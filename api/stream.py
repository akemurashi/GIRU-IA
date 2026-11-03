"""Streaming SSE para el chatbot (POST /preguntar/stream).

Eventos:
    fase   -> etapa de la consulta (recuperando, herramienta, redactando). El
              usuario ve progreso antes de que existan tokens.
    token  -> fragmento de la respuesta segun la escribe el modelo.
    fin    -> respuesta completa, fuentes y metricas (igual que /preguntar).
    error  -> algo fallo; el cliente debe cerrar la conexion.

Como funciona el streaming en GRAIL (verificado en grail/llm/wrapper.py y en
la UI de referencia grail/apps/chat/server.py):
  - set_stream_callback(cb) guarda cb en un ContextVar (no es global: cada
    tarea asyncio ve el suyo, asi que dos usuarios no mezclan tokens).
  - cb debe ser ASINCRONO: el wrapper hace `await cb(fragmento)`.
  - TODA llamada al LLM transmite por el callback mientras este activo.
      * modo global: la fase map genera JSON en paralelo -> no se activa el
        callback; la respuesta llega completa al final.
      * modo agent: GRAIL emite etiquetas <tool_call name="..."> por el mismo
        callback -> se convierten en eventos 'fase', no en texto.

Concurrencia (← GIRU, 15-10-2026): antes se tomaba el lock exclusivo de
grail_svc, una consulta a la vez. Ahora se toma el semaforo acotado
(_sem_consultas, GIRU_CONCURRENCIA=3): varios streams simultaneos. El callback
vive en un ContextVar por tarea, asi que cada stream recibe solo sus tokens.
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from typing import Any, AsyncIterator

from fastapi import APIRouter, HTTPException, status
from fastapi.responses import StreamingResponse

import giru_citas                                   # ← GIRU: filtro de <citas>
import grail_svc
import sesiones
from esquemas import PreguntaEntrada

log = logging.getLogger("giru.stream")
router = APIRouter()

try:
    from grail.llm import set_stream_callback
except Exception:  # pragma: no cover
    set_stream_callback = None  # type: ignore
    log.warning("set_stream_callback no disponible: se enviara la respuesta completa de una vez")

MODOS_SIN_TOKENS = {"global"}          # su fase map emite JSON por el callback
RE_TOOL_CALL = re.compile(r'<tool_call name="([^"]*)">')
NOMBRES_HERRAMIENTA = {
    "local_search": "busqueda local en el grafo",
    "cascade_search": "busqueda en el grafo y el texto",
    "global_search": "resumenes tematicos del corpus",
    "document_search": "busqueda dentro de un documento",
}


def _sse(evento: str, datos: dict[str, Any]) -> str:
    return f"event: {evento}\ndata: {json.dumps(datos, ensure_ascii=False)}\n\n"


async def _generar(body: PreguntaEntrada, modo: str, historial: list[dict[str, Any]]
                   ) -> AsyncIterator[str]:
    cola: asyncio.Queue = asyncio.Queue()
    inicio = time.monotonic()
    # ← GIRU: en streaming los tokens salen al cliente a medida que el modelo
    # escribe, asi que el bloque <citas> se veria en pantalla antes de que
    # nadie pueda quitarlo. El filtro lo corta en vuelo, incluso si la marca
    # llega partida entre dos fragmentos.
    filtro = giru_citas.FiltroStream()
    transmitir = set_stream_callback is not None and modo not in MODOS_SIN_TOKENS

    # ← GIRU: antes era `grail_svc.grail()` dentro de ejecutar(), llamada SINCRONA
    # en el event loop. Con el proceso frio eso carga el grafo y los parquet
    # bloqueando el loop ~40 s: ningun otro request se atiende, ni /salud. Aqui se
    # hace en un hilo y ANTES del semaforo, para no ocupar un cupo mientras carga.
    try:
        g = await asyncio.to_thread(grail_svc.grail)
    except Exception as exc:  # noqa: BLE001
        log.exception("No se pudo cargar el grafo")
        yield _sse("error", {"mensaje": f"No se pudo cargar el grafo: {exc}"})
        return

    async def al_recibir(fragmento: str) -> None:
        """Callback de GRAIL (asincrono). Separa etiquetas de herramienta del texto."""
        m = RE_TOOL_CALL.search(fragmento)
        if m:
            await cola.put(("herramienta", m.group(1)))
            return
        visible = filtro.filtrar(fragmento)            # ← GIRU
        if visible:
            await cola.put(("token", visible))

    async def ejecutar() -> None:
        # El callback se fija DENTRO de la tarea (ContextVar), como hace la UI de GRAIL.
        if transmitir:
            set_stream_callback(al_recibir)
        try:
            hist = grail_svc._historial_para_grail(historial)
            if modo == "agent":
                res = await g.agent_search(query=body.pregunta, conversation_history=hist,
                                           system_prompt=grail_svc.giru_identidad.prompt_agente())
            else:
                consulta = await grail_svc.consulta_para(body.pregunta, modo, hist)  # ← 31-10
                res = await g.search(query=consulta, mode=modo,
                                     conversation_history=hist, document=body.documento)
            await cola.put(("resultado", res))
        except Exception as exc:  # noqa: BLE001
            log.exception("Fallo la consulta en streaming")
            await cola.put(("error", str(exc)))
        finally:
            if transmitir:
                set_stream_callback(None)
            resto = filtro.vaciar()                    # ← GIRU: cola retenida
            if resto:
                await cola.put(("token", resto))
            await cola.put(("cerrar", None))

    detalle = ("Resumiendo los temas del corpus" if modo == "global"
               else "Buscando en el grafo de normativa")
    yield _sse("fase", {"fase": "recuperando", "detalle": detalle})

    resultado = None
    error = None
    hubo_tokens = False          # para re-emitir la fase "redactando" tras una herramienta
    algun_token = False          # si nunca llego un token, se manda la respuesta entera
    async with grail_svc._sem_consultas:        # ← GIRU: antes era el lock exclusivo
        tarea = asyncio.create_task(ejecutar())
        try:
            while True:
                tipo, dato = await cola.get()
                if tipo == "token":
                    algun_token = True
                    if not hubo_tokens:
                        hubo_tokens = True
                        yield _sse("fase", {"fase": "redactando",
                                            "detalle": "Redactando la respuesta"})
                    yield _sse("token", {"t": dato})
                elif tipo == "herramienta":
                    hubo_tokens = False          # tras una herramienta vuelve a redactar
                    yield _sse("fase", {"fase": "herramienta",
                                        "detalle": "Consultando: "
                                        + NOMBRES_HERRAMIENTA.get(dato, dato)})
                elif tipo == "resultado":
                    resultado = dato
                elif tipo == "error":
                    error = dato
                elif tipo == "cerrar":
                    break
        finally:
            if not tarea.done():               # el cliente cerro la conexion
                tarea.cancel()

    if error:
        yield _sse("error", {"mensaje": error})
        return

    # La respuesta final es la de GRAIL (limpia), no la suma de tokens: en modo
    # agent los tokens pueden incluir texto intermedio entre herramientas.
    # ← GIRU: procesar() quita el bloque <citas> y devuelve las fuentes
    # verificadas (o el ranking por evidencia si ninguna cita se pudo comprobar).
    # ← GIRU 31-10: la atribucion en segunda pasada tarda unos segundos; el
    # usuario ya leyo la respuesta, se le avisa que se estan verificando fuentes.
    respaldo = None
    if resultado is not None:
        if grail_svc.giru_atribucion.ACTIVADO:     # la fase solo existe si hay 2a pasada
            yield _sse("fase", {"fase": "verificando", "detalle": "Verificando las fuentes"})
        completo, fuentes, respaldo = await grail_svc.procesar_async(resultado, body.pregunta)
    else:
        completo, fuentes = "", []
    if not algun_token and completo:     # modo global, o streaming no disponible
        yield _sse("token", {"t": completo})
    tiempo = float(getattr(resultado, "completion_time", 0.0) or 0.0) or round(
        time.monotonic() - inicio, 1)
    llamadas = int(getattr(resultado, "llm_calls", 0) or 0)

    await sesiones.agregar_mensaje(
        body.session_id, "assistant", completo,
        meta={"modo": modo, "fuentes": fuentes, "tiempo_s": tiempo},
    )

    yield _sse("fin", {
        "session_id": body.session_id,
        "respuesta": completo,
        "fuentes": fuentes,
        "respaldo": respaldo,
        "modo": modo,
        "tiempo_s": tiempo,
        "llamadas_llm": llamadas,
    })


@router.post("/preguntar/stream")
async def preguntar_stream(body: PreguntaEntrada) -> StreamingResponse:
    s = await sesiones.obtener_sesion(body.session_id, body.user_id)
    if not s:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Sesion no encontrada")

    modo = body.modo or s["modo"]
    if modo == "document" and not body.documento:
        raise HTTPException(status.HTTP_400_BAD_REQUEST,
                            "El modo 'document' necesita el campo 'documento'")

    historial = await sesiones.mensajes(body.session_id, limite=40)
    await sesiones.agregar_mensaje(body.session_id, "user", body.pregunta)
    await sesiones.renombrar_si_nueva(body.session_id, body.pregunta)

    return StreamingResponse(
        _generar(body, modo, historial),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",   # evita que nginx acumule el flujo
        },
    )