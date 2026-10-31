"""API GIRU-IA — servicio on-premise para el backend.

Levantar (Python 3.12 FIRMADO + librerias del .venv; el python.exe del .venv
lo bloquea Smart App Control):
    cd C:\\Users\\investigadores\\dev\\GIRU_IA_72b
    $env:PYTHONPATH = "C:\\Users\\investigadores\\dev\\.venv\\Lib\\site-packages"
    $env:GIRU_API_KEY = "..."
    py -3.12 -m uvicorn main:app --app-dir api --host 0.0.0.0 --port 8000

Variables de entorno:
    GIRU_API_KEY      clave compartida con el backend (obligatoria en produccion)
    GIRU_PROYECTO     carpeta del proyecto GRAIL a montar
    GIRU_MODO         modo por defecto del chatbot (cascade)
    GIRU_SESIONES     carpeta de los JSON de sesiones
    GIRU_BUSCADOR     carpeta del buscador (por defecto dev\\buscador, hermana del grafo)
    GIRU_CONCURRENCIA consultas simultaneas al LLM (3; ponerlo en 1 serializa)
    GIRU_CALENTAR     "0" para no precalentar al arrancar
"""
from __future__ import annotations

import asyncio
import logging
import os
from typing import Optional

from fastapi import Depends, FastAPI, Header, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware

import giru_cache                                        # ← GIRU
import grail_svc
import sesiones
from esquemas import (
    BusquedaEntrada,
    BusquedaSalida,
    Coincidencia,
    Fuente,
    Mensaje,
    PreguntaEntrada,
    PreguntaSalida,
    SesionCrear,
    SesionDetalle,
    SesionResumen,
)

log = logging.getLogger("giru.api")
API_KEY = os.getenv("GIRU_API_KEY", "")
CALENTAR = os.getenv("GIRU_CALENTAR", "1") != "0"        # ← GIRU

app = FastAPI(title="GIRU-IA", version="1.1", description="Consultas normativas UTFSM")

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("GIRU_CORS", "*").split(","),
    allow_methods=["*"],
    allow_headers=["*"],
)


def clave(x_api_key: Optional[str] = Header(None, alias="X-API-Key")) -> None:
    if API_KEY and x_api_key != API_KEY:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Clave de API invalida")


@app.on_event("startup")
async def arranque() -> None:
    await sesiones.init()
    log.info("Almacen de sesiones listo")

    try:
        b = await asyncio.to_thread(grail_svc.buscador)   # precarga del indice de chunks
        log.info("Indice del buscador listo: %d chunks", len(b.documentos))
    except Exception:
        log.exception("No se pudo cargar el indice del buscador (/buscar fallara)")

    # ← GIRU: precalentar el grafo. Sin esto, la PRIMERA consulta del dia paga la
    # carga de los parquet mas el embebido del corpus completo (~40 s), y si el
    # backend tiene un timeout corto ese primer request le falla. Ademas deja la
    # inicializacion perezosa resuelta antes de la primera rafaga concurrente.
    if CALENTAR:
        try:
            info = await grail_svc.calentar()
            log.info("Grafo listo y precalentado: %s", info)
        except Exception:
            log.exception("Fallo el precalentamiento (la API sigue; la 1a consulta sera lenta)")
    else:
        log.info("Precalentamiento desactivado (GIRU_CALENTAR=0)")


# ------------------------------------------------------------------- salud
@app.get("/salud")
async def salud() -> dict[str, str]:
    return {"estado": "ok"}


@app.get("/diagnostico", dependencies=[Depends(clave)])
async def diagnostico() -> dict:
    """Carga el grafo y reporta conteos y columnas. Util para verificar que se
    monto la corrida correcta, que los embeddings estan presentes, y que el
    modelo de generacion y los caches son los esperados."""
    return await asyncio.to_thread(grail_svc.diagnostico)   # ← GIRU: no bloquear el loop


@app.post("/cache/limpiar", dependencies=[Depends(clave)])
async def cache_limpiar() -> dict:                          # ← GIRU
    """Vaciar los caches en memoria. Usarlo DESPUES de reindexar el grafo: si no,
    la API seguiria respondiendo con los parquet viejos. Tras llamarlo, la
    siguiente consulta vuelve a pagar la carga (~40 s)."""
    giru_cache.limpiar()
    grail_svc.olvidar()
    return {"estado": "caches vaciados", "cache": giru_cache.estado()}


# ---------------------------------------------------------------- sesiones
@app.post("/sesiones", response_model=SesionResumen, status_code=201,
          dependencies=[Depends(clave)])
async def crear_sesion(body: SesionCrear) -> SesionResumen:
    s = await sesiones.crear_sesion(body.user_id, body.titulo, body.modo)
    return SesionResumen(**s)


@app.get("/sesiones", response_model=list[SesionResumen], dependencies=[Depends(clave)])
async def listar_sesiones(user_id: str) -> list[SesionResumen]:
    return [SesionResumen(**s) for s in await sesiones.listar_sesiones(user_id)]


@app.get("/sesiones/{sesion_id}", response_model=SesionDetalle,
         dependencies=[Depends(clave)])
async def detalle_sesion(sesion_id: str, user_id: str) -> SesionDetalle:
    s = await sesiones.obtener_sesion(sesion_id, user_id)
    if not s:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Sesion no encontrada")
    msgs = await sesiones.mensajes(sesion_id)
    return SesionDetalle(**s, mensajes=[Mensaje(**m) for m in msgs])


@app.delete("/sesiones/{sesion_id}", status_code=204, dependencies=[Depends(clave)])
async def borrar_sesion(sesion_id: str, user_id: str) -> None:
    if not await sesiones.borrar_sesion(sesion_id, user_id):
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Sesion no encontrada")


# --------------------------------------------------------------- preguntar
@app.post("/preguntar", response_model=PreguntaSalida, dependencies=[Depends(clave)])
async def preguntar(body: PreguntaEntrada) -> PreguntaSalida:
    s = await sesiones.obtener_sesion(body.session_id, body.user_id)
    if not s:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Sesion no encontrada")

    modo = body.modo or s["modo"]
    if modo == "document" and not body.documento:
        raise HTTPException(status.HTTP_400_BAD_REQUEST,
                            "El modo 'document' necesita el campo 'documento'")

    # Historial ANTES de guardar la pregunta actual, para no duplicarla.
    historial = await sesiones.mensajes(body.session_id, limite=40)
    await sesiones.agregar_mensaje(body.session_id, "user", body.pregunta)
    await sesiones.renombrar_si_nueva(body.session_id, body.pregunta)

    try:
        r = await grail_svc.preguntar(body.pregunta, modo, historial, body.documento)
    except Exception as exc:
        log.exception("Fallo la consulta a GRAIL")
        raise HTTPException(status.HTTP_502_BAD_GATEWAY, f"Error consultando el grafo: {exc}")

    await sesiones.agregar_mensaje(
        body.session_id, "assistant", r["respuesta"],
        meta={"modo": r["modo"], "fuentes": r["fuentes"], "tiempo_s": r["tiempo_s"]},
    )

    return PreguntaSalida(
        session_id=body.session_id,
        respuesta=r["respuesta"],
        fuentes=[Fuente(**f) for f in r["fuentes"]],
        respaldo=r.get("respaldo"),                      # ← GIRU 31-10
        modo=r["modo"],
        tiempo_s=r["tiempo_s"],
        llamadas_llm=r["llamadas_llm"],
    )


# ------------------------------------------------------------------ buscar
@app.post("/buscar", response_model=BusquedaSalida, dependencies=[Depends(clave)])
async def buscar(body: BusquedaEntrada) -> BusquedaSalida:
    """Buscador semantico: devuelve documentos rankeados, sin sintesis del LLM.
    El backend usa el nombre de archivo como llave hacia su base de datos."""
    try:
        res = await grail_svc.buscar(body.consulta, body.k)
    except Exception as exc:
        log.exception("Fallo la busqueda")
        raise HTTPException(status.HTTP_502_BAD_GATEWAY, f"Error en la busqueda: {exc}")
    return BusquedaSalida(consulta=body.consulta,
                          resultados=[Coincidencia(**r) for r in res])

# --------------------------------------------------------- preguntar/stream
# Streaming SSE del chatbot (ver stream.py). Misma autenticacion que el resto.
import stream  # noqa: E402

app.include_router(stream.router, dependencies=[Depends(clave)])