"""Capa de servicio sobre GRAIL.

Una sola instancia de GRAIL viva en el proceso (cargar el grafo es caro) y un
semaforo que ACOTA (ya no serializa) las consultas concurrentes.

La busqueda de /buscar NO llama al LLM ni usa el grafo: usa el indice propio de
chunks (carpeta dev\\buscador, generado por indexar_chunks.py) con busqueda
hibrida coseno + BM25 agrupada por documento. Decision de arquitectura: el
indice vectorial de GRAIL es de ENTIDADES (sirve para conectar -> chatbot); para
localizar documentos se usa un indice de chunks (-> buscador).

La busqueda antigua por entidades se conserva como buscar_entidades() para el
experimento de evaluacion (entidades vs chunks); la API no la usa.
"""
from __future__ import annotations

import asyncio
import logging                                        # ← GIRU
import os
import sys
import threading                                      # ← GIRU: lazy init seguro
from pathlib import Path
from typing import Any, Optional

import httpx
import numpy as np
import pandas as pd

import giru_cache                                     # ← GIRU: caches
import giru_citas                                     # ← GIRU: citas verificadas
import giru_recuperacion                              # ← GIRU: rescate de cascade
import giru_identidad                                 # ← GIRU 31-10: se llama Don Fede
import giru_consulta                                  # ← GIRU 31-10: expansion de consulta
import giru_atribucion                                # ← GIRU 31-10: citas en 2a pasada

log = logging.getLogger("giru.svc")                   # ← GIRU

# ← GIRU: el grafo medido y validado el 15-10-2026 es GIRU_IA_72b, no giru_full_72b_v2.
PROYECTO = Path(os.getenv("GIRU_PROYECTO", r"C:\Users\investigadores\dev\GIRU_IA_72b"))
OLLAMA_URL = os.getenv("GIRU_OLLAMA", "http://localhost:11434")
MODELO_EMB = os.getenv("GIRU_MODELO_EMB", "bge-m3")
MODO_DEFECTO = os.getenv("GIRU_MODO", "cascade")
HISTORIAL_MAX = int(os.getenv("GIRU_HISTORIAL", "20"))

# ← GIRU: cuantas consultas al LLM pueden estar en vuelo a la vez.
# Debe ir alineado con OLLAMA_NUM_PARALLEL. Medicion del 15-10-2026 (qwen2.5:32b,
# 5 preguntas): paralelo 1 = 52,1 s | 3 = 38,0 s | 5 = 35,6 s. De 3 a 5 solo se
# gana 6% del total y la latencia por pregunta sube 83%, asi que 3 es el optimo.
# Poner GIRU_CONCURRENCIA=1 restaura el comportamiento serializado anterior.
CONCURRENCIA = int(os.getenv("GIRU_CONCURRENCIA", "3"))

# ← GIRU: cuantas fuentes se devuelven, ya ordenadas por evidencia. El frontend
# muestra la primera como "FUENTE OFICIAL", asi que lo que importa es el orden;
# el tope solo evita mandarle al backend una lista de decenas de documentos que
# apenas rozan la respuesta. 0 = sin tope.
MAX_FUENTES = int(os.getenv("GIRU_MAX_FUENTES", "5")) or 10**9

# Carpeta del buscador: hermana de la carpeta del grafo (dev\buscador).
# api\grail_svc.py -> parents[2] = dev
BUSCADOR_DIR = Path(os.getenv("GIRU_BUSCADOR", Path(__file__).resolve().parents[2] / "buscador"))

# grail.yaml usa rutas RELATIVAS (root_dir: ., custom_prompts, output, lancedb) y
# GRAIL las resuelve contra el directorio de trabajo (grail/prompts/loader.py).
# Se fija el directorio de trabajo en la carpeta del proyecto para que la API
# funcione igual sin importar desde donde se lance.
os.chdir(PROYECTO)

# ← GIRU: antes era un asyncio.Lock() exclusivo que serializaba TODA consulta.
# Ahora un semaforo acotado: permite CONCURRENCIA consultas simultaneas y sigue
# poniendo un techo. Ver la nota "Sobre la concurrencia" al final del archivo.
_sem_consultas = asyncio.Semaphore(CONCURRENCIA)

# ← GIRU: RLock (no Lock) porque artefactos() llama a grail() con el lock tomado;
# con un Lock normal eso seria un deadlock. Es threading y no asyncio porque
# estas funciones tambien se entran desde asyncio.to_thread().
_lock_init = threading.RLock()

_grail = None
_artefactos = None
_matriz: Optional[np.ndarray] = None
_meta: Optional[pd.DataFrame] = None
_buscador = None


# ------------------------------------------------------------ carga perezosa
def grail():
    global _grail
    with _lock_init:                                   # ← GIRU
        if _grail is None:
            from grail.config import load_config
            from grail.core import GRAIL

            # ← GIRU: activar() reemplaza funciones en memoria, asi que tiene que
            # correr ANTES de construir la instancia. Si GRAIL ya existe, el objeto
            # quedo con las referencias originales y el cache no toma efecto.
            giru_cache.activar()
            giru_recuperacion.activar()      # ← GIRU: antes de construir GRAIL
            giru_identidad.activar()         # ← GIRU 31-10: assistant_name = Don Fede
            _grail = GRAIL.from_config(load_config(PROYECTO))
        return _grail


def artefactos():
    global _artefactos
    with _lock_init:                                   # ← GIRU
        if _artefactos is None:
            from grail.query.retrieval import load_artifacts_for_search

            g = grail()
            # Con giru_cache activo esta llamada devuelve la MISMA copia que usa
            # core.search(), en vez de cargar un segundo juego de parquet.
            _artefactos = load_artifacts_for_search(g.storage, g._output_folder())
        return _artefactos


def _col(df: pd.DataFrame, *candidatos: str) -> Optional[str]:
    """Primera columna existente entre varios nombres posibles."""
    for c in candidatos:
        if c in df.columns:
            return c
    return None


def olvidar() -> None:                               # ← GIRU
    """Suelta las referencias en memoria de este modulo (no las de giru_cache).

    Para usar DESPUES de reindexar: si no, la API sigue sirviendo los parquet
    viejos. La instancia de GRAIL tambien se suelta, porque su storage apunta al
    run anterior. Llamar tambien a giru_cache.limpiar().
    """
    global _grail, _artefactos, _matriz, _meta
    with _lock_init:
        _grail = None
        _artefactos = None
        _matriz = None
        _meta = None
    giru_citas.limpiar_cache()
    giru_atribucion.olvidar()                          # ← GIRU 31-10


# ------------------------------------------------- arranque (opcional)
async def calentar() -> dict[str, Any]:              # ← GIRU
    """Llamar desde el evento startup de FastAPI.

    Deja la instancia, los parquet y los embeddings del corpus en memoria para que
    la primera consulta real no pague ~40 s. Tambien evita cualquier carrera de
    inicializacion perezosa bajo la primera rafaga de consultas.
    """
    g = await asyncio.to_thread(grail)
    await asyncio.to_thread(artefactos)
    segundos = await giru_cache.precalentar(g)
    return {"calentamiento_s": round(segundos, 1), "cache": giru_cache.estado()}


# ---------------------------------------------------------------- preguntar
def _historial_para_grail(filas: list[dict[str, Any]]) -> list[dict[str, str]]:
    return [{"role": f["rol"], "content": f["contenido"]} for f in filas][-HISTORIAL_MAX:]


def _basename(ruta: Any) -> str:
    """Nombre de archivo, con '/' o con '\\'.

    No se usa Path(x).name porque depende de la plataforma: en Linux la barra
    invertida no separa rutas, asi que una ruta de Windows volveria entera.
    """
    return str(ruta).replace("\\", "/").rstrip("/").rsplit("/", 1)[-1]


def _ids_de_documento(fila) -> list[str]:
    """document_ids de una fila del contexto; cae a document_id si falta."""
    crudo = fila.get("document_ids")
    if crudo is None or (hasattr(crudo, "__len__") and len(crudo) == 0):
        uno = fila.get("document_id")
        return [str(uno)] if uno else []
    if isinstance(crudo, str):
        return [crudo]
    try:
        return [str(d) for d in crudo if d]
    except TypeError:
        return [str(crudo)]


def _titulos() -> dict[str, str]:
    from grail.query.retrieval import resolve_doc_titles

    a = artefactos()
    return resolve_doc_titles(a.documents, a.mapping)


def procesar(resultado) -> tuple[str, list[dict[str, Any]]]:
    """(respuesta visible, fuentes) — ← GIRU.

    Dos niveles, de mas fuerte a mas debil:

      1. CITAS VERIFICADAS. El modelo cierra con un bloque <citas> de texto
         literal; se comprueba que ese texto exista dentro de alguno de los
         fragmentos que se le enviaron. Si pasa, la fuente trae 'fragmento' y es
         comprobable caracter por caracter.
      2. RANKING POR EVIDENCIA. Si el modelo no cito, o cito algo que no esta en
         ningun fragmento, se cae al orden por evidencia recuperada y las
         fuentes van sin 'fragmento'.

    El bloque <citas> se quita SIEMPRE de la respuesta visible, haya verificado
    o no, para que el usuario nunca lo vea.
    """
    crudo = getattr(resultado, "response", "") if resultado is not None else ""
    texto = crudo if isinstance(crudo, str) else str(crudo or "")
    visible, citas = giru_citas.separar(texto)

    ctx = getattr(resultado, "context_data", None)
    ctx = ctx if isinstance(ctx, dict) else {}

    fuentes: list[dict[str, Any]] = []
    if citas:
        titulos = _titulos()
        verificadas = giru_citas.verificar(
            citas, ctx.get("sources"), titulos,
            # documents trae raw_content: hace falta para saber a CUAL documento
            # pertenece la cita cuando un fragmento cruza el limite entre dos.
            documentos=artefactos().documents,
            maximo=MAX_FUENTES,
        )
        for v in verificadas:
            fuentes.append(
                {"documento": v["documento"], "referencia": None, "fragmento": v["fragmento"]}
            )
        if not verificadas:
            log.info(
                "GIRU: el modelo entrego %d cita(s) pero ninguna se pudo verificar "
                "contra los fragmentos; se usa el ranking por evidencia", len(citas)
            )

    if not fuentes:
        fuentes = [
            {"documento": d, "referencia": None, "fragmento": None}
            for d in _documentos_por_evidencia(ctx)
        ]

    return visible, fuentes


def _fuentes(resultado) -> list[dict[str, Any]]:
    """Compatibilidad: solo las fuentes. Prefiere procesar(), que ademas
    devuelve la respuesta ya limpia del bloque de citas."""
    return procesar(resultado)[1]


def _modelo_generacion() -> str:
    g = grail()
    return (getattr(g.config.search, "local_search_model", None)
            or getattr(g.config.llm, "model", "qwen2.5:32b"))


async def consulta_para(pregunta: str, modo: str, hist: list[dict[str, str]]) -> str:
    """← GIRU 31-10: la consulta que recibe GRAIL (expandida si corresponde).

    El historial guardado en sesiones sigue siendo la pregunta ORIGINAL; la
    expansion solo vive en esta llamada.
    """
    return await giru_consulta.expandir(
        pregunta, modo, hist, ollama=OLLAMA_URL, modelo_defecto=_modelo_generacion())


async def procesar_async(resultado, pregunta: str
                         ) -> tuple[str, list[dict[str, Any]], Optional[float]]:
    """(respuesta visible, fuentes, respaldo) — ← GIRU 31-10.

    Tres niveles, de mas fuerte a mas debil:

      1. ATRIBUCION EN SEGUNDA PASADA (giru_atribucion.py). Se parte la respuesta
         en afirmaciones, se buscan en el indice fino de chunks y un juez LLM
         decide que articulo respalda cada una. Decide la fuente por lo que el
         modelo AFIRMO, no por lo que recupero. Si hay varias versiones, gana la
         vigente.
      2. CITAS VERIFICADAS del bloque <citas> (procesar()). Se agregan al final si
         su documento no quedo ya en la lista.
      3. RANKING POR EVIDENCIA, si ninguno de los anteriores produjo nada.

    `respaldo` = fraccion de afirmaciones con respaldo encontrado (0-1), o None si
    no se pudo medir. Es un indicador de confianza para el frontend.
    """
    visible, previas = procesar(resultado)
    if not giru_atribucion.ACTIVADO or not visible.strip():
        return visible, previas, None

    ctx = getattr(resultado, "context_data", None)
    ctx = ctx if isinstance(ctx, dict) else {}
    titulos = _titulos()
    docs_contexto: set[str] = set()
    src = ctx.get("sources")
    if src is not None and hasattr(src, "empty") and not src.empty:
        for _, fila in src.iterrows():
            for did in _ids_de_documento(fila):
                n = _basename(titulos.get(did) or "")
                if n:
                    docs_contexto.add(n)

    res = await giru_atribucion.atribuir_async(
        pregunta=giru_consulta.pregunta_visible(pregunta), respuesta=visible,
        docs_contexto=docs_contexto, buscador_dir=BUSCADOR_DIR, proyecto=PROYECTO,
        ollama=OLLAMA_URL, modelo_emb=MODELO_EMB, modelo_juez=_modelo_generacion(),
    )
    if not res or not res["fuentes"]:
        return visible, previas, (res or {}).get("respaldo")

    fuentes = list(res["fuentes"])
    ya = {f["documento"] for f in fuentes}
    for f in previas:                       # citas literales del propio modelo
        if f.get("fragmento") and f["documento"] not in ya:
            fuentes.append({**f, "seccion": None})
            ya.add(f["documento"])
    return visible, fuentes[:MAX_FUENTES], res["respaldo"]


def _documentos_por_evidencia(ctx: dict) -> list[str]:
    """Nombres de documento ORDENADOS por evidencia real (respaldo).

    Por que no se usa extract_source_references de GRAIL
    ----------------------------------------------------
    Esa funcion recorre TODAS las tablas de context_data en orden de diccionario
    (entities, relationships, reports, sources) y devuelve los documentos sin
    ningun criterio de relevancia. Dos consecuencias, ambas observadas en
    produccion:

      1. Las entidades van PRIMERO, y el document_ids de una entidad lista todos
         los documentos donde esa entidad aparece. Una entidad frecuente como
         UNIVERSIDAD arrastra decenas de decretos irrelevantes de una sola vez.
      2. 'sources' -los fragmentos de texto que realmente se le pasaron al
         modelo- se procesa AL FINAL, cuando la lista ya quedo inundada.

    El frontend toma la primera fuente y la muestra como "FUENTE OFICIAL", asi
    que ese orden arbitrario se traducia en citar el decreto equivocado.

    Como se ordena aca
    ------------------
    Solo cuentan los fragmentos de texto ('sources'): un documento se gana el
    lugar si su texto efectivamente llego al modelo, no por compartir una
    entidad. Cada fragmento aporta 1/(posicion+1) al documento del que viene, de
    modo que el fragmento mejor rankeado pesa 1,0 y el decimo 0,1. Un documento
    con varios fragmentos buenos sube; uno con un fragmento lejano se queda
    abajo.

    Las entidades se usan SOLO si no hubo fragmentos (sin ellas la lista
    quedaria vacia). Los reportes de comunidad nunca entran: son texto que
    genero el propio modelo al indexar, no una fuente oficial.
    """
    if not ctx:
        return []

    titulos = _titulos()
    puntajes: dict[str, float] = {}

    def acumular(df, peso_base: float) -> bool:
        if df is None or not hasattr(df, "empty") or df.empty:
            return False
        if "document_ids" not in df.columns and "document_id" not in df.columns:
            return False
        visto = False
        for posicion, (_, fila) in enumerate(df.iterrows()):
            peso = peso_base / (posicion + 1)
            dids = _ids_de_documento(fila)
            if not dids:
                continue
            # Un fragmento que cruza el limite entre dos documentos REPARTE su
            # peso: darle el total a cada uno inflaba al vecino de al lado, que
            # era como un decreto del RIOHS terminaba encabezando una pregunta
            # sobre el ROAS.
            reparto = peso / len(dids)
            for did in dids:
                puntajes[did] = puntajes.get(did, 0.0) + reparto
                visto = True
        return visto

    if not acumular(ctx.get("sources"), 1.0):
        acumular(ctx.get("entities"), 1.0)      # respaldo: mejor poco que nada

    if not puntajes:
        return []

    # Dos ids distintos pueden apuntar al mismo archivo: el corpus tiene
    # duplicados exactos. Se fusionan por nombre y se queda el mejor puntaje.
    por_nombre: dict[str, float] = {}
    for did, pts in puntajes.items():
        nombre = _basename(titulos.get(did) or "")
        if nombre:
            por_nombre[nombre] = max(por_nombre.get(nombre, 0.0), pts)

    ordenado = sorted(por_nombre.items(), key=lambda kv: (-kv[1], kv[0]))
    return [n for n, _ in ordenado[:MAX_FUENTES]]


async def preguntar(
    pregunta: str,
    modo: Optional[str],
    historial: list[dict[str, Any]],
    documento: Optional[str] = None,
) -> dict[str, Any]:
    # ← GIRU: la inicializacion perezosa se hace FUERA del semaforo. Si la primera
    # rafaga llega con el proceso frio, una consulta carga el grafo mientras las
    # otras esperan en el RLock, sin ocupar los cupos de consulta.
    g = await asyncio.to_thread(grail)
    m = modo or MODO_DEFECTO
    hist = _historial_para_grail(historial)

    async with _sem_consultas:                         # ← GIRU: antes era _lock
        if m == "agent":
            res = await g.agent_search(query=pregunta, conversation_history=hist,
                                       system_prompt=giru_identidad.prompt_agente())
        else:
            consulta = await consulta_para(pregunta, m, hist)     # ← GIRU 31-10
            res = await g.search(
                query=consulta, mode=m, conversation_history=hist, document=documento
            )

    # ← GIRU 31-10: procesar_async() quita el bloque <citas>, atribuye las fuentes
    # en segunda pasada y cae a las citas verificadas / ranking si no puede.
    # Va FUERA del semaforo: el juez es una llamada corta y Ollama la encola.
    texto, fuentes, respaldo = await procesar_async(res, pregunta)
    return {
        "respuesta": texto,
        "fuentes": fuentes,
        "respaldo": respaldo,
        "modo": m,
        "tiempo_s": float(getattr(res, "completion_time", 0.0) or 0.0),
        "llamadas_llm": int(getattr(res, "llm_calls", 0) or 0),
    }


# ------------------------------------------------------------------- buscar
async def _embeber(texto: str) -> np.ndarray:
    """Ruta NATIVA de Ollama. La compatible con OpenAI (/v1/embeddings) falla
    en este servidor, que es el bug que obligo a escribir reparar.py."""
    async with httpx.AsyncClient(timeout=60) as cli:
        r = await cli.post(
            f"{OLLAMA_URL}/api/embed", json={"model": MODELO_EMB, "input": texto}
        )
        r.raise_for_status()
        datos = r.json()
    vec = datos["embeddings"][0] if "embeddings" in datos else datos["embedding"]
    v = np.asarray(vec, dtype="float32")
    return v / (np.linalg.norm(v) + 1e-9)


def _preparar_indice() -> tuple[np.ndarray, pd.DataFrame]:
    """Matriz normalizada de embeddings de entidad + metadatos para mapear a documento."""
    global _matriz, _meta
    with _lock_init:                                   # ← GIRU
        if _matriz is not None and _meta is not None:
            return _matriz, _meta

        ents = artefactos().entities
        c_emb = _col(ents, "embedding", "description_embedding", "vector", "embeddings")
        if c_emb is None:
            raise RuntimeError(
                f"No encuentro la columna de embeddings en entities. Columnas: {list(ents.columns)}"
            )
        c_tit = _col(ents, "title", "name", "entity") or ents.columns[0]
        c_doc = _col(ents, "document_ids", "doc_ids", "documents")
        c_tu = _col(ents, "text_unit_ids", "text_unit_id", "text_units")

        validas = ents[ents[c_emb].notna()].reset_index(drop=True)
        if validas.empty:
            raise RuntimeError("Las entidades no tienen embeddings; hay que correr reparar.py")

        M = np.vstack([np.asarray(v, dtype="float32") for v in validas[c_emb]])
        M = M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-9)

        meta = pd.DataFrame({"titulo": validas[c_tit].astype(str)})
        meta["documentos"] = (
            validas[c_doc] if c_doc else pd.Series([None] * len(validas))
        )
        meta["text_units"] = validas[c_tu] if c_tu else pd.Series([None] * len(validas))

        _matriz, _meta = M, meta
        return M, meta


def _nombre_documento(doc_id: str) -> Optional[str]:
    docs = artefactos().documents
    if docs.empty:
        return None
    c_id = _col(docs, "id", "document_id") or "id"
    c_nom = _col(docs, "title", "path", "name", "source")
    fila = docs[docs[c_id].astype(str) == str(doc_id)]
    if fila.empty or c_nom is None:
        return None
    return Path(str(fila.iloc[0][c_nom])).name


def _docs_de_text_units(ids: Any) -> list[str]:
    tus = artefactos().text_units
    if tus.empty or ids is None:
        return []
    c_id = _col(tus, "id", "text_unit_id") or "id"
    c_doc = _col(tus, "document_ids", "document_id", "doc_id")
    if c_doc is None:
        return []
    lista = list(ids) if isinstance(ids, (list, tuple, np.ndarray)) else [ids]
    fila = tus[tus[c_id].astype(str).isin([str(i) for i in lista])]
    salida: list[str] = []
    for v in fila[c_doc]:
        salida.extend([str(x) for x in v] if isinstance(v, (list, tuple, np.ndarray)) else [str(v)])
    return salida


async def buscar_entidades(consulta: str, k: int = 10) -> list[dict[str, Any]]:
    """Busqueda ANTIGUA por entidades del grafo. No la usa la API; se conserva
    para el experimento de evaluacion (entidades vs chunks)."""
    M, meta = await asyncio.to_thread(_preparar_indice)
    q = await _embeber(consulta)
    sims = M @ q

    # Se toman bastantes entidades porque varias caen en el mismo documento.
    top = np.argsort(-sims)[: max(k * 8, 80)]

    acumulado: dict[str, dict[str, Any]] = {}
    for i in top:
        fila = meta.iloc[int(i)]
        ids = fila["documentos"]
        lista = (
            [str(x) for x in ids]
            if isinstance(ids, (list, tuple, np.ndarray))
            else ([str(ids)] if ids is not None and not pd.isna(ids) else [])
        )
        if not lista:
            lista = _docs_de_text_units(fila["text_units"])
        for doc_id in lista:
            nombre = _nombre_documento(doc_id) or doc_id
            reg = acumulado.setdefault(
                nombre, {"documento": nombre, "puntaje": 0.0, "entidades": []}
            )
            reg["puntaje"] = max(reg["puntaje"], float(sims[int(i)]))
            if fila["titulo"] not in reg["entidades"] and len(reg["entidades"]) < 5:
                reg["entidades"].append(fila["titulo"])

    ordenado = sorted(acumulado.values(), key=lambda r: -r["puntaje"])
    for r in ordenado:
        r["puntaje"] = round(r["puntaje"], 4)
    return ordenado[:k]


# ------------------------------------------------- buscar (indice de chunks)
def buscador():
    """Carga perezosa del indice de chunks (una vez por proceso)."""
    global _buscador
    with _lock_init:                                   # ← GIRU
        if _buscador is None:
            if str(BUSCADOR_DIR) not in sys.path:
                sys.path.insert(0, str(BUSCADOR_DIR))
            from busqueda import Buscador

            _buscador = Buscador(BUSCADOR_DIR / "lancedb_buscador")
        return _buscador


async def buscar(consulta: str, k: int = 50) -> list[dict[str, Any]]:
    """Documentos rankeados por busqueda hibrida sobre chunks. Solo expone
    documento y puntaje (el fragmento queda disponible internamente via _chunk)."""
    b = await asyncio.to_thread(buscador)
    res = await asyncio.to_thread(b.buscar, consulta, k)
    return [{"documento": r["documento"], "puntaje": r["puntaje"]} for r in res]


# -------------------------------------------------------------- diagnostico
def diagnostico() -> dict[str, Any]:
    g = grail()
    a = artefactos()
    return {
        "proyecto": str(PROYECTO),
        "nombre": g.config.project_name,
        "modelo_llm": getattr(g.config.llm, "model", "?"),
        # ← GIRU: el que de verdad redacta las respuestas; hereda de llm si es None.
        "modelo_generacion": (
            getattr(g.config.search, "local_search_model", None)
            or getattr(g.config.llm, "model", "?")
        ),
        "modelo_embeddings": getattr(g.config.embeddings, "model", "?"),
        "modo_defecto": MODO_DEFECTO,
        "concurrencia": CONCURRENCIA,                  # ← GIRU
        "cache": giru_cache.estado(),                  # ← GIRU
        "recuperacion": giru_recuperacion.estado(),    # ← GIRU
        "asistente": giru_identidad.NOMBRE,            # ← GIRU 31-10
        "expansion": {"activada": giru_consulta.ACTIVADO, "llm": giru_consulta.USAR_LLM,
                      "siglas": len(giru_consulta.glosario())},
        "atribucion": {"activada": giru_atribucion.ACTIVADO,
                       "candidatos": giru_atribucion.N_CANDIDATOS},
        "conteos": {
            "documentos": int(len(a.documents)),
            "text_units": int(len(a.text_units)),
            "entidades": int(len(a.entities)),
            "relaciones": int(len(a.relationships)),
            "reportes_comunidad": int(len(a.community_reports)),
        },
        "buscador": {
            "carpeta": str(BUSCADOR_DIR),
            "chunks": int(len(_buscador.documentos)) if _buscador is not None else "no cargado",
        },
        "columnas": {
            "entities": list(a.entities.columns),
            "documents": list(a.documents.columns),
            "text_units": list(a.text_units.columns),
        },
    }


# ---------------------------------------------------------------------------
# Sobre la concurrencia (← GIRU)
#
# El codigo anterior usaba un asyncio.Lock() exclusivo con esta justificacion:
# "el wrapper de LLM de GRAIL usa estado global de modulo y no es seguro con
# varias consultas simultaneas". Revisado el codigo de GRAIL v0.1.7:
#
#   - core.search() construye un buscador NUEVO en cada llamada (LocalSearch,
#     CascadeSearch, ...): no hay estado compartido entre consultas ahi.
#   - LLMWrapper trae su propio asyncio.Semaphore(concurrent_requests=15) creado
#     en model_post_init: esta disenado para uso concurrente.
#   - La misma ruta se usa concurrentemente durante la indexacion
#     (extraction_concurrency, NUM_PARALLEL=3), que es carga mucho mas alta.
#   - CostTracker usa threading.Lock y LLMCache usa asyncio.Lock: ambos protegidos.
#   - Lo que si era fragil era la inicializacion perezosa de este archivo
#     (_grail, _artefactos, _buscador sin lock): dos consultas simultaneas en frio
#     podian cargar el grafo dos veces. Eso quedo cubierto con _lock_init.
#
# Por eso el lock exclusivo pasa a semaforo acotado. Si aparece cualquier rareza
# bajo carga, GIRU_CONCURRENCIA=1 vuelve al comportamiento anterior sin tocar
# codigo.
# ---------------------------------------------------------------------------