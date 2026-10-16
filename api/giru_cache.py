"""giru_cache.py — Caches de artefactos y de embeddings de chunks para la API de GIRU.

Colocar junto a grail_svc.py. Llamar activar() UNA VEZ antes de instanciar GRAIL.

Qué resuelve (medido el 15-10-2026 sobre GIRU_IA_72b con qwen2.5:32b, 5 preguntas):
    sin cache : 189,1 s totales  (mediana 121,6 s por pregunta)
    con cache :  35,6 s totales  (mediana  33,7 s por pregunta)

Causa, verificada en el codigo de GRAIL v0.1.7 (commit 283943f):
  1. core.search() llama load_artifacts_for_search() en CADA consulta, releyendo
     todos los parquet del run desde disco.
  2. cascade_search.py (~linea 198) llama
     embed_safe(chunk_texts, tag="cascade_chunk_embed") sobre TODOS los text
     units en CADA consulta. Con 1125 chunks y 5 consultas simultaneas serian
     5625 embeddings compitiendo por la misma GPU que esta generando.

No modifica archivos: solo reemplaza dos funciones en memoria.

IMPORTANTE: tras reindexar el grafo hay que reiniciar la API (o llamar limpiar()),
o seguira respondiendo con los parquet viejos que quedaron en memoria.
"""
from __future__ import annotations

import asyncio
import hashlib
import logging
import os
from typing import Any, Optional

log = logging.getLogger(__name__)

_artifacts_cache: dict[str, Any] = {}
_chunk_cache: dict[tuple, Any] = {}
_chunk_lock = asyncio.Lock()
_activado = False


def activar() -> None:
    """Activa los dos caches. Idempotente: llamarla dos veces no hace nada."""
    global _activado
    if _activado:
        return

    import grail.core as core
    import grail.query.retrieval as retrieval
    from grail.llm.embeddings import EmbeddingClient

    # ------------------------------------------------------------------ 1. parquet
    _orig_load = retrieval.load_artifacts_for_search

    def _load_cached(storage, output_folder: str = "output"):
        # Clave = ruta absoluta de la carpeta de salida. A diferencia de id(storage),
        # no se puede reciclar: si el interprete recolecta el objeto storage y otro
        # nuevo reusa su direccion de memoria, una clave por id() devolveria los
        # artefactos de otro grafo. En un proceso que vive semanas eso importa.
        clave = os.path.abspath(output_folder)
        if clave not in _artifacts_cache:
            log.info("GIRU: cargando artefactos desde %s (primera vez)", clave)
            _artifacts_cache[clave] = _orig_load(storage, output_folder)
        return _artifacts_cache[clave]

    # Hay que parchear LOS DOS modulos. core.py hace
    #   from grail.query.retrieval import load_artifacts_for_search
    # al importarse, por lo que grail.core tiene su propio nombre ya ligado a la
    # funcion original: parchear solo retrieval no cambiaria lo que usa search().
    # Parchear ambos hace que core.search() y grail_svc.artefactos() compartan
    # UNA sola copia de los parquet en memoria, en vez de tener dos.
    retrieval.load_artifacts_for_search = _load_cached
    core.load_artifacts_for_search = _load_cached

    # ------------------------------------------------- 2. embeddings de chunks
    _orig_embed_safe = EmbeddingClient.embed_safe

    async def _embed_safe_cached(
        self,
        texts: list[str],
        *,
        endpoint: Optional[str] = None,
        model: Optional[str] = None,
        tag: Optional[str] = None,
        session_id: Optional[str] = None,
    ):
        # Solo se intercepta el re-embebido del corpus de cascade. El embedding de
        # la consulta del usuario (tag="query_embedding") NUNCA se cachea: cada
        # pregunta es distinta y cachearla seria un error de correctitud.
        if tag != "cascade_chunk_embed":
            return await _orig_embed_safe(
                self, texts, endpoint=endpoint, model=model, tag=tag, session_id=session_id
            )

        huella = hashlib.sha1("\x00".join(texts).encode("utf-8", "replace")).hexdigest()
        clave = (model or getattr(self, "model", ""), len(texts), huella)

        # El lock serializa el primer calculo. Si llegan 5 consultas juntas, una
        # embebe el corpus y las otras cuatro esperan y reusan su resultado, en vez
        # de embeberlo cinco veces en paralelo.
        async with _chunk_lock:
            if clave not in _chunk_cache:
                log.info("GIRU: embebiendo %d chunks del corpus (primera vez)", len(texts))
                _chunk_cache[clave] = await _orig_embed_safe(
                    self, texts, endpoint=endpoint, model=model, tag=tag, session_id=session_id
                )
            return _chunk_cache[clave]

    EmbeddingClient.embed_safe = _embed_safe_cached  # type: ignore[assignment]

    _activado = True
    log.info("GIRU: caches de artefactos y chunks activados")


def limpiar() -> None:
    """Vaciar tras reindexar o al cambiar de grafo (GIRU_PROYECTO)."""
    _artifacts_cache.clear()
    _chunk_cache.clear()
    log.info("GIRU: caches vaciados")


def estado() -> dict:
    """Para exponer en /diagnostico."""
    return {
        "activado": _activado,
        "grafos_en_memoria": list(_artifacts_cache.keys()),
        "corpus_embebidos": len(_chunk_cache),
    }


async def precalentar(grail_obj, pregunta: str = "reglamento") -> float:
    """Corre una consulta cascade al levantar la API para dejar los artefactos y
    los embeddings del corpus ya en memoria, de modo que la primera consulta real
    no pague ese costo (~40 s en GIRU_IA_72b).

    Devuelve los segundos que tomo. No propaga excepciones: la API debe levantar
    igual si esto falla, y en ese caso la primera consulta real pagara el costo.
    """
    import time

    t0 = time.perf_counter()
    try:
        await grail_obj.search(pregunta, mode="cascade")
    except Exception as exc:  # noqa: BLE001
        log.warning("GIRU: precalentamiento fallo (%s); se sigue sin el", exc)
    return time.perf_counter() - t0