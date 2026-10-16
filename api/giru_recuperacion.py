"""giru_recuperacion.py — Abre el rescate por texto de cascade, que GRAIL deja
cerrado en 5 y no expone en grail.yaml.

El problema
-----------
cascade_search.py declara:

    top_k_entities: int = 15
    top_k_rescue:   int = 5

`core.py` pasa `top_k_entities` desde `search.local_top_k_entities`, pero NO pasa
`top_k_rescue`: se queda en 5 pase lo que pase.

Ese rescate es el que busca por BM25 + coseno sobre TODOS los fragmentos, fuera de
la puerta de entidades. Es justamente el camino que debería salvar una consulta
cuando el grafo no tiene la entidad adecuada — y está limitado a 5 fragmentos de
1125. La medición del 16-10-2026 sobre 40 preguntas del corpus mostró que en ~70%
de los casos el documento correcto nunca entra al contexto, con recall@1 (27,5%)
casi igual a recall@5 (30%): el ordenamiento funciona, lo que falta es que el
documento llegue.

Cómo
----
`core.search()` importa CascadeSearch DENTRO de la función, así que reemplazar el
nombre en el módulo sí toma efecto. Se envuelve la clase en una fábrica que
inyecta el valor; core.py la construye solo con argumentos por nombre.

Uso
---
En grail_svc.py, junto a giru_cache.activar():

    import giru_recuperacion
    giru_recuperacion.activar()

Se controla por variable de entorno, sin tocar código:

    GIRU_TOP_K_RESCATE=30     (0 o sin definir = se deja el 5 de GRAIL)

Subirlo agranda el contexto recuperado, así que vigila `local_max_tokens`: con
10000 y text_unit_prop 0.5 hay ~5000 tokens para fragmentos, unos 2-3 chunks de
2000. Pasado ese techo los fragmentos extra se descartan al armar el contexto y
subir el rescate deja de servir. Si el rescate no mueve la aguja, el siguiente
paso es subir local_max_tokens o text_unit_prop, no seguir subiendo esto.
"""
from __future__ import annotations

import logging
import os

log = logging.getLogger(__name__)

TOP_K_RESCATE = int(os.getenv("GIRU_TOP_K_RESCATE", "0"))

_activado = False


def activar() -> None:
    """Idempotente. Cada parche se aplica solo si su variable lo pide."""
    global _activado
    if _activado:
        return

    import grail.query.cascade_search as cs

    if TOP_K_RESCATE > 0:
        _Original = cs.CascadeSearch

        def _con_rescate(*args, **kwargs):
            # setdefault y no asignación directa: si algún día core.py empieza a
            # pasarlo desde la configuración, gana la configuración.
            kwargs.setdefault("top_k_rescue", TOP_K_RESCATE)
            return _Original(*args, **kwargs)

        _con_rescate._giru_original = _Original   # type: ignore[attr-defined]
        cs.CascadeSearch = _con_rescate           # type: ignore[assignment]
        log.info("GIRU: top_k_rescue de cascade elevado a %d (GRAIL trae 5)", TOP_K_RESCATE)

    if RESPETAR_RANKING:
        # SOLO el nombre que importó cascade_search. local y document siguen
        # con el original, donde el filtro por entidad sí es necesario.
        cs.build_text_unit_context = _construir_contexto_fragmentos
        log.info("GIRU: cascade conserva su orden de relevancia (nombre=%s)", NOMBRE_EN_FUENTES)

    _activado = True


def estado() -> dict:
    return {
        "activado": _activado,
        "top_k_rescate": TOP_K_RESCATE or 5,
        "respetar_ranking": RESPETAR_RANKING,
        "nombre_en_fuentes": NOMBRE_EN_FUENTES,
    }

# ---------------------------------------------------------------------------
# Respetar el ranking de cascade dentro del contexto de fragmentos
# ---------------------------------------------------------------------------
# cascade ordena sus fragmentos por BM25 + coseno contra la consulta
# (cascade_search.py paso 8, `reranked_text_units`) y se los pasa a
# build_text_unit_context. Esa funcion hace dos cosas que anulan ese trabajo
# (grail/query/retrieval.py ~396):
#
#   1. DESCARTA todo fragmento que no mencione una de las entidades
#      seleccionadas -> los rescatados por texto, que son justamente los que
#      entran cuando el grafo no tiene la entidad adecuada, se pierden casi
#      todos.
#   2. RE-ORDENA por cuantas entidades seleccionadas menciona cada fragmento
#      -> ganan los fragmentos genericos, los de los nodos concentradores, por
#      encima del que responde la pregunta.
#
# Medido el 16-10-2026: subir local_top_k_entities de 10 a 25 BAJO el recall@1
# de 27,5% a 20,0%, porque con mas entidades hay mas fragmentos genericos que
# desplazan al especifico. El sintoma es contraintuitivo (mas contexto, peor
# resultado) y el mecanismo es este.
#
# El parche se aplica SOLO al nombre que importo cascade_search, asi que local
# y document siguen con el comportamiento original: ahi el filtro por entidad
# si hace falta, porque local pasa los 1125 fragmentos sin ordenar y la puerta
# de entidades es su unico criterio.
#
#   GIRU_RESPETAR_RANKING=1    no filtrar, conservar el orden de cascade
#   GIRU_NOMBRE_EN_FUENTES=1   agregar el nombre del documento a cada fragmento
#
# Lo segundo es aparte y vale medirlo solo: hoy el modelo ve
# `<source id="uuid" document_id="uuid">`, puro identificador. El prompt le pide
# citar la fuente por su nombre, pero el nombre no esta en ninguna parte del
# bloque de fragmentos.

RESPETAR_RANKING = os.getenv("GIRU_RESPETAR_RANKING", "0") == "1"
NOMBRE_EN_FUENTES = os.getenv("GIRU_NOMBRE_EN_FUENTES", "0") == "1"


def _construir_contexto_fragmentos(
    text_units, entity_names, *, max_tokens=3000, documents=None, mapping=None
):
    """Reemplazo de build_text_unit_context para cascade.

    Mismo formato de salida que el original (mismas etiquetas, mismo contrato de
    retorno), pero toma los fragmentos en el orden en que llegan -que en cascade
    es el orden de relevancia- y no descarta ninguno.
    """
    from grail.query.retrieval import resolve_doc_titles, tiktoken_len

    if text_units is None or text_units.empty:
        return "", text_units

    titulos = resolve_doc_titles(documents, mapping)

    cabecera, pie = "<sources>", "</sources>"
    lineas, elegidos = [], []
    usado = tiktoken_len(cabecera) + tiktoken_len(pie)

    for idx, fila in text_units.iterrows():
        crudo = fila.get("document_ids")
        if crudo is None or (hasattr(crudo, "__len__") and len(crudo) == 0):
            doc_ids = [fila.get("document_id")]
        else:
            doc_ids = list(crudo)
        doc_ids = [d for d in doc_ids if d]
        doc_id_str = "; ".join(str(d) for d in doc_ids)

        atributo_nombre = ""
        if NOMBRE_EN_FUENTES and doc_ids:
            nombres = []
            for d in doc_ids:
                t = titulos.get(str(d)) or titulos.get(d)
                if t:
                    nombres.append(
                        str(t).replace("\\", "/").rstrip("/").rsplit("/", 1)[-1].replace('"', "'")
                    )
            if nombres:
                atributo_nombre = f' document="{"; ".join(nombres)}"'

        texto = (fila.get("text") or "").replace(chr(10), " ")
        linea = (
            f'<source id="{fila["id"]}" document_id="{doc_id_str}"{atributo_nombre}>\n'
            f"{texto}\n</source>"
        )
        tok = tiktoken_len(linea)
        if usado + tok > max_tokens:
            break
        lineas.append(linea)
        elegidos.append(idx)
        usado += tok

    return "\n".join([cabecera, *lineas, pie]), text_units.loc[elegidos]