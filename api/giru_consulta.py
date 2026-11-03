"""giru_consulta.py — Que la forma de preguntar no cambie la respuesta (GIRU).

El problema
-----------
La recuperacion de cascade depende de las palabras exactas de la pregunta:

  - el BM25 puntua con la pregunta TAL CUAL (cascade_search.py, paso 4);
  - la puerta de entidades (top-10 por embedding) cambia con pocas palabras;
  - el lenguaje del usuario no es el de los reglamentos ("me eche un ramo" vs
    "reprobacion de asignaturas"; "reglamento organico de la administracion
    superior" vs "ROAS").

Caso medido: "¿que reglamento rige el ROAS?" recupero el 362/2024; "¿cual es el
reglamento organico vigente de la administracion superior?" no lo recupero.

Que hace
--------
Antes de buscar, arma una consulta EXPANDIDA en dos capas:

  1. Glosario de siglas (determinista, glosario_siglas.json): si la pregunta trae
     una sigla conocida se agrega su forma larga, y si trae la forma larga se
     agrega la sigla. Sin LLM: no puede inventar.
  2. Reformulacion con el LLM (una llamada corta, temperatura 0): pregunta
     autonoma (resuelve "eso", "ese reglamento" con el historial) y terminos en
     el vocabulario de la normativa. Se le prohibe inventar numeros de decreto,
     anios, nombres de personas y significados de siglas.

La consulta que recibe GRAIL es:

    <pregunta original>
    [[GIRU_EXPANSION]]
    <pregunta autonoma>
    <terminos>

BM25 y embeddings ven TODO; el prompt de respuesta (local_search.py) corta en la
marca, asi que el modelo redacta frente a la pregunta original del usuario.
Solo se aplica en local, cascade y document (agent y global ponen la consulta
tal cual en sus prompts y la marca quedaria a la vista).

    GIRU_EXPANSION=1          activar (por defecto 0 = consulta tal cual)
    GIRU_EXPANSION_LLM=1      usar la capa 2 (0 = solo glosario)
    GIRU_EXPANSION_MODELO     modelo (por defecto, el de generacion)
"""
from __future__ import annotations

import json
import logging
import os
import re
import unicodedata
from pathlib import Path
from typing import Any, Optional

log = logging.getLogger("giru.consulta")

MARCA = "[[GIRU_EXPANSION]]"
ACTIVADO = os.getenv("GIRU_EXPANSION", "0") == "1"   # apagado por defecto
USAR_LLM = os.getenv("GIRU_EXPANSION_LLM", "1") == "1"
MODELO = os.getenv("GIRU_EXPANSION_MODELO", "")
MODOS = {"local", "cascade", "document"}
GLOSARIO_RUTA = Path(os.getenv(
    "GIRU_GLOSARIO", Path(__file__).resolve().parent / "glosario_siglas.json"))

PROMPT = """\
Preparas búsquedas sobre la normativa de la Universidad Técnica Federico Santa \
María (decretos y reglamentos). No respondes la pregunta: solo la reescribes para \
buscarla mejor.

{historial}Pregunta del usuario: {pregunta}

Devuelve SOLO este JSON:
{{"pregunta_autonoma": "...", "terminos": ["...", "..."]}}

- pregunta_autonoma: la pregunta reescrita para que se entienda sola (si dice \
"eso", "ese reglamento" o "y para los de sede", complétala con los turnos \
previos), en el lenguaje formal de los reglamentos: "ramo" → "asignatura", \
"echarse un ramo" → "reprobar una asignatura", "profe" → "académico", \
"la U" → "la Universidad".
- terminos: de 3 a 8 palabras o frases con que un reglamento nombraría este \
tema: nombre formal del procedimiento, del órgano o del reglamento, y sinónimos.
- NO inventes números de decreto, años ni nombres de personas.
- Si la pregunta trae una sigla y no sabes con certeza qué significa, déjala tal \
cual: NO adivines su significado por sus letras.
"""


def _norm(s: str) -> str:
    s = unicodedata.normalize("NFD", s or "")
    return "".join(c for c in s if unicodedata.category(c) != "Mn").lower()


# --------------------------------------------------------------- glosario
_glosario: Optional[dict[str, str]] = None


def glosario() -> dict[str, str]:
    global _glosario
    if _glosario is None:
        try:
            datos = json.loads(GLOSARIO_RUTA.read_text(encoding="utf-8"))
            _glosario = {k: v for k, v in datos.items() if not k.startswith("_")}
        except (OSError, ValueError):
            log.warning("GIRU: no pude leer el glosario %s", GLOSARIO_RUTA)
            _glosario = {}
    return _glosario


def expandir_siglas(texto: str) -> list[str]:
    """Formas largas de las siglas presentes, y siglas de las formas largas."""
    extra: list[str] = []
    t = _norm(texto)
    for sigla, largo in glosario().items():
        if re.search(rf"\b{re.escape(sigla.lower())}\b", t):
            extra.append(largo)
        elif _norm(largo) in t:
            extra.append(sigla)
    return extra


# --------------------------------------------------------------- LLM
async def _reformular(ollama: str, modelo: str, pregunta: str,
                      historial: list[dict[str, Any]]) -> dict[str, Any]:
    import httpx

    previos = [h["content"] for h in historial if h.get("role") == "user"][-2:]
    hist = ("Turnos previos del usuario:\n" + "\n".join(f"- {p}" for p in previos) + "\n\n"
            if previos else "")
    async with httpx.AsyncClient(timeout=60) as cli:
        r = await cli.post(f"{ollama}/api/chat", json={
            "model": modelo,
            "messages": [{"role": "user",
                          "content": PROMPT.format(historial=hist, pregunta=pregunta)}],
            "format": "json",
            "stream": False,
            "options": {"temperature": 0},
        })
        r.raise_for_status()
        contenido = (r.json().get("message") or {}).get("content", "")
    datos = json.loads(contenido)
    return datos if isinstance(datos, dict) else {}


_RE_DECRETO = re.compile(r"\b\d{1,4}\s*[/.-]\s*(19|20)?\d{2}\b")


def _limpiar_terminos(terminos: Any, pregunta: str) -> list[str]:
    """Descarta lo que el modelo no deberia haber inventado: numeros de decreto
    que no estaban en la pregunta."""
    if not isinstance(terminos, list):
        return []
    salida = []
    for t in terminos:
        t = str(t).strip()
        if not t or len(t) > 80:
            continue
        if _RE_DECRETO.search(t) and not _RE_DECRETO.search(pregunta):
            continue
        salida.append(t)
    return salida[:8]


async def expandir(pregunta: str, modo: str, historial: list[dict[str, Any]],
                   *, ollama: str, modelo_defecto: str) -> str:
    """Consulta para GRAIL. Si algo falla, devuelve la pregunta tal cual."""
    if not ACTIVADO or modo not in MODOS:
        return pregunta
    partes: list[str] = []
    if USAR_LLM:
        try:
            d = await _reformular(ollama, MODELO or modelo_defecto, pregunta, historial)
            autonoma = str(d.get("pregunta_autonoma") or "").strip()
            if autonoma and _norm(autonoma) != _norm(pregunta):
                if not (_RE_DECRETO.search(autonoma) and not _RE_DECRETO.search(pregunta)):
                    partes.append(autonoma)
            partes.extend(_limpiar_terminos(d.get("terminos"), pregunta))
        except Exception:                                   # noqa: BLE001
            log.warning("GIRU: fallo la reformulacion; se busca con la pregunta tal cual")
    partes.extend(expandir_siglas(" ".join([pregunta, *partes])))
    if not partes:
        return pregunta
    vistos, unicos = set(), []
    for p in partes:
        if _norm(p) not in vistos:
            vistos.add(_norm(p))
            unicos.append(p)
    consulta = f"{pregunta}\n\n{MARCA}\n" + "\n".join(unicos)
    log.info("GIRU: consulta expandida -> %s", " | ".join(unicos))
    return consulta


def pregunta_visible(texto: str) -> str:
    """Lo que el prompt de respuesta debe ver: todo antes de la marca."""
    return (texto or "").split(MARCA, 1)[0].strip()
