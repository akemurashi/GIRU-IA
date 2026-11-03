"""giru_atribucion.py — Atribucion de fuentes en una segunda pasada (GIRU).

El problema
-----------
Con las citas verificadas (giru_citas.py) ya no hay citas FALSAS, pero muchas
veces la respuesta es correcta y el documento citado no lo es. Causas medidas:

  1. El modelo ve los fragmentos de GRAIL como <source id="uuid">, sin nombre.
  2. Los fragmentos de GRAIL son de 2000 tokens y pueden cruzar dos decretos.
  3. Cuando el documento correcto llega, compite con OTRAS VERSIONES del mismo
     reglamento (291/2017 vs Premio Excelencia 2006, 056/2018 vs DEC.REC 24-94).
  4. La fuente se decide por lo que el modelo RECUPERO, no por lo que AFIRMO.

La idea
-------
Separar "redactar" de "citar". Despues de que GRAIL responde:

  a. Se parte la respuesta en afirmaciones.
  b. Cada afirmacion (+ la pregunta) se busca en el indice FINO de chunks del
     buscador (dev\\buscador: ~7.700 chunks por articulo, un solo documento por
     chunk, con titulo, anio y seccion). Busqueda hibrida coseno + BM25, con un
     bono para los documentos que GRAIL ya habia recuperado.
  c. Un juez LLM (el mismo qwen2.5:32b, una sola llamada) decide que chunks
     respaldan EXPLICITAMENTE cada afirmacion. Los chunks van numerados [1]..[N],
     asi el modelo nunca copia identificadores.
  d. Si varias versiones respaldan lo mismo, gana la vigente / mas reciente.
  e. El fragmento que se devuelve es una oracion del chunk que se comprueba
     LITERALMENTE contra el .md de input/ (se mantiene la garantia de 0 citas
     falsas). Si no hay oracion literal, el documento va con fragmento null.

Prueba offline (31-10-2026, solo BM25, sin juez ni coseno, 40 preguntas de
eval_E): buscar con pregunta + respuesta en el indice de chunks pone el
documento correcto primero en 15/40 y en el top-5 en 24/40; el ranking actual de
GRAIL lo hacia en 10/40 y 15/40.

Variables de entorno
--------------------
  GIRU_ATRIBUCION=1          activar (por defecto 0 = comportamiento anterior)
  GIRU_ATRIB_MODELO          modelo del juez (por defecto, el de generacion)
  GIRU_ATRIB_CANDIDATOS=18   chunks que ve el juez
  GIRU_ATRIB_AFIRMACIONES=8  afirmaciones maximas por respuesta
  GIRU_INPUT                 carpeta de los .md (por defecto <proyecto>\\input)
  GIRU_ESTADOS               JSON opcional con el estado por documento
                             (estado_documentos.json de la entrega a la BD)
"""
from __future__ import annotations

import json
import logging
import math
import os
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Optional

import numpy as np

log = logging.getLogger("giru.atribucion")

ACTIVADO = os.getenv("GIRU_ATRIBUCION", "0") == "1"   # apagado por defecto
MODELO_JUEZ = os.getenv("GIRU_ATRIB_MODELO", "")
N_CANDIDATOS = int(os.getenv("GIRU_ATRIB_CANDIDATOS", "18"))
N_AFIRMACIONES = int(os.getenv("GIRU_ATRIB_AFIRMACIONES", "8"))
POR_AFIRMACION = 5            # chunks que aporta cada afirmacion al pool
BONO_CONTEXTO = 0.15          # documento que GRAIL ya habia recuperado
ALFA = 0.5                    # peso del coseno frente a BM25 (igual que /buscar)
PALABRAS_JUEZ = 220           # se recorta cada chunk para el juez
MIN_CARACTERES_CITA = 30

RE_DEROGADO = re.compile(r"derogad|no[ _-]?vigente|\bNV\b", re.I)


# =================================================================== texto
def normalizar(s: Any) -> str:
    s = unicodedata.normalize("NFD", str(s or ""))
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    return re.sub(r"\s+", " ", s).strip().lower()


def _limpiar_md(s: str) -> str:
    s = re.sub(r"[*_`#>]+", " ", s)
    s = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", s)
    return re.sub(r"\s+", " ", s).strip()


_RE_ORACION = re.compile(r"(?<=[.;:!?])\s+(?=[\"“(¿¡A-ZÁÉÍÓÚÑ0-9])")
_RE_VIÑETA = re.compile(r"^\s*(?:[-*•]|\d+[.)]|[a-z][.)])\s+")
_RE_RELLENO = re.compile(
    r"^(si (necesitas|tienes|requieres|deseas)|te recomiendo|para m[aá]s (informaci[oó]n|detalles)|"
    r"espero que|no (hay|encontr[eé]|tengo|se encontr[oó]|dispongo)|los datos (no|disponibles)|"
    r"soy don fede|¿)", re.I)


def afirmaciones(respuesta: str, maximo: int = N_AFIRMACIONES) -> list[str]:
    """Oraciones con contenido verificable.

    Los items cortos de una lista ("- Decreto 039/2019") no se sostienen solos:
    se pegan a la oracion que los introduce ("El 383/2023 deroga: ...").
    """
    salida: list[str] = []
    for linea in (respuesta or "").splitlines():
        es_vineta = bool(_RE_VIÑETA.match(linea))
        linea = _limpiar_md(_RE_VIÑETA.sub("", linea))
        if not linea:
            continue
        if es_vineta and len(linea.split()) < 7 and salida and len(salida[-1]) < 500:
            salida[-1] += (" " if salida[-1].endswith(":") else "; ") + linea
            continue
        for o in _RE_ORACION.split(linea):
            o = o.strip()
            if len(o.split()) >= 6 and not _RE_RELLENO.match(o):
                salida.append(o)
    vistas, unicas = set(), []
    for o in salida:
        k = normalizar(o)[:120]
        if k not in vistas:
            vistas.add(k)
            unicas.append(o)
    return unicas[:maximo]


def _tokens(texto: str) -> list[str]:
    t = normalizar(texto)
    return [w.lstrip("0") or "0" if w.isdigit() else w
            for w in re.findall(r"\w+", t) if len(w) > 1 or w.isdigit()]


class _BM25:
    """Misma formula que buscador/busqueda.py (copiada para no depender de su import)."""

    def __init__(self, docs: list[str], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.largos = np.zeros(len(docs), dtype=np.float32)
        self.indice: dict[str, list[tuple[int, int]]] = defaultdict(list)
        for i, d in enumerate(docs):
            tf = Counter(_tokens(d))
            self.largos[i] = sum(tf.values())
            for term, n in tf.items():
                self.indice[term].append((i, n))
        self.n = len(docs)
        self.prom = float(self.largos.mean()) if self.n else 0.0

    def puntuar(self, consulta: str) -> np.ndarray:
        s = np.zeros(self.n, dtype=np.float32)
        for term in set(_tokens(consulta)):
            post = self.indice.get(term)
            if not post:
                continue
            idf = math.log(1 + (self.n - len(post) + 0.5) / (len(post) + 0.5))
            ids = np.fromiter((p[0] for p in post), dtype=np.int64)
            tf = np.fromiter((p[1] for p in post), dtype=np.float32)
            den = tf + self.k1 * (1 - self.b + self.b * self.largos[ids] / self.prom)
            s[ids] += idf * tf * (self.k1 + 1) / den
        return s


# ================================================================== indice
class IndiceChunks:
    """La tabla 'chunks' del buscador, con metadatos que Buscador descarta."""

    def __init__(self, ruta_lancedb: Path, carpeta_input: Path,
                 estados: Optional[dict[str, str]] = None):
        import lancedb

        df = lancedb.connect(str(ruta_lancedb)).open_table("chunks").to_pandas()
        self.documento = df["documento"].astype(str).tolist()
        self.ruta = (df["ruta"] if "ruta" in df.columns else df["documento"]).astype(str).tolist()
        self.seccion = (df["seccion"] if "seccion" in df.columns else "").astype(str).tolist()
        self.anio = [self._anio(a, r) for a, r in zip(
            df["anio"] if "anio" in df.columns else [""] * len(df), self.ruta)]
        self.texto = df["texto"].astype(str).tolist()
        m = np.vstack(df["vector"].to_numpy()).astype(np.float32)
        self.matriz = m / (np.linalg.norm(m, axis=1, keepdims=True) + 1e-9)
        self.bm25 = _BM25(df["texto_embed"].astype(str).tolist())
        self.input = carpeta_input
        self.estados = estados or {}
        self._md: dict[str, str] = {}

    @staticmethod
    def _anio(valor: Any, ruta: str) -> int:
        for v in (str(valor or ""), ruta):
            m = re.search(r"(19|20)\d{2}", v)
            if m:
                return int(m.group(0))
        return 0

    def md_normalizado(self, i: int) -> str:
        """Texto del .md original, normalizado, para comprobar citas literales."""
        clave = self.ruta[i]
        if clave not in self._md:
            p = self.input / clave
            if not p.exists():
                encontrados = list(self.input.rglob(self.documento[i]))
                p = encontrados[0] if encontrados else p
            try:
                self._md[clave] = normalizar(p.read_text(encoding="utf-8", errors="replace"))
            except OSError:
                self._md[clave] = ""
        return self._md[clave]

    def derogado(self, i: int) -> bool:
        est = self.estados.get(self.documento[i], "")
        return est.upper().startswith("DEROG") or bool(RE_DEROGADO.search(self.documento[i]))

    def puntuar(self, consulta: str, vec: np.ndarray, docs_contexto: set[str]) -> np.ndarray:
        cos = self.matriz @ vec
        bm = self.bm25.puntuar(consulta)
        s = ALFA * np.clip(cos, 0, None) / max(float(cos.max()), 1e-9) \
            + (1 - ALFA) * bm / max(float(bm.max()), 1e-9)
        if docs_contexto:
            bono = np.fromiter((d in docs_contexto for d in self.documento), dtype=bool)
            s = s + BONO_CONTEXTO * bono
        return s


def cargar_estados(ruta: str) -> dict[str, str]:
    """estado_documentos.json -> {nombre_archivo: estado}. Tolera lista o dict."""
    if not ruta or not Path(ruta).exists():
        return {}
    try:
        datos = json.loads(Path(ruta).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    if isinstance(datos, dict) and "documentos" in datos:
        datos = datos["documentos"]
    salida: dict[str, str] = {}
    if isinstance(datos, list):
        for f in datos:
            if isinstance(f, dict):
                n = f.get("nombre_archivo") or f.get("documento")
                e = f.get("estado") or f.get("estado_documento")
                if n and e:
                    salida[str(n)] = str(e)
    elif isinstance(datos, dict):
        salida = {str(k): str(v.get("estado") if isinstance(v, dict) else v) for k, v in datos.items()}
    return salida


# ================================================================== juez
PROMPT_JUEZ = """\
Eres un verificador de citas para la normativa de la Universidad Técnica Federico \
Santa María. Te doy AFIRMACIONES de una respuesta y FRAGMENTOS numerados de \
documentos oficiales.

Para cada afirmación, indica qué fragmentos la RESPALDAN EXPLÍCITAMENTE: el \
fragmento dice lo mismo que la afirmación (mismo dato, cifra, plazo, órgano o \
regla). No basta con que trate el mismo tema.

- Si varios fragmentos de documentos distintos dicen lo mismo (por ejemplo, \
versiones del mismo reglamento), inclúyelos todos.
- Ordena los números del que mejor respalda al que menos.
- Si ningún fragmento respalda una afirmación, deja su lista vacía.

Responde SOLO con JSON, una clave por afirmación, por ejemplo:
{{"A1": [3, 7], "A2": [], "A3": [1]}}

AFIRMACIONES:
{afirmaciones}

FRAGMENTOS:
{fragmentos}
"""


def _recortar(texto: str, n: int = PALABRAS_JUEZ) -> str:
    p = texto.split()
    return " ".join(p[:n]) + (" …" if len(p) > n else "")


def _parsear_juez(texto: str, n_afirm: int, n_frag: int) -> dict[int, list[int]]:
    m = re.search(r"\{.*\}", texto or "", re.S)
    if not m:
        return {}
    try:
        datos = json.loads(m.group(0))
    except ValueError:
        return {}
    salida: dict[int, list[int]] = {}
    for k, v in (datos.items() if isinstance(datos, dict) else []):
        mk = re.search(r"\d+", str(k))
        if not mk:
            continue
        a = int(mk.group(0))
        if not 1 <= a <= n_afirm or not isinstance(v, list):
            continue
        nums = []
        for x in v:
            try:
                x = int(str(x).strip("[] "))
            except ValueError:
                continue
            if 1 <= x <= n_frag and x not in nums:
                nums.append(x)
        salida[a] = nums
    return salida


# ============================================================ fragmento literal
def _oraciones(texto: str) -> list[str]:
    partes = re.split(r"(?<=[.;:])\s+", texto)
    return [p.strip() for p in partes if len(p.strip()) >= MIN_CARACTERES_CITA]


def fragmento_literal(indice: IndiceChunks, i: int, afirmacion: str) -> Optional[str]:
    """La oracion del chunk que mas se parece a la afirmacion Y existe tal cual
    en el .md. Si ninguna pasa, None (el documento igual se devuelve)."""
    md = indice.md_normalizado(i)
    if not md:
        return None
    t_afirm = set(_tokens(afirmacion))
    candidatas = sorted(
        _oraciones(indice.texto[i]),
        key=lambda o: -len(t_afirm & set(_tokens(o))),
    )
    for o in candidatas[:6]:
        o = o[:450].rsplit(" ", 1)[0] if len(o) > 450 else o
        if normalizar(o) in md:
            return o.lstrip("-•* ").strip()
    return None


# ================================================================ principal
def _orden_version(indice: IndiceChunks, i: int, rango: int) -> tuple:
    """Entre chunks que respaldan lo mismo: vigente antes que derogado, despues el
    mas reciente, despues el mejor rankeado por el juez."""
    return (indice.derogado(i), -indice.anio[i], rango)


def atribuir(
    indice: IndiceChunks,
    pregunta: str,
    respuesta: str,
    vectores: list[np.ndarray],
    afirm: list[str],
    texto_juez: str,
    pool: list[int],
) -> dict[str, Any]:
    """Paso puro (sin red): dado el veredicto del juez, arma las fuentes."""
    veredicto = _parsear_juez(texto_juez, len(afirm), len(pool))
    votos: dict[str, float] = {}
    mejor_chunk: dict[str, tuple[int, str]] = {}
    respaldadas = 0
    for a_idx, afirmacion in enumerate(afirm, start=1):
        nums = veredicto.get(a_idx) or []
        if not nums:
            continue
        respaldadas += 1
        chunks = [pool[n - 1] for n in nums]
        rango = {c: r for r, c in enumerate(chunks)}
        chunks = sorted(chunks, key=lambda c: _orden_version(indice, c, rango[c]))
        elegido = chunks[0]
        doc = indice.documento[elegido]
        votos[doc] = votos.get(doc, 0.0) + 1.0
        # Los demas documentos que tambien respaldan suman poco: desempatan.
        for c in chunks[1:]:
            d = indice.documento[c]
            if d != doc:
                votos[d] = votos.get(d, 0.0) + 0.1
        mejor_chunk.setdefault(doc, (elegido, afirmacion))

    orden = sorted(votos.items(), key=lambda kv: -kv[1])
    fuentes = []
    for doc, _ in orden:
        if doc not in mejor_chunk:
            continue                         # solo desempato, nunca fue el elegido
        i, afirmacion = mejor_chunk[doc]
        fuentes.append({
            "documento": doc,
            "referencia": None,
            "fragmento": fragmento_literal(indice, i, afirmacion),
            "seccion": indice.seccion[i] or None,
        })
    return {
        "fuentes": fuentes,
        "afirmaciones": len(afirm),
        "respaldadas": respaldadas,
        "respaldo": round(respaldadas / len(afirm), 2) if afirm else None,
    }


def armar_pool(indice: IndiceChunks, pregunta: str, afirm: list[str],
               vectores: list[np.ndarray], docs_contexto: set[str]) -> list[int]:
    """Ronda por afirmaciones: cada una aporta sus mejores chunks hasta el tope."""
    listas = []
    for a, v in zip(afirm, vectores):
        s = indice.puntuar(f"{pregunta} {a}", v, docs_contexto)
        listas.append([int(i) for i in np.argsort(-s)[:POR_AFIRMACION * 3]])
    pool: list[int] = []
    for ronda in range(POR_AFIRMACION * 3):
        for lst in listas:
            if ronda < len(lst) and lst[ronda] not in pool:
                pool.append(lst[ronda])
            if len(pool) >= N_CANDIDATOS:
                return pool
    return pool


def texto_para_juez(indice: IndiceChunks, afirm: list[str], pool: list[int]) -> str:
    a = "\n".join(f"A{i}. {x}" for i, x in enumerate(afirm, 1))
    f = "\n\n".join(
        f"[{n}] {indice.documento[i]} — {indice.seccion[i]} ({indice.anio[i] or 's/a'})\n"
        f"{_recortar(indice.texto[i])}"
        for n, i in enumerate(pool, 1)
    )
    return PROMPT_JUEZ.format(afirmaciones=a, fragmentos=f)


# ========================================================= orquestacion async
_indice: Optional[IndiceChunks] = None


def indice(buscador_dir: Path, proyecto: Path) -> IndiceChunks:
    global _indice
    if _indice is None:
        carpeta = Path(os.getenv("GIRU_INPUT", str(proyecto / "input")))
        _indice = IndiceChunks(buscador_dir / "lancedb_buscador", carpeta,
                               cargar_estados(os.getenv("GIRU_ESTADOS", "")))
        log.info("GIRU: indice de atribucion cargado (%d chunks)", len(_indice.documento))
    return _indice


def olvidar() -> None:
    global _indice
    _indice = None


async def _embeber(ollama: str, modelo: str, textos: list[str]) -> list[np.ndarray]:
    import httpx

    async with httpx.AsyncClient(timeout=120) as cli:
        r = await cli.post(f"{ollama}/api/embed", json={"model": modelo, "input": textos})
        r.raise_for_status()
        vecs = r.json()["embeddings"]
    salida = []
    for v in vecs:
        a = np.asarray(v, dtype=np.float32)
        salida.append(a / (np.linalg.norm(a) + 1e-9))
    return salida


async def _juez(ollama: str, modelo: str, prompt: str) -> str:
    import httpx

    # Sin num_ctx: si se pide un contexto distinto al del modelo cargado,
    # Ollama lo RECARGA y la consulta paga decenas de segundos.
    async with httpx.AsyncClient(timeout=180) as cli:
        r = await cli.post(f"{ollama}/api/chat", json={
            "model": modelo,
            "messages": [{"role": "user", "content": prompt}],
            "format": "json",
            "stream": False,
            "options": {"temperature": 0},
        })
        r.raise_for_status()
        return (r.json().get("message") or {}).get("content", "")


async def atribuir_async(
    *, pregunta: str, respuesta: str, docs_contexto: set[str],
    buscador_dir: Path, proyecto: Path, ollama: str, modelo_emb: str, modelo_juez: str,
) -> Optional[dict[str, Any]]:
    """None si no aplica (respuesta sin afirmaciones verificables) o si algo falla:
    el llamador cae al mecanismo anterior. Nunca rompe una respuesta."""
    import asyncio

    afirm = afirmaciones(respuesta)
    if not afirm:
        return None
    try:
        idx = await asyncio.to_thread(indice, buscador_dir, proyecto)
        vecs = await _embeber(ollama, modelo_emb, [f"{pregunta} {a}" for a in afirm])
        pool = await asyncio.to_thread(armar_pool, idx, pregunta, afirm, vecs, docs_contexto)
        if not pool:
            return None
        crudo = await _juez(ollama, MODELO_JUEZ or modelo_juez, texto_para_juez(idx, afirm, pool))
        res = await asyncio.to_thread(atribuir, idx, pregunta, respuesta, vecs, afirm, crudo, pool)
    except Exception:                                      # noqa: BLE001
        log.exception("GIRU: fallo la atribucion; se usa el mecanismo anterior")
        return None
    log.info("GIRU: atribucion %d/%d afirmaciones respaldadas, %d fuentes",
             res["respaldadas"], res["afirmaciones"], len(res["fuentes"]))
    return res
