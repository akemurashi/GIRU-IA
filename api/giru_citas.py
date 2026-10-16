"""giru_citas.py — Citas verificadas para la API de GIRU.

El modelo cierra su respuesta con un bloque:

    <citas>
    "texto literal copiado de un fragmento"
    "otro texto literal"
    </citas>

Este modulo lo separa de la respuesta visible y VERIFICA cada cita: el texto
tiene que aparecer literalmente en alguno de los fragmentos que se le enviaron
al modelo. Solo las que pasan se devuelven como fuente.

Por que la cita es la llave y no un identificador
-------------------------------------------------
GRAIL le muestra al modelo los fragmentos como
``<source id="bc1398a8-0be2-4581-..." document_id="...">``. Esos ids son uuid4
ALEATORIOS (grail/indexing/loader.py:427 -> generate_guid()), asi que pedirle al
modelo que los transcriba es pedirle un trabajo de copista en el que a veces se
equivoca. Buscando el texto citado dentro de los fragmentos se obtiene lo mismo
-el documento de origen- y ademas la verificacion sale gratis: si el texto no
esta, la cita es inventada y se descarta.

Lo que esto garantiza: ninguna cita falsa. El texto existe, es literal, y viene
del documento que se nombra.
Lo que NO garantiza: que el razonamiento del modelo haya salido de ese pasaje.
Eso es inverificable desde fuera con cualquier arquitectura.
"""
from __future__ import annotations

import re
import unicodedata
from typing import Any, Optional

MARCA_INICIO = "<citas>"
MARCA_FIN = "</citas>"

# Una cita mas corta que esto no identifica nada: "la Universidad" aparece en
# medio corpus. Se mide sobre el texto ya normalizado.
MINIMO_CARACTERES = 30

# raw_content normalizado por documento, en memoria. Se llena perezosamente: solo
# los documentos que son candidatos de alguna cita. Se vacia al reindexar.
_cache_docs: dict[str, str] = {}


def limpiar_cache() -> None:
    _cache_docs.clear()

_RE_BLOQUE = re.compile(
    re.escape(MARCA_INICIO) + r"(.*?)(?:" + re.escape(MARCA_FIN) + r"|$)",
    re.DOTALL | re.IGNORECASE,
)
_RE_ENTRECOMILLADO = re.compile(r'["“”«»]([^"“”«»]{10,})["“”«»]')
_RE_ESPACIOS = re.compile(r"\s+")


# --------------------------------------------------------------- separacion
def separar(texto: Any) -> tuple[str, list[str]]:
    """Parte la respuesta en (texto visible, citas crudas).

    Tolera que el modelo olvide cerrar el bloque: en ese caso toma hasta el
    final. Si no hay bloque, devuelve el texto intacto y lista vacia.
    """
    if not isinstance(texto, str) or MARCA_INICIO.lower() not in texto.lower():
        return (texto if isinstance(texto, str) else str(texto or "")), []

    m = _RE_BLOQUE.search(texto)
    if not m:
        return texto, []

    cuerpo = m.group(1)
    visible = (texto[: m.start()] + texto[m.end():]).strip()

    citas: list[str] = []
    for c in _RE_ENTRECOMILLADO.findall(cuerpo):
        c = c.strip()
        if c:
            citas.append(c)
    if not citas:
        # El modelo puede haber puesto una cita por linea sin comillas.
        for linea in cuerpo.splitlines():
            linea = linea.strip().lstrip("-*•").strip()
            linea = re.sub(r"^\[\d+\]\s*", "", linea)   # por si numera
            if len(linea) >= 10:
                citas.append(linea)
    return visible, citas


# ------------------------------------------------------------ verificacion
def _normalizar(s: Any) -> str:
    """Minusculas, sin tildes y con los espacios colapsados.

    Las tildes se quitan porque el corpus viene de OCR y una 'o' con tilde mal
    reconocida no deberia invalidar una cita por lo demas correcta. Se compara
    normalizado, pero se DEVUELVE el texto original del fragmento.
    """
    s = str(s or "")
    s = unicodedata.normalize("NFD", s)
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    return _RE_ESPACIOS.sub(" ", s).strip().lower()


def _trozo_mas_largo(cita: str) -> str:
    """Si el modelo abrevio con puntos suspensivos, se valida el tramo mayor."""
    partes = re.split(r"\.{3}|…|\[\.\.\.\]", cita)
    return max(partes, key=len).strip() if len(partes) > 1 else cita.strip()


def _texto_documento(doc_id: str, documentos) -> str:
    """raw_content del documento, normalizado y cacheado."""
    clave = str(doc_id)
    if clave in _cache_docs:
        return _cache_docs[clave]
    texto = ""
    if documentos is not None and hasattr(documentos, "empty") and not documentos.empty:
        if "id" in documentos.columns and "raw_content" in documentos.columns:
            fila = documentos[documentos["id"].astype(str) == clave]
            if not fila.empty:
                texto = str(fila.iloc[0].get("raw_content") or "")
    _cache_docs[clave] = _normalizar(texto)
    return _cache_docs[clave]


def verificar(
    citas: list[str],
    filas_fuente,
    titulos: dict[str, str],
    *,
    documentos=None,
    maximo: int = 5,
) -> list[dict[str, Optional[str]]]:
    """Devuelve [{documento, fragmento}] solo con las citas comprobadas.

    `filas_fuente` es context_data["sources"]: los fragmentos que de verdad se
    le enviaron al modelo, ya ordenados por relevancia. Una cita vale si su
    texto aparece dentro de alguno de ellos. Si aparece en varios, gana el mejor
    rankeado (el de indice mas bajo).

    `documentos` es artefactos().documents, que trae raw_content. Hace falta por
    esto: el loader de GRAIL concatena TODOS los documentos en un solo texto
    antes de cortar los chunks de 2000 tokens, asi que un chunk puede mezclar el
    final de un decreto con el principio del siguiente. En ese caso su
    document_ids trae DOS documentos y quedarse con el primero atribuye la cita
    al decreto equivocado (caso real: texto del 383/2023 firmado como 321/2023).
    Para evitarlo se comprueba en cual de los documentos candidatos aparece de
    verdad el texto citado. Si no aparece entero en ninguno, la cita se descarta:
    mejor ninguna cita que una mal atribuida.
    """
    if not citas or filas_fuente is None or not hasattr(filas_fuente, "empty"):
        return []
    if filas_fuente.empty:
        return []

    # Se normaliza cada fragmento una sola vez.
    fragmentos: list[tuple[str, str, list[str]]] = []   # (normalizado, original, doc_ids)
    for _, fila in filas_fuente.iterrows():
        texto = fila.get("text") or ""
        crudo = fila.get("document_ids")
        if crudo is None or (hasattr(crudo, "__len__") and len(crudo) == 0):
            uno = fila.get("document_id")
            dids = [str(uno)] if uno else []
        elif isinstance(crudo, str):
            dids = [crudo]
        else:
            try:
                dids = [str(d) for d in crudo if d]
            except TypeError:
                dids = [str(crudo)]
        fragmentos.append((_normalizar(texto), str(texto), dids))

    salida: list[dict[str, Optional[str]]] = []
    ya: set[tuple[str, str]] = set()

    for cita in citas:
        nucleo = _trozo_mas_largo(cita)
        aguja = _normalizar(nucleo)
        if len(aguja) < MINIMO_CARACTERES:
            continue                      # demasiado corta para identificar nada
        for norm, _original, dids in fragmentos:
            pos = norm.find(aguja)
            if pos < 0:
                continue

            # Cual de los documentos del chunk contiene REALMENTE la cita.
            elegido = ""
            if len(dids) == 1:
                elegido = dids[0]
            else:
                for did in dids:
                    if aguja and aguja in _texto_documento(did, documentos):
                        elegido = did
                        break
            if not elegido:
                # El chunk cruza un limite de documento y la cita no aparece
                # entera en ninguno de los dos: no se puede atribuir. Se sigue
                # buscando en los demas fragmentos.
                continue

            t = titulos.get(elegido)
            if not t:
                continue
            nombre = str(t).replace("\\", "/").rstrip("/").rsplit("/", 1)[-1]

            # Si la cita empieza a mitad de palabra (el modelo se comio una
            # letra), se recorta hasta el primer espacio para no mostrar algo
            # como "TR (Veces Tomado...".
            texto_cita = nucleo.strip()
            if pos > 0 and norm[pos - 1].isalnum():
                corte = texto_cita.find(" ")
                if 0 < corte < 25:
                    texto_cita = texto_cita[corte + 1:].strip()

            llave = (nombre, aguja[:80])
            if llave not in ya:
                ya.add(llave)
                salida.append({"documento": nombre, "fragmento": texto_cita})
            break                         # primer fragmento = mejor rankeado
        if len(salida) >= maximo:
            break

    return salida


# ----------------------------------------------------- filtro para streaming
class FiltroStream:
    """Oculta el bloque <citas> del flujo de tokens.

    En streaming los tokens salen al cliente a medida que el modelo escribe, asi
    que el bloque se veria en pantalla antes de que nadie pueda quitarlo. Este
    filtro retiene los ultimos caracteres por si la marca viene partida entre
    dos fragmentos, y corta la emision en cuanto la detecta.
    """

    def __init__(self) -> None:
        self._retenido = ""
        self._cortado = False

    def filtrar(self, fragmento: str) -> str:
        """Devuelve lo que SI se puede emitir de este fragmento."""
        if self._cortado:
            return ""
        self._retenido += fragmento or ""

        pos = self._retenido.lower().find(MARCA_INICIO)
        if pos != -1:
            self._cortado = True
            emitible = self._retenido[:pos]
            self._retenido = ""
            return emitible

        # Puede que la marca venga partida: se guardan los ultimos caracteres
        # (uno menos que la marca) hasta tener la certeza de que no empieza ahi.
        guarda = len(MARCA_INICIO) - 1
        if len(self._retenido) > guarda:
            emitible = self._retenido[:-guarda]
            self._retenido = self._retenido[-guarda:]
            return emitible
        return ""

    def vaciar(self) -> str:
        """Lo que quedo retenido al terminar, si nunca aparecio la marca."""
        if self._cortado:
            return ""
        resto, self._retenido = self._retenido, ""
        return resto