"""Lanzador de la API GIRU-IA para este servidor.

Windows App Control bloquea algunos modulos compilados de la libreria estandar
(_sqlite3, pyexpat). Ninguno hace falta para consultar el grafo, pero entran
por cadenas de importacion ajenas:

    grail -> grail.indexing.leiden -> graspologic -> matplotlib
          -> plistlib -> xml.parsers.expat -> pyexpat   (BLOQUEADO)

Este archivo registra sustitutos vacios en sys.modules ANTES de que se importe
nada, de modo que esas cadenas se corten sin tocar el codigo de la API ni el de
GRAIL. matplotlib solo usa plistlib para tipografias de macOS, y Leiden solo se
usa al indexar, nunca al responder.

Uso:
    & $py run_api.py --check     verifica que GRAIL carga
    & $py run_api.py             levanta la API en el puerto 8000

Correr SIEMPRE parado en la carpeta del proyecto (grail.yaml tiene root_dir: .).
"""
from __future__ import annotations

import sys
import types
from pathlib import Path

RAIZ = Path(__file__).resolve().parent
CARPETA_API = RAIZ / "api"


def _stub(nombre: str, **atributos) -> None:
    """Registra un modulo vacio solo si el real no se puede importar."""
    if nombre in sys.modules:
        return
    try:
        __import__(nombre)
        return  # el real funciona: no lo tocamos
    except ImportError:
        pass
    m = types.ModuleType(nombre)
    for k, v in atributos.items():
        setattr(m, k, v)
    m.__dict__.setdefault("__doc__", f"Sustituto vacio de {nombre} (bloqueado por App Control)")
    sys.modules[nombre] = m
    print(f"[compat] {nombre} bloqueado: se usa un sustituto vacio")


def preparar_entorno() -> None:
    def _no_disponible(*_a, **_k):
        raise RuntimeError("Funcion no disponible en este entorno (modulo bloqueado)")

    # Cadena matplotlib -> plistlib -> pyexpat
    _stub(
        "plistlib",
        load=_no_disponible, loads=_no_disponible,
        dump=_no_disponible, dumps=_no_disponible,
        FMT_XML=1, FMT_BINARY=2,
    )

    if str(CARPETA_API) not in sys.path:
        sys.path.insert(0, str(CARPETA_API))


def comprobar() -> int:
    preparar_entorno()
    fallos = []
    for mod in ("pyarrow", "lancedb", "numpy", "pandas", "httpx", "fastapi", "grail"):
        try:
            __import__(mod)
            print(f"  ok      {mod}")
        except Exception as exc:  # noqa: BLE001
            print(f"  FALLA   {mod}: {exc}")
            fallos.append(mod)
    for mod in ("sesiones", "esquemas", "grail_svc", "main"):
        try:
            __import__(mod)
            print(f"  ok      api/{mod}.py")
        except Exception as exc:  # noqa: BLE001
            print(f"  FALLA   api/{mod}.py: {exc}")
            fallos.append(mod)
    print("\nTODO OK" if not fallos else f"\nFALLARON: {', '.join(fallos)}")
    return 0 if not fallos else 1


def servir() -> None:
    preparar_entorno()
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, log_level="info")


if __name__ == "__main__":
    sys.exit(comprobar() if "--check" in sys.argv else servir() or 0)