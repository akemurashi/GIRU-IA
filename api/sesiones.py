"""Memoria de chatbot por sesion — almacenamiento en archivos JSON.

Version sin SQLite: en este servidor Windows App Control bloquea el modulo
_sqlite3. La interfaz publica (funciones async) es identica a la version con
SQLite, asi que main.py y stream.py no cambian.

Un archivo JSON por sesion en GIRU_SESIONES/ (por defecto ./sesiones_datos/).
Escritura atomica: se escribe a .tmp y se reemplaza, para que una caida a
mitad de guardado no deje un archivo corrupto.
"""
from __future__ import annotations

import asyncio
import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from threading import RLock
from typing import Any, Optional

CARPETA = Path(os.getenv("GIRU_SESIONES", "sesiones_datos"))
_candado = RLock()


def _ahora() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _ruta(sesion_id: str) -> Path:
    # uuid4 como nombre de archivo; se filtra para no permitir rutas raras.
    seguro = "".join(c for c in sesion_id if c.isalnum() or c in "-_")
    return CARPETA / f"{seguro}.json"


def _leer(sesion_id: str) -> Optional[dict[str, Any]]:
    p = _ruta(sesion_id)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def _guardar(datos: dict[str, Any]) -> None:
    p = _ruta(datos["id"])
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(datos, ensure_ascii=False, indent=1), encoding="utf-8")
    tmp.replace(p)


# ------------------------------------------------------------ operaciones
def _init() -> None:
    CARPETA.mkdir(parents=True, exist_ok=True)


def _crear_sesion(user_id: str, titulo: Optional[str], modo: str) -> dict[str, Any]:
    t = _ahora()
    datos = {
        "id": str(uuid.uuid4()),
        "user_id": user_id,
        "titulo": titulo or "Nueva conversacion",
        "modo": modo,
        "creada": t,
        "actualizada": t,
        "mensajes": [],
    }
    with _candado:
        _guardar(datos)
    return {k: v for k, v in datos.items() if k != "mensajes"}


def _obtener_sesion(sesion_id: str, user_id: str) -> Optional[dict[str, Any]]:
    with _candado:
        d = _leer(sesion_id)
    if not d or d.get("user_id") != user_id:
        return None
    return {k: v for k, v in d.items() if k != "mensajes"}


def _listar_sesiones(user_id: str) -> list[dict[str, Any]]:
    salida: list[dict[str, Any]] = []
    with _candado:
        for p in CARPETA.glob("*.json"):
            try:
                d = json.loads(p.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                continue
            if d.get("user_id") == user_id:
                salida.append({k: v for k, v in d.items() if k != "mensajes"})
    return sorted(salida, key=lambda s: s["actualizada"], reverse=True)


def _borrar_sesion(sesion_id: str, user_id: str) -> bool:
    with _candado:
        d = _leer(sesion_id)
        if not d or d.get("user_id") != user_id:
            return False
        _ruta(sesion_id).unlink(missing_ok=True)
    return True


def _agregar_mensaje(
    sesion_id: str, rol: str, contenido: str, meta: Optional[dict[str, Any]] = None
) -> dict[str, Any]:
    t = _ahora()
    msg = {"rol": rol, "contenido": contenido, "creado": t, "meta": meta}
    with _candado:
        d = _leer(sesion_id)
        if d is None:
            raise KeyError(f"Sesion inexistente: {sesion_id}")
        d["mensajes"].append(msg)
        d["actualizada"] = t
        _guardar(d)
    return msg


def _mensajes(sesion_id: str, limite: Optional[int] = None) -> list[dict[str, Any]]:
    with _candado:
        d = _leer(sesion_id)
    msgs = d["mensajes"] if d else []
    return msgs[-limite:] if limite else msgs


def _renombrar_si_nueva(sesion_id: str, pregunta: str) -> Optional[str]:
    with _candado:
        d = _leer(sesion_id)
        if not d or d.get("titulo") != "Nueva conversacion":
            return None
        titulo = pregunta[:60].strip() + ("..." if len(pregunta) > 60 else "")
        d["titulo"] = titulo
        _guardar(d)
    return titulo


# ------------------------------------------- envoltorios async (sin bloquear)
async def init() -> None:
    await asyncio.to_thread(_init)


async def crear_sesion(user_id: str, titulo: Optional[str], modo: str) -> dict[str, Any]:
    return await asyncio.to_thread(_crear_sesion, user_id, titulo, modo)


async def obtener_sesion(sesion_id: str, user_id: str) -> Optional[dict[str, Any]]:
    return await asyncio.to_thread(_obtener_sesion, sesion_id, user_id)


async def listar_sesiones(user_id: str) -> list[dict[str, Any]]:
    return await asyncio.to_thread(_listar_sesiones, user_id)


async def borrar_sesion(sesion_id: str, user_id: str) -> bool:
    return await asyncio.to_thread(_borrar_sesion, sesion_id, user_id)


async def agregar_mensaje(
    sesion_id: str, rol: str, contenido: str, meta: Optional[dict[str, Any]] = None
) -> dict[str, Any]:
    return await asyncio.to_thread(_agregar_mensaje, sesion_id, rol, contenido, meta)


async def mensajes(sesion_id: str, limite: Optional[int] = None) -> list[dict[str, Any]]:
    return await asyncio.to_thread(_mensajes, sesion_id, limite)


async def renombrar_si_nueva(sesion_id: str, pregunta: str) -> Optional[str]:
    return await asyncio.to_thread(_renombrar_si_nueva, sesion_id, pregunta)