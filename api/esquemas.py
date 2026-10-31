"""Contrato de datos entre el backend y el servicio de IA de GIRU."""
from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, Field

Modo = Literal["local", "cascade", "global", "document", "agent"]


# --------------------------------------------------------------- Sesiones
class SesionCrear(BaseModel):
    user_id: str = Field(..., description="Identificador de usuario que entrega el backend")
    titulo: Optional[str] = None
    modo: Modo = "cascade"


class SesionResumen(BaseModel):
    id: str
    user_id: str
    titulo: str
    modo: Modo
    creada: str
    actualizada: str


class Mensaje(BaseModel):
    rol: Literal["user", "assistant"]
    contenido: str
    creado: str
    meta: Optional[dict[str, Any]] = None


class SesionDetalle(SesionResumen):
    mensajes: list[Mensaje]


# --------------------------------------------------------------- Preguntar
class PreguntaEntrada(BaseModel):
    user_id: str
    session_id: str
    pregunta: str
    modo: Optional[Modo] = None
    documento: Optional[str] = Field(
        None, description="Nombre de archivo; obligatorio solo con modo='document'"
    )


class Fuente(BaseModel):
    documento: str = Field(..., description="Nombre de archivo; llave hacia la base de datos")
    referencia: Optional[str] = Field(
        None,
        description="Sin uso. Antes traia un uuid interno de GRAIL que cambiaba en "
                    "cada reindexacion; hoy siempre llega null. Se retira en la v2.",
    )
    fragmento: Optional[str] = Field(
        None,
        description="Texto LITERAL del documento que respalda la respuesta, ya "
                    "verificado contra el fragmento de origen. Cuando viene con "
                    "valor, puede mostrarse entrecomillado como cita. Cuando llega "
                    "null, el documento es el mejor candidato por relevancia pero "
                    "NO hay cita comprobada: no debe maquetarse como cita textual.",
    )
    seccion: Optional[str] = Field(
        None,
        description="Articulo o seccion del documento donde esta el respaldo "
                    "(ej. 'TITULO TERCERO — Articulo 14'). Opcional; null si no se "
                    "pudo ubicar.",
    )


class PreguntaSalida(BaseModel):
    session_id: str
    respuesta: str
    fuentes: list[Fuente]
    respaldo: Optional[float] = Field(
        None,
        description="Fraccion (0-1) de las afirmaciones de la respuesta para las que "
                    "se encontro un articulo que las respalda. Indicador de confianza: "
                    "bajo 0,5 conviene mostrar un aviso. null = no medido.",
    )
    modo: Modo
    tiempo_s: float
    llamadas_llm: int


# ----------------------------------------------------------------- Buscar
class BusquedaEntrada(BaseModel):
    consulta: str
    k: int = Field(50, ge=1, le=100, description="Cantidad de documentos a devolver")


class Coincidencia(BaseModel):
    documento: str = Field(..., description="Nombre de archivo; llave hacia la base de datos")
    puntaje: float = Field(..., description="Solo sirve para ordenar; no es una probabilidad")


class BusquedaSalida(BaseModel):
    consulta: str
    resultados: list[Coincidencia]