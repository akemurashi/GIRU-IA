"""giru_identidad.py — El chatbot se presenta como Don Fede, no como GRAIL.

Por que no basta con editar el prompt
-------------------------------------
GRAIL le pasa al prompt `assistant_name="GRAIL"` de forma EXPLICITA desde cada
buscador (cascade_search.py:116, local_search.py:61, document_search.py:53,
global_search.py:62), y ese valor no se puede cambiar desde grail.yaml.
El override local_search.py usaba `params.setdefault("assistant_name", ...)`,
que no pisa un valor que ya viene puesto: por eso seguia diciendo "GRAIL".

Aca se envuelve PromptRegistry.build para forzar el nombre en TODOS los prompts
de consulta (local, cascade, document y global_reduce), y se deja listo un
system prompt en espanol para el modo agent, cuyo prompt viene fijo en
grail/query/agent.py ("powered by GRAIL").

    GIRU_NOMBRE_ASISTENTE="Don Fede"   (por defecto)
"""
from __future__ import annotations

import logging
import os

log = logging.getLogger("giru.identidad")

NOMBRE = os.getenv("GIRU_NOMBRE_ASISTENTE", "Don Fede")

_activado = False


def activar() -> None:
    global _activado
    if _activado:
        return
    from grail.prompts.loader import PromptRegistry

    original = PromptRegistry.build

    def build(self, name, **params):
        if "assistant_name" in params or name in ("local_search", "global_reduce"):
            params["assistant_name"] = NOMBRE
        return original(self, name, **params)

    build._giru_original = original          # type: ignore[attr-defined]
    PromptRegistry.build = build             # type: ignore[assignment]
    _activado = True
    log.info("GIRU: el asistente se identifica como %s", NOMBRE)


def prompt_agente() -> str:
    """AGENT_SYSTEM_PROMPT de GRAIL con la identidad reemplazada.

    Se reutiliza el texto original (la estrategia de herramientas es buena) y solo
    se cambia el rol, mas una regla de idioma e identidad al final.
    """
    from grail.query.agent import AGENT_SYSTEM_PROMPT

    rol_original = "You are a knowledge-graph research assistant powered by GRAIL."
    rol_nuevo = (
        f"You are {NOMBRE}, the regulations assistant of Universidad Técnica "
        "Federico Santa María (UTFSM)."
    )
    texto = AGENT_SYSTEM_PROMPT.replace(rol_original, rol_nuevo)
    return texto + (
        "\n\n<identidad>\n"
        f"- Tu nombre es {NOMBRE}. Si te preguntan quién eres, responde que eres "
        f"{NOMBRE}, el asistente de normativa de la UTFSM.\n"
        "- Nunca te presentes como GRAIL, GIRU, Qwen, un modelo de lenguaje ni un "
        "grafo de conocimiento.\n"
        "- Responde siempre en español.\n"
        "</identidad>"
    )
