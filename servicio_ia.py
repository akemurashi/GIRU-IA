# servicio_ia.py — Microservicio de IA (GRAIL) para GIRU
#   POST /preguntar  -> chatbot: respuesta + documentos fuente reales (citas)
#   POST /buscar     -> buscador semantico: documentos relevantes por contenido
import httpx
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from grail import GRAIL, load_config
from grail.query.retrieval import load_artifacts_for_search

OLLAMA_EMBED = "http://localhost:11434/api/embed"   # ruta NATIVA (evita el bug de /v1)
EMBED_MODEL = "bge-m3"

app = FastAPI(title="GIRU - IA Normativa")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

grail = GRAIL.from_config(load_config("."))
_arts = load_artifacts_for_search(grail.storage, grail._output_folder())
_ent = _arts.entities.set_index("id", drop=False)
_doc_title = {str(r["id"]): str(r.get("title") or "") for _, r in _arts.documents.iterrows()}
_store = grail._vector_store()

def _list(v):
    try:
        return [] if v is None else [str(x) for x in list(v)]
    except TypeError:
        return []

def _fuentes_desde_sources(sources_df):
    """Extrae los documentos reales que alimentaron la respuesta."""
    docs = {}
    if sources_df is None or getattr(sources_df, "empty", True):
        return []
    for _, row in sources_df.iterrows():
        ids = _list(row.get("document_ids"))
        if not ids and row.get("document_id") is not None:
            ids = [str(row.get("document_id"))]
        snippet = str(row.get("text") or "")[:300]
        for did in ids:
            if not did or did == "None":
                continue
            titulo = _doc_title.get(did, did)
            d = docs.setdefault(titulo, {"documento": titulo, "fragmento": ""})
            if not d["fragmento"] and snippet:
                d["fragmento"] = snippet
    return list(docs.values())

async def _embed(texto: str):
    async with httpx.AsyncClient(timeout=60) as c:
        r = await c.post(OLLAMA_EMBED, json={"model": EMBED_MODEL, "input": texto})
        r.raise_for_status()
        return r.json()["embeddings"][0]

class Consulta(BaseModel):
    pregunta: str
    modo: str = "cascade"

class Busqueda(BaseModel):
    texto: str
    top_k: int = 10

@app.get("/salud")
def salud():
    return {"ok": True, "entidades": int(len(_ent)), "documentos": len(_doc_title)}

@app.post("/preguntar")
async def preguntar(c: Consulta):
    pregunta = (
        f"{c.pregunta}\n\n"
        "IMPORTANTE: Responde SIEMPRE en español, de forma clara y precisa, "
        "citando la normativa correspondiente. No respondas en inglés."
    )
    r = await grail.search(pregunta, mode=c.modo)
    fuentes = _fuentes_desde_sources(r.context_data.get("sources")) if isinstance(r.context_data, dict) else []
    return {"respuesta": r.response, "fuentes": fuentes, "modo": c.modo}

@app.post("/buscar")
async def buscar(b: Busqueda):
    emb = await _embed(b.texto)
    hits = _store.similarity_search_by_vector(query_embedding=emb, k=b.top_k)
    docs = {}
    for pos, h in enumerate(hits):
        eid = h.document.id
        if eid not in _ent.index:
            continue
        row = _ent.loc[eid]
        entidad = str(row.get("title") or row.get("name") or "")
        desc = str(row.get("description") or "")
        for did in _list(row.get("document_ids")):
            titulo = _doc_title.get(did, did)
            d = docs.setdefault(titulo, {"documento": titulo, "entidades": [], "fragmento": "", "relevancia": pos + 1})
            if entidad and entidad not in d["entidades"]:
                d["entidades"].append(entidad)
            if not d["fragmento"] and desc:
                d["fragmento"] = desc[:300]
    return {"resultados": sorted(docs.values(), key=lambda x: x["relevancia"])}