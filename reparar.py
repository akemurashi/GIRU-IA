import asyncio, httpx
import pandas as pd
from grail import GRAIL, load_config
from grail.query.retrieval import load_artifacts_for_search
from grail.indexing.emit_merge import entity_embedding_text

OLLAMA = "http://localhost:11434/api/embed"
MODEL = "bge-m3"
BATCH = 32

def _s(v):
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return ""
    return str(v)

def _rq(v):
    if v is None:
        return []
    try:
        if isinstance(v, float) and pd.isna(v):
            return []
    except Exception:
        pass
    try:
        return [str(x) for x in list(v)]
    except TypeError:
        return []

def embed_batch(client, texts):
    r = client.post(OLLAMA, json={"model": MODEL, "input": texts}, timeout=180)
    r.raise_for_status()
    return r.json()["embeddings"]

async def main():
    grail = GRAIL.from_config(load_config("."))
    arts = load_artifacts_for_search(grail.storage, grail._output_folder())
    ents = arts.entities.copy()
    print(f"Entidades: {len(ents)}")
    texts = [
        entity_embedding_text(_s(r.get("name")), _s(r.get("description")), _rq(r.get("retrieval_queries")))
        for _, r in ents.iterrows()
    ]

    embs = [None] * len(texts)
    with httpx.Client() as client:
        for i in range(0, len(texts), BATCH):
            chunk = texts[i:i+BATCH]
            try:
                vecs = embed_batch(client, chunk)
            except Exception:
                vecs = []
                for t in chunk:
                    try:
                        vecs.append(embed_batch(client, [t])[0])
                    except Exception:
                        vecs.append(None)
            for j, v in enumerate(vecs):
                embs[i+j] = v
            print(f"  {min(i+BATCH, len(texts))}/{len(texts)}", end="\r")

    ok = sum(1 for e in embs if e is not None)
    print(f"\nEmbeddings: {ok}/{len(embs)}")
    if ok == 0:
        print("Fallo total. Revisar Ollama.")
        return
    ents["description_embedding"] = embs
    await grail._update_vector_store(ents)
    print("Indice vectorial reconstruido. Prueba --mode cascade")

asyncio.run(main())