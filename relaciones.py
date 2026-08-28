import json, csv, unicodedata
from grail import GRAIL, load_config
from grail.query.retrieval import load_artifacts_for_search

def norm(s):
    s = str(s or "").lower()
    s = unicodedata.normalize("NFKD", s)
    return "".join(c for c in s if not unicodedata.combining(c))

def _list(v):
    if v is None or isinstance(v, float):
        return []
    try:
        return [str(x) for x in list(v)]
    except TypeError:
        return []

FAMILIAS = [
    ("DEROGA",      ["deroga", "deja sin efecto", "sin efecto", "deja sin vigencia"]),
    ("REEMPLAZA",   ["reemplaza", "reemplaz", "sustituye", "sustituc"]),
    ("MODIFICA",    ["modifica", "modific"]),
    ("COMPLEMENTA", ["complementa", "complement"]),
]

grail = GRAIL.from_config(load_config("."))
arts = load_artifacts_for_search(grail.storage, grail._output_folder())
rels = arts.relationships.copy()
docs = arts.documents.copy()

id2title, id2path = {}, {}
for _, d in docs.iterrows():
    did = str(d.get("id"))
    id2title[did] = str(d.get("title") or "")
    id2path[did] = str(d.get("path") or "")

def familia(desc):
    n = norm(desc)
    for fam, kws in FAMILIAS:
        if any(k in n for k in kws):
            return fam
    return None

out = {fam: [] for fam, _ in FAMILIAS}
for _, r in rels.iterrows():
    fam = familia(r.get("description"))
    if not fam:
        continue
    documentos = [
        {"id": did, "titulo": id2title.get(did, ""), "ruta": id2path.get(did, "")}
        for did in _list(r.get("document_ids"))
    ]
    out[fam].append({
        "source": str(r.get("source")),
        "target": str(r.get("target")),
        "description": str(r.get("description") or ""),
        "documentos": documentos,
    })

print("Relaciones por familia:")
for fam in out:
    print(f"  {fam}: {len(out[fam])}")

with open("relaciones_normativas.json", "w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, indent=2)

with open("relaciones_normativas.csv", "w", newline="", encoding="utf-8-sig") as f:
    w = csv.writer(f)
    w.writerow(["tipo", "source", "target", "documentos", "description"])
    for fam, items in out.items():
        for it in items:
            w.writerow([fam, it["source"], it["target"],
                        " ; ".join(d["titulo"] for d in it["documentos"]),
                        it["description"]])

print("Escrito: relaciones_normativas.json y relaciones_normativas.csv (con nombres de documento)")