r"""
Post-proceso del grafo GRAIL -> tabla de VIGENCIAS normativas (proyecto GIRU).
 
Qué hace, en dos operaciones deterministas sobre final_relationships.parquet:
 
1. FILTRA el subgrafo de vigencias: se queda solo con relaciones NORMA -> NORMA,
   es decir donde tanto source como target son decretos con identificador de la
   forma "DECRETO NNN/AAAA". Descarta el ruido organizacional (cargos, órganos)
   y las relaciones norma<->artículo/reglamento (que no son vigencias entre
   decretos).
 
2. CORRIGE la dirección por año: para los tipos donde la regla temporal es
   universal (DEROGA, MODIFICA, REEMPLAZA), el decreto MÁS NUEVO debe ser el
   source (quien ejerce la acción). Si la arista está invertida, la voltea.
   REFERENCIA y COMPLEMENTA se dejan como las produjo el modelo (la regla
   temporal no es universal para ellas).
 
Por qué post-proceso y no prompt: la dirección se deduce del año del decreto
(dato determinista en el identificador), no de la redacción. Ningún modelo
(14b/32b/70b) acertó la dirección de forma fiable; una regla la acierta el 100%.
 
NO modifica los artefactos de GRAIL. Lee el parquet y escribe archivos nuevos:
  - vigencias.parquet  (tabla limpia y corregida, para el backend)
  - vigencias.json     (mismo contenido, cómodo de inspeccionar)
  - vigencias_reporte.txt (resumen legible de qué se filtró y qué se corrigió)
 
Uso (parado en la carpeta del sandbox, con el .venv de GIRU-IA):
  ...\python.exe postproceso_vigencias.py
o apuntando a un parquet concreto:
  ...\python.exe postproceso_vigencias.py --parquet ruta\a\final_relationships.parquet
"""
from __future__ import annotations
 
import argparse
import glob
import json
import re
import sys
from pathlib import Path
 
import pandas as pd
 
# Tipos cuya dirección temporal es universal: el decreto más nuevo es el source.
DIRECTIONAL_TYPES = {"DEROGA", "MODIFICA", "REEMPLAZA"}
 
# Tipos que se conservan pero NO se corrigen por año (regla no universal).
KEEP_AS_IS_TYPES = {"COMPLEMENTA", "REFERENCIA", "PROMULGA", "APRUEBA"}
 
# Reconoce un identificador de norma tipo "DECRETO 052/2008" y captura el año.
# Acepta años de 4 dígitos (ya normalizados por el prompt). Tolera "DECRETO",
# "DTO", "D.R." o solo "NNN/AAAA" por robustez.
NORMA_RE = re.compile(r"\b(?:DECRETO|DTO\.?|D\.?R\.?)?\s*\d{1,4}\s*/\s*(\d{4})\b", re.IGNORECASE)
 
 
def norma_year(name: str) -> int | None:
    """Devuelve el año (int) si el nombre es una norma con identificador NNN/AAAA; si no, None."""
    if not isinstance(name, str):
        return None
    m = NORMA_RE.search(name)
    return int(m.group(1)) if m else None
 
 
def is_norma(name: str) -> bool:
    return norma_year(name) is not None
 
 
def find_parquet(explicit: str | None) -> Path:
    if explicit:
        p = Path(explicit)
        if not p.exists():
            sys.exit(f"No existe el parquet indicado: {p}")
        return p
    hits = glob.glob("**/final_relationships.parquet", recursive=True)
    if not hits:
        sys.exit("No se encontró final_relationships.parquet bajo la carpeta actual. "
                 "Corre el script parado en la carpeta del sandbox, o usa --parquet.")
    if len(hits) > 1:
        print(f"[aviso] Hay {len(hits)} parquet; uso el primero:\n  {hits[0]}")
    return Path(hits[0])
 
 
def main() -> None:
    ap = argparse.ArgumentParser(description="Post-proceso de vigencias normativas GRAIL.")
    ap.add_argument("--parquet", default=None, help="Ruta a final_relationships.parquet (opcional).")
    ap.add_argument("--out", default=".", help="Carpeta de salida (por defecto: actual).")
    args = ap.parse_args()
 
    src = find_parquet(args.parquet)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
 
    df = pd.read_parquet(src)
    total_in = len(df)
 
    # --- Operación 1: filtrar subgrafo NORMA -> NORMA ---
    df = df.copy()
    df["_src_year"] = df["source"].map(norma_year)
    df["_tgt_year"] = df["target"].map(norma_year)
    mask_norma = df["_src_year"].notna() & df["_tgt_year"].notna()
    vig = df[mask_norma].copy()
    n_norma = len(vig)
 
    # --- Operación 2: corregir dirección por año (solo tipos direccionales) ---
    flipped_rows = []
    def fix_direction(row):
        rt = str(row["relationship_type"]).upper()
        if rt not in DIRECTIONAL_TYPES:
            return row  # COMPLEMENTA/REFERENCIA/etc.: se dejan como están
        sy, ty = int(row["_src_year"]), int(row["_tgt_year"])
        if sy == ty:
            return row  # mismo año: no hay criterio temporal, se deja
        # El más nuevo debe ser source. Si el source es el más viejo, invertir.
        if sy < ty:
            flipped_rows.append((row["source"], row["target"], rt))
            row = row.copy()
            row["source"], row["target"] = row["target"], row["source"]
            row["source_id"], row["target_id"] = row["target_id"], row["source_id"]
            row["_src_year"], row["_tgt_year"] = ty, sy
            row["source_degree"], row["target_degree"] = row["target_degree"], row["source_degree"]
        return row
 
    vig = vig.apply(fix_direction, axis=1)
 
    # --- Salidas ---
    # Columnas útiles para el backend (dejo las de trazabilidad: doc_ids, description).
    cols = ["source", "target", "relationship_type", "description",
            "document_ids", "text_unit_ids", "weight", "confidence"]
    cols = [c for c in cols if c in vig.columns]
    vig_out = vig[cols].reset_index(drop=True)
 
    parquet_out = out_dir / "vigencias.parquet"
    json_out = out_dir / "vigencias.json"
    report_out = out_dir / "vigencias_reporte.txt"
 
    vig_out.to_parquet(parquet_out, index=False)
    # document_ids/text_unit_ids son arrays numpy -> a lista para JSON.
    vig_json = vig_out.copy()
    for c in ("document_ids", "text_unit_ids"):
        if c in vig_json.columns:
            vig_json[c] = vig_json[c].map(lambda x: list(x) if hasattr(x, "__iter__") and not isinstance(x, str) else x)
    vig_json.to_json(json_out, orient="records", force_ascii=False, indent=2)
 
    # Reporte legible
    lines = []
    lines.append("=== POST-PROCESO DE VIGENCIAS ===")
    lines.append(f"Fuente: {src}")
    lines.append(f"Relaciones totales en el grafo: {total_in}")
    lines.append(f"Relaciones NORMA->NORMA (subgrafo de vigencias): {n_norma}")
    lines.append(f"Direcciones corregidas por año: {len(flipped_rows)}")
    if flipped_rows:
        lines.append("\nAristas invertidas (antes -> ahora):")
        for s, t, rt in flipped_rows:
            lines.append(f"  [{rt}] {s} -> {t}   INVERTIDA a   {t} -> {s}")
    lines.append("\n=== TABLA DE VIGENCIAS FINAL ===")
    for _, r in vig_out.iterrows():
        lines.append(f"  {r['source']:22s} --{r['relationship_type']}--> {r['target']}")
    report = "\n".join(lines)
    report_out.write_text(report, encoding="utf-8")
 
    print(report)
    print(f"\nEscritos:\n  {parquet_out}\n  {json_out}\n  {report_out}")
 
 
if __name__ == "__main__":
    main()