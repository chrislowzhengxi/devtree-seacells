#!/usr/bin/env python3
import os
from pathlib import Path
import pandas as pd
import anndata as ad

# Config consistent with your main script
OUT_ROOT = Path("/project/imoskowitz/xyang2/chrislowzhengxi/results/ucell")
SYSTEM   = "Lateral_plate_mesoderm"

# Suffix logic (match your run_suffix())
UCELL_INPUT_LAYER = os.environ.get("UCELL_INPUT_LAYER", None)  # e.g., "log1p_cpm"
SUBSAMPLE_FRAC = float(os.environ.get("SUBSAMPLE_FRAC", "1.0"))
suffix = f"_{UCELL_INPUT_LAYER}" if UCELL_INPUT_LAYER else ""
if SUBSAMPLE_FRAC < 1.0:
    suffix += "_test"

# File paths
outdir    = OUT_ROOT / SYSTEM
qcdir     = outdir / "qc"
h5_scored = outdir / f"{SYSTEM}{suffix}_adata_with_ucell.h5ad"

# Fallback: if the exact file is not found, pick the first matching scored h5ad
if not h5_scored.exists():
    cand = sorted(outdir.glob(f"{SYSTEM}*_adata_with_ucell.h5ad"))
    if not cand:
        raise FileNotFoundError(f"No scored h5ad found under {outdir}")
    h5_scored = cand[0]

print(f"[LOAD] {h5_scored}")
adata = ad.read_h5ad(str(h5_scored))

# Ensure the score column exists
col = "SHH_UCell_score"
if col not in adata.obs.columns:
    raise KeyError(f"Missing '{col}' in adata.obs")

# Build a DataFrame with all obs + index as 'cell_id'
df = adata.obs.copy()
df.insert(0, "cell_id", adata.obs_names.astype(str))

# Write both TSV and CSV
qcdir.mkdir(parents=True, exist_ok=True)
tsv_path = qcdir / f"{SYSTEM}{suffix}_obs_with_SHH_UCell_score.txt"
csv_path = qcdir / f"{SYSTEM}{suffix}_obs_with_SHH_UCell_score.csv"

df.to_csv(tsv_path, sep="\t", index=False)
df.to_csv(csv_path, index=False)

print("[SAVE]", tsv_path)
print("[SAVE]", csv_path)
