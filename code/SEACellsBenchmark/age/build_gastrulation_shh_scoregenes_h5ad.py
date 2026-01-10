#!/usr/bin/env python3
import pandas as pd
import anndata as ad
from pathlib import Path
import numpy as np

# -------------------------
# Paths
# -------------------------
IN_CSV = (
    "/project/imoskowitz/xyang2/chrislowzhengxi/"
    "results/score_genes_new/Gastrulation/"
    "Gastrulation_SHH_scoregenes_percell.csv"
)

OUT_H5AD = (
    "/project/imoskowitz/xyang2/chrislowzhengxi/"
    "results/score_genes_new/Gastrulation/"
    "Gastrulation_adata_with_SHH_scoregenes.h5ad"
)

# -------------------------
# Load per-cell data
# -------------------------
df = pd.read_csv(IN_CSV)

# ---- Sanity checks ----
required = {"SHH_scoregenes"}
if not required.issubset(df.columns):
    raise ValueError(f"Missing required columns: {required - set(df.columns)}")

# Infer cell_id column
if "cell_id" in df.columns:
    cell_id = df["cell_id"].astype(str)
elif "Unnamed: 0" in df.columns:
    cell_id = df["Unnamed: 0"].astype(str)
else:
    # fall back to row index
    cell_id = df.index.astype(str)


# Infer celltype column
if "celltype" in df.columns:
    celltype_col = "celltype"
elif "celltype_update" in df.columns:
    celltype_col = "celltype_update"
else:
    raise ValueError("Cannot infer celltype column")

# -------------------------
# Build obs dataframe
# -------------------------
obs = pd.DataFrame(
    {
        "cell_id": cell_id,
        "celltype_update": df[celltype_col].astype(str),
        "SHH_scoregenes": pd.to_numeric(df["SHH_scoregenes"], errors="coerce"),
    }
)

obs.index = obs["cell_id"]


# -------------------------
# Build minimal AnnData
# -------------------------
# No expression matrix needed; use empty X
adata = ad.AnnData(
    X=np.zeros((obs.shape[0], 0)),
    obs=obs,
)

# -------------------------
# Write
# -------------------------
out_path = Path(OUT_H5AD)
out_path.parent.mkdir(parents=True, exist_ok=True)
adata.write(out_path)

print(f"[DONE] wrote {out_path}")
print(f"n_cells = {adata.n_obs}")
print("obs columns:", list(adata.obs.columns))
