#!/usr/bin/env python3
import scanpy as sc
import pandas as pd
import numpy as np
import os

# -----------------------------
# Config
# -----------------------------
SYSTEM = "Gastrulation"

adata_in = "/project/imoskowitz/xyang2/chrislowzhengxi/data/gastrulation/mouse_gastrulation_sce_clean_singlet.h5ad"
out_dir = "/project/imoskowitz/xyang2/chrislowzhengxi/results/score_genes_new/Gastrulation"
os.makedirs(out_dir, exist_ok=True)

# -----------------------------
# Load data
# -----------------------------
adata = sc.read_h5ad(adata_in)
print(adata)

# -----------------------------
# Normalization (Scanpy-style)
# -----------------------------
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)

# -----------------------------
# Score genes
# -----------------------------
shh_genes = ["Gli1", "Ptch1", "Hhip"]
present = [g for g in shh_genes if g in adata.var_names]

if len(present) == 0:
    raise ValueError("None of Gli1 / Ptch1 / Hhip found in adata.var_names")

print("Using genes:", present)

sc.tl.score_genes(
    adata,
    gene_list=present,
    score_name="SHH_scoregenes"
)

# -----------------------------
# Save per-cell scores
# -----------------------------
cell_csv = os.path.join(out_dir, "Gastrulation_SHH_scoregenes_percell.csv")

adata.obs[["celltype", "SHH_scoregenes"]].to_csv(cell_csv)
print("[SAVE]", cell_csv)

# -----------------------------
# Aggregate per celltype
# -----------------------------
summary = (
    adata.obs
    .groupby("celltype")["SHH_scoregenes"]
    .agg(
        n_cells="count",
        median="median",
        mean="mean",
        q90=lambda x: np.quantile(x, 0.9),
        frac_gt0=lambda x: (x > 0).mean(),
        variance="var",
        std="std"
    )
    .reset_index()
)

summary_csv = os.path.join(out_dir, "Gastrulation_SHH_scoregenes_summary.csv")
summary.to_csv(summary_csv, index=False)
print("[SAVE]", summary_csv)
