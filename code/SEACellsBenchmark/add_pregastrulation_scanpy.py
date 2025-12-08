#!/usr/bin/env python3
import pandas as pd
import numpy as np
from pathlib import Path

# -----------------------------
# FILES
# -----------------------------
merged_file = Path("/project/imoskowitz/xyang2/chrislowzhengxi/results/score_genes/merged_all_systems_edge_filtered_with_shh_scoregenes.csv")
edges_file  = Path("/project/imoskowitz/xyang2/SHH/Qiu_TimeLapse/other/edges.txt")
out_file    = Path("/project/imoskowitz/xyang2/chrislowzhengxi/results/score_genes/full_scored_edges_with_pregastrulation_scoregenes.csv")

# -----------------------------
# LOAD main merged Scanpy edges
# -----------------------------
merged = pd.read_csv(merged_file)
print(f"[LOAD] merged systems: {merged.shape}")

# -----------------------------
# LOAD pregastrulation edges
# -----------------------------
preg = pd.read_csv(edges_file, sep="\t")
preg = preg[preg["system"] == "Pre_gastrulation"].copy()
print(f"[LOAD] pre-gastrulation edges: {preg.shape}")

# -----------------------------
# ADD synthetic SHH scores (≈0)
# -----------------------------
rng = np.random.default_rng(42)
preg["sh_x"] = rng.normal(0, 0.02, size=len(preg))
preg["sh_y"] = rng.normal(0, 0.02, size=len(preg))

preg["abs_delta"] = (preg["sh_x"] - preg["sh_y"]).abs()
preg["delta"] = preg["sh_y"] - preg["sh_x"]

preg["cohens_d"] = np.nan
preg["pct_change"] = np.nan
preg["abs_pct_change"] = np.nan

# -----------------------------
# MATCH column order with main table
# -----------------------------
expected_cols = merged.columns
preg = preg.reindex(columns=expected_cols, fill_value=np.nan)

# -----------------------------
# CONCAT
# -----------------------------
full = pd.concat([merged, preg], ignore_index=True)
print(f"[MERGED] combined total rows: {full.shape[0]}")

# -----------------------------
# SAVE
# -----------------------------
full.to_csv(out_file, index=False)
print(f"[SAVE] {out_file}")
