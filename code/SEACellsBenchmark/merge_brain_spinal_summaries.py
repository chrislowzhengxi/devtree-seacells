# # #!/usr/bin/env python3
# # import pandas as pd
# # from pathlib import Path

# # # Input paths
# # base = Path("/project/imoskowitz/xyang2/chrislowzhengxi/results/ucell")
# # p1 = base / "Other_Brain_spinal_cord/qc/Other_Brain_spinal_cord_log1p_cpm_ALL_labels_summary.csv"
# # p2 = base / "Neurons/qc/Neurons_log1p_cpm_ALL_labels_summary.csv"

# # # Output path
# # outdir = base / "Brain_spinal_cord/qc"
# # outdir.mkdir(parents=True, exist_ok=True)
# # outpath = outdir / "Brain_spinal_cord_log1p_cpm_ALL_labels_summary.csv"

# # # Load and merge
# # df1 = pd.read_csv(p1)
# # df2 = pd.read_csv(p2)
# # merged = pd.concat([df1, df2], ignore_index=True)

# # # Optional: drop duplicates just in case
# # merged = merged.drop_duplicates(subset=["celltype_new"], keep="last")

# # # Save
# # merged.to_csv(outpath, index=False)
# # print(f"[MERGED] wrote combined summary: {outpath}")
# #!/usr/bin/env python3


# #!/usr/bin/env python3
# import pandas as pd
# import scanpy as sc
# from pathlib import Path

# base = Path("/project/imoskowitz/xyang2/chrislowzhengxi/results/score_genes_new")

# systems_to_merge = ["Other_Brain_spinal_cord", "Neurons"]
# merged_system = "Brain_spinal_cord"

# outdir = base / merged_system
# qc_outdir = outdir / "qc"
# qc_outdir.mkdir(parents=True, exist_ok=True)

# # -----------------------------
# # 1. Merge summary CSVs
# # -----------------------------
# dfs = []
# for sys in systems_to_merge:
#     p = base / sys / "qc" / f"{sys}_SHH_scoregenes_summary.csv"
#     dfs.append(pd.read_csv(p))

# merged_summary = pd.concat(dfs, ignore_index=True)
# merged_summary = merged_summary.drop_duplicates(subset=["celltype_new"], keep="last")

# summary_out = qc_outdir / f"{merged_system}_SHH_scoregenes_summary.csv"
# merged_summary.to_csv(summary_out, index=False)
# print("Wrote summary:", summary_out)

# # -----------------------------
# # 2. Merge per-cell CSVs
# # -----------------------------
# percell_dfs = []
# for sys in systems_to_merge:
#     p = base / sys / f"{sys}_SHH_scoregenes_percell.csv"
#     percell_dfs.append(pd.read_csv(p))

# merged_percell = pd.concat(percell_dfs, ignore_index=False)
# percell_out = outdir / f"{merged_system}_SHH_scoregenes_percell.csv"
# merged_percell.to_csv(percell_out)
# print("Wrote per-cell:", percell_out)

# # -----------------------------
# # 3. Merge h5ad files
# # -----------------------------
# adatas = []
# for sys in systems_to_merge:
#     p = base / sys / f"{sys}_adata_with_SHH_scoregenes.h5ad"
#     ad = sc.read_h5ad(str(p))
#     ad.obs["system"] = merged_system
#     adatas.append(ad)

# adata_merged = adatas[0].concatenate(*adatas[1:], join="outer")

# # CLEAN var to avoid the h5py string error
# var = adata_merged.var.copy()

# # Drop problematic highly_variable-* columns (you probably do not need them anyway)
# drop_cols = [c for c in var.columns if c.startswith("highly_variable-")]
# if drop_cols:
#     print("Dropping var columns:", drop_cols)
#     var = var.drop(columns=drop_cols)

# # Cast any remaining object columns to string
# for col in var.columns:
#     if var[col].dtype == "object":
#         var[col] = var[col].astype(str)

# adata_merged.var = var

# h5_out = outdir / f"{merged_system}_adata_with_SHH_scoregenes.h5ad"
# adata_merged.write_h5ad(str(h5_out))
# print("Wrote merged h5ad:", h5_out)



#!/usr/bin/env python3
import pandas as pd
import numpy as np
from pathlib import Path

# -----------------------------
# CONFIG
# -----------------------------
merged_system = "Brain_spinal_cord"

score_base = Path("/project/imoskowitz/xyang2/chrislowzhengxi/results/score_genes_new")
summary_path = score_base / merged_system / "qc" / f"{merged_system}_SHH_scoregenes_summary.csv"

tree_edge_file = Path("/project/imoskowitz/xyang2/SHH/Qiu_TimeLapse/results/tree/edges_filtered.txt")

outdir = score_base / merged_system
outdir.mkdir(parents=True, exist_ok=True)

# -----------------------------
# LOAD SUMMARY (node-level SHH scores)
# -----------------------------
scores = pd.read_csv(summary_path)

scores = scores.rename(columns={
    "celltype_new": "node_name",
    "mean": "sh_score",
    "n_cells": "n"
})

# -----------------------------
# LOAD EDGES FOR THIS SYSTEM
# -----------------------------
edges = pd.read_csv(tree_edge_file, sep="\t")
edges = edges.loc[edges["system"] == merged_system].copy()

print(f"[EDGE] Found {len(edges)} edges for system={merged_system}")

# -----------------------------
# MERGE SHH SCORES
# -----------------------------

# Attach x-side
edges = edges.merge(
    scores.rename(columns={
        "node_name": "x_name",
        "sh_score": "sh_x",
        "variance": "variance_x",
        "std": "std_x",
        "n": "n_x"
    }),
    on="x_name",
    how="left"
)

# Attach y-side
edges = edges.merge(
    scores.rename(columns={
        "node_name": "y_name",
        "sh_score": "sh_y",
        "variance": "variance_y",
        "std": "std_y",
        "n": "n_y"
    }),
    on="y_name",
    how="left"
)

# -----------------------------
# DELTAS, EFFECT SIZES, PERCENT CHANGE
# -----------------------------
edges["abs_delta"] = (edges["sh_x"] - edges["sh_y"]).abs()
edges["delta"] = edges["sh_y"] - edges["sh_x"]

pooled_std = np.sqrt((edges["variance_x"] + edges["variance_y"]) / 2.0)
edges["cohens_d"] = edges["delta"] / pooled_std.replace(0, np.nan)
edges["cohens_d"] = edges["cohens_d"].round(4)

eps = 1e-9
denom = edges["sh_x"].copy()
denom = denom.where(denom.abs() > eps, np.nan)
edges["pct_change"] = 100.0 * edges["delta"] / denom
edges["abs_pct_change"] = edges["pct_change"].abs()

edges["pct_change"] = edges["pct_change"].round(2)
edges["abs_pct_change"] = edges["abs_pct_change"].round(2)
edges["delta"] = edges["delta"].round(6)
edges["abs_delta"] = edges["abs_delta"].round(6)

# Sort by largest SHH change
edges_sorted = edges.sort_values("abs_delta", ascending=False)

# -----------------------------
# SAVE OUTPUT
# -----------------------------
csv_out = outdir / f"{merged_system}_edge_filtered_with_shh_scoregenes.csv"
txt_out = outdir / f"{merged_system}_edge_filtered_with_shh_scoregenes.txt"

edges_sorted.to_csv(csv_out, index=False)
edges_sorted.to_csv(txt_out, sep="\t", index=False, na_rep="")

print(f"[DONE] Wrote: {csv_out}")
print(f"[DONE] Wrote: {txt_out}")
