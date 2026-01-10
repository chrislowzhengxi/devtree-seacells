#!/usr/bin/env python3
from __future__ import annotations

import os
from pathlib import Path
import pandas as pd
import anndata as ad


def summarize_one(h5ad_path: str, system: str) -> pd.DataFrame:
    """
    Summarize Scanpy SHH scoregenes per celltype for one h5ad.
    """
    a = ad.read_h5ad(h5ad_path, backed="r")
    obs = a.obs

    # Resolve celltype column
    if "celltype_new" in obs.columns:
        celltype_col = "celltype_new"
    elif "celltype_update" in obs.columns:
        celltype_col = "celltype_update"
    else:
        raise KeyError(
            f"{os.path.basename(h5ad_path)} missing celltype_new / celltype_update"
        )

    # Resolve Scanpy SHH score column
    score_col = None
    for c in obs.columns:
        if c.lower() in {"shh_scoregenes", "shh_scanpy_score", "shh_score"}:
            score_col = c
            break

    if score_col is None:
        raise KeyError(
            f"{os.path.basename(h5ad_path)} missing SHH scoregenes column"
        )

    # Pull only required columns
    df = obs[[celltype_col, score_col]].copy()
    df.rename(columns={celltype_col: "celltype_new"}, inplace=True)

    df["celltype_new"] = df["celltype_new"].astype(str)
    df[score_col] = pd.to_numeric(df[score_col], errors="coerce")

    out = (
        df.dropna(subset=[score_col])
          .groupby("celltype_new", sort=False)[score_col]
          .agg(
              n_cells="size",
              mean_scanpy="mean",
              pct_scanpy_pos=lambda x: (x > 0).mean(),
          )
          .reset_index()
    )

    out.insert(0, "system", system)
    return out


def combine_weighted(group: pd.DataFrame) -> pd.Series:
    """
    Combine multiple summaries for the same (system, celltype)
    using n_cells weighting.
    """
    n = group["n_cells"].sum()
    if n == 0:
        return pd.Series(
            {"n_cells": 0, "mean_scanpy": float("nan"), "pct_scanpy_pos": float("nan")}
        )

    mean = (group["mean_scanpy"] * group["n_cells"]).sum() / n
    pct = (group["pct_scanpy_pos"] * group["n_cells"]).sum() / n

    return pd.Series(
        {"n_cells": n, "mean_scanpy": mean, "pct_scanpy_pos": pct}
    )


def main():

    # -------------------------
    # Input h5ads with Scanpy SHH scoregenes
    # -------------------------
    files = {
        "Blood": [
            "/project/imoskowitz/xyang2/chrislowzhengxi/results/score_genes_new/Blood/Blood_adata_with_SHH_scoregenes.h5ad",
        ],
        "Gut": [
            "/project/imoskowitz/xyang2/chrislowzhengxi/results/score_genes_new/Gut/Gut_adata_with_SHH_scoregenes.h5ad",
        ],
        "Gastrulation": [
            "/project/imoskowitz/xyang2/chrislowzhengxi/results/score_genes_new/Gastrulation/Gastrulation_adata_with_SHH_scoregenes.h5ad",
        ],
        "Mesoderm": [
            "/project/imoskowitz/xyang2/chrislowzhengxi/results/score_genes_new/Mesoderm/Mesoderm_adata_with_SHH_scoregenes.h5ad",
        ],
        "Notochord": [
            "/project/imoskowitz/xyang2/chrislowzhengxi/results/score_genes_new/Notochord/Notochord_adata_with_SHH_scoregenes.h5ad",
        ],
        "Endothelium": [
            "/project/imoskowitz/xyang2/chrislowzhengxi/results/score_genes_new/Endothelium/Endothelium_adata_with_SHH_scoregenes.h5ad",
        ],
        "Epithelial_cells": [
            "/project/imoskowitz/xyang2/chrislowzhengxi/results/score_genes_new/Epithelial_cells/Epithelial_cells_adata_with_SHH_scoregenes.h5ad",
        ],
        "Eye": [
            "/project/imoskowitz/xyang2/chrislowzhengxi/results/score_genes_new/Eye/Eye_adata_with_SHH_scoregenes.h5ad",
        ],
        "Lateral_plate_mesoderm": [
            "/project/imoskowitz/xyang2/chrislowzhengxi/results/score_genes_new/Lateral_plate_mesoderm/Lateral_plate_mesoderm_adata_with_SHH_scoregenes.h5ad",
        ],
        "Renal": [
            "/project/imoskowitz/xyang2/chrislowzhengxi/results/score_genes_new/Renal/Renal_adata_with_SHH_scoregenes.h5ad",
        ],
        "PNS_neurons": [
            "/project/imoskowitz/xyang2/chrislowzhengxi/results/score_genes_new/PNS_neurons/PNS_neurons_adata_with_SHH_scoregenes.h5ad",
        ],
        "PNS_glia": [
            "/project/imoskowitz/xyang2/chrislowzhengxi/results/score_genes_new/PNS_glia/PNS_glia_adata_with_SHH_scoregenes.h5ad",
        ],
        # Brain = Neurons + Other_Brain_spinal_cord
        "Brain_spinal_cord": [
            "/project/imoskowitz/xyang2/chrislowzhengxi/results/score_genes_new/Neurons/Neurons_adata_with_SHH_scoregenes.h5ad",
            "/project/imoskowitz/xyang2/chrislowzhengxi/results/score_genes_new/Other_Brain_spinal_cord/Other_Brain_spinal_cord_adata_with_SHH_scoregenes.h5ad",
        ],
    }

    rows = []

    for system, paths in files.items():
        for p in paths:
            print(f"[READ] {system}: {Path(p).name}")
            rows.append(summarize_one(p, system))

    df_all = pd.concat(rows, ignore_index=True)

    # -------------------------
    # Merge Brain_spinal_cord pieces safely
    # -------------------------
    df_all = (
        df_all.groupby(["system", "celltype_new"], as_index=False)
              .apply(lambda g: combine_weighted(g), include_groups=False)
              .reset_index()
    )

    out_path = Path("celltype_scanpy_summary_by_system.tsv")
    df_all.to_csv(out_path, sep="\t", index=False)

    print(f"[DONE] wrote {out_path.resolve()}")


if __name__ == "__main__":
    main()
