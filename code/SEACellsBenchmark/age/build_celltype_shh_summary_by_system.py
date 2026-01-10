#!/usr/bin/env python3
from __future__ import annotations

import os
from pathlib import Path
import pandas as pd
import anndata as ad


# def summarize_one(h5ad_path: str, system: str) -> pd.DataFrame:
#     # backed="r" avoids loading X into memory
#     a = ad.read_h5ad(h5ad_path, backed="r")
#     obs = a.obs

#     need = ["celltype_new", "SHH_UCell_score"]
#     for c in need:
#         if c not in obs.columns:
#             raise KeyError(f"{os.path.basename(h5ad_path)} missing obs column: {c}")

#     # Pull only required columns into memory (small relative to full object)
#     df = obs[need].copy()
#     df["celltype_new"] = df["celltype_new"].astype(str)

#     # SHH_UCell_score should be numeric
#     df["SHH_UCell_score"] = pd.to_numeric(df["SHH_UCell_score"], errors="coerce")

#     out = (
#         df.dropna(subset=["SHH_UCell_score"])
#           .groupby("celltype_new", sort=False)["SHH_UCell_score"]
#           .agg(
#               n_cells="size",
#               mean_ucell="mean",
#               pct_ucell_pos=lambda x: (x > 0).mean(),
#           )
#           .reset_index()
#     )
#     out.insert(0, "system", system)
#     return out

def summarize_one(h5ad_path: str, system: str) -> pd.DataFrame:
    a = ad.read_h5ad(h5ad_path, backed="r")
    obs = a.obs

    # --- Resolve celltype column ---
    if "celltype_new" in obs.columns:
        celltype_col = "celltype_new"
    elif "celltype_update" in obs.columns:
        celltype_col = "celltype_update"
    else:
        raise KeyError(
            f"{os.path.basename(h5ad_path)} missing both celltype_new and celltype_update"
        )

    if "SHH_UCell_score" not in obs.columns:
        raise KeyError(
            f"{os.path.basename(h5ad_path)} missing obs column: SHH_UCell_score"
        )

    # Pull only required columns
    df = obs[[celltype_col, "SHH_UCell_score"]].copy()
    df.rename(columns={celltype_col: "celltype_new"}, inplace=True)

    df["celltype_new"] = df["celltype_new"].astype(str)
    df["SHH_UCell_score"] = pd.to_numeric(df["SHH_UCell_score"], errors="coerce")

    out = (
        df.dropna(subset=["SHH_UCell_score"])
          .groupby("celltype_new", sort=False)["SHH_UCell_score"]
          .agg(
              n_cells="size",
              mean_ucell="mean",
              pct_ucell_pos=lambda x: (x > 0).mean(),
          )
          .reset_index()
    )

    out.insert(0, "system", system)
    return out



def main():
    # UCELL score h5ads (no staging needed)
    files = {
        "Blood": [
            "/project/imoskowitz/xyang2/chrislowzhengxi/results/ucell/Blood/Blood_adata_with_ucell.h5ad",
        ],
        "Gut": [
            "/project/imoskowitz/xyang2/chrislowzhengxi/results/ucell/Gut/Gut_adata_with_ucell.h5ad",
        ],
        "Mesoderm": [
            "/project/imoskowitz/xyang2/chrislowzhengxi/results/ucell/Mesoderm/Mesoderm_adata_with_ucell.h5ad",
        ],
        "Lateral_plate_mesoderm": [
            "/project/imoskowitz/xyang2/chrislowzhengxi/results/ucell/Lateral_plate_mesoderm/Lateral_plate_mesoderm_adata_with_ucell.h5ad",
        ],
        "Gastrulation": [
            "/project/imoskowitz/xyang2/chrislowzhengxi/results/ucell/Gastrulation/Gastrulation_adata_with_ucell_with_staging.h5ad",
        ],
        "Notochord": [
            "/project/imoskowitz/xyang2/chrislowzhengxi/results/ucell/Notochord/Notochord_adata_with_ucell.h5ad",
        ],
        "Endothelium": [
            "/project/imoskowitz/xyang2/chrislowzhengxi/results/ucell/Endothelium/Endothelium_adata_with_ucell.h5ad",
        ],
        "Epithelial_cells": [
            "/project/imoskowitz/xyang2/chrislowzhengxi/results/ucell/Epithelial_cells/Epithelial_cells_adata_with_ucell.h5ad",
        ],
        "Eye": [
            "/project/imoskowitz/xyang2/chrislowzhengxi/results/ucell/Eye/Eye_adata_with_ucell.h5ad",
        ],
        "Renal": [
            "/project/imoskowitz/xyang2/chrislowzhengxi/results/ucell/Renal/Renal_adata_with_ucell.h5ad",
        ],
        "PNS_neurons": [
            "/project/imoskowitz/xyang2/chrislowzhengxi/results/ucell/PNS_neurons/PNS_neurons_adata_with_ucell.h5ad",
        ],
        "PNS_glia": [
            "/project/imoskowitz/xyang2/chrislowzhengxi/results/ucell/PNS_glia/PNS_glia_adata_with_ucell.h5ad",
        ],
        "Brain_spinal_cord": [
            "/project/imoskowitz/xyang2/chrislowzhengxi/results/ucell/Neurons/Neurons_adata_with_ucell.h5ad",
            "/project/imoskowitz/xyang2/chrislowzhengxi/results/ucell/Other_Brain_spinal_cord/Other_Brain_spinal_cord_adata_with_ucell.h5ad",
        ],
    }



    rows = []
    for system, paths in files.items():
        for p in paths:
            print(f"[READ] {system}: {Path(p).name}")
            rows.append(summarize_one(p, system))

    df_all = pd.concat(rows, ignore_index=True)

    # If Brain_spinal_cord comes from two files, combine them (weighted by n_cells)
    # Do a safe aggregation:
    def combine(group: pd.DataFrame) -> pd.Series:
        n = group["n_cells"].sum()
        if n == 0:
            return pd.Series({"n_cells": 0, "mean_ucell": float("nan"), "pct_ucell_pos": float("nan")})
        mean = (group["mean_ucell"] * group["n_cells"]).sum() / n
        pct = (group["pct_ucell_pos"] * group["n_cells"]).sum() / n
        return pd.Series({"n_cells": n, "mean_ucell": mean, "pct_ucell_pos": pct})

    df_all = (
        df_all.groupby(["system", "celltype_new"], as_index=False)
              .apply(lambda g: combine(g), include_groups=False)
              .reset_index()
    )

    out_path = Path("celltype_ucell_summary_by_system.tsv")
    df_all.to_csv(out_path, sep="\t", index=False)
    print(f"[DONE] wrote {out_path.resolve()}")


if __name__ == "__main__":
    main()
