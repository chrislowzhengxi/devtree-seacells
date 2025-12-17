#!/usr/bin/env python3
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import scanpy as sc

# Extend this list to include gastrulation stages you actually have
GASTRULATION_STAGE_ORDER = [
    "E6.5", "E6.75", "E7.0", "E7.25", "E7.5", "E7.75",
    "E8.0", "E8.25", "E8.5",
]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-h5ad", required=True)
    ap.add_argument("--meta-csv", required=True)
    ap.add_argument("--out-h5ad", required=True)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    in_h5ad = Path(args.in_h5ad)
    meta_csv = Path(args.meta_csv)
    out_h5ad = Path(args.out_h5ad)

    if out_h5ad.exists() and (not args.overwrite):
        raise SystemExit(f"Output exists. Use --overwrite. {out_h5ad}")

    print("Loading h5ad:", in_h5ad)
    adata = sc.read_h5ad(in_h5ad)

    print("Loading meta:", meta_csv)
    meta = pd.read_csv(meta_csv)

    # Required columns
    if "cell_id" not in meta.columns:
        raise KeyError("meta csv missing cell_id")
    if "day" not in meta.columns:
        raise KeyError("meta csv missing day")

    # Join key for Gastrulation is obs_names like cell_1
    meta = meta.drop_duplicates("cell_id").set_index("cell_id")

    # Add day and staging
    adata.obs["cell_id"] = adata.obs_names.astype(str)
    adata.obs["day"] = adata.obs["cell_id"].map(meta["day"])

    # For Gastrulation, staging can just equal day (E6.5, E7.0, etc.)
    adata.obs["staging"] = adata.obs["day"]

    # Optional extra annotations if present
    for col in ["celltype_update", "celltype_new", "cell_state", "group"]:
        if col in meta.columns:
            adata.obs[col] = adata.obs["cell_id"].map(meta[col])

    # Make ordered categorical
    present = set(adata.obs["staging"].dropna().astype(str).unique())
    order = [s for s in GASTRULATION_STAGE_ORDER if s in present]
    # keep any unexpected stage labels at the end
    extras = sorted(list(present.difference(order)))
    order_final = order + extras

    adata.obs["staging"] = pd.Categorical(adata.obs["staging"], categories=order_final, ordered=True)
    adata.obs["stage_code"] = adata.obs["staging"].cat.codes

    print("Cells:", adata.n_obs)
    print("Stages:", len(order_final))
    print("Top stage counts:")
    print(adata.obs["staging"].value_counts(dropna=False).head(15))

    out_h5ad.parent.mkdir(parents=True, exist_ok=True)
    print("Writing:", out_h5ad)
    adata.write(out_h5ad)
    print("Done.")

if __name__ == "__main__":
    main()
