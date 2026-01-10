#!/usr/bin/env python3
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc

STAGING_ORDER = [
    "E8.0", "E8.25", "E8.5", "E8.75",
    "E9.0", "E9.25", "E9.5", "E9.75",
    "E10.0", "E10.25", "E10.5", "E10.75",
    "E11.0", "E11.25", "E11.5", "E11.75",
    "E12.0", "E12.25", "E12.5", "E12.75",
    "E13.0", "E13.25", "E13.5", "E13.75",
    "E14.0", "E14.25", "E14.5", "E14.75",
    "E15.0", "E15.25", "E15.5", "E15.75",
    "E16.0", "E16.25", "E16.5", "E16.75",
    "E17.0", "E17.25", "E17.5", "E17.75",
    "E18.0", "E18.25", "E18.5", "E18.75", "P0"
]

def build_staging_from_day_and_somite(df_meta: pd.DataFrame) -> pd.Series:
    def _map_staging(row):
        d = row["day"]
        if d in ("E8", "E8.0-E8.5", "E8.5"):
            try:
                scount = int(str(row["somite_count"]).split()[0])
            except Exception:
                return np.nan
            if scount <= 3:
                return "E8.0"
            if scount <= 7:
                return "E8.25"
            if scount <= 11:
                return "E8.5"
            return "E8.5+"
        return d
    return df_meta.apply(_map_staging, axis=1)

def add_meta_and_staging_to_adata(adata, df_meta: pd.DataFrame, staging_order):
    if "cell_id" not in adata.obs.columns:
        raise KeyError("adata.obs is missing 'cell_id'")

    df_meta = df_meta.drop_duplicates("cell_id").copy()
    meta_idx = df_meta.set_index("cell_id")

    cols_to_add = [
        "day",
        "staging",
        "somite_count",
        "embryo_id",
        "experimental_id",
        "system",
        "meta_group",
        "celltype_new",
        "embryo_sex",
    ]

    for c in cols_to_add:
        if c not in meta_idx.columns:
            raise KeyError(f"Metadata CSV missing required column: {c}")

    for c in cols_to_add:
        adata.obs[c] = adata.obs["cell_id"].map(meta_idx[c])

    # normalize embryo_sex -> sex
    adata.obs["embryo_sex"] = (
        adata.obs["embryo_sex"].astype(str).str.strip().str.upper()
    )
    adata.obs["embryo_sex"].replace({"M": "MALE", "F": "FEMALE"}, inplace=True)
    adata.obs.rename(columns={"embryo_sex": "sex"}, inplace=True)
    adata.obs["sex"] = adata.obs["sex"].astype("category")

    staging_order_final = list(staging_order)
    if (adata.obs["staging"].astype(str) == "E8.5+").any() and ("E8.5+" not in staging_order_final):
        try:
            i = staging_order_final.index("E8.5")
            staging_order_final.insert(i + 1, "E8.5+")
        except ValueError:
            staging_order_final.append("E8.5+")

    adata.obs["staging"] = pd.Categorical(
        adata.obs["staging"],
        categories=staging_order_final,
        ordered=True,
    )
    return adata, staging_order_final

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--h5-root", required=True, type=str)
    parser.add_argument("--csv-meta", required=True, type=str)
    parser.add_argument("--out-root", required=True, type=str)
    parser.add_argument("--systems", nargs="+", required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    h5_root = Path(args.h5_root)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    req_cols = [
        "cell_id",
        "day",
        "somite_count",
        "embryo_id",
        "experimental_id",
        "system",
        "celltype_new",
        "meta_group",
        "embryo_sex",
    ]
    df_meta = pd.read_csv(args.csv_meta, usecols=req_cols)

    # Debug: show what systems actually exist in metadata
    sys_counts = df_meta["system"].value_counts().head(20)
    print("\n[DEBUG] Top metadata systems:")
    print(sys_counts.to_string())
    print("")

    # This mapping is the key fix.
    # Your metadata system is likely "Brain_spinal_cord" for BOTH of these.
    def meta_system_for(system_tag: str) -> str:
        if system_tag in ["Neurons", "Other_Brain_spinal_cord"]:
            return "Brain_spinal_cord"
        return system_tag

    for system_tag in args.systems:
        in_h5 = h5_root / f"{system_tag}_adata_scale.h5ad"
        out_h5 = out_root / f"{system_tag}_adata_scale_with_staging.h5ad"

        print(f"\n=== {system_tag} ===")
        print(f"[DEBUG] Input h5ad: {in_h5}")
        print(f"[DEBUG] Output h5ad: {out_h5}")

        if not in_h5.exists():
            raise FileNotFoundError(f"Missing input h5ad: {in_h5}")

        if out_h5.exists() and (not args.overwrite):
            raise RuntimeError(f"Output exists but --overwrite not set: {out_h5}")

        adata = sc.read_h5ad(in_h5)
        print(f"[DEBUG] Cells in input h5ad: {adata.n_obs:,}")

        meta_key = meta_system_for(system_tag)
        df_sys = df_meta[df_meta["system"] == meta_key].copy()
        print(f"[DEBUG] Metadata system used: {meta_key}")
        print(f"[DEBUG] Metadata rows after system filter: {len(df_sys):,}")

        if len(df_sys) == 0:
            raise RuntimeError(
                f"Metadata filter returned 0 rows for meta_key='{meta_key}'. "
                f"Check df_meta['system'] values."
            )

        df_sys["staging"] = build_staging_from_day_and_somite(df_sys)
        df_sys = df_sys.drop_duplicates("cell_id")

        # Debug overlap before filtering
        cell_ids_h5 = pd.Index(adata.obs["cell_id"].astype(str))
        cell_ids_meta = pd.Index(df_sys["cell_id"].astype(str))
        overlap = cell_ids_h5.intersection(cell_ids_meta)
        print(f"[DEBUG] Unique cell_id in h5ad: {cell_ids_h5.nunique():,}")
        print(f"[DEBUG] Unique cell_id in meta: {cell_ids_meta.nunique():,}")
        print(f"[DEBUG] Overlap cell_id count: {len(overlap):,}")

        keep = adata.obs["cell_id"].astype(str).isin(cell_ids_meta)
        kept_n = int(keep.sum())
        print(f"[DEBUG] Cells kept after join: {kept_n:,}")

        if kept_n == 0:
            raise RuntimeError(
                "Join kept 0 cells. This means cell_id values do not match between "
                "the input h5ad and the metadata CSV (or you filtered the wrong system)."
            )

        adata = adata[keep].copy()
        adata, staging_order_final = add_meta_and_staging_to_adata(adata, df_sys, STAGING_ORDER)
        adata.obs["stage_code"] = adata.obs["staging"].cat.codes

        # Final safety checks before writing
        if adata.n_obs == 0:
            raise RuntimeError("Refusing to write an empty AnnData object (n_obs==0).")

        staging_nonnull = adata.obs["staging"].notna().sum()
        print(f"[DEBUG] staging non-null cells: {int(staging_nonnull):,} / {adata.n_obs:,}")
        print(f"[DEBUG] staging categories saved: {len(staging_order_final)}")

        print(f"[DEBUG] Writing: {out_h5}")
        adata.write(out_h5)

        # Read back quickly to verify it is not empty on disk
        ad2 = sc.read_h5ad(out_h5)
        print(f"[DEBUG] Re-read output n_obs: {ad2.n_obs:,}")

    print("\nDone.")

if __name__ == "__main__":
    main()
