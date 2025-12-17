"""
Docstring for code.SEACellsBenchmark.add_staging_to_all_systems

ls -lh /project/imoskowitz/xyang2/chrislowzhengxi/results/raw_added_with_staging

Use job_staging.sh to run
Notochord PNS_glia PNS_neurons Renal Other_Brain_spinal_cord Neurons
"""

#!/usr/bin/env python3
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc


# Global staging order from your code
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
    """
    Matches your _map_staging(row) logic.
    Creates a string staging value per row.
    """
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


def add_meta_and_staging_to_adata(
    adata,
    df_meta: pd.DataFrame,
    staging_order,
) :
    """
    Adds day, staging, somite_count, embryo_id, experimental_id, system,
    meta_group, celltype_new, sex to adata.obs.
    """

    if "cell_id" not in adata.obs.columns:
        raise KeyError("adata.obs is missing 'cell_id'. This join needs cell_id.")

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

    # normalize embryo_sex to sex, same as your code
    if "embryo_sex" in adata.obs.columns:
        adata.obs["embryo_sex"] = (
            adata.obs["embryo_sex"]
            .astype(str)
            .str.strip()
            .str.upper()
        )
        adata.obs["embryo_sex"].replace({"M": "MALE", "F": "FEMALE"}, inplace=True)
        adata.obs.rename(columns={"embryo_sex": "sex"}, inplace=True)
        adata.obs["sex"] = adata.obs["sex"].astype("category")

    # If "E8.5+" exists, it is not in your STAGING_ORDER list.
    # If we do nothing, it becomes NaN after categorical casting.
    # This keeps your order, but adds E8.5+ right after E8.5.
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

    parser.add_argument(
        "--systems",
        nargs="+",
        default=[
            "Blood",
            "Brain_spinal_cord",
            "Endothelium",
            "Epithelial_cells",
            "Eye",
            "Gut",
            "Lateral_plate_mesoderm",
            "Mesoderm",
            "Notochord",
            "PNS_glia",
            "PNS_neurons",
            "Renal",
        ],
    )

    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    h5_root = Path(args.h5_root)
    csv_meta = Path(args.csv_meta)
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
    df_meta = pd.read_csv(csv_meta, usecols=req_cols)

    for system_tag in args.systems:
        in_h5 = h5_root / f"{system_tag}_adata_scale.h5ad"
        out_h5 = out_root / f"{system_tag}_adata_scale_with_staging.h5ad"

        if (not in_h5.exists()):
            print(f"[SKIP] missing input: {in_h5}")
            continue

        if out_h5.exists() and (not args.overwrite):
            print(f"[SKIP] output exists (use --overwrite): {out_h5}")
            continue

        print(f"\n=== {system_tag} ===")
        print(f"Loading: {in_h5}")
        adata = sc.read_h5ad(in_h5)

        df_sys = df_meta[df_meta["system"] == system_tag].copy()
        print(f"Metadata rows (system filtered): {len(df_sys):,}")

        # build staging
        df_sys["staging"] = build_staging_from_day_and_somite(df_sys)

        # keep only cells that exist in both adata and metadata
        df_sys = df_sys.drop_duplicates("cell_id")
        keep = adata.obs["cell_id"].isin(df_sys["cell_id"])
        print(f"Cells in h5ad: {adata.n_obs:,}")
        print(f"Cells kept after join: {int(keep.sum()):,}")
        adata = adata[keep].copy()

        # add columns + staging categorical
        adata, staging_order_final = add_meta_and_staging_to_adata(
            adata,
            df_sys,
            STAGING_ORDER,
        )

        # Optional numeric code. Comment out if you do not want it saved.
        adata.obs["stage_code"] = adata.obs["staging"].cat.codes

        print("Staging categories saved:", len(staging_order_final))
        print("Writing:", out_h5)
        adata.write(out_h5)

        del adata

    print("\nDone.")


if __name__ == "__main__":
    main()
