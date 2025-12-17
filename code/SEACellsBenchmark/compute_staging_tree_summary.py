#!/usr/bin/env python3
import os
import numpy as np
import pandas as pd


# Paths
META_PATH = "/project/imoskowitz/xyang2/chrislowzhengxi/data/df_cell_celltyp_new_merged.csv"
OUT_DIR   = "/project/imoskowitz/xyang2/chrislowzhengxi/results/ucell/tree"
OUT_FILE  = os.path.join(OUT_DIR, "tree_node_staging_summary.csv")

E9_CUTOFF = 9.0  # E9 and earlier


def map_staging(row):
    """
    Reproduce the staging logic from the SEACell initialize_data function.

    Input columns:
      row["day"], row["somite_count"]

    Returns a string like 'E8.0', 'E8.25', 'E8.5', 'E8.5+', 'E9.0', 'E10.0', etc.
    """
    d = row["day"]

    # Handle special E8 cases where somite_count refines the stage
    if d in ("E8", "E8.0-E8.5", "E8.5"):
        try:
            scount = int(str(row["somite_count"]).split()[0])
        except Exception:
            return np.nan

        if 0 <= scount <= 3:
            return "E8.0"
        elif 4 <= scount <= 7:
            return "E8.25"
        elif 8 <= scount <= 11:
            return "E8.5"
        else:
            return "E8.5+"

    else:
        # For everything else, just keep the day as is
        return d


def parse_stage_to_numeric(stage):
    """
    Convert staging strings to numeric E days.

    Examples:
      'E8.25'  -> 8.25
      'E8.5+'  -> 8.5
      'E9.0'   -> 9.0
      '9.5'    -> 9.5
    """
    if isinstance(stage, str):
        s = stage.strip()

        # Drop leading 'E' if present
        if s.startswith("E"):
            s = s[1:]

        # Remove trailing '+' if present
        s = s.replace("+", "")

        try:
            return float(s)
        except ValueError:
            return np.nan

    # If it is already numeric or something else
    try:
        return float(stage)
    except Exception:
        return np.nan


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    print("Reading metadata:", META_PATH, flush=True)
    df = pd.read_csv(META_PATH, low_memory=False)
    print("Metadata shape:", df.shape, flush=True)
    print("Columns:", df.columns.tolist(), flush=True)

    # Sanity checks
    required_cols = ["cell_id", "day", "somite_count", "system", "celltype_new"]
    for col in required_cols:
        if col not in df.columns:
            raise KeyError(f"Required column '{col}' not found in metadata")

    # Build staging
    print("\nComputing staging from day and somite_count ...", flush=True)
    df["staging"] = df.apply(map_staging, axis=1)

    print("Staging value counts (top 20):")
    print(df["staging"].value_counts(dropna=False).head(20), "\n")

    # Convert staging to numeric E day
    print("Converting staging to numeric (staging_num) ...", flush=True)
    df["staging_num"] = df["staging"].map(parse_stage_to_numeric)

    print("staging_num summary:")
    print(df["staging_num"].describe(), "\n")

    # Flag cells collected at or before E9
    df["before_E9"] = df["staging_num"] <= E9_CUTOFF

    overall_frac = df["before_E9"].mean()
    print(f"Overall fraction of cells collected at or before E9: {overall_frac:.3f}", flush=True)

    # Drop rows where staging_num is missing for the summary
    df_nonan = df.dropna(subset=["staging_num"]).copy()
    print("After dropping NA staging_num:", df_nonan.shape, flush=True)

    # Grouping keys for tree nodes
    group_cols = ["system", "celltype_new"]

    print("\nSummarizing by system and celltype_new ...", flush=True)
    node_staging = (
        df_nonan
        .groupby(group_cols)
        .agg(
            n_cells=("staging_num", "size"),
            median_collection_day=("staging_num", "median"),
            frac_before_E9=("before_E9", "mean")
        )
        .reset_index()
    )

    node_staging["pct_before_E9"] = 100.0 * node_staging["frac_before_E9"]

    print("Preview of node_staging:")
    print(node_staging.head(), "\n")

    # For sanity, show Eye system only if present
    if "Eye" in node_staging["system"].unique():
        print("Preview for Eye system:")
        print(node_staging[node_staging["system"] == "Eye"].head(), "\n")

    # Write output
    node_staging.to_csv(OUT_FILE, index=False)
    print("Wrote node staging summary to:", OUT_FILE, flush=True)
    print("Done.", flush=True)


if __name__ == "__main__":
    main()
