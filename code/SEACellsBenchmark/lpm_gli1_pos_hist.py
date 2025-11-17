#!/usr/bin/env python3
import os
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    ap = argparse.ArgumentParser(
        description="LPM: histogram of SHH_UCell_score for cells with Gli1_raw > 0"
    )
    ap.add_argument(
        "--extended-tsv",
        required=True,
        help="Path to Lateral_plate_mesoderm_extended_obs.tsv",
    )
    ap.add_argument(
        "--outdir",
        required=True,
        help="Output directory for figures and tables",
    )
    ap.add_argument("--low-thr", type=float, default=0.35)
    ap.add_argument("--high-thr", type=float, default=0.65)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # Load extended obs
    df = pd.read_csv(args.extended_tsv, sep="\t")

    # Basic counts
    n_total = len(df)

    # Define Gli1 positive cells (raw counts > 0)
    df["Gli1_pos_raw_flag"] = df["Gli1_raw"] > 0
    n_gli1_pos = int(df["Gli1_pos_raw_flag"].sum())
    pct_gli1_pos = 100.0 * n_gli1_pos / n_total if n_total > 0 else 0.0

    print(f"Total LPM cells: {n_total:,}")
    print(f"Gli1_raw > 0 cells: {n_gli1_pos:,} ({pct_gli1_pos:.2f} %)")

    # Restrict to cells with SHH_UCell_score > 0 for the histogram
    df_nonzero_shh = df[df["SHH_UCell_score"] > 0].copy()
    n_nonzero_shh = len(df_nonzero_shh)

    # Among those, keep only Gli1 positive cells
    df_hist = df_nonzero_shh[df_nonzero_shh["Gli1_pos_raw_flag"]].copy()
    n_hist = len(df_hist)

    print(f"Cells used in histogram (SHH_UCell_score > 0 and Gli1_raw > 0): {n_hist:,}")

    # Histogram of SHH_UCell_score for Gli1+ cells
    vals = df_hist["SHH_UCell_score"].astype(float).values

    plt.figure(figsize=(6, 4))
    plt.hist(vals, bins=np.linspace(0.0, 1.0, 51),
             edgecolor="black", linewidth=0.5)
    plt.xlabel("SHH_UCell_score")
    plt.ylabel("Count")
    plt.title(f"LPM Gli1_raw > 0 cells (no SHH zeros, n={n_hist:,})")

    # Vertical lines at thresholds
    plt.axvline(args.low_thr, linestyle="--", linewidth=1.5, color="red")
    plt.axvline(args.high_thr, linestyle="--", linewidth=1.5, color="blue")

    plt.tight_layout()
    out_pdf = os.path.join(args.outdir, "LPM_Gli1pos_SHH_hist.pdf")
    plt.savefig(out_pdf)
    plt.close()
    print("Saved histogram:", out_pdf)

    # Simple summary table
    # 1) overall Gli1+ fraction
    # 2) Gli1+ counts by SHH_UCell_score bin (low, middle, high)
    bins = [0.0, args.low_thr, args.high_thr, 1.0 + 1e-9]
    labels = ["Low", "Middle", "High"]
    df_nonzero_shh["SHH_bin"] = pd.cut(
        df_nonzero_shh["SHH_UCell_score"].astype(float),
        bins=bins,
        labels=labels,
        right=False,
    )

    by_bin = (
        df_nonzero_shh.groupby("SHH_bin")
        .agg(
            n_cells=("SHH_UCell_score", "size"),
            n_Gli1_pos=("Gli1_pos_raw_flag", "sum"),
        )
        .reset_index()
    )
    by_bin["frac_Gli1_pos"] = by_bin["n_Gli1_pos"] / by_bin["n_cells"]
    by_bin["pct_Gli1_pos"] = 100.0 * by_bin["frac_Gli1_pos"]

    # Add overall row at top
    overall = pd.DataFrame(
        {
            "SHH_bin": ["All nonzero SHH cells"],
            "n_cells": [n_nonzero_shh],
            "n_Gli1_pos": [int(df_nonzero_shh["Gli1_pos_raw_flag"].sum())],
        }
    )
    overall["frac_Gli1_pos"] = overall["n_Gli1_pos"] / overall["n_cells"]
    overall["pct_Gli1_pos"] = 100.0 * overall["frac_Gli1_pos"]

    summary = pd.concat([overall, by_bin], ignore_index=True)

    out_csv = os.path.join(args.outdir, "LPM_Gli1_raw_pos_summary.csv")
    summary.to_csv(out_csv, index=False)
    print("Saved table:", out_csv)


if __name__ == "__main__":
    main()
