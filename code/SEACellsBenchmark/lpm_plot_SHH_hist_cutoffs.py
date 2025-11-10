#!/usr/bin/env python3
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def plot_hist_rlike(values, title, out_path, cutoffs=(0.35, 0.66), n_bins=60):
    # R-ish aesthetics
    plt.rcParams.update({
        "figure.dpi": 300,
        "axes.facecolor": "white",
        "axes.edgecolor": "black",
        "axes.linewidth": 0.8,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "font.family": "DejaVu Serif"  # simple serif similar to many R defaults
    })

    fig, ax = plt.subplots(figsize=(6.8, 4.6))
    # Histogram with gray fill and black borders, boxed axes
    counts, bins, patches = ax.hist(
        values,
        bins=np.linspace(0, 1, n_bins + 1),
        color=(0.75, 0.75, 0.75),
        edgecolor="black",
        linewidth=0.6
    )

    ax.set_xlim(0, 1)
    ax.set_xlabel("SHH_UCell_score")
    ax.set_ylabel("Count")
    ax.set_title(title)

    # Keep top/right spines to match base R box look
    for side in ["top", "right", "bottom", "left"]:
        ax.spines[side].set_visible(True)

    # Cutoff lines and horizontal labels near the baseline
    ymax = ax.get_ylim()[1]
    ytext = ymax * 0.02
    colors = ["red", "blue"]
    for x, c in zip(cutoffs, colors):
        ax.axvline(x, color=c, linestyle="--", linewidth=1.0)
        ax.text(x, ytext, f"{x:.2f}", color=c, ha="center", va="bottom")

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"[save] {out_path}")

def main():
    ap = argparse.ArgumentParser(description="R-like SHH histogram with cutoffs")
    ap.add_argument("--tsv", required=True, help="TSV with SHH_UCell_score column")
    ap.add_argument("--outdir", required=True, help="Directory to write plots")
    ap.add_argument("--bins", type=int, default=60, help="Number of bins (default 60)")
    ap.add_argument("--cut1", type=float, default=0.35, help="First cutoff")
    ap.add_argument("--cut2", type=float, default=0.66, help="Second cutoff")
    args = ap.parse_args()

    df = pd.read_csv(args.tsv, sep="\t")
    if "SHH_UCell_score" not in df.columns:
        raise ValueError("Missing SHH_UCell_score column")

    s = pd.to_numeric(df["SHH_UCell_score"], errors="coerce").dropna()
    s_nz = s[s > 0]

    # no-zeros panel
    title_nz = f"LPM All cells (no zeros, n={len(s_nz):,})"
    plot_hist_rlike(
        s_nz.values,
        title_nz,
        f"{args.outdir}/LPM_SHH_hist_nozeros_rlike.pdf",
        cutoffs=(args.cut1, args.cut2),
        n_bins=args.bins,
    )

    # all-cells panel
    title_all = f"LPM All cells (all, n={len(s):,}, zero={100.0*(s==0).mean():.1f}%)"
    plot_hist_rlike(
        s.values,
        title_all,
        f"{args.outdir}/LPM_SHH_hist_all_rlike.pdf",
        cutoffs=(args.cut1, args.cut2),
        n_bins=args.bins,
    )

if __name__ == "__main__":
    main()
