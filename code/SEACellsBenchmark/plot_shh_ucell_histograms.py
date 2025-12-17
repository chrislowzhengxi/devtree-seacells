#!/usr/bin/env python3
"""
python plot_shh_ucell_histograms.py \
  --base-dir /project/imoskowitz/xyang2/chrislowzhengxi/results/ucell \
  --outdir /project/imoskowitz/xyang2/chrislowzhengxi/results/ucell/histograms
"""
import os
import argparse
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# SYSTEMS = [
#     "Blood",
#     "Brain_spinal_cord",
#     "Endothelium",
#     "Epithelial_cells",
#     "Eye",
#     "Gut",
#     "Lateral_plate_mesoderm",
#     "Mesoderm",
#     "Notochord",
#     "PNS_glia",
#     "PNS_neurons",
#     "Renal",
# ]
SYSTEMS = [
    "Brain_spinal_cord"
]


def load_scores(csv_path, score_col="SHH_UCell_score"):
    df = pd.read_csv(csv_path, low_memory=False)
    if score_col not in df.columns:
        raise ValueError(f"Column '{score_col}' not found in {csv_path}")
    scores = pd.to_numeric(df[score_col], errors="coerce")
    scores = scores.dropna()
    return scores


def plot_histograms_for_system(scores, system, outdir):
    os.makedirs(outdir, exist_ok=True)

    # All cells
    n_all = len(scores)
    n_zero = (scores == 0).sum()
    zero_pct = 100.0 * n_zero / n_all if n_all > 0 else 0.0

    plt.figure(figsize=(6, 4))
    plt.hist(scores, bins=100, edgecolor="black")
    plt.xlabel("SHH_UCell")
    plt.ylabel("Count")
    plt.title(f"{system} ... All cells (all, n={n_all}, zero={zero_pct:.1f}%)")
    plt.tight_layout()
    out_all = os.path.join(outdir, f"{system}_SHH_UCell_hist_all.pdf")
    plt.savefig(out_all)
    plt.close()

    # Non-zero only
    scores_nz = scores[scores != 0]
    n_nz = len(scores_nz)

    plt.figure(figsize=(6, 4))
    if n_nz > 0:
        plt.hist(scores_nz, bins=100, edgecolor="black")
    plt.xlabel("SHH_UCell")
    plt.ylabel("Count")
    plt.title(f"{system} ... All cells (no-zeros, n={n_nz})")
    plt.tight_layout()
    out_nz = os.path.join(outdir, f"{system}_SHH_UCell_hist_nozeros.pdf")
    plt.savefig(out_nz)
    plt.close()

    print(f"[OK] {system}: saved {out_all} and {out_nz}")


def main():
    parser = argparse.ArgumentParser(
        description="Plot SHH UCell histograms (all vs non-zero) for each system."
    )
    parser.add_argument(
        "--base-dir",
        required=True,
        help="Base directory containing system subfolders, e.g. "
             "/project/imoskowitz/xyang2/chrislowzhengxi/results/ucell",
    )
    parser.add_argument(
        "--outdir",
        required=True,
        help="Directory where all histogram PDFs will be written.",
    )
    args = parser.parse_args()

    for system in SYSTEMS:
        csv_path = os.path.join(
            args.base_dir,
            system,
            f"{system}_SHH_UCell_scores.csv",
        )
        if not os.path.isfile(csv_path):
            print(f"[WARN] Missing file for {system}: {csv_path}")
            continue

        print(f"[INFO] Processing {system} from {csv_path}")
        scores = load_scores(csv_path)
        plot_histograms_for_system(scores, system, args.outdir)

    print("✓ Done.")


if __name__ == "__main__":
    main()
