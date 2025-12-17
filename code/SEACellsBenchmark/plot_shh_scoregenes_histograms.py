#!/usr/bin/env python3
import os
import argparse
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# List of systems to process (no Gastrulation)
SYSTEMS = [
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
]


def load_scores(csv_path, score_col="SHH_scoregenes"):
    """
    Read a *_SHH_scoregenes_percell.csv file and return a 1D Series of scores.
    Assumes the first column is the cell_id and the second is SHH_scoregenes.
    """
    df = pd.read_csv(csv_path, low_memory=False)
    # If there is an unnamed first column with cell IDs, set it as index
    if df.columns[0] == "":
        df = df.set_index(df.columns[0])
    if score_col not in df.columns:
        raise ValueError(f"Column '{score_col}' not found in {csv_path}")
    scores = pd.to_numeric(df[score_col], errors="coerce")
    scores = scores.dropna()
    return scores


def plot_histograms_for_system(scores, system, outdir):
    """
    Make two histograms:
      1) All scores (including zeros)
      2) Only non-zero scores
    """
    os.makedirs(outdir, exist_ok=True)

    # 1. All cells
    n_all = len(scores)
    n_zero = (scores == 0).sum()
    zero_pct = 100.0 * n_zero / n_all if n_all > 0 else 0.0

    plt.figure(figsize=(6, 4))
    plt.hist(scores, bins=100, edgecolor="black")
    plt.xlabel("SHH_scoregenes")
    plt.ylabel("Count")
    plt.title(f"{system} ... All cells (all, n={n_all}, zero={zero_pct:.1f}%)")
    plt.tight_layout()
    out_all = os.path.join(outdir, f"{system}_SHH_scoregenes_hist_all.pdf")
    plt.savefig(out_all)
    plt.close()

    # 2. Non-zero cells only
    scores_nz = scores[scores != 0]
    n_nz = len(scores_nz)

    plt.figure(figsize=(6, 4))
    if n_nz > 0:
        plt.hist(scores_nz, bins=100, edgecolor="black")
    plt.xlabel("SHH_scoregenes")
    plt.ylabel("Count")
    plt.title(f"{system} ... All cells (no-zeros, n={n_nz})")
    plt.tight_layout()
    out_nz = os.path.join(outdir, f"{system}_SHH_scoregenes_hist_nozeros.pdf")
    plt.savefig(out_nz)
    plt.close()

    print(f"[OK] {system}: saved {out_all} and {out_nz}")


def main():
    parser = argparse.ArgumentParser(
        description="Plot SHH score_genes histograms (all vs non-zero) for each system."
    )
    parser.add_argument(
        "--base-dir",
        required=True,
        help="Base directory containing system subfolders, e.g. "
             "/project/imoskowitz/xyang2/chrislowzhengxi/results/score_genes",
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
            f"{system}_SHH_scoregenes_percell.csv",
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
