import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def gaussian_kernel(sigma_bins: float, radius: int):
    xs = np.arange(-radius, radius + 1)
    w = np.exp(-(xs**2) / (2 * sigma_bins**2))
    w = w / w.sum()
    return w

def smooth_1d(y: np.ndarray, sigma_bins: float = 2.0):
    radius = int(max(3, round(4 * sigma_bins)))
    w = gaussian_kernel(sigma_bins=sigma_bins, radius=radius)
    return np.convolve(y, w, mode="same")

def find_local_extrema(y: np.ndarray):
    # indices where derivative changes sign
    dy = np.diff(y)
    s = np.sign(dy)
    ds = np.diff(s)

    # peak: + to -
    peak_idx = np.where(ds < 0)[0] + 1
    # valley: - to +
    valley_idx = np.where(ds > 0)[0] + 1
    return peak_idx, valley_idx

def choose_valley_between_modes(bin_centers, y_smooth):
    peak_idx, valley_idx = find_local_extrema(y_smooth)

    if len(peak_idx) < 2 or len(valley_idx) < 1:
        return None

    # pick strongest peak on negative side and strongest peak on positive side
    neg_peaks = [i for i in peak_idx if bin_centers[i] < 0]
    pos_peaks = [i for i in peak_idx if bin_centers[i] > 0]

    if len(neg_peaks) == 0 or len(pos_peaks) == 0:
        return None

    neg_peak = max(neg_peaks, key=lambda i: y_smooth[i])
    pos_peak = max(pos_peaks, key=lambda i: y_smooth[i])

    left = min(neg_peak, pos_peak)
    right = max(neg_peak, pos_peak)

    candidate_valleys = [i for i in valley_idx if left < i < right]
    if len(candidate_valleys) == 0:
        return None

    best_valley = min(candidate_valleys, key=lambda i: y_smooth[i])
    return float(bin_centers[best_valley])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="CSV with per-cell scanpy scores")
    ap.add_argument("--col", default="SHH_scoregenes", help="Column name for scanpy score")
    ap.add_argument("--nozeros", action="store_true", help="Drop zeros before plotting")
    ap.add_argument("--bins", type=int, default=100)
    ap.add_argument("--smooth_sigma_bins", type=float, default=2.0)
    ap.add_argument("--out", default="scanpy_hist_with_threshold.png")
    ap.add_argument("--title", default="Scanpy score_genes histogram")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    x = df[args.col].astype(float).to_numpy()

    if args.nozeros:
        x = x[x != 0]

    # histogram
    counts, edges = np.histogram(x, bins=args.bins)
    centers = 0.5 * (edges[:-1] + edges[1:])

    # smooth counts and find threshold
    counts_s = smooth_1d(counts.astype(float), sigma_bins=args.smooth_sigma_bins)
    thr = choose_valley_between_modes(centers, counts_s)

    # plot
    plt.figure()
    plt.hist(x, bins=args.bins)
    plt.xlabel(args.col)
    plt.ylabel("Count")
    plt.title(args.title)

    if thr is not None:
        plt.axvline(thr, linewidth=2)
        # light shading for the "positive" group
        plt.axvspan(thr, np.max(x), alpha=0.15)
        plt.text(
            thr, max(counts) * 0.95,
            f"threshold = {thr:.3f}",
            rotation=90, va="top", ha="right"
        )
    else:
        plt.text(
            0.02, 0.98,
            "Could not auto-detect valley.\nUse a manual threshold.",
            transform=plt.gca().transAxes,
            va="top"
        )

    plt.tight_layout()
    plt.savefig(args.out, dpi=300)
    print(f"[SAVE] {args.out}")
    if thr is not None:
        print(f"[THRESHOLD] {thr:.6f}")

if __name__ == "__main__":
    main()
