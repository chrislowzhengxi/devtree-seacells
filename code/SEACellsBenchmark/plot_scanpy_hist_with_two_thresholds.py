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

def find_local_peaks(y: np.ndarray):
    dy = np.diff(y)
    s = np.sign(dy)
    ds = np.diff(s)
    peak_idx = np.where(ds < 0)[0] + 1
    return peak_idx

def find_valley_between(i_left: int, i_right: int, y_smooth: np.ndarray, centers: np.ndarray):
    if i_left > i_right:
        i_left, i_right = i_right, i_left
    if i_right - i_left < 3:
        return None
    segment = y_smooth[i_left:i_right + 1]
    j = int(np.argmin(segment))
    idx = i_left + j
    return float(centers[idx])

def pick_strongest_peak_in_range(peak_idx, centers, y_smooth, lo, hi):
    candidates = [i for i in peak_idx if (centers[i] >= lo and centers[i] <= hi)]
    if not candidates:
        return None
    best = max(candidates, key=lambda i: y_smooth[i])
    return best

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--col", default="SHH_scoregenes")
    ap.add_argument("--nozeros", action="store_true")
    ap.add_argument("--bins", type=int, default=120)
    ap.add_argument("--smooth_sigma_bins", type=float, default=2.5)
    ap.add_argument("--out", required=True)
    ap.add_argument("--title", default="Scanpy score_genes histogram (no-zeros)")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    x = df[args.col].astype(float).to_numpy()
    if args.nozeros:
        x = x[x != 0]

    counts, edges = np.histogram(x, bins=args.bins)
    centers = 0.5 * (edges[:-1] + edges[1:])
    counts_s = smooth_1d(counts.astype(float), sigma_bins=args.smooth_sigma_bins)

    peak_idx = find_local_peaks(counts_s)

    # Three regions. You can tweak these if needed.
    neg_peak = pick_strongest_peak_in_range(peak_idx, centers, counts_s, lo=-10, hi=0)
    mid_peak = pick_strongest_peak_in_range(peak_idx, centers, counts_s, lo=0, hi=4)
    high_peak = pick_strongest_peak_in_range(peak_idx, centers, counts_s, lo=4, hi=20)

    thr1 = None
    thr2 = None

    if neg_peak is not None and mid_peak is not None:
        thr1 = find_valley_between(neg_peak, mid_peak, counts_s, centers)

    if mid_peak is not None and high_peak is not None:
        thr2 = find_valley_between(mid_peak, high_peak, counts_s, centers)

    plt.figure()
    plt.hist(x, bins=args.bins)
    plt.xlabel(args.col)
    plt.ylabel("Count")
    plt.title(args.title)

    ymax = max(counts) if len(counts) else 1

    if thr1 is not None:
        plt.axvline(thr1, linewidth=2)
        plt.text(thr1, ymax * 0.95, f"thr1 = {thr1:.3f}", rotation=90, va="top", ha="right")

    if thr2 is not None:
        plt.axvline(thr2, linewidth=2)
        plt.text(thr2, ymax * 0.95, f"thr2 = {thr2:.3f}", rotation=90, va="top", ha="left")

    # Optional shading to show three groups
    if thr1 is not None and thr2 is not None:
        plt.axvspan(thr1, thr2, alpha=0.12)
        plt.axvspan(thr2, np.max(x), alpha=0.12)

    plt.tight_layout()
    plt.savefig(args.out, dpi=300)
    print(f"[SAVE] {args.out}")
    if thr1 is not None:
        print(f"[THR1] {thr1:.6f}")
    if thr2 is not None:
        print(f"[THR2] {thr2:.6f}")

if __name__ == "__main__":
    main()
