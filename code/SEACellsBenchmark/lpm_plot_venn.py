#!/usr/bin/env python3
import argparse, os, glob
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib_venn import venn3

def plot_one(tsv_path, outdir, title=None):
    df = pd.read_csv(tsv_path, sep="\t")
    m = dict(zip(df["region"], df["count"]))

    onlyA = int(m.get("Gli1", 0))
    onlyB = int(m.get("Ptch1", 0))
    onlyC = int(m.get("Hhip", 0))
    AB    = int(m.get("Gli1&Ptch1", 0))
    AC    = int(m.get("Gli1&Hhip", 0))
    BC    = int(m.get("Ptch1&Hhip", 0))
    ABC   = int(m.get("Gli1&Ptch1&Hhip", 0))

    # Order expected by venn3: 100,010,110,001,101,011,111
    subsets = (onlyA, onlyB, AB, onlyC, AC, BC, ABC)

    if title is None:
        title = os.path.splitext(os.path.basename(tsv_path))[0]

    plt.figure(figsize=(5.3, 5.0))
    v = venn3(subsets=subsets, set_labels=("Gli1", "Ptch1", "Hhip"))
    plt.title(title)
    os.makedirs(outdir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(tsv_path))[0]
    plt.savefig(os.path.join(outdir, f"{stem}.png"), dpi=300)
    plt.savefig(os.path.join(outdir, f"{stem}.pdf"))
    plt.close()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tsv", default=None)
    ap.add_argument("--folder", default=None)
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()

    if args.tsv:
        plot_one(args.tsv, args.outdir)
    elif args.folder:
        for f in sorted(glob.glob(os.path.join(args.folder, "venn_counts_*.tsv"))):
            suffix = os.path.basename(f).replace("venn_counts_", "").replace(".tsv", "")
            plot_one(f, args.outdir, title=f"Overlap in SHH bin: {suffix}")
    else:
        raise SystemExit("Provide --tsv or --folder")

if __name__ == "__main__":
    main()
