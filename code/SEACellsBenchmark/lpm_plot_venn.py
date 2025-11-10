#!/usr/bin/env python3
import argparse, os, glob, warnings
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib_venn import venn3, venn3_unweighted

REGION_IDS = ["100","010","110","001","101","011","111"]  # A,B,AB,C,AC,BC,ABC

def set_all_labels(v, values):
    # values ordered as subsets=(A,B,AB,C,AC,BC,ABC)
    for rid, val in zip(REGION_IDS, values):
        lab = v.get_label_by_id(rid)
        if lab is None:
            # region turned off by library for zero area; create a label anyway
            v.set_labels(("Gli1","Ptch1","Hhip"))  # no-op for safety
            lab = v.get_label_by_id(rid)
        if lab is not None:
            lab.set_text(f"{int(val):,}")

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
    subsets = (onlyA, onlyB, AB, onlyC, AC, BC, ABC)

    if title is None:
        title = os.path.splitext(os.path.basename(tsv_path))[0]

    plt.figure(figsize=(6.8, 6.4))

    used_unweighted = False
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        v = venn3(subsets=subsets, set_labels=("Gli1","Ptch1","Hhip"))
        bad = any("Bad circle positioning" in str(x.message) for x in w)
    # fallback if optimizer could not place circles well
    if v is None or bad:
        plt.clf()
        v = venn3_unweighted(subsets=subsets, set_labels=("Gli1","Ptch1","Hhip"))
        used_unweighted = True

    # ensure all region labels show comma-formatted numbers, including zeros
    set_all_labels(v, subsets)

    plt.title(title)
    out_png = os.path.join(outdir, os.path.splitext(os.path.basename(tsv_path))[0] + ("_unweighted.png" if used_unweighted else ".png"))
    out_pdf = out_png.replace(".png", ".pdf")
    os.makedirs(outdir, exist_ok=True)
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.savefig(out_pdf, bbox_inches="tight")
    plt.close()
    print(f"[save] {out_png}\n[save] {out_pdf}" + (" [unweighted]" if used_unweighted else ""))

def main():
    ap = argparse.ArgumentParser(description="Robust Venn plots from venn_counts_*.tsv with auto fallback")
    ap.add_argument("--tsv", default=None)
    ap.add_argument("--folder", default=None)
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()

    if args.tsv:
        plot_one(args.tsv, args.outdir)
    elif args.folder:
        files = sorted(glob.glob(os.path.join(args.folder, "venn_counts_*.tsv")))
        for f in files:
            suffix = os.path.basename(f).replace("venn_counts_", "").replace(".tsv", "")
            plot_one(f, args.outdir, title=f"Overlap in SHH bin: {suffix}")
    else:
        raise SystemExit("Provide --tsv or --folder")

if __name__ == "__main__":
    main()
