#!/usr/bin/env python3
import argparse
from pathlib import Path
import pandas as pd
import numpy as np

DEFAULT_GENES = ["Gli1", "Ptch1", "Hhip"]

def bin_shh(s, low_thr=0.35, high_thr=0.65):
    # 0: exactly zero. Low: (0, low_thr). Middle: [low_thr, high_thr). High: >= high_thr.
    out = np.empty(len(s), dtype=object)
    s = pd.to_numeric(s, errors="coerce").fillna(0.0).values
    out[s == 0.0] = "0"
    out[(s > 0.0) & (s < low_thr)] = "Low"
    out[(s >= low_thr) & (s < high_thr)] = "Middle"
    out[s >= high_thr] = "High"
    return pd.Categorical(out, categories=["0", "Low", "Middle", "High"], ordered=True)

def summarize_gene(df, gene, low_thr, high_thr):
    raw_col = f"{gene}_raw"
    if raw_col not in df.columns:
        raise ValueError(f"Column '{raw_col}' not found in input.")
    sub = df[["SHH_UCell_score", raw_col]].copy()
    sub["shh_bin"] = bin_shh(sub["SHH_UCell_score"], low_thr=low_thr, high_thr=high_thr)
    sub["pos"] = (pd.to_numeric(sub[raw_col], errors="coerce").fillna(0.0).values > 0).astype(int)

    grp = sub.groupby("shh_bin", observed=False)
    out = pd.DataFrame({
        "gene": gene,
        "shh_bin": grp.size().index,
        "n_total": grp.size().values,
        "n_pos": grp["pos"].sum().values
    })
    out["n_neg"] = out["n_total"] - out["n_pos"]
    out["pct_pos"] = np.where(out["n_total"] > 0, 100.0 * out["n_pos"] / out["n_total"], np.nan)
    return out


def main():
    ap = argparse.ArgumentParser(description="Build SHH bin tables for Gli1/Ptch1/Hhip from extended_obs.tsv")
    ap.add_argument("--extended-tsv", required=True, help="Path to *_extended_obs.tsv")
    ap.add_argument("--genes", nargs="*", default=DEFAULT_GENES)
    ap.add_argument("--low-thr", type=float, default=0.35)
    ap.add_argument("--high-thr", type=float, default=0.65)
    ap.add_argument("--outdir", default=None, help="Directory for outputs. Default: same folder as input")
    args = ap.parse_args()

    in_path = Path(args.extended_tsv)
    outdir = Path(args.outdir) if args.outdir else in_path.parent
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(in_path, sep="\t")
    if "SHH_UCell_score" not in df.columns:
        raise ValueError("Input is missing SHH_UCell_score column.")

    # # Bind threshold function with chosen cutoffs
    # global bin_shh
    # def bin_shh_local(s):  # shadow with chosen thresholds
    #     return bin_shh(s, low_thr=args.low_thr, high_thr=args.high_thr)
    # bin_shh = bin_shh_local  # rebind for summarize_gene

    all_tables = []
    for g in args.genes:
        t = summarize_gene(df, g, args.low_thr, args.high_thr)
        all_tables.append(t)
        per_path = outdir / f"{g}_by_SHH_bin.tsv"
        t.to_csv(per_path, sep="\t", index=False)
        print(f"[save] {per_path}")

    comb = pd.concat(all_tables, ignore_index=True)
    comb_path = outdir / "SHH_bins_by_gene.tsv"
    comb.to_csv(comb_path, sep="\t", index=False)
    print(f"[save] {comb_path}")

    print("\n=== Summary (counts and % positive) ===")
    order = ["0","Low","Middle","High"]
    for g in args.genes:
        tt = comb[comb["gene"] == g].set_index("shh_bin").reindex(order)
        print(f"\n{g}:")
        print(tt[["n_total","n_pos","n_neg","pct_pos"]].to_string())


if __name__ == "__main__":
    main()
