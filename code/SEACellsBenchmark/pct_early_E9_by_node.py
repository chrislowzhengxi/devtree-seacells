#!/usr/bin/env python3
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# SYSTEMS = [
#     "Blood",
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
#     "Neurons",
#     "Other_Brain_spinal_cord",
# ]

SYSTEM_RENAME = {
    "Neurons": "Brain_spinal_cord",
    "Other_Brain_spinal_cord": "Brain_spinal_cord",
}

SYSTEMS = [
    "Lateral_plate_mesoderm",
]


def staging_to_numeric(x):
    """
    Convert staging strings like:
    E8.0, E8.25, E8.5+, E9.0, E9.75, P0 -> numeric
    Rule for E8.5+ is 8.5.
    P0 becomes a large number so it is never <= E9.
    """
    if pd.isna(x):
        return np.nan

    s = str(x).strip()
    if s == "":
        return np.nan

    if s.upper() == "P0":
        return 100.0

    if s.startswith("E"):
        s2 = s[1:]
        if s2.endswith("+"):
            s2 = s2[:-1]
        try:
            return float(s2)
        except Exception:
            return np.nan

    return np.nan


def build_node_df_for_one_system(fp, system_name, node_col, staging_col, cutoff_E):
    adata = sc.read_h5ad(fp)

    if staging_col not in adata.obs.columns:
        raise KeyError(f"{system_name}: missing staging column '{staging_col}'")

    if node_col not in adata.obs.columns:
        raise KeyError(f"{system_name}: missing node column '{node_col}'")

    df = adata.obs[[node_col, staging_col]].copy()
    df.columns = ["node", "staging"]

    df["stage_num"] = df["staging"].map(staging_to_numeric)
    df["is_early"] = df["stage_num"] <= cutoff_E

    # Per node counts
    g = df.groupby("node", dropna=False)

    out = pd.DataFrame({
        "system": system_name,
        "node": g.size().index.astype(str),
        "n_cells_total": g.size().values,
        "n_cells_with_stage": g["stage_num"].apply(lambda x: np.isfinite(x).sum()).values,
        "n_cells_early_leq_E9": g["is_early"].sum().values,
    })

    # Percent among cells that have staging (recommended)
    out["pct_early_among_staged"] = (
        out["n_cells_early_leq_E9"] / out["n_cells_with_stage"].replace(0, np.nan)
    )

    # Percent among all cells in node (more strict)
    out["pct_early_among_all"] = (
        out["n_cells_early_leq_E9"] / out["n_cells_total"].replace(0, np.nan)
    )

    return out


def plot_box_with_points(df_nodes, ycol, out_pdf):
    """
    Matplotlib-only boxplot, with jittered points.
    """
    systems = sorted(df_nodes["system"].unique().tolist())
    data = []
    for s in systems:
        vals = df_nodes.loc[df_nodes["system"] == s, ycol].dropna().values
        data.append(vals)

    plt.figure(figsize=(12, 5))
    plt.boxplot(data, labels=systems, showfliers=False)

    # jittered points
    rng = np.random.default_rng(0)
    for i, s in enumerate(systems, start=1):
        vals = df_nodes.loc[df_nodes["system"] == s, ycol].dropna().values
        if len(vals) == 0:
            continue
        x = i + rng.normal(0, 0.06, size=len(vals))
        plt.scatter(x, vals, s=12, alpha=0.7)

    plt.ylabel(ycol)
    plt.xlabel("")
    plt.xticks(rotation=45, ha="right")
    plt.title(f"Node early fraction by system (cutoff <= E9.0), metric={ycol}")
    plt.tight_layout()
    plt.savefig(out_pdf)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--staging-root",
        required=True,
        help="Folder containing *_adata_scale_with_staging.h5ad",
    )
    ap.add_argument(
        "--outdir",
        required=True,
        help="Output folder",
    )
    ap.add_argument(
        "--node-col",
        default="meta_group",
        help="Node column in adata.obs, eg meta_group or celltype_new or leiden",
    )
    ap.add_argument(
        "--staging-col",
        default="staging",
        help="Staging column in adata.obs",
    )
    ap.add_argument(
        "--cutoff",
        default=9.0,
        type=float,
        help="E stage cutoff, <= cutoff counts as early. Use 9.0 for 'includes E9'.",
    )
    ap.add_argument(
        "--min-cells",
        default=1,
        type=int,
        help="Drop nodes with fewer than this many total cells",
    )
    args = ap.parse_args()

    staging_root = Path(args.staging_root)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    all_rows = []
    printed_cols = False

    for system in SYSTEMS:
        fp = staging_root / f"{system}_adata_scale_with_staging.h5ad"
        if not fp.exists():
            print(f"[SKIP] missing file: {fp}")
            continue

        print(f"\n=== {system} ===")
        adata_peek = sc.read_h5ad(fp, backed="r")
        if not printed_cols:
            print("[DEBUG] obs columns (first 40):")
            print(list(adata_peek.obs.columns)[:40])
            printed_cols = True
        del adata_peek

        system_out = SYSTEM_RENAME.get(system, system)
        df_sys = build_node_df_for_one_system(
            fp=str(fp),
            system_name=system_out,
            node_col=args.node_col,
            staging_col=args.staging_col,
            cutoff_E=args.cutoff,
        )

        if args.min_cells > 1:
            df_sys = df_sys[df_sys["n_cells_total"] >= args.min_cells].copy()

        all_rows.append(df_sys)

    if len(all_rows) == 0:
        raise RuntimeError("No systems were processed. Check paths and filenames.")

    df_nodes = pd.concat(all_rows, ignore_index=True)

    # Collapse Neurons + Other_Brain_spinal_cord into Brain_spinal_cord
    # (sum counts per (system, node) then recompute percentages)
    df_nodes = (
        df_nodes.groupby(["system", "node"], as_index=False)
        .agg(
            n_cells_total=("n_cells_total", "sum"),
            n_cells_with_stage=("n_cells_with_stage", "sum"),
            n_cells_early_leq_E9=("n_cells_early_leq_E9", "sum"),
        )
    )
    df_nodes["pct_early_among_staged"] = (
        df_nodes["n_cells_early_leq_E9"]
        / df_nodes["n_cells_with_stage"].replace(0, np.nan)
    )
    df_nodes["pct_early_among_all"] = (
        df_nodes["n_cells_early_leq_E9"]
        / df_nodes["n_cells_total"].replace(0, np.nan)
    )

    # Save per node
    out_nodes_csv = outdir / "node_pct_early_leq_E9_by_system.csv"
    df_nodes.to_csv(out_nodes_csv, index=False)
    print(f"\n[SAVE] {out_nodes_csv}")

    # Save per system summary table (median and mean across nodes)
    df_sys_summary = (
        df_nodes.groupby("system")
        .agg(
            n_nodes=("node", "nunique"),
            median_pct_early=("pct_early_among_staged", "median"),
            mean_pct_early=("pct_early_among_staged", "mean"),
            median_pct_early_all=("pct_early_among_all", "median"),
            mean_pct_early_all=("pct_early_among_all", "mean"),
        )
        .reset_index()
    )
    out_sys_csv = outdir / "system_summary_pct_early_leq_E9.csv"
    df_sys_summary.to_csv(out_sys_csv, index=False)
    print(f"[SAVE] {out_sys_csv}")

    # Plot (PDF) using the percent among staged cells
    out_pdf = outdir / "boxplot_node_pct_early_leq_E9_among_staged.pdf"
    plot_box_with_points(df_nodes, "pct_early_among_staged", str(out_pdf))
    print(f"[SAVE] {out_pdf}")

    # Optional second plot using percent among all cells
    out_pdf2 = outdir / "boxplot_node_pct_early_leq_E9_among_all.pdf"
    plot_box_with_points(df_nodes, "pct_early_among_all", str(out_pdf2))
    print(f"[SAVE] {out_pdf2}")

    print("\nDone.")



if __name__ == "__main__":
    main()
