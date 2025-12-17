#!/usr/bin/env python3
import argparse
from pathlib import Path
import re

import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt


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


def normalize_system_label(system: str) -> str:
    if system in {"Other_Brain_spinal_cord", "Neurons"}:
        return "Brain_spinal_cord"
    return system


def stage_to_float(x) -> float:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return np.nan
    s = str(x).strip()
    if s == "" or s.lower() == "nan":
        return np.nan
    if s.upper() == "P0":
        return 19.0
    m = re.match(r"^[Ee](\d+(\.\d+)?)([a-zA-Z\+]+)?$", s)
    if not m:
        return np.nan
    base = float(m.group(1))
    suffix = m.group(3) or ""
    if "+" in suffix:
        return base + 0.05
    if suffix.lower() in {"a", "b"}:
        return base
    return base


def find_ucell_column(obs_cols) -> str:
    cols = list(obs_cols)
    preferred = []
    for c in cols:
        s = c.lower()
        if "ucell" in s and "shh" in s:
            preferred.append(c)
    if preferred:
        return preferred[0]
    any_ucell = [c for c in cols if "ucell" in c.lower()]
    if any_ucell:
        return any_ucell[0]
    raise KeyError(
        "Could not find a UCell column in the ucell h5ad obs. "
        f"Available obs columns (first 40): {cols[:40]}"
    )


def find_node_column(obs_cols, user_node_col=None) -> str:
    cols = list(obs_cols)
    if user_node_col and user_node_col in cols:
        return user_node_col
    for c in ["meta_group", "celltype_new", "celltype_update", "leiden"]:
        if c in cols:
            return c
    raise KeyError(
        "Could not infer node column. Pass --node-col explicitly. "
        f"Available obs columns (first 40): {cols[:40]}"
    )


def get_ucell_key_series(adata_ucell) -> pd.Series:
    obs = adata_ucell.obs
    if "cell_id" in obs.columns:
        return obs["cell_id"].astype(str)
    # fall back to obs_names, like cell_1, cell_2
    return pd.Series(adata_ucell.obs_names.astype(str), index=adata_ucell.obs_names)


def load_and_join_one_system(system: str, staging_root: Path, ucell_root: Path, node_col_hint: str):
    # staging file (always from raw_added_with_staging)
    if system == "Brain_spinal_cord":
        # Option A: combine staging from both files
        staging_files = [
            staging_root / "Other_Brain_spinal_cord_adata_scale_with_staging.h5ad",
            staging_root / "Neurons_adata_scale_with_staging.h5ad",
        ]
        ucell_files = [
            ucell_root / "Other_Brain_spinal_cord" / "Other_Brain_spinal_cord_adata_with_ucell.h5ad",
            ucell_root / "Neurons" / "Neurons_adata_with_ucell.h5ad",
        ]
    else:
        staging_files = [staging_root / f"{system}_adata_scale_with_staging.h5ad"]
        ucell_files = [ucell_root / system / f"{system}_adata_with_ucell.h5ad"]

    # load staging mapping: cell_id -> staging
    map_parts = []
    for sf in staging_files:
        if not sf.exists():
            print(f"[SKIP] missing staging: {sf}")
            continue
        a_s = sc.read_h5ad(str(sf), backed="r")
        if "cell_id" not in a_s.obs.columns:
            raise KeyError(f"Staging file missing cell_id: {sf}")
        if "staging" not in a_s.obs.columns:
            raise KeyError(f"Staging file missing staging: {sf}")
        tmp = pd.DataFrame({
            "cell_id": a_s.obs["cell_id"].astype(str).values,
            "staging": a_s.obs["staging"].astype(str).values,
        }).drop_duplicates("cell_id")
        map_parts.append(tmp)

    if not map_parts:
        return pd.DataFrame()

    staging_map = pd.concat(map_parts, ignore_index=True).drop_duplicates("cell_id")
    staging_map = staging_map.set_index("cell_id")["staging"]

    # now load ucell files, join staging, then summarize per node
    node_rows = []
    for uf in ucell_files:
        if not uf.exists():
            print(f"[SKIP] missing ucell: {uf}")
            continue

        a_u = sc.read_h5ad(str(uf), backed="r")
        node_col = find_node_column(a_u.obs.columns, user_node_col=node_col_hint)
        ucell_col = find_ucell_column(a_u.obs.columns)

        key_series = get_ucell_key_series(a_u)
        # map staging
        staging_vals = key_series.map(staging_map)

        df = pd.DataFrame({
            "system": system,
            "node": a_u.obs[node_col].astype(str).values,
            "ucell": pd.to_numeric(a_u.obs[ucell_col], errors="coerce").values,
            "staging": staging_vals.values,
        })

        df["stage_float"] = [stage_to_float(v) for v in df["staging"]]
        df = df.dropna(subset=["node", "ucell", "stage_float"])

        g = df.groupby(["system", "node"], observed=True)
        out = pd.DataFrame({
            "n_cells": g.size().astype(int),
            "mean_stage": g["stage_float"].mean(),
            "median_stage": g["stage_float"].median(),
            "mean_ucell": g["ucell"].mean(),
        }).reset_index()

        node_rows.append(out)

    if not node_rows:
        return pd.DataFrame()

    # if Brain_spinal_cord has two ucell files, combine their node summaries by pooling rows
    df_nodes = pd.concat(node_rows, ignore_index=True)

    # recompute by node with weighting by n_cells
    def wmean(x, w):
        return (x * w).sum() / w.sum()

    pooled = []
    for (sys, node), sub in df_nodes.groupby(["system", "node"], observed=True):
        w = sub["n_cells"].astype(float)
        pooled.append({
            "system": sys,
            "node": node,
            "n_cells": int(sub["n_cells"].sum()),
            "mean_stage": wmean(sub["mean_stage"], w),
            "median_stage": wmean(sub["median_stage"], w),
            "mean_ucell": wmean(sub["mean_ucell"], w),
        })
    return pd.DataFrame(pooled)


def plot_scatter(df_nodes: pd.DataFrame, x_col: str, outpath: Path, title: str):
    systems = sorted(df_nodes["system"].unique().tolist())
    fig, ax = plt.subplots(figsize=(10, 6))
    for sys in systems:
        sub = df_nodes[df_nodes["system"] == sys]
        ax.scatter(sub[x_col], sub["mean_ucell"], s=10, alpha=0.7, label=sys)
    ax.set_xlabel(x_col)
    ax.set_ylabel("mean UCell score per node")
    ax.set_title(title)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
    fig.tight_layout()
    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--staging-root", required=True, type=str)
    ap.add_argument("--ucell-root", required=True, type=str)
    ap.add_argument("--outdir", required=True, type=str)
    ap.add_argument("--node-col", default="meta_group", type=str)
    args = ap.parse_args()

    staging_root = Path(args.staging_root)
    ucell_root = Path(args.ucell_root)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    all_nodes = []
    for sys in SYSTEMS:
        print(f"\n=== {sys} ===")
        df = load_and_join_one_system(sys, staging_root, ucell_root, args.node_col)
        if not df.empty:
            all_nodes.append(df)

    if not all_nodes:
        raise RuntimeError("No systems produced node summaries. Check paths and columns.")

    df_nodes = pd.concat(all_nodes, ignore_index=True)

    csv_path = outdir / "node_stage_ucell_summary.csv"
    df_nodes.to_csv(csv_path, index=False)
    print(f"\n[SAVE] {csv_path}")

    plot_scatter(
        df_nodes,
        x_col="mean_stage",
        outpath=outdir / "ucell_vs_mean_stage_by_system.pdf",
        title="Mean UCell per node vs mean stage per node (colored by system)"
    )
    plot_scatter(
        df_nodes,
        x_col="median_stage",
        outpath=outdir / "ucell_vs_median_stage_by_system.pdf",
        title="Mean UCell per node vs median stage per node (colored by system)"
    )

    plot_scatter(
        df_nodes,
        x_col="mean_stage",
        outpath=outdir / "ucell_vs_mean_stage_by_system.png",
        title="Mean UCell per node vs mean stage per node (colored by system)"
    )
    plot_scatter(
        df_nodes,
        x_col="median_stage",
        outpath=outdir / "ucell_vs_median_stage_by_system.png",
        title="Mean UCell per node vs median stage per node (colored by system)"
    )

    print("[DONE]")


if __name__ == "__main__":
    main()
