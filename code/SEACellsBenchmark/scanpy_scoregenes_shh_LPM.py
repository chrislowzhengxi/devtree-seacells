#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Scanpy score_genes scoring for SHH genes (Gli1, Ptch1, Hhip).

- Uses scanpy.tl.score_genes with ctrl_as_ref=False, n_bins=50.
- Input:   Lateral_plate_mesoderm_adata_scale.h5ad
- Output root: /project/imoskowitz/xyang2/chrislowzhengxi/results/score_genes
- Outputs:
    - h5ad with .obs["SHH_scoregenes"]
    - per cell CSV
    - per node summary CSV
    - violin plot per celltype_new
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse
import matplotlib.pyplot as plt

# -------------------- CONFIG --------------------

H5_ROOT = Path("/project/imoskowitz/xyang2/SHH/Qiu_TimeLapse/results/raw_added")

OUT_ROOT = Path("/project/imoskowitz/xyang2/chrislowzhengxi/results/score_genes")

SYSTEMS = ["Lateral_plate_mesoderm"]

SHH_GENES = ["Gli1", "Ptch1", "Hhip"]

LOG_LAYER_NAME = "log1p_cpm"   # will be created if missing

SCORE_NAME = "SHH_scoregenes"

# ------------------------------------------------


def ensure_log1p_cpm_layer(adata, layer_name="log1p_cpm", target_sum=1_000_000):
    """
    Create a log1p CPM layer from adata.raw, without densifying if the
    input is sparse.

    - Uses adata.raw.X as counts.
    - Normalizes so each cell sums to target_sum.
    - Applies log1p in place on that layer.
    """
    if adata.raw is None:
        raise ValueError("adata.raw is None. Need raw counts to build log-normalized layer.")

    n_obs, n_vars = adata.n_obs, adata.n_vars
    make_sparse = sparse.issparse(adata.X)

    # start with empty container
    if make_sparse:
        layer = sparse.csr_matrix((n_obs, n_vars), dtype=np.float32)
    else:
        layer = np.zeros((n_obs, n_vars), dtype=np.float32)

    raw = adata.raw.to_adata()

    # align genes between main and raw
    common = adata.var_names.intersection(raw.var_names)
    if len(common) == 0:
        raise ValueError("No overlap between adata.var_names and adata.raw.var_names.")

    var_idx = adata.var_names.get_indexer(common)
    raw_idx = raw.var_names.get_indexer(common)

    if sparse.issparse(layer):
        L = layer.tolil()
        sub = raw.X[:, raw_idx]
        if not sparse.issparse(sub):
            sub = sparse.csr_matrix(sub)
        L[:, var_idx] = sub
        layer = L.tocsr()
    else:
        sub = raw.X[:, raw_idx]
        if sparse.issparse(sub):
            sub = sub.toarray()
        layer[:, var_idx] = np.asarray(sub, dtype=np.float32)

    adata.layers[layer_name] = layer

    # normalize and log1p in place on this layer
    sc.pp.normalize_total(adata, target_sum=target_sum, layer=layer_name)
    sc.pp.log1p(adata, layer=layer_name)

    return layer_name


def run_score_genes_shh(adata, layer_name, score_name=SCORE_NAME):
    """
    Run scanpy.tl.score_genes on SHH_GENES using the specified layer.
    """

    # sanity check that requested genes exist
    present = [g for g in SHH_GENES if g in adata.var_names]
    print(f"Requested SHH genes: {SHH_GENES}")
    print(f"Present in var_names: {present}")

    if len(present) == 0:
        raise ValueError("None of Gli1, Ptch1, Hhip found in adata.var_names.")

    # run score_genes
    sc.tl.score_genes(
        adata,
        gene_list=SHH_GENES,
        ctrl_as_ref=False,
        ctrl_size=50,
        gene_pool=None,
        n_bins=50,
        score_name=score_name,
        random_state=0,
        use_raw=False,
        layer=layer_name,
        copy=False,
    )

    print(f"[SCORE_GENES] Wrote scores to adata.obs['{score_name}'].")
    print(adata.obs[score_name].describe())

    return adata.obs[score_name]


def save_per_cell_outputs(adata, out_dir, system_tag, score_name=SCORE_NAME):
    """
    Save scored AnnData and a per cell table with the SHH score.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    h5_out = out_dir / f"{system_tag}_adata_with_{score_name}.h5ad"
    csv_out = out_dir / f"{system_tag}_{score_name}_percell.csv"

    # keep full object for now
    adata.write_h5ad(str(h5_out))
    adata.obs[[score_name]].to_csv(csv_out, index=True)

    print(f"[SAVE] {h5_out}")
    print(f"[SAVE] {csv_out}")


def make_summary_by_celltype(adata, system_tag, out_root, score_name=SCORE_NAME):
    """
    Compute per celltype_new summary for the SHH_scoregenes column.
    """
    out_dir = out_root / system_tag / "qc"
    out_dir.mkdir(parents=True, exist_ok=True)

    if "celltype_new" not in adata.obs.columns:
        raise KeyError("adata.obs does not contain 'celltype_new'.")

    df = adata.obs[["celltype_new", score_name]].copy()

    counts = df["celltype_new"].value_counts().sort_index()
    grp = df.groupby("celltype_new", observed=False)[score_name]

    summary = pd.DataFrame({
        "celltype_new": counts.index,
        "n_cells": counts.values,
        "median": grp.median().reindex(counts.index).values,
        "mean": grp.mean().reindex(counts.index).values,
        "q90": grp.quantile(0.90).reindex(counts.index).values,
        "frac>0": grp.apply(lambda s: (s > 0).mean()).reindex(counts.index).values,
        "variance": grp.var(ddof=1).reindex(counts.index).values,
        "std": grp.std(ddof=1).reindex(counts.index).values,
    })

    csv_path = out_dir / f"{system_tag}_{score_name}_summary.csv"
    summary.to_csv(csv_path, index=False)
    print(f"[QC] wrote summary: {csv_path}")

    return summary


def plot_violins_by_celltype(adata, system_tag, out_root, score_name=SCORE_NAME):
    """
    Violin plots of SHH_scoregenes per celltype_new.
    """
    out_dir = out_root / system_tag / "qc"
    out_dir.mkdir(parents=True, exist_ok=True)

    df = adata.obs[["celltype_new", score_name]].copy()
    labels = list(pd.unique(df["celltype_new"]))

    data = [df.loc[df["celltype_new"] == lab, score_name].to_numpy()
            for lab in labels]
    ns = [len(arr) for arr in data]

    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.violinplot(data, showmeans=False, showmedians=False, showextrema=False)

    medians = [np.median(arr) if len(arr) else np.nan for arr in data]
    q90s = [np.quantile(arr, 0.90) if len(arr) else np.nan for arr in data]
    x = np.arange(1, len(labels) + 1)

    ax.plot(x, medians, "o", label="Median")
    ax.plot(x, q90s, "^", label="90th pct")

    for xi, n in zip(x, ns):
        if len(data[xi - 1]):
            ytop = np.nanmax(data[xi - 1])
            ax.text(xi, ytop + 0.02, f"n={n}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel(score_name)
    ax.set_title(f"{system_tag} – {score_name} per node")

    max_val = max((np.nanmax(arr) if len(arr) else 0.0) for arr in data)
    ax.set_ylim(0, float(max_val) + 0.1)

    ax.legend(frameon=False, ncol=2)

    png = out_dir / f"{system_tag}_{score_name}_violins.png"
    pdf = out_dir / f"{system_tag}_{score_name}_violins.pdf"
    fig.savefig(png, dpi=300)
    fig.savefig(pdf, dpi=300)
    plt.close(fig)

    print(f"[PLOT] wrote {png}")
    print(f"[PLOT] wrote {pdf}")


def main():
    print("== Scanpy score_genes SHH scoring ==")
    print(f"h5-root   : {H5_ROOT}")
    print(f"out-root  : {OUT_ROOT}")
    print(f"systems   : {SYSTEMS}")
    print(f"score name: {SCORE_NAME}")
    print(f"log layer : {LOG_LAYER_NAME}")

    for system_tag in SYSTEMS:
        print("\n" + "=" * 60)
        print(f"[SYSTEM] {system_tag}")

        raw_h5 = H5_ROOT / f"{system_tag}_adata_scale.h5ad"
        if not raw_h5.exists():
            print(f"[SKIP] Missing input h5ad: {raw_h5}")
            continue

        print(f"[LOAD] {raw_h5}")
        adata = sc.read_h5ad(str(raw_h5))

        # Optional: add system column if not present
        if "system" not in adata.obs.columns:
            adata.obs["system"] = system_tag

        # Create log-normalized layer if it is not already there
        if LOG_LAYER_NAME not in adata.layers:
            print(f"[LAYER] Creating {LOG_LAYER_NAME} from adata.raw.")
            ensure_log1p_cpm_layer(adata, layer_name=LOG_LAYER_NAME, target_sum=1_000_000)
        else:
            print(f"[LAYER] Found existing layer '{LOG_LAYER_NAME}'.")

        # Run score_genes on SHH genes
        run_score_genes_shh(adata, layer_name=LOG_LAYER_NAME, score_name=SCORE_NAME)

        # Save per cell outputs
        out_dir = OUT_ROOT / system_tag
        save_per_cell_outputs(adata, out_dir, system_tag, score_name=SCORE_NAME)

        # Per node summary and violins
        try:
            make_summary_by_celltype(adata, system_tag, OUT_ROOT, score_name=SCORE_NAME)
            plot_violins_by_celltype(adata, system_tag, OUT_ROOT, score_name=SCORE_NAME)
        except Exception as e:
            print(f"[WARN] Summary or violin plot failed: {e}")

    print("\nDone.")


if __name__ == "__main__":
    main()
