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

# SYSTEMS = ["Mesoderm"]   # R 42703522 bigmem
SYSTEMS = ["Neurons"]  # R 42703535 amd-hm

# SYSTEMS = ["Blood", "Notochord"]   # Done bigmem 42702803
# SYSTEMS = ["Eye", "PNS_glia", "PNS_neurons", "Renal"]   # Done 42701709 bigmem
# SYSTEMS = ["Endothelium"]  # Done 42701714 caslake 
# SYSTEMS = ["Gut"] # Done  42701721 amd
# SYSTEMS = ["Epithelial_cells"]  # Done 42702528 caslake 
# SYSTEMS = ["Neuroectoderm"]  # Done bigmem 42702841
# SYSTEMS = ["Other_Brain_spinal_cord"]   # Done amd-hm 42702828


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
    If none of the genes are present, fill scores with 0 and continue.
    """
    varnames = pd.Index(adata.var_names.astype(str))

    present = [g for g in SHH_GENES if g in varnames]
    print(f"Requested SHH genes: {SHH_GENES}")
    print(f"Present in var_names: {present}")

    if len(present) == 0:
        print(f"[WARN] None of {SHH_GENES} found in adata.var_names for this system.")
        print("[WARN] Filling adata.obs['{score_name}'] with zeros and skipping score_genes.")
        adata.obs[score_name] = 0.0
        return adata.obs[score_name]

    if layer_name not in adata.layers:
        raise KeyError(f"Layer '{layer_name}' not found in adata.layers")

    print(f"[SCORE_GENES] Using layer '{layer_name}' as adata.X for scoring.")
    adata.X = adata.layers[layer_name].copy()

    sc.tl.score_genes(
        adata,
        gene_list=present,
        ctrl_as_ref=False,
        ctrl_size=50,
        gene_pool=None,
        n_bins=50,
        score_name=score_name,
        random_state=0,
        use_raw=False,
        copy=False,
    )

    print(f"[SCORE_GENES] Wrote scores to adata.obs['{score_name}'].")
    print(adata.obs[score_name].describe())
    return adata.obs[score_name]


# def run_score_genes_shh(adata, layer_name, score_name=SCORE_NAME):
#     """
#     Run scanpy.tl.score_genes on SHH_GENES using the specified layer.
#     For older Scanpy (no 'layer' arg), we temporarily set adata.X to that layer.
#     """

#     # sanity check that requested genes exist
#     present = [g for g in SHH_GENES if g in adata.var_names]
#     print(f"Requested SHH genes: {SHH_GENES}")
#     print(f"Present in var_names: {present}")

#     if len(present) == 0:
#         raise ValueError("None of Gli1, Ptch1, Hhip found in adata.var_names.")

#     # make sure the layer exists
#     if layer_name not in adata.layers:
#         raise KeyError(f"Layer '{layer_name}' not found in adata.layers")

#     # Temporarily point X to the chosen layer
#     print(f"[SCORE_GENES] Using layer '{layer_name}' as adata.X for scoring.")
#     adata.X = adata.layers[layer_name].copy()

#     sc.tl.score_genes(
#         adata,
#         gene_list=SHH_GENES,      # keep all 3; Scanpy will ignore missing ones
#         ctrl_as_ref=False,
#         ctrl_size=50,
#         gene_pool=None,
#         n_bins=50,
#         score_name=score_name,
#         random_state=0,
#         use_raw=False,
#         copy=False,
#     )

#     print(f"[SCORE_GENES] Wrote scores to adata.obs['{score_name}'].")
#     print(adata.obs[score_name].describe())

#     return adata.obs[score_name]


def integrate_shh_scoregenes_with_edges_and_plot(system_tag: str, out_root: Path):
    """
    Use SHH_scoregenes per-node summary to build an edge table like
    Lateral_plate_mesoderm_edge_filtered_with_shh.csv, but using Scanpy
    score_genes instead of UCell.

    Outputs:
      - <system>_edge_filtered_with_shh_scoregenes.csv
      - <system>_edge_filtered_with_shh_scoregenes.txt
      - optional graph plot if plot_shh_graph is available.
    """
    import numpy as np

    # 1) Per-node SHH_scoregenes summary
    score_csv = out_root / system_tag / "qc" / f"{system_tag}_{SCORE_NAME}_summary.csv"
    if not score_csv.exists():
        print(f"[EDGE] Missing score summary: {score_csv}")
        return

    scores = pd.read_csv(score_csv)

    # Ensure variance/std exist
    if "variance" not in scores.columns:
        scores["variance"] = np.nan
    if "std" not in scores.columns:
        scores["std"] = np.sqrt(scores["variance"])

    # Standardize column names for merging
    scores = scores.rename(columns={
        "celltype_new": "node_name",
        "mean": "sh_score",
        "n_cells": "n"
    })

    # 2) Edges from Holly's file
    # edge_file = Path("/project/imoskowitz/xyang2/SHH/Qiu_TimeLapse/Holly_desktop/edges_filtered.txt")
    edge_file = Path("/project/imoskowitz/xyang2/SHH/Qiu_TimeLapse/results/tree/edges_filtered.txt")
    if not edge_file.exists():
        print(f"[EDGE] Missing edge file: {edge_file}")
        return

    edges = pd.read_csv(edge_file, sep="\t")
    edges = edges.loc[edges["system"] == system_tag].copy()

    # 3) Special-case: ensure SHF -> Atrial CM exists for LPM
    if system_tag != "Lateral_plate_mesoderm":
        edges = edges[~((edges["x_name"] == "Second heart field") &
                        (edges["y_name"] == "Atrial cardiomyocytes"))]

    if system_tag == "Lateral_plate_mesoderm":
        need_manual = ~((edges["x_name"] == "Second heart field") &
                        (edges["y_name"] == "Atrial cardiomyocytes")).any()
        if need_manual:
            new_row = {
                "system": system_tag,
                "x": "L_M22",
                "y": "L_M5",
                "x_name": "Second heart field",
                "y_name": "Atrial cardiomyocytes",
                "edge_type": "Developmental progression",
                "x_number": np.nan, "y_number": np.nan,
                "x_id": np.nan, "y_id": np.nan,
            }
            edges = pd.concat([edges, pd.DataFrame([new_row])], ignore_index=True)

    # 4) Merge SHH_scoregenes means onto x and y nodes
    edges = edges.merge(
        scores.rename(columns={
            "node_name": "x_name",
            "sh_score": "sh_x",
            "variance": "variance_x",
            "std": "std_x",
            "n": "n_x"
        }),
        on="x_name", how="left"
    )
    edges = edges.merge(
        scores.rename(columns={
            "node_name": "y_name",
            "sh_score": "sh_y",
            "variance": "variance_y",
            "std": "std_y",
            "n": "n_y"
        }),
        on="y_name", how="left"
    )

    # 5) Deltas and effect sizes
    edges["abs_delta"] = (edges["sh_x"] - edges["sh_y"]).abs()
    edges["delta"] = edges["sh_y"] - edges["sh_x"]

    pooled_std = np.sqrt((edges["variance_x"] + edges["variance_y"]) / 2.0)
    edges["cohens_d"] = edges["delta"] / pooled_std.replace(0, np.nan)
    edges["cohens_d"] = edges["cohens_d"].round(4)

    # Percent change relative to source node (sh_x)
    eps = 1e-9
    denom = edges["sh_x"].copy()
    denom = denom.where(denom.abs() > eps, np.nan)
    edges["pct_change"] = 100.0 * edges["delta"] / denom
    edges["abs_pct_change"] = edges["pct_change"].abs()

    edges["pct_change"] = edges["pct_change"].round(2)
    edges["abs_pct_change"] = edges["abs_pct_change"].round(2)
    edges["delta"] = edges["delta"].round(6)
    edges["abs_delta"] = edges["abs_delta"].round(6)

    # Sort by absolute delta for convenience
    edges_sorted = edges.sort_values("abs_delta", ascending=False)

    outdir = out_root / system_tag
    outdir.mkdir(parents=True, exist_ok=True)

    out_csv = outdir / f"{system_tag}_edge_filtered_with_shh_scoregenes.csv"
    out_txt = outdir / f"{system_tag}_edge_filtered_with_shh_scoregenes.txt"
    edges_sorted.to_csv(out_csv, index=False)
    edges_sorted.to_csv(out_txt, sep="\t", index=False, na_rep="")
    print(f"[EDGE] wrote (score_genes) edges: {out_csv}")
    print(f"[EDGE] wrote (score_genes txt): {out_txt}")

    # Optional graph plot if you copy plot_shh_graph() into this file
    if "plot_shh_graph" in globals():
        scores_for_plot = scores.rename(columns={"node_name": "celltype_new"})[
            ["celltype_new", "sh_score"]
        ]
        plot_shh_graph(
            edges_sorted, system_tag, outdir, scores_for_plot,
            file_stem=f"{system_tag}_shh_scoregenes_graph"
        )


def save_per_cell_outputs(adata, out_dir, system_tag, score_name=SCORE_NAME):
    out_dir.mkdir(parents=True, exist_ok=True)

    h5_out = out_dir / f"{system_tag}_adata_with_{score_name}.h5ad"
    csv_out = out_dir / f"{system_tag}_{score_name}_percell.csv"

    # --- Fix adata.var (main) index/column collision ---
    adata.var_names = adata.var_names.astype(str)
    if hasattr(adata, "var_names_make_unique"):
        adata.var_names_make_unique()

    iname = adata.var.index.name
    if iname and (iname in adata.var.columns):
        same = (
            adata.var.index.astype(str).to_series().values
            == adata.var[iname].astype(str).values
        )
        if not bool(np.all(same)):
            adata.var.rename(columns={iname: f"{iname}_col"}, inplace=True)
    adata.var.index.name = None

    # --- Fix adata.raw.var index/column collision ---
    if adata.raw is not None:
        raw = adata.raw.to_adata()
        raw.var_names = raw.var_names.astype(str)
        if hasattr(raw, "var_names_make_unique"):
            raw.var_names_make_unique()

        riname = raw.var.index.name
        if riname and (riname in raw.var.columns):
            same = (
                raw.var.index.astype(str).to_series().values
                == raw.var[riname].astype(str).values
            )
            if not bool(np.all(same)):
                raw.var.rename(columns={riname: f"{riname}_col"}, inplace=True)
        raw.var.index.name = None
        adata.raw = raw

    # now safe to write
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


SUBSAMPLE_FRAC = 1.0

def main():
    print("== Scanpy score_genes SHH scoring ==")
    print(f"h5-root   : {H5_ROOT}")
    print(f"out-root  : {OUT_ROOT}")
    print(f"systems   : {SYSTEMS}")
    print(f"score name: {SCORE_NAME}")
    print(f"log layer : {LOG_LAYER_NAME}")
    print(f"subsample : {SUBSAMPLE_FRAC}")

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
        
        
        # Deal with this: pandas.errors.InvalidIndexError: Reindexing only valid with uniquely valued Index objects
        adata.var_names = adata.var_names.astype(str)
        if hasattr(adata, "var_names_make_unique"):
            adata.var_names_make_unique()
        if adata.raw is not None:
            raw = adata.raw.to_adata()
            raw.var_names = raw.var_names.astype(str)
            if hasattr(raw, "var_names_make_unique"):
                raw.var_names_make_unique()
            adata.raw = raw
        
        if SUBSAMPLE_FRAC is not None and SUBSAMPLE_FRAC < 1.0:
            n = adata.n_obs
            k = max(1, int(round(n * float(SUBSAMPLE_FRAC))))
            rng = np.random.default_rng(0)
            sel = rng.choice(n, size=k, replace=False)
            adata = adata[adata.obs_names[sel]].copy()
            print(f"[SUBSAMPLE] {n} → {adata.n_obs} cells (fraction={SUBSAMPLE_FRAC})")

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
            integrate_shh_scoregenes_with_edges_and_plot(system_tag, OUT_ROOT)
        except Exception as e:
            print(f"[WARN] Summary or violin plot failed: {e}")

    print("\nDone.")


if __name__ == "__main__":
    main()
