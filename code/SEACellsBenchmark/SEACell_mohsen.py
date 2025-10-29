#!/usr/bin/env python3
import sys
# 1) Put our SEACells clone at the top of PYTHONPATH
sys.path.insert(0, "/project/xyang2/TOOLS/SEACells")

import os
import re
import random
import warnings
import numpy as np
import pandas as pd
import anndata as ad
import scanpy as sc
import seaborn as sns
import matplotlib.pyplot as plt
import SEACells
from scipy import sparse
import scipy.sparse as sp

# ---------------------- Reproducibility and plotting ---------------------- #
warnings.filterwarnings("ignore", category=FutureWarning)
random.seed(0)
np.random.seed(0)

sns.set_style("ticks")
plt.rcParams["figure.figsize"] = [4, 4]
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42
plt.rcParams["figure.dpi"] = 300

# ---------------------- I/O paths. Edit if needed ------------------------- #
INPUT_FILE  = "/project/imoskowitz/xyang2/SHH/Qiu_TimeLapse/results/raw_added/PNS_neurons_adata_scale.h5ad"
RESULTS_DIR = "/project/imoskowitz/xyang2/chrislowzhengxi/results/shendure_test_small"
SYSTEM_TAG  = os.path.basename(INPUT_FILE).split("_")[0]   # e.g. "Eye"
FIG_DIR     = os.path.join(RESULTS_DIR, "figures")

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)
sc.settings.figdir = FIG_DIR  # so scanpy saves into ./figures

def tag(fname: str, tag_txt: str = SYSTEM_TAG) -> str:
    base, ext = os.path.splitext(fname)
    return f"{base}_{tag_txt}{ext}"

def plot_and_save(fig_name, **save_kw):
    outname = tag(fig_name)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, outname), **save_kw)
    plt.close()

# ---------------------- Data init. Blend of both codes -------------------- #
def initialize_data(
    h5_path: str,
    csv_meta: str = "/project/imoskowitz/xyang2/chrislowzhengxi/data/df_cell_celltyp_new_merged.csv",
    min_cells_per_embryo: int = 50,
    subsample_frac: float | None = None,
    random_state: int = 0
):
    """
    1) Load AnnData.
    2) Read metadata. Build staging from day and somite_count.
    3) Map metadata into .obs.
    4) Drop tiny embryos. Optional subsample.
    5) Ensure PCA exists. Ensure neighbors and UMAP exist for plotting.
    """
    print("Loading expression:", h5_path, flush=True)
    adata = sc.read_h5ad(h5_path)
    print("AnnData shape:", adata.shape, flush=True)

    # Make var names fixable
    if adata.raw is not None and isinstance(adata.raw.var.index, pd.CategoricalIndex):
        adata.raw.var.index = adata.raw.var.index.astype(str)
    if isinstance(adata.var_names, pd.CategoricalIndex):
        adata.var_names = adata.var_names.astype(str)

    # Choose X for PCA and kernel. Prefer a normalized layer if present.
    # You can switch to counts if needed.
    if "data" in adata.layers:
        adata.X = adata.layers["data"].astype(np.float32)
    else:
        adata.X = adata.X.astype(np.float32)

    # Set adata.raw for later summarization. Prefer true counts if available.
    if "counts" in adata.layers:
        ad_raw = sc.AnnData(X=adata.layers["counts"].copy())
        ad_raw.obs_names = adata.obs_names.copy()
        ad_raw.var_names = adata.var_names.copy()
        adata.raw = ad_raw
    else:
        adata.raw = adata

    # ------------------- metadata and staging mapping ------------------- #
    req_cols = [
        "cell_id", "day", "somite_count", "embryo_id",
        "experimental_id", "system", "celltype_new", "meta_group"
    ]
    print("Reading metadata CSV …", flush=True)
    df_meta = pd.read_csv(csv_meta, usecols=req_cols)

    if "somite_count" not in df_meta.columns or "day" not in df_meta.columns:
        raise ValueError("Metadata must have somite_count and day.")

    df_meta = df_meta[df_meta["system"] == SYSTEM_TAG].copy()
    print(f"Filtered to system {SYSTEM_TAG}: {len(df_meta):,} rows", flush=True)
    print(f"AnnData cells: {adata.n_obs:,}", flush=True)

    def _map_staging(row):
        d = row["day"]
        if d in ("E8", "E8.0-E8.5", "E8.5"):
            try:
                scount = int(str(row["somite_count"]).split()[0])
            except Exception:
                return np.nan
            if 0 <= scount <= 3:
                return "E8.0"
            elif 4 <= scount <= 7:
                return "E8.25"
            elif 8 <= scount <= 11:
                return "E8.5"
            else:
                return "E8.5+"
        else:
            return d

    df_meta["staging"] = df_meta.apply(_map_staging, axis=1)
    print("\nSample staging rows:")
    print(df_meta.loc[df_meta["day"].astype(str).str.contains("E8"),
                      ["day", "somite_count", "staging"]].head(), "\n")

    # Drop duplicate cell_id in metadata
    df_meta = df_meta.drop_duplicates("cell_id", keep="first")

    # Keep only cells present in metadata
    orphan = set(adata.obs["cell_id"]) - set(df_meta["cell_id"])
    if orphan:
        print(f"{len(orphan):,} cells in AnnData lack metadata. Dropping them.", flush=True)
    adata = adata[adata.obs["cell_id"].isin(df_meta["cell_id"])].copy()

    # Map columns to adata.obs
    meta_idx = df_meta.set_index("cell_id")
    for col in [
        "day", "staging", "somite_count", "embryo_id",
        "experimental_id", "system", "meta_group", "celltype_new"
    ]:
        adata.obs[col] = adata.obs["cell_id"].map(meta_idx[col])

    print("Mapped to obs:", ["staging", "somite_count", "experimental_id"], flush=True)

    # Drop tiny embryos
    counts = adata.obs["embryo_id"].value_counts()
    keep_emb = counts[counts >= min_cells_per_embryo].index
    adata = adata[adata.obs["embryo_id"].isin(keep_emb)].copy()
    print(f"Kept {len(keep_emb)} embryos. {adata.n_obs:,} cells remain.", flush=True)

    # Optional subsample
    if subsample_frac:
        sc.pp.subsample(adata, fraction=subsample_frac, random_state=random_state, copy=False)
        print(f"Subsampled to {adata.n_obs} cells", flush=True)

    # PCA
    if "X_pca" in adata.obsm:
        adata.obsm["X_pca"] = adata.obsm["X_pca"][:, :20]
    else:
        sc.tl.pca(adata, n_comps=20, random_state=random_state)

    # Neighbors and UMAP for plotting
    sc.pp.neighbors(adata, use_rep="X_pca", n_neighbors=15)
    sc.tl.umap(adata)

    return adata

# ---------------------- SEACells core. Mohsen style ----------------------- #
def run_seacells(adata):
    print(f"Starting SEACells on {adata.n_obs} cells", flush=True)

    # Mohsen used about 1 metacell per 100 cells for lower density.
    # Tutorial suggests 1 per 75 cells. You can change this later.
    n_SEACells = max(10, adata.n_obs // 100)

    model = SEACells.core.SEACells(
        adata,
        build_kernel_on="X_pca",
        n_SEACells=n_SEACells,
        n_waypoint_eigs=10,
        convergence_epsilon=1e-5,
        use_sparse=True
    )

    model.construct_kernel_matrix()
    print("Kernel built", flush=True)

    # Light clustermap for a small block
    try:
        g = sns.clustermap(model.kernel_matrix[:200, :200].toarray(), cmap="viridis")
        g.savefig(os.path.join(FIG_DIR, tag("kernel_clustermap.pdf")), dpi=300)
        plt.close(g.fig)
    except Exception as e:
        print("Skip clustermap:", e, flush=True)

    model.initialize_archetypes()
    print("Archetypes initialized", flush=True)

    SEACells.plot.plot_initialization(
        adata, model,
        save_as=os.path.join(FIG_DIR, tag("init_umap.pdf"))
    )

    model.fit(min_iter=10, max_iter=50)
    for _ in range(5):
        model.step()

    # Ensure A_ is dense for downstream steps
    if hasattr(model, "A_") and sparse.issparse(model.A_):
        model.A_ = model.A_.toarray()

    model.plot_convergence(save_as=os.path.join(FIG_DIR, tag("convergence.pdf")))
    print("SEACells converged", flush=True)
    return model

# ---------------------- Diagnostics and metrics --------------------------- #
def summarize_and_evaluate(adata, model):
    # 1) Non trivial assignment histogram
    plt.figure(figsize=(3, 2))
    sns.histplot((model.A_.T > 0.1).sum(axis=1), bins=30)
    plt.title("Non-trivial (>0.1) assignments per cell")
    plt.xlabel("# Non-trivial SEACell Assignments")
    plt.ylabel("# Cells")
    plot_and_save("nontrivial_assignments_hist.pdf")

    # 2) Top 5 strongest assignments
    plt.figure(figsize=(3, 2))
    b = np.partition(model.A_.T, -5, axis=1)
    sns.heatmap(np.sort(b[:, -5:], axis=1)[:, ::-1], cmap="viridis", vmin=0)
    plt.title("Top 5 strongest assignments")
    plt.xlabel("$n^{th}$ strongest assignment")
    plot_and_save("top5_assignment_heatmap.pdf")

    # 3) UMAP cells only
    SEACells.plot.plot_2D(
        adata, key="X_umap", colour_metacells=False,
        save_as=os.path.join(FIG_DIR, tag("umap_cells.pdf"))
    )
    # 4) UMAP with metacells
    SEACells.plot.plot_2D(
        adata, key="X_umap", colour_metacells=True,
        save_as=os.path.join(FIG_DIR, tag("umap_metacells.pdf"))
    )
    # 5) SEACell size distribution
    SEACells.plot.plot_SEACell_sizes(
        adata, bins=5,
        save_as=os.path.join(FIG_DIR, tag("seacell_sizes.pdf"))
    )

    # Purity, compactness, separation
    metrics = [
        SEACells.evaluate.compute_celltype_purity,
        SEACells.evaluate.compactness,
        SEACells.evaluate.separation
    ]
    for fn in metrics:
        if fn is SEACells.evaluate.compute_celltype_purity:
            label_key = "celltype_new"
            if label_key not in adata.obs:
                print("Skip purity. obs['celltype_new'] not found.", flush=True)
                continue
            df = fn(adata, label_key)
            col = f"{label_key}_purity"
        elif fn is SEACells.evaluate.compactness:
            df = fn(adata, "X_pca")
            col = "compactness"
        else:
            df = fn(adata, "X_pca", nth_nbr=1)
            col = "separation"

        plt.figure(figsize=(4, 4))
        sns.boxplot(data=df, y=col)
        plt.title(col.capitalize())
        sns.despine()
        plot_and_save(f"{col}.pdf")

# ---------------------- Composition and exports --------------------------- #
def write_mappings(adata, model):
    df = adata.obs[["celltype_new"]].copy()
    df = df.rename(columns={"celltype_new": "orig_cluster"})
    df["metacell"] = model.get_hard_assignments()
    out_csv = os.path.join(RESULTS_DIR, tag("cell_to_metacell_map.csv"))
    df.to_csv(out_csv, index=True)
    print("Wrote cell to metacell map:", out_csv, flush=True)

def write_metacell_composition(adata, results_dir, cluster_key="celltype_new"):
    """
    Count how many cells of each original cluster fall into each SEACell.
    """
    df = pd.DataFrame({
        "metacell_id": adata.obs["SEACell"].values,
        "orig_cluster": adata.obs[cluster_key].values
    })
    comp_counts = (
        df.groupby("metacell_id")["orig_cluster"]
          .value_counts()
          .unstack(fill_value=0)
          .sort_index()
    )
    out_csv = os.path.join(results_dir, tag("metacell_composition_counts.csv"))
    comp_counts.to_csv(out_csv)
    print("Wrote metacell composition table:", out_csv, flush=True)

def aggregate_metacells_by_timepoint(adata, results_dir, time_key="day"):
    if time_key not in adata.obs:
        raise KeyError(f"{time_key} not found in adata.obs")

    df = adata.obs[[time_key, "SEACell"]].copy()
    df["group"] = df[time_key].astype(str) + "_mc" + df["SEACell"].astype(str)

    X = adata.X.toarray() if sp.issparse(adata.X) else adata.X
    expr_df = pd.DataFrame(X, index=adata.obs_names, columns=adata.var_names)
    expr_df["group"] = df["group"].values
    pseudobulk = expr_df.groupby("group").mean()

    new_obs = pd.DataFrame(index=pseudobulk.index)
    parts = new_obs.index.to_series().str.split("_mc", expand=True)
    new_obs[time_key] = parts[0]
    new_obs["SEACell"] = parts[1].astype(int)

    new_adata = ad.AnnData(X=pseudobulk.values, obs=new_obs, var=adata.var)
    out_file = os.path.join(results_dir, tag(f"metacell_pseudobulk_by_{time_key}.h5ad"))
    new_adata.write(out_file)
    print("Wrote aggregated metacell AnnData:", out_file, flush=True)

# ---------------------- Optional SEACell summaries ------------------------ #
def write_seacell_summaries(adata, model):
    """
    Write hard and soft SEACell summaries. Uses adata.raw genes.
    """
    try:
        se_ad = SEACells.core.summarize_by_SEACell(
            adata,
            SEACells_label="SEACell",
            summarize_layer="raw",
            ad_raw_var_names=True
        )
        se_ad.write(os.path.join(RESULTS_DIR, tag("SEACell_summary.h5ad")))
        print("Wrote SEACell hard summary", flush=True)
    except Exception as e:
        print("Skip hard summary:", e, flush=True)

    try:
        se_soft = SEACells.core.summarize_by_soft_SEACell(
            adata,
            model.A_,
            celltype_label="celltype_new",
            summarize_layer="raw",
            minimum_weight=0.05,
            ad_raw_var_names=True
        )
        se_soft.write(os.path.join(RESULTS_DIR, tag("SEACell_soft_summary.h5ad")))
        print("Wrote SEACell soft summary", flush=True)
    except Exception as e:
        print("Skip soft summary:", e, flush=True)

# ---------------------- Main ------------------------------------------------ #
def main():
    assert os.path.exists(INPUT_FILE), f"{INPUT_FILE} not found"
    meta_csv = "/project/imoskowitz/xyang2/chrislowzhengxi/data/df_cell_celltyp_new_merged.csv"
    assert os.path.exists(meta_csv), "Metadata CSV not found"

    adata = initialize_data(INPUT_FILE)
    model = run_seacells(adata)

    # Hard assignments into obs
    adata.obs["SEACell"] = model.get_hard_assignments()

    # AnnData write out
    adata.var.index.name = "var_index"
    if adata.raw is not None and hasattr(adata.raw, "var"):
        adata.raw.var.index.name = "var_index"

    out_h5ad = os.path.join(RESULTS_DIR, tag("with_SEACells_full.h5ad"))
    adata.write(out_h5ad)
    print("Wrote AnnData with SEACell labels:", out_h5ad, flush=True)

    # Plots and metrics
    summarize_and_evaluate(adata, model)

    # Mappings and compositions
    write_mappings(adata, model)
    write_metacell_composition(adata, RESULTS_DIR)

    # Aggregate by time
    aggregate_metacells_by_timepoint(adata, RESULTS_DIR, time_key="day")

    # Optional SEACell summaries
    write_seacell_summaries(adata, model)

    print("All outputs generated", flush=True)

if __name__ == "__main__":
    main()
