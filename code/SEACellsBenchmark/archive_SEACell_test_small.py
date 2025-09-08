#!/usr/bin/env python3
import sys
# ensure we load the clone under /project/xyang2/TOOLS/SEACells
# 1) point to your SEACells install
sys.path.insert(0, "/project/xyang2/TOOLS/SEACells")

import os, random, numpy as np

# sys.path.append('/project/xyang2/TOOLS')
import SEACells

# 2) standard imports
import scanpy as sc
import anndata as ad
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

# 3) reproducibility & plotting
random.seed(0)
np.random.seed(0)
sns.set_style('ticks')
matplotlib.rcParams['figure.figsize'] = [4, 4]
matplotlib.rcParams['figure.dpi'] = 100

# 4) paths
INPUT_FILE = '/project/imoskowitz/xyang2/SHH/Qiu_TimeLapse/results/raw_added/Eye_adata_scale.h5ad'
RESULTS_DIR = '/project/imoskowitz/xyang2/chrislowzhengxi/results/shendure_test_small'
FIG_DIR     = os.path.join(RESULTS_DIR, 'figures')
os.makedirs(FIG_DIR, exist_ok=True)

def plot_and_save(fig, name):
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, name), dpi=150)
    plt.close(fig)


def initialize_data(path):
    adata = sc.read_h5ad(path)
    adata.var.index.name = None

    # for quick‐test: subsample 5%
    sc.pp.subsample(adata, fraction=0.05, random_state=0)

    # stash raw counts _and_ layer
    adata.raw = adata.copy()  
    adata.layers['raw'] = adata.raw.X
    # stash raw counts as a sparse matrix layer
    from scipy import sparse
    adata.raw = adata.copy()
    adata.layers['raw'] = sparse.csr_matrix(adata.raw.X)

    return adata


# Archived code for initializing data 
# def initialize_data(path):
#     adata = sc.read_h5ad(path)

#     ######
#     # # make sure gene names are unique
#     # adata.var_names_make_unique()
#     adata.var_names = adata.var_names.astype(str)
#     adata.var_names_make_unique()

#     # (optional) basic QC & normalization
#     # filter genes in <3 cells
#     sc.pp.filter_genes(adata, min_cells=3)
#     # normalize total counts per cell then log-transform
#     sc.pp.normalize_total(adata, target_sum=1e4)
#     sc.pp.log1p(adata)
#     # stash raw for later summarization
#     adata.raw = adata.copy()       # DOES SeaCell use adata.raw? ### 
#     ###### 

#     print("Loaded data from", path, flush=True)
#     # subsample 5% of cells
#     sc.pp.subsample(adata, fraction=0.05, random_state=0)
#     print(f"Subsampled to {adata.n_obs} cells", flush=True)
    
#     # # HVG Selection (Neccsary?)
#     # sc.pp.highly_variable_genes(
#     #     adata,
#     #     n_top_genes=3000,
#     #     flavor="seurat",
#     #     n_bins=50,
#     #     subset=True
#     # )
#     # print(f"Kept {adata.n_vars} HVGs", flush=True)
#     ######## 


#     # # use normalized layer
#     # adata.X = adata.layers['data'].astype(np.float32)

#     # use the main X (now normalized & log1p’d)
#     adata.X = adata.X.astype(np.float32)


#     # replace any NaN or Inf in adata.X with 0.0
#     adata.X = np.nan_to_num(adata.X, nan=0.0, posinf=0.0, neginf=0.0)

#     sc.pp.pca(adata, n_comps=30)
#     print("PCA done (30 comps)", flush=True)
#     # stash raw for summarization
#     raw_ad = ad.AnnData(adata.X)
#     raw_ad.obs_names, raw_ad.var_names = adata.obs_names, adata.var_names
#     adata.raw = raw_ad
#     return adata


def run_seacells(adata):
    print("Starting SEACells on", adata.n_obs, "cells", flush=True)
    n_cells = adata.n_obs
    n_SEACells = max(10, n_cells // 200)
    model = SEACells.core.SEACells(
        adata,
        build_kernel_on='X_pca',
        n_SEACells=n_SEACells,
        n_waypoint_eigs=10,
        convergence_epsilon=1e-5
    )
    model.construct_kernel_matrix()
    print("Kernel matrix constructed", flush=True)
    # kernel heatmap (first 200×200 for speed)
    g = sns.clustermap(model.kernel_matrix.toarray()[:200, :200], cmap='viridis')
    g.savefig(os.path.join(FIG_DIR, 'kernel_matrix_clustermap.png'), dpi=150)
    plt.close('all')

    model.initialize_archetypes()
    print("Archetypes initialized", flush=True)

    SEACells.plot.plot_initialization(adata, model,
        save_as=os.path.join(FIG_DIR, 'initialization_umap.png'))
    model.fit(min_iter=10, max_iter=50)
    for _ in range(5):
        model.step()
    print("SEACells fitted (convergence reached)", flush=True)


    model.plot_convergence(save_as=os.path.join(FIG_DIR, 'rss_convergence.png'))
    print("Convergence plot saved", flush=True)

    return model

def summarize_and_evaluate(adata, model):
    # non-trivial assignments
    fig = plt.figure()
    sns.histplot((model.A_.T > 0.1).sum(axis=1), bins=30)
    fig.savefig(os.path.join(FIG_DIR, 'nontrivial_assignments_hist.png'))
    plt.close(fig)
    # top 5 assignment heatmap
    fig = plt.figure()
    b = np.partition(model.A_.T, -5, axis=1)
    sns.heatmap(np.sort(b[:, -5:], axis=1)[:, ::-1], cmap='viridis', vmin=0)
    fig.savefig(os.path.join(FIG_DIR, 'top5_assignment_heatmap.png'))
    plt.close(fig)
    # UMAPs & sizes
    SEACells.plot.plot_2D(adata, key='X_umap', colour_metacells=False,
        save_as=os.path.join(FIG_DIR, 'umap_cells.png'))
    SEACells.plot.plot_2D(adata, key='X_umap', colour_metacells=True,
        save_as=os.path.join(FIG_DIR, 'umap_metacells.png'))
    SEACells.plot.plot_SEACell_sizes(adata, bins=5,
        save_as=os.path.join(FIG_DIR, 'seacell_sizes.png'))
    # purity, compactness, separation - USE celltype_new instead of leiden_0.5?
    purity = SEACells.evaluate.compute_celltype_purity(adata, 'celltype_new')
    fig = plt.figure()
    sns.boxplot(data=purity, y='celltype_new_purity')
    fig.savefig(os.path.join(FIG_DIR, 'celltype_new_purity.png'))
    plt.close(fig)

    compactness = SEACells.evaluate.compactness(adata, 'X_pca')
    fig = plt.figure()
    sns.boxplot(data=compactness, y='compactness')
    fig.savefig(os.path.join(FIG_DIR, 'compactness.png'))
    plt.close(fig)

    separation = SEACells.evaluate.separation(adata, 'X_pca', nth_nbr=1)
    fig = plt.figure()
    sns.boxplot(data=separation, y='separation')
    fig.savefig(os.path.join(FIG_DIR, 'separation.png'))
    plt.close(fig)

def main():
    adata = initialize_data(INPUT_FILE)
    model = run_seacells(adata)
    # write hard assignments
    adata.obs['SEACell'] = model.get_hard_assignments()

    # clear var‐index name to avoid H5AD write error
    adata.var.index.name = None

    adata.write(os.path.join(RESULTS_DIR, 'output_with_SEACells.h5ad'))
    print("Wrote hard-assignment AnnData", flush=True)

    summarize_and_evaluate(adata, model)
    print("Evaluation plots generated", flush=True)

    # write summaries
    seac_ad = SEACells.core.summarize_by_SEACell(
        adata, SEACells_label='SEACell', summarize_layer='raw')
    seac_ad.write(os.path.join(RESULTS_DIR, 'SEACell_summary.h5ad'))
    print("Wrote SEACell_summary.h5ad", flush=True)

    soft_ad = SEACells.core.summarize_by_soft_SEACell(
        adata, model.A_, celltype_label='celltype_new',
        summarize_layer='raw', minimum_weight=0.05)
    soft_ad.write(os.path.join(RESULTS_DIR, 'SEACell_soft_summary.h5ad'))

    # now compute purity on your real labels
    purity = SEACells.evaluate.compute_celltype_purity(adata, 'celltype_new')
    print("Computed purity on 'celltype_new'", flush=True)
    print("Wrote SEACell_soft_summary.h5ad", flush=True)

if __name__ == '__main__':
    main()
