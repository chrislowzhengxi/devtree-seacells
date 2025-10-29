import os
import numpy as np
import pandas as pd
import scanpy as sc

import my_SEACells as SEACells

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

import random
np.random.seed(0)
random.seed(0)

'''
import dask.array as da
from dask.distributed import Client, LocalCluster

from dask_jobqueue import SLURMCluster
cluster = SLURMCluster(cores=48, memory='180GB', queue='caslake', account='rcc-staff', walltime='24:00:00')
cluster.scale(jobs=4) # Request 4 Slurm jobs, each with 24 cores
client = Client(cluster)
'''


# Some plotting aesthetics
sns.set_style('ticks')
matplotlib.rcParams['figure.figsize'] = [4, 4]
matplotlib.rcParams['figure.dpi'] = 100
FIG_DIR = 'figures'
os.makedirs(FIG_DIR, exist_ok=True)

def plot_and_save(fig_name):
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, fig_name))
    plt.close()
    
def initialize_data(filepath):
    ad = sc.read(filepath)
    print('Load data:', ad.layers)  # !xy
    
    #### For debugging
    #cell_indices = np.random.choice(ad.n_obs, size=2000, replace=False)
    #gene_indices = np.random.choice(ad.n_vars, size=1000, replace=False)
    #ad = ad[cell_indices, gene_indices]
    #ad = ad[cell_indices, :]
    
    ####
    # ad.layers['data']	 normalized counts + log1p transformed (what you’d use for e.g. scanpy.pp.highly_variable_genes, clustering, DE).
    # ad.layers['scale.data']  z-scored data (mean=0, sd=1) for clustering or differential analysis, distances, etc.
    # ad.X  this is scale.data (negative values typical of scaled z-scores).
    
    # this is only for Holly's BioTIP work using SCT normalization
    #ad.X = ad.layers['scale.data']
    ad.X = ad.layers['data'] #.toarray()  # based on 4/16/2025 email
    ad.X = ad.X.astype(np.float32)
    sc.pp.pca(ad, n_comps=30)
    ####
    
    diffdays_mapping = {
        'day0': 0, 'day1': 1, 'day3': 3,
        'day5': 5, 'day7': 7, 'day11': 11, 'day15': 15
    }
    ad.obs["time"] = ad.obs["diffday"].map(diffdays_mapping).astype(int)
    
    # raw_ad = sc.AnnData(ad.X)
    # raw_ad.obs_names, raw_ad.var_names = ad.obs_names, ad.raw_names
    # ad.raw = raw_ad
    
    ## correct the raw count assignment by Holly
    print('raw count: ', ad.raw.X.shape)  # (230786, 38847) # !xy
    
    if not ad.raw.var_names.is_unique:
        temp_raw_ad = ad.raw(ad.raw.X, var=ad.raw.var)
        temp_raw_ad.var_names_make_unique()
        ad.raw = temp_raw_ad
        print("\nmake adata.raw.var_names unique.")
    else:
        print("\nadata.raw.var_names are already unique.")
    
    ad.raw.obs = ad.obs
    
    return ad


def run_seacells(ad, n_SEACells, build_kernel_on='X_pca', n_waypoint_eigs=10):
    model = SEACells.core.SEACells(
        ad,
        build_kernel_on=build_kernel_on,
        n_SEACells=n_SEACells,
        n_waypoint_eigs=n_waypoint_eigs,
        convergence_epsilon=1e-5
    )
    model.construct_kernel_matrix()
    sns.clustermap(model.kernel_matrix[:500, :500].toarray())
    plot_and_save("kernel_matrix_clustermap.png")

    model.initialize_archetypes()
    SEACells.plot.plot_initialization(ad, model, save_as="figures/initialization_umap.png")
    #plot_and_save("initialization_umap.png")

    model.fit(min_iter=10, max_iter=50)
    for _ in range(5):
        model.step()

    model.plot_convergence(save_as="figures/rss_convergence.png")
    #plot_and_save("rss_convergence.png")
    return model

def summarize_and_evaluate(ad, model):
    plt.figure(figsize=(3, 2))
    sns.histplot((model.A_.T > 0.1).sum(axis=1), bins=30)
    plt.title('Non-trivial (>0.1) assignments per cell')
    plt.xlabel('# Non-trivial SEACell Assignments')
    plt.ylabel('# Cells')
    plot_and_save("nontrivial_assignments_hist.png")

    plt.figure(figsize=(3, 2))
    b = np.partition(model.A_.T, -5, axis=1)
    sns.heatmap(np.sort(b[:, -5:], axis=1)[:, ::-1], cmap='viridis', vmin=0)
    plt.title('Top 5 strongest assignments')
    plt.xlabel('$n^{th}$ strongest assignment')
    plot_and_save("top5_assignment_heatmap.png")

    SEACells.plot.plot_2D(ad, key='X_umap', colour_metacells=False, save_as="figures/umap_cells.png")
    #plot_and_save("umap_cells.png")
    SEACells.plot.plot_2D(ad, key='X_umap', colour_metacells=True, save_as="figures/umap_metacells.png")
    #plot_and_save("umap_metacells.png")
    SEACells.plot.plot_SEACell_sizes(ad, bins=5, save_as="figures/seacell_sizes.png")
    #plot_and_save("seacell_sizes.png")

    #if 'type' not in ad.obs:
    #    raise KeyError("Missing 'celltype' in ad.obs for purity evaluation.")

    
    purity = SEACells.evaluate.compute_celltype_purity(ad, 'leiden_0.5')
    plt.figure(figsize=(4, 4))
    sns.boxplot(data=purity, y='leiden_0.5_purity')
    plt.title('Celltype Purity')
    sns.despine()
    plot_and_save("celltype_purity.png")

    compactness = SEACells.evaluate.compactness(ad, 'X_pca')
    plt.figure(figsize=(4, 4))
    sns.boxplot(data=compactness, y='compactness')
    plt.title('Compactness')
    sns.despine()
    plot_and_save("compactness.png")

    separation = SEACells.evaluate.separation(ad, 'X_pca', nth_nbr=1)
    plt.figure(figsize=(4, 4))
    sns.boxplot(data=separation, y='separation')
    plt.title('Separation')
    sns.despine()
    plot_and_save("separation.png")

def main():
    input_file = '/project/imoskowitz/xyang2/heart_dev/GSE175634_iPSC_CM/sctransformed.sct3k_reclustered.h5ad'
    
    ad = initialize_data(input_file)
    
    # Save reference UMAP plots
    sc.pl.scatter(ad, basis='umap', color='leiden_0.5_type', frameon=False, save="_type.png")    
    sc.pl.scatter(ad, basis='umap', color='leiden_0.5', frameon=False, save="_leiden_0.5.png")

    n_cells = ad.shape[0]
    n_SEACells = n_cells // 200 #75
    model = run_seacells(ad, n_SEACells)

    # Save results
    ad.obs['SEACell'] = model.get_hard_assignments()
    ad.write("results/SEACells/output_with_SEACells.h5ad")
    
    summarize_and_evaluate(ad, model)

    # Summarized versions
    SEACell_ad = SEACells.core.summarize_by_SEACell(ad, SEACells_label='SEACell', summarize_layer='raw', ad_raw_var_names=True)
    SEACell_ad.write("results/SEACells/SEACell_summary.h5ad")

    SEACell_soft_ad = SEACells.core.summarize_by_soft_SEACell(ad, model.A_, celltype_label='leiden_0.5', summarize_layer='raw', 
                                                            minimum_weight=0.05, ad_raw_var_names=True)
    SEACell_soft_ad.write("results/SEACells/SEACell_soft_summary.h5ad")

if __name__ == "__main__":
    main()
    #client.close()
