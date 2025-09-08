import os
import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse
import seaborn as sns
import statsmodels.api as sm
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import LogNorm
from matplotlib import cm
from matplotlib.colors import Normalize
from pathlib import Path

# ---------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------
H5_ROOT      = Path("/project/imoskowitz/xyang2/SHH/Qiu_TimeLapse/results/raw_added")
CSV_META     = Path("/project/imoskowitz/xyang2/chrislowzhengxi/data/df_cell_celltyp_new_merged.csv")
RESULTS_ROOT = Path("/project/imoskowitz/xyang2/chrislowzhengxi/results")
RESULTS_SUBFOLDER = "system_qc"   # will contain one subfolder per system

sc.settings.figdir = RESULTS_ROOT

# plotting defaults
random_state = 0
sns.set_style("ticks")
plt.rcParams.update({"figure.dpi":300, "pdf.fonttype":42, "ps.fonttype":42})
TOP_N_EMBRYOS = 20

# staging ordering across all systems:
staging_order = [
    "E8.0", "E8.25", "E8.5", "E8.75",
    "E9.0", "E9.25", "E9.5", "E9.75",
    "E10.0", "E10.25", "E10.5", "E10.75",
    "E11.0", "E11.25", "E11.5", "E11.75",
    "E12.0", "E12.25", "E12.5", "E12.75",
    "E13.0", "E13.25", "E13.5", "E13.75",
    "E14.0", "E14.25", "E14.5", "E14.75",
    "E15.0", "E15.25", "E15.5", "E15.75",
    "E16.0", "E16.25", "E16.5", "E16.75",
    "E17.0", "E17.25", "E17.5", "E17.75",
    "E18.0", "E18.25", "E18.5", "E18.75", "P0"
]

# ---------------------------------------------------------------------
# DATA INITIALIZATION
# ---------------------------------------------------------------------
def initialize_data(h5_path, system_tag, csv_meta, min_cells_per_embryo=50,
                    subsample_frac=None, random_state=0):
    # load .h5ad
    print(f"Loading {{system_tag}} data from: {{h5_path}}")
    adata = sc.read_h5ad(h5_path)
    
    # fix categorical var names
    if adata.raw is not None and isinstance(adata.raw.var.index, pd.CategoricalIndex):
        adata.raw.var.index = adata.raw.var.index.astype(str)
    if isinstance(adata.var_names, pd.CategoricalIndex):
        adata.var_names = adata.var_names.astype(str)

    # read metadata
    req_cols = ["cell_id","day","somite_count","embryo_id",
                "experimental_id","system","celltype_new","meta_group", "embryo_sex"]
    df_meta = pd.read_csv(csv_meta, usecols=req_cols)
    # keep only this system
    df_meta = df_meta[df_meta.system == system_tag].copy()

    # build staging
    def _map_staging(row):
        d = row.day
        if d in ("E8","E8.0-E8.5","E8.5"):
            try:
                scount = int(str(row.somite_count).split()[0])
            except:
                return np.nan
            if scount <= 3:   return "E8.0"
            if scount <= 7:   return "E8.25"
            if scount <= 11:  return "E8.5"
            return "E8.5+"
        return d
    df_meta["staging"] = df_meta.apply(_map_staging, axis=1)

    # drop dupes, join to adata
    df_meta = df_meta.drop_duplicates('cell_id')
    adata = adata[adata.obs.cell_id.isin(df_meta.cell_id)].copy()
    meta_idx = df_meta.set_index('cell_id')
    for c in ['day','staging','somite_count','embryo_id','experimental_id','system','meta_group','celltype_new', 'embryo_sex']:
        adata.obs[c] = adata.obs['cell_id'].map(meta_idx[c])
    
    if "embryo_sex" in adata.obs:
        # normalize values a bit
        adata.obs["embryo_sex"] = adata.obs["embryo_sex"].astype(str).str.strip().str.upper()
        adata.obs["embryo_sex"].replace({"M":"MALE","F":"FEMALE"}, inplace=True)
        adata.obs.rename(columns={"embryo_sex": "sex"}, inplace=True)
        adata.obs["sex"] = adata.obs["sex"].astype("category")
    
    # enforce staging order
    adata.obs["staging"] = pd.Categorical(
    adata.obs["staging"],
    categories=staging_order,
    ordered=True,
    )

    # filter small embryos
    emb_counts = adata.obs.embryo_id.value_counts()
    keep_emb = emb_counts[emb_counts >= min_cells_per_embryo].index
    adata = adata[adata.obs.embryo_id.isin(keep_emb)].copy()

    # subsample
    if subsample_frac:
        sc.pp.subsample(adata, fraction=subsample_frac, random_state=random_state)

    # ensure PCA
    adata.var_names_make_unique()
    if 'X_pca' in adata.obsm:
        adata.obsm['X_pca'] = adata.obsm['X_pca'][:,:20]
    else:
        sc.tl.pca(adata, n_comps=20, random_state=random_state)

    # for building the pseudotime.
    sc.pp.neighbors(adata, use_rep="X_pca")

    if adata.raw is None:
        adata.raw = adata.copy()
    else:
        raw_adata = adata.raw.to_adata()  
    
    raw_adata.var_names_make_unique()
    adata.raw = raw_adata

    return adata



def ensure_log1p_cpm_layer(adata, layer_name="log1p_cpm", target_sum=1_000_000):
    """
    Create adata.layers[layer_name] with the SAME shape as adata.X, filled from adata.raw
    for overlapping genes, then CPM-normalize + log1p on that layer.
    Works for any dense/sparse combination of adata.X and raw.X.
    """
    n_obs, n_vars = adata.n_obs, adata.n_vars
    make_sparse = sparse.issparse(adata.X)

    # 1) allocate layer with the SAME shape/sparsity as adata.X
    layer = sparse.csr_matrix((n_obs, n_vars), dtype=np.float32) if make_sparse \
            else np.zeros((n_obs, n_vars), dtype=np.float32)

    # 2) overlap mapping
    raw = adata.raw.to_adata()
    common = adata.var_names.intersection(raw.var_names)
    if len(common) == 0:
        raise ValueError("No overlap between adata.var_names and adata.raw.var_names")

    var_idx = adata.var_names.get_indexer(common)   # cols in layer
    raw_idx = raw.var_names.get_indexer(common)     # cols in raw.X

    # 3) copy values from raw -> layer, respecting sparsity of the TARGET (layer)
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

    # 4) CPM + log1p on that layer
    sc.pp.normalize_total(adata, target_sum=target_sum, layer=layer_name)
    sc.pp.log1p(adata, layer=layer_name)
    return layer_name


def _genes_in_varnames(adata, genes):
    vset = {g.upper() for g in adata.var_names}
    return all(g.upper() in vset for g in genes)

def norm_raw_view(adata, genes, target_sum=1_000_000):
    """Return an AnnData view with X = log1p(CPM) from adata.raw for the requested genes."""
    if adata.raw is None:
        raise ValueError("adata.raw is None; cannot build raw-based view")

    raw = adata.raw.to_adata()
    # case‑insensitive map
    lut = {g.upper(): g for g in raw.var_names}
    keep = [lut[g.upper()] for g in genes if g.upper() in lut]
    if not keep:
        raise ValueError("None of the requested genes are in adata.raw.var_names")

    ad = raw[:, keep].copy()
    # normalize on X (counts from raw), then log1p
    sc.pp.normalize_total(ad, target_sum=target_sum)
    sc.pp.log1p(ad)

    # carry over needed obs columns for grouping/labels
    for c in ("staging", "day", "celltype_new"):
        if c in adata.obs:
            ad.obs[c] = adata.obs[c].values
    # enforce staging order
    ad.obs["staging"] = pd.Categorical(ad.obs["staging"], categories=staging_order, ordered=True)
    ad.obs["staging"] = ad.obs["staging"].cat.remove_unused_categories()
    return ad




# ---------------------------------------------------------------------
# PLOTTING
# ---------------------------------------------------------------------
def plot_system_qc(adata, outdir, system_tag, staging_order, top_n=20):
    obs = adata.obs.copy()
    print(f"[{system_tag}] plot_system_qc: saving to {outdir}")
    os.makedirs(outdir, exist_ok=True)

    # 1) cells per staging
    print(f"[{system_tag}]   • plotting cells per staging")
    df_cells = (obs.groupby('staging').size()
                   .reindex(staging_order).dropna()
                   .reset_index(name='n_cells'))
    fig,ax = plt.subplots(figsize=(10,4))
    sns.barplot(df_cells, x='staging', y='n_cells', color='#3182bd', ax=ax)
    ax.set(xlabel='staging', ylabel='n_cells', title=f'Cells per staging ({system_tag})')
    ax.set_yscale('log')

    ax.set_xticklabels(ax.get_xticklabels(),
                   rotation=75,
                   ha="center")

    fig.subplots_adjust(bottom=0.25,left=0.1)
    sns.despine(ax=ax)
    fig.savefig(Path(outdir)/f'cells_per_staging_{system_tag}.pdf')
    plt.close(fig)

    # 3) top-N embryos bar
    print(f"[{system_tag}]   • plotting top {top_n} embryos")
    df_emb = (obs.groupby('embryo_id').size().nlargest(top_n)
                 .reset_index(name='n_cells').sort_values('n_cells'))
    fig,ax = plt.subplots(figsize=(6, top_n*0.3+1))
    sns.barplot(df_emb, y='embryo_id', x='n_cells', color='#2ca02c', orient='h', ax=ax)
    ax.set(xlabel='n_cells',ylabel='embryo_id', title=f'Top {top_n} embryos ({system_tag})')
    fig.subplots_adjust(left=0.35)
    sns.despine(ax=ax,left=True,bottom=True)
    fig.savefig(Path(outdir)/f'top{top_n}_embryos_{system_tag}.pdf')
    plt.close(fig)

    # 4) heatmap top-N
    print(f"[{system_tag}]   • plotting heatmaps")
    top_ids = df_emb.embryo_id.tolist()
    pivot = (obs[obs.embryo_id.isin(top_ids)]
             .groupby(['embryo_id','staging']).size()
             .unstack(fill_value=0).reindex(columns=staging_order,fill_value=0)
             .loc[top_ids])
    fig,ax = plt.subplots(figsize=(12,6))
    sns.heatmap(pivot, cmap='viridis',
                norm=LogNorm(vmin=max(pivot.values.min(),1), vmax=pivot.values.max()),
                linewidths=0.3,cbar_kws={'label':'n_cells (log scale)'}, ax=ax)
    ax.set(title=f'Cells per embryo × staging (top {top_n})', xlabel='staging', ylabel='embryo_id')

    n = pivot.shape[1]
    ax.set_xticks(np.arange(n) + 0.5)
    ax.set_xticklabels(pivot.columns, rotation=75, ha="center")

    
    fig.subplots_adjust(bottom=0.3,left=0.15)
    sns.despine(ax=ax)
    fig.savefig(Path(outdir)/f'heatmap_top{top_n}_embryo_{system_tag}.pdf')
    plt.close(fig)

    # 5) heatmap all embryos
    pivot_all = (obs.groupby(['embryo_id','staging']).size()
                   .unstack(fill_value=0).reindex(columns=staging_order,fill_value=0))
    fig,ax = plt.subplots(figsize=(12, max(6,pivot_all.shape[0]*0.1)))
    sns.heatmap(pivot_all, cmap='viridis',
                norm=LogNorm(vmin=max(pivot_all.values.min(),1), vmax=pivot_all.values.max()),
                linewidths=0.3, cbar_kws={'label':'n_cells'}, ax=ax)
    ax.set(title=f'Cells per embryo × staging (all — {system_tag})', xlabel='staging', ylabel='embryo_id')

    n_all = pivot_all.shape[1]
    ax.set_xticks(np.arange(n_all) + 0.5)
    ax.set_xticklabels(pivot_all.columns, rotation=75, ha="center")

    fig.subplots_adjust(bottom=0.25,left=0.15)
    fig.savefig(Path(outdir)/f'heatmap_all_embryos_{system_tag}.pdf')
    plt.close(fig)

    print(f"[{system_tag}] plot_system_qc done\n")

    

# # Added legend_loc to sc.pl.umap calls
def _umap_color_from_log1p_cpm(adata, genes):
    cols = []
    for g in genes:
        col = f"expr_{g}_log1pcpm"
        adata.obs[col] = _gene_expr(adata, g, layer="log1p_cpm")  # uses layer; falls back to raw
        cols.append(col)
    return cols

def plot_umaps(adata, outdir, system_tag, genes):
    print(f"[{system_tag}] plot_umaps: saving to {outdir}")
    os.makedirs(outdir, exist_ok=True)

    # day / celltype
    for key in ["day", "celltype_new", "sex"]:
        sc.pl.umap(
            adata, color=key, use_raw=False, size=20,
            title=f"{system_tag} UMAP — {key}",
            save=f"_{system_tag}_{key}.pdf", legend_loc="right margin",
        )

    gene_cols = _umap_color_from_log1p_cpm(adata, genes)
    custom_cmap = cm.get_cmap("magma").copy()
    custom_cmap.set_under("lightgray") 

    # genes from normalized values in obs
    gene_cols = _umap_color_from_log1p_cpm(adata, genes)
    sc.pl.umap(
        adata,
        color=gene_cols,
        use_raw=False,
        size=20,
        title=[f"{system_tag} UMAP — {g}" for g in genes],
        legend_loc="on data",
        cmap=custom_cmap,
        vmin=0.01,
        ncols=len(genes),
        save=f"_{system_tag}_genes_umap.pdf",
    )



def _pick_root_cell(adata, system_tag, rng_seed=0):
    """
    Root selection:
      • GUT (Holly-style): pick a most-differentiated anchor.
          - Rank candidate top-of-tree lineages by median stage_code (later = larger).
          - Within the chosen lineage at its latest stage, pick PCA-centroid cell.
      • RENAL: 'Anterior intermediate mesoderm' at its earliest present stage; representative cell.
      • Else: absolute earliest stage overall.
    Returns: (root_obs_name, "Celltype @ Stage")
    """
    rng = np.random.default_rng(rng_seed)

    # ensure ordered categorical + numeric code
    adata.obs["staging"] = pd.Categorical(adata.obs["staging"],
                                          categories=staging_order, ordered=True)
    adata.obs["stage_code"] = adata.obs["staging"].cat.codes
    valid_stage = adata.obs["stage_code"] >= 0

    obs = adata.obs

    def _representative_index(mask):
        """Pick cell nearest to the subset centroid in X_pca (falls back to random if missing)."""
        idxs = np.flatnonzero(mask.values)
        if idxs.size == 0:
            return None
        if "X_pca" not in adata.obsm or adata.obsm["X_pca"] is None:
            return int(rng.choice(idxs))
        X = adata.obsm["X_pca"][idxs, :]
        c = X.mean(axis=0)
        d = ((X - c) ** 2).sum(axis=1)  # L2 distance
        return idxs[int(np.argmin(d))]

    if system_tag == "Gut":
        candidates_all = [
            "Lung progenitor cells",
            "Hepatocytes",
            "Pancreatic acinar cells",
            "Foregut epithelial cells",
            "Midgut/Hindgut epithelial cells",
            "Pancreatic islets",
        ]
        present_types = set(obs["celltype_new"].unique())
        candidates = [ct for ct in candidates_all if ct in present_types]

        if candidates:
            # score by differentiation proxy: median stage_code (tie-breaker: max)
            scored = []
            for ct in candidates:
                m = (obs["celltype_new"] == ct) & valid_stage
                if m.any():
                    med_stage = int(obs.loc[m, "stage_code"].median())
                    max_stage = int(obs.loc[m, "stage_code"].max())
                    scored.append((ct, med_stage, max_stage))
            if scored:
                scored.sort(key=lambda x: (x[1], x[2]), reverse=True)
                chosen_ct, _, _ = scored[0]
                latest_stage = int(obs.loc[(obs["celltype_new"] == chosen_ct) & valid_stage, "stage_code"].max())
                mask = (obs["celltype_new"] == chosen_ct) & (obs["stage_code"] == latest_stage) & valid_stage
            else:
                # no valid staged cells among candidates → fall back
                latest_stage = int(obs.loc[valid_stage, "stage_code"].max())
                mask = (obs["stage_code"] == latest_stage) & valid_stage
        else:
            # no candidates present → absolute latest stage overall
            latest_stage = int(obs.loc[valid_stage, "stage_code"].max())
            mask = (obs["stage_code"] == latest_stage) & valid_stage

        pick = _representative_index(mask)
        if pick is None:
            # final fallback: any cell among candidates or any valid staged cell
            fallback_mask = obs["celltype_new"].isin(candidates) if candidates else valid_stage
            pick = _representative_index(fallback_mask)
            if pick is None:
                pick = 0  # last resort
        root = obs.index[pick]
        return root, f"{obs.loc[root,'celltype_new']} @ {obs.loc[root,'staging']}"

    elif system_tag == "Renal":
        lineage = "Anterior intermediate mesoderm"
        ct_mask = (obs["celltype_new"] == lineage) & valid_stage
        earliest = None
        for s in staging_order:
            m = ct_mask & (obs["staging"] == s)
            if m.any():
                earliest = s
                break
        if earliest is None:
            # fallback to earliest stage overall
            for s in staging_order:
                m = valid_stage & (obs["staging"] == s)
                if m.any():
                    earliest = s
                    break
            mask = valid_stage & (obs["staging"] == earliest)
        else:
            mask = ct_mask & (obs["staging"] == earliest)

        pick = _representative_index(mask)
        root = obs.index[pick]
        return root, f"{obs.loc[root,'celltype_new']} @ {earliest}"
    
    elif system_tag == "PNS_neurons":
        lineage = "Neural crest (PNS neurons)"
        ct_mask = (obs["celltype_new"] == lineage) & valid_stage
        earliest = None
        for s in staging_order:
            m = ct_mask & (obs["staging"] == s)
            if m.any():
                earliest = s
                break
        if earliest is None:
            # fallback to earliest stage overall
            for s in staging_order:
                m = valid_stage & (obs["staging"] == s)
                if m.any():
                    earliest = s
                    break
            mask = valid_stage & (obs["staging"] == earliest)
        else:
            mask = ct_mask & (obs["staging"] == earliest)

        pick = _representative_index(mask)
        root = obs.index[pick]
        return root, f"{obs.loc[root,'celltype_new']} @ {earliest}"

    elif system_tag == "Lateral_plate_mesoderm":
        lineage = "Atrial cardiomyocytes"
        ct_mask = (obs["celltype_new"] == lineage) & valid_stage

        # restrict to embryo 161 if present
        emb_mask = (obs["embryo_id"].astype(str) == "161")
        ct_mask = ct_mask & emb_mask

        # pick the LATEST stage where atrial cardiomyocytes appear
        latest = None
        for s in reversed(staging_order):
            m = ct_mask & (obs["staging"] == s)
            if m.any():
                latest = s
                break

        if latest is None:
            # fallback: latest stage overall if no atrial cells in 161
            for s in reversed(staging_order):
                m = valid_stage & (obs["staging"] == s)
                if m.any():
                    latest = s
                    break
            mask = valid_stage & (obs["staging"] == latest)
        else:
            mask = ct_mask & (obs["staging"] == latest)

        pick = _representative_index(mask)
        if pick is None:
            pick = _representative_index(ct_mask) or _representative_index(valid_stage)
        root = obs.index[int(pick)]
        return root, f"{obs.loc[root,'celltype_new']} @ {latest}"




    else:
        # generic fallback: absolute earliest stage overall
        for s in staging_order:
            m = valid_stage & (obs["staging"] == s)
            if m.any():
                pick = _representative_index(m)
                root = obs.index[pick]
                return root, f"{obs.loc[root,'celltype_new']} @ {s}"


def plot_pseudotime_expression(adata, genes, outdir, system_tag, file_tag="", rng_seed=0):
    LAYER = "log1p_cpm"

    print(f"[{system_tag}] plot_pseudotime_expression: saving to {outdir}")
    os.makedirs(outdir, exist_ok=True)
    print(f"[{system_tag}]   • computing neighbors + diffusion map + DPT")

    # graph + diffusion
    sc.pp.neighbors(adata, use_rep="X_pca")
    sc.tl.diffmap(adata, n_comps=10)

    # clean any prior DPT
    adata.obs.drop(columns=["dpt_pseudotime"], errors="ignore", inplace=True)

    # choose root via the unified function
    root_name, root_label = _pick_root_cell(adata, system_tag, rng_seed=rng_seed)
    root_idx = adata.obs_names.get_loc(root_name)
    adata.uns["iroot"] = int(root_idx)
    print(f"[{system_tag}] Using root {root_name} ({root_label}); idx={root_idx}")

    # DPT (Holly: n_dcs=6)
    sc.tl.dpt(adata, n_dcs=6)
    print(f"[{system_tag}]   ✓ DPT done")

    # Flip direction for Gut to read progenitor → differentiated (1 - pt)
    if system_tag == "Gut":
        adata.obs["dpt_pseudotime_raw"] = adata.obs["dpt_pseudotime"].copy()
        adata.obs["dpt_pseudotime"] = 1.0 - adata.obs["dpt_pseudotime"]

    pt = adata.obs["dpt_pseudotime"]

    # plots
    for gene in genes:
        print(f"[{system_tag}]   • plotting pseudotime for {gene}")
        expr = _gene_expr(adata, gene, layer=LAYER)
        df = pd.DataFrame({
            "pseudotime": pt.values,
            "expr": expr,
            "day": adata.obs["day"],
            "celltype": adata.obs["celltype_new"],
        })

        fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
        for ax, color_by, cmap in zip(axes, ["day", "celltype"], ["viridis", "tab20"]):
            sns.scatterplot(
                data=df, x="pseudotime", y="expr",
                hue=color_by, palette=cmap,
                alpha=0.4, s=10, ax=ax, legend=False
            )
            smoothed = sm.nonparametric.lowess(df["expr"], df["pseudotime"], frac=0.3)
            ax.plot(smoothed[:, 0], smoothed[:, 1], color="k", lw=2)
            ax.set(xlabel="pseudotime", title=f"{gene} expression colored by {color_by}")
        axes[0].set(ylabel=f"expression ({LAYER} or raw fallback)")

        flip_note = " (1 − DPT)" if system_tag == "Gut" else ""
        fig.suptitle(f"{system_tag}: {gene} vs pseudotime{flip_note}\nroot: {root_label}",
                     y=1.02, fontsize=10)
        fig.tight_layout()
        fig.savefig(Path(outdir) / f"pseudotime{file_tag}_{system_tag}_{gene}.pdf",
                    bbox_inches="tight")
        plt.close(fig)

    print(f"[{system_tag}] plot_pseudotime_expression done\n")


# ---------------------------
# Reverse pseudotime chooser
# ---------------------------
def _ensure_stage_code(adata):
    adata.obs["staging"] = pd.Categorical(adata.obs["staging"],
                                          categories=staging_order, ordered=True)
    adata.obs["stage_code"] = adata.obs["staging"].cat.codes
    return adata.obs["stage_code"]

def _ensure_dpt_if_missing(adata, system_tag, rng_seed=0):
    if "dpt_pseudotime" in adata.obs:
        return
    sc.pp.neighbors(adata, use_rep="X_pca")
    sc.tl.diffmap(adata, n_comps=10)
    root_name, _ = _pick_root_cell(adata, system_tag, rng_seed=rng_seed)
    adata.uns["iroot"] = int(adata.obs_names.get_loc(root_name))
    sc.tl.dpt(adata, n_dcs=6)

# def choose_anchor_and_make_reverse_pseudotime(
#     adata,
#     outdir,
#     system_tag,
#     candidates=("Airway goblet cells", "Lung cells (Eln+)"),
#     late_quantile=0.75,
#     compare_by=("staging", "pseudotime"),   # can be ("staging",), ("pseudotime",) or both
#     pick_on="staging",                      # which comparison decides the anchor
#     file_tag=""
# ):
#     """
#     1) Compare the two candidate bottom lineages by how enriched they are in *late* cells.
#        - 'staging': top quartile of stage_code across all cells
#        - 'pseudotime': top quartile of (unflipped) DPT values
#     2) Pick the winner (by `pick_on`) and write reverse pseudotime = 1 - base_pt to obs.
#     3) Save simple barplots of late proportions for transparency.

#     Returns: dict with chosen anchors and proportions.
#     """
#     os.makedirs(outdir, exist_ok=True)
#     _ensure_stage_code(adata)
#     _ensure_dpt_if_missing(adata, system_tag)

#     obs = adata.obs
#     # If Gut was previously flipped, recover the original orientation if available
#     base_pt = obs["dpt_pseudotime_raw"] if "dpt_pseudotime_raw" in obs else obs["dpt_pseudotime"]

#     # thresholds for "late"
#     out = {}
#     if "staging" in compare_by:
#         stage_thr = obs.loc[obs["stage_code"] >= 0, "stage_code"].quantile(late_quantile)
#         rows = []
#         for ct in candidates:
#             m = (obs["celltype_new"] == ct) & (obs["stage_code"] >= 0)
#             n_tot = int(m.sum())
#             n_late = int(((obs["stage_code"] >= stage_thr) & m).sum())
#             prop = n_late / n_tot if n_tot > 0 else np.nan
#             rows.append({"candidate": ct, "n_total": n_tot, "n_late": n_late, "prop_late": prop})
#         df_stage = pd.DataFrame(rows)
#         # plot
#         fig, ax = plt.subplots(figsize=(4.5, 3.2))
#         sns.barplot(df_stage, x="candidate", y="prop_late", ax=ax)
#         ax.set_ylabel(f"Late-stage proportion (≥ Q{int(late_quantile*100)})")
#         ax.set_xlabel("")
#         ax.set_title(f"{system_tag}: Late by staging")
#         plt.xticks(rotation=20, ha="right")
#         plt.tight_layout()
#         plt.savefig(Path(outdir)/f"{system_tag}_anchor_compare_staging{file_tag}.pdf")
#         plt.close(fig)
#         out["staging"] = {"table": df_stage, "winner": df_stage.sort_values("prop_late").iloc[-1]["candidate"]}

#     if "pseudotime" in compare_by:
#         pt_thr = pd.Series(base_pt).quantile(late_quantile)
#         rows = []
#         for ct in candidates:
#             m = (obs["celltype_new"] == ct) & (~pd.isna(base_pt))
#             n_tot = int(m.sum())
#             n_late = int(((base_pt >= pt_thr) & m).sum())
#             prop = n_late / n_tot if n_tot > 0 else np.nan
#             rows.append({"candidate": ct, "n_total": n_tot, "n_late": n_late, "prop_late": prop})
#         df_pt = pd.DataFrame(rows)
#         # plot
#         fig, ax = plt.subplots(figsize=(4.5, 3.2))
#         sns.barplot(df_pt, x="candidate", y="prop_late", ax=ax)
#         ax.set_ylabel(f"Late-PT proportion (≥ Q{int(late_quantile*100)})")
#         ax.set_xlabel("")
#         ax.set_title(f"{system_tag}: Late by pseudotime")
#         plt.xticks(rotation=20, ha="right")
#         plt.tight_layout()
#         plt.savefig(Path(outdir)/f"{system_tag}_anchor_compare_pseudotime{file_tag}.pdf")
#         plt.close(fig)
#         out["pseudotime"] = {"table": df_pt, "winner": df_pt.sort_values("prop_late").iloc[-1]["candidate"]}

#     # pick the anchor by the requested criterion
#     if pick_on not in out:
#         raise ValueError(f"`pick_on`='{pick_on}' not in computed comparisons {list(out.keys())}")
#     chosen_anchor = out[pick_on]["winner"]

#     # compute reverse pseudotime (store; leave original untouched)
#     adata.obs["reverse_pseudotime"] = 1.0 - np.asarray(base_pt, dtype=float)
#     adata.uns["reverse_pt_anchor"] = {
#         "candidates": list(candidates),
#         "late_quantile": late_quantile,
#         "compare_by": list(compare_by),
#         "pick_on": pick_on,
#         "chosen_anchor": chosen_anchor,
#     }
#     print(f"[{system_tag}] Reverse pseudotime written to obs['reverse_pseudotime'] "
#           f"(anchor picked by {pick_on}: {chosen_anchor})")

#     return out
# # ---------------------------
# # End of Reverse pseudotime chooser
# # ---------------------------

def choose_anchor_and_make_reverse_pseudotime(
    adata,
    outdir,
    system_tag,
    candidates=None,                 # None -> auto-pick top_k candidates from celltype_new
    top_k=2,                         # how many auto candidates to consider if candidates=None
    late_quantile=0.75,
    compare_by=("staging", "pseudotime"),   # any subset of {"staging","pseudotime"}
    pick_on="staging",                       # which comparison decides the anchor
    file_tag="",
    # optional biasing toward a specific embryo's latest stage (general, not system specific)
    embryo_bias_id=None,             # e.g., "161"
    embryo_bias_weight=1.0,          # add this to "late" counts for cells at the embryo's latest stage
):
    """
    Generalized reverse pseudotime chooser.

    Workflow:
      1) Ensure stage codes and DPT exist. Use base_pt = original DPT
         (if you had flipped earlier, we use dpt_pseudotime_raw to avoid double flipping).
      2) If candidates is None, auto-pick top_k candidate celltypes by fraction of "late" cells
         using the first metric in `compare_by` (usually 'staging').
      3) For each candidate, compute proportion of "late" cells under each metric in compare_by.
         Optionally add a small bias for cells at a specific embryo's latest stage.
      4) Choose the winner according to `pick_on`.
      5) Write reverse pseudotime = 1 - base_pt to obs and save barplots.

    Returns:
      dict with keys in compare_by:
        { metric: {"table": df, "winner": <celltype>} }, plus the chosen anchor in adata.uns.
    """
    os.makedirs(outdir, exist_ok=True)
    _ensure_stage_code(adata)
    _ensure_dpt_if_missing(adata, system_tag)

    obs = adata.obs
    base_pt = obs["dpt_pseudotime_raw"] if "dpt_pseudotime_raw" in obs else obs["dpt_pseudotime"]
    out = {}

    # ---------------------------
    # Helper: compute "late" proportions per celltype under a metric
    # ---------------------------
    def _late_table(metric):
        rows = []
        if metric == "staging":
            mask_valid = obs["stage_code"] >= 0
            thr = obs.loc[mask_valid, "stage_code"].quantile(late_quantile)
            is_late = (obs["stage_code"] >= thr) & mask_valid
        elif metric == "pseudotime":
            s = pd.Series(base_pt)
            thr = s.quantile(late_quantile)
            is_late = ~pd.isna(base_pt) & (base_pt >= thr)
        else:
            raise ValueError(f"Unknown metric: {metric}")

        # optional embryo bias toward its latest stage
        bias = pd.Series(0.0, index=obs.index)
        if embryo_bias_id is not None and "embryo_id" in obs and "staging" in obs:
            try:
                # find latest stage present for this embryo
                emb_mask = (obs["embryo_id"].astype(str) == str(embryo_bias_id)) & obs["staging"].notna()
                if emb_mask.any():
                    latest_code = (obs.loc[emb_mask, "staging"]
                                     .cat.codes.replace({-1: np.nan}).max())
                    late_stage_mask = emb_mask & (obs["staging"].cat.codes == latest_code)
                    # add bias only to cells that are "late" under the metric and also embryo-late stage
                    bias.loc[late_stage_mask & is_late] = float(embryo_bias_weight)
            except Exception:
                pass

        # choose candidate set
        cts = sorted(obs["celltype_new"].dropna().astype(str).unique())
        if not cts:
            return pd.DataFrame(columns=["candidate", "n_total", "n_late", "prop_late"])

        # preselect candidate pool if needed
        pool = cts
        if candidates is not None:
            pool = [c for c in candidates if c in cts]

        # If candidates not supplied, autoselect by the first metric in compare_by
        if candidates is None and metric == (compare_by[0] if len(compare_by) else "staging"):
            # rank all by late proportion under this metric
            stats_all = []
            for ct in cts:
                m = (obs["celltype_new"].astype(str) == ct)
                n_tot = int(m.sum())
                if n_tot == 0:
                    continue
                # biased late count
                n_late = float(((is_late & m).sum())) + float(bias[m].sum())
                prop = n_late / (n_tot + 1e-9)
                stats_all.append((ct, n_tot, n_late, prop))
            stats_all.sort(key=lambda x: x[3], reverse=True)
            pool = [t[0] for t in stats_all[:max(1, top_k)]]

        # make the table for the current pool
        for ct in pool:
            m = (obs["celltype_new"].astype(str) == ct)
            n_tot = int(m.sum())
            if n_tot == 0:
                rows.append({"candidate": ct, "n_total": 0, "n_late": 0.0, "prop_late": np.nan})
                continue
            n_late = float(((is_late & m).sum())) + float(bias[m].sum())
            prop = n_late / (n_tot + 1e-9)
            rows.append({"candidate": ct, "n_total": n_tot, "n_late": n_late, "prop_late": prop})

        df = pd.DataFrame(rows)
        if not df.empty:
            df = df.sort_values("prop_late", ascending=False)
        return df

    # ---------------------------
    # Build tables for each requested metric
    # ---------------------------
    for metric in compare_by:
        df_metric = _late_table(metric)
        if df_metric is None or df_metric.empty:
            continue

        # plot bar
        fig, ax = plt.subplots(figsize=(max(4.5, 0.45*len(df_metric)), 3.2))
        sns.barplot(df_metric, x="candidate", y="prop_late", ax=ax)
        ax.set_ylabel(f"Late proportion (≥ Q{int(late_quantile*100)})")
        ax.set_xlabel("")
        ax.set_title(f"{system_tag}: Late by {metric}")
        plt.xticks(rotation=20, ha="right")
        plt.tight_layout()
        plt.savefig(Path(outdir)/f"{system_tag}_anchor_compare_{metric}{file_tag}.pdf")
        plt.close(fig)

        out[metric] = {
            "table": df_metric.reset_index(drop=True),
            "winner": None if df_metric.empty else df_metric.iloc[0]["candidate"]
        }

    # pick winner
    if pick_on not in out or out[pick_on]["winner"] is None:
        raise ValueError(
            f"`pick_on`='{pick_on}' not available or empty. "
            f"Computed metrics: {list(out.keys())}. "
            f"Check that `celltype_new` exists and candidates are present."
        )
    chosen_anchor = out[pick_on]["winner"]

    # write reverse pseudotime
    adata.obs["reverse_pseudotime"] = 1.0 - np.asarray(base_pt, dtype=float)
    adata.uns["reverse_pt_anchor"] = {
        "candidates": (list(candidates) if candidates is not None else "auto"),
        "auto_top_k": top_k if candidates is None else None,
        "late_quantile": late_quantile,
        "compare_by": list(compare_by),
        "pick_on": pick_on,
        "chosen_anchor": chosen_anchor,
        "embryo_bias_id": embryo_bias_id,
        "embryo_bias_weight": embryo_bias_weight,
    }
    print(f"[{system_tag}] Reverse pseudotime written to obs['reverse_pseudotime'] "
          f"(anchor picked by {pick_on}: {chosen_anchor})")

    return out



# ---------------------------
# Start of Holly's requested clustermap and dotplots  
# ---------------------------
# -------- helpers --------
def _get_expr_from(adata, gene, layer="log1p_cpm", fallback_to_raw=True):
    """1D np.array expression for `gene`. Tries adata.layers[layer] (aligned to adata.var_names),
    then falls back to adata.raw if requested. Case-insensitive gene lookup."""
    # case-insensitive map
    lut = {g.upper(): i for i, g in enumerate(adata.var_names)}
    idx = lut.get(gene.upper(), None)

    if (layer in adata.layers) and (idx is not None):
        M = adata.layers[layer]
        col = M[:, idx]
        return col.toarray().ravel() if sparse.issparse(col) else np.asarray(col).ravel()

    if fallback_to_raw and (adata.raw is not None):
        rnames = pd.Index(adata.raw.var_names)
        ridx = {g.upper(): i for i, g in enumerate(rnames)}.get(gene.upper(), None)
        if ridx is not None:
            col = adata.raw.X[:, ridx]
            return col.toarray().ravel() if sparse.issparse(col) else np.asarray(col).ravel()

    # not found → zeros (keeps plotting code robust)
    return np.zeros(adata.n_obs, dtype=float)

def _mean_matrix(adata, gene, groupx, groupy, layer="log1p_cpm",
                 fallback_to_raw=True, drop_allzero_cols=True):
    """Return a (groupy × groupx) DataFrame of mean expression for `gene`."""
    # pull expression
    expr = _get_expr_from(adata, gene, layer=layer, fallback_to_raw=fallback_to_raw)

    # ensure both grouping keys exist
    if groupx not in adata.obs or groupy not in adata.obs:
        raise KeyError(f"Missing obs key(s): {groupx if groupx not in adata.obs else ''} "
                       f"{groupy if groupy not in adata.obs else ''}")

    df = pd.DataFrame({
        groupx: adata.obs[groupx].astype("category"),
        groupy: adata.obs[groupy].astype("category"),
        "_expr": expr
    })

    # keep declared category order if present (esp. your `staging_order`)
    def _order(series):
        if isinstance(series.dtype, pd.CategoricalDtype):
            # drop unused to speed up plots
            return series.cat.remove_unused_categories()
        return series

    df[groupx] = _order(df[groupx])
    df[groupy] = _order(df[groupy])

    mat = (df.groupby([groupy, groupx], observed=False)["_expr"]
             .mean().unstack(fill_value=0))

    if drop_allzero_cols:
        # drop groupx levels with no signal at all (helps clustermap width)
        keep_cols = mat.columns[(mat > 0).any(axis=0)]
        if len(keep_cols) > 0:
            mat = mat.loc[:, keep_cols]

    return mat  # rows: groupy, cols: groupx

# -------- dotplot (obsB × obsA) --------
def plot_gene_dotplot_2d(adata, gene, groupx, groupy, outdir, system_tag,
                         file_tag="", layer="log1p_cpm", fallback_to_raw=True,
                         vmin=0.0, vmax=None, cmap="RdBu_r", min_dot=20, max_dot=200):
    """
    Dot color = mean expression; dot size = mean expression (scaled).
    groupx on X (e.g., 'staging'); groupy on Y (e.g., 'celltype_new').
    """
    Path(outdir).mkdir(parents=True, exist_ok=True)
    mat = _mean_matrix(adata, gene, groupx, groupy, layer, fallback_to_raw)
    if mat.empty:
        print(f"[{system_tag}] {gene}: no data to plot for {groupx}×{groupy}")
        return

    # long-form for scatter
    df = mat.reset_index().melt(id_vars=groupy, var_name=groupx, value_name="mean_expr")
    # keep only positive signal to reduce clutter (matches Holly’s look)
    df = df[df["mean_expr"] > 0].copy()
    if df.empty:
        print(f"[{system_tag}] {gene}: no positive mean expression to plot.")
        return

    if vmax is None:
        vmax = float(df["mean_expr"].quantile(0.99)) or float(df["mean_expr"].max() or 1.0)

    # sequential size scaling
    size = min_dot + (max_dot - min_dot) * (df["mean_expr"] / vmax).clip(0, 1)
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

    # figure size adapts to category counts
    fig_w = min(14, 0.18 * mat.shape[1] + 4)
    fig_h = max(5, 0.33 * mat.shape[0])
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    sca = ax.scatter(df[groupx], df[groupy], c=df["mean_expr"], s=size,
                     cmap=cmap, norm=norm, edgecolor="none", alpha=0.9)

    ax.set_xlabel(groupx); ax.set_ylabel(groupy)
    ax.set_title(f"{system_tag}: {gene} — mean expression by {groupy} × {groupx}")
    plt.xticks(rotation=90)
    cb = plt.colorbar(sca, ax=ax, pad=0.01)
    cb.set_label("Mean expression")

    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.grid(False)
    fig.tight_layout()

    fname = f"{gene}_{system_tag}_dotplot_{groupy}_by_{groupx}{file_tag}.pdf"
    fig.savefig(Path(outdir)/fname, bbox_inches="tight")
    plt.close(fig)

# -------- clustermap (obsB × obsA) --------
def plot_gene_clustermap_2d(
    adata, gene, groupx, groupy, outdir, system_tag,
    file_tag="", layer="log1p_cpm", fallback_to_raw=True,
    vmin=0.0, vmax=None, cmap="RdBu_r",
    col_cluster=False, row_cluster=True,
):
    Path(outdir).mkdir(parents=True, exist_ok=True)
    mat = _mean_matrix(adata, gene, groupx, groupy, layer, fallback_to_raw)
    if mat.empty:
        print(f"[{system_tag}] {gene}: no data to plot for {groupx}×{groupy}")
        return

    # keep only rows with any signal
    mat = mat.loc[mat.index[(mat > 0).any(axis=1)]].astype(float).fillna(0.0)
    if mat.empty:
        print(f"[{system_tag}] {gene}: matrix all zero after filtering.")
        return

    if vmax is None:
        vmax = float(np.nanpercentile(mat.values, 99)) or float(mat.values.max() or 1.0)

    # Decide layout
    rclust = bool(row_cluster and mat.shape[0] >= 3)
    cclust = bool(col_cluster and mat.shape[1] >= 3)

    # Figure size
    fig_w = min(16, 0.22 * mat.shape[1] + 4)
    fig_h = max(4.5, 0.40 * mat.shape[0])

    title = f"{system_tag}: {gene} — mean expression\n ( {groupy} × {groupx} )"

    if rclust or cclust:
        # Reserve a right margin for the colorbar so it never covers the title
        g = sns.clustermap(
            mat, cmap=cmap, vmin=vmin, vmax=vmax,
            row_cluster=rclust, col_cluster=cclust,
            figsize=(fig_w, fig_h),
            cbar_pos=(0.1, 0.3, 0.03, 0.5),
            dendrogram_ratio=(0.08, 0.08),
            cbar_kws={"label": "Mean expression"},
        )

        g.cax.set_ylabel("Mean expression", rotation=0, labelpad=10, ha="left")
        g.fig.suptitle(title, x=0.9, y=0.98, fontsize=11)
        g.fig.subplots_adjust(top=0.90, bottom=0.15, left=0.25, right=0.95)

        # Hide empty dendrogram axes if clustering is off on an axis
        if not rclust and hasattr(g, "ax_row_dendrogram"):
            g.ax_row_dendrogram.set_visible(False)
        if not cclust and hasattr(g, "ax_col_dendrogram"):
            g.ax_col_dendrogram.set_visible(False)

        g.ax_heatmap.set_xlabel(groupx)
        g.ax_heatmap.set_ylabel(groupy)
        for lab in g.ax_heatmap.get_xticklabels():
            lab.set_rotation(90)
            lab.set_ha("center")

        # Add title with padding; make room at top
        g.fig.suptitle(title, y=0.98, fontsize=11)
        g.fig.subplots_adjust(top=0.90, right=0.88, left=0.22, bottom=0.15)
        

        fname = f"{gene}_{system_tag}_clustermap_{groupy}_by_{groupx}{file_tag}.pdf"
        g.savefig(Path(outdir)/fname, bbox_inches="tight")
        plt.close(g.fig)

    else:
        # Tiny matrix → plain heatmap with a separate colorbar axis on the right
        fig = plt.figure(figsize=(fig_w, fig_h))
        # axes: [left, bottom, width, height] in figure coords
        ax = fig.add_axes([0.20, 0.15, 0.65, 0.70])
        cax = fig.add_axes([0.88, 0.20, 0.02, 0.60])

        hm = sns.heatmap(
            mat, ax=ax, cbar=True, cbar_ax=cax,
            cmap=cmap, vmin=vmin, vmax=vmax,
            cbar_kws={"label": "Mean expression"},
        )
        ax.set_xlabel(groupx); ax.set_ylabel(groupy)
        for lab in ax.get_xticklabels():
            lab.set_rotation(90); lab.set_ha("center")

        fig.suptitle(title, y=0.98, fontsize=11)
        fname = f"{gene}_{system_tag}_heatmap_{groupy}_by_{groupx}{file_tag}.pdf"
        fig.savefig(Path(outdir)/fname, bbox_inches="tight")
        plt.close(fig)


# -------- convenience wrapper (like your stacked violin) --------
def plot_dot_and_clustermap_per_gene(
    adata, genes, outdir, system_tag, file_tag="",
    groupby=("staging","celltype_new"),
    layer="log1p_cpm", fallback_to_raw=True, cmap="RdBu_r",
):
    if not (isinstance(groupby, (list, tuple)) and len(groupby) == 2):
        raise ValueError("groupby must be a 2-tuple/list like ('staging','celltype_new').")
    groupx, groupy = groupby

    if groupx == "staging" and "staging" in adata.obs:
        adata.obs["staging"] = pd.Categorical(
            adata.obs["staging"],
            categories=[s for s in staging_order if s in set(adata.obs["staging"].astype(str))],
            ordered=True,
        )

    for gene in genes:
        try:
            print(f"[{system_tag}] • {gene} → dotplot & clustermap by {groupy} × {groupx}")
            plot_gene_dotplot_2d(
                adata, gene, groupx, groupy, outdir, system_tag,
                file_tag=file_tag, layer=layer, fallback_to_raw=fallback_to_raw, cmap=cmap
            )
            plot_gene_clustermap_2d(
                adata, gene, groupx, groupy, outdir, system_tag,
                file_tag=file_tag, layer=layer, fallback_to_raw=fallback_to_raw, cmap=cmap
            )
        except Exception as e:
            print(f"[{system_tag}] Skipping {gene} for groupby={groupby} due to: {e}")
            continue


# ---------------------------
# End of Holly's requested clustermap and dotplots  
# ---------------------------


def _gene_expr(adata, gene, layer="log1p_cpm"):
    """
    Return a 1D numpy array of expression for `gene`.
    Prefer a matrix layer aligned to adata.var_names; fallback to adata.raw.
    Case-insensitive gene match.
    """
    # case-insensitive match in current var_names (layer columns)
    lut = {g.upper(): i for i, g in enumerate(adata.var_names)}
    idx = lut.get(gene.upper(), None)

    if (layer in adata.layers) and (idx is not None):
        M = adata.layers[layer]
        col = M[:, idx]
        return col.toarray().ravel() if sparse.issparse(col) else np.asarray(col).ravel()

    # fallback to raw (case-insensitive there too)
    if adata.raw is not None:
        raw_names = pd.Index(adata.raw.var_names)
        lut_raw = {g.upper(): i for i, g in enumerate(raw_names)}
        ridx = lut_raw.get(gene.upper(), None)
        if ridx is not None:
            col = adata.raw.X[:, ridx]
            return col.toarray().ravel() if sparse.issparse(col) else np.asarray(col).ravel()

    # final fallback: zeros (and warn)
    print(f"⚠️  {gene}: not found in layer '{layer}' or adata.raw; using zeros")
    return np.zeros(adata.n_obs, dtype=float)




def plot_reverse_pseudotime_violin_by_day(adata, outdir, system_tag, file_tag=""):
    # identical style to your original, but uses reverse_pseudotime
    adata.obs["staging"] = pd.Categorical(
        adata.obs["staging"], categories=staging_order, ordered=True,
    )
    df = pd.DataFrame({
        "staging": adata.obs["staging"],
        "pseudotime": adata.obs["reverse_pseudotime"],
    })
    fig, ax = plt.subplots(figsize=(12, 4))
    sns.violinplot(
        data=df, x="staging", y="pseudotime",
        order=staging_order, inner="quartile", scale="width", ax=ax
    )
    ax.set_xticklabels(ax.get_xticklabels(), rotation=75, ha="center", fontsize=8)
    ax.set_xlabel("Collection day (staging)")
    ax.set_ylabel("Reverse pseudotime (1 − DPT)")
    ax.set_title(f"{system_tag}: reverse pseudotime by day")
    plt.tight_layout()
    fig.savefig(Path(outdir) / f"{system_tag}_reverse_pseudotime_violin_by_day{file_tag}.pdf",
                bbox_inches="tight")
    plt.close(fig)




def _add_pseudotime_bins(adata, pseudo_col="reverse_pseudotime", n_bins=20,
                         label_prefix="pt", method="quantile"):
    if pseudo_col not in adata.obs:
        raise KeyError(f"Missing pseudotime column: {pseudo_col}. Run DPT first.")

    x = adata.obs[pseudo_col].astype(float)

    if method == "quantile":
        cuts = pd.qcut(x, q=n_bins, duplicates="drop")
        cats = list(dict.fromkeys(map(str, cuts.cat.categories)))
    else:
        edges = np.linspace(x.min(), x.max(), n_bins + 1)
        cuts  = pd.cut(x, bins=edges, include_lowest=True)
        cats  = list(dict.fromkeys(map(str, cuts.cat.categories)))
        adata.uns[f"{pseudo_col}_bin_edges"] = edges  # save edges

    labels = [f"{label_prefix}{i:02d}" for i in range(len(cats))]
    lut = dict(zip(cats, labels))
    adata.obs["pt_bin"] = cuts.astype(str).map(lut)
    adata.obs["pt_bin"] = pd.Categorical(
        adata.obs["pt_bin"],
        categories=labels,  # keep full ordered list
        ordered=True
    )


def save_ptbin_by_staging_table(
    adata, outdir, system_tag,
    pseudo_col="dpt_pseudotime", n_bins=20, method="quantile",
    fname_tag=""
):
    """
    Ensures pt_bin exists (for the chosen pseudotime), builds a crosstab:
      rows = staging, cols = pt_bin (pt00..pt19), values = counts.
    Saves CSV + a quick heatmap PDF for QC. Returns the dataframe.
    """
    # make bins if missing / wrong length
    if "pt_bin" not in adata.obs or len(pd.Categorical(adata.obs["pt_bin"]).categories) != n_bins:
        _add_pseudotime_bins(adata, pseudo_col=pseudo_col, n_bins=n_bins, method=method)

    # enforce ordered staging categories that are actually present
    present = set(map(str, pd.Series(adata.obs["staging"]).dropna().unique()))
    ordered_present = [s for s in staging_order if s in present]
    adata.obs["staging"] = pd.Categorical(adata.obs["staging"], categories=ordered_present, ordered=True)

    # ordered pt labels
    pt_labels = [f"pt{i:02d}" for i in range(n_bins)]
    if isinstance(adata.obs["pt_bin"].dtype, pd.CategoricalDtype):
        adata.obs["pt_bin"] = adata.obs["pt_bin"].cat.set_categories(pt_labels, ordered=True)

    # build table
    tbl = pd.crosstab(adata.obs["staging"], adata.obs["pt_bin"]).reindex(index=ordered_present).fillna(0).astype(int)

    # save CSV
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    csv_path = outdir / f"{system_tag}_staging_by_ptbin_{pseudo_col}_nb{n_bins}{fname_tag}.csv"
    tbl.to_csv(csv_path)

    # optional quick heatmap (handy to spot imbalances)
    import matplotlib.pyplot as plt, seaborn as sns
    fig_h = max(4.5, 0.35 * max(1, len(ordered_present)))
    fig_w = max(6, 0.35 * n_bins + 3)
    plt.figure(figsize=(fig_w, fig_h))
    ax = sns.heatmap(tbl, cmap="viridis", cbar_kws={"label":"# cells"})
    ax.set_xlabel("pseudotime bin")
    ax.set_ylabel("staging")
    ax.set_title(f"{system_tag}: cells per staging × pt_bin ({pseudo_col})")
    plt.tight_layout()
    plt.savefig(outdir / f"{system_tag}_staging_by_ptbin_{pseudo_col}_nb{n_bins}{fname_tag}.pdf", bbox_inches="tight")
    plt.close()

    print(f"[{system_tag}] Saved crosstab to {csv_path}")
    return tbl





def plot_stacked_violin_per_gene_by_day(
    adata, genes, outdir, system_tag, file_tag="",
    groupby=("staging",),             # accepts str or list/tuple[str]
    n_pt_bins=8, pt_method="quantile"
):
    """
    Examples:
      groupby="staging"
      groupby="pt_bin"
      groupby=["celltype_new","staging"]
      groupby=["celltype_new","pt_bin"]
    """
    import os
    import numpy as np
    import pandas as pd
    import scanpy as sc
    import matplotlib.pyplot as plt
    import seaborn as sns
    from pathlib import Path

    # normalize groupby into list
    if isinstance(groupby, str):
        group_keys = [groupby]
    else:
        group_keys = list(groupby)

    print(f"[{system_tag}] stacked violin per gene × {group_keys}")
    os.makedirs(outdir, exist_ok=True)

    # --- fix staging order to only present categories (avoids mismatch) ---
    if "staging" in adata.obs:
        # use adata.obs["staging"].astype(str).unique() to tolerate non-categorical input
        present = set(map(str, pd.Series(adata.obs["staging"]).dropna().unique()))
        present_stages = [s for s in staging_order if s in present]
        adata.obs["staging"] = pd.Categorical(
            adata.obs["staging"], categories=present_stages, ordered=True
        )

    # --- add pseudotime bins if requested ---
    needs_pt = any(k == "pt_bin" for k in group_keys)
    if needs_pt:
        _add_pseudotime_bins(
            adata, pseudo_col="dpt_pseudotime",
            n_bins=n_pt_bins, method=pt_method
        )
        if isinstance(adata.obs["pt_bin"].dtype, pd.CategoricalDtype):
            present_bins = [b for b in adata.obs["pt_bin"].cat.categories
                            if (adata.obs["pt_bin"] == b).any()]
            adata.obs["pt_bin"] = adata.obs["pt_bin"].cat.set_categories(
                present_bins, ordered=True
            )

    # helper: categories order only for SINGLE categorical key
    def _single_categories_order(key):
        if key not in adata.obs:
            return None
        ser = adata.obs[key]
        if isinstance(ser.dtype, pd.CategoricalDtype):
            cats = [c for c in ser.cat.categories if (ser == c).any()]
            return cats
        return None

    for gene in genes:
        print(f"[{system_tag}]  • {gene} grouped by {group_keys}")

        # one-gene, log1p-CPM view from raw (keeps your normalization consistent)
        ad_plot = norm_raw_view(adata, [gene])

        # align needed obs columns onto the view
        for col in group_keys:
            if col in adata.obs:
                ad_plot.obs[col] = adata.obs.loc[ad_plot.obs_names, col]

        # If multi-key, sanitize rows and build a combined column
        categories_order = None
        combo_col = None
        df_plot = None

        if len(group_keys) == 1:
            categories_order = _single_categories_order(group_keys[0])
            groupby_arg = group_keys[0]
        else:
            ok = np.ones(ad_plot.n_obs, dtype=bool)
            for k in group_keys:
                ok &= ad_plot.obs[k].notna()
            if (~ok).any():
                print(f"[{system_tag}]   • dropping {(~ok).sum()} cells with missing values in {group_keys}")
            ad_plot = ad_plot[ok].copy()
            # make a combined grouping col (stable and explicit)
            combo_col = "__groupcombo__"
            ad_plot.obs[combo_col] = ad_plot.obs[group_keys].astype(str).agg(" | ".join, axis=1)
            groupby_arg = combo_col

        by_tag = "_x_".join(group_keys)
        save_path = Path(outdir) / f"_{system_tag}_stacked_violin_{gene}_by_{by_tag}{file_tag}.pdf"

        # --- try Scanpy's stacked_violin first (fast path) ---
        try:
            plot_kwargs = dict(
                adata=ad_plot,
                var_names={gene: [gene]},
                groupby=groupby_arg,
                use_raw=False,
                swap_axes=False,
                dendrogram=False,
                show=False,
                save=str(save_path.name),  # scanpy joins with sc.settings.figdir
            )
            if categories_order is not None:
                plot_kwargs["categories_order"] = categories_order

            # make sure Scanpy saves into outdir
            old_figdir = sc.settings.figdir
            sc.settings.figdir = str(outdir)
            sc.pl.stacked_violin(**plot_kwargs)
            sc.settings.figdir = old_figdir
            continue  # success → next gene

        except Exception as e:
            print(f"[{system_tag}]   • stacked_violin fallback (Seaborn) due to: {e}")

        # --- fallback: Seaborn violin (one gene) ---
        # ad_plot.X is (n_cells, 1) with our gene
        X = ad_plot.X
        if hasattr(X, "toarray"):
            X = X.toarray()
        expr = np.asarray(X).ravel()

        grp = ad_plot.obs[groupby_arg].astype(str).values
        df_plot = pd.DataFrame({"group": grp, "expr": expr})

        # order: single-key → categories_order; multi-key → observed order
        if categories_order is not None:
            order = [str(c) for c in categories_order]
        else:
            # preserve appearance order while stable-sorting for readability
            order = list(pd.Series(df_plot["group"]).astype(str).dropna().unique())

        plt.figure(figsize=(max(8, 0.45*len(order)+4), 4))
        ax = sns.violinplot(
            data=df_plot, x="group", y="expr",
            order=order, inner="quartile", scale="width", cut=0
        )
        ax.set_xticklabels(ax.get_xticklabels(), rotation=75, ha="center", fontsize=8)
        ax.set_xlabel(" | ".join(group_keys) if len(group_keys) > 1 else group_keys[0])
        ax.set_ylabel(f"{gene} (log1p-CPM from raw)")
        ax.set_title(f"{system_tag}: {gene} by {' × '.join(group_keys)}")
        plt.tight_layout()
        plt.savefig(save_path, bbox_inches="tight")
        plt.close()






def plot_expression_dot_by_day(adata, genes, outdir, system_tag, file_tag=""):
    print(f"[{system_tag}] plot_expression_dot_by_day: saving to {outdir}")

    # choose the matrix we’ll plot from
    genes_in_var = all(g.upper() in {v.upper() for v in adata.var_names} for g in genes)
    if genes_in_var:
        ad_plot = adata
        layer = "log1p_cpm"     # we created this to mirror raw but normalized
    else:
        ad_plot = norm_raw_view(adata, genes)  # X already holds log1p-CPM for these genes
        layer = None

    # make the staging column consistent on the object we are plotting
    ad_plot.obs["staging"] = pd.Categorical(
        ad_plot.obs["staging"], categories=staging_order, ordered=True
    )
    ad_plot.obs["staging"] = ad_plot.obs["staging"].cat.remove_unused_categories()

    # order must match the categories on *ad_plot*
    cats_present = list(ad_plot.obs["staging"].cat.categories)

    sc.pl.dotplot(
        ad_plot,
        var_names=genes,
        groupby="staging",
        dendrogram=False,
        categories_order=cats_present,
        standard_scale="var",
        use_raw=False,
        layer=layer,
        save=f"_{system_tag}_dotplot_by_day{file_tag}.pdf",
    )
    print(f"[{system_tag}] plot_expression_dot_by_day done\n")



def plot_violin_by_day(adata, genes, outdir, system_tag, file_tag=""):
    print(f"[{system_tag}] plot_violin_by_day (one file per gene): saving to {outdir}")
    os.makedirs(outdir, exist_ok=True)

    # ordered categories actually present
    adata.obs["staging"] = pd.Categorical(adata.obs["staging"],
                                          categories=staging_order, ordered=True)
    cats_present = [s for s in staging_order if (adata.obs["staging"] == s).any()]

    for gene in genes:
        # choose data source
        if gene.upper() in {g.upper() for g in adata.var_names}:
            ad_plot, layer = adata, "log1p_cpm"      # genes are in var_names → use your layer
        else:
            ad_plot, layer = norm_raw_view(adata, [gene]), None  # tiny raw-normalized view

        ax = sc.pl.violin(
            ad_plot,
            keys=gene,
            groupby="staging",
            order=cats_present,     # ensure order matches present categories
            use_raw=False,          # we’re always reading from X or the layer
            layer=layer,            # None for the raw-view; "log1p_cpm" on the main adata
            stripplot=False,
            jitter=False,
            multi_panel=False,
            rotation=90,
            show=False,
            save=f"_{system_tag}_violin_by_day_{gene}{file_tag}.pdf",
        )

        # clean legend + labels if you’re not letting scanpy save for you
        if hasattr(ax, "get_legend") and ax.get_legend() is not None:
            ax.get_legend().remove()



def plot_expr_vs_day_scatter(adata, genes, outdir, system_tag):
    print(f"[{system_tag}] plot_expr_vs_day_scatter: saving to {outdir}")
    os.makedirs(outdir, exist_ok=True)

    # turn staging into an ordered categorical (if you haven't already)
    adata.obs["staging"] = pd.Categorical(
        adata.obs["staging"],
        categories=staging_order,
        ordered=True,
    )

    # get a numeric code for each stage:
    adata.obs["stage_code"] = adata.obs["staging"].cat.codes

    for gene in genes:
        # pull expression out of raw
        mat = adata.raw[:, gene].X
        expr = mat.toarray().ravel() if sparse.issparse(mat) else mat.ravel()

        df = pd.DataFrame({
            "stage_code":    adata.obs["stage_code"],
            "expr":          expr,
            "celltype_new":  adata.obs["celltype_new"],
        })

        plt.figure(figsize=(6,4))
        sns.scatterplot(
            data=df,
            x="stage_code",      # numeric codes 0,1,2… for E8.0, E8.25, etc.
            y="expr",
            hue="celltype_new",
            alpha=0.4,
            s=10,
            legend=False,
        )

        # replace x‐axis ticks with actual stage names and rotate nicely
        ax = plt.gca()
        ax.set_xticks(range(len(staging_order)))
        ax.set_xticklabels(staging_order) 
        # now rotate & align
        plt.setp(ax.get_xticklabels(), rotation=75, ha="center", fontsize=8)

        plt.xlabel("Collection day (staging)")
        plt.ylabel(f"{gene} expression (raw count)")
        plt.title(f"{system_tag}: {gene} per cell by day")
        plt.tight_layout()
        plt.savefig(Path(outdir)/f"{system_tag}_scatter_{gene}_by_day.pdf")
        plt.close()
    print(f"[{system_tag}] plot_expr_vs_day_scatter done\n")


signaling_pathways = {
    'Hedgehog': ['GLI1', 'PTCH1', 'HHIP', 'FOXF1', 'FOXF2'],
    'RA': ['RARB', 'HOXA1', 'HOXA2', 'HOXA3', 'HOXB1', 'HOXB2', 'HOXB3'],
    'Wnt': ['AXIN1', 'LEF1', 'AXIN2', 'MYC', 'CCND1',  'DKK1'],
    'FGF': ['SPRY1', 'SPRY2',  'DUSP6', 'DUSP14', 'ETV4', 'ETV5', 'FOS', 'JUNB', 'MYC'],   
    'Notch': ['HES1', 'HEY1', 'HEY2', 'NOTCH3'],
    'BMP': ['ID1', 'ID2', 'ID3', 'ID4',  'SMAD7', 'MSX1',  'MSX2','BAMBI', 'BMPER'],    
    'Ribosomal': [
        # Large ribosomal subunit (60S) proteins
        'RPL3', 'RPL4', 'RPL5', 'RPL6', 'RPL7', 'RPL7A', 'RPL8', 'RPL9', 'RPL10', 'RPL10A',
        'RPL11', 'RPL12', 'RPL13', 'RPL13A', 'RPL14', 'RPL15', 'RPL17', 'RPL18', 'RPL18A', 'RPL19',
        'RPL21', 'RPL22', 'RPL23', 'RPL23A', 'RPL24', 'RPL26', 'RPL27', 'RPL27A', 'RPL28', 'RPL29',
        'RPL30', 'RPL31', 'RPL32', 'RPL34', 'RPL35', 'RPL35A', 'RPL36', 'RPL36A', 'RPL37', 'RPL37A',
        'RPL38', 'RPL39', 'RPL40', 'RPL41', 'RPLP0', 'RPLP1', 'RPLP2',
        # Small ribosomal subunit (40S) proteins  
        'RPS2', 'RPS3', 'RPS3A', 'RPS4X', 'RPS4Y1', 'RPS5', 'RPS6', 'RPS7', 'RPS8', 'RPS9',
        'RPS10', 'RPS11', 'RPS12', 'RPS13', 'RPS14', 'RPS15', 'RPS15A', 'RPS16', 'RPS17', 'RPS18',
        'RPS19', 'RPS20', 'RPS21', 'RPS23', 'RPS24', 'RPS25', 'RPS26', 'RPS27', 'RPS27A', 'RPS28', 
        'RPS29', 'RPSA'
    ]
}



def compute_signaling_scores(adata, signaling_pathways, target_sum=1_000_000):
    if adata.raw is None:
        raise ValueError("adata.raw is None; cannot score pathways")

    # work on a full-gene object from raw
    raw = adata.raw.to_adata().copy()
    # normalize + log1p on raw.X (counts) so we can set use_raw=False
    sc.pp.normalize_total(raw, target_sum=target_sum)
    sc.pp.log1p(raw)

    # case-insensitive lookup
    raw_names = pd.Index(raw.var_names)
    lut = {g.upper(): g for g in raw_names}

    for pathway, genes in signaling_pathways.items():
        used = [lut[g.upper()] for g in genes if g.upper() in lut]
        if not used:
            print(f"⚠️ {pathway}: 0/{len(genes)} genes present in raw; skipping")
            continue

        sc.tl.score_genes(
            raw,
            gene_list=used,
            score_name=f"{pathway}_score",
            use_raw=False,     # we normalized/logged raw.X just above
        )
        # copy back to main adata (same cells / order)
        adata.obs[f"{pathway}_score"] = raw.obs[f"{pathway}_score"].values
        print(f"✓ {pathway}: {len(used)}/{len(genes)} genes scored")





def plot_signaling_scores_vs_pseudotime(
    adata, outdir, system_tag, pseudo_col="reverse_pseudotime", file_tag=""
):
    """
    Binned mean curves of pathway scores vs pseudotime.
    Writes:
      ribosomal_signaling_{pseudo_col}{file_tag}.pdf
      other_signaling_{pseudo_col}{file_tag}.pdf
    """
    os.makedirs(outdir, exist_ok=True)
    if pseudo_col not in adata.obs:
        raise KeyError(f"Missing pseudotime column: {pseudo_col}")

    # choose pathways by what was actually computed
    scored = [c[:-6] for c in adata.obs.columns if c.endswith("_score")]
    if not scored:
        print("No *_score columns found; did you call compute_signaling_scores?")
        return

    # nice colors (no need to be perfect)
    palette = {
        "Hedgehog": "#e91e63", "RA": "#3498db", "Wnt": "#9b59b6",
        "FGF": "#f39c12", "Notch": "#27ae60", "BMP": "#ff6b35",
        "Ribosomal": "#1abc9c",
    }

    # common binning
    pt = adata.obs[pseudo_col].dropna().values
    bins = np.linspace(pt.min(), pt.max(), 60)
    centers = (bins[:-1] + bins[1:]) / 2

    def _binned_mean(series):
        df = adata.obs[[pseudo_col]].copy()
        df["y"] = series.values
        df = df.dropna()  # drop rows where either pseudotime or y is NaN
        digitized = np.digitize(df[pseudo_col].values, bins)
        xs, means = [], []
        for i in range(1, len(bins)):
            vals = df["y"].values[digitized == i]
            if vals.size > 20:     # require enough cells per bin
                means.append(vals.mean())
                xs.append(centers[i-1])
        return xs, means


    # 2a) Ribosomal only
    plt.figure(figsize=(8, 5))
    if "Ribosomal" in scored:
        x, y = _binned_mean(adata.obs["Ribosomal_score"])
        if x:
            plt.plot(x, y, lw=3, color=palette["Ribosomal"], label="Ribosomal")
    plt.xlabel(pseudo_col); plt.ylabel("Pathway score")
    plt.title(f"{system_tag}: Ribosomal vs pseudotime")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(Path(outdir)/f"ribosomal_signaling_{pseudo_col}{file_tag}.pdf", bbox_inches="tight")
    plt.close()

    # 2b) All other pathways
    plt.figure(figsize=(9, 6))
    for p in scored:
        if p == "Ribosomal": 
            continue
        x, y = _binned_mean(adata.obs[f"{p}_score"])
        if x:
            plt.plot(x, y, lw=2.2, color=palette.get(p, None), label=p)
    plt.xlabel(pseudo_col); plt.ylabel("Pathway score")
    plt.title(f"{system_tag}: Other signaling vs pseudotime")
    plt.grid(alpha=0.25); plt.legend(bbox_to_anchor=(1.02,1), loc="upper left")
    plt.tight_layout()
    plt.savefig(Path(outdir)/f"other_signaling_{pseudo_col}{file_tag}.pdf", bbox_inches="tight")
    plt.close()

def plot_signaling_scores_by_staging(
    adata, outdir, system_tag, file_tag=""
):
    """
    Line plots of mean pathway scores across collection days (staging).
    Writes:
      ribosomal_signaling_by_staging{file_tag}.pdf
      other_signaling_by_staging{file_tag}.pdf
    """
    os.makedirs(outdir, exist_ok=True)

    if "staging" not in adata.obs:
        raise KeyError("Missing obs['staging'].")

    # make staging ordered and drop unused levels
    adata.obs["staging"] = pd.Categorical(
        adata.obs["staging"], categories=staging_order, ordered=True
    )
    adata.obs["staging"] = adata.obs["staging"].cat.remove_unused_categories()
    stages = list(adata.obs["staging"].cat.categories)

    # which pathway scores are present?
    scored = [c[:-6] for c in adata.obs.columns if c.endswith("_score")]
    if not scored:
        print("No *_score columns found; did you call compute_signaling_scores?")
        return

    # helper: mean per stage (optionally filter stages with very few cells)
    counts = adata.obs["staging"].value_counts().reindex(stages).fillna(0).astype(int)
    keep_stages = [s for s in stages if counts.loc[s] > 0]

    def _mean_per_stage(series):
        df = pd.DataFrame({"staging": adata.obs["staging"], "y": series.values})
        g = df.groupby("staging", observed=False)["y"].mean().reindex(keep_stages)
        return keep_stages, g.values

    palette = {
        "Hedgehog": "#e91e63", "RA": "#3498db", "Wnt": "#9b59b6",
        "FGF": "#f39c12", "Notch": "#27ae60", "BMP": "#ff6b35",
        "Ribosomal": "#1abc9c",
    }

    # 1) Ribosomal alone
    if "Ribosomal" in scored:
        xlabels, y = _mean_per_stage(adata.obs["Ribosomal_score"])
        plt.figure(figsize=(max(8, 0.45*len(xlabels)+4), 5))
        plt.plot(range(len(xlabels)), y, lw=3, color=palette["Ribosomal"], label="Ribosomal")
        plt.xticks(range(len(xlabels)), xlabels, rotation=75, ha="center")
        plt.xlabel("staging"); plt.ylabel("Pathway score")
        plt.title(f"{system_tag}: Ribosomal vs staging")
        plt.grid(alpha=0.25); plt.tight_layout()
        plt.savefig(Path(outdir)/f"ribosomal_signaling_by_staging{file_tag}.pdf", bbox_inches="tight")
        plt.close()

    # 2) All other pathways
    plt.figure(figsize=(max(9, 0.5*len(keep_stages)+4), 6))
    for p in scored:
        if p == "Ribosomal":
            continue
        xlabels, y = _mean_per_stage(adata.obs[f"{p}_score"])
        plt.plot(range(len(xlabels)), y, lw=2.2, label=p, color=palette.get(p, None))
    plt.xticks(range(len(xlabels)), xlabels, rotation=75, ha="center")
    plt.xlabel("staging"); plt.ylabel("Pathway score")
    plt.title(f"{system_tag}: Other signaling vs staging")
    plt.grid(alpha=0.25); plt.legend(bbox_to_anchor=(1.02,1), loc="upper left")
    plt.tight_layout()
    plt.savefig(Path(outdir)/f"other_signaling_by_staging{file_tag}.pdf", bbox_inches="tight")
    plt.close()

def plot_hedgehog_vs_others_by_staging(
    adata, outdir, system_tag, file_tag=""
):
    """
    Pairwise plots of Hedgehog score vs each other pathway score across staging.
    Writes 6 PDFs:
      Hedgehog_vs_RA_by_staging{file_tag}.pdf
      Hedgehog_vs_Wnt_by_staging{file_tag}.pdf
      Hedgehog_vs_FGF_by_staging{file_tag}.pdf
      Hedgehog_vs_Notch_by_staging{file_tag}.pdf
      Hedgehog_vs_BMP_by_staging{file_tag}.pdf
      Hedgehog_vs_Ribosomal_by_staging{file_tag}.pdf
    """
    os.makedirs(outdir, exist_ok=True)

    if "staging" not in adata.obs:
        raise KeyError("Missing obs['staging'].")

    # staging ordered and cleaned
    adata.obs["staging"] = pd.Categorical(
        adata.obs["staging"], categories=staging_order, ordered=True
    )
    adata.obs["staging"] = adata.obs["staging"].cat.remove_unused_categories()
    stages = list(adata.obs["staging"].cat.categories)

    # make sure Hedgehog is scored
    if "Hedgehog_score" not in adata.obs:
        raise KeyError("Hedgehog_score missing — did you run compute_signaling_scores()?")

    # which other pathways are available
    scored = [c[:-6] for c in adata.obs.columns if c.endswith("_score")]
    others = [p for p in scored if p != "Hedgehog"]

    palette = {
        "Hedgehog": "#e91e63", "RA": "#3498db", "Wnt": "#9b59b6",
        "FGF": "#f39c12", "Notch": "#27ae60", "BMP": "#ff6b35",
        "Ribosomal": "#1abc9c",
    }

    def _mean_per_stage(series):
        df = pd.DataFrame({"staging": adata.obs["staging"], "y": series.values})
        g = df.groupby("staging", observed=False)["y"].mean().reindex(stages)
        return stages, g.values

    # loop over other pathways
    for p in others:
        xlabels, y_hh = _mean_per_stage(adata.obs["Hedgehog_score"])
        _, y_p = _mean_per_stage(adata.obs[f"{p}_score"])

        plt.figure(figsize=(max(8, 0.45*len(xlabels)+4), 5))
        plt.plot(range(len(xlabels)), y_hh, lw=2.5, color=palette["Hedgehog"], label="Hedgehog")
        plt.plot(range(len(xlabels)), y_p, lw=2.5, color=palette.get(p, None), label=p)
        plt.xticks(range(len(xlabels)), xlabels, rotation=75, ha="center")
        plt.xlabel("staging"); plt.ylabel("Pathway score")
        plt.title(f"{system_tag}: Hedgehog vs {p} by staging")
        plt.grid(alpha=0.25); plt.legend()
        plt.tight_layout()
        plt.savefig(Path(outdir)/f"Hedgehog_vs_{p}_by_staging{file_tag}.pdf", bbox_inches="tight")
        plt.close()




neuro_markers = [
    "PHOX2B", "ISL1", "SOX2", "HEY1", "SOX9",
    "NOTCH1", "NOTCH3", "MYC", "PLK1", "IFI16",
]

# Neurons ONLY 
def plot_neuro_markers_dotplots(adata, genes, outdir, system_tag, file_tag=""):
    """
    Makes two dotplots (genes on rows):
      1) grouped by celltype_new
      2) grouped by staging (ordered by staging_order)
    Saves:
      _{system_tag}_neuro_markers_by_celltype_new{file_tag}.pdf
      _{system_tag}_neuro_markers_by_staging{file_tag}.pdf
    """
    from pathlib import Path
    Path(outdir).mkdir(parents=True, exist_ok=True)

    # Case-insensitive presence check in current var_names
    var_upper_to_name = {v.upper(): v for v in adata.var_names}
    all_in_layer = all(g.upper() in var_upper_to_name for g in genes)

    if all_in_layer:
        ad_plot = adata
        layer = "log1p_cpm"  # uses your CPM+log1p layer aligned to adata.var_names
        # map to exact casing in var_names (keeps requested order)
        var_names = [var_upper_to_name[g.upper()] for g in genes]
    else:
        # Build a tiny view from raw with X = log1p(CPM) for just these genes
        ad_plot = norm_raw_view(adata, genes)
        layer = None
        # map to exact casing present in that tiny view, preserving requested order
        tiny_upper_to_name = {v.upper(): v for v in ad_plot.var_names}
        var_names = [tiny_upper_to_name[g.upper()] for g in genes if g.upper() in tiny_upper_to_name]

    # Ensure staging is ordered by your global staging_order and drop unused
    if "staging" in ad_plot.obs:
        ad_plot.obs["staging"] = pd.Categorical(
            ad_plot.obs["staging"],
            categories=[s for s in staging_order if s in set(map(str, ad_plot.obs["staging"].dropna().unique()))],
            ordered=True,
        )
        ad_plot.obs["staging"] = ad_plot.obs["staging"].cat.remove_unused_categories()
        stages_present = list(ad_plot.obs["staging"].cat.categories)
    else:
        stages_present = None

    # Helper to compute a clean category order
    def _order_for(key):
        if key not in ad_plot.obs:
            return None
        ser = ad_plot.obs[key]
        if isinstance(ser.dtype, pd.CategoricalDtype):
            # keep only used categories
            return [c for c in ser.cat.categories if (ser == c).any()]
        # for non-categorical, order by frequency descending
        counts = ser.value_counts()
        return list(counts.index)

    # 1) Dotplot by celltype_new
    if "celltype_new" in ad_plot.obs:
        ctype_order = _order_for("celltype_new")
        sc.pl.dotplot(
            ad_plot,
            var_names=var_names,
            groupby="celltype_new",
            dendrogram=False,
            categories_order=ctype_order,
            standard_scale="var",
            use_raw=False,
            layer=layer,
            save=f"_{system_tag}_neuro_markers_by_celltype_new{file_tag}.pdf",
        )
    else:
        print(f"[{system_tag}] Warning: obs['celltype_new'] missing; skipping dotplot by celltype_new")

    # 2) Dotplot by staging
    if stages_present:
        sc.pl.dotplot(
            ad_plot,
            var_names=var_names,
            groupby="staging",
            dendrogram=False,
            categories_order=stages_present,
            standard_scale="var",
            use_raw=False,
            layer=layer,
            save=f"_{system_tag}_neuro_markers_by_staging{file_tag}.pdf",
        )
    else:
        print(f"[{system_tag}] Warning: obs['staging'] missing; skipping dotplot by staging")





# ---------------------------------------------------------------------
# MAIN: loop over systems
# ---------------------------------------------------------------------
def main():
    # systems = ["Renal","Gut"]
    # systems = ["PNS_neurons"]
    systems = ["Lateral_plate_mesoderm"]
    genes = ["Gli1","Gli2","Gli3","Ptch1","Ptch2","Hhip","Foxf1","Foxf2"]   # Verify

    # run on full data for all comparisons/plots
    subsample_frac = 1.0
    tag = "_full" if subsample_frac == 1.0 else "_testp"

    groupbys = [
        ("staging", "celltype_new"),
        # ("pt_bin",  "celltype_new"),
    ]

    for system_tag in systems:
        h5_file = H5_ROOT / f"{system_tag}_adata_scale.h5ad"
        outdir = RESULTS_ROOT / RESULTS_SUBFOLDER / system_tag / "figures"
        outdir.mkdir(parents=True, exist_ok=True)
        sc.settings.figdir = str(outdir)

        print(f"\n=== processing {system_tag} ===")

        # ---------- load + prep ----------
        adata = initialize_data(
            str(h5_file), system_tag, str(CSV_META),
            min_cells_per_embryo=50, subsample_frac=subsample_frac, random_state=0
        )
        ensure_log1p_cpm_layer(adata)

        # Optional sanity: show the latest stage present for embryo 161
        try:
            s161 = (adata.obs.query("embryo_id == '161' and staging.notna()")
                        .staging.cat.codes.replace({-1: np.nan}).max())
            print(f"[{system_tag}] embryo 161 latest stage_code = {s161}")
        except Exception as _:
            pass


        # ---------- UMAP by sex (and other keys) ----------
        plot_umaps(
            adata,
            outdir=str(outdir),
            system_tag=system_tag,
            genes=genes
        )


        # ---------- pseudotime (normal) ----------
        # plot_pseudotime_expression(
        #     adata,
        #     genes=genes,
        #     outdir=str(outdir),
        #     system_tag=system_tag,
        #     file_tag=tag,
        #     rng_seed=0
        # )


        # Init not commented out 
        # # ---------- choose reverse anchor + write reverse_pseudotime ----------
        # # Also saves two small barplots showing which endpoint is later by staging / by DPT.
        # _ = choose_anchor_and_make_reverse_pseudotime(
        #     adata,
        #     outdir=str(outdir),
        #     system_tag=system_tag,
        #     candidates=("Airway goblet cells", "Lung cells (Eln+)"),
        #     late_quantile=0.75,
        #     compare_by=("staging", "pseudotime"),
        #     pick_on="staging",      # change to "pseudotime" if you prefer that criterion
        #     file_tag=tag



        # _ = choose_anchor_and_make_reverse_pseudotime(
        #     adata,
        #     outdir=str(outdir),
        #     system_tag=system_tag,
        #     candidates=("Atrial cardiomyocytes",),   # lock to atrial
        #     compare_by=("staging", "pseudotime"),
        #     pick_on="pseudotime",                    # decide winner by PT-late
        #     embryo_bias_id="161",                    # bias toward embryo 161's latest stage
        #     embryo_bias_weight=1.0,                  # small nudge; increase if ties are common
        #     file_tag=tag
        # )





        # # --- pick which pseudotime to use for bins/plots ---
        # # Gut → reverse pseudotime; Renal → regular DPT
        # pseudo_col = "reverse_pseudotime" if "reverse_pseudotime" in adata.obs else "dpt_pseudotime"

        # # build (or rebuild) pt_bin from the chosen pseudotime
        # _add_pseudotime_bins(
        #     adata,
        #     pseudo_col=pseudo_col,
        #     n_bins=20,
        #     method="quantile"
        # )

        # # save the cross‑table (#cells per staging × pt_bin) for this orientation
        # _ = save_ptbin_by_staging_table(
        #     adata,
        #     outdir=str(outdir),
        #     system_tag=system_tag,
        #     pseudo_col=pseudo_col,
        #     n_bins=20,
        #     method="quantile",
        #     fname_tag=tag,
        # )
        # ---------- pseudotime (normal) ----------



        # ---------- violin by day: normal + reverse ----------
        # if "reverse_pseudotime" in adata.obs:
        #     plot_reverse_pseudotime_violin_by_day(adata, str(outdir), system_tag, file_tag=tag)


        # ---------- optional: dot/extra QC/signaling ----------
        # plot_expression_dot_by_day(adata, genes, str(outdir), system_tag, file_tag=tag)
        # plot_violin_by_day(adata, genes, str(outdir), system_tag)
        # plot_expr_vs_day_scatter(adata, genes, str(outdir), system_tag)

        # ---------- NEW: 2-D per-gene dotplot + clustermap ----------
        for gb in groupbys:
            try:
                plot_dot_and_clustermap_per_gene(
                    adata, genes=genes, outdir=str(outdir), system_tag=system_tag,
                    file_tag=tag, groupby=gb, layer="log1p_cpm", fallback_to_raw=True,
                    cmap="RdBu_r"
                )
            except Exception as e:
                print(f"[{system_tag}] Skipping 2D plots for groupby={gb} due to: {e}")
         # ---------- End of NEW: 2-D per-gene dotplot + clustermap ----------

       
        # compute_signaling_scores(adata, signaling_pathways)
        # plot_signaling_scores_vs_pseudotime(
        #     adata,
        #     outdir=str(outdir),
        #     system_tag=system_tag,
        #     pseudo_col="reverse_pseudotime",
        #     file_tag=tag
        # )

        compute_signaling_scores(adata, signaling_pathways)
        plot_signaling_scores_by_staging(
            adata, outdir=str(outdir), system_tag=system_tag, file_tag=tag
        )
        plot_hedgehog_vs_others_by_staging(adata, outdir=str(outdir), system_tag=system_tag, file_tag=tag)


        del adata
        print(f"=== DONE  {system_tag} ===\n")

    print("\n✅ All systems complete!")


# def main():
#     # --- config for this run ---
#     system_tag = "PNS_neurons"
#     h5_file = H5_ROOT / f"{system_tag}_adata_scale.h5ad"
#     outdir = RESULTS_ROOT / RESULTS_SUBFOLDER / system_tag / "figures"
#     outdir.mkdir(parents=True, exist_ok=True)
#     sc.settings.figdir = str(outdir)

#     # neuro markers from Holly
#     neuro_markers = [
#         "PHOX2B", "ISL1", "SOX2", "HEY1", "SOX9",
#         "NOTCH1", "NOTCH3", "MYC", "PLK1", "IFI16",
#     ]

#     # choose test fraction or full
#     subsample_frac = 1.0   # quick test (1% of cells)
#     tag = "_full" if subsample_frac == 1.0 else "_testp"

#     print(f"\n=== processing {system_tag} — neuro dotplots only ({tag}) ===")

#     # 1) load + prep
#     adata = initialize_data(
#         str(h5_file), system_tag, str(CSV_META),
#         min_cells_per_embryo=50,
#         subsample_frac=subsample_frac,
#         random_state=0
#     )
#     ensure_log1p_cpm_layer(adata)

#     # 2) dotplots: genes on rows, grouped by celltype_new and staging
#     plot_neuro_markers_dotplots(
#         adata, genes=neuro_markers, outdir=str(outdir),
#         system_tag=system_tag, file_tag=tag
#     )

#     del adata
#     print(f"=== DONE  {system_tag} ===\n")
#     print("\n✅ Neuro dotplots complete!")


if __name__ == "__main__":
    main()