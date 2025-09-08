import os
import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse
import seaborn as sns
import statsmodels.api as sm
import matplotlib.pyplot as plt
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
                "experimental_id","system","celltype_new","meta_group"]
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
    for c in ['day','staging','somite_count','embryo_id','experimental_id','system','meta_group','celltype_new']:
        adata.obs[c] = adata.obs['cell_id'].map(meta_idx[c])
    
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
    for key in ["day", "celltype_new"]:
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

def choose_anchor_and_make_reverse_pseudotime(
    adata,
    outdir,
    system_tag,
    candidates=("Airway goblet cells", "Lung cells (Eln+)"),
    late_quantile=0.75,
    compare_by=("staging", "pseudotime"),   # can be ("staging",), ("pseudotime",) or both
    pick_on="staging",                      # which comparison decides the anchor
    file_tag=""
):
    """
    1) Compare the two candidate bottom lineages by how enriched they are in *late* cells.
       - 'staging': top quartile of stage_code across all cells
       - 'pseudotime': top quartile of (unflipped) DPT values
    2) Pick the winner (by `pick_on`) and write reverse pseudotime = 1 - base_pt to obs.
    3) Save simple barplots of late proportions for transparency.

    Returns: dict with chosen anchors and proportions.
    """
    os.makedirs(outdir, exist_ok=True)
    _ensure_stage_code(adata)
    _ensure_dpt_if_missing(adata, system_tag)

    obs = adata.obs
    # If Gut was previously flipped, recover the original orientation if available
    base_pt = obs["dpt_pseudotime_raw"] if "dpt_pseudotime_raw" in obs else obs["dpt_pseudotime"]

    # thresholds for "late"
    out = {}
    if "staging" in compare_by:
        stage_thr = obs.loc[obs["stage_code"] >= 0, "stage_code"].quantile(late_quantile)
        rows = []
        for ct in candidates:
            m = (obs["celltype_new"] == ct) & (obs["stage_code"] >= 0)
            n_tot = int(m.sum())
            n_late = int(((obs["stage_code"] >= stage_thr) & m).sum())
            prop = n_late / n_tot if n_tot > 0 else np.nan
            rows.append({"candidate": ct, "n_total": n_tot, "n_late": n_late, "prop_late": prop})
        df_stage = pd.DataFrame(rows)
        # plot
        fig, ax = plt.subplots(figsize=(4.5, 3.2))
        sns.barplot(df_stage, x="candidate", y="prop_late", ax=ax)
        ax.set_ylabel(f"Late-stage proportion (≥ Q{int(late_quantile*100)})")
        ax.set_xlabel("")
        ax.set_title(f"{system_tag}: Late by staging")
        plt.xticks(rotation=20, ha="right")
        plt.tight_layout()
        plt.savefig(Path(outdir)/f"{system_tag}_anchor_compare_staging{file_tag}.pdf")
        plt.close(fig)
        out["staging"] = {"table": df_stage, "winner": df_stage.sort_values("prop_late").iloc[-1]["candidate"]}

    if "pseudotime" in compare_by:
        pt_thr = pd.Series(base_pt).quantile(late_quantile)
        rows = []
        for ct in candidates:
            m = (obs["celltype_new"] == ct) & (~pd.isna(base_pt))
            n_tot = int(m.sum())
            n_late = int(((base_pt >= pt_thr) & m).sum())
            prop = n_late / n_tot if n_tot > 0 else np.nan
            rows.append({"candidate": ct, "n_total": n_tot, "n_late": n_late, "prop_late": prop})
        df_pt = pd.DataFrame(rows)
        # plot
        fig, ax = plt.subplots(figsize=(4.5, 3.2))
        sns.barplot(df_pt, x="candidate", y="prop_late", ax=ax)
        ax.set_ylabel(f"Late-PT proportion (≥ Q{int(late_quantile*100)})")
        ax.set_xlabel("")
        ax.set_title(f"{system_tag}: Late by pseudotime")
        plt.xticks(rotation=20, ha="right")
        plt.tight_layout()
        plt.savefig(Path(outdir)/f"{system_tag}_anchor_compare_pseudotime{file_tag}.pdf")
        plt.close(fig)
        out["pseudotime"] = {"table": df_pt, "winner": df_pt.sort_values("prop_late").iloc[-1]["candidate"]}

    # pick the anchor by the requested criterion
    if pick_on not in out:
        raise ValueError(f"`pick_on`='{pick_on}' not in computed comparisons {list(out.keys())}")
    chosen_anchor = out[pick_on]["winner"]

    # compute reverse pseudotime (store; leave original untouched)
    adata.obs["reverse_pseudotime"] = 1.0 - np.asarray(base_pt, dtype=float)
    adata.uns["reverse_pt_anchor"] = {
        "candidates": list(candidates),
        "late_quantile": late_quantile,
        "compare_by": list(compare_by),
        "pick_on": pick_on,
        "chosen_anchor": chosen_anchor,
    }
    print(f"[{system_tag}] Reverse pseudotime written to obs['reverse_pseudotime'] "
          f"(anchor picked by {pick_on}: {chosen_anchor})")

    return out
# ---------------------------
# End of Reverse pseudotime chooser
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




def plot_pseudotime_violin_by_day(adata, outdir, system_tag, file_tag=""):
    adata.obs["staging"] = pd.Categorical(
        adata.obs["staging"], categories=staging_order, ordered=True,
    )

    df = pd.DataFrame({
        "staging": adata.obs["staging"],
        "pseudotime": adata.obs["dpt_pseudotime"],
    })

    fig, ax = plt.subplots(figsize=(12, 4))
    sns.violinplot(
        data=df, x="staging", y="pseudotime",
        order=staging_order, inner="quartile", scale="width", ax=ax
    )

    ax.set_xticklabels(ax.get_xticklabels(), rotation=75, ha="center", fontsize=8)
    ax.set_xlabel("Collection day (staging)")
    ax.set_ylabel("Pseudotime (DPT)")
    ax.set_title(f"{system_tag}: pseudotime by day")

    plt.tight_layout()
    fig.savefig(Path(outdir) / f"{system_tag}_pseudotime_violin_by_day{file_tag}.pdf", bbox_inches="tight")
    plt.close(fig)







# def plot_stacked_violin_per_gene_by_day(adata, genes, outdir, system_tag, file_tag=""):
#     print(f"[{system_tag}] stacked violin per gene × staging")
#     os.makedirs(outdir, exist_ok=True)

#     # ordered, drop unused
#     adata.obs["staging"] = pd.Categorical(
#         adata.obs["staging"], categories=staging_order, ordered=True
#     )
#     cats_present = [s for s in staging_order if (adata.obs["staging"] == s).any()]

#     for gene in genes:
#         print(f"[{system_tag}]  • {gene} grouped by staging")
#         # make a 1-gene, CPM+log1p normalized view from RAW
#         ad_plot = norm_raw_view(adata, [gene])
#         ad_plot.obs["staging"] = ad_plot.obs["staging"].cat.remove_unused_categories()
#         cats_here = [s for s in cats_present if s in ad_plot.obs["staging"].cat.categories]

#         sc.pl.stacked_violin(
#             ad_plot,
#             {gene: [gene]},
#             groupby="staging",
#             use_raw=False,              # we already normalized into X
#             swap_axes=False,
#             dendrogram=False,
#             categories_order=cats_here,
#             show=False,
#             save=f"_{system_tag}_stacked_violin_{gene}_by_staging{file_tag}.pdf",
#         )
def _add_pseudotime_bins(
    adata,
    pseudo_col="dpt_pseudotime",
    n_bins=8,
    label_prefix="pt",
    method="quantile",   # "quantile" or "uniform"
):
    """Add adata.obs['pt_bin'] as an ordered categorical."""
    if pseudo_col not in adata.obs:
        raise KeyError(f"Missing pseudotime column: {pseudo_col}. Run DPT first.")

    s = adata.obs[pseudo_col].astype(float)
    s = s.dropna()

    if method == "quantile":
        # equal-count bins
        cuts = pd.qcut(adata.obs[pseudo_col], q=n_bins, duplicates="drop")
        # build compact labels in rank order
        cats = list(dict.fromkeys(str(c) for c in cuts.cat.categories))
        labels = [f"{label_prefix}{i:02d}" for i in range(len(cats))]
        lut = {c: l for c, l in zip(cats, labels)}
        adata.obs["pt_bin"] = cuts.astype(str).map(lut)
    else:
        # equal-width bins
        cuts = pd.cut(adata.obs[pseudo_col], bins=n_bins, include_lowest=True)
        cats = list(dict.fromkeys(str(c) for c in cuts.cat.categories))
        labels = [f"{label_prefix}{i:02d}" for i in range(len(cats))]
        lut = {c: l for c, l in zip(cats, labels)}
        adata.obs["pt_bin"] = cuts.astype(str).map(lut)

    # ordered categorical
    adata.obs["pt_bin"] = pd.Categorical(
        adata.obs["pt_bin"],
        categories=[f"{label_prefix}{i:02d}" for i in range(len(set(adata.obs["pt_bin"].dropna())))],
        ordered=True,
    )


def plot_stacked_violin_per_gene_by_day(
    adata, genes, outdir, system_tag, file_tag="", groupby="staging",
    n_pt_bins=8, pt_method="quantile"
):
    """
    groupby options:
      - "staging": group by collection day (default, same as before)
      - "pt_bin": group by pseudotime bins
      - "staging_x_pt": combined 'staging | pt_bin' category
    """
    print(f"[{system_tag}] stacked violin per gene × {groupby}")
    os.makedirs(outdir, exist_ok=True)

    # staging as ordered categorical
    adata.obs["staging"] = pd.Categorical(
        adata.obs["staging"], categories=staging_order, ordered=True
    )

    # add pseudotime bins if needed
    if groupby in ("pt_bin", "staging_x_pt"):
        _add_pseudotime_bins(adata, pseudo_col="dpt_pseudotime",
                             n_bins=n_pt_bins, method=pt_method)

    # build categories and order
    if groupby == "staging":
        cats_present = [s for s in staging_order if (adata.obs["staging"] == s).any()]
        group_col = "staging"
        cats_order = cats_present

    elif groupby == "pt_bin":  
        # only keep non-null bins, preserve declared order
        bins = [b for b in adata.obs["pt_bin"].cat.categories if (adata.obs["pt_bin"] == b).any()]
        group_col = "pt_bin"
        cats_order = bins

    elif groupby == "staging_x_pt":
        # combined label like "E9.5 | pt03"
        valid = adata.obs["staging"].notna() & adata.obs["pt_bin"].notna()
        comb = adata.obs.loc[valid, ["staging", "pt_bin"]].astype(str).agg(" | ".join, axis=1)
        adata.obs.loc[valid, "staging_ptbin"] = comb.values
        # order: iterate staging outer, pt_bin inner
        cats_staging = [s for s in staging_order if (adata.obs["staging"] == s).any()]
        cats_pt = [b for b in adata.obs["pt_bin"].cat.categories if (adata.obs["pt_bin"] == b).any()]
        order = []
        for s in cats_staging:
            for b in cats_pt:
                label = f"{s} | {b}"
                if (adata.obs.get("staging_ptbin") == label).any():
                    order.append(label)
        adata.obs["staging_ptbin"] = pd.Categorical(adata.obs["staging_ptbin"], categories=order, ordered=True)
        group_col = "staging_ptbin"
        cats_order = order

    else:
        raise ValueError(f"Unknown groupby: {groupby}")

    # plot per gene
    for gene in genes:
        print(f"[{system_tag}]  • {gene} grouped by {group_col}")
        # one-gene, log1p-CPM view from raw (keeps your normalization consistent)
        ad_plot = norm_raw_view(adata, [gene])
        # align grouping columns onto the view
        for col in [c for c in (group_col, "staging", "pt_bin", "staging_ptbin") if c in adata.obs]:
            ad_plot.obs[col] = adata.obs.loc[ad_plot.obs_names, col]
        # drop unused categories in the view
        if group_col in ("staging", "pt_bin", "staging_ptbin"):
            ad_plot.obs[group_col] = ad_plot.obs[group_col].astype("category")
            # keep only categories that are present in the view
            present = [c for c in cats_order if (ad_plot.obs[group_col] == c).any()]
            ad_plot.obs[group_col] = ad_plot.obs[group_col].cat.set_categories(present, ordered=True)

        sc.pl.stacked_violin(
            ad_plot,
            var_names={gene: [gene]},
            groupby=group_col,      ### groupby=['celltype_new','pt_bin']. Rerun with groupby=['celltype_new','staging']
            use_raw=False,          
            swap_axes=False,
            dendrogram=False,
            categories_order=list(ad_plot.obs[group_col].cat.categories),
            show=False,
            save=f"_{system_tag}_stacked_violin_{gene}_by_{group_col}{file_tag}.pdf",
        )





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
    adata, outdir, system_tag, pseudo_col="dpt_pseudotime", file_tag=""
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


# ---------------------------------------------------------------------
# MAIN: loop over systems
# ---------------------------------------------------------------------
def main():
    # systems = ["Renal","Gut"]  
    # systems = ["Renal"]
    systems = ["Gut"]
    genes = ["Gli1","Ptch1","Hhip","Foxf1"]

    # --- test settings ---
    subsample_frac = 0.05
    tag = "_test5p" if subsample_frac == 0.05 else ""

    for system_tag in systems:
        h5_file = H5_ROOT / f"{system_tag}_adata_scale.h5ad"
        outdir = RESULTS_ROOT/RESULTS_SUBFOLDER/system_tag/"figures"
        outdir.mkdir(parents=True, exist_ok=True)
        sc.settings.figdir = str(outdir)

        print(f"\n=== processing {system_tag} ===")

        subsample_frac = 1
        tag = "_test5p" if subsample_frac == 0.05 else ""

        adata = initialize_data(str(h5_file), system_tag, str(CSV_META),
                                 min_cells_per_embryo=50, subsample_frac=subsample_frac, random_state=0)
        layer_name = ensure_log1p_cpm_layer(adata) 

        # -------- Plotting --------
        # # plot_system_qc(adata, str(outdir), system_tag, staging_order, TOP_N_EMBRYOS)
        # plot_umaps(adata, str(outdir), system_tag, genes)

        # # ------------------------------------------------------
        plot_pseudotime_expression(
            adata, genes=["Gli1","Ptch1","Hhip","Foxf1"],
            outdir=str(outdir), system_tag=system_tag, file_tag=tag, rng_seed=0
        )

        _ = choose_anchor_and_make_reverse_pseudotime(
            adata,
            outdir=str(outdir),
            system_tag=system_tag,
            candidates=("Airway goblet cells", "Lung cells (Eln+)"),
            late_quantile=0.75,
            compare_by=("staging", "pseudotime"),
            pick_on="staging",   # or "pseudotime" if you prefer that criterion
            file_tag=tag
        )
        
        # plot_pseudotime_violin_by_day(adata, str(outdir), system_tag, file_tag=tag)
        # # # ------------------------------------------------------

        # # # ------------------------------------------------------
        # plot_stacked_violin_per_gene_by_day(
        #     adata, genes=["Gli1","Ptch1","Hhip","Foxf1"], outdir=str(outdir), system_tag=system_tag, file_tag=tag
        # )
        # plot_expression_dot_by_day(adata, genes, outdir, system_tag, file_tag=tag)
        # # ------------------------------------------------------

        # plot_violin_by_day(adata, genes, outdir, system_tag)
        # plot_expr_vs_day_scatter(adata, genes, str(outdir), system_tag)

        # compute_signaling_scores(adata, signaling_pathways)
        # plot_signaling_scores_vs_pseudotime(
        #     adata,
        #     outdir=str(outdir),           
        #     system_tag=system_tag,
        #     pseudo_col="dpt_pseudotime",  
        #     file_tag=tag          
        # )
        # # -------- Plotting --------

        # # same as before
        # plot_stacked_violin_per_gene_by_day(adata, genes=["Gli1","Ptch1"], outdir=str(outdir),
        #                                     system_tag=system_tag, groupby="staging")

        # # pseudotime only
        # plot_stacked_violin_per_gene_by_day(adata, genes=["Gli1","Ptch1"], outdir=str(outdir),
        #                                     system_tag=system_tag, groupby="pt_bin", n_pt_bins=8)

        # # combined grid-like ordering: "E9.5 | pt03"
        # plot_stacked_violin_per_gene_by_day(adata, genes=["Gli1","Ptch1"], outdir=str(outdir),
        #                                     system_tag=system_tag, groupby="staging_x_pt", n_pt_bins=8)


        del adata
        print(f"=== DONE  {system_tag} ===\n")
    print("\n✅ All systems complete!")

if __name__ == "__main__":
    main()