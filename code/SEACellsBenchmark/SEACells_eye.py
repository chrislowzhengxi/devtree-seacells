#!/usr/bin/env python3
import sys
# 1) force our SEACells clone at top of PYTHONPATH
sys.path.insert(0, "/project/xyang2/TOOLS/SEACells")

import warnings 
# warnings.filterwarnings("ignore", category=FutureWarning)
# warnings.filterwarnings("ignore", category=UserWarning)

import os, random, numpy as np, pandas as pd
import anndata as ad
import matplotlib.pyplot as plt
import re
import scanpy as sc
import seaborn as sns
import SEACells
from scipy import sparse
import scipy.sparse as sp

# reproducibility & plotting
random.seed(0)
np.random.seed(0)
sns.set_style('ticks')
plt.rcParams['figure.figsize'] = [4,4]
plt.rcParams['pdf.fonttype'] = 42   # Editable text in PDFs 
plt.rcParams['ps.fonttype'] = 42
plt.rcParams['figure.dpi'] = 300  # 300 dpi for publication quality

# I/O paths: – EDIT HERE if running on another machine (LOOP?)
INPUT_FILE  = '/project/imoskowitz/xyang2/SHH/Qiu_TimeLapse/results/raw_added/Renal_adata_scale.h5ad'
RESULTS_DIR = '/project/imoskowitz/xyang2/chrislowzhengxi/results/shendure_test_small'
SYSTEM_TAG = os.path.basename(INPUT_FILE).split("_")[0]        # e.g. "Eye"
FIG_DIR     = os.path.join(RESULTS_DIR, 'figures')
os.makedirs(FIG_DIR, exist_ok=True)

def plot_and_save(fig_name, **save_kw):
    outname = tag(fig_name)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, outname), **save_kw)
    plt.close()

def tag(fname: str, tag_txt: str = SYSTEM_TAG):
    """insert _<tag> before the file extension"""
    base, ext = os.path.splitext(fname)
    return f"{base}_{tag_txt}{ext}"


# Data information
# adata.shape                : (166852, 4000)
# adata.X.shape              : (166852, 4000)
# adata.raw.X.shape          : (166852, 24552)
# adata.raw.X is sparse?     : True
# list(adata.layers.keys()))
# adata.layers before setting: []
# 5% subsample of the original data: 8342 cells (166852 * 0.05)

# Wrong: 
# After `adata.raw = adata.copy()`
#   adata.raw.X.shape        : (166852, 4000)
#   adata.raw.X is sparse?   : False


# def initialize_data(
#     h5_path,
#     csv_meta="/project/imoskowitz/xyang2/chrislowzhengxi/data/df_cell_celltyp_new_merged.csv",
#     min_cells_per_embryo=50,
#     subsample_frac=None,     # e.g. 0.05 for 5%
#     random_state=0
# ):
#     """
#     • load raw .h5ad  
#     • safely map CSV metadata on cell_id (with diagnostics)  
#     • drop embryos with < min_cells_per_embryo  
#     • optional subsample  
#     • record & fix duplicate gene symbols
#     """
#     print("Loading expression:", h5_path, flush=True)
#     adata = sc.read_h5ad(h5_path)

#     # ensure var_names are strings for .var_names_make_unique()
#     if adata.raw is not None and isinstance(adata.raw.var.index, pd.CategoricalIndex):
#         adata.raw.var.index = adata.raw.var.index.astype(str)
#     if isinstance(adata.var_names, pd.CategoricalIndex):
#         adata.var_names = adata.var_names.astype(str)

#     # ---------- 1) load & inspect metadata ----------
#     print("Reading metadata CSV …", flush=True)
#     df_meta = pd.read_csv(
#         csv_meta,
#         usecols=["cell_id", "day", "embryo_id", "system", "celltype_new", "meta_group"]
#     )

#     # only keep metadata for this system
#     df_meta = df_meta[df_meta["system"] == SYSTEM_TAG]
#     print(f"Filtered to system {SYSTEM_TAG}: {len(df_meta):,} rows")
#     print(f"AnnData cells: {adata.n_obs:,}  |  metadata rows: {len(df_meta):,}", flush=True)
#     # └── Holly Q1: “Is the dimension of df_meta the same as the number of cells?”  
#       # We print both counts so you can compare.

#     # ---------- 2) detect & log duplicate cell_ids in metadata ----------
#     dupes = df_meta["cell_id"][df_meta["cell_id"].duplicated()].unique()
#     if dupes.size:
#         print(f"⚠️  {dupes.size} duplicate cell_id rows in metadata → keeping first", flush=True)
#         pd.Series(dupes).to_csv(
#             os.path.join(RESULTS_DIR, tag("duplicate_cell_ids_in_metadata.txt")),
#             index=False, header=False
#         )
#         df_meta = df_meta.drop_duplicates("cell_id", keep="first")
#     # └── Holly Q2: “Are all adata.obs.cell_id in df_meta.cell_id?”  
#       # Next we log any missing IDs.

#     missing = set(adata.obs["cell_id"]) - set(df_meta["cell_id"])
#     if missing:
#         print(f"⚠️  {len(missing):,} cells in AnnData lack metadata", flush=True)
#         pd.Series(sorted(missing)).to_csv(
#             os.path.join(RESULTS_DIR, tag("cells_missing_from_metadata.txt")),
#             index=False, header=False
#         )
#     # └── Holly Q3: “Why celltype_new isn’t identical?”  
#       # Any unmatched IDs become NaN below.

#     # # ---------- 3) map metadata columns one-by-one ----------
#     # meta_idx = df_meta.set_index("cell_id")
#     # for col in ["day", "embryo_id", "system", "meta_group", "celltype_new"]:
#     #     if col in meta_idx.columns:
#     #         adata.obs[col] = adata.obs["cell_id"].map(meta_idx[col])

#     # # report how many ended up NaN
#     # n_nan = adata.obs["celltype_new"].isna().sum()
#     # print(f"`celltype_new` mapped: {adata.n_obs - n_nan:,} annotated | {n_nan:,} missing", flush=True)
#     # # └── Now you can see exactly if & why `celltype_new` didn’t match: missing entries are logged.


#     # ---------- 3) map & verify celltype_new, then the other columns ----------
#     meta_idx = df_meta.set_index("cell_id")

#     # 3a) helper column for debugging
#     adata.obs["celltype_new_meta"] = adata.obs["cell_id"].map(meta_idx["celltype_new"])

#     # 3b) your official column
#     adata.obs["celltype_new"]      = adata.obs["cell_id"].map(meta_idx["celltype_new"])

#     # 3c) verify they are bit-for-bit identical
#     identical = (adata.obs["celltype_new_meta"] == adata.obs["celltype_new"]).all()
#     print(f"🔍 celltype_new_meta == celltype_new? {identical}", flush=True)

#     # 3d) drop the helper now that we've verified
#     adata.obs.drop(columns=["celltype_new_meta"], inplace=True)

#     # 3e) map the rest of the metadata
#     for col in ["day", "embryo_id", "system", "meta_group"]:
#         if col in meta_idx.columns:
#             adata.obs[col] = adata.obs["cell_id"].map(meta_idx[col])

#     # 3f) report how many ended up NaN (should be zero if identical==True)
#     n_nan = adata.obs["celltype_new"].isna().sum()
#     print(f"`celltype_new` mapped: {adata.n_obs - n_nan:,} annotated | {n_nan:,} missing", flush=True)



#     # ---------- 4) drop & log cells missing embryo_id ----------
#     to_log = adata.obs["cell_id"][adata.obs["embryo_id"].isna()]
#     if not to_log.empty:
#         to_log.to_csv(
#             os.path.join(RESULTS_DIR, tag("discarded_cells_missing_embryo_id.txt")),
#             index=False, header=False
#         )
#     adata = adata[~adata.obs["embryo_id"].isna()].copy()
#     print(f"after dropping missing embryos: {adata.n_obs} cells", flush=True)

#     # ---------- 5) drop & log tiny embryos ----------
#     emb_counts = adata.obs["embryo_id"].value_counts()
#     keep_emb = emb_counts[emb_counts >= min_cells_per_embryo].index
#     to_drop = adata.obs["cell_id"][~adata.obs["embryo_id"].isin(keep_emb)]
#     if not to_drop.empty:
#         to_drop.to_csv(
#             os.path.join(RESULTS_DIR, tag("discarded_cells_small_embryos.txt")),
#             index=False, header=False
#         )
#     adata = adata[adata.obs["embryo_id"].isin(keep_emb)].copy()
#     print(f"kept {len(keep_emb)} embryos ({adata.n_obs} cells)", flush=True)

#     # ---------- 6) optional subsample ----------
#     if subsample_frac:
#         sc.pp.subsample(adata, fraction=subsample_frac,
#                         random_state=random_state, copy=False)
#         print(f"subsampled to {adata.n_obs} cells", flush=True)

#     # ---------- 7) record & fix duplicate genes ----------
#     dupes = adata.raw.var.index[adata.raw.var.index.duplicated()] if adata.raw is not None else []
#     if len(dupes):
#         with open(os.path.join(RESULTS_DIR, tag("duplicate_genes.txt")), "w") as f:
#             f.write("\n".join(dupes))
#         print(f"{len(dupes)} duplicate gene IDs logged", flush=True)
#     adata.var_names_make_unique()

#     # ---------- 8) PCA trim or compute ----------
#     if "X_pca" in adata.obsm:
#         adata.obsm["X_pca"] = adata.obsm["X_pca"][:, :20]
#     else:
#         sc.tl.pca(adata, n_comps=20, random_state=random_state)

#     return adata


def initialize_data(
        h5_path,
        csv_meta="/project/imoskowitz/xyang2/chrislowzhengxi/data/df_cell_celltyp_new_merged.csv",
        min_cells_per_embryo=50,
        subsample_frac=None,
        random_state=0):
    """
    • load .h5ad
    • join metadata (adds `staging` + experimental_id)
    • drop tiny embryos / optional subsample
    • basic gene-duplication + PCA handling
    """
    # ------------------------------------------------------------------ #
    # 0) read expression
    # ------------------------------------------------------------------ #
    print("Loading expression:", h5_path, flush=True)
    adata = sc.read_h5ad(h5_path)

    # make var-names fixable
    if adata.raw is not None and isinstance(adata.raw.var.index, pd.CategoricalIndex):
        adata.raw.var.index = adata.raw.var.index.astype(str)
    if isinstance(adata.var_names, pd.CategoricalIndex):
        adata.var_names = adata.var_names.astype(str)

    # ------------------------------------------------------------------ #
    # 1) read metadata CSV and build the new `staging` column
    # ------------------------------------------------------------------ #
    req_cols = ["cell_id", "day", "somite_count", "embryo_id",
                "experimental_id", "system",
                "celltype_new", "meta_group"]
    print("Reading metadata CSV …", flush=True)
    df_meta = pd.read_csv(csv_meta, usecols=req_cols)

    # ---------- sanity: columns exist ---------- #
    missing_cols = [c for c in ["somite_count", "day"] if c not in df_meta.columns]
    if missing_cols:
        raise ValueError(f"Expected column(s) {missing_cols} not found in metadata!")

    # keep only the current system
    df_meta = df_meta[df_meta["system"] == SYSTEM_TAG].copy()

    # Should return True - sampling onto the current system
    print(f"Filtered to system {SYSTEM_TAG}: {len(df_meta):,} rows")
    print(f"AnnData cells: {adata.n_obs:,}  |  metadata rows: {len(df_meta):,}", flush=True)

    # ---- build `staging` ------------------------------------------------
    def _map_staging(row):
        d = row["day"]
        if d in ("E8", "E8.0-E8.5", "E8.5"):          # the earliest bucket
            try:
                scount = int(str(row["somite_count"]).split()[0])  # "8 somites" → 8
            except Exception:
                return np.nan
            if   0 <= scount <= 3:  return "E8.0"
            elif 4 <= scount <= 7:  return "E8.25"
            elif 8 <= scount <= 11: return "E8.5"
            else:                   return "E8.5+"      # fallback if >11
        else:
            return d

    df_meta["staging"] = df_meta.apply(_map_staging, axis=1)

    # quick debug print
    print("\nSample rows after staging mapping:")
    print(df_meta.loc[df_meta["day"].str.contains("E8"),      # only E8.* rows
                      ["day", "somite_count", "staging"]]
          .head(), "\n")

    # ------------------------------------------------------------------ #
    # 2) basic diagnostics: duplicates / missing cell-IDs
    # ------------------------------------------------------------------ #
    df_meta = df_meta.drop_duplicates("cell_id", keep="first")
    print(f"Metadata rows for {SYSTEM_TAG}: {len(df_meta):,}")
    print(f"AnnData cells: {adata.n_obs:,}")

    orphan_cells = set(adata.obs["cell_id"]) - set(df_meta["cell_id"])
    if orphan_cells:
        print(f"⚠️  {len(orphan_cells):,} AnnData cells lack metadata (dropped).")
    adata = adata[adata.obs["cell_id"].isin(df_meta["cell_id"])].copy()

    # ------------------------------------------------------------------ #
    # 3) map columns into .obs  (including NEW ones)
    # ------------------------------------------------------------------ #
    meta_idx = df_meta.set_index("cell_id")
    for col in ["day", "staging", "somite_count",
                "embryo_id", "experimental_id",
                "system", "meta_group", "celltype_new"]:
        adata.obs[col] = adata.obs["cell_id"].map(meta_idx[col])

    # confirm mapping
    print("Mapped columns → adata.obs:", ["staging", "somite_count", "experimental_id"])

    # ------------------------------------------------------------------ #
    # 4) drop embryos with < min_cells_per_embryo
    # ------------------------------------------------------------------ #
    good_embryos = adata.obs["embryo_id"].value_counts()
    keep_emb = good_embryos[good_embryos >= min_cells_per_embryo].index
    adata = adata[adata.obs["embryo_id"].isin(keep_emb)].copy()
    print(f"Kept {len(keep_emb)} embryos  |  {adata.n_obs:,} cells after filtering")

    # ------------------------------------------------------------------ #
    # 5) optional subsample
    # ------------------------------------------------------------------ #
    if subsample_frac:
        sc.pp.subsample(adata, fraction=subsample_frac,
                        random_state=random_state, copy=False)
        print(f"Subsampled to {adata.n_obs} cells")

    # ------------------------------------------------------------------ #
    # 6) duplicate genes -> unique; make sure 20-PC matrix exists
    # ------------------------------------------------------------------ #
    adata.var_names_make_unique()
    if "X_pca" in adata.obsm:
        adata.obsm["X_pca"] = adata.obsm["X_pca"][:, :20]
    else:
        sc.tl.pca(adata, n_comps=20, random_state=random_state)



    # # --------------------------Making sure-------------------------- #
    # # select only the rows that started life as the E8.0-E8.5 bucket:
    # obs = adata.obs.reset_index(drop=True)
    # mask = obs["day"] == "E8.0-E8.5"
    # # show the first 10 of them, with somite_count → staging
    # print(obs.loc[mask, ["day", "somite_count", "staging"]].head(10))
    return adata



def run_seacells(adata):
    print(f"Starting SEACells on {adata.n_obs} cells", flush=True)

    # 75 single cells per metacell was recommeneded in the tutorial
    # Lower density: 1 metacell per 100 cells
    n_SEACells = max(10, adata.n_obs // 100)   

    model = SEACells.core.SEACells(
        adata,
        build_kernel_on='X_pca',
        n_SEACells=n_SEACells,
        n_waypoint_eigs=10,
        convergence_epsilon=1e-5,
        use_sparse = True
    )
    model.construct_kernel_matrix()
    print("→ kernel built", flush=True)

    # (optional) small cluster‐gram of the first 200×200 block
    g = sns.clustermap(model.kernel_matrix[:200,:200].toarray(), cmap='viridis')
    # g.savefig(os.path.join(FIG_DIR, 'kernel_clustermap.pdf'), dpi=300)
    g.savefig(os.path.join(FIG_DIR, tag("kernel_clustermap.pdf")), dpi=300)
    plt.close(g.fig)


    model.initialize_archetypes();  print("→ archetypes initialized", flush=True)
    SEACells.plot.plot_initialization(
        adata, model,
        save_as=os.path.join(FIG_DIR, tag("init_umap.pdf"))       
    )
    model.fit(min_iter=10, max_iter=50)    
    for _ in range(5): model.step()

    # Sparse turn into numpy array (for use_sparse to work)
    if hasattr(model, 'A_') and sparse.issparse(model.A_):
        model.A_ = model.A_.toarray()

    print("→ SEACells converged", flush=True)

    model.plot_convergence(save_as=os.path.join(FIG_DIR, tag("convergence.pdf"))) # tag
    return model



def summarize_and_evaluate(adata, model):
    # # 1. non‐trivial assignment histogram
    # fig = plt.figure()
    # sns.histplot((model.A_.T > 0.1).sum(axis=1), bins=30)
    # fig.savefig(os.path.join(FIG_DIR, 'assignments_hist.pdf'))
    # plt.close(fig)

    # 1. non‐trivial assignment histogram
    plt.figure(figsize=(3,2))
    sns.histplot((model.A_.T > 0.1).sum(axis=1), bins=30)
    plt.title('Non-trivial (>0.1) assignments per cell')
    plt.xlabel('# Non-trivial SEACell Assignments')
    plt.ylabel('# Cells')
    plot_and_save("nontrivial_assignments_hist.pdf")   # nontrivial_assignments_hist_eye

    # 2. top-5 strongest assignments heatmap
    plt.figure(figsize=(3,2))
    b = np.partition(model.A_.T, -5, axis=1)
    sns.heatmap(np.sort(b[:, -5:], axis=1)[:, ::-1], cmap='viridis', vmin=0)
    plt.title('Top 5 strongest assignments')
    plt.xlabel('$n^{th}$ strongest assignment')
    plot_and_save("top5_assignment_heatmap.pdf")

    # 3. UMAP of cells only (no meta overlay)
    SEACells.plot.plot_2D(
        adata, key='X_umap', colour_metacells=False,
        save_as=os.path.join(FIG_DIR, tag("umap_cells.pdf"))
    )

    # 4. UMAP metacell plot
    SEACells.plot.plot_2D(
        adata, key='X_umap', colour_metacells=True,
        save_as=os.path.join(FIG_DIR, tag("umap_metacells.pdf"))
    )

    # 5. SEACell size distribution
    SEACells.plot.plot_SEACell_sizes(
        adata,
        bins=5,
        save_as=os.path.join(FIG_DIR, tag("seacell_sizes.pdf"))
    )

    # 5. purity / compactness / separation on your real labels
    metrics = [
        SEACells.evaluate.compute_celltype_purity,
        SEACells.evaluate.compactness,
        SEACells.evaluate.separation
    ]
    for fn in metrics:
        if fn is SEACells.evaluate.compute_celltype_purity:
            df = fn(adata, 'celltype_new')
            col = 'celltype_new_purity'
        elif fn is SEACells.evaluate.compactness:
            df = fn(adata, 'X_pca')
            col = 'compactness'
        else:  # separation
            df = fn(adata, 'X_pca', nth_nbr=1)
            col = 'separation'

        plt.figure(figsize=(4,4))
        sns.boxplot(data=df, y=col)
        plt.title(col.capitalize())
        sns.despine()
        plot_and_save(f"{col}.pdf")



def write_mappings(adata, model):
    df = adata.obs[['celltype_new']].copy()
    # rename for clarity
    df = df.rename(columns={'celltype_new': 'orig_cluster'})
    # add the metacell column (this is guaranteed 1D, same length as obs)
    df['metacell'] = model.get_hard_assignments()
    df.to_csv(os.path.join(RESULTS_DIR, tag('cell_to_metacell_map.csv')))


# # Go to Mohsen's code 
def write_metacell_composition(adata, results_dir, cluster_key='celltype_new'):
    """
    CSV: how many cells of each original cluster (e.g. 'celltype_new')
    fall into each SEACell (metacell).

    Param: 
        adata : AnnData object with 'SEACell' and cluster_key in .obs
        results_dir : path to write output
        cluster_key : column in adata.obs with original labels
    """
    df = pd.DataFrame({
        'metacell_id': adata.obs['SEACell'].values,
        'orig_cluster': adata.obs[cluster_key].values
    })

    comp_counts = (
        df
        .groupby('metacell_id')['orig_cluster']
        .value_counts()
        .unstack(fill_value=0)
        .sort_index()
    )

    out_csv = os.path.join(results_dir, tag('metacell_composition_counts.csv'))
    comp_counts.to_csv(out_csv)
    print(f"→ wrote metacell composition table to {out_csv}", flush=True)



def aggregate_metacells_by_timepoint(adata, results_dir, time_key='day'):
    # 1) check the day column exists
    if time_key not in adata.obs:
        raise KeyError(f"{time_key} not found in adata.obs")

    # 2) build your group‐labels: e.g. "E14.0_mc0", "E14.0_mc1", …
    df = adata.obs[[time_key, 'SEACell']].copy()
    df['group'] = df[time_key].astype(str) + '_mc' + df['SEACell'].astype(str)

    # 3) average the expression per group
    X = adata.X.toarray() if sp.issparse(adata.X) else adata.X
    expr_df = pd.DataFrame(X, index=adata.obs_names, columns=adata.var_names)
    expr_df['group'] = df['group'].values
    pseudobulk = expr_df.groupby('group').mean()

    # 4) rebuild obs from the group‐index
    #    index looks like ["E14.0_mc0", "E14.0_mc1", …]
    new_obs = pd.DataFrame(index=pseudobulk.index)
    parts   = new_obs.index.to_series().str.split('_mc', expand=True)
    new_obs[time_key] = parts[0]
    new_obs['SEACell'] = parts[1].astype(int)

    # 5) write out new AnnData
    new_adata = ad.AnnData(X=pseudobulk.values, obs=new_obs, var=adata.var)
    out_file = os.path.join(results_dir, tag(f'metacell_pseudobulk_by_{time_key}.h5ad'))
    new_adata.write(out_file)
    print(f"→ wrote aggregated metacell AnnData to {out_file}")


def main():
    assert os.path.exists(INPUT_FILE), f"{INPUT_FILE} not found"
    assert os.path.exists("/project/imoskowitz/xyang2/chrislowzhengxi/data/df_cell_celltyp_new_merged.csv"), \
        "metadata CSV not found"

    adata = initialize_data(INPUT_FILE)  # subsample 5% for quick test
    # model = run_seacells(adata)


    # adata.obs['SEACell'] = np.argmax(model.A_, axis=0)

    # # avoid h5ad write‐out conflict by giving the var index its own name
    # adata.var.index.name = "var_index"
    # if adata.raw is not None:
    #     adata.raw.var.index.name = "var_index"
        
    # # write out the metacell‐annotated AnnData (quick test)
    # adata.write(os.path.join(RESULTS_DIR, tag("with_SEACells_full.h5ad")))
    # print("→ wrote test AnnData", flush=True)

    # summarize_and_evaluate(adata, model)
    # write_mappings(adata, model)
    # write_metacell_composition(adata, RESULTS_DIR)
    # aggregate_metacells_by_timepoint(adata, RESULTS_DIR,
    #                              time_key="day")
    # print("→ all outputs generated", flush=True)

if __name__=='__main__':
    main()