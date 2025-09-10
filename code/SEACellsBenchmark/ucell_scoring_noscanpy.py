#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
UCell scoring for SHH genes (GLI1, PTCH1, HHIP), rank-based per cell.
No Scanpy dependency (uses anndata only).
Writes .obs['SHH_UCell_score'] and saves updated h5ad + per-cell CSV.

Run by calling the env's python directly; no module/conda activate needed:
  ENVROOT=/project/xyang2/software-packages/env/velocity_2025Feb_xy
  $ENVROOT/bin/python ucell_scoring_noscanpy.py
"""

import os
from pathlib import Path
import numpy as np
import pandas as pd
import anndata as ad
from scipy import sparse
import scanpy as sc

# rpy2 / R bridge
import rpy2.robjects as ro
from rpy2.robjects import numpy2ri, pandas2ri
from rpy2.robjects.packages import importr
from rpy2.robjects.vectors import StrVector, ListVector
numpy2ri.activate()
pandas2ri.activate()

# -------------------- CONFIG --------------------
H5_ROOT   = Path("/project/imoskowitz/xyang2/SHH/Qiu_TimeLapse/results/raw_added")
OUT_ROOT  = Path("/project/imoskowitz/xyang2/chrislowzhengxi/results/ucell")
# SYSTEMS = ["Lateral_plate_mesoderm", "Renal", "Gut", "PNS_neurons"]
SYSTEMS = ["Lateral_plate_mesoderm"]
SUBSAMPLE_FRAC = 1.0 # 0.01 for quick test, 1.0 for full
NCORES = int(os.environ.get("SLURM_CPUS_PER_TASK", "1"))
SHH_GENES = ["Gli1", "Ptch1", "Hhip"]
UCELL_INPUT_LAYER = os.environ.get("UCELL_INPUT_LAYER", None)  # e.g., "log1p_cpm"

FORCE_SYMBOL_COL = None
# ------------------------------------------------

def load_adata(h5_file: Path) -> ad.AnnData:
    print(f"[LOAD] {h5_file}")
    adata = ad.read_h5ad(str(h5_file))

    # Ensure string var names and uniqueness (important!)
    adata.var_names = adata.var_names.astype(str)
    if hasattr(adata, "var_names_make_unique"):
        adata.var_names_make_unique()

    # Ensure .raw exists & is normalized to plain strings, unique too
    if getattr(adata, "raw", None) is None:
        adata.raw = adata.copy()
    else:
        raw = adata.raw.to_adata()
        raw.var_names = raw.var_names.astype(str)
        if hasattr(raw, "var_names_make_unique"):
            raw.var_names_make_unique()
        adata.raw = raw

    return adata


def sanitize_anndata_for_h5ad(adata):
    """Fix var/raw.var string/unique and resolve index/column name collisions."""
    # Main
    adata.var_names = adata.var_names.astype(str)
    if hasattr(adata, "var_names_make_unique"):
        adata.var_names_make_unique()
    # If index name also appears as a column with different values, rename the column
    iname = adata.var.index.name
    if iname and (iname in adata.var.columns):
        same = (adata.var.index.astype(str).to_series().values ==
                adata.var[iname].astype(str).values)
        if not bool(np.all(same)):
            adata.var.rename(columns={iname: f"{iname}_col"}, inplace=True)
    # Clear index name to be safe
    adata.var.index.name = None

    # Raw
    if adata.raw is not None:
        raw = adata.raw.to_adata()
        raw.var_names = raw.var_names.astype(str)
        if hasattr(raw, "var_names_make_unique"):
            raw.var_names_make_unique()
        riname = raw.var.index.name
        if riname and (riname in raw.var.columns):
            same = (raw.var.index.astype(str).to_series().values ==
                    raw.var[riname].astype(str).values)
            if not bool(np.all(same)):
                raw.var.rename(columns={riname: f"{riname}_col"}, inplace=True)
        raw.var.index.name = None
        adata.raw = raw


def maybe_subsample(adata: ad.AnnData, fraction: float, seed: int = 0) -> ad.AnnData:
    if fraction is None or fraction >= 1.0:
        return adata
    n = adata.n_obs
    k = max(1, int(round(n * float(fraction))))
    rng = np.random.default_rng(seed)
    sel = rng.choice(n, size=k, replace=False)
    adata = adata[adata.obs_names[sel]].copy()
    print(f"[SUBSAMPLE] {n} → {adata.n_obs} cells (fraction={fraction})")
    return adata


def _matrix_from_adata_for_ucell(adata: ad.AnnData):
    """
    Return (X_T, gene_names, cell_names) for R/UCell: genes x cells.
    PREFER raw counts (adata.raw) -> adata.X -> layers (as last resort).
    """
    import numpy as np
    from scipy import sparse as sp

    # 1) Prefer raw (these files store the “true” gene space here)
    if adata.raw is not None:
        raw = adata.raw.to_adata()
        X = raw.X
        genes = list(raw.var_names.astype(str))
    else:
        # 2) Fall back to main matrix
        X = adata.X
        genes = list(adata.var_names.astype(str))

        # 3) Only if X is missing/empty, try layers (counts/raw_counts/etc.)
        if X is None or (sp.issparse(X) and X.nnz == 0) or (not sp.issparse(X) and np.all(X == 0)):
            for layer_name in ("counts", "raw_counts", "X_raw"):
                if hasattr(adata, "layers") and layer_name in adata.layers:
                    X = adata.layers[layer_name]
                    genes = list(adata.var_names.astype(str))
                    break

    if sp.issparse(X):
        X = X.toarray()
    X = np.asarray(X, dtype=np.float32)

    # R expects genes x cells
    X_T = X.T
    cells = list(adata.obs_names)
    return X_T, genes, cells



def _ensure_ucell_installed():
    try:
        return importr("UCell")
    except Exception as e:
        raise RuntimeError(
            "Could not import R package 'UCell'. This script expects the conda env "
            "at /project/xyang2/software-packages/env/velocity_2025Feb_xy to have R+UCell. "
            "Make sure PATH and R_HOME point to that env before running."
        ) from e


def _map_signature_genes_to_matrix_rows(adata, requested):
    raw = adata.raw.to_adata() if adata.raw is not None else adata
    rownames = pd.Index(raw.var_names)

    # 1) direct case-insensitive match to rownames
    lut = {g.upper(): g for g in rownames}
    used = [lut.get(g.upper()) for g in requested if g.upper() in lut]
    if used:
        return used

    # 2) symbol columns: prioritize gene_short_name
    candidate_cols = ["gene_short_name", "gene_symbol", "symbol", "SYMBOL",
                      "Gene", "gene", "gene_name", "features", "Feature", "GeneSymbol"]
    for c in candidate_cols:
        if c in raw.var.columns:
            col = raw.var[c].astype(str)
            # build symbol -> rowname map
            sym2row = {}
            for row, sym in zip(rownames, col):
                u = sym.upper()
                if u and u not in sym2row:
                    sym2row[u] = row
            used = [sym2row.get(g.upper()) for g in requested if g.upper() in sym2row]
            used = [u for u in used if u is not None]
            if used:
                return used
    return []


def _pick_rowlabels_for_ucell(adata):
    """
    Return a list of labels to use as M's rownames (prefer gene symbols when available).
    Tries, in order:
      1) FORCE_SYMBOL_COL if set and present
      2) common symbol-like columns
      3) var_names
    Ensures uniqueness (UCell wants unique rownames).
    """
    raw = adata.raw.to_adata() if adata.raw is not None else adata

    # 1) Forced column only if configured AND present
    if FORCE_SYMBOL_COL:
        if FORCE_SYMBOL_COL in raw.var.columns:
            col = FORCE_SYMBOL_COL
        else:
            print(f"[WARN] FORCE_SYMBOL_COL='{FORCE_SYMBOL_COL}' not found in raw.var; falling back.")
            col = None
    else:
        col = None

    # 2) If no forced column, try common symbol columns
    if col is None:
        for c in ["gene_short_name", "gene_symbol", "symbol", "SYMBOL",
                  "Gene", "gene", "gene_name", "features", "Feature", "GeneSymbol"]:
            if c in raw.var.columns:
                col = c
                break

    # 3) Build labels
    if col is not None:
        labels = raw.var[col].astype(str).fillna("")
        labels = labels.where(labels != "", other=raw.var_names.astype(str))
    else:
        # fallback: var_names
        labels = raw.var_names.astype(str)

    # De-duplicate to ensure uniqueness (UCell requirement)
    seen, uniq = set(), []
    for x in labels:
        if x not in seen:
            uniq.append(x); seen.add(x)
        else:
            i = 1
            y = f"{x}.{i}"
            while y in seen:
                i += 1
                y = f"{x}.{i}"
            uniq.append(y)
            seen.add(y)

    return uniq, col



def run_ucell_scores_shh(adata: ad.AnnData, shh_genes=SHH_GENES, ncores: int = 1, use_layer: str | None = None) -> pd.Series:
    from scipy import sparse as sp
    ucell = _ensure_ucell_installed()

    # --- choose matrix (cells x genes) ---
    if use_layer:
        if use_layer not in adata.layers:
            raise ValueError(f"Requested layer '{use_layer}' not found in adata.layers")
        X = adata.layers[use_layer]
        print(f"[INPUT] using layer='{use_layer}' (cells x genes={X.shape})")
        # IMPORTANT: labels must align to adata.var_names when using a layer
        rowlabels = list(map(str, adata.var_names))
        rowlabels_src = "adata.var_names"
    else:
        raw = adata.raw.to_adata() if adata.raw is not None else adata
        X = raw.X
        print(f"[INPUT] using {'adata.raw.X' if adata.raw is not None else 'adata.X'} (cells x genes={X.shape})")
        # When using raw/X, build labels from the same object
        # If you prefer symbol columns, call _pick_rowlabels_for_ucell on *raw*
        # (current helper looks at adata; simplest is to use raw.var_names directly)
        rowlabels = list(map(str, raw.var_names))
        rowlabels_src = "raw.var_names" if adata.raw is not None else "adata.var_names"

    cells = list(map(str, adata.obs_names))

    # Sanity: length must match gene dimension
    if X.shape[1] != len(rowlabels):
        raise RuntimeError(f"rowlabels length ({len(rowlabels)}) does not match X n_vars ({X.shape[1]}) from {rowlabels_src}")

    # ---- 2) Map requested genes to the EXACT case used in rowlabels ----
    lut = {}
    for g in rowlabels:
        up = g.upper()
        if up not in lut:
            lut[up] = g
    feats_exact = [lut[g.upper()] for g in shh_genes if g.upper() in lut]

    print(f"[UCELL] rownames source: {rowlabels_src}")
    print(f"[UCELL] SHH genes requested: {shh_genes}")
    print(f"[UCELL] SHH genes present  : {feats_exact} ({len(feats_exact)}/{len(shh_genes)})")
    if not feats_exact:
        print("⚠️  No requested SHH genes matched rownames — returning zeros.")
        return pd.Series(0.0, index=adata.obs_names, name="SHH_UCell_score")

    # ---- 3) Build M in R (genes x cells) WITHOUT densifying if sparse ----
    if sp.issparse(X):
        Xgc = X.T.tocoo(copy=False)  # genes x cells
        from rpy2.robjects.vectors import IntVector, FloatVector
        ro.globalenv["i"] = IntVector((Xgc.row + 1).astype(np.int32))
        ro.globalenv["j"] = IntVector((Xgc.col + 1).astype(np.int32))
        ro.globalenv["x"] = FloatVector(Xgc.data.astype(np.float32))
        ro.globalenv["dims"] = IntVector([Xgc.shape[0], Xgc.shape[1]])
        ro.globalenv["G"] = StrVector(rowlabels)
        ro.globalenv["C"] = StrVector(cells)
        ro.r("""
            suppressMessages(library(Matrix))
            M <- sparseMatrix(i=i, j=j, x=x, dims=dims)
            rownames(M) <- G; colnames(M) <- C
        """)
    else:
        X_T = np.asarray(X, dtype=np.float32).T
        ro.globalenv["M"] = ro.r["as.matrix"](X_T)
        ro.globalenv["G"] = StrVector(rowlabels)
        ro.globalenv["C"] = StrVector(cells)
        ro.r("rownames(M) <- G; colnames(M) <- C")


    # ---- 4) Safety on rownames & presence sanity ----
    ro.r("""
        rn <- rownames(M)
        if (is.null(rn)) stop('M rownames are NULL')
        if (anyNA(rn)) stop('M rownames contain NA')
        if (any(duplicated(rn))) rownames(M) <- make.unique(rn)
    """)
    ro.globalenv["FEATS"] = StrVector(feats_exact)
    ro.r('cat("Check presence:", sum(rownames(M) %in% FEATS), "of", length(FEATS), "present\\n")')
    ro.r('stopifnot(all(FEATS %in% rownames(M)))')

    # ---- 5) Precompute ranks and score (version-safe path) ----
    ro.r("ranks <- UCell::StoreRankings_UCell(M)")
    res = ucell.ScoreSignatures_UCell(
        **{'precalc.ranks': ro.globalenv["ranks"]},
        features=ListVector({"SHH": StrVector(feats_exact)}),
        ncores=int(ncores)
    )

    # ---- 6) Convert R -> pandas (1 col per signature, 1 row per cell) ----
    try:
        res_df_r = ro.r("as.data.frame")(res)
        res_pd = pandas2ri.rpy2py(res_df_r)
        colname = "SHH_UCell" if "SHH_UCell" in res_pd.columns else ("SHH" if "SHH" in res_pd.columns else res_pd.columns[0])
        vals = res_pd[colname].to_numpy()
    except Exception:
        arr = numpy2ri.rpy2py(res)
        cols = list(ro.r("colnames")(res))
        if "SHH_UCell" in cols:
            idx = cols.index("SHH_UCell"); vals = np.asarray(arr)[:, idx]
        elif "SHH" in cols:
            idx = cols.index("SHH"); vals = np.asarray(arr)[:, idx]
        else:
            vals = np.asarray(arr).ravel()

    if len(vals) != adata.n_obs:
        raise ValueError(f"UCell returned {len(vals)} scores, expected {adata.n_obs} (per cell).")

    # Optional guard against accidental all-zeros
    if not np.any(vals):
        raise RuntimeError("UCell returned all zeros. Check features/rownames or UCell version.")

    return pd.Series(vals, index=adata.obs_names, name="SHH_UCell_score")



def ensure_log1p_cpm_layer(adata, layer_name="log1p_cpm", target_sum=1_000_000):
    n_obs, n_vars = adata.n_obs, adata.n_vars
    make_sparse = sparse.issparse(adata.X)
    layer = sparse.csr_matrix((n_obs, n_vars), dtype=np.float32) if make_sparse else np.zeros((n_obs, n_vars), dtype=np.float32)

    raw = adata.raw.to_adata()
    common = adata.var_names.intersection(raw.var_names)
    if len(common) == 0:
        raise ValueError("No overlap between adata.var_names and adata.raw.var_names")

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

    sc.pp.normalize_total(adata, target_sum=target_sum, layer=layer_name)
    sc.pp.log1p(adata, layer=layer_name)
    return layer_name



def save_outputs(adata: ad.AnnData, out_dir: Path, system: str):
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = run_suffix()
    h5_out = out_dir / f"{system}{suffix}_adata_with_ucell.h5ad"
    csv_out = out_dir / f"{system}{suffix}_SHH_UCell_scores.csv"
    adata.write_h5ad(str(h5_out))
    adata.obs[["SHH_UCell_score"]].to_csv(csv_out, index=True)
    print(f"[SAVE] {h5_out}")
    print(f"[SAVE] {csv_out}")



# ============================
# System-aware auto plotting
# ============================
import matplotlib.pyplot as plt

# 1) Configure per-system label order (edit/extend as you learn them)
SYSTEM_ORDER = {
    "Lateral_plate_mesoderm": None,
    "Renal": [
        "Anterior intermediate mesoderm",
        "Posterior intermediate mesoderm",
        "Metanephric mesenchyme",
        "Nephron progenitors",
        "Ureteric bud",
        "Ureteric bud stalk",
        "Proximal tubule cells",
        "Ascending loop of Henle",
        "Distal convoluted tubule",
        "Connecting tubule",
        "Collecting duct principal cells",
        "Collecting duct intercalated cells",
        "Renal pericytes and mesangial cells",
        "Podocytes",
    ],
    # If you know the intended lineage paths for Gut / PNS_neurons, put them here:
    "Gut": None,          # None -> plot whatever labels exist, in df order
    "PNS_neurons": None,  # same
}

# 2) Optional per-system tree ID maps (only fill what you know)
TREE_IDS = {}

def _xticklabels_with_ids(system: str, labels: list[str]) -> list[str]:
    nid_map = TREE_IDS.get(system, {})
    out = []
    for lab in labels:
        nid = nid_map.get(lab)
        out.append(f"{lab}\nID {nid}" if nid is not None else lab)
    return out

def _run_suffix():
    s_layer = f"_{UCELL_INPUT_LAYER}" if UCELL_INPUT_LAYER else ""
    s_run   = "_test" if SUBSAMPLE_FRAC < 1.0 else ""
    return f"{s_layer}{s_run}"

def plot_shh_for_system(system: str, out_root: Path):
    qcdir = (out_root / system / "qc")
    suffix = _run_suffix()
    # csv = qcdir / f"{system}{suffix}_cardiac_gold_standard_summary.csv"

    csv_all = qcdir / f"{system}{suffix}_ALL_labels_summary.csv"
    csv_alt = qcdir / f"{system}{suffix}_cardiac_gold_standard_summary.csv"  # fallback (e.g., LPM if you still write it)
    csv = csv_all if csv_all.exists() else csv_alt
    if not csv.exists():
        print(f"[PLOT] Missing summary CSV for {system}: {csv}")
        return

    df = pd.read_csv(csv)
    # keep rows that actually have cells
    df = df[df["n_cells"].fillna(0) > 0].copy()
    if df.empty:
        print(f"[PLOT] No labeled rows with cells for {system} — skipping plots.")
        return

    # Decide display order
    desired = SYSTEM_ORDER.get(system)
    if desired:
        cats = [lab for lab in desired if lab in set(df["celltype_new"])]
        if not cats:
            print(f"[PLOT] None of desired labels present for {system}; plotting all labels as-is.")
            cats = list(df["celltype_new"])
        df["celltype_new"] = pd.Categorical(df["celltype_new"], categories=cats, ordered=True)
        df = df.sort_values("celltype_new")
    else:
        # no predefined order: keep input order (or sort by n_cells descending)
        # df = df.sort_values("n_cells", ascending=False)
        pass

    df = df.reset_index(drop=True)
    # Build tick labels (with IDs if configured)
    labels = df["celltype_new"].tolist()
    xticklabels = _xticklabels_with_ids(system, labels)

    # Guard: if all summary stats are zero, note it (your renal example)
    all_zero = (df[["median","mean","q90"]].fillna(0).to_numpy() == 0).all()
    if all_zero:
        print(f"[PLOT] All SHH summary stats are zero for {system}. Plots will be flat (check gene presence/layer).")

    # ---------- Fig 1: bars (median) + markers (mean, 90th)
    fig1, ax1 = plt.subplots(figsize=(8.8, 4.6))
    x = range(len(df))
    # ax1.bar(x, df["median"], width=0.6, label="Median",
    #         edgecolor="black", linewidth=0.7, color="#D0D7DE")
    ax1.bar(x, df["mean"], width=0.6, label="Mean",
        edgecolor="black", linewidth=0.7, color="#D0D7DE")
    ax1.plot(x, df["mean"], marker="o", linestyle="none",
             label="Mean", markersize=6, markeredgecolor="black", markerfacecolor="black")
    ax1.plot(x, df["q90"], marker="^", linestyle="none",
             label="90th pct", markersize=7, markeredgecolor="black", markerfacecolor="#F5A623")

    ax1.set_xticks(list(x)); ax1.set_xticklabels(xticklabels, rotation=30, ha="right")
    # ymax = float(df[["median","mean","q90"]].max().max())
    # ax1.set_ylim(0, max(1.0, ymax * 1.05))
    # ax1.set_ylabel("SHH UCell score"); ax1.set_title(f"{system} – SHH UCell score by label")
    ymax = float(df[["mean", "q90"]].max().max()) + 0.1
    ax1.set_ylim(0, ymax)
    ax1.set_ylabel("SHH UCell score"); ax1.set_title(f"{system} – SHH UCell score by label (mean)")

    ax1.legend(frameon=False, ncol=3); fig1.tight_layout()
    fig1.savefig(qcdir / f"{system}{suffix}_SHH_UCell_summary.pdf", dpi=300)
    fig1.savefig(qcdir / f"{system}{suffix}_SHH_UCell_summary.png", dpi=300)

    # ---------- Fig 2: lineage-style line (using chosen order)
    fig2, ax2 = plt.subplots(figsize=(8.2, 3.9))
    # ax2.plot(list(x), df["median"], marker="o", linewidth=2)
    ax2.plot(list(x), df["mean"], marker="o", linewidth=2)
    for xn, m, n in zip(x, df["mean"], df["n_cells"]):
        ax2.text(xn, m + 0.02, f"n={int(n)}", fontsize=8, va="bottom", ha="center")

    ax2.set_xticks(list(x)); ax2.set_xticklabels(xticklabels, rotation=30, ha="right")
    ax2.set_xticklabels(xticklabels, rotation=45, ha="right", fontsize=7)
    ax2.set_ylim(0, float(df["mean"].max()) + 0.1)
    ax2.set_ylabel("Mean SHH UCell"); ax2.set_title(f"{system} – SHH along lineage (mean)")
    ax2.text(xn, m, f"n={int(n)}", fontsize=6, va="bottom", ha="center")
    fig2.tight_layout()
    fig2.savefig(qcdir / f"{system}{suffix}_SHH_UCell_lineage_mean.pdf", dpi=300)
    fig2.savefig(qcdir / f"{system}{suffix}_SHH_UCell_lineage_mean.png", dpi=300)

    # ---------- MASTER table update
    keep_cols = ["celltype_new", "n_cells", "median", "mean", "q90", "frac>0"]
    tab = df[keep_cols].copy(); tab.insert(0, "system", system)
    master_path = out_root / "SHH_UCell_cardio_summary_MASTER.csv"
    if master_path.exists():
        master = pd.read_csv(master_path)
        master = pd.concat([master, tab], ignore_index=True)
        master.drop_duplicates(subset=["system", "celltype_new"], keep="last", inplace=True)
    else:
        master = tab
    master.to_csv(master_path, index=False)

    print(f"[PLOT] Saved for {system}: summary + lineage + MASTER updated")



def integrate_shh_with_edges(system_tag: str, outdir: Path):
    """
    - Subset the linkage to LPM ("system")
    - Manually add second heart field, atrial cardiomyocytes, as a new row
    - Use your calculated uscore mean value as sh_x, sh_y (add as new columns for edge_filtered.txt)
    - Calculate absolute delta, add it as a new column
    - Plot it into graph (igraph). x_name, y_name will be labels of the node, node 
    colored by the SHH score (sh_score). Edge width by abs delta. Title the graph with system ID 

    Merge SHH scores with lineage edges for one system and save extended table.
    """
    # Inputs
    suffix = run_suffix()
    score_csv = outdir / "qc" / f"{system_tag}{suffix}_ALL_labels_summary.csv"
    edge_file = Path("/project/xyang2/SHH/Qiu_TimeLapse/Holly_desktop/edges_filtered.txt")

    if not score_csv.exists():
        print(f"[EDGE] Missing score file: {score_csv}")
        return

    scores = pd.read_csv(score_csv)[["celltype_new", "mean"]]
    scores = scores.rename(columns={"mean": "sh_score"})

    edges = pd.read_csv(edge_file, sep="\t")
    edges = edges.loc[edges["system"] == system_tag].copy()

    # --- Manual addition: Second heart field → Atrial cardiomyocytes ---
    new_row = {
        "system": system_tag,
        "x": "L_M22",  # fill with x_number or symbol if available
        "y": "L_M5",
        "x_name": "Second heart field",
        "y_name": "Atrial cardiomyocytes",
        "edge_type": "Developmental progression",
        "x_number": None, "y_number": None, "x_id": None, "y_id": None,
    }

    need_manual = ~((edges["x_name"] == "Second heart field") & 
                (edges["y_name"] == "Atrial cardiomyocytes")).any()
    if need_manual:
        edges = pd.concat([edges, pd.DataFrame([new_row])], ignore_index=True)

    # --- Merge SHH scores ---
    edges = edges.merge(scores.rename(columns={"celltype_new": "x_name", "sh_score": "sh_x"}),
                        on="x_name", how="left")
    edges = edges.merge(scores.rename(columns={"celltype_new": "y_name", "sh_score": "sh_y"}),
                        on="y_name", how="left")

    # --- Delta ---
    edges["abs_delta"] = (edges["sh_x"] - edges["sh_y"]).abs()

    # --- Save ---
    out_file = outdir / f"{system_tag}_edge_filtered_with_shh.csv"
    edges.to_csv(out_file, index=False)
    print(f"[EDGE] wrote merged edges with SHH: {out_file}")


def run_suffix():
    s_layer = f"_{UCELL_INPUT_LAYER}" if UCELL_INPUT_LAYER else ""
    s_run   = "_test" if SUBSAMPLE_FRAC < 1.0 else ""
    return f"{s_layer}{s_run}"



def main():
    print("== UCELL SHH scoring (no Scanpy) ==")
    print(f"h5-root   : {H5_ROOT}")
    print(f"out-root  : {OUT_ROOT}")
    print(f"systems   : {SYSTEMS}")
    print(f"subsample : {SUBSAMPLE_FRAC}")
    print(f"ncores    : {NCORES}")

    _ensure_ucell_installed()

    for system_tag in SYSTEMS:
        h5_file = H5_ROOT / f"{system_tag}_adata_scale.h5ad"
        if not h5_file.exists():
            print(f"[SKIP] Missing: {h5_file}")
            continue
        outdir = OUT_ROOT / system_tag
        adata = load_adata(h5_file)
        adata = maybe_subsample(adata, SUBSAMPLE_FRAC, seed=0)
        sanitize_anndata_for_h5ad(adata)

        # ensure per-cell system is available for joins
        if "system" not in adata.obs.columns:
            adata.obs["system"] = system_tag

        # If we want to use log1p CPM, create the layer once (non-destructive)
        if UCELL_INPUT_LAYER:
            if UCELL_INPUT_LAYER not in adata.layers:
                ensure_log1p_cpm_layer(adata, layer_name=UCELL_INPUT_LAYER, target_sum=1_000_000)
            print(f"[LAYER] Ready: {UCELL_INPUT_LAYER} in adata.layers")


        # >>> INSERT DEBUG PRINTS HERE <<<
        raw_dbg = adata.raw.to_adata() if adata.raw is not None else adata
        print("Matrix shape genes x cells:", (raw_dbg.n_vars, raw_dbg.n_obs))
        print("Example rownames (genes):", list(map(str, raw_dbg.var_names[:5])))
        print("Example colnames (cells):", list(map(str, adata.obs_names[:5])))
        # <<< END DEBUG PRINTS >>>

        # >>> INSERT DIAGNOSTICS HERE <<<
        from scipy import sparse as sp
        raw = adata.raw.to_adata() if adata.raw is not None else adata
        rowlabels, _ = _pick_rowlabels_for_ucell(adata)
        want = ["Gli1", "Ptch1", "Hhip"]
        if not any(g in rowlabels for g in want):
            print("[DIAG] None of Gli1/Ptch1/Hhip are in rowlabels (unexpected).")
            print("       First 10 rowlabels:", rowlabels[:10])

        X = raw.X
        if sp.issparse(X):
            X = X.tocsr()
        for g in want:
            if g in rowlabels:
                i = rowlabels.index(g)
                col = X[:, i]
                if sp.issparse(col):
                    nz = int(col.nnz)
                    mean_val = float(col.sum() / raw.n_obs)
                else:
                    nz = int((col != 0).sum())
                    mean_val = float(col.mean())
                print(f"[DIAG] {g}: nonzero cells={nz} / {raw.n_obs}, mean={mean_val:.4g}")
            else:
                print(f"[DIAG] {g}: NOT in rowlabels")
        # <<< END DIAGNOSTICS >>>

        print(f"[UCELL] {system_tag}: computing SHH_UCell_score (cells={adata.n_obs})")


        rowlabels, _ = _pick_rowlabels_for_ucell(adata)
        print("[CHECK] Any exact-case hits?", [g for g in ["Gli1","Ptch1","Hhip"] if g in rowlabels])

        suffix = run_suffix()
        summary_csv = outdir / "qc" / f"{system_tag}{suffix}_ALL_labels_summary.csv"

        if not summary_csv.exists():
            shh = run_ucell_scores_shh(adata, shh_genes=SHH_GENES, ncores=NCORES, use_layer=UCELL_INPUT_LAYER)
            adata.obs["SHH_UCell_score"] = shh.values

            print(f"[UCELL] {system_tag}: summary\n{adata.obs['SHH_UCell_score'].describe()}")

            # --- Optional QC: summarize scores by celltype_new ---
            if "celltype_new" in adata.obs:
                summary = (adata.obs[["celltype_new", "SHH_UCell_score"]]
                    .groupby("celltype_new", observed=False)["SHH_UCell_score"]
                    .median().sort_values(ascending=False))
                (outdir / "qc").mkdir(parents=True, exist_ok=True)

                suffix = run_suffix()
                summary.to_csv(outdir / "qc" / f"{system_tag}{suffix}_SHH_UCell_by_celltype_median.csv")
                for label in ["Second heart field", "Atrial cardiomyocytes", "atrial CM"]:
                    if label in summary.index:
                        print(f"[QC] {label}: median SHH_UCell_score = {summary.loc[label]:.4f}")
            save_outputs(adata, outdir, system_tag)
        else:
            print(f"[SKIP] Using existing summaries: {summary_csv}")

        # --- GENERAL QC (all labels) ---
        if "celltype_new" in adata.obs and "SHH_UCell_score" in adata.obs:
            df = adata.obs[["celltype_new", "SHH_UCell_score"]].copy()
            counts  = df["celltype_new"].value_counts().sort_index()
            grp     = df.groupby("celltype_new", observed=False)["SHH_UCell_score"]
            summary = pd.DataFrame({
                "celltype_new": counts.index,
                "n_cells": counts.values,
                "median":  grp.median().reindex(counts.index).values,
                "mean":    grp.mean().reindex(counts.index).values,
                "q90":     grp.quantile(0.90).reindex(counts.index).values,
                "frac>0":  grp.apply(lambda s: (s > 0).mean()).reindex(counts.index).values,
            })
            (outdir / "qc").mkdir(parents=True, exist_ok=True)
            suffix = run_suffix()
            summary.to_csv(outdir / "qc" / f"{system_tag}{suffix}_ALL_labels_summary.csv", index=False)
            print("[QC] wrote:", outdir / "qc" / f"{system_tag}{suffix}_ALL_labels_summary.csv")
        else:
            print("[QC] Skipping general QC: SHH_UCell_score not in adata.obs")

        integrate_shh_with_edges(system_tag, outdir)  

        try:
            plot_shh_for_system(system_tag, OUT_ROOT)
        except Exception as e:
            print(f"[PLOT] Error plotting {system_tag}: {e}")

        del adata

    print("✓ Done.")


if __name__ == "__main__":
    main()
