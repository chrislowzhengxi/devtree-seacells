#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
UCell scoring for SHH genes (GLI1, PTCH1, HHIP), rank-based per cell.
Writes score to .obs['SHH_UCell_score'] and saves updated h5ad + per-cell CSV.

Run:
  module load python/anaconda-2022.05
  source activate /project/xyang2/software-packages/env/velocity_2025Feb_xy
  python ucell_scoring.py
"""

import os
from pathlib import Path
import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse

# ---- rpy2 / R bridge ----
import rpy2.robjects as ro
from rpy2.robjects import numpy2ri, pandas2ri
from rpy2.robjects.packages import importr
from rpy2.robjects.vectors import StrVector, ListVector

numpy2ri.activate()
pandas2ri.activate()

# ---------------------------------------------------------------------
# CONFIG (edit here)
# ---------------------------------------------------------------------
H5_ROOT   = Path("/project/imoskowitz/xyang2/SHH/Qiu_TimeLapse/results/raw_added")
OUT_ROOT  = Path("/project/imoskowitz/xyang2/chrislowzhengxi/results/ucell")

# # systems = ["Renal","Gut"]
# # systems = ["PNS_neurons"]
# systems = ["Lateral_plate_mesoderm"]
SYSTEMS   = ["Renal"]   # e.g. ["Renal","Gut"] or ["PNS_neurons"]
SUBSAMPLE_FRAC = 0.01                    # 0.01 for quick test, 1.0 for full
NCORES = int(os.environ.get("SLURM_CPUS_PER_TASK", "1"))

# SHH panel Holly asked to try first
SHH_GENES = ["GLI1", "PTCH1", "HHIP"]

# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def load_adata(h5_file: Path) -> sc.AnnData:
    print(f"[LOAD] {h5_file}")
    adata = sc.read_h5ad(str(h5_file))
    if isinstance(adata.var_names, pd.CategoricalIndex):
        adata.var_names = adata.var_names.astype(str)
    else:
        adata.var_names = adata.var_names.astype(str)
    if adata.raw is None:
        adata.raw = adata.copy()
    else:
        raw = adata.raw.to_adata()
        if isinstance(raw.var_names, pd.CategoricalIndex):
            raw.var_names = raw.var_names.astype(str)
        adata.raw = raw
    return adata

def maybe_subsample(adata: sc.AnnData, fraction: float, seed: int = 0) -> sc.AnnData:
    if fraction is None or fraction >= 1.0:
        return adata
    n_before = adata.n_obs
    sc.pp.subsample(adata, fraction=fraction, random_state=seed)
    print(f"[SUBSAMPLE] {n_before} → {adata.n_obs} cells (fraction={fraction})")
    return adata

def _matrix_from_adata_for_ucell(adata: sc.AnnData):
    """Return (X_T, gene_names, cell_names) for R/UCell: genes x cells matrix."""
    if adata.raw is not None:
        raw = adata.raw.to_adata()
        X = raw.X
        genes = list(raw.var_names)
    else:
        X = adata.X
        genes = list(adata.var_names)
    if sparse.issparse(X):
        X = X.toarray()
    X = np.asarray(X, dtype=np.float32)
    X_T = X.T  # R wants genes x cells
    cells = list(adata.obs_names)
    return X_T, genes, cells

def _ensure_ucell_installed():
    try:
        return importr("UCell")
    except Exception as e:
        raise RuntimeError(
            "Could not import R package 'UCell'. Use Holly's env:\n"
            "  module load python/anaconda-2022.05\n"
            "  source activate /project/xyang2/software-packages/env/velocity_2025Feb_xy"
        ) from e

def run_ucell_scores_shh(adata: sc.AnnData, shh_genes=SHH_GENES, ncores: int = 1) -> pd.Series:
    _ensure_ucell_installed()
    ucell = importr("UCell")
    X_T, gene_names, cell_names = _matrix_from_adata_for_ucell(adata)

    # case-insensitive matching
    lut = {g.upper(): g for g in gene_names}
    used = [lut[g.upper()] for g in shh_genes if g.upper() in lut]
    if not used:
        print("⚠️  SHH genes not found in matrix — returning zeros.")
        return pd.Series(0.0, index=adata.obs_names, name="SHH_UCell_score")

    # Build R matrix and dimnames via globalenv (robust with rpy2)
    r_mat = ro.r["as.matrix"](X_T)
    ro.globalenv["M"] = r_mat
    ro.globalenv["G"] = StrVector(gene_names)
    ro.globalenv["C"] = StrVector(cell_names)
    ro.r("rownames(M) <- G; colnames(M) <- C")

    sigs = ListVector({"SHH": StrVector(used)})
    res_df = ucell.ScoreSignatures_UCell(ro.globalenv["M"], features=sigs, ncores=int(ncores))
    res_pd = pandas2ri.rpy2py(res_df)

    if "SHH_UCell" in res_pd.columns:
        score = res_pd["SHH_UCell"].to_numpy()
    elif "SHH" in res_pd.columns:
        score = res_pd["SHH"].to_numpy()
    else:
        score = res_pd.iloc[:, 0].to_numpy()

    if len(score) != adata.n_obs:
        raise ValueError(f"UCell returned {len(score)} scores for {adata.n_obs} cells.")
    return pd.Series(score, index=adata.obs_names, name="SHH_UCell_score")

def save_outputs(adata: sc.AnnData, out_dir: Path, system: str):
    out_dir.mkdir(parents=True, exist_ok=True)
    h5_out = out_dir / f"{system}_adata_with_ucell.h5ad"
    csv_out = out_dir / f"{system}_SHH_UCell_scores.csv"
    adata.write(str(h5_out))
    adata.obs[["SHH_UCell_score"]].to_csv(csv_out, index=True)
    print(f"[SAVE] {h5_out}")
    print(f"[SAVE] {csv_out}")

# ---------------------------------------------------------------------
# MAIN: loop over systems (edit SYSTEMS / SUBSAMPLE_FRAC above)
# ---------------------------------------------------------------------
def main():
    print("== UCELL SHH scoring ==")
    print(f"h5-root   : {H5_ROOT}")
    print(f"out-root  : {OUT_ROOT}")
    print(f"systems   : {SYSTEMS}")
    print(f"subsample : {SUBSAMPLE_FRAC}")
    print(f"ncores    : {NCORES}")

    # sanity: check UCell availability once
    _ensure_ucell_installed()

    for system_tag in SYSTEMS:
        h5_file = H5_ROOT / f"{system_tag}_adata_scale.h5ad"
        if not h5_file.exists():
            print(f"[SKIP] Missing: {h5_file}")
            continue

        outdir = OUT_ROOT / system_tag
        adata = load_adata(h5_file)
        adata = maybe_subsample(adata, SUBSAMPLE_FRAC, seed=0)

        print(f"[UCELL] {system_tag}: computing SHH_UCell_score (cells={adata.n_obs})")
        shh = run_ucell_scores_shh(adata, shh_genes=SHH_GENES, ncores=NCORES)
        adata.obs["SHH_UCell_score"] = shh.values

        print(f"[UCELL] {system_tag}: summary\n{adata.obs['SHH_UCell_score'].describe()}")
        save_outputs(adata, outdir, system_tag)
        del adata

    print("✓ Done.")

if __name__ == "__main__":
    main()
