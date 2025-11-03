#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build an "extended obs" table for SHH genes.
- Reads a scored .h5ad
- Pulls raw counts for Gli1, Ptch1, Hhip from .raw
- Pulls normalized values from a chosen layer (creates log1p_cpm if missing)
- Adds simple positivity flags (raw > 0)
- Writes a TSV to <system>/qc/<system>_extended_obs.tsv

Usage
-----
python make_extended_obs.py \
  --h5ad /project/.../Lateral_plate_mesoderm_adata_with_ucell.h5ad \
  --layer log1p_cpm

You can change the layer name with --layer. If the layer is missing, it will be created from .raw using CPM then log1p.
"""

import os
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import anndata as ad
from scipy import sparse

DEFAULT_GENES = ["Gli1", "Ptch1", "Hhip"]
CAND_SYMBOL_COLS = ["gene_short_name","gene_symbol","symbol","SYMBOL","Gene","gene",
                    "gene_name","features","Feature","GeneSymbol"]


def _ensure_var_unique(adata):
    try:
        adata.var_names = adata.var_names.astype(str)
        if hasattr(adata, "var_names_make_unique"):
            adata.var_names_make_unique()
    except Exception:
        pass
    if adata.raw is not None:
        raw = adata.raw.to_adata()
        raw.var_names = raw.var_names.astype(str)
        if hasattr(raw, "var_names_make_unique"):
            raw.var_names_make_unique()
        adata.raw = raw
    return adata


def ensure_log1p_cpm_layer_np(adata, layer_name="log1p_cpm", target_sum=1_000_000):
    if adata.raw is None:
        raise ValueError("adata.raw is None. Cannot construct normalized layer without raw counts.")
    raw = adata.raw.to_adata()

    # Align genes
    common = adata.var_names.intersection(raw.var_names)
    if len(common) == 0:
        raise ValueError("No overlap between adata.var_names and adata.raw.var_names")
    var_idx = adata.var_names.get_indexer(common)
    raw_idx = raw.var_names.get_indexer(common)

    # Make sure indices are plain int64 arrays (prevents SciPy setitem issues)
    var_idx = np.asarray(var_idx, dtype=np.int64)
    raw_idx = np.asarray(raw_idx, dtype=np.int64)

    X = raw.X[:, raw_idx]
    n_obs, n_vars = adata.n_obs, adata.n_vars

    if sparse.issparse(X):
        X = X.tocsr(copy=True)
        rs = np.asarray(X.sum(axis=1)).ravel()
        with np.errstate(divide="ignore", invalid="ignore"):
            scale = np.divide(float(target_sum), rs, out=np.zeros_like(rs, dtype=float), where=rs > 0)
        # scale rows
        S = sparse.diags(scale)
        X = S @ X
        # log1p on data
        X.data = np.log1p(X.data, dtype=np.float64)

        # Assign by columns into CSC (column-efficient), then convert to CSR
        L = sparse.csc_matrix((n_obs, n_vars), dtype=np.float32)
        L[:, var_idx] = X.astype(np.float32)
        adata.layers[layer_name] = L.tocsr()
    else:
        X = X.astype(np.float64, copy=True)
        rs = X.sum(1)
        with np.errstate(divide="ignore", invalid="ignore"):
            scale = np.divide(float(target_sum), rs, out=np.zeros_like(rs, dtype=float), where=rs > 0)
        X = (X.T * scale).T
        np.log1p(X, out=X)

        # Dense path is straightforward
        L = np.zeros((n_obs, n_vars), dtype=np.float32)
        L[:, var_idx] = X.astype(np.float32)
        adata.layers[layer_name] = L

    return layer_name


def _match_genes(raw_var, genes):
    """Case-insensitive match to raw.var_names first."""
    lut = {g.upper(): g for g in map(str, raw_var.index.tolist())}
    found = {}
    missing = []
    for g in genes:
        gg = lut.get(g.upper())
        if gg is not None:
            found[g] = gg
        else:
            missing.append(g)
    return found, missing


def _match_genes_by_symbol_column(raw_var, genes):
    """Fallback: try a symbol-like column inside raw.var."""
    for col in CAND_SYMBOL_COLS:
        if col in raw_var.columns:
            series = raw_var[col].astype(str)
            sym2row = {}
            for row, sym in zip(raw_var.index.astype(str), series):
                up = sym.upper()
                if up and up not in sym2row:
                    sym2row[up] = row
            found = {}
            missing = []
            for g in genes:
                row = sym2row.get(g.upper())
                if row is not None:
                    found[g] = row
                else:
                    missing.append(g)
            if found:
                return found, missing, col
    return {}, genes, None


def _extract_raw_columns(adata, rownames_map):
    """Return dict gene -> raw vector using adata.raw rownames."""
    raw = adata.raw.to_adata()
    rows = [rownames_map[g] for g in rownames_map]
    idx = raw.var_names.get_indexer(rows)
    X = raw.X[:, idx]
    if sparse.issparse(X):
        X = X.toarray()
    out = {}
    for j, g in enumerate(rownames_map):
        out[g] = X[:, j].astype(np.float64, copy=False)
    return out


def _match_genes_in_main_var(adata, rows_from_raw):
    """Map the rownames selected in raw back to adata.var_names for layer extraction."""
    main = adata.var
    out = {}
    not_found = []

    main_names = set(map(str, main.index.tolist()))
    for g_req, raw_row in rows_from_raw.items():
        if raw_row in main_names:
            out[g_req] = raw_row
        else:
            not_found.append((g_req, raw_row))

    if not not_found:
        return out

    for col in CAND_SYMBOL_COLS:
        if col in main.columns:
            series = main[col].astype(str)
            sym2row = {}
            for row, sym in zip(main.index.astype(str), series):
                up = sym.upper()
                if up and up not in sym2row:
                    sym2row[up] = row
            still_missing = []
            for g_req, raw_row in not_found:
                rowname = sym2row.get(g_req.upper())
                if rowname is not None:
                    out[g_req] = rowname
                else:
                    still_missing.append((g_req, raw_row))
            not_found = still_missing
            if not not_found:
                break

    if not_found:
        for g_req, raw_row in not_found:
            out[g_req] = raw_row
    return out


def _extract_layer_columns(adata, layer_name, var_rowname_map):
    """Return dict gene -> normalized vector using adata.layers[layer_name]."""
    if layer_name not in adata.layers:
        raise ValueError(f"Layer '{layer_name}' not present in adata.layers")
    L = adata.layers[layer_name]
    out = {}
    for g_req, rowname in var_rowname_map.items():
        if rowname not in adata.var_names:
            out[g_req] = np.full(adata.n_obs, np.nan, dtype=float)
            continue
        j = int(np.where(adata.var_names == rowname)[0][0])
        col = L[:, j]
        if sparse.issparse(col):
            col = col.toarray().ravel()
        else:
            col = np.asarray(col).ravel()
        out[g_req] = col.astype(np.float64, copy=False)
    return out

def _normalized_log1p_from_raw(raw_adata, raw_row_indices, target_sum=1_000_000):
    """Return a dict: raw_rowname -> log1p(CPM) vector for selected rows, no full layer."""
    from scipy import sparse as sp
    Xraw = raw_adata.X
    # per-cell library sizes
    if sp.issparse(Xraw):
        rs = np.asarray(Xraw.sum(axis=1)).ravel().astype(np.float64)
    else:
        rs = Xraw.sum(axis=1).astype(np.float64)

    # safe scaling
    with np.errstate(divide="ignore", invalid="ignore"):
        scale = np.divide(float(target_sum), rs, out=np.zeros_like(rs), where=rs > 0)

    out = {}
    for j in raw_row_indices:
        col = Xraw[:, j]
        if sp.issparse(col):
            col = col.toarray().ravel().astype(np.float64, copy=False)
        else:
            col = np.asarray(col).ravel().astype(np.float64, copy=False)
        v = col * scale
        np.log1p(v, out=v)
        out[int(j)] = v  # key by raw column index
    return out


def build_extended_obs(h5ad_path, outdir=None, layer="log1p_cpm", genes=None):
    p = Path(h5ad_path)
    adata = ad.read_h5ad(str(p))
    _ensure_var_unique(adata)

    if outdir is None:
        outdir = p.parent / "qc"
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    genes = genes or DEFAULT_GENES

    if adata.raw is None:
        raise ValueError("This .h5ad has no .raw. Cannot proceed.")

    found, missing = _match_genes(adata.raw.var, genes)
    symbol_col_used = None
    if missing:
        found2, missing2, symbol_col_used = _match_genes_by_symbol_column(adata.raw.var, missing)
        found.update(found2)
        missing = missing2

    main_map = _match_genes_in_main_var(adata, found)

    # raw values as before
    raw_vals = _extract_raw_columns(adata, found)

    # compute normalized values ONLY for requested genes, no full layer
    raw = adata.raw.to_adata()
    # indices of requested genes in raw
    raw_idx_vec = raw.var_names.get_indexer([found[g] for g in genes if g in found])
    raw_idx_vec = raw_idx_vec[raw_idx_vec >= 0]
    norm_by_raw_idx = _normalized_log1p_from_raw(raw, raw_idx_vec, target_sum=1_000_000)

    # map back to requested gene names
    norm_vals = {}
    for g in genes:
        if g in found:
            j = int(np.where(raw.var_names == found[g])[0][0])
            norm_vals[g] = norm_by_raw_idx.get(j, np.full(adata.n_obs, np.nan, dtype=float))
        else:
            norm_vals[g] = np.full(adata.n_obs, np.nan, dtype=float)

    # for reporting
    created_layer = False


    df = pd.DataFrame(index=adata.obs_names.astype(str))
    if "cell_id" in adata.obs.columns:
        df.insert(0, "cell_id", adata.obs["cell_id"].astype(str).values)
    else:
        df.insert(0, "cell_id", df.index.values.astype(str))

    for col in ["celltype_new", "system"]:
        if col in adata.obs.columns:
            df[col] = adata.obs[col].astype(str).values
    if "SHH_UCell_score" in adata.obs.columns:
        df["SHH_UCell_score"] = pd.to_numeric(adata.obs["SHH_UCell_score"], errors="coerce").values

    for g in genes:
        col_raw = np.full(adata.n_obs, np.nan, dtype=float)
        if g in raw_vals:
            col_raw = raw_vals[g]
        df[f"{g}_raw"] = col_raw

        col_norm = np.full(adata.n_obs, np.nan, dtype=float)
        if g in norm_vals:
            col_norm = norm_vals[g]
        df[f"{g}_norm"] = col_norm

        df[f"{g}_pos_raw"] = (df[f"{g}_raw"].values > 0).astype(int)

    system_tag = None
    if "system" in df.columns:
        vals = pd.unique(df["system"])
        if len(vals) == 1:
            system_tag = str(vals[0])
    if system_tag is None:
        system_tag = p.stem.split("_")[0]

    out_tsv = outdir / f"{system_tag}_extended_obs.tsv"
    df.to_csv(out_tsv, sep="\t", index=False)

    print("== Extended obs report ==")
    print("Input:", p)
    print("Output:", out_tsv)
    print("Layer used:", layer, "(created now)" if created_layer else "(existing)")
    if symbol_col_used:
        print("Matched genes via symbol column:", symbol_col_used)
    if missing:
        print("Requested but not found in raw:", missing)
    print("N cells:", adata.n_obs)
    for g in genes:
        if f"{g}_raw" in df.columns:
            pos = int((df[f"{g}_raw"] > 0).sum())
            pct = 100.0 * pos / len(df)
            print(f"{g}: raw-positive = {pos} / {len(df)} ({pct:.2f}%)")
    return str(out_tsv)


def parse_args():
    ap = argparse.ArgumentParser(description="Create extended obs with SHH gene raw and normalized values.")
    ap.add_argument("--h5ad", required=True, help="Path to the scored .h5ad")
    ap.add_argument("--outdir", default=None, help="Output directory. Default: <h5ad dir>/qc")
    ap.add_argument("--layer", default="log1p_cpm", help="Normalized layer name to use or create")
    ap.add_argument("--genes", nargs="*", default=DEFAULT_GENES, help="Genes to extract")
    return ap.parse_args()


def main():
    args = parse_args()
    build_extended_obs(args.h5ad, args.outdir, args.layer, args.genes)


if __name__ == "__main__":
    main()
