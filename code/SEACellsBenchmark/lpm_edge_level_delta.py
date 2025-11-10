#!/usr/bin/env python3
"""
Build and rank edge-level deltas for three metrics within one system:

A) % SHH_UCell_score > 0
B) % SHH in Middle/High (cutoffs low_thr, high_thr)
C) % <gene> raw > 0  (default Gli1; switch with --gene)

Inputs
- --extended-tsv  .../<system>/qc/<system>_extended_obs.tsv
    requires columns: celltype_new, SHH_UCell_score, <gene>_raw
- --edges         /project/xyang2/SHH/Qiu_TimeLapse/Holly_desktop/edges_filtered.txt
    requires columns: system, x_name, y_name
- --system        system name to filter edges

Outputs written to the same folder as the extended TSV:
- edges_ranked_shh_nonzero.tsv
- edges_ranked_shh_midhigh_{low}_{high}.tsv
- edges_ranked_pos_raw_<gene>.tsv
Also prints the gold-edge rank and stats.
"""

from pathlib import Path
import argparse
import numpy as np
import pandas as pd

def ensure_manual_edges(edges: pd.DataFrame, system_tag: str) -> pd.DataFrame:
    """
    Add hand-curated edges that are drawn in the graph but not present
    in edges_filtered.txt so they also appear in rankings.
    """
    edges = edges.copy()
    # Lateral_plate_mesoderm: Second heart field -> Atrial cardiomyocytes
    if system_tag == "Lateral_plate_mesoderm":
        need = ~((edges.get("x_name") == "Second heart field") &
                 (edges.get("y_name") == "Atrial cardiomyocytes")).any()
        if need:
            new_row = {
                "system": system_tag,
                "x_name": "Second heart field",
                "y_name": "Atrial cardiomyocytes",
                # optional placeholders for other columns if present
                "edge_type": "Developmental progression",
            }
            edges = pd.concat([edges, pd.DataFrame([new_row])], ignore_index=True)
    return edges

def pct(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.where(b > 0, 100.0 * a / b, np.nan)
    return out

def bin_shh(s, low_thr=0.35, high_thr=0.65):
    x = pd.to_numeric(s, errors="coerce").fillna(0.0).values
    out = np.empty(x.shape[0], dtype=object)
    out[x == 0.0] = "0"
    out[(x > 0.0) & (x < low_thr)] = "Low"
    out[(x >= low_thr) & (x < high_thr)] = "Middle"
    out[x >= high_thr] = "High"
    return pd.Categorical(out, categories=["0", "Low", "Middle", "High"], ordered=True)

def load_inputs(extended_tsv: Path, edges_txt: Path, system: str):
    df = pd.read_csv(extended_tsv, sep="\t")
    edges = pd.read_csv(edges_txt, sep="\t")
    if "system" in edges.columns:
        edges = edges.loc[edges["system"] == system].copy()
    need_cols = {"x_name", "y_name"}
    if not need_cols.issubset(edges.columns):
        raise ValueError("edges file must have columns x_name and y_name")
    return df, edges

def per_node_tables(df: pd.DataFrame, low_thr: float, high_thr: float, gene: str):
    if "celltype_new" not in df.columns:
        raise ValueError("extended TSV must have column celltype_new")
    # totals per node
    base = df[["celltype_new"]].copy()
    base["one"] = 1
    tot = base.groupby("celltype_new", observed=False)["one"].sum().rename("n_total")

    # A) % SHH > 0
    shh = pd.to_numeric(df["SHH_UCell_score"], errors="coerce").fillna(0.0)
    nz = (shh > 0).astype(int)
    n_nz = nz.groupby(df["celltype_new"], observed=False).sum().rename("n_shh_nonzero")
    A = pd.concat([tot, n_nz], axis=1).fillna(0)
    A["pct_shh_nonzero"] = pct(A["n_shh_nonzero"], A["n_total"])
    A = A.reset_index().rename(columns={"celltype_new": "node"})

    # B) % SHH in Middle+High
    bins = bin_shh(shh, low_thr=low_thr, high_thr=high_thr)
    df_bins = pd.DataFrame({"node": df["celltype_new"], "bin": bins})
    counts = df_bins.value_counts().rename("n").reset_index()
    wide = counts.pivot(index="node", columns="bin", values="n").fillna(0)
    for c in ["0", "Low", "Middle", "High"]:
        if c not in wide.columns:
            wide[c] = 0
    wide["n_total"] = wide[["0", "Low", "Middle", "High"]].sum(axis=1)
    wide["n_midhigh"] = wide["Middle"] + wide["High"]
    wide["pct_midhigh"] = pct(wide["n_midhigh"], wide["n_total"])
    B = wide.reset_index()[["node", "n_total", "pct_midhigh"]]

    # C) % <gene> raw > 0
    raw_col = f"{gene}_raw"
    if raw_col not in df.columns:
        raise ValueError(f"Missing column {raw_col} in extended TSV")
    pos = (pd.to_numeric(df[raw_col], errors="coerce").fillna(0.0) > 0).astype(int)
    n_pos = pos.groupby(df["celltype_new"], observed=False).sum().rename(f"n_{gene}_pos")
    C = pd.concat([tot, n_pos], axis=1).fillna(0)
    C[f"pct_{gene}_pos"] = pct(C[f"n_{gene}_pos"], C["n_total"])
    C = C.reset_index().rename(columns={"celltype_new": "node"})

    # a single n_total per node
    N = A[["node", "n_total"]].copy()
    return N, A[["node", "pct_shh_nonzero"]], B[["node", "pct_midhigh"]], C[["node", f"pct_{gene}_pos"]]

def edge_table_from_nodes(edges: pd.DataFrame,
                          node_counts: pd.DataFrame,
                          node_vals: pd.DataFrame,
                          val_col: str,
                          label: str):
    # counts for sizes
    nmap = node_counts.rename(columns={"node": "name", "n_total": "n_total"})
    # values to x and y
    left = node_vals.rename(columns={"node": "x_name", val_col: f"value_x"})
    right = node_vals.rename(columns={"node": "y_name", val_col: f"value_y"})

    df = edges.merge(left, on="x_name", how="left").merge(right, on="y_name", how="left")

    # add n_x and n_y
    nx = nmap.rename(columns={"name": "x_name", "n_total": "n_x"})
    ny = nmap.rename(columns={"name": "y_name", "n_total": "n_y"})
    df = df.merge(nx, on="x_name", how="left").merge(ny, on="y_name", how="left")

    # compute deltas
    df["delta"] = df["value_y"] - df["value_x"]
    df["abs_delta"] = df["delta"].abs()
    df["metric"] = label
    return df

def add_ranks(df: pd.DataFrame):
    out = df.sort_values(["metric", "abs_delta"], ascending=[True, False]).copy()
    out["rank"] = out.groupby("metric", observed=False).cumcount() + 1
    return out

def print_gold(df_ranked: pd.DataFrame, gold_x: str, gold_y: str):
    sel = df_ranked[(df_ranked["x_name"] == gold_x) & (df_ranked["y_name"] == gold_y)]
    if sel.empty:
        sel = df_ranked[(df_ranked["x_name"] == gold_y) & (df_ranked["y_name"] == gold_x)]
    if sel.empty:
        print(f"[gold] Edge not found: {gold_x} -> {gold_y}")
        return
    for metric, sub in sel.groupby("metric", observed=False):
        r = int(sub["rank"].iloc[0])
        ad = float(sub["abs_delta"].iloc[0])
        nx = int(sub["n_x"].iloc[0]) if pd.notna(sub["n_x"].iloc[0]) else -1
        ny = int(sub["n_y"].iloc[0]) if pd.notna(sub["n_y"].iloc[0]) else -1
        print(f"[gold] {metric}: rank={r}, abs_delta={ad:.6f}, n_x={nx}, n_y={ny}, size_total={nx+ny if nx>=0 and ny>=0 else 'NA'}")

def main():
    ap = argparse.ArgumentParser(description="Compute and rank edge deltas for SHH metrics.")
    ap.add_argument("--extended-tsv", required=True, help=".../<system>/qc/<system>_extended_obs.tsv")
    ap.add_argument("--edges", required=True, help="edges_filtered.txt")
    ap.add_argument("--system", required=True, help="system name to filter edges")
    ap.add_argument("--gene", default="Gli1")
    ap.add_argument("--low-thr", type=float, default=0.35)
    ap.add_argument("--high-thr", type=float, default=0.65)
    ap.add_argument("--gold-x", default="Second heart field")
    ap.add_argument("--gold-y", default="Atrial cardiomyocytes")
    args = ap.parse_args()

    ext = Path(args.extended_tsv)
    outdir = ext.parent
    outdir.mkdir(parents=True, exist_ok=True)

    df, edges = load_inputs(ext, Path(args.edges), args.system)
    edges = ensure_manual_edges(edges, args.system) 

    node_counts, Avals, Bvals, Cvals = per_node_tables(df, args.low_thr, args.high_thr, args.gene)

    tblA = edge_table_from_nodes(edges, node_counts, Avals, "pct_shh_nonzero", "SHH>0")
    tblB = edge_table_from_nodes(edges, node_counts, Bvals, "pct_midhigh", f"SHH_MidHigh_{args.low_thr}_{args.high_thr}")
    tblC = edge_table_from_nodes(edges, node_counts, Cvals, f"pct_{args.gene}_pos", f"{args.gene}_raw>0")

    ranked = add_ranks(pd.concat([tblA, tblB, tblC], ignore_index=True))

    # Write separate ranked TSVs per metric
    def save_metric(dfm, fname):
        cols = ["rank","x_name","y_name","n_x","n_y","value_x","value_y","delta","abs_delta"]
        dfm = dfm.sort_values("rank").reset_index(drop=True)
        dfm.to_csv(outdir / fname, sep="\t", index=False, columns=[c for c in cols if c in dfm.columns])

    save_metric(ranked[ranked["metric"] == "SHH>0"],
                "edges_ranked_shh_nonzero.tsv")
    save_metric(ranked[ranked["metric"] == f"SHH_MidHigh_{args.low_thr}_{args.high_thr}"],
                f"edges_ranked_shh_midhigh_{args.low_thr}_{args.high_thr}.tsv")
    save_metric(ranked[ranked["metric"] == f"{args.gene}_raw>0"],
                f"edges_ranked_pos_raw_{args.gene}.tsv")

    # Console summaries
    print_gold(ranked, args.gold_x, args.gold_y)
    print("\nTop 5 by metric:")
    for metric, sub in ranked.groupby("metric", observed=False):
        top = sub.sort_values("rank").head(5)[["rank","x_name","y_name","abs_delta"]]
        print(f"\n[{metric}]")
        print(top.to_string(index=False))

    print(f"\nSaved ranked edge tables in {outdir}")

if __name__ == "__main__":
    main()
