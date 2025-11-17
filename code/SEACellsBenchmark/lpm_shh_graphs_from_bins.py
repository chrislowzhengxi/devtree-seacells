#!/usr/bin/env python3
"""
Make three lineage graphs for one system using the extended_obs table:
  A) node color = % SHH_UCell_score > 0
  B) node color = % SHH_UCell_score in {Low, Middle, High} with thresholds
  C) node color = % <gene> raw > 0  (default Gli1; switch with --gene)

Inputs
- extended TSV:  <system>/qc/<system>_extended_obs.tsv
- edges file:    /project/xyang2/SHH/Qiu_TimeLapse/Holly_desktop/edges_filtered.txt

Outputs
- <system>/qc/graph_%positive_shh_nonzero.{png,pdf}
- <system>/qc/graph_%shh_lowmidhigh_{tlow}_{thigh}.{png,pdf}
- <system>/qc/graph_%pos_raw_<gene>.{png,pdf}
- Helper CSVs with per-node percentages used for coloring.
"""

from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx

def load_inputs(extended_tsv: Path, edges_txt: Path, system: str):
    df = pd.read_csv(extended_tsv, sep="\t")
    edges = pd.read_csv(edges_txt, sep="\t")
    edges = edges.loc[edges["system"] == system].copy()
    return df, edges

def pct(a, b):
    """Safe percent = 100*a/b."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.where(b > 0, 100.0 * a / b, np.nan)
    return out

def bin_shh(s, low_thr=0.35, high_thr=0.65):
    """Return categorical bins: '0', 'Low', 'Middle', 'High'."""
    x = pd.to_numeric(s, errors="coerce").fillna(0.0).values
    out = np.empty(x.shape[0], dtype=object)
    out[x == 0.0] = "0"
    out[(x > 0.0) & (x < low_thr)] = "Low"
    out[(x >= low_thr) & (x < high_thr)] = "Middle"
    out[x >= high_thr] = "High"
    return pd.Categorical(out, categories=["0","Low","Middle","High"], ordered=True)

def aggregate_per_node(df: pd.DataFrame, low_thr: float, high_thr: float, raw_gene: str):
    """Compute per-node counts and percentages used by all three plots."""
    if "celltype_new" not in df.columns:
        raise ValueError("extended_obs.tsv must contain column 'celltype_new'")

    # Totals per node
    base = df[["celltype_new"]].copy()
    base["one"] = 1
    tot = base.groupby("celltype_new", observed=False)["one"].sum().rename("n_total")

    # A) % SHH non-zero
    nz = (pd.to_numeric(df["SHH_UCell_score"], errors="coerce").fillna(0.0) > 0).astype(int)
    pct_nonzero = nz.groupby(df["celltype_new"], observed=False).sum().rename("n_shh_nonzero")
    tbl_A = pd.concat([tot, pct_nonzero], axis=1).fillna(0)
    tbl_A["pct_shh_nonzero"] = pct(tbl_A["n_shh_nonzero"], tbl_A["n_total"])

    # B) % Low or Middle or High (i.e., non-zero but also binned by thresholds)
    bins = bin_shh(df["SHH_UCell_score"], low_thr=low_thr, high_thr=high_thr)
    df_bins = pd.DataFrame({"node": df["celltype_new"], "bin": bins})
    counts = df_bins.value_counts().rename("n").reset_index()
    wide = counts.pivot(index="node", columns="bin", values="n").fillna(0)
    # Ensure all columns exist
    for c in ["0","Low","Middle","High"]:
        if c not in wide.columns:
            wide[c] = 0

    wide["n_total"] = wide[["0","Low","Middle","High"]].sum(axis=1)

    # Only Middle + High
    wide["n_midhigh"] = wide["Middle"] + wide["High"]
    wide["pct_midhigh"] = pct(wide["n_midhigh"], wide["n_total"])

    # keep Low/Mid/High counts if you still want them saved
    tbl_B = wide.reset_index().rename(columns={"node":"celltype_new"})


    # C) % raw > 0 for chosen gene
    raw_col = f"{raw_gene}_raw"
    if raw_col not in df.columns:
        raise ValueError(f"Column '{raw_col}' not found in extended_obs.tsv")
    pos_raw = (pd.to_numeric(df[raw_col], errors="coerce").fillna(0.0) > 0).astype(int)
    pos_raw_node = pos_raw.groupby(df["celltype_new"], observed=False).sum().rename(f"n_{raw_gene}_pos")
    tbl_C = pd.concat([tot, pos_raw_node], axis=1).fillna(0)
    tbl_C[f"pct_{raw_gene}_pos"] = pct(tbl_C[f"n_{raw_gene}_pos"], tbl_C["n_total"])

    # Merge a single frame for convenience

    merged = (
        tbl_A.reset_index()
        .merge(tbl_B[["celltype_new", "pct_midhigh"]], on="celltype_new", how="left")
        .merge(tbl_C.reset_index(), on="celltype_new", how="left")
        .fillna(0)
    )
    return merged, tbl_A.reset_index(), tbl_B, tbl_C.reset_index()

def _build_graph(edges_df: pd.DataFrame):
    """Build a DiGraph of the system from edges with x_name, y_name."""
    G = nx.DiGraph()
    for r in edges_df.itertuples(index=False):
        x = getattr(r, "x_name", None)
        y = getattr(r, "y_name", None)
        if pd.isna(x) or pd.isna(y):
            continue
        G.add_node(x)
        G.add_node(y)
        G.add_edge(x, y)
    return G

def _node_positions(G):
    """Try topological layout. Fallback to spring layout."""
    try:
        _ = list(nx.topological_sort(G))
        pos = nx.spring_layout(G, k=0.6, seed=4)
    except nx.NetworkXUnfeasible:
        pos = nx.spring_layout(G, k=0.6, seed=4)
    return pos

def _draw_graph(G, pos, node_values: dict, title: str, out_png: Path, vlabel: str):
    # Convert values to array aligned to G.nodes
    vals = np.array([node_values.get(n, np.nan) for n in G.nodes()])
    vmin = np.nanmin(vals) if np.isfinite(np.nanmin(vals)) else 0.0
    vmax = np.nanmax(vals) if np.isfinite(np.nanmax(vals)) else 1.0

    # Normalize 0..100 if these are percentages
    if vmax > 1.0:
        vmax = 100.0
        vmin = 0.0

    cmap = plt.cm.GnBu
    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    colors = [cmap(norm(node_values.get(n, np.nan))) if n in node_values else (0.9,0.9,0.9,1.0) for n in G.nodes()]

    fig, ax = plt.subplots(figsize=(14, 8), constrained_layout=True)
    # nx.draw_networkx_edges(G, pos, width=1.6, arrows=True, arrowstyle="-|>", arrowsize=12,
    #                        connectionstyle="arc3,rad=0.05", ax=ax)
    # Scale edge width by absolute delta if we have values for both nodes
    # Build edge widths from absolute change between node values (0 to 100 scale)
    edges_with_delta = []
    widths = []
    for (x, y) in G.edges():
        vx = node_values.get(x, np.nan)
        vy = node_values.get(y, np.nan)
        if np.isfinite(vx) and np.isfinite(vy):
            delta = abs(float(vx) - float(vy))  # in percentage points
        else:
            delta = 0.0
        # Scale to a readable width range: 1.0 to ~7.0
        widths.append(1.0 + 10.0 * (delta / 100.0))
        edges_with_delta.append((x, y))

    nx.draw_networkx_edges(
        G, pos,
        edgelist=edges_with_delta,
        width=widths,
        arrows=True, arrowstyle="-|>", arrowsize=12,
        connectionstyle="arc3,rad=0.05", ax=ax
    )


    nx.draw_networkx_nodes(G, pos, node_color=colors, node_size=520,
                           linewidths=0.8, edgecolors="black", ax=ax)
    nx.draw_networkx_labels(G, pos, font_size=8, font_color="darkred", ax=ax)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label(vlabel)

    ax.set_title(title)
    ax.set_axis_off()

    out_pdf = out_png.with_suffix(".pdf")
    fig.savefig(out_png, dpi=300)
    fig.savefig(out_pdf)
    plt.close(fig)

def _ensure_manual_edges(edges: pd.DataFrame, system_tag: str) -> pd.DataFrame:
    edges = edges.copy()
    if system_tag == "Lateral_plate_mesoderm":
        need = ~((edges["x_name"] == "Second heart field") &
                 (edges["y_name"] == "Atrial cardiomyocytes")).any()
        if need:
            new_row = {
                "system": system_tag,
                "x": "L_M22",
                "y": "L_M5",
                "x_name": "Second heart field",
                "y_name": "Atrial cardiomyocytes",
                "edge_type": "Developmental progression",
                "x_number": np.nan, "y_number": np.nan,
                "x_id": np.nan, "y_id": np.nan,
            }
            edges = pd.concat([edges, pd.DataFrame([new_row])], ignore_index=True)
    return edges

def main():
    ap = argparse.ArgumentParser(description="Lineage graphs colored by SHH bins and raw-positives.")
    ap.add_argument("--extended-tsv", required=True, help=".../<system>/qc/<system>_extended_obs.tsv")
    ap.add_argument("--edges", default="/project/xyang2/SHH/Qiu_TimeLapse/Holly_desktop/edges_filtered.txt")
    ap.add_argument("--system", default="Lateral_plate_mesoderm")
    ap.add_argument("--gene", default="Gli1", help="Which gene to use for raw-positive plot")
    ap.add_argument("--low-thr", type=float, default=0.35)
    ap.add_argument("--high-thr", type=float, default=0.65)
    args = ap.parse_args()

    ext = Path(args.extended_tsv)
    outdir = ext.parent  # write plots next to the TSV
    outdir.mkdir(parents=True, exist_ok=True)

    df, edges = load_inputs(ext, Path(args.edges), args.system)
    edges = _ensure_manual_edges(edges, args.system)

    merged, tblA, tblB, tblC = aggregate_per_node(df, args.low_thr, args.high_thr, args.gene)

    # Save helper tables
    tblA.to_csv(outdir / "per_node_pct_shh_nonzero.csv", index=False)
    tblB.to_csv(outdir / f"per_node_pct_midhigh_{args.low_thr}_{args.high_thr}.csv", index=False)
    tblC.to_csv(outdir / f"per_node_pct_pos_raw_{args.gene}.csv", index=False)

    # Build graph and layout once
    G = _build_graph(edges)
    pos = _node_positions(G)

    # A) % SHH non-zero
    vals_A = dict(zip(merged["celltype_new"], merged["pct_shh_nonzero"]))
    _draw_graph(
        G, pos, vals_A,
        title=f"{args.system} – % cells with SHH_UCell_score > 0",
        out_png=outdir / "graph_pct_shh_nonzero.png",
        vlabel="% SHH>0"
    )

    # B) % Low or Middle or High
    # Only Middle+High
    vals_B = dict(zip(tblB["celltype_new"], tblB["pct_midhigh"]))

    if np.isclose(args.low_thr, args.high_thr):
        title = f"{args.system} – % cells with SHH_UCell_score > {args.high_thr}"
        vlabel = f"% SHH > {args.high_thr}"
    else:
        title = f"{args.system} – % cells with SHH in Middle/High (cutoffs {args.low_thr} / {args.high_thr})"
        vlabel = "% SHH in Middle/High"

    _draw_graph(
        G, pos, vals_B,
        title=title,
        out_png=outdir / f"graph_pct_shh_midhigh_{args.low_thr}_{args.high_thr}.png",
        vlabel=vlabel,
    )


    # C) % raw-positive for chosen gene
    vals_C = dict(zip(merged["celltype_new"], merged[f"pct_{args.gene}_pos"]))
    _draw_graph(
        G, pos, vals_C,
        title=f"{args.system} – % cells {args.gene} raw>0",
        out_png=outdir / f"graph_pct_pos_raw_{args.gene}.png",
        vlabel=f"% {args.gene} raw>0"
    )

    print("Saved graphs and per-node tables to:", outdir)

if __name__ == "__main__":
    main()
