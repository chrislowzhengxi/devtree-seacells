#!/usr/bin/env python3
import os
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------
# Paths (edit if needed)
EDGE_CSV = "/project/imoskowitz/xyang2/chrislowzhengxi/results/ucell/Lateral_plate_mesoderm/Lateral_plate_mesoderm_edge_filtered_with_shh.csv"
OBS_TSV  = "/project/imoskowitz/xyang2/chrislowzhengxi/results/ucell/Lateral_plate_mesoderm/qc/Lateral_plate_mesoderm_extended_obs.tsv"
OUT_PDF  = "/project/imoskowitz/xyang2/chrislowzhengxi/results/ucell/Lateral_plate_mesoderm/qc/LPM_tree_Gli1_ge0p35.pdf"

SYSTEM_LABEL = "Lateral_plate_mesoderm"
M_THR = 0.35   # "medium" threshold, same as SHH workflow
# ---------------------------------------------------------------------


def build_graph_from_edges(edge_csv):
    """Build a directed graph whose nodes are celltype names."""
    edges = pd.read_csv(edge_csv)

    G = nx.DiGraph()
    for _, row in edges.iterrows():
        # if there is a 'system' column, keep only LPM rows
        if "system" in row and row["system"] != SYSTEM_LABEL:
            continue

        src = row.get("x_name", row.get("x_id"))
        dst = row.get("y_name", row.get("y_id"))
        if pd.isna(src) or pd.isna(dst):
            continue

        # edge weight; fall back to 1.0 if not present
        w = row.get("abs_delta", np.nan)
        if pd.isna(w):
            w = row.get("delta", 1.0)

        G.add_edge(str(src), str(dst), weight=float(w))

    return G


def compute_gli1_pct_per_node(obs_tsv, m_thr):
    """Return DataFrame with pct of cells Gli1_norm >= m_thr per celltype_new."""
    df = pd.read_csv(obs_tsv, sep="\t")

    df["Gli1_ge_M"] = df["Gli1_norm"] >= m_thr

    summary = (
        df.groupby("celltype_new")["Gli1_ge_M"]
          .mean()
          .reset_index(name="pct_Gli1_ge_M")
    )
    # this is a fraction in [0,1]; convert to percent if you prefer later
    return summary


def plot_tree_with_gli1(G, gli_df, out_pdf):
    """Color nodes by pct_Gli1_ge_M and draw tree."""
    # map node name -> pct
    pct_map = dict(zip(gli_df["celltype_new"].astype(str),
                       gli_df["pct_Gli1_ge_M"]))

    # attach as node attributes (missing nodes get NaN)
    nx.set_node_attributes(G, {n: pct_map.get(n, np.nan)
                               for n in G.nodes()}, "pct_Gli1_ge_M")

    # layout
    try:
        from networkx.drawing.nx_agraph import graphviz_layout
        pos = graphviz_layout(G, prog="dot")
    except Exception:
        pos = nx.spring_layout(G, seed=0)

    # node colors
    node_values = np.array([
        G.nodes[n].get("pct_Gli1_ge_M", np.nan) for n in G.nodes()
    ], dtype=float)

    vmin = np.nanmin(node_values)
    vmax = np.nanmax(node_values)
    if not np.isfinite(vmin) or not np.isfinite(vmax):
        vmin, vmax = 0.0, 1.0

    # nodes with NaN will be light gray
    cmap = plt.cm.Blues

    plt.figure(figsize=(10, 6))

    # draw edges first
    edge_weights = [G[u][v].get("weight", 1.0) for u, v in G.edges()]
    max_w = max(edge_weights) if edge_weights else 1.0
    # rescale widths a bit
    widths = [1.0 + 4.0 * (w / max_w) for w in edge_weights]

    nx.draw_networkx_edges(
        G, pos,
        arrows=True,
        arrowstyle="-|>",
        arrowsize=10,
        width=widths,
        edge_color="black",
        alpha=0.8,
    )

    # draw nodes, coloring by Gli1 percentage
    nodes = nx.draw_networkx_nodes(
        G,
        pos,
        node_color=node_values,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        node_size=600,
        linewidths=0.5,
        edgecolors="black",
    )

    # labels
    nx.draw_networkx_labels(
        G,
        pos,
        font_size=7,
        font_color="darkred",
    )

    # colorbar
    cbar = plt.colorbar(nodes)
    cbar.set_label("Fraction of cells with Gli1_norm ≥ 0.35")

    plt.title(f"{SYSTEM_LABEL}: nodes colored by % Gli1_norm ≥ 0.35")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_pdf)
    plt.close()
    print("Saved tree figure:", out_pdf)


def main():
    os.makedirs(os.path.dirname(OUT_PDF), exist_ok=True)

    print("Building graph from:", EDGE_CSV)
    G = build_graph_from_edges(EDGE_CSV)

    print("Computing Gli1 fractions from:", OBS_TSV)
    gli_df = compute_gli1_pct_per_node(OBS_TSV, M_THR)

    print("Plotting tree with Gli1 node colors")
    plot_tree_with_gli1(G, gli_df, OUT_PDF)


if __name__ == "__main__":
    main()
