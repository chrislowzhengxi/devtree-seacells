"""
python /project/imoskowitz/xyang2/chrislowzhengxi/code/SEACellsBenchmark/make_scanpy_node_table_and_plot.py \
  --edges_csv /project/imoskowitz/xyang2/chrislowzhengxi/results/score_genes_new/full_scored_edges_with_pregastrulation_scoregenes.csv \
  --out_dir /project/imoskowitz/xyang2/chrislowzhengxi/results/LPM_cross_method \
  --system_name ALL
"""
import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams['pdf.fonttype'] = 42  
plt.rcParams['ps.fonttype'] = 42


SYSTEMS_IN_ORDER = [
    "Blood",
    "Brain_spinal_cord",
    "Endothelium",
    "Epithelial_cells",
    "Eye",
    "Gastrulation",
    "Gut",
    "Lateral_plate_mesoderm",
    "Mesoderm",
    "Notochord",
    "PNS_glia",
    "PNS_neurons",
    "Pre_gastrulation",
    "Renal",
]

PALETTE = {
    "Blood": "#E41A1C",
    "Brain_spinal_cord": "#864F70",
    "Endothelium": "#3881AF",
    "Epithelial_cells": "#00CED1",
    "Eye": "#6FBF73",
    "Gastrulation": "#CAB2D6",
    "Gut": "#6A3D9A",
    "Lateral_plate_mesoderm": "#C65A14",
    "Mesoderm": "#D3D3D3",
    "Notochord": "#FFEB2B",
    "PNS_glia": "#DCBD2E",
    "PNS_neurons": "#800000",
    "Renal": "#F781BF",
}


def build_nodes_from_edges(df_edges: pd.DataFrame) -> pd.DataFrame:
    # x-side nodes
    df_x = pd.DataFrame()
    df_x["system"] = df_edges["system"].astype(str)
    df_x["node_id"] = df_edges["x"].astype(str)
    df_x["node_name"] = df_edges["x_name"].astype(str)
    df_x["n_cells"] = df_edges["n_x"].astype(float)
    df_x["median"] = df_edges["median_x"].astype(float)
    df_x["mean_scanpy"] = df_edges["sh_x"].astype(float)
    df_x["q90"] = df_edges["q90_x"].astype(float)
    df_x["pct_scanpy"] = df_edges["frac>0_x"].astype(float)
    df_x["variance"] = df_edges["variance_x"].astype(float)
    df_x["std"] = df_edges["std_x"].astype(float)

    # y-side nodes
    df_y = pd.DataFrame()
    df_y["system"] = df_edges["system"].astype(str)
    df_y["node_id"] = df_edges["y"].astype(str)
    df_y["node_name"] = df_edges["y_name"].astype(str)
    df_y["n_cells"] = df_edges["n_y"].astype(float)
    df_y["median"] = df_edges["median_y"].astype(float)
    df_y["mean_scanpy"] = df_edges["sh_y"].astype(float)
    df_y["q90"] = df_edges["q90_y"].astype(float)
    df_y["pct_scanpy"] = df_edges["frac>0_y"].astype(float)
    df_y["variance"] = df_edges["variance_y"].astype(float)
    df_y["std"] = df_edges["std_y"].astype(float)

    # stack and deduplicate
    df_nodes = pd.concat([df_x, df_y], axis=0, ignore_index=True)
    df_nodes = df_nodes.drop_duplicates(subset=["system", "node_id"], keep="first")

    df_nodes = df_nodes.replace([np.inf, -np.inf], np.nan)
    df_nodes = df_nodes.dropna(subset=["mean_scanpy", "pct_scanpy"])

    return df_nodes


# def plot_mean_vs_pct_scanpy(df_nodes: pd.DataFrame, out_png: str, title: str):
#     x = df_nodes["mean_scanpy"].to_numpy()
#     y = df_nodes["pct_scanpy"].to_numpy()

#     rho = pd.Series(x).corr(pd.Series(y), method="spearman")

#     plt.figure(figsize=(6.5, 5.5))
#     plt.scatter(x, y, s=14)

#     # regression line for visualization
#     if len(x) >= 2:
#         m, b = np.polyfit(x, y, 1)
#         xs = np.array([np.min(x), np.max(x)])
#         ys = m * xs + b
#         plt.plot(xs, ys)

#     plt.xlabel("mean_scanpy (node mean)")
#     plt.ylabel("%_scanpy (node frac>threshold)")
#     plt.title(f"{title}\nSpearman rho = {rho:.3f}")

#     plt.tight_layout()
#     plt.savefig(out_png, dpi=300)
#     print(f"[SAVE] {out_png}")


def plot_mean_vs_pct_scanpy(df_nodes: pd.DataFrame, out_png: str, title: str):
    rho = df_nodes["mean_scanpy"].corr(df_nodes["pct_scanpy"], method="spearman")

    plt.figure(figsize=(7.5, 6))

    # plot each system separately
    for system in sorted(df_nodes["system"].unique()):
        # sub = df_nodes[df_nodes["system"] == system]
        # plt.scatter(
        #     sub["mean_scanpy"],
        #     sub["pct_scanpy"],
        #     s=18,
        #     alpha=0.8,
        #     label=system
        # )
        sub = df_nodes[df_nodes["system"] == system]
        if len(sub) == 0:
            continue

        plt.scatter(
            sub["mean_scanpy"],
            sub["pct_scanpy"],
            s=18,
            color=PALETTE.get(system, "black"),
            alpha=0.85,
            label=system,
            edgecolors="none",
        )

    # regression line across ALL nodes
    x = df_nodes["mean_scanpy"].to_numpy()
    y = df_nodes["pct_scanpy"].to_numpy()

    if len(x) >= 2:
        m, b = np.polyfit(x, y, 1)
        xs = np.array([x.min(), x.max()])
        ys = m * xs + b
        plt.plot(xs, ys)

    plt.xlabel("mean_scanpy (node mean)")
    plt.ylabel("%_scanpy (node frac > threshold)")
    plt.title(f"{title}\nSpearman rho = {rho:.3f}")

    plt.legend(
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        frameon=False
    )


    plt.tight_layout()

    # PNG
    plt.savefig(out_png, dpi=300, bbox_inches="tight")

    # PDF (vector)
    out_pdf = out_png.replace(".png", ".pdf")
    plt.savefig(out_pdf, bbox_inches="tight")

    plt.close()
    print(f"[SAVE] {out_png}")
    print(f"[SAVE] {out_pdf}")



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--edges_csv", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--system_name", default="ALL")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    df_edges = pd.read_csv(args.edges_csv)
    df_nodes = build_nodes_from_edges(df_edges)

    out_nodes = os.path.join(
        args.out_dir, f"{args.system_name}_nodes_scanpy_summary.csv"
    )
    df_nodes.to_csv(out_nodes, index=False)
    print(f"[SAVE] {out_nodes}")

    out_png = os.path.join(
        args.out_dir, f"{args.system_name}_dot_meanScanpy_vs_pctScanpy.png"
    )
    plot_mean_vs_pct_scanpy(
        df_nodes,
        out_png,
        title=f"{args.system_name} node dots: mean_scanpy vs %_scanpy",
    )


if __name__ == "__main__":
    main()
