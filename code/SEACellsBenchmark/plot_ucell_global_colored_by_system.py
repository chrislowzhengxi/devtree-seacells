import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def build_nodes_from_edges(df_edges: pd.DataFrame) -> pd.DataFrame:
    df_x = pd.DataFrame()
    df_x["system"] = df_edges["system"].astype(str)
    df_x["node_id"] = df_edges["x"].astype(str)
    df_x["node_name"] = df_edges["x_name"].astype(str)
    df_x["n_cells"] = df_edges["n_x"].astype(float)
    df_x["mean_ucell"] = df_edges["sh_x"].astype(float)
    df_x["pct_ucell"] = df_edges["frac>0_x"].astype(float)

    df_y = pd.DataFrame()
    df_y["system"] = df_edges["system"].astype(str)
    df_y["node_id"] = df_edges["y"].astype(str)
    df_y["node_name"] = df_edges["y_name"].astype(str)
    df_y["n_cells"] = df_edges["n_y"].astype(float)
    df_y["mean_ucell"] = df_edges["sh_y"].astype(float)
    df_y["pct_ucell"] = df_edges["frac>0_y"].astype(float)

    df_nodes = pd.concat([df_x, df_y], axis=0, ignore_index=True)
    df_nodes = df_nodes.drop_duplicates(subset=["system", "node_id"], keep="first").copy()

    df_nodes = df_nodes.replace([np.inf, -np.inf], np.nan)
    df_nodes = df_nodes.dropna(subset=["mean_ucell", "pct_ucell"])

    return df_nodes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--edges_csv", required=True)
    ap.add_argument("--out_png", required=True)
    ap.add_argument("--title", default="ALL systems: mean_UCell vs %_UCell (colored by system)")
    ap.add_argument("--legend_top_n", type=int, default=12,
                    help="Show legend only for top N systems by node count. 0 means no legend.")
    args = ap.parse_args()

    df_edges = pd.read_csv(args.edges_csv)
    df_nodes = build_nodes_from_edges(df_edges)

    # Spearman for all nodes combined
    rho = df_nodes["mean_ucell"].corr(df_nodes["pct_ucell"], method="spearman")

    # Determine which systems to show in legend
    sys_counts = df_nodes["system"].value_counts()
    if args.legend_top_n > 0:
        top_systems = list(sys_counts.index[:args.legend_top_n])
    else:
        top_systems = []

    plt.figure(figsize=(7.2, 6.2))

    # Plot top systems with labels
    for sys in top_systems:
        sub = df_nodes[df_nodes["system"] == sys]
        plt.scatter(sub["mean_ucell"], sub["pct_ucell"], s=12, label=sys)

    # Plot remaining systems without cluttering legend
    if len(top_systems) > 0:
        other = df_nodes[~df_nodes["system"].isin(top_systems)]
        if len(other) > 0:
            plt.scatter(other["mean_ucell"], other["pct_ucell"], s=12)

    # Regression line for all points (visual only)
    x = df_nodes["mean_ucell"].to_numpy(dtype=float)
    y = df_nodes["pct_ucell"].to_numpy(dtype=float)
    if len(x) >= 2:
        m, b = np.polyfit(x, y, 1)
        xs = np.array([np.min(x), np.max(x)], dtype=float)
        ys = m * xs + b
        plt.plot(xs, ys)

    plt.xlabel("mean_UCell (node mean)")
    plt.ylabel("%_UCell (node frac>0)")
    plt.title(f"{args.title}\nSpearman rho = {rho:.3f}")

    if args.legend_top_n > 0:
        plt.legend(loc="best", fontsize=8, frameon=False)

    plt.tight_layout()
    os.makedirs(os.path.dirname(args.out_png), exist_ok=True)
    plt.savefig(args.out_png, dpi=300)
    plt.close()
    print(f"[SAVE] {args.out_png}")


if __name__ == "__main__":
    main()
