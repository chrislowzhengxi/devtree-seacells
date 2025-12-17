import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def build_nodes_from_edges(df_edges: pd.DataFrame) -> pd.DataFrame:
    # Build node table from the x side
    df_x = pd.DataFrame()
    df_x["system"] = df_edges["system"].astype(str)
    df_x["node_id"] = df_edges["x"].astype(str)
    df_x["node_name"] = df_edges["x_name"].astype(str)
    df_x["n_cells"] = df_edges["n_x"].astype(float)
    df_x["median"] = df_edges["median_x"].astype(float)
    df_x["mean_ucell"] = df_edges["sh_x"].astype(float)
    df_x["q90"] = df_edges["q90_x"].astype(float)
    df_x["pct_ucell"] = df_edges["frac>0_x"].astype(float)
    df_x["variance"] = df_edges["variance_x"].astype(float)
    df_x["std"] = df_edges["std_x"].astype(float)

    # Build node table from the y side
    df_y = pd.DataFrame()
    df_y["system"] = df_edges["system"].astype(str)
    df_y["node_id"] = df_edges["y"].astype(str)
    df_y["node_name"] = df_edges["y_name"].astype(str)
    df_y["n_cells"] = df_edges["n_y"].astype(float)
    df_y["median"] = df_edges["median_y"].astype(float)
    df_y["mean_ucell"] = df_edges["sh_y"].astype(float)
    df_y["q90"] = df_edges["q90_y"].astype(float)
    df_y["pct_ucell"] = df_edges["frac>0_y"].astype(float)
    df_y["variance"] = df_edges["variance_y"].astype(float)
    df_y["std"] = df_edges["std_y"].astype(float)

    # Stack and deduplicate
    df_nodes_raw = pd.concat([df_x, df_y], axis=0, ignore_index=True)

    # If a node appears multiple times, keep the first instance.
    # These stats should be identical across edges, so dropping duplicates is fine.
    df_nodes = df_nodes_raw.drop_duplicates(subset=["system", "node_id"], keep="first").copy()

    # Clean up any obvious bad rows
    df_nodes = df_nodes.replace([np.inf, -np.inf], np.nan)
    df_nodes = df_nodes.dropna(subset=["mean_ucell", "pct_ucell"])

    return df_nodes


def plot_mean_vs_pct_ucell(df_nodes: pd.DataFrame, out_png: str, title: str):
    x = df_nodes["mean_ucell"].astype(float).to_numpy()
    y = df_nodes["pct_ucell"].astype(float).to_numpy()

    # Spearman correlation
    rho = pd.Series(x).corr(pd.Series(y), method="spearman")

    plt.figure(figsize=(6.5, 5.5))
    plt.scatter(x, y, s=14)

    # Regression line for visualization
    if len(x) >= 2:
        m, b = np.polyfit(x, y, 1)
        xs = np.array([np.min(x), np.max(x)], dtype=float)
        ys = m * xs + b
        plt.plot(xs, ys)

    plt.xlabel("mean_UCell (node mean)")
    plt.ylabel("%_UCell (node frac>0)")
    plt.title(f"{title}\nSpearman rho = {rho:.3f}")

    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    print(f"[SAVE] {out_png}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--edges_csv", required=True, help="Edge-level CSV with *_x and *_y node stats")
    ap.add_argument("--out_dir", required=True, help="Directory to write outputs")
    ap.add_argument("--system_name", default="Lateral_plate_mesoderm", help="For plot title")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    df_edges = pd.read_csv(args.edges_csv)

    df_nodes = build_nodes_from_edges(df_edges)

    out_nodes_csv = os.path.join(args.out_dir, f"{args.system_name}_nodes_ucell_summary.csv")
    df_nodes.to_csv(out_nodes_csv, index=False)
    print(f"[SAVE] {out_nodes_csv}")

    out_png = os.path.join(args.out_dir, f"{args.system_name}_dot_meanUCell_vs_pctUCell.png")
    plot_mean_vs_pct_ucell(
        df_nodes=df_nodes,
        out_png=out_png,
        title=f"{args.system_name} node dots: mean_UCell vs %_UCell"
    )


if __name__ == "__main__":
    main()
