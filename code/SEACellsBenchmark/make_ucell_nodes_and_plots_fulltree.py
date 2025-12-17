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
    df_x["median"] = df_edges["median_x"].astype(float)
    df_x["mean_ucell"] = df_edges["sh_x"].astype(float)
    df_x["q90"] = df_edges["q90_x"].astype(float)
    df_x["pct_ucell"] = df_edges["frac>0_x"].astype(float)
    df_x["variance"] = df_edges["variance_x"].astype(float)
    df_x["std"] = df_edges["std_x"].astype(float)

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

    df_nodes_raw = pd.concat([df_x, df_y], axis=0, ignore_index=True)

    df_nodes = df_nodes_raw.drop_duplicates(
        subset=["system", "node_id"],
        keep="first"
    ).copy()

    df_nodes = df_nodes.replace([np.inf, -np.inf], np.nan)
    df_nodes = df_nodes.dropna(subset=["mean_ucell", "pct_ucell"])

    return df_nodes


def plot_mean_vs_pct_ucell(df_nodes: pd.DataFrame, out_png: str, title: str):
    x = df_nodes["mean_ucell"].to_numpy(dtype=float)
    y = df_nodes["pct_ucell"].to_numpy(dtype=float)

    rho = pd.Series(x).corr(pd.Series(y), method="spearman")

    plt.figure(figsize=(6.5, 5.5))
    plt.scatter(x, y, s=12)

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
    plt.close()
    print(f"[SAVE] {out_png}")


def safe_name(s: str) -> str:
    return "".join(c if c.isalnum() or c in ["_", "-"] else "_" for c in s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--edges_csv", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--system_filter", default="", help="If set, only plot this one system")
    ap.add_argument("--per_system", action="store_true", help="If set, write one plot per system")
    ap.add_argument("--write_nodes_csv", action="store_true", help="Write node-level CSV(s)")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    df_edges = pd.read_csv(args.edges_csv)
    df_nodes = build_nodes_from_edges(df_edges)

    if args.system_filter:
        df_nodes = df_nodes[df_nodes["system"] == args.system_filter].copy()

    if args.per_system:
        for sys in sorted(df_nodes["system"].unique()):
            sub = df_nodes[df_nodes["system"] == sys].copy()

            if args.write_nodes_csv:
                out_nodes_csv = os.path.join(args.out_dir, f"{safe_name(sys)}_nodes_ucell_summary.csv")
                sub.to_csv(out_nodes_csv, index=False)
                print(f"[SAVE] {out_nodes_csv}")

            out_png = os.path.join(args.out_dir, f"{safe_name(sys)}_dot_meanUCell_vs_pctUCell.png")
            plot_mean_vs_pct_ucell(
                sub,
                out_png=out_png,
                title=f"{sys} node dots: mean_UCell vs %_UCell"
            )
    else:
        if args.write_nodes_csv:
            out_nodes_csv = os.path.join(args.out_dir, "ALL_systems_nodes_ucell_summary.csv")
            df_nodes.to_csv(out_nodes_csv, index=False)
            print(f"[SAVE] {out_nodes_csv}")

        out_png = os.path.join(args.out_dir, "ALL_systems_dot_meanUCell_vs_pctUCell.png")
        plot_mean_vs_pct_ucell(
            df_nodes,
            out_png=out_png,
            title="ALL systems node dots: mean_UCell vs %_UCell"
        )


if __name__ == "__main__":
    main()
