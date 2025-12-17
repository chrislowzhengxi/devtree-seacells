import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plot_scatter_with_spearman(
    x, y, xlabel, ylabel, title, out_png
):
    rho = pd.Series(x).corr(pd.Series(y), method="spearman")

    plt.figure(figsize=(6.5, 5.5))
    plt.scatter(x, y, s=14)

    # regression line (visual only)
    if len(x) >= 2:
        m, b = np.polyfit(x, y, 1)
        xs = np.array([np.min(x), np.max(x)])
        ys = m * xs + b
        plt.plot(xs, ys)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(f"{title}\nSpearman rho = {rho:.3f}")
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    print(f"[SAVE] {out_png}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ucell_nodes", required=True)
    ap.add_argument("--scanpy_nodes", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--system_name", default="Lateral_plate_mesoderm")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    df_u = pd.read_csv(args.ucell_nodes)
    df_s = pd.read_csv(args.scanpy_nodes)

    # merge on node
    df = df_u.merge(
        df_s,
        on=["system", "node_id"],
        suffixes=("_ucell", "_scanpy"),
        how="inner",
    )

    # ---- plot 1: mean_UCell vs mean_scanpy ----
    plot_scatter_with_spearman(
        x=df["mean_ucell"].to_numpy(),
        y=df["mean_scanpy"].to_numpy(),
        xlabel="mean_UCell (node mean)",
        ylabel="mean_scanpy (node mean)",
        title=f"{args.system_name}: mean_UCell vs mean_scanpy",
        out_png=os.path.join(
            args.out_dir,
            f"{args.system_name}_dot_meanUCell_vs_meanScanpy.png",
        ),
    )

    # ---- plot 2: %_UCell vs %_scanpy ----
    plot_scatter_with_spearman(
        x=df["pct_ucell"].to_numpy(),
        y=df["pct_scanpy"].to_numpy(),
        xlabel="%_UCell (node frac>threshold)",
        ylabel="%_scanpy (node frac>threshold)",
        title=f"{args.system_name}: %_UCell vs %_scanpy",
        out_png=os.path.join(
            args.out_dir,
            f"{args.system_name}_dot_pctUCell_vs_pctScanpy.png",
        ),
    )


if __name__ == "__main__":
    main()
