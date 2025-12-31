# import argparse
# import os
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt


# def build_nodes_from_edges(df_edges: pd.DataFrame, prefix: str) -> pd.DataFrame:
#     # prefix is just for column naming downstream, not used for reading
#     df_x = pd.DataFrame()
#     df_x["system"] = df_edges["system"].astype(str)
#     df_x["node_id"] = df_edges["x"].astype(str)
#     df_x[f"mean_{prefix}"] = df_edges["sh_x"].astype(float)
#     df_x[f"pct_{prefix}"] = df_edges["frac>0_x"].astype(float)

#     df_y = pd.DataFrame()
#     df_y["system"] = df_edges["system"].astype(str)
#     df_y["node_id"] = df_edges["y"].astype(str)
#     df_y[f"mean_{prefix}"] = df_edges["sh_y"].astype(float)
#     df_y[f"pct_{prefix}"] = df_edges["frac>0_y"].astype(float)

#     df_nodes = pd.concat([df_x, df_y], axis=0, ignore_index=True)
#     df_nodes = df_nodes.drop_duplicates(subset=["system", "node_id"], keep="first").copy()

#     df_nodes = df_nodes.replace([np.inf, -np.inf], np.nan)
#     df_nodes = df_nodes.dropna(subset=[f"mean_{prefix}", f"pct_{prefix}"])

#     return df_nodes


# def plot_colored_by_system(df, xcol, ycol, out_png, title, legend_top_n=12):
#     rho = df[xcol].corr(df[ycol], method="spearman")

#     sys_counts = df["system"].value_counts()
#     top_systems = list(sys_counts.index[:legend_top_n]) if legend_top_n > 0 else []

#     plt.figure(figsize=(7.2, 6.2))

#     for sys in top_systems:
#         sub = df[df["system"] == sys]
#         plt.scatter(sub[xcol], sub[ycol], s=12, label=sys)

#     if len(top_systems) > 0:
#         other = df[~df["system"].isin(top_systems)]
#         if len(other) > 0:
#             plt.scatter(other[xcol], other[ycol], s=12)

#     # regression line for all points (visual only)
#     x = df[xcol].to_numpy(dtype=float)
#     y = df[ycol].to_numpy(dtype=float)
#     if len(x) >= 2:
#         m, b = np.polyfit(x, y, 1)
#         xs = np.array([np.min(x), np.max(x)], dtype=float)
#         ys = m * xs + b
#         plt.plot(xs, ys)

#     plt.xlabel(xcol)
#     plt.ylabel(ycol)
#     plt.title(f"{title}\nSpearman rho = {rho:.3f}")

#     if legend_top_n > 0:
#         plt.legend(loc="best", fontsize=8, frameon=False)

#     plt.tight_layout()
#     os.makedirs(os.path.dirname(out_png), exist_ok=True)
#     plt.savefig(out_png, dpi=300)
#     plt.close()
#     print(f"[SAVE] {out_png}")


# def main():
#     ap = argparse.ArgumentParser()
#     ap.add_argument("--ucell_edges", required=True)
#     ap.add_argument("--scanpy_edges", required=True)
#     ap.add_argument("--out_dir", required=True)
#     ap.add_argument("--legend_top_n", type=int, default=12)
#     args = ap.parse_args()

#     df_ucell_edges = pd.read_csv(args.ucell_edges)
#     df_scanpy_edges = pd.read_csv(args.scanpy_edges)
#     df_scanpy_edges = df_scanpy_edges[df_scanpy_edges["system"] != "Gastrulation_E8.5b"].copy()

#     needed = ["system", "x", "y", "sh_x", "sh_y", "frac>0_x", "frac>0_y"]
#     df_scanpy_edges = df_scanpy_edges.dropna(subset=needed).copy()

#     df_ucell_nodes = build_nodes_from_edges(df_ucell_edges, prefix="ucell")
#     df_scanpy_nodes = build_nodes_from_edges(df_scanpy_edges, prefix="scanpy")

#     # merge to align node sets
#     df = df_ucell_nodes.merge(df_scanpy_nodes, on=["system", "node_id"], how="inner")

#     # 1) mean_scanpy vs %_scanpy
#     plot_colored_by_system(
#         df=df,
#         xcol="mean_scanpy",
#         ycol="pct_scanpy",
#         out_png=os.path.join(args.out_dir, "ALL_systems_dot_meanScanpy_vs_pctScanpy_coloredBySystem.png"),
#         title="ALL systems: mean_scanpy vs %_scanpy (colored by system)",
#         legend_top_n=args.legend_top_n,
#     )

#     # 2) mean_UCell vs mean_scanpy
#     plot_colored_by_system(
#         df=df,
#         xcol="mean_ucell",
#         ycol="mean_scanpy",
#         out_png=os.path.join(args.out_dir, "ALL_systems_dot_meanUCell_vs_meanScanpy_coloredBySystem.png"),
#         title="ALL systems: mean_UCell vs mean_scanpy (colored by system)",
#         legend_top_n=args.legend_top_n,
#     )

#     # 3) %_UCell vs %_scanpy
#     plot_colored_by_system(
#         df=df,
#         xcol="pct_ucell",
#         ycol="pct_scanpy",
#         out_png=os.path.join(args.out_dir, "ALL_systems_dot_pctUCell_vs_pctScanpy_coloredBySystem.png"),
#         title="ALL systems: %_UCell vs %_scanpy (colored by system)",
#         legend_top_n=args.legend_top_n,
#     )


# if __name__ == "__main__":
#     main()


import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def build_nodes_from_edges(df_edges: pd.DataFrame, prefix: str) -> pd.DataFrame:
    df_x = pd.DataFrame({
        "system": df_edges["system"].astype(str),
        "node_id": df_edges["x"].astype(str),
        f"mean_{prefix}": df_edges["sh_x"].astype(float),
        f"pct_{prefix}": df_edges["frac>0_x"].astype(float),
    })

    df_y = pd.DataFrame({
        "system": df_edges["system"].astype(str),
        "node_id": df_edges["y"].astype(str),
        f"mean_{prefix}": df_edges["sh_y"].astype(float),
        f"pct_{prefix}": df_edges["frac>0_y"].astype(float),
    })

    df = pd.concat([df_x, df_y], ignore_index=True)
    df = df.drop_duplicates(subset=["system", "node_id"], keep="first")
    df = df.replace([np.inf, -np.inf], np.nan)

    return df


def plot_colored_by_system(df, xcol, ycol, out_png, title):
    df = df.dropna(subset=[xcol, ycol]).copy()

    rho = df[xcol].corr(df[ycol], method="spearman")

    plt.figure(figsize=(7.4, 6.2))

    for sys in sorted(df["system"].unique()):
        sub = df[df["system"] == sys]
        plt.scatter(sub[xcol], sub[ycol], s=14, label=sys)

    # regression line
    if len(df) >= 2:
        m, b = np.polyfit(df[xcol], df[ycol], 1)
        xs = np.array([df[xcol].min(), df[xcol].max()])
        plt.plot(xs, m * xs + b)

    plt.xlabel(xcol)
    plt.ylabel(ycol)
    plt.title(f"{title}\nSpearman rho = {rho:.3f}")

    plt.legend(
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        frameon=False,
        fontsize=8,
    )

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[SAVE] {out_png}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ucell_edges", required=True)
    ap.add_argument("--scanpy_edges", required=True)
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    df_ucell_edges = pd.read_csv(args.ucell_edges)
    df_scanpy_edges = pd.read_csv(args.scanpy_edges)

    df_ucell_nodes = build_nodes_from_edges(df_ucell_edges, "ucell")
    df_scanpy_nodes = build_nodes_from_edges(df_scanpy_edges, "scanpy")

    # OUTER merge keeps Gastrulation and partial overlaps
    df = df_ucell_nodes.merge(
        df_scanpy_nodes,
        on=["system", "node_id"],
        how="outer"
    )

    plot_colored_by_system(
        df,
        "mean_ucell",
        "mean_scanpy",
        os.path.join(args.out_dir, "ALL_meanUCell_vs_meanScanpy.png"),
        "ALL systems: mean_UCell vs mean_scanpy (colored by system)",
    )

    plot_colored_by_system(
        df,
        "pct_ucell",
        "pct_scanpy",
        os.path.join(args.out_dir, "ALL_pctUCell_vs_pctScanpy.png"),
        "ALL systems: %_UCell vs %_scanpy (colored by system)",
    )

    plot_colored_by_system(
        df,
        "mean_scanpy",
        "pct_scanpy",
        os.path.join(args.out_dir, "ALL_meanScanpy_vs_pctScanpy.png"),
        "ALL systems: mean_scanpy vs %_scanpy (colored by system)",
    )


if __name__ == "__main__":
    main()
