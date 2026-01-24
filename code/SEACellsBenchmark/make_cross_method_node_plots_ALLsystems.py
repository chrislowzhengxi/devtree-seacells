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

    # for sys in sorted(df["system"].unique()):
    #     sub = df[df["system"] == sys]
    #     plt.scatter(sub[xcol], sub[ycol], s=14, label=sys)

    for sys in SYSTEMS_IN_ORDER:
        sub = df[df["system"] == sys]
        if len(sub) == 0:
            continue

        plt.scatter(
            sub[xcol],
            sub[ycol],
            s=18,
            color=PALETTE.get(sys, "black"),
            alpha=0.85,
            label=sys,
            edgecolors="none",
        )


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

    df = df[df["system"] != "Pre_gastrulation"].copy()

    plot_colored_by_system(
        df,
        "mean_ucell",
        "mean_scanpy",
        os.path.join(args.out_dir, "ALL_meanUCell_vs_meanScanpy.pdf"),
        "ALL systems: mean_UCell vs mean_scanpy (colored by system)",
    )

    plot_colored_by_system(
        df,
        "pct_ucell",
        "pct_scanpy",
        os.path.join(args.out_dir, "ALL_pctUCell_vs_pctScanpy.pdf"),
        "ALL systems: %_UCell vs %_scanpy (colored by system)",
    )

    plot_colored_by_system(
        df,
        "mean_scanpy",
        "pct_scanpy",
        os.path.join(args.out_dir, "ALL_meanScanpy_vs_pctScanpy.pdf"),
        "ALL systems: mean_scanpy vs %_scanpy (colored by system)",
    )


if __name__ == "__main__":
    main()



