#!/usr/bin/env python3
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CSV = "/project/imoskowitz/xyang2/chrislowzhengxi/results/node_stage_plots/node_stage_ucell_summary.csv"
OUTDIR = "/project/imoskowitz/xyang2/chrislowzhengxi/results/node_stage_plots"

def scatter_by_system(df, xcol, ycol, title, out_pdf):
    plt.figure(figsize=(10, 7))

    for sys in sorted(df["system"].unique()):
        sub = df[df["system"] == sys].copy()
        s = (sub["n_cells"].clip(lower=1) ** 0.5) * 2.0
        plt.scatter(sub[xcol], sub[ycol], s=s, alpha=0.7, label=sys)

    plt.xlabel(xcol)
    plt.ylabel(ycol)
    plt.title(title)
    plt.legend(loc="best", fontsize=8, frameon=False)
    plt.tight_layout()
    plt.savefig(out_pdf, format="pdf")
    plt.close()

def main():
    outdir = Path(OUTDIR)
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(CSV)
    df = df.dropna(subset=["system", "node", "n_cells", "mean_stage", "median_stage", "mean_ucell"]).copy()

    scatter_by_system(
        df,
        xcol="mean_stage",
        ycol="mean_ucell",
        title="Mean UCell vs mean stage (per node)",
        out_pdf=str(outdir / "node_ucell_vs_mean_stage.pdf"),
    )

    scatter_by_system(
        df,
        xcol="median_stage",
        ycol="mean_ucell",
        title="Mean UCell vs median stage (per node)",
        out_pdf=str(outdir / "node_ucell_vs_median_stage.pdf"),
    )

    print("[DONE] wrote:")
    print(outdir / "node_ucell_vs_mean_stage.pdf")
    print(outdir / "node_ucell_vs_median_stage.pdf")

if __name__ == "__main__":
    main()
