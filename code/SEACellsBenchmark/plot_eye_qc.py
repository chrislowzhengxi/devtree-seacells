#!/usr/bin/env python3
"""
plot_all_systems_qc.py  –  Generate Holly’s six QC bar-plots for every system
"""

import os, random, sys
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
import matplotlib.pyplot as plt

# ───────────────────────── CONFIG ─────────────────────────────────── #
RAW_DIR      = Path("/project/imoskowitz/xyang2/SHH/Qiu_TimeLapse/results/raw_added")
CSV_META     = Path("/project/imoskowitz/xyang2/chrislowzhengxi/data/df_cell_celltyp_new_merged.csv")
RESULTS_ROOT = Path("/project/imoskowitz/xyang2/chrislowzhengxi/results/all_systems_qc")
MIN_CELLS_PER_EMBRYO = 50        # drop very tiny embryos
TOP_N_EMBRYOS        = 20        # for cleaner “cells per embryo” plot

# plotting style
random.seed(0);  np.random.seed(0)
sns.set_style("ticks")
plt.rcParams.update({"figure.dpi":300, "pdf.fonttype":42, "ps.fonttype":42})

# ─────────────────────── HELPER FUNCTIONS ─────────────────────────── #
def tagged(fname, system):
    b, e = os.path.splitext(fname)
    return f"{b}_{system}{e}"

def save(fig, outdir, fname, system):
    fig.tight_layout()
    fig.savefig(outdir / tagged(fname, system))
    plt.close(fig)

def barplot(df, x, y, hue, title, outdir, fname, system, figsize=(6,4), horiz=False):
    fig = plt.figure(figsize=figsize)
    if horiz:
        sns.barplot(data=df, y=x, x=y, hue=hue, dodge=False)
        plt.ylabel(x); plt.xlabel(y)
    else:
        sns.barplot(data=df, x=x, y=y, hue=hue, dodge=False)
        plt.xticks(rotation=90, ha="right")
    plt.title(title);  sns.despine()
    save(fig, outdir, fname, system)

# ─────────────────── INITIALISE ONE SYSTEM ────────────────────────── #
def init_system(h5, csv_meta, min_cells):
    system = h5.name.split("_")[0]
    adata  = sc.read_h5ad(h5)

    # simplest: drop `.raw`
    adata.raw = None      

    # … keep raw but subset it in parallel
    df_meta = (pd.read_csv(csv_meta,
               usecols=["cell_id","day","embryo_id","system","celltype_new"])
               .query("system == @system")
               .drop_duplicates("cell_id"))
    keep    = adata.obs["cell_id"].isin(df_meta["cell_id"]).values
    # subset obs / layers
    adata   = adata[keep].copy()

    idx = df_meta.set_index("cell_id")
    for col in ["day","embryo_id","system","celltype_new"]:
        adata.obs[col] = adata.obs["cell_id"].map(idx[col])

    # drop tiny embryos
    big_emb = adata.obs["embryo_id"].value_counts()
    keep_emb = big_emb[big_emb >= min_cells].index
    mask = adata.obs["embryo_id"].isin(keep_emb).values
    adata = adata[mask].copy()

    return adata

# ────────────────   PLOT THE SIX QC FIGURES   ─────────────────────── #
def qc_plots(adata, outdir, system):
    obs = adata.obs

    # 1  – top-20 embryos by cell count (horizontal for readability)
    df_emb = (obs.groupby(["system","embryo_id"]).size()
                 .reset_index(name="n_cells")
                 .nlargest(TOP_N_EMBRYOS, "n_cells")
                 .sort_values("n_cells"))
    barplot(df_emb, "embryo_id", "n_cells", "system",
            f"Top {TOP_N_EMBRYOS} embryos (cells)", outdir,
            "cells_per_topN_embryo.pdf", system, figsize=(5,8), horiz=True)

    # 2  – total cells for THIS system
    df_tot = pd.DataFrame({
        "system": [system],
        "n_cells": [obs.shape[0]]
    })
    barplot(df_tot, "system", "n_cells", None,
            "Total cells per system", outdir,
            "total_cells_per_system.pdf", system)

    # 3/4/5/6 – day/system cross-tabs
    df_cells = (obs.groupby(["day","system"]).size()
                .reset_index(name="n_cells"))
    df_embry = (obs.drop_duplicates(["embryo_id","day","system"])
                   .groupby(["day","system"]).size()
                   .reset_index(name="n_embryos"))

    barplot(df_cells,  "day","n_cells","system",
            "Cells per day", outdir,"cells_per_day_per_system.pdf",system)
    barplot(df_embry, "day","n_embryos","system",
            "Embryos per day", outdir,"embryos_per_day_per_system.pdf",system)
    barplot(df_cells,  "system","n_cells","day",
            "Cells per system (col by day)", outdir,
            "cells_per_system_per_day.pdf",system)
    barplot(df_embry, "system","n_embryos","day",
            "Embryos per system (col by day)", outdir,
            "embryos_per_system_per_day.pdf",system)

# ---------------------------------------------------------------------#
# 7) CROSS‐SYSTEM COMPARISONS                                         #
# ---------------------------------------------------------------------#
def cross_system_plots(csv_meta, raw_dir, outdir):
    # 1) Total cells per system (counting obs in each h5ad)
    totals = []
    for h5 in sorted(raw_dir.glob("*_adata_scale.h5ad")):
        sys_tag = h5.name.split("_")[0]
        ad = sc.read_h5ad(h5)
        totals.append({"system": sys_tag, "n_cells": ad.n_obs})
    df_tot = pd.DataFrame(totals)
    barplot(df_tot, "system", "n_cells", None,
            "Total cells per system (all)", outdir,
            "total_cells_per_system_all.pdf", system="", horiz=False)

    # 2) Cells per day coloured by system
    #    (we can pull day & system from the master CSV)
    df_meta = (pd.read_csv(csv_meta, usecols=["cell_id","day","system"])
                  .drop_duplicates("cell_id"))
    df_cd   = df_meta.groupby(["day","system"]).size().reset_index(name="n_cells")
    barplot(df_cd, "day", "n_cells", "system",
            "Cells per day (all systems)", outdir,
            "cells_per_day_all_systems.pdf", system="", horiz=False)

    # 3) Embryos per day coloured by system
    df_ed = (df_meta.drop_duplicates(["system","day","cell_id"])
                   .groupby(["day","system"])
                   .size().reset_index(name="n_embryos"))
    barplot(df_ed, "day", "n_embryos", "system",
            "Embryos per day (all systems)", outdir,
            "embryos_per_day_all_systems.pdf", system="", horiz=False)

    # 4/5) Cells & Embryos per system (col by day)
    barplot(df_cd, "system", "n_cells", "day",
            "Cells per system (by day)", outdir,
            "cells_per_system_all.pdf", system="", horiz=False)
    barplot(df_ed, "system", "n_embryos", "day",
            "Embryos per system (by day)", outdir,
            "embryos_per_system_all.pdf", system="", horiz=False)


# ───────────────────────────── MAIN ───────────────────────────────── #
def main():
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)

    for h5 in sorted(RAW_DIR.glob("*_adata_scale.h5ad")):
        system = h5.name.split("_")[0]
        outdir = RESULTS_ROOT / system / "figures"
        outdir.mkdir(parents=True, exist_ok=True)

        print(f"\n▶ {system}")
        adata = init_system(h5, CSV_META, MIN_CELLS_PER_EMBRYO)
        print(f"   {adata.n_obs:,} cells | {adata.obs['embryo_id'].nunique()} embryos")

        qc_plots(adata, outdir, system)
        print(f"   QC plots → {outdir}")

if __name__ == "__main__":
    main()
    CROSS_DIR = RESULTS_ROOT / "ALL_SYSTEMS"
    CROSS_DIR.mkdir(exist_ok=True)
    cross_system_plots(CSV_META, RAW_DIR, CROSS_DIR)
    print("✓ Cross‐system comparison plots →", CROSS_DIR)
