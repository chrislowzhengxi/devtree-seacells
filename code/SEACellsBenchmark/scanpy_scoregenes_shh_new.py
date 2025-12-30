#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Scanpy SHH score_genes on normalized raw counts (Holly-style).

- Uses adata.raw only
- Normalizes raw counts (1e4)
- log1p transform
- No HVG restriction
- Does NOT modify adata.X
- Copies scores back to original AnnData
"""

from pathlib import Path
import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt

# -------------------- CONFIG --------------------

H5_ROOT = Path("/project/imoskowitz/xyang2/SHH/Qiu_TimeLapse/results/raw_added")
OUT_ROOT = Path("/project/imoskowitz/xyang2/chrislowzhengxi/results/score_genes_new")

# SYSTEMS = ["Blood"]  # Done
# SYSTEMS = ["Endothelium", "Epithelial_cells", "Eye", "Gut", "Notochord", "PNS_glia", "PNS_neurons", "Renal", "Lateral_plate_mesoderm", "Neurons"]
# amd-hm 43754137
SYSTEMS = ["Other_Brain_spinal_cord", "Mesoderm"]  # 43754199

SHH_GENES = ["Gli1", "Ptch1", "Hhip"]
SCORE_NAME = "SHH_scoregenes"

# ------------------------------------------------


def fix_var_index_column_conflict(adata):
    """
    Ensure adata.var (and adata.raw.var) do not have
    a column with the same name as the index.
    Required for writing h5ad safely.
    """
    if adata.var.index.name is not None:
        idx_name = adata.var.index.name
        if idx_name in adata.var.columns:
            adata.var = adata.var.rename(columns={idx_name: f"{idx_name}_col"})
        adata.var.index.name = None

    if adata.raw is not None:
        raw = adata.raw.to_adata()
        if raw.var.index.name is not None:
            idx_name = raw.var.index.name
            if idx_name in raw.var.columns:
                raw.var = raw.var.rename(columns={idx_name: f"{idx_name}_col"})
            raw.var.index.name = None
        adata.raw = raw


def run_scanpy_shh_from_raw(adata):
    """
    Run scanpy score_genes on normalized raw counts.
    """

    if adata.raw is None:
        raise ValueError("adata.raw is None. Raw counts required.")

    # 1) Temporary AnnData from raw
    adata_tmp = adata.raw.to_adata()

    # 2) Normalize + log1p
    sc.pp.normalize_total(adata_tmp, target_sum=1e4)
    sc.pp.log1p(adata_tmp)

    # 3) Check SHH genes
    present = [g for g in SHH_GENES if g in adata_tmp.var_names]
    print(f"Requested SHH genes: {SHH_GENES}")
    print(f"Present in normalized raw: {present}")

    if len(present) == 0:
        print("[WARN] No SHH genes present. Filling score with zeros.")
        adata.obs[SCORE_NAME] = 0.0
        return

    # 4) score_genes
    sc.tl.score_genes(
        adata_tmp,
        gene_list=present,
        ctrl_as_ref=False,
        ctrl_size=50,
        n_bins=50,
        score_name=SCORE_NAME,
        random_state=0,
        use_raw=False,
        copy=False,
    )

    # 5) Copy scores back
    adata.obs[SCORE_NAME] = adata_tmp.obs[SCORE_NAME].values

    print(adata.obs[SCORE_NAME].describe())


def save_outputs(adata, system_tag):
    out_dir = OUT_ROOT / system_tag
    qc_dir = out_dir / "qc"
    qc_dir.mkdir(parents=True, exist_ok=True)

    # ---- FIX before writing ----
    fix_var_index_column_conflict(adata)

    h5_out = out_dir / f"{system_tag}_adata_with_{SCORE_NAME}.h5ad"
    csv_out = out_dir / f"{system_tag}_{SCORE_NAME}_percell.csv"

    adata.write_h5ad(h5_out)
    adata.obs[[SCORE_NAME]].to_csv(csv_out)

    print(f"[SAVE] {h5_out}")
    print(f"[SAVE] {csv_out}")

    # ---- per-node summary ----
    grp = adata.obs.groupby("celltype_new")[SCORE_NAME]

    summary = pd.DataFrame({
        "celltype_new": grp.size().index,
        "n_cells": grp.size().values,
        "mean": grp.mean().values,
        "median": grp.median().values,
        "q90": grp.quantile(0.9).values,
        "frac>0": grp.apply(lambda x: (x > 0).mean()).values,
        "variance": grp.var(ddof=1).values,
        "std": grp.std(ddof=1).values,
    })

    summary_csv = qc_dir / f"{system_tag}_{SCORE_NAME}_summary.csv"
    summary.to_csv(summary_csv, index=False)
    print(f"[QC] wrote summary: {summary_csv}")

    # ---- violin plot ----
    fig, ax = plt.subplots(figsize=(10, 4.8))
    data = [grp.get_group(ct).values for ct in summary["celltype_new"]]
    ax.violinplot(data, showextrema=False)
    ax.set_xticks(range(1, len(data) + 1))
    ax.set_xticklabels(summary["celltype_new"], rotation=30, ha="right")
    ax.set_ylabel(SCORE_NAME)
    ax.set_title(f"{system_tag} – {SCORE_NAME}")

    png = qc_dir / f"{system_tag}_{SCORE_NAME}_violins.png"
    fig.savefig(png, dpi=300)
    plt.close(fig)
    print(f"[PLOT] wrote {png}")


def integrate_shh_scoregenes_with_edges_and_plot(system_tag: str, out_root: Path):
    """
    Use SHH_scoregenes per-node summary to build an edge table using Scanpy scores.
    """
    import numpy as np

    score_csv = out_root / system_tag / "qc" / f"{system_tag}_{SCORE_NAME}_summary.csv"
    if not score_csv.exists():
        print(f"[EDGE] Missing score summary: {score_csv}")
        return

    scores = pd.read_csv(score_csv)

    if "variance" not in scores.columns:
        scores["variance"] = np.nan
    if "std" not in scores.columns:
        scores["std"] = np.sqrt(scores["variance"])

    scores = scores.rename(columns={
        "celltype_new": "node_name",
        "mean": "sh_score",
        "n_cells": "n"
    })

    edge_file = Path(
        "/project/imoskowitz/xyang2/SHH/Qiu_TimeLapse/results/tree/edges_filtered.txt"
    )
    if not edge_file.exists():
        print(f"[EDGE] Missing edge file: {edge_file}")
        return

    edges = pd.read_csv(edge_file, sep="\t")
    edges = edges.loc[edges["system"] == system_tag].copy()

    edges = edges.merge(
        scores.rename(columns={
            "node_name": "x_name",
            "sh_score": "sh_x",
            "variance": "variance_x",
            "std": "std_x",
            "n": "n_x"
        }),
        on="x_name", how="left"
    )

    edges = edges.merge(
        scores.rename(columns={
            "node_name": "y_name",
            "sh_score": "sh_y",
            "variance": "variance_y",
            "std": "std_y",
            "n": "n_y"
        }),
        on="y_name", how="left"
    )

    edges["abs_delta"] = (edges["sh_x"] - edges["sh_y"]).abs()
    edges["delta"] = edges["sh_y"] - edges["sh_x"]

    pooled_std = np.sqrt((edges["variance_x"] + edges["variance_y"]) / 2.0)
    edges["cohens_d"] = edges["delta"] / pooled_std.replace(0, np.nan)

    outdir = out_root / system_tag
    outdir.mkdir(parents=True, exist_ok=True)

    out_csv = outdir / f"{system_tag}_edge_filtered_with_shh_scoregenes.csv"
    out_txt = outdir / f"{system_tag}_edge_filtered_with_shh_scoregenes.txt"

    edges.to_csv(out_csv, index=False)
    edges.to_csv(out_txt, sep="\t", index=False, na_rep="")

    print(f"[EDGE] wrote {out_csv}")



def main():
    print("== Scanpy SHH scoring (normalized raw) ==")

    for system_tag in SYSTEMS:
        print("\n" + "=" * 60)
        print(f"[SYSTEM] {system_tag}")

        h5 = H5_ROOT / f"{system_tag}_adata_scale.h5ad"
        if not h5.exists():
            print(f"[SKIP] missing {h5}")
            continue

        adata = sc.read_h5ad(h5)

        if "system" not in adata.obs:
            adata.obs["system"] = system_tag

        # ---- Fix categorical var_names safely ----
        adata.var.index = adata.var.index.astype(str)
        adata.var_names_make_unique()

        if adata.raw is not None:
            raw = adata.raw.to_adata()
            raw.var.index = raw.var.index.astype(str)
            raw.var_names_make_unique()
            adata.raw = raw

        # ---- Run scoring ----
        run_scanpy_shh_from_raw(adata)

        # ---- Save ----
        save_outputs(adata, system_tag)

        integrate_shh_scoregenes_with_edges_and_plot(
            system_tag,
            OUT_ROOT
        )

    print("\nDone.")


if __name__ == "__main__":
    main()
