# #!/usr/bin/env python3
# import os
# import argparse
# import numpy as np
# import pandas as pd
# import anndata as ad
# import matplotlib
# matplotlib.use("Agg")  # headless
# import matplotlib.pyplot as plt
# import seaborn as sns


# # ------------------------ UMAP helpers ------------------------
# def ensure_umap(adata, use_rep="X_pca", n_neighbors=15, random_state=0):
#     """
#     Compute neighbors/UMAP if missing. Uses X_pca by default.
#     """
#     import scanpy as sc
#     if "X_umap" not in adata.obsm_keys():
#         if "neighbors" not in adata.uns:
#             k = min(n_neighbors, max(2, adata.n_obs - 1))
#             sc.pp.neighbors(adata, use_rep=use_rep, n_neighbors=k)
#         sc.tl.umap(adata, random_state=random_state)
#     return adata

# # --------------------- Score merge helpers --------------------
# def add_ucell_scores_to_obs(
#     adata,
#     score_csv=None,
#     id_col_csv="cell_id",
#     score_col_csv="ucell_score",
#     id_col_obs="cell_id",
# ):
#     """
#     Populate adata.obs['ucell_score'] from either:
#       1) an existing column in adata.obs (e.g., 'SHH_UCell_score' or 'ucell_score'), OR
#       2) a per-cell CSV with columns [cell_id, score] (or first column as IDs).

#     Set id_col_obs='index' to match by obs_names.
#     """
#     # Case A: no CSV -> try to pick up from obs
#     if score_csv is None:
#         # prefer explicit score_col_csv, else common fallback 'SHH_UCell_score'
#         for cand in [score_col_csv, "SHH_UCell_score"]:
#             if cand in adata.obs.columns:
#                 adata.obs["ucell_score"] = pd.to_numeric(adata.obs[cand], errors="coerce")
#                 return adata
#         raise ValueError("No score_csv provided and no score column (e.g. 'SHH_UCell_score') found in adata.obs.")

#     # Case B: CSV provided -> read and map
#     df = pd.read_csv(score_csv, low_memory=False)

#     # If ID col not present, auto-use the first column (handles 'Unnamed: 0')
#     if id_col_csv not in df.columns:
#         auto_id = df.columns[0]
#         print(f"[info] Using '{auto_id}' as ID column from CSV (was expecting '{id_col_csv}').")
#         id_col_csv = auto_id

#     # If requested score col missing, try 'SHH_UCell_score'
#     if score_col_csv not in df.columns and "SHH_UCell_score" in df.columns:
#         print(f"[info] Using 'SHH_UCell_score' as score column (requested '{score_col_csv}' not found).")
#         score_col_csv = "SHH_UCell_score"

#     if score_col_csv not in df.columns:
#         raise ValueError(
#             f"Score column '{score_col_csv}' not in CSV. Found: {list(df.columns)[:10]}..."
#         )

#     # Build mapping from CSV
#     m = df.set_index(id_col_csv)[score_col_csv]
#     m = pd.to_numeric(m, errors="coerce")

#     # Map to obs
#     if id_col_obs == "index":
#         idx = adata.obs_names.astype(str)
#         adata.obs["ucell_score"] = m.reindex(idx).values
#     else:
#         if id_col_obs not in adata.obs.columns:
#             raise ValueError(
#                 f"AnnData .obs is missing id column '{id_col_obs}'. "
#                 f"Have: {list(adata.obs.columns)[:20]}"
#             )
#         adata.obs["ucell_score"] = adata.obs[id_col_obs].map(m)

#     missing = int(adata.obs["ucell_score"].isna().sum())
#     if missing:
#         print(f"[warn] {missing} cells had no matching score. They will be ignored for plotting.")
#     return adata

# # --------------------------- Plots -----------------------------
# def save_histograms(adata, cluster_col, score_col, outdir, system_label="LPM"):
#     figdir = os.path.join(outdir, "figures")
#     os.makedirs(figdir, exist_ok=True)

#     df = adata.obs[[cluster_col, score_col]].dropna().copy()
#     df[cluster_col] = df[cluster_col].astype(str)

#     # Faceted overview (also save PNG for PPT)
#     try:
#         g = sns.displot(
#             data=df, x=score_col, col=cluster_col, col_wrap=5,
#             binwidth=0.025, height=2.0, facet_kws={"sharex": True, "sharey": False}
#         )
#         g.set_axis_labels("SHH UCell score", "Cell count")
#         g.fig.subplots_adjust(top=0.9)
#         g.fig.suptitle(f"{system_label}: cell-level SHH UCell score distributions by cluster", y=1.02)
#         out_pdf = os.path.join(figdir, f"{system_label}_SHH_histograms_facet.pdf")
#         out_png = os.path.join(figdir, f"{system_label}_SHH_histograms_facet.png")
#         g.savefig(out_pdf, dpi=300)
#         g.savefig(out_png, dpi=300)
#         plt.close(g.fig)
#     except Exception as e:
#         print(f"[warn] Facet histograms skipped: {e}")

#     # Per-cluster PDFs/PNGs
#     for c in sorted(df[cluster_col].unique(), key=lambda x: (x.lower(), x)):
#         sub = df[df[cluster_col] == c]
#         if sub.empty:
#             continue
#         plt.figure(figsize=(3.2, 2.4))
#         sns.histplot(sub[score_col], bins=np.arange(0, 1.0001, 0.025))
#         sns.despine()
#         plt.title(f"{c} (n={len(sub)})", fontsize=9)
#         plt.xlabel("SHH UCell score"); plt.ylabel("Cell count")
#         plt.tight_layout()
#         safe = c.replace(" ", "_").replace("/", "_")
#         plt.savefig(os.path.join(figdir, f"{system_label}_hist_{safe}.pdf"), dpi=300)
#         plt.savefig(os.path.join(figdir, f"{system_label}_hist_{safe}.png"), dpi=300)
#         plt.close()
#     return figdir

# def save_umap_plots(adata, cluster_col, score_col, focus_clusters, outdir, system_label="LPM"):
#     figdir = os.path.join(outdir, "figures")
#     os.makedirs(figdir, exist_ok=True)
#     if "X_umap" not in adata.obsm_keys():
#         print("[info] No UMAP found. Skipping UMAP plots.")
#         return figdir, []

#     coords = adata.obsm["X_umap"]
#     df = adata.obs[[cluster_col, score_col]].copy()
#     df = df.join(pd.DataFrame(coords, index=adata.obs_names, columns=["umap_1", "umap_2"]))
#     df = df.dropna(subset=[score_col])

#     # All cells colored by score
#     plt.figure(figsize=(4.6, 4.0))
#     sca = plt.scatter(df["umap_1"], df["umap_2"], c=df[score_col], s=4, linewidths=0, alpha=0.9)
#     plt.colorbar(sca, label="SHH UCell score")
#     plt.xticks([]); plt.yticks([]); sns.despine(left=True, bottom=True)
#     plt.title(f"{system_label} UMAP colored by SHH UCell score")
#     plt.tight_layout()
#     all_pdf = os.path.join(figdir, f"{system_label}_UMAP_SHH_score_all.pdf")
#     all_png = os.path.join(figdir, f"{system_label}_UMAP_SHH_score_all.png")
#     plt.savefig(all_pdf, dpi=300); plt.savefig(all_png, dpi=300)
#     plt.close()

#     # Zooms for focus clusters
#     zoom_pngs = []
#     x_min, x_max = df["umap_1"].min(), df["umap_1"].max()
#     y_min, y_max = df["umap_2"].min(), df["umap_2"].max()
#     for c in focus_clusters:
#         sub = df[df[cluster_col] == c]
#         if sub.empty:
#             print(f"[info] Focus cluster not found or empty: {c}")
#             continue
#         plt.figure(figsize=(4.6, 4.0))
#         plt.scatter(df["umap_1"], df["umap_2"], c="lightgray", s=2, linewidths=0, alpha=0.35)
#         sca = plt.scatter(sub["umap_1"], sub["umap_2"], c=sub[score_col], s=8, linewidths=0, alpha=0.95)
#         plt.colorbar(sca, label="SHH UCell score")
#         plt.xlim(x_min, x_max); plt.ylim(y_min, y_max)
#         plt.xticks([]); plt.yticks([]); sns.despine(left=True, bottom=True)
#         plt.title(f"UMAP zoom: {c}")
#         plt.tight_layout()
#         safe = c.replace(" ", "_").replace("/", "_")
#         out_pdf = os.path.join(figdir, f"{system_label}_UMAP_zoom_{safe}.pdf")
#         out_png = os.path.join(figdir, f"{system_label}_UMAP_zoom_{safe}.png")
#         plt.savefig(out_pdf, dpi=300); plt.savefig(out_png, dpi=300)
#         plt.close()
#         zoom_pngs.append(out_png)
#     return figdir, [all_png] + zoom_pngs


# # ---------------------------- CLI ------------------------------
# def main():
#     ap = argparse.ArgumentParser(description="LPM cell-level histograms & UMAP from AnnData + per-cell UCell scores")
#     ap.add_argument("--h5ad", required=True, help="Path to system .h5ad (e.g., Lateral_Plate_Mesoderm_adata_scale.h5ad)")
#     ap.add_argument("--score-csv", default=None, help="Per-cell score CSV (optional if score already in .obs)")
#     ap.add_argument("--obs-id-col", default="cell_id", help="ID column in adata.obs (use 'index' to match obs_names)")
#     ap.add_argument("--csv-id-col", default="cell_id", help="ID column in score CSV (first col auto-used if missing)")
#     ap.add_argument("--score-col", default="ucell_score", help="Score column in obs/CSV (auto-fallsback to SHH_UCell_score)")
#     ap.add_argument("--cluster-col", default="celltype_new", help="Cluster label in obs")
#     ap.add_argument("--focus", nargs="*", default=["First heart field", "Second heart field", "Atrial cardiomyocytes"])
#     ap.add_argument("--outdir", required=True, help="Output directory")
#     ap.add_argument("--compute-umap", action="store_true", help="Compute UMAP if missing (using X_pca)")
#     ap.add_argument("--system-label", default="LPM", help="Short system tag for filenames/titles")
#     args = ap.parse_args()

#     os.makedirs(args.outdir, exist_ok=True)
#     figdir = os.path.join(args.outdir, "figures"); os.makedirs(figdir, exist_ok=True)

#     # Load AnnData
#     adata = ad.read_h5ad(args.h5ad)
#     try:
#         adata.var_names_make_unique()
#     except Exception:
#         pass

#     # Merge or pick up scores
#     adata = add_ucell_scores_to_obs(
#         adata,
#         score_csv=args.score_csv,
#         id_col_csv=args.csv_id_col,
#         score_col_csv=args.score_col,
#         id_col_obs=args.obs_id_col,
#     )

#     # UMAP if asked / needed
#     if args.compute_umap:
#         adata = ensure_umap(adata)

#     # Style
#     sns.set_style("ticks")
#     plt.rcParams["pdf.fonttype"] = 42
#     plt.rcParams["ps.fonttype"]  = 42
#     plt.rcParams["figure.dpi"]   = 300

#     # Plots
#     save_histograms(adata, args.cluster_col, "ucell_score", args.outdir, system_label=args.system_label)
#     figdir, _ = save_umap_plots(adata, args.cluster_col, "ucell_score", args.focus, args.outdir, system_label=args.system_label)

#     print("All done.")
#     print("Figures dir:", figdir)

# if __name__ == "__main__":
#     main()


#!/usr/bin/env python3
import os
import argparse
import numpy as np
import pandas as pd
import anndata as ad
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns


# ------------------------ UMAP helpers ------------------------
def ensure_umap(adata, use_rep="X_pca", n_neighbors=15, random_state=0):
    import scanpy as sc
    if "X_umap" not in adata.obsm_keys():
        if "neighbors" not in adata.uns:
            k = min(n_neighbors, max(2, adata.n_obs - 1))
            sc.pp.neighbors(adata, use_rep=use_rep, n_neighbors=k)
        sc.tl.umap(adata, random_state=random_state)
    return adata


# --------------------- Score merge helpers --------------------
def add_ucell_scores_to_obs(
    adata,
    score_csv=None,
    id_col_csv="cell_id",
    score_col_csv="ucell_score",
    id_col_obs="cell_id",
):
    if score_csv is None:
        for cand in [score_col_csv, "SHH_UCell_score"]:
            if cand in adata.obs.columns:
                adata.obs["ucell_score"] = pd.to_numeric(adata.obs[cand], errors="coerce")
                return adata
        raise ValueError("No score_csv provided and no score column found in adata.obs.")

    df = pd.read_csv(score_csv, low_memory=False)
    if id_col_csv not in df.columns:
        id_col_csv = df.columns[0]
        print(f"[info] Using '{id_col_csv}' as ID column from CSV.")
    if score_col_csv not in df.columns and "SHH_UCell_score" in df.columns:
        score_col_csv = "SHH_UCell_score"
        print(f"[info] Using 'SHH_UCell_score' as score column.")

    m = df.set_index(id_col_csv)[score_col_csv]
    m = pd.to_numeric(m, errors="coerce")

    if id_col_obs == "index":
        idx = adata.obs_names.astype(str)
        adata.obs["ucell_score"] = m.reindex(idx).values
    else:
        adata.obs["ucell_score"] = adata.obs[id_col_obs].map(m)

    missing = int(adata.obs["ucell_score"].isna().sum())
    if missing:
        print(f"[warn] {missing} cells had no matching score. They will be ignored for plotting.")
    return adata


# --------------------------- Plots -----------------------------
# def save_histograms(adata, cluster_col, score_col, outdir, system_label="LPM", focus_clusters=None):
#     figdir = os.path.join(outdir, "figures")
#     os.makedirs(figdir, exist_ok=True)
#     df = adata.obs[[cluster_col, score_col]].dropna().copy()
#     df[cluster_col] = df[cluster_col].astype(str)

#     # Plot only focus clusters for clarity
#     if focus_clusters:
#         df_focus = df[df[cluster_col].isin(focus_clusters)].copy()
#     else:
#         df_focus = df.copy()

#     # Faceted overview of focus clusters
#     try:
#         g = sns.displot(
#             data=df_focus, x=score_col, col=cluster_col, col_wrap=3,
#             binwidth=0.02, height=2.8, facet_kws={"sharex": True, "sharey": False}
#         )
#         g.set_axis_labels("SHH UCell score", "Cell count")
#         g.fig.subplots_adjust(top=0.9)
#         g.fig.suptitle(f"{system_label}: SHH UCell score distributions (focus clusters)", y=1.02)
#         out_pdf = os.path.join(figdir, f"{system_label}_SHH_histograms_focus.pdf")
#         g.savefig(out_pdf, dpi=300)
#         plt.close(g.fig)
#     except Exception as e:
#         print(f"[warn] Facet histograms skipped: {e}")

#     # Individual histograms with median and q90 lines
#     for c in sorted(df_focus[cluster_col].unique(), key=lambda x: (x.lower(), x)):
#         sub = df_focus[df_focus[cluster_col] == c]
#         if sub.empty:
#             continue
#         plt.figure(figsize=(3.5, 2.6))
#         sns.histplot(sub[score_col], bins=np.arange(0, 1.001, 0.02))
#         sns.despine()
#         plt.xlim(0, 1)
#         median = float(sub[score_col].median())
#         q90 = float(sub[score_col].quantile(0.90))
#         plt.axvline(median, ls="--", lw=1, color="black")
#         plt.axvline(q90, ls=":", lw=1, color="black")
#         plt.title(f"{c} (n={len(sub)})", fontsize=9)
#         plt.xlabel("SHH UCell score")
#         plt.ylabel("Cell count")
#         plt.tight_layout()
#         safe = c.replace(" ", "_").replace("/", "_")
#         plt.savefig(os.path.join(figdir, f"{system_label}_hist_{safe}.pdf"), dpi=300)
#         plt.close()
#     return figdir

def save_histograms(adata, cluster_col, score_col, outdir, system_label="LPM"):
    figdir = os.path.join(outdir, "figures")
    os.makedirs(figdir, exist_ok=True)

    df = adata.obs[[cluster_col, score_col]].dropna().copy()
    df[cluster_col] = df[cluster_col].astype(str)

    # --- NEW: log10(score + 1) transform for histograms ---
    df["score_log10p1"] = np.log10(df[score_col].astype(float) + 1.0)
    log_max = np.log10(2.0)  # scores in [0,1] -> log10 in [0, ~0.301]

    # Faceted overview
    try:
        g = sns.displot(
            data=df, x="score_log10p1", col=cluster_col, col_wrap=5,
            bins=100, binrange=(0, log_max), height=2.0,
            facet_kws={"sharex": True, "sharey": False}
        )
        g.set_axis_labels("log10(SHH UCell + 1)", "Cell count")
        g.fig.subplots_adjust(top=0.9)
        g.fig.suptitle(f"{system_label}: log10 histograms by cluster", y=1.02)
        g.savefig(os.path.join(figdir, f"{system_label}_SHH_histograms_facet_log10p1.pdf"), dpi=300)
        plt.close(g.fig)
    except Exception as e:
        print(f"[warn] Facet histograms skipped: {e}")

    # Per-cluster PDFs/PNGs
    for c in sorted(df[cluster_col].unique(), key=lambda x: (x.lower(), x)):
        sub = df[df[cluster_col] == c]
        if sub.empty:
            continue
        plt.figure(figsize=(3.2, 2.4))
        sns.histplot(sub["score_log10p1"], bins=100, binrange=(0, log_max))
        sns.despine()
        plt.title(f"{c} (n={len(sub)})", fontsize=9)
        plt.xlabel("log10(SHH UCell + 1)"); plt.ylabel("Cell count")
        plt.tight_layout()
        safe = c.replace(" ", "_").replace("/", "_")
        plt.savefig(os.path.join(figdir, f"{system_label}_hist_{safe}_log10p1.pdf"), dpi=300)
        plt.savefig(os.path.join(figdir, f"{system_label}_hist_{safe}_log10p1.png"), dpi=300)
        plt.close()
    return figdir


def save_histograms_focus(adata, cluster_col, score_col, focus_clusters, outdir, system_label="LPM"):
    figdir = os.path.join(outdir, "figures"); os.makedirs(figdir, exist_ok=True)
    df = adata.obs[[cluster_col, score_col]].dropna().copy()
    df["score_log10p1"] = np.log10(df[score_col].astype(float) + 1.0)
    log_max = np.log10(2.0)
    df = df[df[cluster_col].isin(focus_clusters)]
    g = sns.displot(
        data=df, x="score_log10p1", col=cluster_col, col_wrap=3,
        bins=100, binrange=(0, log_max), height=2.2,
        facet_kws={"sharex": True, "sharey": False}
    )
    g.set_axis_labels("log10(SHH UCell + 1)", "Cell count")
    g.fig.subplots_adjust(top=0.85)
    g.fig.suptitle(f"{system_label}: SHH UCell log10 histograms (focus clusters)", y=1.02)
    g.savefig(os.path.join(figdir, f"{system_label}_SHH_hist_focus_log10p1.pdf"), dpi=300)
    plt.close(g.fig)
    return figdir


def save_umap_plots(adata, cluster_col, score_col, focus_clusters, outdir, system_label="LPM"):
    figdir = os.path.join(outdir, "figures")
    os.makedirs(figdir, exist_ok=True)
    if "X_umap" not in adata.obsm_keys():
        print("[info] No UMAP found. Skipping UMAP plots.")
        return figdir, []

    coords = adata.obsm["X_umap"]
    df = adata.obs[[cluster_col, score_col]].copy()
    df = df.join(pd.DataFrame(coords, index=adata.obs_names, columns=["umap_1", "umap_2"]))
    df = df.dropna(subset=[score_col])

    # All cells colored by score (fixed 0–1 scale)
    plt.figure(figsize=(4.8, 4.2))
    sca = plt.scatter(df["umap_1"], df["umap_2"],
                      c=df[score_col], vmin=0, vmax=1, s=4, linewidths=0, alpha=0.9)
    plt.colorbar(sca, label="SHH UCell score")
    plt.xticks([]); plt.yticks([]); sns.despine(left=True, bottom=True)
    plt.title(f"{system_label} UMAP colored by SHH UCell score")
    plt.tight_layout()
    all_pdf = os.path.join(figdir, f"{system_label}_UMAP_SHH_score_all.pdf")
    plt.savefig(all_pdf, dpi=300)
    plt.close()

    # Zooms with high/low (top/bottom 10%) highlight
    x_min, x_max = df["umap_1"].min(), df["umap_1"].max()
    y_min, y_max = df["umap_2"].min(), df["umap_2"].max()
    for c in focus_clusters:
        sub = df[df[cluster_col] == c].copy()
        if sub.empty:
            print(f"[info] Focus cluster not found or empty: {c}")
            continue
        q10 = sub[score_col].quantile(0.10)
        q90 = sub[score_col].quantile(0.90)
        sub["band"] = np.where(
            sub[score_col] >= q90, "high (top 10%)",
            np.where(sub[score_col] <= q10, "low (bottom 10%)", "mid")
        )

        plt.figure(figsize=(4.6, 4.0))
        plt.scatter(df["umap_1"], df["umap_2"], c="lightgray", s=2, alpha=0.25, linewidths=0)
        for band, color in [("low (bottom 10%)", "blue"), ("high (top 10%)", "red")]:
            sb = sub[sub["band"] == band]
            plt.scatter(sb["umap_1"], sb["umap_2"], s=10, linewidths=0, color=color, label=f"{band} (n={len(sb)})")
        plt.legend(frameon=False, fontsize=8, loc="upper right")
        plt.xlim(x_min, x_max); plt.ylim(y_min, y_max)
        plt.xticks([]); plt.yticks([]); sns.despine(left=True, bottom=True)
        plt.title(f"UMAP: {c} – high/low SHH cells")
        plt.tight_layout()
        safe = c.replace(" ", "_").replace("/", "_")
        plt.savefig(os.path.join(figdir, f"{system_label}_UMAP_zoom_{safe}_high_low.pdf"), dpi=300)
        plt.close()
    return figdir


# ---------------------------- CLI ------------------------------
def main():
    ap = argparse.ArgumentParser(description="LPM SHH UCell: histograms + UMAP (focus clusters)")
    ap.add_argument("--h5ad", required=True)
    ap.add_argument("--score-csv", default=None)
    ap.add_argument("--obs-id-col", default="cell_id")
    ap.add_argument("--csv-id-col", default="cell_id")
    ap.add_argument("--score-col", default="ucell_score")
    ap.add_argument("--cluster-col", default="celltype_new")
    ap.add_argument("--focus", nargs="*", default=["First heart field", "Second heart field", "Atrial cardiomyocytes"])
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--compute-umap", action="store_true")
    ap.add_argument("--system-label", default="LPM")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    adata = ad.read_h5ad(args.h5ad)
    adata.var_names_make_unique()

    adata = add_ucell_scores_to_obs(
        adata,
        score_csv=args.score_csv,
        id_col_csv=args.csv_id_col,
        score_col_csv=args.score_col,
        id_col_obs=args.obs_id_col,
    )

    if args.compute_umap:
        adata = ensure_umap(adata)

    sns.set_style("ticks")
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"]  = 42
    plt.rcParams["figure.dpi"]   = 300

    # save_histograms(adata, args.cluster_col, "ucell_score", args.outdir,
    #                 system_label=args.system_label)
    save_histograms_focus(adata, args.cluster_col, "ucell_score", args.focus,
                      args.outdir, system_label=args.system_label)
    save_umap_plots(adata, args.cluster_col, "ucell_score", args.focus,
                    args.outdir, system_label=args.system_label)
    print("✓ Done. All figures in:", os.path.join(args.outdir, "figures"))


if __name__ == "__main__":
    main()

