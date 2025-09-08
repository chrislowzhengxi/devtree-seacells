#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# --- paths ---
csv = Path("/project/imoskowitz/xyang2/chrislowzhengxi/results/ucell/Lateral_plate_mesoderm/qc/Lateral_plate_mesoderm_log1p_cpm_cardiac_gold_standard_summary.csv")
outdir = csv.parent  # save plots in the same qc/ folder

# infer system name (e.g., Lateral_plate_mesoderm) from path
system = outdir.parent.name

# --- load ---
df = pd.read_csv(csv)
df = df[df["n_cells"].fillna(0) > 0].copy()  # keep labeled rows only

# lineage order (tree): CPM -> SHF -> FHF -> Ventricles -> Atria
display_order = [
    "Cardiopharyngeal mesoderm",
    "Second heart field",
    "First heart field",
    "Ventricular cardiomyocytes",
    "Atrial cardiomyocytes",
]
df["celltype_new"] = pd.Categorical(df["celltype_new"], categories=display_order, ordered=True)
df = df.sort_values("celltype_new")

# =============================================================================
# Fig 1: barplot of MEDIAN with distinct markers for MEAN (●) and 90th pct (▲)
# =============================================================================
fig1, ax1 = plt.subplots(figsize=(8.5, 4.5))
x = range(len(df))

# make the bar color light so the markers stand out
bars = ax1.bar(x, df["median"], width=0.6, label="Median", edgecolor="black", linewidth=0.7, color="#D0D7DE")

# overlay markers (different shapes)
ax1.plot(x, df["mean"], marker="o", linestyle="none", label="Mean", markersize=6, markeredgecolor="black", markerfacecolor="black")
ax1.plot(x, df["q90"],  marker="^", linestyle="none", label="90th pct", markersize=7, markeredgecolor="black", markerfacecolor="#F5A623")

ax1.set_xticks(list(x))
ax1.set_xticklabels(df["celltype_new"], rotation=30, ha="right")
ax1.set_ylim(0, 1)  # UCell is 0..1
ax1.set_ylabel("SHH UCell score")
ax1.set_title(f"{system} – SHH UCell score by cardiac label")
ax1.legend(frameon=False, ncol=3)
fig1.tight_layout()
fig1.savefig(outdir / f"{system}_SHH_UCell_cardiac_summary.pdf", dpi=300)
fig1.savefig(outdir / f"{system}_SHH_UCell_cardiac_summary.png", dpi=300)

# =============================================================================
# Fig 2: lineage-style line of MEDIANS to show where the biggest change occurs
# =============================================================================
fig2, ax2 = plt.subplots(figsize=(8.0, 3.8))
ax2.plot(list(x), df["median"], marker="o", linewidth=2)

# annotate with n
for xn, m, n in zip(x, df["median"], df["n_cells"]):
    ax2.text(xn, m, f"n={int(n)}", fontsize=8, va="bottom", ha="center")

ax2.set_xticks(list(x))
ax2.set_xticklabels(df["celltype_new"], rotation=30, ha="right")
ax2.set_ylim(0, 1)
ax2.set_ylabel("Median SHH UCell")
ax2.set_title(f"{system} – SHH along cardiac lineage (median)")
fig2.tight_layout()
fig2.savefig(outdir / f"{system}_SHH_UCell_lineage_median.pdf", dpi=300)
fig2.savefig(outdir / f"{system}_SHH_UCell_lineage_median.png", dpi=300)

# =============================================================================
# Extendable master table (appendable across systems)
# =============================================================================
# Keep the canonical columns and add 'system'
keep_cols = ["celltype_new", "n_cells", "median", "mean", "q90", "frac>0"]
tab = df[keep_cols].copy()
tab.insert(0, "system", system)

root = Path("/project/imoskowitz/xyang2/chrislowzhengxi/results/ucell")
master_path = root / "SHH_UCell_cardio_summary_MASTER.csv"
# append (create if missing), then drop any accidental duplicates
if master_path.exists():
    master = pd.read_csv(master_path)
    master = pd.concat([master, tab], ignore_index=True)
    master.drop_duplicates(subset=["system", "celltype_new"], keep="last", inplace=True)
else:
    master = tab

master.to_csv(master_path, index=False)

print("Saved:")
print(outdir / f"{system}_SHH_UCell_cardiac_summary.pdf")
print(outdir / f"{system}_SHH_UCell_cardiac_summary.png")
print(outdir / f"{system}_SHH_UCell_lineage_median.pdf")
print(outdir / f"{system}_SHH_UCell_lineage_median.png")
print(master_path)
