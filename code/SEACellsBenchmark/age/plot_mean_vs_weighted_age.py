import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# -------------------------
# Matplotlib settings
# -------------------------
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

# -------------------------
# Constants
# -------------------------
AGE_TSV = (
    "/project/imoskowitz/xyang2/chrislowzhengxi/code/"
    "SEACellsBenchmark/age/"
    "age_mean_and_weighted_by_celltype_by_system_final.tsv"
)

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

# -------------------------
# Load data
# -------------------------
df = pd.read_csv(AGE_TSV, sep="\t")

# -------------------------
# Plot
# -------------------------
fig, ax = plt.subplots(figsize=(7.5, 5))

for system in SYSTEMS_IN_ORDER:
    sub = df[df["system"] == system]
    if sub.empty:
        continue

    ax.scatter(
        sub["age_mean"],
        sub["age_weighted"],
        label=system,
        color=PALETTE.get(system, "gray"),
        s=30,
        alpha=0.7,
        edgecolors="none",
    )

# -------------------------
# Reference diagonal y = x
# -------------------------
xmin = min(df["age_mean"].min(), df["age_weighted"].min())
xmax = max(df["age_mean"].max(), df["age_weighted"].max())
ax.plot(
    [xmin, xmax],
    [xmin, xmax],
    linestyle="--",
    linewidth=1,
    color="black",
    alpha=0.6,
)

# -------------------------
# Labels and legend
# -------------------------
ax.set_xlabel("Mean developmental age")
ax.set_ylabel("Weighted developmental age")
ax.set_title("Weighted vs mean developmental age by cell type")

ax.legend(
    title="system",
    bbox_to_anchor=(1.05, 1),
    loc="upper left",
    frameon=False,
)

plt.tight_layout()
plt.savefig("scatter_mean_vs_weighted_age.pdf")
plt.close()

print("Wrote scatter_mean_vs_weighted_age.pdf")
