import pandas as pd
import matplotlib.pyplot as plt

# -------------------------
# Matplotlib settings
# -------------------------
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42

# -------------------------
# Paths
# -------------------------
AGE_TSV = (
    "/project/imoskowitz/xyang2/chrislowzhengxi/code/SEACellsBenchmark/age/age_mean_and_weighted_by_celltype_by_system_final.tsv"
)

SCANPY_TSV = "celltype_scanpy_summary_by_system.tsv"
UCELL_TSV  = "celltype_ucell_summary_by_system.tsv"

# -------------------------
# Systems and colors
# -------------------------
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

BIN_WIDTH = 4.0  # hours

# -------------------------
# Load data
# -------------------------
age = pd.read_csv(AGE_TSV, sep="\t")
scanpy = pd.read_csv(SCANPY_TSV, sep="\t")
ucell  = pd.read_csv(UCELL_TSV, sep="\t")

df_scanpy = age.merge(
    scanpy,
    on=["system", "celltype_new"],
    how="inner",
)

df_ucell = age.merge(
    ucell,
    on=["system", "celltype_new"],
    how="inner",
)

# Ensure numeric
for df in [df_scanpy, df_ucell]:
    df["age_weighted"] = pd.to_numeric(df["age_weighted"], errors="coerce")
    df["total_cells"] = pd.to_numeric(df["total_cells"], errors="coerce")

df_scanpy["mean_scanpy"]      = pd.to_numeric(df_scanpy["mean_scanpy"], errors="coerce")
df_scanpy["pct_scanpy_pos"]   = pd.to_numeric(df_scanpy["pct_scanpy_pos"], errors="coerce")
df_ucell["mean_ucell"]        = pd.to_numeric(df_ucell["mean_ucell"], errors="coerce")
df_ucell["pct_ucell_pos"]     = pd.to_numeric(df_ucell["pct_ucell_pos"], errors="coerce")

# -------------------------
# Helper: smoothed plot
# -------------------------
def plot_smoothed(
    df,
    value_col,
    ylabel,
    title,
    out_pdf,
):
    fig, ax = plt.subplots(figsize=(8, 5))

    for system in SYSTEMS_IN_ORDER:
        sub = df[df["system"] == system].copy()
        if sub.empty:
            continue

        # Raw points (faint)
        ax.scatter(
            sub["age_weighted"],
            sub[value_col],
            color=PALETTE.get(system, "gray"),
            s=10,
            alpha=0.08,
            edgecolors="none",
        )

        # Bin age
        sub["age_bin"] = (sub["age_weighted"] / BIN_WIDTH).round() * BIN_WIDTH

        # Weighted mean per bin
        binned = (
            sub.groupby("age_bin", as_index=False)
               .apply(
                   lambda x: pd.Series({
                       value_col: (
                           (x[value_col] * x["total_cells"]).sum()
                           / x["total_cells"].sum()
                       )
                   })
               )
               .reset_index(drop=True)
               .sort_values("age_bin")
        )

        ax.plot(
            binned["age_bin"],
            binned[value_col],
            label=system,
            color=PALETTE.get(system, "gray"),
            linewidth=2.8,
            alpha=0.85,
        )

    ax.set_xlabel("Weighted developmental age (hours)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    ax.legend(
        title="system",
        bbox_to_anchor=(1.05, 1),
        loc="upper left",
        frameon=False,
    )

    plt.tight_layout()
    plt.savefig(out_pdf)
    plt.close()

    print(f"Wrote {out_pdf}")

# ============================================================
# Generate all four plots
# ============================================================

plot_smoothed(
    df_scanpy,
    value_col="mean_scanpy",
    ylabel="Mean SHH Scanpy score",
    title="System-level SHH Scanpy activity across development",
    out_pdf="line_age_vs_scanpy_mean_by_system_smoothed.pdf",
)

plot_smoothed(
    df_scanpy,
    value_col="pct_scanpy_pos",
    ylabel="Fraction SHH Scanpy score > 0",
    title="System-level SHH Scanpy positivity across development",
    out_pdf="line_age_vs_scanpy_pct_by_system_smoothed.pdf",
)

plot_smoothed(
    df_ucell,
    value_col="mean_ucell",
    ylabel="Mean SHH UCell score",
    title="System-level SHH UCell activity across development",
    out_pdf="line_age_vs_ucell_mean_by_system_smoothed.pdf",
)

plot_smoothed(
    df_ucell,
    value_col="pct_ucell_pos",
    ylabel="Fraction of cells with positive SHH UCell score",
    title="System-level SHH UCell positivity across development",
    out_pdf="line_age_vs_ucell_pct_by_system_smoothed.pdf",
)
