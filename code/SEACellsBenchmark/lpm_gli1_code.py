import pandas as pd

path = "/project/imoskowitz/xyang2/chrislowzhengxi/results/ucell/Lateral_plate_mesoderm/qc/Lateral_plate_mesoderm_extended_obs.tsv"
df = pd.read_csv(path, sep="\t")

# Fixed "medium" threshold from the SHH histogram workflow
M = 0.35
print(f"Using M = {M:.2f} for Gli1_norm")

# Flag cells with Gli1_norm >= M
df["Gli1_above_M"] = df["Gli1_norm"] >= M

# Summarize per celltype_new
summary = (
    df.groupby("celltype_new")
      .agg(
          n_cells=("cell_id", "size"),
          n_Gli1_above_M=("Gli1_above_M", "sum"),
      )
      .reset_index()
)

summary["pct_Gli1_above_M"] = 100.0 * summary["n_Gli1_above_M"] / summary["n_cells"]

# Rank by percentage (high to low)
summary = summary.sort_values("pct_Gli1_above_M", ascending=False).reset_index(drop=True)
summary["rank"] = summary.index + 1
summary["total_celltypes"] = len(summary)

print("\n=== LPM: % of cells with Gli1_norm >= 0.35 by celltype_new ===")
print(summary)

# Gold standards
gs_list = ["Atrial cardiomyocytes", "Second heart field"]
gs_rows = summary[summary["celltype_new"].isin(gs_list)]

print("\n=== Gold standard ranks (Gli1_norm >= 0.35) ===")
print(gs_rows)

# Save table for the slide
out_csv = "/project/imoskowitz/xyang2/chrislowzhengxi/results/ucell/Lateral_plate_mesoderm/qc/LPM_pct_Gli1_ge0p35_per_celltype.csv"
summary.to_csv(out_csv, index=False)
print("\nSaved table to:", out_csv)
