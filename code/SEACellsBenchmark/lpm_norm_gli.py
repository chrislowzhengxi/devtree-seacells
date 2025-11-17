import pandas as pd

# Load file
path = "/project/imoskowitz/xyang2/chrislowzhengxi/results/ucell/Lateral_plate_mesoderm/qc/Lateral_plate_mesoderm_extended_obs.tsv"
df = pd.read_csv(path, sep="\t")

# Mean of normalized Gli1 per celltype_new
df_mean = (
    df.groupby("celltype_new")["Gli1_norm"]
      .mean()
      .reset_index()
      .rename(columns={"Gli1_norm": "mean_Gli1_norm"})
      .sort_values("mean_Gli1_norm", ascending=False)
)

# Save full table
out1 = "/project/imoskowitz/xyang2/chrislowzhengxi/results/ucell/Lateral_plate_mesoderm/qc/mean_Gli1_norm_per_celltype.csv"
df_mean.to_csv(out1, index=False)

# Compute and save gold standard rank
gs_name = "Atrial cardiomyocytes"

rank_df = df_mean.reset_index(drop=True)
rank_df["rank"] = rank_df.index + 1
rank_df["total"] = len(rank_df)

gs_row = rank_df[rank_df["celltype_new"] == gs_name]

out2 = "/project/imoskowitz/xyang2/chrislowzhengxi/results/ucell/Lateral_plate_mesoderm/qc/gold_standard_rank.csv"
gs_row.to_csv(out2, index=False)

print("Saved:")
print(out1)
print(out2)
