#!/usr/bin/env python3
import pandas as pd
from pathlib import Path

# Input paths
base = Path("/project/imoskowitz/xyang2/chrislowzhengxi/results/ucell")
p1 = base / "Other_Brain_spinal_cord/qc/Other_Brain_spinal_cord_log1p_cpm_ALL_labels_summary.csv"
p2 = base / "Neurons/qc/Neurons_log1p_cpm_ALL_labels_summary.csv"

# Output path
outdir = base / "Brain_spinal_cord/qc"
outdir.mkdir(parents=True, exist_ok=True)
outpath = outdir / "Brain_spinal_cord_log1p_cpm_ALL_labels_summary.csv"

# Load and merge
df1 = pd.read_csv(p1)
df2 = pd.read_csv(p2)
merged = pd.concat([df1, df2], ignore_index=True)

# Optional: drop duplicates just in case
merged = merged.drop_duplicates(subset=["celltype_new"], keep="last")

# Save
merged.to_csv(outpath, index=False)
print(f"[MERGED] wrote combined summary: {outpath}")
