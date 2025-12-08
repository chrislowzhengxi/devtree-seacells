#!/usr/bin/env python3
import pandas as pd
from pathlib import Path

# Paths
base = Path("/project/imoskowitz/xyang2/chrislowzhengxi/results/score_genes")

full_path = base / "full_scored_edges_with_pregastrulation_scoregenes.csv"
gastr_path = base / "Gastrulation" / "Gastrulation_edge_filtered_with_shh_scoregenes.csv"

print("[LOAD] full:", full_path)
full = pd.read_csv(full_path)

print("[LOAD] gastrulation:", gastr_path)
gastr = pd.read_csv(gastr_path)

# Just in case there is already some Gastrulation in the file, drop it first
before = full.shape[0]
full = full[full["system"] != "Gastrulation"].copy()
after = full.shape[0]
print(f"[FILTER] removed {before - after} existing Gastrulation rows (if any).")

# Append Gastrulation rows
combined = pd.concat([full, gastr], ignore_index=True)
print(f"[MERGE] combined rows: {combined.shape[0]}")

# Optional: keep a backup
backup_path = base / "full_scored_edges_with_pregastrulation_scoregenes_backup_before_gastr.csv"
full.to_csv(backup_path, index=False)
print("[BACKUP] wrote:", backup_path)

# Overwrite the main file
combined.to_csv(full_path, index=False)
print("[SAVE] updated:", full_path)
