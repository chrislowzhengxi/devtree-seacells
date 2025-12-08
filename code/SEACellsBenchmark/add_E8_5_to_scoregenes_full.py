#!/usr/bin/env python3
import pandas as pd
from pathlib import Path

# Source with E8.5b edges (UCell-scored)
ucell_devtree = Path(
    "/project/imoskowitz/xyang2/chrislowzhengxi/results/ucell/tree/devtree_edges_for_graph.tsv"
)

# Target Scanpy combined file
scanpy_full = Path(
    "/project/imoskowitz/xyang2/chrislowzhengxi/results/score_genes/full_scored_edges_with_pregastrulation_scoregenes.csv"
)

print("[LOAD] UCell devtree:", ucell_devtree)
uc = pd.read_csv(ucell_devtree, sep="\t")

if "system" not in uc.columns:
    raise SystemExit("No 'system' column in devtree_edges_for_graph.tsv")

e85 = uc[uc["system"] == "Gastrulation_E8.5b"].copy()
print(f"[E8.5] rows found in devtree file: {e85.shape[0]}")

if e85.empty:
    raise SystemExit("No Gastrulation_E8.5b rows found, nothing to do.")

print("[LOAD] Scanpy full:", scanpy_full)
sg_orig = pd.read_csv(scanpy_full)

# Remove any existing E8.5b rows from Scanpy file (just in case)
before = sg_orig.shape[0]
sg = sg_orig[sg_orig["system"] != "Gastrulation_E8.5b"].copy()
print(f"[FILTER] removed {before - sg.shape[0]} existing E8.5b rows from Scanpy file")

# Keep only columns that actually exist in the Scanpy file,
# so we do not introduce weird extra columns from devtree.
common_cols = [c for c in e85.columns if c in sg.columns]
e85_for_merge = e85[common_cols].copy()

combined = pd.concat([sg, e85_for_merge], ignore_index=True)
print(f"[MERGE] combined rows: {combined.shape[0]}")

# Backup original Scanpy file
backup = scanpy_full.with_suffix(".backup_before_E8_5_from_devtree.csv")
sg_orig.to_csv(backup, index=False)
print("[BACKUP] wrote:", backup)

# Save updated file
combined.to_csv(scanpy_full, index=False)
print("[SAVE] updated:", scanpy_full)
