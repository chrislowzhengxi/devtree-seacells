#!/usr/bin/env python3
import pandas as pd
from pathlib import Path

# Base directory for Scanpy score-gen codes
base = Path("/project/imoskowitz/xyang2/chrislowzhengxi/results/score_genes_new")

# The full list of systems
systems = [
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

dfs = []

for sys in systems:
    f = base / sys / f"{sys}_edge_filtered_with_shh_scoregenes.csv"
    if f.exists():
        df = pd.read_csv(f)
        df["system_name"] = sys
        dfs.append(df)
        print(f"[OK] loaded {sys}")
    else:
        print(f"[WARN] missing: {f}")

# Combine all systems
merged = pd.concat(dfs, ignore_index=True)
print(f"Total merged rows: {len(merged):,}")

# Output file
outpath = base / "merged_all_systems_edge_filtered_with_shh_scoregenes.csv"
merged.to_csv(outpath, index=False)
print(f"[MERGED] wrote {outpath}")
