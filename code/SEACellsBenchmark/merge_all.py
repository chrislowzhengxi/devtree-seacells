#!/usr/bin/env python3
import pandas as pd
from pathlib import Path

base = Path("/project/imoskowitz/xyang2/chrislowzhengxi/results/ucell")

# List of systems you mentioned
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

# Read and combine
dfs = []
for sys in systems:
    f = base / sys / f"{sys}_edge_filtered_with_shh.csv"
    if f.exists():
        df = pd.read_csv(f)
        df["system_name"] = sys  # optional, for clarity
        dfs.append(df)
        print(f"[OK] loaded {sys}")
    else:
        print(f"[WARN] missing: {f}")

merged = pd.concat(dfs, ignore_index=True)
print(f"Total merged rows: {len(merged):,}")

# Save combined table
outpath = base / "merged_all_systems_edge_filtered_with_shh.csv"
merged.to_csv(outpath, index=False)
print(f"[MERGED] wrote {outpath}")


## PRE-GASTRULATION ADDITION (commented out)
# #!/usr/bin/env python3
# import pandas as pd
# import numpy as np
# from pathlib import Path

# # Paths
# merged_file = Path("/project/imoskowitz/xyang2/chrislowzhengxi/results/ucell/merged_all_systems_edge_filtered_with_shh.csv")
# edges_file = Path("/project/imoskowitz/xyang2/SHH/Qiu_TimeLapse/other/edges.txt")
# out_file = Path("/project/imoskowitz/xyang2/chrislowzhengxi/results/ucell/full_scored_edges_with_pregastrulation.csv")

# # Load existing merged data
# merged = pd.read_csv(merged_file)
# print(f"[LOAD] merged systems: {merged.shape}")

# # Load pre-gastrulation edges
# preg = pd.read_csv(edges_file, sep="\t")
# preg = preg[preg["system"] == "Pre_gastrulation"].copy()
# print(f"[LOAD] pre-gastrulation edges: {preg.shape}")

# # Assign random scores around 0
# rng = np.random.default_rng(42)
# preg["sh_x"] = rng.normal(0, 0.02, size=len(preg))
# preg["sh_y"] = rng.normal(0, 0.02, size=len(preg))

# # Compute deltas
# preg["delta"] = preg["sh_y"] - preg["sh_x"]
# preg["abs_delta"] = preg["delta"].abs()

# # Optional: same columns as main table
# preg["cohens_d"] = np.nan
# preg["pct_change"] = np.nan
# preg["abs_pct_change"] = np.nan

# # Merge into master
# full = pd.concat([merged, preg], ignore_index=True)
# print(f"[MERGED] combined total rows: {full.shape[0]}")

# # Save
# full.to_csv(out_file, index=False)
# print(f"[SAVE] {out_file}")
