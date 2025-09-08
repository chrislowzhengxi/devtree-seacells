# #!/usr/bin/env python3
# import os
# import scanpy as sc
# import pandas as pd
# import matplotlib.pyplot as plt

# import warnings
# warnings.filterwarnings("ignore", category=UserWarning)

# # 1) where your per‐organ h5ads live
# RAW_DIR = "/project/imoskowitz/xyang2/SHH/Qiu_TimeLapse/results/raw_added"
# OUT_CSV = "/project/imoskowitz/xyang2/chrislowzhengxi/results/cell_counts_per_organ.csv"
# OUT_PDF = "/project/imoskowitz/xyang2/chrislowzhengxi/results/cell_counts_per_organ.pdf"

# # 2) find all the .h5ad files in there
# files = sorted(f for f in os.listdir(RAW_DIR) if f.endswith(".h5ad"))

# counts = []
# for fn in files:
#     organ = fn.replace("_adata_scale.h5ad","")  # e.g. "Eye", "Blood", ...
#     path  = os.path.join(RAW_DIR, fn)
#     adata = sc.read_h5ad(path, backed='r')
#     counts.append({
#       "organ":    organ,
#       "n_cells":  adata.n_obs
#     })

# # 3) write out CSV
# df = pd.DataFrame(counts)
# df.to_csv(OUT_CSV, index=False)
# print(f"Wrote counts to {OUT_CSV}")

# # 4) bar‐plot and save PDF
# plt.figure(figsize=(8,4))
# plt.bar(df["organ"], df["n_cells"])
# plt.xticks(rotation=45, ha="right")
# plt.ylabel("Number of cells")
# plt.title("Cells per organ")
# plt.tight_layout()
# plt.savefig(OUT_PDF)
# print(f"Wrote barplot to {OUT_PDF}")




# import os
# import scanpy as sc
# import pandas as pd
# RAW_DIR = "/project/imoskowitz/xyang2/SHH/Qiu_TimeLapse/results/raw_added"
# OUT_CSV = "/project/imoskowitz/xyang2/chrislowzhengxi/results/cell_counts_per_organ.csv"

# counts = []

# for fn in sorted(f for f in os.listdir(RAW_DIR) if f.endswith(".h5ad")):
#     organ = fn.replace("_adata_scale.h5ad", "")
#     path = os.path.join(RAW_DIR, fn)

#     try:
#         adata = sc.read_h5ad(path, backed='r')
#         counts.append({
#             "organ": organ,
#             "n_cells": adata.shape[0]
#         })
#         del adata  # just in case
#     except Exception as e:
#         print(f"Failed to process {organ}: {e}")

# df = pd.DataFrame(counts)
# df.to_csv(OUT_CSV, index=False)
# print(f"Wrote cell counts to {OUT_CSV}")




#!/usr/bin/env python3
import os
import scanpy as sc
import pandas as pd
import matplotlib.pyplot as plt

# Directories
RAW_DIR = "/project/imoskowitz/xyang2/SHH/Qiu_TimeLapse/results/raw_added"
OUT_CSV = "/project/imoskowitz/xyang2/chrislowzhengxi/results/cell_counts_per_organ.csv"
OUT_PDF = "/project/imoskowitz/xyang2/chrislowzhengxi/results/cell_counts_per_organ.pdf"

# Scan directory and count cells
counts = []
for fname in sorted(f for f in os.listdir(RAW_DIR) if f.endswith(".h5ad")):
    organ = fname.replace("_adata_scale.h5ad", "")
    path = os.path.join(RAW_DIR, fname)
    try:
        adata = sc.read_h5ad(path, backed="r")  # read in backed mode to save memory
        counts.append({
            "organ": organ,
            "n_cells": adata.n_obs
        })
        adata.file.close()  # release file handle
    except Exception as e:
        print(f"⚠️ Error reading {fname}: {e}")

# Save CSV
df = pd.DataFrame(counts)
df.to_csv(OUT_CSV, index=False)
print(f"✅ Saved counts to {OUT_CSV}")
