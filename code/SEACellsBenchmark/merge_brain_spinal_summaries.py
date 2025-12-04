# #!/usr/bin/env python3
# import pandas as pd
# from pathlib import Path

# # Input paths
# base = Path("/project/imoskowitz/xyang2/chrislowzhengxi/results/ucell")
# p1 = base / "Other_Brain_spinal_cord/qc/Other_Brain_spinal_cord_log1p_cpm_ALL_labels_summary.csv"
# p2 = base / "Neurons/qc/Neurons_log1p_cpm_ALL_labels_summary.csv"

# # Output path
# outdir = base / "Brain_spinal_cord/qc"
# outdir.mkdir(parents=True, exist_ok=True)
# outpath = outdir / "Brain_spinal_cord_log1p_cpm_ALL_labels_summary.csv"

# # Load and merge
# df1 = pd.read_csv(p1)
# df2 = pd.read_csv(p2)
# merged = pd.concat([df1, df2], ignore_index=True)

# # Optional: drop duplicates just in case
# merged = merged.drop_duplicates(subset=["celltype_new"], keep="last")

# # Save
# merged.to_csv(outpath, index=False)
# print(f"[MERGED] wrote combined summary: {outpath}")
#!/usr/bin/env python3


#!/usr/bin/env python3
import pandas as pd
import scanpy as sc
from pathlib import Path

base = Path("/project/imoskowitz/xyang2/chrislowzhengxi/results/score_genes")

systems_to_merge = ["Other_Brain_spinal_cord", "Neurons"]
merged_system = "Brain_spinal_cord"

outdir = base / merged_system
qc_outdir = outdir / "qc"
qc_outdir.mkdir(parents=True, exist_ok=True)

# -----------------------------
# 1. Merge summary CSVs
# -----------------------------
dfs = []
for sys in systems_to_merge:
    p = base / sys / "qc" / f"{sys}_SHH_scoregenes_summary.csv"
    dfs.append(pd.read_csv(p))

merged_summary = pd.concat(dfs, ignore_index=True)
merged_summary = merged_summary.drop_duplicates(subset=["celltype_new"], keep="last")

summary_out = qc_outdir / f"{merged_system}_SHH_scoregenes_summary.csv"
merged_summary.to_csv(summary_out, index=False)
print("Wrote summary:", summary_out)

# -----------------------------
# 2. Merge per-cell CSVs
# -----------------------------
percell_dfs = []
for sys in systems_to_merge:
    p = base / sys / f"{sys}_SHH_scoregenes_percell.csv"
    percell_dfs.append(pd.read_csv(p))

merged_percell = pd.concat(percell_dfs, ignore_index=False)
percell_out = outdir / f"{merged_system}_SHH_scoregenes_percell.csv"
merged_percell.to_csv(percell_out)
print("Wrote per-cell:", percell_out)

# -----------------------------
# 3. Merge h5ad files
# -----------------------------
adatas = []
for sys in systems_to_merge:
    p = base / sys / f"{sys}_adata_with_SHH_scoregenes.h5ad"
    ad = sc.read_h5ad(str(p))
    ad.obs["system"] = merged_system
    adatas.append(ad)

adata_merged = adatas[0].concatenate(*adatas[1:], join="outer")
h5_out = outdir / f"{merged_system}_adata_with_SHH_scoregenes.h5ad"
adata_merged.write_h5ad(str(h5_out))
print("Wrote merged h5ad:", h5_out)
