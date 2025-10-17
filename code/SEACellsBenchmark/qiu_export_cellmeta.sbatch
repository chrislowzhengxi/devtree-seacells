#!/bin/bash -l
#SBATCH --job-name=qiu_export_meta
#SBATCH --output=outs/qiu_export_meta_%j.out
#SBATCH --error=outs/qiu_export_meta_%j.err
#SBATCH --account=pi-imoskowitz
#SBATCH --partition=caslake
#SBATCH --mem=4G
#SBATCH --cpus-per-task=2
#SBATCH --time=00:10:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=chrislowzhengxi@uchicago.edu
set -eo pipefail

# === Use env directly; no modules/activation needed ===
ENVROOT=/project/xyang2/software-packages/env/velocity_2025Feb_xy
PY=$ENVROOT/bin/python
export PATH="$ENVROOT/bin:$PATH"
export R_HOME="$ENVROOT/lib/R"
unset PYTHONPATH
export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export NUMEXPR_MAX_THREADS=$SLURM_CPUS_PER_TASK

mkdir -p outs
ls -t outs/qiu_export_meta_*.out 2>/dev/null | tail -n +6 | xargs -r rm -f
ls -t outs/qiu_export_meta_*.err 2>/dev/null | tail -n +6 | xargs -r rm -f

# Sanity print
$PY - <<'PY'
import sys
print("PYEXE:", sys.executable)
PY

# Run the tiny export (backed='r' to avoid loading matrices)
$PY - <<'PY'
from pathlib import Path
import anndata as ad
import pandas as pd

H5_ROOT = Path("/project/imoskowitz/xyang2/SHH/Qiu_TimeLapse/results/raw_added")
OUT_DIR = Path("/project/imoskowitz/xyang2/chrislowzhengxi/results/gastrulation")
OUT_DIR.mkdir(parents=True, exist_ok=True)

SYSTEMS = ["Lateral_plate_mesoderm"]  # extend list as needed

for system in SYSTEMS:
    h5 = H5_ROOT / f"{system}_adata_scale.h5ad"
    print("[LOAD obs] ", h5)
    adata = ad.read_h5ad(str(h5), backed="r")   # low memory read
    obs = adata.obs.copy().rename_axis("cell").reset_index()
    keep = [c for c in ["celltype_updated","celltype_new","system","stage"] if c in obs.columns]
    obs = obs[["cell", *keep]]
    out_csv = OUT_DIR / f"qiu_{system}_cellmeta.csv"
    obs.to_csv(out_csv, index=False)
    print("wrote:", out_csv)
PY
