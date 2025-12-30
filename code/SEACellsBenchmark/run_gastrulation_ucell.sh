#!/bin/bash -l
#SBATCH --job-name=ucell_gastr
#SBATCH --output=outs/ucell_gastr_%j.out
#SBATCH --error=outs/ucell_gastr_%j.err
#SBATCH --account=pi-imoskowitz
##SBATCH --account=pi-xyang2

##SBATCH --partition=bigmem
##SBATCH --partition=caslake
#SBATCH --partition=amd-hm
##SBATCH --partition=amd

## Tune these as needed:
#SBATCH --mem=700G
##SBATCH --mem=256G
##SBATCH --mem=512G
#SBATCH --cpus-per-task=32

##SBATCH --time=04:00:00
#SBATCH --time=1-00:00:00

#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=chrislowzhengxi@uchicago.edu

set -eo pipefail

# === Use env directly; no modules, no conda activate ===
ENVROOT=/project/xyang2/software-packages/env/velocity_2025Feb_xy

# Ensure we use R/Python from this env
export PATH="$ENVROOT/bin:$PATH"
export R_HOME="$ENVROOT/lib/R"
unset PYTHONPATH
export PYTHONNOUSERSITE=1

# Threads (match SLURM)
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export NUMEXPR_MAX_THREADS=${SLURM_CPUS_PER_TASK:-1}

# Temp space (faster scratch for R tmp files)
mkdir -p /scratch/midway3/$USER/tmp_${SLURM_JOB_ID}
export TMPDIR=/scratch/midway3/$USER/tmp_${SLURM_JOB_ID}

# Logs directory + simple rotation (keep last 5)
mkdir -p outs
ls -t outs/ucell_gastr_*.out 2>/dev/null | tail -n +6 | xargs -r rm -f
ls -t outs/ucell_gastr_*.err 2>/dev/null | tail -n +6 | xargs -r rm -f

# ---- Sanity prints (to log) ----
echo "ENVROOT: $ENVROOT"
echo "R:       $(which R)"; R --version | head -n 1 || true
echo "Rscript: $(which Rscript)"
echo "CPUS:    ${SLURM_CPUS_PER_TASK:-1}   MEM: ${SLURM_MEM_PER_NODE:-NA}"

# Optional Python check (since env includes python/rpy2)
python - <<'PY' || true
import sys
print("PYEXE:", sys.executable)
try:
    import numpy as np; print("NumPy:", np.__version__)
except Exception as e:
    print("NumPy import failed:", e)
try:
    import rpy2; print("rpy2  :", rpy2.__version__)
except Exception as e:
    print("rpy2 import failed:", e)
PY

# ---- Paths ----
R_SCRIPT=/project/imoskowitz/xyang2/chrislowzhengxi/code/SEACellsBenchmark/run_gastrulation_ucell.R
RESULTS_DIR=/project/imoskowitz/xyang2/chrislowzhengxi/results/ucell/Gastrulation
DATA_OUT=/project/imoskowitz/xyang2/chrislowzhengxi/data/gastrulation

echo "Will write results under: $RESULTS_DIR"
echo "Will write SCE RDS under: $DATA_OUT"

# Ensure parent dirs exist
mkdir -p "$RESULTS_DIR" "$DATA_OUT"

# ---- Run ----
cd /project/imoskowitz/xyang2/chrislowzhengxi || exit 1
Rscript --vanilla "$R_SCRIPT"

echo "Done."
