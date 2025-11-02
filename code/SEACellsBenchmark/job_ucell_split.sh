#!/bin/bash -l
#SBATCH --job-name=ucell_shh_direct
#SBATCH --output=outs/ucell_shh_direct_%j.out
#SBATCH --error=outs/ucell_shh_direct_%j.err
#SBATCH --account=pi-imoskowitz

##SBATCH --partition=bigmem
##SBATCH --partition=caslake
#SBATCH --partition=amd-hm
##SBATCH --partition=amd

##SBATCH --mem=64G
##SBATCH --mem=128G
##SBATCH --mem=256G
##SBATCH --mem=512G
#SBATCH --mem=700G
##SBATCH --mem=1T
##SBATCH --mem=2T


##SBATCH --cpus-per-task=1
#SBATCH --cpus-per-task=8
##SBATCH --cpus-per-task=16
##SBATCH --cpus-per-task=32


##SBATCH --time=01:00:00
#SBATCH --time=1-12:00:00  

#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=chrislowzhengxi@uchicago.edu

set -eo pipefail

# === Use env directly; no modules, no conda activate ===
ENVROOT=/project/xyang2/software-packages/env/velocity_2025Feb_xy
PY=$ENVROOT/bin/python

# Ensure we use R/Python from this env
export PATH="$ENVROOT/bin:$PATH"
export R_HOME="$ENVROOT/lib/R"
unset PYTHONPATH
export PYTHONNOUSERSITE=1

# Scratch + chunking (script reads these)
export TMP_DIR=/scratch/midway3/chrislowzhengxi
# Optional: force a chunk size, else script auto-estimates
# export UCELL_CHUNK_SIZE=600000
# Optional: adjust headroom for 2^31 limit
# export UCELL_HEADROOM=0.80

# Optional: use normalized layer the script will build if missing
export UCELL_INPUT_LAYER=log1p_cpm

# Threads: use 1 for BLAS to avoid oversubscription; ncores comes from SLURM_CPUS_PER_TASK
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_MAX_THREADS=1
export SLURM_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK:-8}

mkdir -p outs
ls -t outs/ucell_shh_direct_*.out 2>/dev/null | tail -n +6 | xargs -r rm -f
ls -t outs/ucell_shh_direct_*.err 2>/dev/null | tail -n +6 | xargs -r rm -f

# Sanity print (goes to log)
$PY - <<'PY'
import sys, numpy
print("PYEXE:", sys.executable)
print("NumPy :", numpy.__version__)
try:
    import anndata as ad; print("AnnData:", ad.__version__)
except Exception as e:
    print("AnnData import failed:", e)
try:
    import rpy2; print("rpy2  :", rpy2.__version__)
except Exception as e:
    print("rpy2 import failed:", e)
PY

# Run (no activation needed)
cd /project/imoskowitz/xyang2/chrislowzhengxi/code/SEACellsBenchmark/
$PY ucell_scoring_noscanpy_split_copy.py
