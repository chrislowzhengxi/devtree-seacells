#!/bin/bash -l
#SBATCH --job-name=ucell_shh_direct
#SBATCH --output=outs/ucell_shh_direct_%j.out
#SBATCH --error=outs/ucell_shh_direct_%j.err
#SBATCH --account=pi-xyang2

##SBATCH --partition=bigmem
##SBATCH --partition=caslake
#SBATCH --partition=amd-hm


#SBATCH --mem=64G
##SBATCH --mem=700G

#SBATCH --cpus-per-task=8
##SBATCH --cpus-per-task=16

#SBATCH --time=01:00:00
##SBATCH --time=1-00:00:00  

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

# Threads
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export NUMEXPR_MAX_THREADS=$SLURM_CPUS_PER_TASK

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
$PY ucell_scoring_noscanpy.py
