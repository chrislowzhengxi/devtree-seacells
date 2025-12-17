#!/bin/bash -l
#SBATCH --job-name=shh_scoregenes_hist
#SBATCH --output=outs/shh_scoregenes_hist_%j.out
#SBATCH --error=outs/shh_scoregenes_hist_%j.err
#SBATCH --account=pi-imoskowitz
##SBATCH --account=pi-xyang2

##SBATCH --partition=bigmem
##SBATCH --partition=caslake
##SBATCH --partition=amd-hm
#SBATCH --partition=amd

##SBATCH --mem=64G
##SBATCH --mem=128G
#SBATCH --mem=200G
##SBATCH --mem=256G
##SBATCH --mem=512G
##SBATCH --mem=1T
##SBATCH --mem=2T

##SBATCH --cpus-per-task=1
#SBATCH --cpus-per-task=16
##SBATCH --cpus-per-task=32

##SBATCH --time=01:00:00
#SBATCH --time=12:00:00
##SBATCH --time=1-12:00:00

#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=chrislowzhengxi@uchicago.edu

set -eo pipefail

# === Use env directly; no modules, no conda activate ===
ENVROOT=/project/xyang2/software-packages/env/velocity_2025Feb_xy
PY=$ENVROOT/bin/python

# Ensure we use Python from this env
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
ls -t outs/shh_scoregenes_hist_*.out 2>/dev/null | tail -n +6 | xargs -r rm -f
ls -t outs/shh_scoregenes_hist_*.err 2>/dev/null | tail -n +6 | xargs -r rm -f

# Sanity print (goes to log)
$PY - <<'PY'
import sys
print("PYEXE:", sys.executable)
try:
    import pandas as pd; import matplotlib
    print("pandas:", pd.__version__)
    print("matplotlib:", matplotlib.__version__)
except Exception as e:
    print("Import check failed:", e)
PY

# Paths for script and data
CODE_DIR=/project/imoskowitz/xyang2/chrislowzhengxi/code/SEACellsBenchmark
BASE_DIR=/project/imoskowitz/xyang2/chrislowzhengxi/results/score_genes
OUTDIR=$BASE_DIR/histograms_scanpy

mkdir -p "$OUTDIR"

cd "$CODE_DIR"

# Run histogram script
$PY plot_shh_scoregenes_histograms.py \
    --base-dir "$BASE_DIR" \
    --outdir "$OUTDIR"
