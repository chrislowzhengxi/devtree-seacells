#!/bin/bash -l
#SBATCH --job-name=node_stage_tbl
#SBATCH --output=outs/node_stage_tbl_%j.out
#SBATCH --error=outs/node_stage_tbl_%j.err

#SBATCH --account=pi-imoskowitz
#SBATCH --partition=amd-hm
#SBATCH --mem=700G
#SBATCH --cpus-per-task=1
#SBATCH --time=12:00:00

#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=chrislowzhengxi@uchicago.edu

set -euo pipefail

# ===== Environment =====
ENVROOT=/project/xyang2/software-packages/env/velocity_2025Feb_xy

export PATH="$ENVROOT/bin:$PATH"
export R_HOME="$ENVROOT/lib/R"
unset PYTHONPATH
export PYTHONNOUSERSITE=1

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_MAX_THREADS=1

mkdir -p outs

# ===== Sanity checks =====
R --version
which R

# ===== Run =====
cd /project/imoskowitz/xyang2/chrislowzhengxi/code/SEACellsBenchmark/age/

Rscript chris_build_node_staging_table.R
