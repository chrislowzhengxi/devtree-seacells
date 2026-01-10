#!/bin/bash -l
#SBATCH --job-name=build_ucell_sum
#SBATCH --account=pi-imoskowitz
#SBATCH --partition=amd-hm
#SBATCH --mem=700GB
#SBATCH --cpus-per-task=1
#SBATCH --time=12:00:00
#SBATCH --output=/project/imoskowitz/xyang2/chrislowzhengxi/code/SEACellsBenchmark/outs/build_ucell_sum_%j.out
#SBATCH --error=/project/imoskowitz/xyang2/chrislowzhengxi/code/SEACellsBenchmark/outs/build_ucell_sum_%j.err

set -euo pipefail

ENVROOT=/project/xyang2/software-packages/env/velocity_2025Feb_xy
export PATH="$ENVROOT/bin:$PATH"
unset PYTHONPATH
export PYTHONNOUSERSITE=1

cd /project/imoskowitz/xyang2/chrislowzhengxi/code/SEACellsBenchmark/age

python build_celltype_scanpy_summary_by_system.py
