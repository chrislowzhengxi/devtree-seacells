#!/bin/bash -l
#SBATCH --job-name=node_stage_tbl
#SBATCH --account=pi-imoskowitz
#SBATCH --partition=bigmem
##SBATCH --partition=amd-hm


#SBATCH --mem=500GB
##SBATCH --mem=1000GB

#SBATCH --cpus-per-task=1
#SBATCH --time=12:00:00
#SBATCH --output=/project/imoskowitz/xyang2/chrislowzhengxi/code/SEACellsBenchmark/outs/node_stage_tbl_%j.out
#SBATCH --error=/project/imoskowitz/xyang2/chrislowzhengxi/code/SEACellsBenchmark/outs/node_stage_tbl_%j.err

set -euo pipefail

ENVROOT=/project/xyang2/software-packages/env/velocity_2025Feb_xy
export PATH="$ENVROOT/bin:$PATH"
export R_HOME="$ENVROOT/lib/R"

cd /project/imoskowitz/xyang2/chrislowzhengxi/code/SEACellsBenchmark/age

Rscript compute_5.R
