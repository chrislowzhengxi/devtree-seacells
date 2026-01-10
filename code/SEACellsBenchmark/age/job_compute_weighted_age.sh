#!/bin/bash -l
#SBATCH --job-name=weighted_age_gastrulation
#SBATCH --account=pi-imoskowitz
#SBATCH --partition=amd-hm
#SBATCH --mem=700G
#SBATCH --cpus-per-task=1
#SBATCH --time=12:00:00

#SBATCH --output=/project/imoskowitz/xyang2/chrislowzhengxi/code/SEACellsBenchmark/outs/weighted_age_gastrulation_%j.out
#SBATCH --error=/project/imoskowitz/xyang2/chrislowzhengxi/code/SEACellsBenchmark/outs/weighted_age_gastrulation_%j.err

set -euo pipefail

# ---- Environment ----
ENVROOT=/project/xyang2/software-packages/env/velocity_2025Feb_xy
export PATH="$ENVROOT/bin:$PATH"
export R_HOME="$ENVROOT/lib/R"

# Avoid old RevoUtils / site-library pollution
export R_LIBS_USER=$HOME/R/clean_libs
mkdir -p "$R_LIBS_USER"

# ---- Working directory ----
cd /project/imoskowitz/xyang2/chrislowzhengxi/code/SEACellsBenchmark/age

# ---- Run ----
Rscript --vanilla compute_weighted_age_all_systems.R
