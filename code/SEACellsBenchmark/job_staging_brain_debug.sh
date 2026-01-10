#!/bin/bash -l
#SBATCH --job-name=add_staging_brain_dbg
#SBATCH --output=outs/add_staging_brain_dbg_%j.out
#SBATCH --error=outs/add_staging_brain_dbg_%j.err
#SBATCH --account=pi-imoskowitz
#SBATCH --partition=amd-hm
#SBATCH --mem=700G
#SBATCH --cpus-per-task=1
#SBATCH --time=02:00:00
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=chrislowzhengxi@uchicago.edu

set -euo pipefail

ENVROOT=/project/xyang2/software-packages/env/velocity_2025Feb_xy
PY=$ENVROOT/bin/python
export PATH="$ENVROOT/bin:$PATH"
unset PYTHONPATH
export PYTHONNOUSERSITE=1

mkdir -p outs

cd /project/imoskowitz/xyang2/chrislowzhengxi/code/SEACellsBenchmark/

$PY add_staging_brain_debug.py \
  --h5-root /project/imoskowitz/xyang2/SHH/Qiu_TimeLapse/results/raw_added \
  --csv-meta /project/imoskowitz/xyang2/chrislowzhengxi/data/df_cell_celltyp_new_merged.csv \
  --out-root /project/imoskowitz/xyang2/chrislowzhengxi/results/raw_added_with_staging \
  --systems Neurons Other_Brain_spinal_cord \
  --overwrite
