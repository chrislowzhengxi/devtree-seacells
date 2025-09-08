#!/bin/bash -l

#SBATCH --job-name=count_cells_per_organ
#SBATCH --output=outs/count_cells_output.out
#SBATCH --error=outs/count_cells_error.err

## ---- Resources ----
#SBATCH --partition=caslake
#SBATCH --account=pi-xyang2
#SBATCH --time=01:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=1

## ---- Notifications ----
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=chrislowzhengxi@uchicago.edu

## ---- Environment ----
module load python
source activate /project/xyang2/anaconda/py311
export PYTHONPATH=$PYTHONPATH:/project/xyang2/TOOLS
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK

## ---- Run your script ----
cd /project/imoskowitz/xyang2/chrislowzhengxi/code/SEACellsBenchmark/
python count_cells_per_organ.py
