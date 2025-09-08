#!/bin/bash -l

#SBATCH --job-name=seacell_test_small
#SBATCH --output=outs/job_output_test_small_%j.out
#SBATCH --error=outs/job_error_test_small_%j.err

## ---- Partition/Account Options ----
#SBATCH --account=pi-xyang2
##SBATCH --account=pi-imoskowitz


##SBATCH --partition=caslake
##SBATCH --partition=amd
#SBATCH --partition=bigmem
##SBATCH --partition=amd-hm

## ---- Memory Options ----
##SBATCH --mem=64G
##SBATCH --mem=128G
#SBATCH --mem=256G
##SBATCH --mem=512G
##SBATCH --mem=768G
##SBATCH --mem=1536G
##SBATCH --mem=2048G  # For amd-hm only

## ---- CPU Options ----
#SBATCH --cpus-per-task=4
##SBATCH --cpus-per-task=16
##SBATCH --cpus-per-task=32
##SBATCH --cpus-per-task=48
##SBATCH --cpus-per-task=64
##SBATCH --cpus-per-task=128  # For amd-hm only

## ---- Time ----
#SBATCH --time=01:00:00
##SBATCH --time=04:00:00
##SBATCH --time=06:00:00
##SBATCH --time=08:00:00
##SBATCH --time=1-00:00:00  
##SBATCH --time=1-12:00:00

## ---- Notifications ----
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=chrislowzhengxi@uchicago.edu

# ---- Environment Setup ----
module load python
source activate /project/xyang2/anaconda/py311
export NUMEXPR_MAX_THREADS=128 # For amd-hm only
export PYTHONPATH=$PYTHONPATH:/project/xyang2/TOOLS
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK

# ---- Cleanup: keep only the 5 most recent logs ----
ls -t outs/job_output_test_small_*.out | tail -n +6 | xargs -r rm -f
ls -t outs/job_error_test_small_*.err | tail -n +6 | xargs -r rm -f

# ---- Code Execution ----
cd /project/imoskowitz/xyang2/chrislowzhengxi/code/SEACellsBenchmark/
python plot_gut_copy.py
