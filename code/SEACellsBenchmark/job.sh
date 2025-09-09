#!/bin/bash -l
#SBATCH --job-name=seacell_test_small
#SBATCH --output=outs/job_output_test_small.out
#SBATCH --error=outs/job_error_test_small.err
#SBATCH --partition=${PARTITION:-caslake}
#SBATCH --account=pi-xyang2
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --time=05:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=chrislowzhengxi@uchicago.edu

module load python
source activate /project/xyang2/anaconda/py311
export PYTHONPATH=$PYTHONPATH:/project/xyang2/TOOLS

cd /project/imoskowitz/xyang2/chrislowzhengxi/code/SEACellsBenchmark/
python SEACell_test_small.py


