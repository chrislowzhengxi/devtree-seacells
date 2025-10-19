#!/bin/bash
# Conda-based variant (Variant A)
# One-time conda setup (run interactively once):
# conda create -n r443 -c conda-forge -c bioconda r-base=4.4.2 r-essentials r-base-curl=8.2.1 libcurl=8.2.1 r-biocmanager r-experimenthub r-mousegastrulationdata r-singlecellexperiment r-spatialexperiment r-bioconductor-biocfilecache -y
# Or, more conservatively, create env then install in the env with conda/bioconda/cran as needed.
# After creating: source ~/.bashrc; conda activate r443

#SBATCH --job-name=MGD_fetch_conda
#SBATCH --account=pi-imoskowitz
#SBATCH --partition=bigmem
#SBATCH --mem=256G
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/mgd_fetch_%j.out
#SBATCH --error=logs/mgd_fetch_%j.err

set -euo pipefail
export PROJECT_DIR="/project/imoskowitz/xyang2/chrislowzhengxi"
export WORKDIR="$PROJECT_DIR/code/gastrulation_check"
export R_CACHE="$PROJECT_DIR/.ehcache"
export R_LIBS_USER="/home/chrislowzhengxi/R/x86_64-pc-linux-gnu-library/4.4"

mkdir -p "$WORKDIR" "$PROJECT_DIR/results" "$PROJECT_DIR/.ehcache" logs

# Prefer node-local scratch if available
if [ -n "${TMPDIR:-}" ]; then
  export TMPDIR="$TMPDIR"
elif [ -d "/scratch" ]; then
  export TMPDIR="/scratch/$USER/$SLURM_JOB_ID"
  mkdir -p "$TMPDIR"
fi

cd "$WORKDIR"

echo "Starting Conda-based MGD fetch at $(date)"
echo "Job: $SLURM_JOB_ID on $(hostname)"
echo "R_LIBS_USER=$R_LIBS_USER"
echo "R_CACHE=$R_CACHE"

# Load conda and activate environment
source ~/.bashrc || true
if command -v conda >/dev/null 2>&1; then
  conda activate r443 || { echo "Failed to activate conda env r443"; exit 2; }
else
  echo "conda not found in PATH. Ensure conda is installed and ~/.bashrc sources it.";
  exit 3
fi

# Run R script
Rscript --vanilla fetch_mgd_processed.R

echo "Finished Conda-based MGD fetch at $(date)"
#!/bin/bash
#SBATCH --job-name=MGD_fetch
#SBATCH --account=pi-imoskowitz
#SBATCH --partition=bigmem
#SBATCH --mem=256G
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/mgd_fetch_%j.out
#SBATCH --error=logs/mgd_fetch_%j.err

module purge
module load R/4.4.2+gcc-13.2.0   # <-- use this newer module
# (Skip ImageMagick; not needed just to fetch MGD)

export R_LIBS_USER="$HOME/R/x86_64-pc-linux-gnu-library/4.4"
cd /project/imoskowitz/xyang2/chrislowzhengxi/code/gastrulation_check
Rscript --vanilla fetch_mgd_processed.R
