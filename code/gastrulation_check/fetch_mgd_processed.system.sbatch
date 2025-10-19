#!/bin/bash
# System-R-based variant (Variant B) - call site R directly and add libcurl diagnostics
#SBATCH --job-name=MGD_fetch_systemR
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

echo "Starting System-R MGD fetch at $(date)"
echo "Job: $SLURM_JOB_ID on $(hostname)"
echo "R_LIBS_USER=$R_LIBS_USER"
echo "R_CACHE=$R_CACHE"

# Path to system Rscript binary (avoid module load to bypass Tcl/modulefile errors)
RSCRIPT_BIN="/software/R-4.4.1-el8-x86_64/bin/Rscript"
if [ ! -x "$RSCRIPT_BIN" ]; then
  echo "Rscript not found or not executable at $RSCRIPT_BIN"; exit 4
fi

# Attempt to set LD_LIBRARY_PATH so R picks up compatible libcurl. Adjust the path below if your site has a different libcurl.
# Common candidate: /software/libcurl/<version>/lib or the lib directory that shipped with the system R
export LD_LIBRARY_PATH="/software/lib64/:/software/R-4.4.1-el8-x86_64/lib64:${LD_LIBRARY_PATH:-}"

echo "LD_LIBRARY_PATH=${LD_LIBRARY_PATH}"
echo "ldd diagnostics for curl.so (R's curl package) - if this fails, adjust LD_LIBRARY_PATH as needed"
if [ -f "/software/R-4.4.1-el8-x86_64/lib64/R/library/curl/libs/curl.so" ]; then
  ldd /software/R-4.4.1-el8-x86_64/lib64/R/library/curl/libs/curl.so || true
else
  echo "curl.so not found at expected site path; skip ldd"
fi

# Export R env vars for the script
export R_LIBS_USER

# Run R script via system Rscript
"$RSCRIPT_BIN" --vanilla fetch_mgd_processed.R

echo "Finished System-R MGD fetch at $(date)"
