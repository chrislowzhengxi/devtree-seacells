#!/usr/bin/env Rscript
## Robust fetch for MouseGastrulationData::EmbryoAtlasData(type = "processed")
## Writes ExperimentHub cache under project dir and saves an RDS for future fast loads.

options(repos = c(CRAN = "https://cloud.r-project.org"))
project_dir <- "/project/imoskowitz/xyang2/chrislowzhengxi"
cache_dir <- file.path(project_dir, ".ehcache")
out_rds <- file.path(project_dir, "results", "MouseGastrulation_processed.rds")
dir.create(cache_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(dirname(out_rds), recursive = TRUE, showWarnings = FALSE)

start_time <- Sys.time()
cat("R.version.string:", R.version.string, "\n")
cat(".libPaths():", paste(.libPaths(), collapse = ":"), "\n")
cat("R_LIBS_USER env:", Sys.getenv("R_LIBS_USER"), "\n")
cat("ExperimentHub cache dir:", cache_dir, "\n")

options(download.file.method = "libcurl")

safe_require <- function(pkg){
  if(!suppressWarnings(requireNamespace(pkg, quietly = TRUE))){
    cat(sprintf("Required package '%s' not installed in user lib. Please install before running.\n", pkg))
    quit(status = 11)
  }
}

for(p in c("ExperimentHub", "MouseGastrulationData", "SingleCellExperiment", "BiocFileCache")) safe_require(p)

suppressPackageStartupMessages({
  library(ExperimentHub)
  library(MouseGastrulationData)
  library(SingleCellExperiment)
})

cat("Setting ExperimentHub cache to:", cache_dir, "\n")
tryCatch({
  ExperimentHub::setExperimentHubOption("CACHE", cache_dir)
}, error = function(e){
  cat("Warning: failed to set CACHE option:", conditionMessage(e), "\n")
})

# Initialize ExperimentHub to warm index and print diagnostics
eh <- NULL
try({
  eh <- ExperimentHub()
  cat("ExperimentHub initialized. Cache path from options():", ExperimentHub::getExperimentHubOption("CACHE"), "\n")
}, silent = TRUE)

if(is.null(eh)){
  cat("Failed to initialize ExperimentHub. Exiting.\n")
  quit(status = 12)
}

try_fetch <- function(){
  cat("Attempting EmbryoAtlasData(type = 'processed') fetch...\n")
  sce <- NULL
  try({
    sce <- MouseGastrulationData::EmbryoAtlasData(type = "processed")
  }, silent = TRUE)
  if(!is.null(sce)) return(sce)

  cat("Direct call failed or returned NULL — attempting alternative retrieval via query()...\n")
  q <- NULL
  try({
    q <- query(eh, "MouseGastrulationData")
    cat("Query returned", length(q), "records\n")
  }, silent = TRUE)

  if(is.null(q) || length(q) == 0){
    cat("No resources found for MouseGastrulationData in ExperimentHub index.\n")
    return(NULL)
  }

  # Try to fetch first matching resource that looks like the processed dataset
  idx <- seq_along(q)
  for(i in idx){
    resname <- names(q)[i]
    cat(sprintf("Trying resource %d: %s\n", i, resname))
    sce <- NULL
    try({
      sce <- eh[[i]]
    }, silent = TRUE)
    if(!is.null(sce)){
      cat(sprintf("Successfully loaded resource %d -> %s\n", i, resname))
      return(sce)
    }
  }
  return(NULL)
}

remove_partial_cache <- function(){
  cat("Scanning cache dir for partial/failed MouseGastrulationData files to remove...\n")
  b <- list.files(cache_dir, recursive = TRUE, full.names = TRUE)
  bad <- grepl("MouseGastrulationData", basename(b), ignore.case = TRUE)
  if(any(bad)){
    to_rm <- b[bad]
    cat("Removing", length(to_rm), "partial cache files:\n")
    print(to_rm)
    unlink(to_rm, recursive = TRUE, force = TRUE)
  } else {
    cat("No obvious partial cache files found.\n")
  }
}

sce <- try_fetch()
if(is.null(sce)){
  # attempt cleanup and retry once
  remove_partial_cache()
  cat("Retrying fetch after cleanup...\n")
  sce <- try_fetch()
}

if(is.null(sce)){
  cat("Failed to obtain EmbryoAtlasData after retry.\n")
  quit(status = 13)
}

cat("Save RDS to:", out_rds, "\n")
saveRDS(sce, out_rds)
cat("Saved RDS. Object size (MB):", round(as.numeric(object.size(sce))/1024^2, 2), "\n")
cat("Elapsed:", difftime(Sys.time(), start_time, units = "mins"), "minutes\n")
cat("sessionInfo():\n")
print(sessionInfo())

invisible(NULL)
suppressPackageStartupMessages({
