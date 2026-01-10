suppressPackageStartupMessages({
  library(zellkonverter)
  library(SingleCellExperiment)
})

## =========================
## Directory with h5ads
## =========================
H5_DIR <- "/project/imoskowitz/xyang2/chrislowzhengxi/results/raw_added_with_staging"

h5_files <- list.files(
  H5_DIR,
  pattern = "_adata_scale_with_staging\\.h5ad$",
  full.names = TRUE
)

cat("Found", length(h5_files), "h5ad files\n\n")

## =========================
## Inspect each file
## =========================
for (f in h5_files) {
  cat("========================================\n")
  cat("FILE:", basename(f), "\n")

  sce <- readH5AD(f)

  meta <- colData(sce)
  cols <- colnames(meta)

  cat("n_cells:", nrow(meta), "\n")
  cat("obs columns:\n")
  print(cols)

  shh_cols <- cols[grepl("SHH|shh", cols)]
  if (length(shh_cols) > 0) {
    cat("SHH-related columns:\n")
    print(shh_cols)
  } else {
    cat("NO SHH-related columns found\n")
  }

  cat("\n")
}
