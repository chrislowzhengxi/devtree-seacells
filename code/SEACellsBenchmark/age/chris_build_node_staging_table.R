#!/usr/bin/env Rscript

library(zellkonverter)  # readH5AD
library(dplyr)
library(purrr)
library(readr)
library(stringr)

IN_DIR  <- "/project/imoskowitz/xyang2/chrislowzhengxi/results/raw_added_with_staging"
OUT_DIR <- "/project/imoskowitz/xyang2/chrislowzhengxi/results/staging"
dir.create(OUT_DIR, recursive = TRUE, showWarnings = FALSE)

h5_files <- list.files(IN_DIR, pattern = "_adata_scale_with_staging\\.h5ad$", full.names = TRUE)
stopifnot(length(h5_files) > 0)

message("Found ", length(h5_files), " h5ad files")

node_stage_tbl <- map_dfr(h5_files, function(f) {
  message("Reading: ", basename(f))
  sce <- readH5AD(f)

  df <- as.data.frame(colData(sce))
  stopifnot(all(c("meta_group", "system", "stage_code") %in% colnames(df)))

  df %>%
    filter(!is.na(meta_group), !is.na(stage_code)) %>%
    group_by(meta_group, system) %>%
    summarise(
      mean_stage_code = mean(stage_code),
      n_cells = n(),
      .groups = "drop"
    )
})

out_tsv <- file.path(OUT_DIR, "node_stage_code.tsv")
write_tsv(node_stage_tbl, out_tsv)

message("Saved: ", out_tsv)
