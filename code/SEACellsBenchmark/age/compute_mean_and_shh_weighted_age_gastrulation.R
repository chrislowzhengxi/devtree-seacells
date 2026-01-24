#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(zellkonverter)
  library(SingleCellExperiment)
  library(dplyr)
  library(readr)
})

## =========================
## Input
## =========================
h5ad_file <- "/project/imoskowitz/xyang2/chrislowzhengxi/results/ucell/Gastrulation/Gastrulation_adata_with_ucell_with_staging.h5ad"

sce <- zellkonverter::readH5AD(h5ad_file)
meta <- as.data.frame(colData(sce))

## =========================
## Sanity checks
## =========================
stopifnot(all(c(
  "celltype_update",
  "stage_code",
  "SHH_UCell_score"
) %in% colnames(meta)))

## =========================
## Clean + compute ages
## =========================
df_age <- meta %>%
  filter(!is.na(stage_code)) %>%
  mutate(
    celltype_update = as.character(celltype_update),
    stage_code      = as.numeric(stage_code),
    shh_score       = as.numeric(SHH_UCell_score)
  ) %>%
  group_by(celltype_update) %>%
  summarise(
    total_cells = n(),

    ## Mean age (unweighted)
    age_mean = mean(stage_code, na.rm = TRUE),

    ## SHH-weighted age
    age_shh_weighted =
      sum(stage_code * shh_score, na.rm = TRUE) /
      sum(shh_score, na.rm = TRUE),

    .groups = "drop"
  ) %>%
  arrange(age_shh_weighted)

## =========================
## Output
## =========================
write_tsv(
  df_age,
  "age_mean_and_shh_weighted_by_celltype_gastrulation.tsv"
)

cat("DONE: age_mean_and_shh_weighted_by_celltype_gastrulation.tsv\n")
