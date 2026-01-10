## =========================
## Packages
## =========================
suppressPackageStartupMessages({
  library(zellkonverter)
  library(SingleCellExperiment)
  library(dplyr)
  library(stringr)
  library(gtools)
  library(tibble)
  library(readr)
})

## =========================
## Robust stage -> hours or rank
## =========================
stage_to_weights <- function(stages) {

  parse_embryo_day <- function(x) {
    x0 <- tolower(trimws(x))
    nums <- stringr::str_extract_all(x0, "(?<=e)\\d+\\.?\\d*")[[1]]
    if (length(nums) == 0) return(NA_real_)
    mean(as.numeric(nums))
  }

  uniq <- unique(stages)
  days <- vapply(uniq, parse_embryo_day, numeric(1))

  # Case 1: numeric embryo days exist
  if (any(!is.na(days))) {
    w <- (days - min(days, na.rm = TRUE)) * 24
  } else {
    # Case 2: pure rank fallback
    ord <- gtools::mixedorder(uniq)
    w <- seq_along(ord) - 1
  }

  tibble(stage = uniq, w = w)
}

## =========================
## Paths
## =========================
AGE_TSV <- "age_score_by_celltype_by_system.tsv"

BRAIN_FILES <- c(
  "/project/imoskowitz/xyang2/chrislowzhengxi/results/raw_added_with_staging/Neurons_adata_scale_with_staging.h5ad",
  "/project/imoskowitz/xyang2/chrislowzhengxi/results/raw_added_with_staging/Other_Brain_spinal_cord_adata_scale_with_staging.h5ad"
)

## =========================
## Load existing table
## =========================
age_all <- read_tsv(AGE_TSV, show_col_types = FALSE)

## =========================
## Read + merge brain metadata
## =========================
meta_brain <- lapply(BRAIN_FILES, function(f) {
  message("Reading ", basename(f))
  sce <- zellkonverter::readH5AD(f)
#   as.data.frame(colData(sce)) %>%
#     filter(staging != "P0") %>%
#     mutate(
#       system = "Brain_spinal_cord",
#       celltype_new = as.character(celltype_new),
#       stage = as.character(staging)
#     )
  as.data.frame(colData(sce)) %>%
    mutate(staging = as.character(staging)) %>%   # <<< FIX
    filter(staging != "P0") %>%
    mutate(
        system = "Brain_spinal_cord",
        celltype_new = as.character(celltype_new),
        stage = staging
    )
}) %>% bind_rows()

## =========================
## Compute age per celltype
## =========================
brain_age <- meta_brain %>%
  count(celltype_new, stage, name = "k") %>%
  group_by(celltype_new) %>%
  mutate(k_prop = k / sum(k)) %>%
  group_modify(~ {
    stage_map <- stage_to_weights(.x$stage)

    .x %>%
      left_join(stage_map, by = "stage") %>%
      summarise(
        total_cells = sum(k),
        age_weighted = sum(k_prop * w)
      )
  }) %>%
  ungroup() %>%
  mutate(system = "Brain_spinal_cord")

## =========================
## Append + overwrite
## =========================
age_all_fixed <- age_all %>%
  filter(system != "Brain_spinal_cord") %>%
  bind_rows(brain_age) %>%
  arrange(system, age_weighted)

write_tsv(age_all_fixed, AGE_TSV)

cat("DONE. Brain_spinal_cord appended correctly.\n")
