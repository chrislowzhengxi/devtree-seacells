## =========================
## Packages
## =========================
suppressPackageStartupMessages({
  library(zellkonverter)         # readH5AD
  library(SingleCellExperiment)
  library(dplyr)
  library(tibble)
  library(stringr)
  library(gtools)                # mixedorder
  library(readr)
})

## =========================
## Input: ONE dataset only
## =========================
h5ad_file <- "/project/imoskowitz/xyang2/chrislowzhengxi/results/raw_added_with_staging/Gastrulation_adata_scale_with_staging.h5ad"

sce <- zellkonverter::readH5AD(h5ad_file)

## =========================
## Helper: infer stage weights
## =========================
infer_stage_weights <- function(stages,
                                unit = c("hours", "days", "rank"),
                                scale01 = FALSE,
                                origin = c("min", "absolute")) {

  unit   <- match.arg(unit)
  origin <- match.arg(origin)

  lev <- unique(as.character(stages))
  lev <- lev[!is.na(lev)]
  if (length(lev) < 2) stop("Need >= 2 unique stages")

  lev <- lev[gtools::mixedorder(lev)]

  parse_embryo_day <- function(x) {
    x0 <- tolower(trimws(x))

    if (str_detect(x0, "e\\s*\\d")) {
      nums <- str_extract_all(x0, "(?<=e)\\s*\\d+\\.?\\d*")[[1]]
      nums <- gsub("\\s+", "", nums)
      if (length(nums) >= 1) return(mean(as.numeric(nums)))
    }

    suppressWarnings(num <- as.numeric(x0))
    if (!is.na(num)) return(num)

    return(NA_real_)
  }

  t_day <- vapply(lev, parse_embryo_day, numeric(1))

  if (unit == "rank" || any(is.na(t_day))) {
    w <- seq_along(lev) - 1
  } else {
    w <- t_day
    if (origin == "min") w <- w - min(w)
    if (unit == "hours") w <- w * 24
  }

  if (scale01) {
    if (max(w) == min(w)) {
      w <- rep(0, length(w))
    } else {
      w <- (w - min(w)) / (max(w) - min(w))
    }
  }

  tibble(stage = lev, w = w)
}

## =========================
## Extract metadata
## =========================
meta <- as.data.frame(colData(sce)) %>%
  filter(staging != "P0") %>%                # EXCLUDE P0
  mutate(
    celltype_new = as.character(celltype_new),
    stage = as.character(staging)
  )

stopifnot(all(c("celltype_new", "stage") %in% colnames(meta)))

## =========================
## Count cells per (celltype, stage)
## =========================
df_counts <- meta %>%
  count(celltype_new, stage, name = "k") %>%
  group_by(celltype_new) %>%
  mutate(k_prop = k / sum(k)) %>%
  ungroup()

## =========================
## Stage → time (hours)
## =========================
stage_map <- infer_stage_weights(
  df_counts$stage,
  unit = "hours",
  scale01 = FALSE,
  origin = "min"
)

df_counts <- df_counts %>%
  left_join(stage_map, by = "stage")

## =========================
## Weighted age per celltype
## =========================
df_age <- df_counts %>%
  group_by(celltype_new) %>%
  summarise(
    total_cells = sum(k),
    age_weighted = sum(k_prop * w),
    .groups = "drop"
  ) %>%
  arrange(age_weighted)

## =========================
## Output
## =========================
write_tsv(df_age, "age_score_by_celltype.tsv")

cat("DONE. Wrote age_score_by_celltype.tsv\n")
