## amd-hm 44476975

## =========================
## Packages
## =========================
suppressPackageStartupMessages({
  library(zellkonverter)
  library(SingleCellExperiment)
  library(dplyr)
  library(tibble)
  library(stringr)
  library(gtools)
  library(readr)
})

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
      return(mean(as.numeric(nums)))
    }
    return(NA_real_)
  }

  t_day <- vapply(lev, parse_embryo_day, numeric(1))

  if (any(is.na(t_day))) {
    w <- seq_along(lev) - 1
  } else {
    w <- t_day
    if (origin == "min") w <- w - min(w)
    if (unit == "hours") w <- w * 24
  }

  tibble(stage = lev, w = w)
}

## =========================
## Input directory
## =========================
INPUT_DIR <- "/project/imoskowitz/xyang2/chrislowzhengxi/results/raw_added_with_staging"

files <- list.files(
  INPUT_DIR,
  pattern = "_adata_scale_with_staging\\.h5ad$",
  full.names = TRUE
)

## =========================
## Loop over systems
## =========================
all_results <- list()

for (f in files) {

  message("Processing: ", basename(f))

  sce <- zellkonverter::readH5AD(f)
  meta <- as.data.frame(colData(sce))

  system_name <- basename(f) |>
    str_replace("_adata_scale_with_staging.h5ad", "")

  if (system_name %in% c("Neurons", "Other_Brain_spinal_cord")) {
    system_name <- "Brain_spinal_cord"
  }

  meta <- meta %>%
    filter(staging != "P0") %>%
    mutate(
      system = system_name,
      celltype_new = as.character(celltype_new),
      stage = as.character(staging)
    )

  if (nrow(meta) == 0) next

  df_counts <- meta %>%
    count(celltype_new, stage, name = "k") %>%
    ungroup()

  stage_map <- infer_stage_weights(
    df_counts$stage,
    unit = "hours",
    scale01 = FALSE,
    origin = "min"
  )

  df_age <- df_counts %>%
    left_join(stage_map, by = "stage") %>%
    group_by(celltype_new) %>%
    summarise(
      total_cells = sum(k),
      age_mean     = sum(k * w) / sum(k),
      age_weighted = sum((k / sum(k)) * w),
      .groups = "drop"
    ) %>%
    mutate(system = system_name)

  all_results[[system_name]] <- df_age
}

## =========================
## Combine + output
## =========================
age_all <- bind_rows(all_results) %>%
  select(system, celltype_new, total_cells, age_mean, age_weighted) %>%
  arrange(system, age_weighted)

write_tsv(
  age_all,
  "age_mean_and_weighted_by_celltype_by_system.tsv"
)

cat("DONE. Wrote age_mean_and_weighted_by_celltype_by_system.tsv\n")
