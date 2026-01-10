## =========================
## Packages
## =========================
suppressPackageStartupMessages({
  library(zellkonverter)         # readH5AD()
  library(SingleCellExperiment)  # colData()
  library(dplyr)
  library(tibble)
  library(stringr)
  library(gtools)                # mixedorder()
  library(ggplot2)
  library(readr)
})

## =========================
## Input
## =========================
# Option A (most common): read from an .h5ad file
h5ad_file <- "path/to/adata.h5ad"
sce <- zellkonverter::readH5AD(h5ad_file)

# Option B: if you already have a python AnnData object `adata` via reticulate:
# sce <- zellkonverter::AnnData2SCE(adata)

## =========================
## Helper: infer stage weights w_s
## =========================
infer_stage_weights <- function(stages,
                                unit = c("hours", "days", "rank"),
                                scale01 = FALSE,
                                origin = c("min", "absolute")) {
  unit   <- match.arg(unit)
  origin <- match.arg(origin)

  lev <- unique(as.character(stages))
  lev <- lev[!is.na(lev)]
  if (length(lev) < 2) stop("Need >= 2 unique stages to compute weights.")

  # Order stage labels in a human/numeric-friendly way (e.g., E6.5 < E10)
  lev <- lev[gtools::mixedorder(lev)]

  # Try to parse embryo day from labels like "E6.5" or ranges like "E8.0-E8.5"
  parse_embryo_day <- function(x) {
    x0 <- tolower(trimws(x))

    # Catch things like E6.5 or E8.0-E8.5 (take mean if multiple E-values)
    if (str_detect(x0, "e\\s*\\d")) {
      nums <- str_extract_all(x0, "(?<=e)\\s*\\d+\\.?\\d*")[[1]]
      nums <- gsub("\\s+", "", nums)
      if (length(nums) >= 1) return(mean(as.numeric(nums)))
    }

    # Fallback: purely numeric stage already (e.g., "6.5")
    suppressWarnings(num <- as.numeric(x0))
    if (!is.na(num)) return(num)

    return(NA_real_)
  }

  t_day <- vapply(lev, parse_embryo_day, numeric(1))

  if (unit == "rank" || any(is.na(t_day))) {
    # Rank-based assignment: 0,1,2,... following lev order
    w <- seq_along(lev) - 1
  } else {
    # Time-based assignment using embryo days
    w <- t_day
    if (origin == "min") w <- w - min(w)

    if (unit == "hours") w <- w * 24
    # unit == "days" leaves it in days
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
## Extract obs (colData) and sanity-check columns
## =========================
meta <- as.data.frame(colData(sce))
stopifnot(all(c("celltype_new", "stage") %in% colnames(meta)))

meta <- meta %>%
  mutate(
    celltype_new = as.character(celltype_new),
    stage        = as.character(stage)
  )

## =========================
## k_{i,s} counts and within-celltype proportions k_s
## =========================
df_counts <- meta %>%
  count(celltype_new, stage, name = "k") %>%
  group_by(celltype_new) %>%
  mutate(k_prop = k / sum(k)) %>%   # this is k_s in the age-score formula
  ungroup()

## =========================
## Choose w_s
## =========================
# Recommended default for development datasets with E-labels:
#   unit="hours" -> hours since earliest stage
# If you want 0..1 scaling for cross-dataset comparability, set scale01=TRUE
stage_map <- infer_stage_weights(df_counts$stage,
                                 unit   = "hours",
                                 scale01 = FALSE,
                                 origin = "min")

df_counts <- df_counts %>%
  left_join(stage_map, by = "stage") %>%
  mutate(stage = factor(stage, levels = stage_map$stage))

## =========================
## Age_score per celltype: sum_s k_s * w_s
## =========================
df_age <- df_counts %>%
  group_by(celltype_new) %>%
  summarise(
    total_cells = sum(k),
    age_score   = sum(k_prop * w),
    .groups = "drop"
  ) %>%
  arrange(age_score)

## =========================
## Output (3): text table (celltype_new, age_score)
## =========================
age_table <- df_age %>% select(celltype_new, age_score)
print(age_table)
readr::write_tsv(age_table, "age_score_by_celltype.tsv")

## =========================
## Output (1): bar plot (counts), x=celltype_new, fill=stage
## =========================
p_bar <- ggplot(df_counts, aes(x = celltype_new, y = k, fill = stage)) +
  geom_col(width = 0.9) +
  theme_bw() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
  labs(
    x = "celltype_new",
    y = "Cell population (k_{i,s} counts)",
    fill = "stage",
    title = "Cell population by cell type and stage"
  )

# Optional alternative: proportions instead of counts
p_bar_prop <- ggplot(df_counts, aes(x = celltype_new, y = k, fill = stage)) +
  geom_col(position = "fill", width = 0.9) +
  theme_bw() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
  labs(
    x = "celltype_new",
    y = "Proportion within cell type",
    fill = "stage",
    title = "Stage composition within each cell type"
  )

## =========================
## Output (2): line plot (age_score vs celltype_new)
## =========================
df_age <- df_age %>%
  mutate(celltype_new = factor(celltype_new, levels = celltype_new)) # keep sorted

p_line <- ggplot(df_age, aes(x = celltype_new, y = age_score, group = 1)) +
  geom_line() +
  geom_point() +
  theme_bw() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
  labs(
    x = "celltype_new",
    y = "age_score",
    title = "Age score by cell type"
  )

## =========================
## Save plots
## =========================
ggsave("cell_population_by_stage_barplot.pdf", p_bar, width = 11, height = 5)
ggsave("cell_population_by_stage_barplot_prop.pdf", p_bar_prop, width = 11, height = 5)
ggsave("age_score_by_celltype_lineplot.pdf", p_line, width = 11, height = 4)

# If running interactively (RStudio), you can also just print plots:
print(p_bar)
print(p_bar_prop)
print(p_line)
