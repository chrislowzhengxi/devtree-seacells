#############################################
## build_node_age_table_step1to3.R
## Steps 1–3:
##  1) Standardize + combine age tables (celltype-level)
##  2) Merge with nodes.txt (celltype -> meta_group)
##  3) Aggregate to node-level ages
#############################################

suppressPackageStartupMessages({
  library(readr)
  library(dplyr)
  library(stringr)
})

## ---------------------------
## Paths
## ---------------------------

AGE_ALL_PATH <- "/project/imoskowitz/xyang2/chrislowzhengxi/code/SEACellsBenchmark/age/age_mean_and_weighted_by_celltype_by_system_final.tsv"

AGE_GAST_PATH <- "/project/imoskowitz/xyang2/chrislowzhengxi/code/SEACellsBenchmark/age/age_mean_and_shh_weighted_by_celltype_gastrulation.tsv"

NODES_PATH <- "/project/imoskowitz/xyang2/SHH/Qiu_TimeLapse/other/nodes.txt"

OUT_DIR <- "/project/imoskowitz/xyang2/chrislowzhengxi/code/SEACellsBenchmark/age"
OUT_CELLTYPE_COMBINED <- file.path(OUT_DIR, "age_celltype_combined_standardized.tsv")
OUT_CELLTYPE_WITH_NODES <- file.path(OUT_DIR, "age_celltype_with_nodes.tsv")
OUT_NODE_AGE_TABLE <- file.path(OUT_DIR, "node_age_table_step1to3.tsv")

dir.create(OUT_DIR, showWarnings = FALSE, recursive = TRUE)

## ---------------------------
## Step 1A: Load non-gastrulation age table (already has system)
## ---------------------------

age_all <- read_tsv(AGE_ALL_PATH, show_col_types = FALSE)

age_all_std <- age_all %>%
  transmute(
    system = as.character(system),
    celltype_new = as.character(celltype_new),
    total_cells = as.numeric(total_cells),
    age_mean = as.numeric(age_mean),
    age_weighted_celltype = as.numeric(age_weighted)
  )

## ---------------------------
## Step 1B: Load gastrulation table, rename columns, add system
## ---------------------------

age_gast <- read_tsv(AGE_GAST_PATH, show_col_types = FALSE)

age_gast_std <- age_gast %>%
  transmute(
    system = "Gastrulation",
    celltype_new = as.character(celltype_update),
    total_cells = as.numeric(total_cells),
    age_mean = as.numeric(age_mean),
    age_weighted_celltype = as.numeric(age_shh_weighted)
  )

## ---------------------------
## Step 1C: Combine (celltype-level)
## ---------------------------

age_celltype_combined <- bind_rows(age_all_std, age_gast_std) %>%
  mutate(
    system = str_trim(system),
    celltype_new = str_trim(celltype_new)
  )

write_tsv(age_celltype_combined, OUT_CELLTYPE_COMBINED)

## ---------------------------
## Step 2: Load nodes.txt and join (celltype -> meta_group)
## ---------------------------

nodes <- read_tsv(NODES_PATH, show_col_types = FALSE) %>%
  transmute(
    system = str_trim(as.character(system)),
    meta_group = as.character(meta_group),
    celltype_new = str_trim(as.character(celltype_new))
  )

age_celltype_with_nodes <- age_celltype_combined %>%
  left_join(nodes, by = c("system", "celltype_new"))

## Quick diagnostics (prints to console)
cat("\n--- Join diagnostics ---\n")
cat("Rows in combined age table:", nrow(age_celltype_combined), "\n")
cat("Rows after join:", nrow(age_celltype_with_nodes), "\n")
cat("Rows with missing meta_group:", sum(is.na(age_celltype_with_nodes$meta_group)), "\n")

write_tsv(age_celltype_with_nodes, OUT_CELLTYPE_WITH_NODES)

## ---------------------------
## Step 3: Aggregate to node-level ages
## Recommended: cell-weighted mean to avoid tiny celltypes dominating.
## ---------------------------

node_age_table <- age_celltype_with_nodes %>%
  filter(!is.na(meta_group)) %>%
  group_by(system, meta_group) %>%
  summarise(
    total_cells_node = sum(total_cells, na.rm = TRUE),

    ## Node average age (cell-weighted average of age_mean)
    node_age_mean = weighted.mean(age_mean, w = total_cells, na.rm = TRUE),

    ## Node weighted age (cell-weighted average of age_weighted_celltype)
    node_age_weighted = weighted.mean(age_weighted_celltype, w = total_cells, na.rm = TRUE),

    ## Useful QC counts
    n_celltypes_in_node = dplyr::n(),
    .groups = "drop"
  ) %>%
  arrange(system, meta_group)

write_tsv(node_age_table, OUT_NODE_AGE_TABLE)

cat("\nWrote outputs:\n")
cat("  ", OUT_CELLTYPE_COMBINED, "\n")
cat("  ", OUT_CELLTYPE_WITH_NODES, "\n")
cat("  ", OUT_NODE_AGE_TABLE, "\n")
