#############################################
## merge_pregastrulation.R
## Build Pre-gastrulation pseudo-ages
## and merge into node_age_table
#############################################

suppressPackageStartupMessages({
  library(readr)
  library(dplyr)
  library(stringr)
})

## ---------------------------
## Paths
## ---------------------------

NODES_PATH <- "/project/imoskowitz/xyang2/SHH/Qiu_TimeLapse/other/nodes.txt"

NODE_AGE_PATH <- "/project/imoskowitz/xyang2/chrislowzhengxi/code/SEACellsBenchmark/age/tree/node_age_table_step1to3.tsv"

OUT_NODE_AGE_FINAL <- "/project/imoskowitz/xyang2/chrislowzhengxi/code/SEACellsBenchmark/age/tree/node_age_table_with_pregastrulation.tsv"

## ---------------------------
## Step 1: Build Pre-gastrulation pseudo-ages
## ---------------------------

nodes_pg <- read_tsv(NODES_PATH, show_col_types = FALSE) %>%
  filter(system == "Pre_gastrulation") %>%
  mutate(
    ## PGa_M1 -> 1, PGa_M16 -> 16
    stage_index = as.numeric(str_extract(meta_group, "\\d+"))
  ) %>%
  arrange(stage_index)

stopifnot(!any(is.na(nodes_pg$stage_index)))

pg_pseudo_age <- nodes_pg %>%
  mutate(
    node_age_mean =
      (stage_index - min(stage_index)) /
      (max(stage_index) - min(stage_index)),
    node_age_weighted = node_age_mean
  ) %>%
  select(
    system,
    meta_group,
    node_age_mean,
    node_age_weighted
  )

cat("Built Pre-gastrulation pseudo-ages:\n")
print(pg_pseudo_age)

## ---------------------------
## Step 2: Load existing node age table
## ---------------------------

node_age <- read_tsv(NODE_AGE_PATH, show_col_types = FALSE)

## ---------------------------
## Step 3: Append Pre-gastrulation nodes
## ---------------------------

## Identify Pre-gastrulation rows already present (likely zero)
existing_pg <- node_age %>%
  filter(system == "Pre_gastrulation")

## Build full Pre-gastrulation node rows
pg_full <- pg_pseudo_age %>%
  mutate(
    total_cells_node = NA_real_,
    n_celltypes_in_node = 1
  )

## Combine
node_age_final <- bind_rows(
  node_age %>% filter(system != "Pre_gastrulation"),
  pg_full
)

## Optional: order nicely
node_age_final <- node_age_final %>%
  arrange(system, meta_group)


## ---------------------------
## Step 4: Write output
## ---------------------------

write_tsv(node_age_final, OUT_NODE_AGE_FINAL)

cat("\nWrote final node age table:\n")
cat(OUT_NODE_AGE_FINAL, "\n")
