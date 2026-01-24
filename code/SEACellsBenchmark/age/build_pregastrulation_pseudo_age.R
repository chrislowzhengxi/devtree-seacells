#############################################
## build_pregastrulation_pseudo_age.R
#############################################

suppressPackageStartupMessages({
  library(dplyr)
  library(readr)
  library(stringr)
})

## ---------------------------
## Paths
## ---------------------------

NODES_PATH <- "/project/imoskowitz/xyang2/SHH/Qiu_TimeLapse/other/nodes.txt"

NODE_AGE_PATH <- "/project/imoskowitz/xyang2/chrislowzhengxi/code/SEACellsBenchmark/age/tree/node_age_table_step1to3.tsv"

OUT_NODE_AGE_FINAL <- "/project/imoskowitz/xyang2/chrislowzhengxi/code/SEACellsBenchmark/age/tree/node_age_table_with_pregastrulation.tsv"

## ---------------------------
## Load nodes and extract Pre-gastrulation order
## ---------------------------

nodes_pg <- read_tsv(NODES_PATH, show_col_types = FALSE) %>%
  filter(system == "Pre_gastrulation") %>%
  mutate(
    ## Extract numeric stage from meta_group: PGa_M1 -> 1
    stage_index = as.numeric(str_extract(meta_group, "\\d+"))
  ) %>%
  arrange(stage_index)

stopifnot(!any(is.na(nodes_pg$stage_index)))

## ---------------------------
## Convert stage index -> pseudo-age
## ---------------------------

pg_pseudo_age <- nodes_pg %>%
  mutate(
    node_age_mean = (stage_index - min(stage_index)) /
                    (max(stage_index) - min(stage_index)),
    node_age_weighted = node_age_mean
  ) %>%
  select(
    system,
    meta_group,
    node_age_mean,
    node_age_weighted
  )

print(pg_pseudo_age)
