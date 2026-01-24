#############################################
## compute_node_deltas.R
## Correct version for tbl_graph
#############################################

suppressPackageStartupMessages({
  library(tidygraph)
  library(dplyr)
  library(readr)
})

## ---------------------------
## Paths
## ---------------------------

TREE_PATH <- "/project/imoskowitz/xyang2/chrislowzhengxi/results/ucell/tree/devtree_graph.rds"

NODE_AGE_PATH <- "/project/imoskowitz/xyang2/chrislowzhengxi/code/SEACellsBenchmark/age/tree/node_age_table_with_pregastrulation.tsv"

OUT_NODE_DELTA_PATH <- "/project/imoskowitz/xyang2/chrislowzhengxi/code/SEACellsBenchmark/age/tree/node_age_table_with_delta.tsv"

## ---------------------------
## Load tree
## ---------------------------

g <- readRDS(TREE_PATH)

## ---------------------------
## Extract edges (tbl_graph-safe)
## ---------------------------

edge_df <- g %>%
  activate(edges) %>%
  as_tibble() %>%
  select(from, to, delta) %>%
  mutate(delta = ifelse(is.na(delta), 0, delta))

## ---------------------------
## Map node indices -> meta_group
## ---------------------------

node_df <- g %>%
  activate(nodes) %>%
  as_tibble() %>%
  mutate(node_index = row_number()) %>%
  select(node_index, meta_group)

edge_df <- edge_df %>%
  left_join(node_df, by = c("from" = "node_index")) %>%
  rename(from_meta = meta_group) %>%
  left_join(node_df, by = c("to" = "node_index")) %>%
  rename(to_meta = meta_group)

## ---------------------------
## Compute delta_input (incoming)
## ---------------------------

delta_input <- edge_df %>%
  group_by(to_meta) %>%
  summarise(
    delta_input = sum(delta),
    .groups = "drop"
  ) %>%
  rename(meta_group = to_meta)

## ---------------------------
## Compute delta_output (outgoing)
## ---------------------------

delta_output <- edge_df %>%
  group_by(from_meta) %>%
  summarise(
    delta_output = sum(delta),
    .groups = "drop"
  ) %>%
  rename(meta_group = from_meta)

## ---------------------------
## Combine deltas cleanly
## ---------------------------

delta_table <- full_join(
  delta_input,
  delta_output,
  by = "meta_group"
) %>%
  mutate(
    delta_input = ifelse(is.na(delta_input), 0, delta_input),
    delta_output = ifelse(is.na(delta_output), 0, delta_output),
    delta_delta = delta_input - delta_output
  )

## ---------------------------
## Merge into node age table
## ---------------------------

node_age <- read_tsv(NODE_AGE_PATH, show_col_types = FALSE)

node_final <- node_age %>%
  left_join(delta_table, by = "meta_group") %>%
  mutate(
    delta_input = ifelse(is.na(delta_input), 0, delta_input),
    delta_output = ifelse(is.na(delta_output), 0, delta_output),
    delta_delta = ifelse(is.na(delta_delta), 0, delta_delta)
  )

## ---------------------------
## Sanity checks
## ---------------------------

cat("\nDelta_delta summary:\n")
print(summary(node_final$delta_delta))

## ---------------------------
## Write output
## ---------------------------

write_tsv(node_final, OUT_NODE_DELTA_PATH)

cat("\nWrote node table with deltas:\n")
cat(OUT_NODE_DELTA_PATH, "\n")
