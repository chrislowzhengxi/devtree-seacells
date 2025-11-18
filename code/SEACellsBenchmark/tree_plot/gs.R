library(dplyr)
library(igraph)
library(tidygraph)

tree_dir  <- "/project/imoskowitz/xyang2/chrislowzhengxi/results/ucell/tree"
nodes_path <- file.path(tree_dir, "devtree_nodes_for_graph.tsv")
edges_path <- file.path(tree_dir, "devtree_edges_for_graph.tsv")

nodes <- read.delim(nodes_path, sep = "\t", stringsAsFactors = FALSE)
edges <- read.delim(edges_path, sep = "\t", stringsAsFactors = FALSE)

colnames(edges)
# [1] "system" "x" "y" "x_name" "y_name" "edge_type" "abs_delta" "delta" "sh_x" "sh_y" "x_id" "y_id"

# Get node ids for L_M22 and L_M5
x_id_val <- nodes$id[nodes$meta_group == "L_M22"]
y_id_val <- nodes$id[nodes$meta_group == "L_M5"]

stopifnot(length(x_id_val) == 1, length(y_id_val) == 1)

# Only add if it does not already exist
existing <- edges %>%
  filter(x == "L_M22", y == "L_M5", edge_type == "Developmental progression")

if (nrow(existing) == 0) {
  new_edge <- data.frame(
    system    = "Lateral_plate_mesoderm",
    x         = "L_M22",
    y         = "L_M5",
    x_name    = "Second heart field",
    y_name    = "Atrial cardiomyocytes",
    edge_type = "Developmental progression",
    # from your long row: abs_delta and delta
    abs_delta = 0.063567,
    delta     = -0.063567,
    # sh_x, sh_y not actually used for plotting (node sh comes from nodes),
    # so we can safely set them to NA
    sh_x      = NA_real_,
    sh_y      = NA_real_,
    x_id      = x_id_val,
    y_id      = y_id_val,
    stringsAsFactors = FALSE
  )

  edges2 <- bind_rows(edges, new_edge)

  # Overwrite TSV (or change filename if you want a versioned one)
  write.table(
    edges2,
    edges_path,
    sep = "\t",
    quote = FALSE,
    row.names = FALSE
  )

  cat("Added SHF → Atrial cardiomyocytes edge and updated devtree_edges_for_graph.tsv\n")
} else {
  cat("Edge already present, no change made.\n")
}
