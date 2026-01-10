# Not ran yet 
#!/usr/bin/env Rscript

library(ggraph)
library(tidygraph)
library(igraph)
library(dplyr)
library(readr)
library(viridis)
library(scales)
library(grid)

TREE_DIR   <- "/project/imoskowitz/xyang2/chrislowzhengxi/results/ucell/tree"
STAGE_TSV  <- "/project/imoskowitz/xyang2/chrislowzhengxi/results/staging/node_stage_code.tsv"
OUT_DIR    <- TREE_DIR

g <- readRDS(file.path(TREE_DIR, "devtree_graph.rds"))
stage_tbl <- read_tsv(STAGE_TSV, show_col_types = FALSE)

# Attach staging to vertices by meta_group
g <- g %>%
  activate(nodes) %>%
  left_join(stage_tbl, by = "meta_group")

# Sanity check
stopifnot(sum(is.na(V(g)$mean_stage_code)) == 0)

# Consistent scale across whole tree
stage_rng <- range(V(g)$mean_stage_code, na.rm = TRUE)

pdf(file.path(OUT_DIR, "devtree_tree_staging_magma_v1.pdf"), width = 24, height = 12)

p <- ggraph(g, layout = "tree") +
  geom_edge_link(
    arrow = arrow(length = unit(2, "mm")),
    end_cap = circle(2, "mm"),
    colour = "grey70",
    linewidth = 0.3
  ) +
  geom_node_point(
    aes(colour = mean_stage_code),
    size = 2
  ) +
  scale_colour_viridis(
    option = "magma",
    limits = stage_rng,
    name = "Mean developmental stage\n(stage_code)"
  ) +
  theme_void() +
  theme(
    legend.position = "bottom",
    plot.margin = margin(10, 10, 10, 10)
  )

print(p)
dev.off()

pdf(file.path(OUT_DIR, "devtree_sugiyama_staging_magma_v1.pdf"), width = 24, height = 12)

p2 <- ggraph(g, layout = "sugiyama") +
  geom_edge_link(
    arrow = arrow(length = unit(2, "mm")),
    end_cap = circle(2, "mm"),
    colour = "grey70",
    linewidth = 0.3
  ) +
  geom_node_point(
    aes(colour = mean_stage_code),
    size = 2
  ) +
  scale_colour_viridis(
    option = "magma",
    limits = stage_rng,
    name = "Mean developmental stage\n(stage_code)"
  ) +
  theme_void() +
  theme(
    legend.position = "bottom",
    plot.margin = margin(10, 10, 10, 10)
  )

print(p2)
dev.off()

message("Saved staging trees to: ", OUT_DIR)
