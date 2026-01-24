#############################################
## plot_tree_by_age.R
## Tree colored by node age
#############################################

suppressPackageStartupMessages({
  library(ggraph)
  library(tidygraph)
  library(igraph)
  library(readr)
  library(dplyr)
  library(colorspace)
})

## ---------------------------
## Paths
## ---------------------------

TREE_PATH <- "/project/imoskowitz/xyang2/chrislowzhengxi/results/ucell/tree/devtree_graph.rds"

NODE_AGE_PATH <- "/project/imoskowitz/xyang2/chrislowzhengxi/code/SEACellsBenchmark/age/tree/node_age_table_with_pregastrulation.tsv"

OUT_DIR <- "/project/imoskowitz/xyang2/chrislowzhengxi/code/SEACellsBenchmark/age/tree"

dir.create(OUT_DIR, showWarnings = FALSE, recursive = TRUE)

## ---------------------------
## Load tree
## ---------------------------

g <- readRDS(TREE_PATH)

## ---------------------------
## Load node age table
## ---------------------------

node_age <- read_tsv(NODE_AGE_PATH, show_col_types = FALSE)

## Build named vectors for fast attach
age_mean_map <- setNames(node_age$node_age_mean, node_age$meta_group)
age_weighted_map <- setNames(node_age$node_age_weighted, node_age$meta_group)

## Attach to graph
V(g)$age_mean <- age_mean_map[V(g)$meta_group]
V(g)$age_weighted <- age_weighted_map[V(g)$meta_group]

## ---------------------------
## Color palette
## ---------------------------

age_pal <- colorspace::sequential_hcl(
  n = 256,
  palette = "Plasma"
)

## ---------------------------
## A. Tree colored by average age
## ---------------------------

pdf(
  file.path(OUT_DIR, "devtree_tree_colored_by_age_mean.pdf"),
  width = 24,
  height = 12
)

p_age_mean <- ggraph(g, layout = "tree") +
  geom_edge_link(
    arrow = arrow(length = unit(2, "mm")),
    end_cap = circle(2, "mm"),
    colour = "grey70",
    linewidth = 0.3
  ) +
  geom_node_point(
    aes(colour = age_mean),
    size = 2
  ) +
  scale_colour_gradientn(
    colours = age_pal,
    name = "Average age",
    na.value = "grey85"
  ) +
  theme_void() +
  theme(
    legend.position = "bottom",
    plot.margin = margin(10, 10, 10, 10)
  )

print(p_age_mean)
dev.off()

## ---------------------------
## B. Tree colored by weighted age
## ---------------------------

pdf(
  file.path(OUT_DIR, "devtree_tree_colored_by_age_weighted.pdf"),
  width = 24,
  height = 12
)

p_age_weighted <- ggraph(g, layout = "tree") +
  geom_edge_link(
    arrow = arrow(length = unit(2, "mm")),
    end_cap = circle(2, "mm"),
    colour = "grey70",
    linewidth = 0.3
  ) +
  geom_node_point(
    aes(colour = age_weighted),
    size = 2
  ) +
  scale_colour_gradientn(
    colours = age_pal,
    name = "Weighted age",
    na.value = "grey85"
  ) +
  theme_void() +
  theme(
    legend.position = "bottom",
    plot.margin = margin(10, 10, 10, 10)
  )

print(p_age_weighted)
dev.off()

cat("Wrote age-colored trees to:\n", OUT_DIR, "\n")
