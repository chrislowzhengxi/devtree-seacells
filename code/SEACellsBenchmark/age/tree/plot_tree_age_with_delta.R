#############################################
## plot_tree_age_with_delta.R
## Node color = age
## Node size  = |delta_delta|
#############################################

suppressPackageStartupMessages({
  library(ggraph)
  library(tidygraph)
  library(igraph)
  library(readr)
  library(dplyr)
  library(colorspace)
  library(scales)
})

## ---------------------------
## Paths
## ---------------------------

TREE_PATH <- "/project/imoskowitz/xyang2/chrislowzhengxi/results/ucell/tree/devtree_graph.rds"

NODE_TABLE_PATH <- "/project/imoskowitz/xyang2/chrislowzhengxi/code/SEACellsBenchmark/age/tree/node_age_table_with_delta.tsv"

OUT_DIR <- "/project/imoskowitz/xyang2/chrislowzhengxi/code/SEACellsBenchmark/age/tree"

dir.create(OUT_DIR, showWarnings = FALSE, recursive = TRUE)

## ---------------------------
## Load tree
## ---------------------------
age_pal_orange <- colorspace::sequential_hcl(
  256,
  palette = "YlOrRd",
  rev = FALSE
)


g <- readRDS(TREE_PATH)
E(g)$delta     <- tidyr::replace_na(E(g)$delta, 0)
E(g)$abs_delta <- tidyr::replace_na(E(g)$abs_delta, 0)

## ---------------------------
## Load node table with delta
## ---------------------------

node_tbl <- read_tsv(NODE_TABLE_PATH, show_col_types = FALSE)

## Attach node attributes
age_mean_map <- setNames(node_tbl$node_age_mean, node_tbl$meta_group)
age_weighted_map <- setNames(node_tbl$node_age_weighted, node_tbl$meta_group)
delta_delta_map <- setNames(node_tbl$delta_delta, node_tbl$meta_group)

V(g)$age_mean <- age_mean_map[V(g)$meta_group]
V(g)$age_weighted <- age_weighted_map[V(g)$meta_group]
V(g)$delta_delta <- delta_delta_map[V(g)$meta_group]

## Size transform for visibility
V(g)$delta_size <- sqrt(abs(V(g)$delta_delta))

## ---------------------------
## Color palette
## ---------------------------

age_pal <- colorspace::sequential_hcl(
  n = 256,
  palette = "YlOrRd",
  rev = FALSE
)

## ---------------------------
## A. Tree: MEAN age + delta_delta size
## ---------------------------

pdf(
  file.path(OUT_DIR, "devtree_tree_meanAge_nodeSize_deltaDelta.pdf"),
  width = 24,
  height = 12
)

# p_mean <- ggraph(g, layout = "tree") +
#     geom_edge_link(
#     aes(
#         edge_colour = delta,
#         edge_width  = pmax(abs(delta), 0.002)
#     ),
#     arrow = arrow(length = unit(2, "mm")),
#     end_cap = circle(2, "mm"),
#     alpha = 0.85,
#     show.legend = TRUE
#     )+
#   geom_node_point(
#     aes(
#       colour = age_mean,
#       size = delta_size
#     ),
#     alpha = 0.9
#   ) +
#   scale_colour_gradientn(
#     colours = age_pal_orange,
#     name = "Mean age",
#     na.value = "grey85"
#   ) +
#   scale_size_continuous(
#     range = c(1.5, 7),
#     name = "|delta_delta|"
#   ) +
#   theme_void() +
#   theme(
#     legend.position = "bottom",
#     plot.margin = margin(10, 10, 10, 10)
#   )
p_mean <- ggraph(g, layout = "tree") +
  geom_edge_link(
    aes(
      edge_colour = delta,
      edge_width  = pmax(abs(delta), 0.002)
    ),
    arrow = arrow(length = unit(2, "mm")),
    end_cap = circle(2, "mm"),
    alpha = 0.85,
    show.legend = TRUE
  ) +
  scale_edge_colour_gradient2(
    low = "blue",
    mid = "grey80",
    high = "red",
    midpoint = 0,
    name = "Edge delta"
  ) +
  scale_edge_width(
    range = c(0.25, 1.2),
    name = "|edge delta|"
  ) +
  geom_node_point(
    aes(
      colour = age_mean,
      size = delta_size
    ),
    alpha = 0.9
  ) +
  scale_colour_gradientn(
    colours = age_pal_orange,
    name = "Mean age",
    na.value = "grey85"
  ) +
  scale_size_continuous(
    range = c(1.5, 7),
    name = "|delta_delta|"
  ) +
  theme_void() +
  theme(
    legend.position = "bottom",
    plot.margin = margin(10, 10, 10, 10)
  )


print(p_mean)
dev.off()

## ---------------------------
## B. Tree: WEIGHTED age + delta_delta size
## ---------------------------

pdf(
  file.path(OUT_DIR, "devtree_tree_weightedAge_nodeSize_deltaDelta.pdf"),
  width = 24,
  height = 12
)

p_weighted <- ggraph(g, layout = "tree") +
  geom_edge_link(
    aes(
      edge_colour = delta,
      edge_width  = pmax(abs(delta), 0.002)
    ),
    arrow = arrow(length = unit(2, "mm")),
    end_cap = circle(2, "mm"),
    alpha = 0.8,
    show.legend = TRUE
  ) +
  scale_edge_colour_gradient2(
    low = "blue",
    mid = "grey80",
    high = "red",
    midpoint = 0,
    name = "Edge delta"
  ) +
  scale_edge_width(
    range = c(0.25, 1.2),
    name = "|edge delta|"
  ) +
  geom_node_point(
    aes(
      colour = age_weighted,
      size = delta_size
    ),
    alpha = 0.9
  ) +
  scale_colour_gradientn(
    colours = age_pal_orange,
    name = "Weighted age",
    na.value = "grey85"
  ) +
  scale_size_continuous(
    range = c(1.5, 7),
    name = "|delta_delta|"
  ) +
  theme_void() +
  theme(
    legend.position = "bottom",
    plot.margin = margin(10, 10, 10, 10)
  )


print(p_weighted)
dev.off()

cat("Wrote delta-sized trees to:\n", OUT_DIR, "\n")
