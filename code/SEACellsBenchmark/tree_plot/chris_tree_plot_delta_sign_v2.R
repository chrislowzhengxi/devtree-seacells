#############################################
## chris_tree_plot_delta_sign_v2.R
## V2: delta is ONLY on EDGES
##   positive = red, negative = blue, near 0 = grey
## Nodes are neutral (no sh coloring)
## Does NOT overwrite your previous PDFs
#############################################

library(ggraph)
library(tidygraph)
library(igraph)
library(scales)
library(RColorBrewer)
library(tidyr)

OUTPUT_CHRIS_TREE <- "/project/imoskowitz/xyang2/chrislowzhengxi/results/ucell/tree"

# 1) Load graph
g <- readRDS(file.path(OUTPUT_CHRIS_TREE, "devtree_graph.rds"))

# 1a) Attach new_id from nodes_filtered.txt (only used for labeled plots)
nodes_filt <- read.delim(
  file.path(OUTPUT_CHRIS_TREE, "nodes_filtered.txt"),
  sep = "\t",
  stringsAsFactors = FALSE
)
new_id_map <- setNames(nodes_filt$new_id, nodes_filt$meta_group)
V(g)$new_id <- new_id_map[V(g)$meta_group]

# ---------------------------
# Edge aesthetics
# ---------------------------
# Width based on magnitude
E(g)$ew <- rescale(replace_na(E(g)$abs_delta, 0), to = c(0.2, 2.0))

# Color based on SIGNED delta
# Transform makes most edges grey, only strong deltas saturated
d <- replace_na(E(g)$delta, 0)
E(g)$dc <- sign(d) * (abs(d)^(1/4))   # more grey than sqrt()

# ---------------------------
# Node aesthetics (neutral)
# ---------------------------
V(g)$node_col <- "grey20"

############################################################
##  A. TREE layout (labels), edge color = signed delta
############################################################

pdf(file.path(OUTPUT_CHRIS_TREE, "devtree_tree_deltaSign_edgesOnly_v2.pdf"),
    width = 24, height = 12)

p_tree_lbl <- ggraph(g, layout = "tree") +
  geom_edge_link(aes(edge_width = ew, edge_colour = dc),
                 arrow = arrow(length = unit(2, "mm")),
                 end_cap = circle(2, "mm"),
                 show.legend = TRUE) +
  scale_edge_colour_gradient2(
    low = "blue",
    mid = "grey80",
    high = "red",
    midpoint = 0,
    name = "delta (signed)",
    na.value = "grey80"
  ) +
  scale_edge_width(
    range = c(0.2, 2.0),
    guide = guide_legend(title = "abs_delta")
  ) +
  geom_node_point(color = "grey20", size = 1.6) +
  geom_node_text(aes(label = new_id), size = 1.5,
                 vjust = -0.6, check_overlap = TRUE) +
  theme_void() +
  theme(
    legend.position = "bottom",
    plot.margin = margin(10, 10, 10, 10)
  )

print(p_tree_lbl)
dev.off()

############################################################
##  B. TREE layout (unlabeled), edge color = signed delta
############################################################

pdf(file.path(OUTPUT_CHRIS_TREE, "devtree_tree_deltaSign_edgesOnly_unlabeled_v2.pdf"),
    width = 24, height = 12)

p_tree_unlbl <- ggraph(g, layout = "tree") +
  geom_edge_link(aes(edge_width = ew, edge_colour = dc),
                 arrow = arrow(length = unit(2, "mm")),
                 end_cap = circle(2, "mm"),
                 show.legend = TRUE) +
  scale_edge_colour_gradient2(
    low = "blue",
    mid = "grey80",
    high = "red",
    midpoint = 0,
    name = "delta (signed)",
    na.value = "grey80"
  ) +
  scale_edge_width(
    range = c(0.2, 2.0),
    guide = guide_legend(title = "abs_delta")
  ) +
  geom_node_point(color = "grey20", size = 1.6) +
  theme_void() +
  theme(
    legend.position = "bottom",
    plot.margin = margin(10, 10, 10, 10)
  )

print(p_tree_unlbl)
dev.off()

############################################################
##  C. SUGIYAMA layout (labels), edge color = signed delta
############################################################

pdf(file.path(OUTPUT_CHRIS_TREE, "devtree_sugiyama_deltaSign_edgesOnly_v2.pdf"),
    width = 24, height = 12)

p_sug_lbl <- ggraph(g, layout = "sugiyama") +
  geom_edge_link(aes(edge_width = ew, edge_colour = dc),
                 arrow = arrow(length = unit(2, "mm")),
                 end_cap = circle(2, "mm"),
                 show.legend = TRUE) +
  scale_edge_colour_gradient2(
    low = "blue",
    mid = "grey80",
    high = "red",
    midpoint = 0,
    name = "delta (signed)",
    na.value = "grey80"
  ) +
  scale_edge_width(
    range = c(0.2, 2.0),
    guide = guide_legend(title = "abs_delta")
  ) +
  geom_node_point(color = "grey20", size = 1.6) +
  geom_node_text(aes(label = new_id), size = 1.5,
                 vjust = -0.6, check_overlap = TRUE) +
  theme_void() +
  theme(
    legend.position = "bottom",
    plot.margin = margin(10, 10, 10, 10)
  )

print(p_sug_lbl)
dev.off()

############################################################
##  D. SUGIYAMA layout (unlabeled), edge color = signed delta
############################################################

pdf(file.path(OUTPUT_CHRIS_TREE, "devtree_sugiyama_deltaSign_edgesOnly_unlabeled_v2.pdf"),
    width = 24, height = 12)

p_sug_unlbl <- ggraph(g, layout = "sugiyama") +
  geom_edge_link(aes(edge_width = ew, edge_colour = dc),
                 arrow = arrow(length = unit(2, "mm")),
                 end_cap = circle(2, "mm"),
                 show.legend = TRUE) +
  scale_edge_colour_gradient2(
    low = "blue",
    mid = "grey80",
    high = "red",
    midpoint = 0,
    name = "delta (signed)",
    na.value = "grey80"
  ) +
  scale_edge_width(
    range = c(0.2, 2.0),
    guide = guide_legend(title = "abs_delta")
  ) +
  geom_node_point(color = "grey20", size = 1.6) +
  theme_void() +
  theme(
    legend.position = "bottom",
    plot.margin = margin(10, 10, 10, 10)
  )

print(p_sug_unlbl)
dev.off()
