# Full structural tree like Holy's, system colored, no SHH overlay
library(ggraph)
library(RColorBrewer)
OUTPUT_CHRIS_TREE <- "/project/imoskowitz/xyang2/chrislowzhengxi/results/ucell/tree"

# Build graph from the merged, de-duplicated objects
g_full <- graph_from_data_frame(
  d = tbs_final$edges[, c("x_id","y_id")],
  vertices = tbs_final$nodes[, c("new_id","meta_group","celltype_new","system")],
  directed = TRUE
) %>% as_tbl_graph()

# Colors per system
n_colors <- length(unique(tbs_final$nodes$system))
palette  <- colorRampPalette(brewer.pal(8, "Set1"))(n_colors)

# PDF 1. Tree layout with labels
pdf(file.path(OUTPUT_CHRIS_TREE, "full_tree_skeleton.pdf"), width = 20, height = 10)
print(
  ggraph(g_full, layout = "tree") +
    geom_edge_link(arrow = arrow(length = unit(2, "mm")), end_cap = circle(2, "mm"), colour = "grey60", size = 0.3) +
    geom_node_point(aes(color = as.factor(system)), size = 1.8) +
    geom_node_text(aes(label = celltype_new), size = 1.6, vjust = -0.6, check_overlap = TRUE) +
    scale_color_manual(values = palette, name = "System") +
    theme_void() + theme(legend.position = "bottom", plot.margin = margin(10,10,10,10))
)
dev.off()

# PDF 2. Same, unlabeled
pdf(file.path(OUTPUT_CHRIS_TREE, "full_tree_skeleton_unlabeled.pdf"), width = 20, height = 10)
print(
  ggraph(g_full, layout = "tree") +
    geom_edge_link(arrow = arrow(length = unit(2, "mm")), end_cap = circle(2, "mm"), colour = "grey60", size = 0.3) +
    geom_node_point(aes(color = as.factor(system)), size = 2) +
    scale_color_manual(values = palette, name = "System") +
    theme_void() + theme(legend.position = "bottom", plot.margin = margin(10,10,10,10))
)
dev.off()
