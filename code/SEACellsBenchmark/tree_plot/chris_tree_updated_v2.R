#############################################
## chris_tree_plot_v2.R
## Plot global HH tree from devtree_graph.rds
#############################################

library(ggraph)
library(tidygraph)
library(igraph)
library(viridis)
library(scales)
library(RColorBrewer)

OUTPUT_CHRIS_TREE <- "/project/imoskowitz/xyang2/chrislowzhengxi/results/ucell/tree"

# 1) Load graph built by chris_tree_updated.R
g <- readRDS(file.path(OUTPUT_CHRIS_TREE, "devtree_graph.rds"))

# Edge aesthetics (reuse if already present, but recompute to be safe)
E(g)$ew <- rescale(replace_na(E(g)$abs_delta, 0), to = c(0.2, 2))
E(g)$ec <- abs(replace_na(E(g)$delta, 0))

# Node aesthetics
V(g)$nv <- replace_na(V(g)$sh, 0)

############################################################
##  A. TREE layout with HH overlay (labels)
############################################################

pdf(file.path(OUTPUT_CHRIS_TREE, "devtree_tree_HH_overlay_v2.pdf"),
    width = 24, height = 12)

p_tree_lbl <- ggraph(g, layout = "tree") +
  geom_edge_link(aes(edge_width = ew, edge_colour = ec),
                 arrow = arrow(length = unit(2, "mm")),
                 end_cap = circle(2, "mm"),
                 show.legend = TRUE) +
  scale_edge_colour_viridis(name = "|delta|", option = "C", direction = 1,
                            na.value = "grey80") +
  scale_edge_width(range = c(0.2, 2.0), guide = guide_legend(title = "abs_delta")) +
  geom_node_point(aes(colour = nv), size = 2) +
  scale_colour_viridis(name = "Node Hh (sh)", option = "B", direction = 1) +
  geom_node_text(aes(label = celltype_new), size = 1.5,
                 vjust = -0.6, check_overlap = TRUE) +
  theme_void() +
  theme(legend.position = "bottom",
        plot.margin = margin(10, 10, 10, 10))
print(p_tree_lbl)
dev.off()

############################################################
##  B. TREE layout with HH overlay (unlabeled)
############################################################

pdf(file.path(OUTPUT_CHRIS_TREE, "devtree_tree_HH_overlay_unlabeled_v2.pdf"),
    width = 24, height = 12)

p_tree_unlbl <- ggraph(g, layout = "tree") +
  geom_edge_link(aes(edge_width = ew, edge_colour = ec),
                 arrow = arrow(length = unit(2, "mm")),
                 end_cap = circle(2, "mm"),
                 show.legend = TRUE) +
  scale_edge_colour_viridis(name = "|delta|", option = "C", direction = 1,
                            na.value = "grey80") +
  scale_edge_width(range = c(0.2, 2.0), guide = guide_legend(title = "abs_delta")) +
  geom_node_point(aes(colour = nv), size = 2) +
  scale_colour_viridis(name = "Node Hh (sh)", option = "B", direction = 1) +
  theme_void() +
  theme(legend.position = "bottom",
        plot.margin = margin(10, 10, 10, 10))
print(p_tree_unlbl)
dev.off()

############################################################
##  C. SUGIYAMA layout with HH overlay (labels)
############################################################

pdf(file.path(OUTPUT_CHRIS_TREE, "devtree_sugiyama_HH_overlay_v2.pdf"),
    width = 24, height = 12)

p_sug_lbl <- ggraph(g, layout = "sugiyama") +
  geom_edge_link(aes(edge_width = ew, edge_colour = ec),
                 arrow = arrow(length = unit(2, "mm")),
                 end_cap = circle(2, "mm"),
                 show.legend = TRUE) +
  scale_edge_colour_viridis(name = "|delta|", option = "C", direction = 1,
                            na.value = "grey80") +
  scale_edge_width(range = c(0.2, 2.0), guide = guide_legend(title = "abs_delta")) +
  geom_node_point(aes(colour = nv), size = 2) +
  scale_colour_viridis(name = "Node Hh (sh)", option = "B", direction = 1) +
  geom_node_text(aes(label = celltype_new), size = 1.5,
                 vjust = -0.6, check_overlap = TRUE) +
  theme_void() +
  theme(legend.position = "bottom",
        plot.margin = margin(10, 10, 10, 10))
print(p_sug_lbl)
dev.off()

############################################################
##  D. SUGIYAMA layout with HH overlay (unlabeled)
############################################################

pdf(file.path(OUTPUT_CHRIS_TREE, "devtree_sugiyama_HH_overlay_unlabeled_v2.pdf"),
    width = 24, height = 12)

p_sug_unlbl <- ggraph(g, layout = "sugiyama") +
  geom_edge_link(aes(edge_width = ew, edge_colour = ec),
                 arrow = arrow(length = unit(2, "mm")),
                 end_cap = circle(2, "mm"),
                 show.legend = TRUE) +
  scale_edge_colour_viridis(name = "|delta|", option = "C", direction = 1,
                            na.value = "grey80") +
  scale_edge_width(range = c(0.2, 2.0), guide = guide_legend(title = "abs_delta")) +
  geom_node_point(aes(colour = nv), size = 2) +
  scale_colour_viridis(name = "Node Hh (sh)", option = "B", direction = 1) +
  theme_void() +
  theme(legend.position = "bottom",
        plot.margin = margin(10, 10, 10, 10))
print(p_sug_unlbl)
dev.off()

############################################################
##  E. System-colored structural skeleton (TREE layout)
##     (like old chris_tree_combine.R)
############################################################

sys_vec   <- as.factor(V(g)$system)
n_colors  <- length(levels(sys_vec))
sys_pal   <- colorRampPalette(brewer.pal(8, "Set1"))(n_colors)

pdf(file.path(OUTPUT_CHRIS_TREE, "devtree_system_skeleton_v2.pdf"),
    width = 20, height = 10)

p_sys <- ggraph(g, layout = "tree") +
  geom_edge_link(arrow = arrow(length = unit(2, "mm")),
                 end_cap = circle(2, "mm"),
                 colour = "grey60",
                 linewidth = 0.3) +
  geom_node_point(aes(color = sys_vec), size = 1.8) +
  geom_node_text(aes(label = celltype_new), size = 1.6,
                 vjust = -0.6, check_overlap = TRUE) +
  scale_color_manual(values = sys_pal, name = "System") +
  theme_void() +
  theme(legend.position = "bottom",
        plot.margin = margin(10, 10, 10, 10))
print(p_sys)
dev.off()
