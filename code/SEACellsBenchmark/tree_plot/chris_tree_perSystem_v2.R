#############################################
## chris_tree_perSystem_v2.R
## Per-system structural trees from devtree_graph.rds
#############################################

library(ggraph)
library(tidygraph)
library(igraph)
library(RColorBrewer)
library(gridExtra)
library(grid)

OUTPUT_CHRIS_TREE <- "/project/imoskowitz/xyang2/chrislowzhengxi/results/ucell/tree"

# Load the global graph
g <- readRDS(file.path(OUTPUT_CHRIS_TREE, "devtree_graph.rds"))

# Systems present on nodes
sys_vec   <- as.factor(V(g)$system)
all_sys   <- levels(sys_vec)

# Exclude pre-gastrulation and gastrulation from the per-system loop
sub_sys <- setdiff(all_sys, c("Pre_gastrulation", "Gastrulation"))

# Color palette for systems
n_colors <- length(all_sys)
palette  <- colorRampPalette(brewer.pal(8, "Set1"))(n_colors)
names(palette) <- all_sys

pdf(file.path(OUTPUT_CHRIS_TREE, "devtree_perSystem_v2.pdf"),
    width = 8, height = 6)

for (i in sub_sys) {

  # Subgraph that keeps Pre_gastrulation, Gastrulation, and system i
  g_sub <- g %>%
    activate(nodes) %>%
    filter(system %in% c("Pre_gastrulation", "Gastrulation", i)) %>%
    as_tbl_graph()

  p <- ggraph(g_sub, layout = "tree") +
    geom_edge_link(
      arrow   = arrow(length = unit(2, "mm")),
      end_cap = circle(3, "mm")
    ) +
    geom_node_point(aes(color = as.factor(system)), size = 4) +
    geom_node_text(aes(label = name), vjust = 0.5, hjust = 0.5, size = 2) +
    scale_color_manual(values = palette, name = "System") +
    theme_void() +
    theme(legend.position = "top") +
    labs(
      title = paste("Per-system tree:", i),
      color = "System"
    )

  # Build the id_celltype label block for Gastrulation + system i
  g_sub_sys <- g_sub %>%
    activate(nodes) %>%
    filter(system %in% c("Gastrulation", i)) %>%
    as_tbl_graph()

  combined_labels <- paste(
        paste(V(g_sub_sys)$name, V(g_sub_sys)$celltype_new, sep = "_"),
        collapse = "\n"
    )

  text_grob <- textGrob(
    combined_labels,
    x  = 0,
    just = "left",
    gp = gpar(fontsize = 5)
  )

  # Arrange plot and labels side by side
  grid.arrange(p, text_grob, ncol = 2, widths = c(3, 1))

  Sys.sleep(0.5)
}

dev.off()
