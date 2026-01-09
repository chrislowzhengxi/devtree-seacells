#############################################
## chris_tree_plot_ucell_pct_v1.R
## Plot global HH tree
## Nodes = UCell % (>0)
## Edges = signed delta (unchanged topology)
#############################################

library(dplyr)
library(tidyr)
library(readr)
library(ggraph)
library(tidygraph)
library(igraph)
library(colorspace)
library(scales)
library(RColorBrewer)

# ------------------------------------------------------------
# Paths
# ------------------------------------------------------------

OUTPUT_CHRIS_TREE <- "/project/imoskowitz/xyang2/chrislowzhengxi/results/ucell/tree"

EDGES_PCT_CSV <- "/project/imoskowitz/xyang2/chrislowzhengxi/results/ucell/tree/full_scored_edges_with_pregastrulation.csv"


# ------------------------------------------------------------
# 1) Load canonical tree (DO NOT CHANGE)
# ------------------------------------------------------------

g <- readRDS(file.path(OUTPUT_CHRIS_TREE, "devtree_graph.rds"))

# ------------------------------------------------------------
# 2) Load node labels (for labeled plots)
# ------------------------------------------------------------

nodes_filt <- read.delim(
  file.path(OUTPUT_CHRIS_TREE, "nodes_filtered.txt"),
  sep = "\t",
  stringsAsFactors = FALSE
)

new_id_map <- setNames(nodes_filt$new_id, nodes_filt$meta_group)
V(g)$new_id <- new_id_map[V(g)$meta_group]

# ------------------------------------------------------------
# 3) Load UCell % edge summaries
# ------------------------------------------------------------

edges_pct <- read_csv(EDGES_PCT_CSV, show_col_types = FALSE)

# Helper: pick frac>0 if present, otherwise sh
get_ucell_pct <- function(frac, sh) {
  ifelse(!is.na(frac), frac, sh)
}

# Long table: one row per node appearance
node_ucell_long <- bind_rows(
  edges_pct %>%
    transmute(
      meta_group = x,
      ucell_pct  = get_ucell_pct(`frac>0_x`, sh_x),
      n          = if ("n_x" %in% names(edges_pct)) n_x else 1
    ),
  edges_pct %>%
    transmute(
      meta_group = y,
      ucell_pct  = get_ucell_pct(`frac>0_y`, sh_y),
      n          = if ("n_y" %in% names(edges_pct)) n_y else 1
    )
) %>%
  filter(!is.na(meta_group), !is.na(ucell_pct))

# ------------------------------------------------------------
# 4) Aggregate to node-level UCell %
# ------------------------------------------------------------

node_ucell <- node_ucell_long %>%
  group_by(meta_group) %>%
  summarise(
    ucell_pct = weighted.mean(ucell_pct, w = n, na.rm = TRUE),
    .groups = "drop"
  )

# ------------------------------------------------------------
# 5) Attach to graph nodes
# ------------------------------------------------------------

ucell_map <- setNames(node_ucell$ucell_pct, node_ucell$meta_group)
V(g)$ucell_pct <- ucell_map[V(g)$meta_group]
V(g)$ucell_pct <- replace_na(V(g)$ucell_pct, 0)

# ------------------------------------------------------------
# 6) Edge aesthetics (UNCHANGED)
# ------------------------------------------------------------

E(g)$ew <- rescale(replace_na(E(g)$abs_delta, 0), to = c(0.2, 2.0))
E(g)$dc <- replace_na(E(g)$delta, 0)

# ------------------------------------------------------------
# 7) Node aesthetics (UCell %)
# ------------------------------------------------------------

pal_ucell <- colorspace::sequential_hcl(
  256, palette = "YlOrRd", rev = FALSE
)

V(g)$nv <- V(g)$ucell_pct

# ------------------------------------------------------------
# 8) TREE layout (labeled)
# ------------------------------------------------------------

pdf(file.path(OUTPUT_CHRIS_TREE,
              "devtree_tree_ucell_pct_v1.pdf"),
    width = 24, height = 12)

p_tree <- ggraph(g, layout = "tree") +
  geom_edge_link(aes(edge_width = ew, edge_colour = dc),
                 arrow = arrow(length = unit(2, "mm")),
                 end_cap = circle(2, "mm"),
                 show.legend = TRUE) +
  scale_edge_colour_gradient2(
    low = "blue",
    mid = "grey80",
    high = "red",
    midpoint = 0,
    name = "delta (signed)"
  ) +
  scale_edge_width(
    range = c(0.2, 2.0),
    guide = guide_legend(title = "abs_delta")
  ) +
  geom_node_point(aes(colour = nv), size = 2.2) +
  scale_colour_gradientn(
    colours = pal_ucell,
    limits = c(0, 1),
    name = "UCell % (>0)"
  ) +
  geom_node_text(aes(label = new_id),
                 size = 1.5,
                 vjust = -0.6,
                 check_overlap = TRUE) +
  theme_void() +
  theme(
    legend.position = "bottom",
    plot.margin = margin(10, 10, 10, 10)
  )

print(p_tree)
dev.off()

# ------------------------------------------------------------
# 9) SUGIYAMA layout (unlabeled)
# ------------------------------------------------------------

pdf(file.path(OUTPUT_CHRIS_TREE,
              "devtree_sugiyama_ucell_pct_unlabeled_v1.pdf"),
    width = 24, height = 12)

p_sug <- ggraph(g, layout = "sugiyama") +
  geom_edge_link(aes(edge_width = ew, edge_colour = dc),
                 arrow = arrow(length = unit(2, "mm")),
                 end_cap = circle(2, "mm")) +
  scale_edge_colour_gradient2(
    low = "blue",
    mid = "grey80",
    high = "red",
    midpoint = 0,
    name = "delta (signed)"
  ) +
  scale_edge_width(range = c(0.2, 2.0)) +
  geom_node_point(aes(colour = nv), size = 2.0) +
  scale_colour_gradientn(
    colours = pal_ucell,
    limits = c(0, 1),
    name = "UCell % (>0)"
  ) +
  theme_void() +
  theme(
    legend.position = "bottom",
    plot.margin = margin(10, 10, 10, 10)
  )

print(p_sug)
dev.off()

cat("UCell % tree plots generated.\n")
