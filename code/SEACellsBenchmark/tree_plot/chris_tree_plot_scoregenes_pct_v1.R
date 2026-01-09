#############################################
## chris_tree_plot_scoregenes_pct_v1.R
## Plot Scanpy score_genes % (>0) on dev tree
## Nodes = fraction of cells with score_genes > 0
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

OUTPUT_CHRIS_TREE <- "/project/imoskowitz/xyang2/chrislowzhengxi/results/score_genes_new/tree"

EDGES_PCT_CSV <- "/project/imoskowitz/xyang2/chrislowzhengxi/results/score_genes_new/full_scored_edges_with_pregastrulation_scoregenes.csv"

# ------------------------------------------------------------
# 1) Load canonical Scanpy tree (DO NOT REBUILD)
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
mapped <- unname(new_id_map[as.character(V(g)$meta_group)])
mapped[is.na(mapped)] <- as.character(V(g)$meta_group)[is.na(mapped)]
V(g)$new_id <- mapped

# ------------------------------------------------------------
# 3) Load Scanpy score_genes edge summaries
# ------------------------------------------------------------

edges_pct <- read_csv(EDGES_PCT_CSV, show_col_types = FALSE)

stopifnot(all(c(
  "x","y",
  "frac>0_x","frac>0_y",
  "n_x","n_y"
) %in% names(edges_pct)))

# ------------------------------------------------------------
# 4) Build long table of node-level Scanpy %
# ------------------------------------------------------------

node_pct_long <- bind_rows(
  edges_pct %>%
    transmute(
      meta_group = x,
      pct        = `frac>0_x`,
      n          = n_x
    ),
  edges_pct %>%
    transmute(
      meta_group = y,
      pct        = `frac>0_y`,
      n          = n_y
    )
) %>%
  filter(!is.na(meta_group), !is.na(pct))

# ------------------------------------------------------------
# 5) Aggregate to node-level Scanpy % (>0)
# ------------------------------------------------------------

node_pct <- node_pct_long %>%
  group_by(meta_group) %>%
  summarise(
    scoregenes_pct = weighted.mean(pct, w = n, na.rm = TRUE),
    .groups = "drop"
  )

# ------------------------------------------------------------
# 6) Attach Scanpy % to graph nodes
# ------------------------------------------------------------

pct_map <- setNames(node_pct$scoregenes_pct, node_pct$meta_group)
V(g)$scoregenes_pct <- pct_map[V(g)$meta_group]
V(g)$scoregenes_pct <- replace_na(V(g)$scoregenes_pct, 0)

# ------------------------------------------------------------
# 7) Edge aesthetics (UNCHANGED)
# ------------------------------------------------------------

E(g)$ew <- rescale(replace_na(E(g)$abs_delta, 0), to = c(0.2, 2.0))
E(g)$dc <- replace_na(E(g)$delta, 0)

# ------------------------------------------------------------
# 8) Node aesthetics (Scanpy %)
# ------------------------------------------------------------

pal_pct <- colorspace::sequential_hcl(
  256, palette = "YlOrRd", rev = FALSE
)

V(g)$nv <- V(g)$scoregenes_pct

# ------------------------------------------------------------
# 9) TREE layout (labeled)
# ------------------------------------------------------------

pdf(file.path(OUTPUT_CHRIS_TREE,
              "devtree_tree_scoregenes_pct_v1.pdf"),
    width = 24, height = 12)

p_tree <- ggraph(g, layout = "tree") +
  geom_edge_link(
    aes(edge_width = ew, edge_colour = dc),
    arrow = arrow(length = unit(2, "mm")),
    end_cap = circle(2, "mm"),
    show.legend = TRUE
  ) +
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
    colours = pal_pct,
    limits  = c(0, 1),
    name    = "Scanpy % (>0)"
  ) +
  geom_node_text(
    aes(label = new_id),
    size = 1.5,
    vjust = -0.6,
    check_overlap = TRUE
  ) +
  theme_void() +
  theme(
    legend.position = "bottom",
    plot.margin = margin(10, 10, 10, 10)
  )

print(p_tree)
dev.off()

# ------------------------------------------------------------
# 10) SUGIYAMA layout (unlabeled)
# ------------------------------------------------------------

pdf(file.path(OUTPUT_CHRIS_TREE,
              "devtree_sugiyama_scoregenes_pct_unlabeled_v1.pdf"),
    width = 24, height = 12)

p_sug <- ggraph(g, layout = "sugiyama") +
  geom_edge_link(
    aes(edge_width = ew, edge_colour = dc),
    arrow = arrow(length = unit(2, "mm")),
    end_cap = circle(2, "mm")
  ) +
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
    colours = pal_pct,
    limits  = c(0, 1),
    name    = "Scanpy % (>0)"
  ) +
  theme_void() +
  theme(
    legend.position = "bottom",
    plot.margin = margin(10, 10, 10, 10)
  )

print(p_sug)
dev.off()

cat("Scanpy % (>0) tree plots generated.\n")
