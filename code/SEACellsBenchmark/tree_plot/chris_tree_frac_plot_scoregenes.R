#############################################
## chris_tree_frac_plot_scoregenes.R
## Global tree for Scanpy score_genes with:
##   - node color = % SHH_scoregenes > 0  (frac_w)
##   - edge width = abs_delta
##   - edge color = |delta|
## Backbone topology comes from devtree_graph.rds
#############################################

library(dplyr)
library(readr)
library(tidyr)
library(igraph)
library(tidygraph)
library(ggraph)
library(colorspace)
library(scales)

tree_dir <- "/project/imoskowitz/xyang2/chrislowzhengxi/results/score_genes/tree"

# ---------- 1) Load inputs ----------

# Edge table that already contains frac>0_x, frac>0_y, n_x, n_y
edges_frac_path <- "/project/imoskowitz/xyang2/chrislowzhengxi/results/score_genes/full_scored_edges_with_pregastrulation_scoregenes.csv"

# Existing backbone tree graph (has meta_group on vertices, abs_delta/delta on edges)
g <- readRDS(file.path(tree_dir, "devtree_graph.rds"))

edges_frac <- read_csv(edges_frac_path, show_col_types = FALSE)

# Optional labels file
nodes_filt_path <- file.path(tree_dir, "nodes_filtered.txt")
nodes_filt <- NULL
if (file.exists(nodes_filt_path)) {
  nodes_filt <- read.delim(nodes_filt_path, sep = "\t", stringsAsFactors = FALSE)
}

# ---------- 2) Compute node-level % scoregenes > 0 ----------

frac_long <- bind_rows(
  edges_frac %>%
    transmute(meta_group = x, frac = `frac>0_x`, n = n_x),
  edges_frac %>%
    transmute(meta_group = y, frac = `frac>0_y`, n = n_y)
) %>%
  filter(!is.na(meta_group), !is.na(frac)) %>%
  mutate(
    frac = as.numeric(frac),
    n    = as.numeric(n)
  )

node_frac <- frac_long %>%
  group_by(meta_group) %>%
  summarise(
    n_tot  = sum(replace_na(n, 0)),
    frac_w = ifelse(
      n_tot > 0,
      weighted.mean(replace_na(frac, 0), w = replace_na(n, 0)),
      mean(replace_na(frac, 0))
    ),
    .groups = "drop"
  )

# ---------- 3) Attach frac_w + labels to the existing graph ----------

# Build a vertex table keyed by meta_group from the graph
vtab <- data.frame(
  meta_group = as.character(V(g)$meta_group),
  stringsAsFactors = FALSE
)

# Join frac_w
vtab <- vtab %>%
  left_join(node_frac, by = "meta_group") %>%
  mutate(frac_w = replace_na(frac_w, 0))

# Join new_id labels if available
if (!is.null(nodes_filt) && all(c("meta_group","new_id") %in% names(nodes_filt))) {
  vtab <- vtab %>%
    left_join(nodes_filt %>% select(meta_group, new_id), by = "meta_group")
} else {
  vtab$new_id <- NA_character_
}

# Fallback label if new_id missing
vtab <- vtab %>%
  mutate(label_id = ifelse(is.na(new_id) | new_id == "", meta_group, new_id))

# Write attributes back to graph
V(g)$frac_w   <- vtab$frac_w
V(g)$label_id <- vtab$label_id

# ---------- 4) Edge aesthetics (from existing edge attrs) ----------

# These should already exist on the backbone graph
stopifnot(all(c("abs_delta", "delta") %in% igraph::edge_attr_names(g)))

E(g)$ew <- scales::rescale(replace_na(E(g)$abs_delta, 0), to = c(0.2, 2.0))
E(g)$ec <- abs(replace_na(E(g)$delta, 0))

# Save graph with frac info
saveRDS(g, file.path(tree_dir, "devtree_graph_frac_scoregenes.rds"))

# Convert to tbl_graph for ggraph
g_frac <- as_tbl_graph(g)

# ---------- 5) Palettes ----------

pal_mint_r   <- colorspace::sequential_hcl(256, palette = "Mint",   rev = TRUE)  # nodes
pal_rocket_r <- colorspace::sequential_hcl(256, palette = "Rocket", rev = TRUE)  # edges

# ---------- 6) Plot: TREE layout ----------

pdf(file.path(tree_dir, "devtree_tree_frac_overlay_scoregenes.pdf"),
    width = 24, height = 12)

p_tree_frac <- ggraph(g_frac, layout = "tree") +
  geom_edge_link(
    aes(edge_width = ew, edge_colour = ec),
    arrow = arrow(length = unit(2, "mm")),
    end_cap = circle(2, "mm"),
    show.legend = TRUE
  ) +
  scale_edge_colour_gradientn(
    colours = pal_rocket_r,
    name = "|delta|",
    na.value = "grey80"
  ) +
  scale_edge_width(
    range = c(0.2, 2.0),
    guide = guide_legend(title = "abs_delta")
  ) +
  geom_node_point(aes(colour = frac_w), size = 2) +
  scale_colour_gradientn(
    colours = pal_mint_r,
    name = "% scoregenes > 0"
  ) +
  geom_node_text(aes(label = label_id), size = 1.5,
                 vjust = -0.6, check_overlap = TRUE) +
  theme_void() +
  theme(
    legend.position = "bottom",
    plot.margin = margin(10, 10, 10, 10)
  )

print(p_tree_frac)
dev.off()

# ---------- 7) Plot: SUGIYAMA layout ----------

pdf(file.path(tree_dir, "devtree_sugiyama_frac_overlay_scoregenes.pdf"),
    width = 24, height = 12)

p_sug_frac <- ggraph(g_frac, layout = "sugiyama") +
  geom_edge_link(
    aes(edge_width = ew, edge_colour = ec),
    arrow = arrow(length = unit(2, "mm")),
    end_cap = circle(2, "mm"),
    show.legend = TRUE
  ) +
  scale_edge_colour_gradientn(
    colours = pal_rocket_r,
    name = "|delta|",
    na.value = "grey80"
  ) +
  scale_edge_width(
    range = c(0.2, 2.0),
    guide = guide_legend(title = "abs_delta")
  ) +
  geom_node_point(aes(colour = frac_w), size = 2, alpha = 1) +
  scale_colour_gradientn(
    colours = pal_mint_r,
    name = "% scoregenes > 0"
  ) +
  geom_node_text(aes(label = label_id), size = 1.5,
                 vjust = -0.6, check_overlap = TRUE) +
  theme_void() +
  theme(
    legend.position = "bottom",
    plot.margin = margin(10, 10, 10, 10)
  )

print(p_sug_frac)
dev.off()

cat("Finished:\n")
cat("  devtree_tree_frac_overlay_scoregenes.pdf\n")
cat("  devtree_sugiyama_frac_overlay_scoregenes.pdf\n")
cat("Saved:\n")
cat("  devtree_graph_frac_scoregenes.rds\n")
