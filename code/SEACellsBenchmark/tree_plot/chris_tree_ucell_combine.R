# === Inputs ===
library(dplyr)
library(readr)
library(ggraph)
library(igraph)
library(tidygraph)
library(viridis)
library(scales)
library(RColorBrewer)

EDGES_CSV         <- "/project/imoskowitz/xyang2/chrislowzhengxi/results/ucell/full_scored_edges_with_pregastrulation.csv"
OUTPUT_CHRIS_TREE <- "/project/imoskowitz/xyang2/chrislowzhengxi/results/ucell/tree"
dir.create(OUTPUT_CHRIS_TREE, recursive = TRUE, showWarnings = FALSE)

# === 1) Node Hh score from CSV (weighted by n) ===
edges_scored <- read_csv(EDGES_CSV, show_col_types = FALSE)

node_scores <- bind_rows(
  edges_scored %>% transmute(meta_group = x, celltype_new = x_name, n = n_x, sh = sh_x),
  edges_scored %>% transmute(meta_group = y, celltype_new = y_name, n = n_y, sh = sh_y)
) %>%
  filter(!is.na(meta_group)) %>%
  group_by(meta_group) %>%
  summarise(
    n_tot = sum(replace_na(n, 0)),
    sh_w  = ifelse(n_tot > 0,
                   weighted.mean(replace_na(sh, 0), w = replace_na(n, 0)),
                   mean(replace_na(sh, 0))),
    .groups = "drop"
  )

# attach to tbs_final nodes by meta_group
nodes_full <- tbs_final$nodes %>%
  left_join(node_scores, by = "meta_group") %>%
  mutate(node_HH = replace_na(sh_w, 0))

# === 2) Edge deltas on the merged edges ===
edges_full <- tbs_final$edges

# If abs_delta and delta are missing in tbs_final$edges, merge from CSV by names
if (!all(c("abs_delta","delta") %in% names(edges_full))) {
  edges_full <- edges_full %>%
    left_join(
      edges_scored %>%
        select(x_name, y_name, abs_delta, delta),
      by = c("x_name","y_name")
    )
}
edges_full <- edges_full %>%
  mutate(
    abs_delta = replace_na(abs_delta, 0),
    delta     = replace_na(delta, 0),
    edge_w    = rescale(abs_delta, to = c(0.2, 2.0)),
    edge_c    = abs(delta)
  )

# === 3) Build the full graph and plot with overlays ===
g_full <- graph_from_data_frame(
  d = edges_full %>% transmute(from = x_id, to = y_id, edge_w, edge_c),
  vertices = nodes_full %>% transmute(id = new_id, meta_group, celltype_new, system, node_HH),
  directed = TRUE
) %>% as_tbl_graph()

# Color palette for systems (used only for legend if you want a system view later)
n_colors <- length(unique(nodes_full$system))
sys_pal  <- colorRampPalette(brewer.pal(8, "Set1"))(n_colors)

# === Full tree with Hh overlays (labels on) ===
pdf(file.path(OUTPUT_CHRIS_TREE, "full_tree_HH_overlay.pdf"), width = 24, height = 12)
print(
  ggraph(g_full, layout = "tree") +
    geom_edge_link(aes(edge_width = edge_w, edge_colour = edge_c),
                   arrow = arrow(length = unit(2, "mm")),
                   end_cap = circle(2, "mm")) +
    scale_edge_colour_viridis(name = "|delta|", option = "C", direction = 1) +
    scale_edge_width(range = c(0.2, 2.0), guide = guide_legend(title = "abs_delta")) +
    geom_node_point(aes(colour = node_HH), size = 2) +
    scale_colour_viridis(name = "Node Hh (sh)", option = "B", direction = 1) +
    geom_node_text(aes(label = celltype_new), size = 1.5, vjust = -0.6, check_overlap = TRUE) +
    theme_void() + theme(legend.position = "bottom", plot.margin = margin(10,10,10,10))
)
dev.off()

# === Full tree with Hh overlays (unlabeled) ===
pdf(file.path(OUTPUT_CHRIS_TREE, "full_tree_HH_overlay_unlabeled.pdf"), width = 24, height = 12)
print(
  ggraph(g_full, layout = "tree") +
    geom_edge_link(aes(edge_width = edge_w, edge_colour = edge_c),
                   arrow = arrow(length = unit(2, "mm")),
                   end_cap = circle(2, "mm")) +
    scale_edge_colour_viridis(name = "|delta|", option = "C", direction = 1) +
    scale_edge_width(range = c(0.2, 2.0), guide = guide_legend(title = "abs_delta")) +
    geom_node_point(aes(colour = node_HH), size = 2) +
    scale_colour_viridis(name = "Node Hh (sh)", option = "B", direction = 1) +
    theme_void() + theme(legend.position = "bottom", plot.margin = margin(10,10,10,10))
)
dev.off()
