# --- Setup ---
library(dplyr)
library(tidyr)
library(readr)
library(igraph)
library(tidygraph)
library(ggraph)
library(viridis)
library(scales)

INPUT_QIU_OTHER   <- "/project/imoskowitz/xyang2/SHH/Qiu_TimeLapse/other"
EDGES_CSV        <- "/project/imoskowitz/xyang2/chrislowzhengxi/results/ucell/full_scored_edges_with_pregastrulation.csv"
OUTPUT_CHRIS_TREE <- "/project/imoskowitz/xyang2/chrislowzhengxi/results/ucell/tree"
dir.create(OUTPUT_CHRIS_TREE, recursive = TRUE, showWarnings = FALSE)

# --- Load inputs ---
nodes_raw <- read.table(file.path(INPUT_QIU_OTHER, "nodes.txt"),
                        header = TRUE, sep = "\t", as.is = TRUE)
edges_raw <- read_csv(EDGES_CSV, show_col_types = FALSE)

# Quick sanity: required columns
stopifnot(all(c("x","y","x_name","y_name","edge_type") %in% names(edges_raw)))
stopifnot(all(c("meta_group","celltype_new","system") %in% names(nodes_raw)))

# --- Build per-node Hh scores from both x_* and y_* sides ---
# We will use "sh" as the node score. Weighted by n_*.
# You can switch to median or q90 by changing the column names below.

to_long <- function(df) {
  bind_rows(
    df %>% transmute(meta_group = x,
                     celltype_new = x_name,
                     system       = .data$system,
                     n            = n_x,
                     sh           = sh_x),
    df %>% transmute(meta_group = y,
                     celltype_new = y_name,
                     system       = .data$system,
                     n            = n_y,
                     sh           = sh_y)
  )
}

node_scores_long <- to_long(edges_raw) %>%
  filter(!is.na(meta_group), !is.na(celltype_new))

# Weighted mean of sh by n, per node label
node_scores <- node_scores_long %>%
  group_by(meta_group, celltype_new) %>%
  summarise(n_tot = sum(replace_na(n, 0)),
            sh_w  = ifelse(sum(replace_na(n,0)) > 0,
                           weighted.mean(replace_na(sh, 0), w = replace_na(n,0)),
                           mean(replace_na(sh,0))),
            .groups = "drop")

# Attach system from nodes table and keep a unique row per meta_group
nodes <- nodes_raw %>%
  select(system, meta_group, celltype_new) %>%
  left_join(node_scores %>% select(meta_group, sh_w), by = "meta_group") %>%
  distinct(meta_group, .keep_all = TRUE)

# --- Choose edges: drop Spatial continuity by default ---
edges_filt <- edges_raw %>%
  filter(edge_type != "Spatial continuity") %>%
  select(system, x, y, x_name, y_name, edge_type,
         abs_delta, delta)

# Map node ids
nodes$id <- match(nodes$meta_group, nodes$meta_group)  # 1..n
id_map <- setNames(nodes$id, nodes$meta_group)
edges_filt <- edges_filt %>%
  mutate(x_id = id_map[x],
         y_id = id_map[y]) %>%
  filter(!is.na(x_id), !is.na(y_id))

# --- Build graph ---
g <- graph_from_data_frame(
  d = edges_filt %>% transmute(from = x_id, to = y_id, edge_type, abs_delta, delta),
  vertices = nodes %>% transmute(id, meta_group, celltype_new, system, sh = sh_w),
  directed = TRUE
) %>% as_tbl_graph()

# --- Aesthetics helpers ---
# Edge width by abs_delta (rescale to a readable range)
E(g)$ew <- rescale(replace_na(E(g)$abs_delta, 0), to = c(0.2, 2))

# Edge color by |delta|
E(g)$ec <- abs(replace_na(E(g)$delta, 0))

# Node color by sh (Hh node score)
V(g)$nv <- replace_na(V(g)$sh, 0)

# --- Plot 1: labeled by celltype_new, sugiyama layout ---
pdf(file.path(OUTPUT_CHRIS_TREE, "devtree_sugiyama_HH_overlay.pdf"), width = 20, height = 10)
p <- ggraph(g, layout = "sugiyama") +
  geom_edge_link(aes(edge_width = ew, edge_colour = ec),
                 arrow = arrow(length = unit(2, "mm")),
                 end_cap = circle(2, "mm"),
                 show.legend = TRUE) +
  scale_edge_colour_viridis(name = "|delta|", option = "C", direction = 1) +
  scale_edge_width(range = c(0.2, 2), guide = guide_legend(title = "abs_delta")) +
  geom_node_point(aes(colour = nv), size = 2) +
  scale_colour_viridis(name = "Node Hh (sh)", option = "B", direction = 1) +
  geom_node_text(aes(label = celltype_new), size = 1.6, vjust = -0.6, check_overlap = TRUE) +
  theme_void() +
  theme(legend.position = "bottom", plot.margin = margin(10, 10, 10, 10))
print(p)
dev.off()

# --- Plot 2: unlabeled (cleaner) ---
pdf(file.path(OUTPUT_CHRIS_TREE, "devtree_sugiyama_HH_overlay_unlabeled.pdf"), width = 20, height = 10)
p2 <- ggraph(g, layout = "sugiyama") +
  geom_edge_link(aes(edge_width = ew, edge_colour = ec),
                 arrow = arrow(length = unit(2, "mm")),
                 end_cap = circle(2, "mm")) +
  scale_edge_colour_viridis(name = "|delta|", option = "C", direction = 1) +
  scale_edge_width(range = c(0.2, 2), guide = guide_legend(title = "abs_delta")) +
  geom_node_point(aes(colour = nv), size = 2) +
  scale_colour_viridis(name = "Node Hh (sh)", option = "B", direction = 1) +
  theme_void() +
  theme(legend.position = "bottom", plot.margin = margin(10, 10, 10, 10))
print(p2)
dev.off()

# --- Optional: per-system facets keeping Pre_gastrulation + Gastrulation as backbone ---
pdf(file.path(OUTPUT_CHRIS_TREE, "devtree_perSystem_HH_overlay.pdf"), width = 20, height = 10)
sys_levels <- sort(unique(V(g)$system))
for (s in sys_levels) {
  g_sub <- g %>%
    activate(nodes) %>%
    filter(system %in% c("Pre_gastrulation", "Gastrulation", s)) %>%
    as_tbl_graph()
  if (gorder(g_sub) == 0) next
  p3 <- ggraph(g_sub, layout = "sugiyama") +
    geom_edge_link(aes(edge_width = abs(replace_na(abs_delta,0)),
                       edge_colour = abs(replace_na(delta,0))),
                   arrow = arrow(length = unit(2, "mm")), end_cap = circle(2, "mm")) +
    scale_edge_colour_viridis(name = "|delta|", option = "C", direction = 1) +
    scale_edge_width(range = c(0.2, 2), guide = guide_legend(title = "abs_delta")) +
    geom_node_point(aes(colour = replace_na(sh,0)), size = 2) +
    scale_colour_viridis(name = "Node Hh (sh)", option = "B", direction = 1) +
    geom_node_text(aes(label = celltype_new), size = 1.6, vjust = -0.6, check_overlap = TRUE) +
    ggtitle(paste("System:", s)) +
    theme_void() +
    theme(legend.position = "bottom", plot.margin = margin(10, 10, 10, 10))
  print(p3)
}
dev.off()
