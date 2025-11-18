#############################################
## chris_tree_updated.R
## Build global HH tree from edges_all_final
#############################################

# --- Setup ---
library(dplyr)
library(tidyr)
library(readr)
library(igraph)
library(tidygraph)
library(ggraph)
library(viridis)
library(scales)

INPUT_QIU_OTHER    <- "/project/imoskowitz/xyang2/SHH/Qiu_TimeLapse/other"
EDGES_CSV          <- "/project/imoskowitz/xyang2/chrislowzhengxi/results/ucell/edges_all_final.csv"
OUTPUT_CHRIS_TREE  <- "/project/imoskowitz/xyang2/chrislowzhengxi/results/ucell/tree"
dir.create(OUTPUT_CHRIS_TREE, recursive = TRUE, showWarnings = FALSE)

# --- Load inputs ---
nodes_raw <- read.table(file.path(INPUT_QIU_OTHER, "nodes.txt"),
                        header = TRUE, sep = "\t", as.is = TRUE)

edges_raw <- read.csv(EDGES_CSV, check.names = FALSE)

# Quick sanity: required columns
stopifnot(all(c("x","y","x_name","y_name","edge_type") %in% names(edges_raw)))
stopifnot(all(c("meta_group","celltype_new","system") %in% names(nodes_raw)))

# --- Build per node HH scores -------------------------------------------

has_n <- all(c("n_x","n_y") %in% names(edges_raw))

to_long <- function(df, use_n = FALSE) {
  if (use_n) {
    bind_rows(
      df %>%
        transmute(meta_group = x,
                  celltype_new = x_name,
                  system       = .data$system,
                  n            = n_x,
                  sh           = sh_x),
      df %>%
        transmute(meta_group = y,
                  celltype_new = y_name,
                  system       = .data$system,
                  n            = n_y,
                  sh           = sh_y)
    )
  } else {
    bind_rows(
      df %>%
        transmute(meta_group = x,
                  celltype_new = x_name,
                  system       = .data$system,
                  n            = 1,
                  sh           = sh_x),
      df %>%
        transmute(meta_group = y,
                  celltype_new = y_name,
                  system       = .data$system,
                  n            = 1,
                  sh           = sh_y)
    )
  }
}

node_scores_long <- to_long(edges_raw, use_n = has_n) %>%
  filter(!is.na(meta_group), !is.na(celltype_new), !is.na(sh))

node_scores <- node_scores_long %>%
  group_by(meta_group, celltype_new) %>%
  summarise(
    n_tot = sum(replace_na(n, 0)),
    sh_w  = ifelse(
      n_tot > 0,
      weighted.mean(replace_na(sh, 0), w = replace_na(n, 0)),
      mean(replace_na(sh, 0))
    ),
    .groups = "drop"
  )

# --- Attach node scores to nodes.txt ------------------------------------

nodes <- nodes_raw %>%
  select(system, meta_group, celltype_new) %>%
  left_join(node_scores %>% select(meta_group, sh_w),
            by = "meta_group") %>%
  distinct(meta_group, .keep_all = TRUE)

# --- Filter edges: drop Spatial continuity only -------------------------

edges_filt <- edges_raw %>%
  filter(edge_type != "Spatial continuity") %>%
  # keep basic structure and HH edge scores if present
  select(
    system, x, y, x_name, y_name, edge_type,
    dplyr::any_of(c("abs_delta","delta","sh_x","sh_y"))
  )

# --- Map node ids -------------------------------------------------------

nodes$id <- seq_len(nrow(nodes))
id_map <- setNames(nodes$id, nodes$meta_group)

edges_filt <- edges_filt %>%
  mutate(
    x_id = id_map[x],
    y_id = id_map[y]
  ) %>%
  filter(!is.na(x_id), !is.na(y_id))

# --- Build graph object -------------------------------------------------

g <- graph_from_data_frame(
  d = edges_filt %>%
    transmute(
      from      = x_id,
      to        = y_id,
      edge_type = edge_type,
      abs_delta = if ("abs_delta" %in% names(edges_filt)) abs_delta else NA_real_,
      delta     = if ("delta"     %in% names(edges_filt)) delta     else NA_real_
    ),
  vertices = nodes %>%
    transmute(
      id          = id,
      meta_group  = meta_group,
      celltype_new = celltype_new,
      system      = system,
      sh          = sh_w
    ),
  directed = TRUE
) %>% as_tbl_graph()

# --- Edge and node aesthetics (no plotting yet) -------------------------

E(g)$ew <- rescale(replace_na(E(g)$abs_delta, 0), to = c(0.2, 2))
E(g)$ec <- abs(replace_na(E(g)$delta, 0))
V(g)$nv <- replace_na(V(g)$sh, 0)

# --- Save objects for later use -----------------------------------------

saveRDS(g,      file.path(OUTPUT_CHRIS_TREE, "devtree_graph.rds"))
write.table(
  as.data.frame(nodes),
  file = file.path(OUTPUT_CHRIS_TREE, "devtree_nodes_for_graph.tsv"),
  sep = "\t", quote = FALSE, row.names = FALSE
)
write.table(
  as.data.frame(edges_filt),
  file = file.path(OUTPUT_CHRIS_TREE, "devtree_edges_for_graph.tsv"),
  sep = "\t", quote = FALSE, row.names = FALSE
)

cat("Tree graph built and saved as devtree_graph.rds\n")

############################################################
## Optional plotting code. Leave commented for now.
############################################################

## Uncomment when you are ready to plot.

pdf(file.path(OUTPUT_CHRIS_TREE, "devtree_sugiyama_HH_overlay.pdf"), width = 20, height = 10)
p <- ggraph(g, layout = "sugiyama") +
  geom_edge_link(aes(edge_width = ew, edge_colour = ec),
                 arrow = arrow(length = unit(2, "mm")),
                 end_cap = circle(2, "mm"),
                 show.legend = TRUE) +
  scale_edge_colour_viridis(name = "|delta|", option = "C", direction = 1,
                            na.value = "grey80") +
  scale_edge_width(range = c(0.2, 2), guide = guide_legend(title = "abs_delta")) +
  geom_node_point(aes(colour = nv), size = 2) +
  scale_colour_viridis(name = "Node Hh (sh)", option = "B", direction = 1) +
  geom_node_text(aes(label = celltype_new), size = 1.6, vjust = -0.6, check_overlap = TRUE) +
  theme_void() +
  theme(legend.position = "bottom", plot.margin = margin(10, 10, 10, 10))
print(p)
dev.off()

pdf(file.path(OUTPUT_CHRIS_TREE, "devtree_sugiyama_HH_overlay_unlabeled.pdf"), width = 20, height = 10)
p2 <- ggraph(g, layout = "sugiyama") +
  geom_edge_link(aes(edge_width = ew, edge_colour = ec),
                 arrow = arrow(length = unit(2, "mm")),
                 end_cap = circle(2, "mm")) +
  scale_edge_colour_viridis(name = "|delta|", option = "C", direction = 1,
                            na.value = "grey80") +
  scale_edge_width(range = c(0.2, 2), guide = guide_legend(title = "abs_delta")) +
  geom_node_point(aes(colour = nv), size = 2) +
  scale_colour_viridis(name = "Node Hh (sh)", option = "B", direction = 1) +
  theme_void() +
  theme(legend.position = "bottom", plot.margin = margin(10, 10, 10, 10))
print(p2)
dev.off()
