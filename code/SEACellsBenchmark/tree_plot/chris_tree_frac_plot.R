# #############################################
# ## chris_tree_frac_plot.R
# ## Global tree with node color = % SHH > 0
# #############################################

# library(dplyr)
# library(readr)
# library(tidyr)
# library(igraph)
# library(tidygraph)
# library(ggraph)
# library(colorspace)

# tree_dir <- "/project/imoskowitz/xyang2/chrislowzhengxi/results/ucell/tree"

# # ---------- 1) Load inputs ----------

# edges_frac_path   <- file.path(tree_dir, "edges_all_with_frac.csv")
# nodes_graph_path  <- file.path(tree_dir, "devtree_nodes_for_graph.tsv")
# nodes_filt_path   <- file.path(tree_dir, "nodes_filtered.txt")
# dev_edges_path    <- file.path(tree_dir, "devtree_edges_for_graph.tsv")

# edges_frac  <- read_csv(edges_frac_path, show_col_types = FALSE)
# nodes_graph <- read.delim(nodes_graph_path, sep = "\t", stringsAsFactors = FALSE)
# nodes_filt  <- read.delim(nodes_filt_path,  sep = "\t", stringsAsFactors = FALSE)
# dev_edges   <- read.delim(dev_edges_path,   sep = "\t", stringsAsFactors = FALSE)

# # ---------- 2) Compute node-level %SHH > 0 ----------

# frac_long <- bind_rows(
#   edges_frac %>%
#     transmute(meta_group = x,
#               frac      = `frac>0_x`,
#               n         = n_x),
#   edges_frac %>%
#     transmute(meta_group = y,
#               frac      = `frac>0_y`,
#               n         = n_y)
# ) %>%
#   filter(!is.na(meta_group), !is.na(frac)) %>%
#   mutate(
#     # force to numeric in case they came in as character
#     frac = as.numeric(frac),
#     n    = as.numeric(n)
#   )

# node_frac <- frac_long %>%
#   group_by(meta_group) %>%
#   summarise(
#     n_tot  = sum(replace_na(n, 0)),
#     frac_w = ifelse(
#       n_tot > 0,
#       weighted.mean(replace_na(frac, 0), w = replace_na(n, 0)),
#       mean(replace_na(frac, 0))
#     ),
#     .groups = "drop"
#   )

# # ---------- 3) Attach new_id and frac_w to the graph nodes ----------

# nodes2 <- nodes_graph %>%
#   left_join(nodes_filt %>% select(meta_group, new_id),
#             by = "meta_group") %>%
#   left_join(node_frac, by = "meta_group") %>%
#   mutate(
#     frac_w = replace_na(frac_w, 0)
#   )

# # ---------- 4) Build graph (structure from devtree_edges_for_graph) ----------

# g_frac <- graph_from_data_frame(
#   d = dev_edges %>%
#     transmute(from = x_id, to = y_id),
#   vertices = nodes2 %>%
#     transmute(
#       name        = id,                         # igraph vertex id
#       new_id      = ifelse(is.na(new_id), id, new_id),
#       meta_group  = meta_group,
#       celltype_new = celltype_new,
#       system      = system,
#       frac_w      = frac_w
#     ),
#   directed = TRUE
# ) %>% as_tbl_graph()

# saveRDS(g_frac, file.path(tree_dir, "devtree_graph_frac.rds"))

# # ---------- 5) Plot: TREE and SUGIYAMA layouts ----------

# pal_mint_r <- colorspace::sequential_hcl(256, palette = "Mint", rev = TRUE)

# ## A) TREE layout
# pdf(file.path(tree_dir, "devtree_tree_frac_overlay.pdf"),
#     width = 24, height = 12)

# p_tree_frac <- ggraph(g_frac, layout = "tree") +
#   geom_edge_link(
#     arrow    = arrow(length = unit(2, "mm")),
#     end_cap  = circle(2, "mm"),
#     colour   = "grey75",
#     linewidth = 0.3
#   ) +
#   geom_node_point(aes(colour = frac_w), size = 2) +
#   scale_colour_gradientn(
#     colours = pal_mint_r,
#     name    = "% SHH > 0"
#   ) +
#   geom_node_text(aes(label = new_id), size = 1.5,
#                  vjust = -0.6, check_overlap = TRUE) +
#   theme_void() +
#   theme(
#     legend.position = "bottom",
#     plot.margin = margin(10, 10, 10, 10)
#   )

# print(p_tree_frac)
# dev.off()

# ## B) SUGIYAMA layout
# pdf(file.path(tree_dir, "devtree_sugiyama_frac_overlay.pdf"),
#     width = 24, height = 12)

# p_sug_frac <- ggraph(g_frac, layout = "sugiyama") +
#   geom_edge_link(
#     arrow    = arrow(length = unit(2, "mm")),
#     end_cap  = circle(2, "mm"),
#     colour   = "grey75",
#     linewidth = 0.3
#   ) +
#   geom_node_point(aes(colour = frac_w), size = 2) +
#   scale_colour_gradientn(
#     colours = pal_mint_r,
#     name    = "% SHH > 0"
#   ) +
#   geom_node_text(aes(label = new_id), size = 1.5,
#                  vjust = -0.6, check_overlap = TRUE) +
#   theme_void() +
#   theme(
#     legend.position = "bottom",
#     plot.margin = margin(10, 10, 10, 10)
#   )

# print(p_sug_frac)
# dev.off()

# cat("Finished: devtree_tree_frac_overlay.pdf and devtree_sugiyama_frac_overlay.pdf\n")


#############################################
## chris_tree_frac_plot.R
## Global tree with:
##   - node color = % SHH > 0 (frac_w)
##   - edge width = abs_delta
##   - edge color = |delta|
#############################################

library(dplyr)
library(readr)
library(tidyr)
library(igraph)
library(tidygraph)
library(ggraph)
library(colorspace)
library(scales)

tree_dir <- "/project/imoskowitz/xyang2/chrislowzhengxi/results/ucell/tree"

# ---------- 1) Load inputs ----------

edges_frac_path   <- file.path(tree_dir, "edges_all_with_frac.csv")
nodes_graph_path  <- file.path(tree_dir, "devtree_nodes_for_graph.tsv")
nodes_filt_path   <- file.path(tree_dir, "nodes_filtered.txt")
dev_edges_path    <- file.path(tree_dir, "devtree_edges_for_graph.tsv")

edges_frac  <- read_csv(edges_frac_path, show_col_types = FALSE)
nodes_graph <- read.delim(nodes_graph_path, sep = "\t", stringsAsFactors = FALSE)
nodes_filt  <- read.delim(nodes_filt_path,  sep = "\t", stringsAsFactors = FALSE)
dev_edges   <- read.delim(dev_edges_path,   sep = "\t", stringsAsFactors = FALSE)

# ---------- 2) Compute node-level %SHH > 0 ----------

frac_long <- bind_rows(
  edges_frac %>%
    transmute(meta_group = x,
              frac      = `frac>0_x`,
              n         = n_x),
  edges_frac %>%
    transmute(meta_group = y,
              frac      = `frac>0_y`,
              n         = n_y)
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

# ---------- 3) Attach new_id and frac_w to the graph nodes ----------

nodes2 <- nodes_graph %>%
  left_join(nodes_filt %>% select(meta_group, new_id),
            by = "meta_group") %>%
  left_join(node_frac, by = "meta_group") %>%
  mutate(
    frac_w = replace_na(frac_w, 0)
  )

# ---------- 4) Build graph and add edge aesthetics ----------

g_frac <- graph_from_data_frame(
  d = dev_edges %>%
    transmute(from = x_id, to = y_id),
  vertices = nodes2 %>%
    transmute(
      name        = id,                         # igraph vertex id
      new_id      = ifelse(is.na(new_id), id, new_id),
      meta_group  = meta_group,
      celltype_new = celltype_new,
      system      = system,
      frac_w      = frac_w
    ),
  directed = TRUE
) %>% as_tbl_graph()

# Attach edge attributes: abs_delta and delta from dev_edges
# (same row order)
E(g_frac)$abs_delta <- dev_edges$abs_delta
E(g_frac)$delta     <- dev_edges$delta

# Edge aesthetics: width from abs_delta, color from |delta|
E(g_frac)$ew <- scales::rescale(
  replace_na(E(g_frac)$abs_delta, 0),
  to = c(0.2, 2.0)
)
E(g_frac)$ec <- abs(replace_na(E(g_frac)$delta, 0))

saveRDS(g_frac, file.path(tree_dir, "devtree_graph_frac.rds"))

# ---------- 5) Palettes ----------

pal_mint_r   <- colorspace::sequential_hcl(256, palette = "Mint",   rev = TRUE)  # nodes
pal_rocket_r <- colorspace::sequential_hcl(256, palette = "Rocket", rev = TRUE)  # edges

# ---------- 6) Plot: TREE layout ----------

pdf(file.path(tree_dir, "devtree_tree_frac_overlay.pdf"),
    width = 24, height = 12)

p_tree_frac <- ggraph(g_frac, layout = "tree") +
  geom_edge_link(
    aes(edge_width = ew, edge_colour = ec),
    arrow    = arrow(length = unit(2, "mm")),
    end_cap  = circle(2, "mm"),
    show.legend = TRUE
  ) +
  scale_edge_colour_gradientn(
    colours = pal_rocket_r,
    name    = "|delta|",
    na.value = "grey80"
  ) +
  scale_edge_width(
    range = c(0.2, 2.0),
    guide = guide_legend(title = "abs_delta")
  ) +
  geom_node_point(aes(colour = frac_w), size = 2) +
  scale_colour_gradientn(
    colours = pal_mint_r,
    name    = "% SHH > 0"
  ) +
  geom_node_text(aes(label = new_id), size = 1.5,
                 vjust = -0.6, check_overlap = TRUE) +
  theme_void() +
  theme(
    legend.position = "bottom",
    plot.margin = margin(10, 10, 10, 10)
  )

print(p_tree_frac)
dev.off()

# ---------- 7) Plot: SUGIYAMA layout ----------

pdf(file.path(tree_dir, "devtree_sugiyama_frac_overlay.pdf"),
    width = 24, height = 12)

p_sug_frac <- ggraph(g_frac, layout = "sugiyama") +
  geom_edge_link(
    aes(edge_width = ew, edge_colour = ec),
    arrow    = arrow(length = unit(2, "mm")),
    end_cap  = circle(2, "mm"),
    show.legend = TRUE
  ) +
  scale_edge_colour_gradientn(
    colours = pal_rocket_r,
    name    = "|delta|",
    na.value = "grey80"
  ) +
  scale_edge_width(
    range = c(0.2, 2.0),
    guide = guide_legend(title = "abs_delta")
  ) +
  geom_node_point(aes(colour = frac_w), size = 2) +
  scale_colour_gradientn(
    colours = pal_mint_r,
    name    = "% SHH > 0"
  ) +
  geom_node_text(aes(label = new_id), size = 1.5,
                 vjust = -0.6, check_overlap = TRUE) +
  theme_void() +
  theme(
    legend.position = "bottom",
    plot.margin = margin(10, 10, 10, 10)
  )

print(p_sug_frac)
dev.off()

cat("Finished: devtree_tree_frac_overlay.pdf and devtree_sugiyama_frac_overlay.pdf\n")
