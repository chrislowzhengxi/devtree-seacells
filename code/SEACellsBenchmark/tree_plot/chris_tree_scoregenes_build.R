# library(igraph)
# library(dplyr)
# library(ggraph)
# library(tidygraph)
# library(RColorBrewer)
# library(scales)
# library(viridis)

# # ==========================
# #  PATHS
# # ==========================

# INPUT_QIU_OTHER   <- "/project/imoskowitz/xyang2/SHH/Qiu_TimeLapse/other"
# OUTPUT_CHRIS_TREE <- "/project/imoskowitz/xyang2/chrislowzhengxi/results/score_genes_new/tree"

# dir.create(OUTPUT_CHRIS_TREE, recursive = TRUE, showWarnings = FALSE)

# QiuFile_path <- INPUT_QIU_OTHER

# nodes_path <- file.path(INPUT_QIU_OTHER, "nodes.txt")

# edges_csv <- file.path(
#   "/project/imoskowitz/xyang2/chrislowzhengxi/results/score_genes_new",
#   "full_scored_edges_with_pregastrulation_scoregenes.csv"
# )
# edges_dir <- dirname(edges_csv)

# # ==========================
# #  LOAD NODES / EDGES
# # ==========================

# nodes <- read.table(nodes_path, header = TRUE, sep = "\t", as.is = TRUE)
# dim(nodes)
# length(unique(nodes$system))
# length(unique(nodes$celltype_new))

# edges <- read.csv(edges_csv, check.names = FALSE)
# dim(edges)
# table(edges$edge_type)

# # ==========================
# #  HELPER: MERGE REDUNDANT CELLTYPES
# # ==========================

# mergeRedundantCelltype <- function(nodes_df, edges_df,
#                                    merge_by = "celltype_new",
#                                    filtered_by = "Spatial continuity") {
#   ed <- edges_df
#   if (!is.null(filtered_by) && "edge_type" %in% names(ed)) {
#     ed <- ed[ed$edge_type != filtered_by, , drop = FALSE]
#   }

#   # one representative row per celltype
#   nodes_rep <- nodes_df %>%
#     dplyr::group_by(.data[[merge_by]]) %>%
#     dplyr::slice(1) %>%
#     dplyr::ungroup()
#   nodes_rep$new_id <- seq_len(nrow(nodes_rep))

#   id_map <- stats::setNames(nodes_rep$new_id, nodes_rep[[merge_by]])
#   ed$x_id <- id_map[ed$x_name]
#   ed$y_id <- id_map[ed$y_name]
#   ed <- ed[!is.na(ed$x_id) & !is.na(ed$y_id), , drop = FALSE]

#   list(nodes = nodes_rep, edges = ed)
# }

# # ==========================
# #  BUILD INITIAL GRAPH (ALL EDGES)
# # ==========================

# nodes$id <- seq_len(nrow(nodes))
# node_name_map <- setNames(nodes$id, nodes$meta_group)

# edges$x_number <- node_name_map[edges$x]
# edges$y_number <- node_name_map[edges$y]

# g_all <- graph_from_data_frame(
#   d = edges[, c("x_number", "y_number")],
#   vertices = nodes[, c("id", "meta_group", "celltype_new", "system")],
#   directed = TRUE
# ) %>% as_tbl_graph()

# E(g_all)$label <- edges$edge_type

# # ==========================
# #  REMOVE SPATIAL CONTINUITY, COLLAPSE REDUNDANT LABELS
# # ==========================

# g_sub <- g_all %>%
#   activate(edges) %>%
#   filter(label != "Spatial continuity") %>%
#   as_tbl_graph()

# length(unique(V(g_sub)$celltype_new))

# nodes_sub <- subset(nodes, celltype_new %in% V(g_sub)$celltype_new)

# tbs_final <- mergeRedundantCelltype(
#   nodes_df = nodes_sub,
#   edges_df = edges,
#   merge_by = "celltype_new",
#   filtered_by = "Spatial continuity"
# )

# dim(tbs_final$nodes)   # should be 262 x ...
# dim(tbs_final$edges)   # ~ 355 x ...

# write.table(
#   tbs_final$nodes,
#   file = file.path(OUTPUT_CHRIS_TREE, "nodes_filtered.txt"),
#   row.names = FALSE, sep = "\t", quote = FALSE
# )

# write.table(
#   tbs_final$edges,
#   file = file.path(OUTPUT_CHRIS_TREE, "edges_filtered.txt"),
#   row.names = FALSE, sep = "\t", quote = FALSE
# )

# # ==========================
# #  BUILD FINAL GRAPH OBJECT
# # ==========================

# g <- graph_from_data_frame(
#   d = tbs_final$edges[, c("x_id", "y_id")],
#   vertices = tbs_final$nodes[, c("new_id", "meta_group", "celltype_new", "system")],
#   directed = TRUE
# ) %>% as_tbl_graph()

# V(g)$id <- tbs_final$nodes$new_id

# # ==========================
# #  ATTACH EDGE DELTAS (abs_delta / delta) FROM SCORE_GENES
# # ==========================

# attach_edge_deltas <- function(tbs_edges, search_root, fallback_csv) {
#   found_files <- list.files(
#     search_root,
#     pattern = "(edge|Edges).*\\.(csv|rds)$",
#     full.names = TRUE,
#     recursive = TRUE
#   )

#   found_files <- unique(c(found_files, fallback_csv))

#   read_any <- function(f) {
#     ext <- tolower(tools::file_ext(f))
#     if (ext == "csv") {
#       tryCatch(read.csv(f, check.names = FALSE), error = function(e) NULL)
#     } else if (ext == "rds") {
#       tryCatch(readRDS(f), error = function(e) NULL)
#     } else {
#       NULL
#     }
#   }

#   for (f in found_files) {
#     df <- read_any(f)
#     if (is.null(df)) next

#     if (all(c("x_name", "y_name", "abs_delta", "delta") %in% names(df))) {
#       message("Merging SHH deltas from: ", f)

#       tbs_edges$.key <- paste(tbs_edges$x_name, tbs_edges$y_name, sep = " :: ")
#       df$.key        <- paste(df$x_name,     df$y_name,     sep = " :: ")

#       keep_cols <- c(".key", "abs_delta", "delta")
#       out <- merge(
#         tbs_edges,
#         df[, keep_cols, drop = FALSE],
#         by = ".key",
#         all.x = TRUE
#       )
#       out$.key <- NULL
#       return(out)
#     }
#   }

#   message("No file with both 'abs_delta' and 'delta' found. Proceeding without edge deltas.")
#   tbs_edges
# }

# tbs_final$edges <- attach_edge_deltas(
#   tbs_edges   = tbs_final$edges,
#   search_root = edges_dir,
#   fallback_csv = edges_csv
# )

# # Ensure the columns exist, even if we could not find real values
# has_delta_cols <- all(c("abs_delta", "delta") %in% names(tbs_final$edges))

# if (!has_delta_cols) {
#   message("Warning: abs_delta / delta columns not found. Creating NA placeholders.")
#   if (!("abs_delta" %in% names(tbs_final$edges))) {
#     tbs_final$edges$abs_delta <- NA_real_
#   }
#   if (!("delta" %in% names(tbs_final$edges))) {
#     tbs_final$edges$delta <- NA_real_
#   }
# }

# stopifnot(length(E(g)) == nrow(tbs_final$edges))

# E(g)$abs_delta <- tbs_final$edges$abs_delta
# E(g)$delta     <- tbs_final$edges$delta


# # optional: give systems colors for later plotting
# n_colors <- length(unique(tbs_final$nodes$system))
# palette <- colorRampPalette(brewer.pal(8, "Set1"))(n_colors)
# V(g)$color <- palette[as.factor(tbs_final$nodes$system)]
# E(g)$label <- tbs_final$edges$edge_type

# # ==========================
# #  SAVE GRAPH FOR PLOTTING SCRIPT
# # ==========================

# saveRDS(g, file.path(OUTPUT_CHRIS_TREE, "devtree_graph.rds"))

# cat("Saved devtree_graph.rds and nodes/edges_filtered.txt into:\n",
#     OUTPUT_CHRIS_TREE, "\n")
