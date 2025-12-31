#!/usr/bin/env Rscript
suppressPackageStartupMessages({
  library(dplyr)
  library(readr)
})

message("== Gastrulation score_genes + Edge integration ==")

# ------------------------------------------------------------------
# Config
# ------------------------------------------------------------------
system_tag <- "Gastrulation"

edges_txt <- "/project/imoskowitz/xyang2/SHH/Qiu_TimeLapse/other/edges.txt"
nodes_txt <- "/project/imoskowitz/xyang2/SHH/Qiu_TimeLapse/other/nodes.txt"

# score_genes QC summary (already computed by Scanpy)
qc_csv <- "/project/imoskowitz/xyang2/chrislowzhengxi/results/score_genes_new/Gastrulation/Gastrulation_SHH_scoregenes_summary.csv"

out_root  <- "/project/imoskowitz/xyang2/chrislowzhengxi/results/score_genes_new"
edges_dir <- file.path(out_root, system_tag)
dir.create(edges_dir, recursive = TRUE, showWarnings = FALSE)

edges_out_csv <- file.path(
  edges_dir,
  paste0(system_tag, "_edge_filtered_with_shh_scoregenes.csv")
)

edges_out_txt <- file.path(
  edges_dir,
  paste0(system_tag, "_edge_filtered_with_shh_scoregenes.txt")
)

# ------------------------------------------------------------------
# Load inputs
# ------------------------------------------------------------------
message("[LOAD] score_genes QC: ", qc_csv)
qc <- read_csv(qc_csv, show_col_types = FALSE)

# ---- normalize column names (robust to Scanpy outputs) ----
qc <- qc |>
  rename(
    celltype_update = celltype,
    sh_score = mean,
    n        = n_cells,
    `frac>0` = frac_gt0
  )

edges <- read.delim(edges_txt, sep = "\t", stringsAsFactors = FALSE) |>
  filter(system == system_tag)

nodes <- read.delim(nodes_txt, sep = "\t", stringsAsFactors = FALSE) |>
  filter(system == system_tag) |>
  transmute(
    system,
    node_id     = meta_group,
    node_name   = celltype_new,
    node_number = celltype_num
  )

# ------------------------------------------------------------------
# Build edge table
# ------------------------------------------------------------------
message("[EDGE] Integrating score_genes with edges/nodes")

edges2 <- edges |>
  mutate(
    x_id = x,
    y_id = y
  ) |>
  # numeric node codes
  left_join(
    nodes |> rename(x_name = node_name, x_number = node_number),
    by = c("system", "x_name")
  ) |>
  left_join(
    nodes |> rename(y_name = node_name, y_number = node_number),
    by = c("system", "y_name")
  ) |>
  # score_genes stats
  left_join(
    qc |> rename(
      x_name     = celltype_update,
      sh_x       = sh_score,
      n_x        = n,
      median_x  = median,
      q90_x     = q90,
      `frac>0_x`= `frac>0`,
      variance_x= variance,
      std_x     = std
    ),
    by = "x_name"
  ) |>
  left_join(
    qc |> rename(
      y_name     = celltype_update,
      sh_y       = sh_score,
      n_y        = n,
      median_y  = median,
      q90_y     = q90,
      `frac>0_y`= `frac>0`,
      variance_y= variance,
      std_y     = std
    ),
    by = "y_name"
  ) |>
  mutate(
    abs_delta = abs(sh_x - sh_y),
    delta     = sh_y - sh_x,
    cohens_d  = {
      pooled_sd <- sqrt((variance_x + variance_y) / 2)
      round(delta / ifelse(is.na(pooled_sd) | pooled_sd == 0,
                           NA_real_, pooled_sd), 4)
    },
    pct_change     = 100 * delta / ifelse(abs(sh_x) < 1e-9, NA_real_, sh_x),
    abs_pct_change = abs(pct_change)
  )

edges3 <- edges2 |>
  select(any_of(c(
    "system","x","y","x_name","y_name","edge_type",
    "x_number","y_number","x_id","y_id",
    "n_x","median_x","sh_x","q90_x","frac>0_x","variance_x","std_x",
    "n_y","median_y","sh_y","q90_y","frac>0_y","variance_y","std_y",
    "abs_delta","delta","cohens_d","pct_change","abs_pct_change"
  )))

# ------------------------------------------------------------------
# Write outputs
# ------------------------------------------------------------------
write_csv(edges3, edges_out_csv, na = "")
write.table(edges3, edges_out_txt,
            sep = "\t", quote = FALSE,
            row.names = FALSE, na = "")

message("[EDGE] wrote: ", edges_out_csv)
message("[EDGE] wrote: ", edges_out_txt)
message("\n✓ Done (score_genes).")
