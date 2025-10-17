#!/usr/bin/env Rscript
suppressPackageStartupMessages({
  library(SingleCellExperiment)
  library(Matrix)
  library(UCell)
  library(dplyr)
  library(readr)
})

message("== Gastrulation UCell + Edge integration ==")

# ------------------------------------------------------------------
# Config (edit only if your layout changes)
# ------------------------------------------------------------------
system_tag    <- "Gastrulation"

sce_in        <- "/project/imoskowitz/xyang2/chrislowzhengxi/data/gastrulation/mouse_gastrulation_sce_overlap_pd_labeled.rds"
sce_out_ucell <- "/project/imoskowitz/xyang2/chrislowzhengxi/data/gastrulation/mouse_gastrulation_sce_overlap_pd_labeled_ucell.rds"

edges_txt     <- "/project/imoskowitz/xyang2/SHH/Qiu_TimeLapse/other/edges.txt"
nodes_txt     <- "/project/imoskowitz/xyang2/SHH/Qiu_TimeLapse/other/nodes.txt"

out_root      <- "/project/imoskowitz/xyang2/chrislowzhengxi/results/ucell"
qc_dir        <- file.path(out_root, system_tag, "qc")
edges_dir     <- file.path(out_root, system_tag)
dir.create(qc_dir,    recursive = TRUE, showWarnings = FALSE)
dir.create(edges_dir, recursive = TRUE, showWarnings = FALSE)

qc_csv        <- file.path(qc_dir,    paste0(system_tag, "_ALL_labels_summary.csv"))
edges_out_csv <- file.path(edges_dir, paste0(system_tag, "_edge_filtered_with_shh.csv"))
edges_out_txt <- file.path(edges_dir, paste0(system_tag, "_edge_filtered_with_shh.txt"))

# Threads (respect SLURM)
ncores <- as.integer(Sys.getenv("SLURM_CPUS_PER_TASK", unset = "4"))
Sys.setenv(OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
           VECLIB_MAXIMUM_THREADS="1", NUMEXPR_NUM_THREADS="1")
message(sprintf("[INFO] ncores=%d", ncores))

# ------------------------------------------------------------------
# Load SCE and ensure labels/system
# ------------------------------------------------------------------
message("[LOAD] ", sce_in)
sce <- readRDS(sce_in)

stopifnot("celltype_update" %in% colnames(colData(sce)))
if (!("system" %in% colnames(colData(sce)))) {
  colData(sce)$system <- system_tag
}

# ------------------------------------------------------------------
# Ensure SYMBOL rownames for UCell (your SCE rownames are Ensembl IDs)
# ------------------------------------------------------------------
syms <- as.character(rowData(sce)$SYMBOL)
keep <- !is.na(syms) & nzchar(syms)
sce  <- sce[keep, , drop = FALSE]
syms <- syms[keep]
rownames(sce) <- make.unique(syms)

# Column-compressed sparse for speed
cts <- counts(sce)
if (!inherits(cts, "CsparseMatrix")) {
  counts(sce) <- as(cts, "CsparseMatrix")
}
rm(cts); invisible(gc())

# ------------------------------------------------------------------
# UCell SHH scoring
# ------------------------------------------------------------------
sig_genes <- c("Gli1","Ptch1","Hhip")
present   <- sig_genes[sig_genes %in% rownames(sce)]
if (!length(present)) stop("None of Gli1/Ptch1/Hhip found after SYMBOL mapping.")
if (length(present) < length(sig_genes)) {
  message("[WARN] Missing: ", paste(setdiff(sig_genes, present), collapse = ", "))
}

message("[UCELL] Store rankings…")
ranks <- StoreRankings_UCell(counts(sce))

message(sprintf("[UCELL] Score signatures (ncores=%d)…", ncores))
sc <- ScoreSignatures_UCell(precalc.ranks = ranks, features = list(SHH = present), ncores = ncores)

colname <- if ("SHH_UCell" %in% colnames(sc)) "SHH_UCell" else "SHH"
sce$SHH_UCell_score <- sc[, colname]
rm(ranks, sc); invisible(gc())

# ------------------------------------------------------------------
# QC table grouped by celltype_update (schema used downstream)
# ------------------------------------------------------------------
message("[QC] Building per-label summary…")
qc <- as.data.frame(colData(sce)) |>
  transmute(celltype_update = .data$celltype_update,
            SHH = sce$SHH_UCell_score) |>
  group_by(celltype_update) |>
  summarize(
    n_cells = n(),
    median  = median(SHH),
    mean    = mean(SHH),
    q90     = quantile(SHH, 0.90),
    `frac>0`= mean(SHH > 0),
    variance= var(SHH),
    std     = sd(SHH),
    .groups = "drop"
  )

write_csv(qc, qc_csv)
message("[QC] wrote: ", qc_csv)

# Save SCE with scores (compress=FALSE for speed on /project)
saveRDS(sce, sce_out_ucell, compress = FALSE)
message("[SAVE] SCE w/ SHH_UCell_score: ", sce_out_ucell)

# ------------------------------------------------------------------
# Edge integration: join with edges.txt and nodes.txt
# ------------------------------------------------------------------
message("[EDGE] Integrating with edges/nodes for system: ", system_tag)

edges <- read.delim(edges_txt, sep = "\t", stringsAsFactors = FALSE) |>
  filter(system == system_tag)

nodes <- read.delim(nodes_txt, sep = "\t", stringsAsFactors = FALSE) |>
  filter(system == system_tag) |>
  transmute(system,
            node_id     = meta_group,       # e.g., PGa_M1 / L_M24
            node_name   = celltype_new,     # e.g., "Oocyte"
            node_number = celltype_num)     # numeric code (may be missing for some systems)

# Per-node scores (for merges)
scores <- qc |>
  rename(node_name = celltype_update,
         sh_score  = mean,
         n         = n_cells)

# Build edge table: attach IDs, numbers, and SHH stats
edges2 <- edges |>
  mutate(
    x_id = x,  # use IDs provided in edges.txt
    y_id = y
  ) |>
  # numeric codes from nodes (if present)
  left_join(nodes |> rename(x_name = node_name,
                            x_number = node_number),
            by = c("system","x_name")) |>
  left_join(nodes |> rename(y_name = node_name,
                            y_number = node_number),
            by = c("system","y_name")) |>
  # attach SHH stats for x and y
  left_join(scores |> rename(x_name = node_name,
                             sh_x = sh_score, n_x = n,
                             median_x = median, q90_x = q90,
                             `frac>0_x` = `frac>0`,
                             variance_x = variance, std_x = std),
            by = "x_name") |>
  left_join(scores |> rename(y_name = node_name,
                             sh_y = sh_score, n_y = n,
                             median_y = median, q90_y = q90,
                             `frac>0_y` = `frac>0`,
                             variance_y = variance, std_y = std),
            by = "y_name") |>
  mutate(
    abs_delta = abs(sh_x - sh_y),
    delta     = sh_y - sh_x,
    cohens_d  = {
      pooled_sd <- sqrt((variance_x + variance_y)/2)
      round(delta / ifelse(is.na(pooled_sd) | pooled_sd == 0, NA_real_, pooled_sd), 4)
    },
    pct_change     = 100 * delta / ifelse(abs(sh_x) < 1e-9, NA_real_, sh_x),
    abs_pct_change = abs(pct_change)
  )

# Robust column selection (tolerates missing optional fields)
edges3 <- edges2 |>
  dplyr::select(dplyr::any_of(c(
    "system","x","y","x_name","y_name","edge_type",
    "x_number","y_number","x_id","y_id",
    "n_x","median_x","sh_x","q90_x","frac>0_x","variance_x","std_x",
    "n_y","median_y","sh_y","q90_y","frac>0_y","variance_y","std_y",
    "abs_delta","delta","cohens_d","pct_change","abs_pct_change"
  )))

write_csv(edges3, edges_out_csv, na = "")
write.table(edges3, edges_out_txt, sep = "\t", quote = FALSE, row.names = FALSE, na = "")

message("[EDGE] wrote: ", edges_out_csv)
message("[EDGE] wrote: ", edges_out_txt)

message("\n✓ Done.")
