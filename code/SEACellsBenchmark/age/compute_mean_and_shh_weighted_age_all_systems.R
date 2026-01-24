# #!/usr/bin/env Rscript weighted mean age by celltype by system
# # memory-safe version of 44479740

# suppressPackageStartupMessages({
#   library(zellkonverter)
#   library(SingleCellExperiment)
#   library(dplyr)
#   library(readr)
# })

# STAGING_ROOT <- "/project/imoskowitz/xyang2/chrislowzhengxi/results/raw_added_with_staging"
# UCELL_ROOT   <- "/project/imoskowitz/xyang2/chrislowzhengxi/results/ucell"

# systems <- c(
#   "Blood",
#   "Gut",
#   "Notochord",
#   "Endothelium",
#   "Epithelial_cells",
#   "Eye",
#   "Renal",
#   "Mesoderm",
#   "Lateral_plate_mesoderm",
#   "PNS_neurons",
#   "PNS_glia",
#   "Neurons",
#   "Other_Brain_spinal_cord"
# )

# out_file <- "age_mean_and_shh_weighted_by_celltype_by_system.tsv"
# first_write <- TRUE

# for (sys in systems) {

#   staging_h5ad <- file.path(
#     STAGING_ROOT,
#     paste0(sys, "_adata_scale_with_staging.h5ad")
#   )

#   ucell_h5ad <- file.path(
#     UCELL_ROOT,
#     sys,
#     paste0(sys, "_adata_with_ucell.h5ad")
#   )

#   if (!file.exists(staging_h5ad) || !file.exists(ucell_h5ad)) {
#     message("[SKIP] ", sys)
#     next
#   }

#   message("[READ] ", sys)

#   ## ---- read staging metadata only ----
#   sce_stage <- readH5AD(staging_h5ad, backed = TRUE)
#   assays(sce_stage) <- NULL

#   meta_stage <- as.data.frame(colData(sce_stage)) %>%
#     select(cell_id, staging, stage_code, celltype_new)

#   rm(sce_stage)
#   gc()

#   ## ---- read ucell metadata only ----
#   sce_ucell <- readH5AD(ucell_h5ad, backed = TRUE)
#   assays(sce_ucell) <- NULL

#   meta_ucell <- as.data.frame(colData(sce_ucell)) %>%
#     select(cell_id, SHH_UCell_score)

#   rm(sce_ucell)
#   gc()

#   ## ---- join + clean ----
#   meta <- meta_stage %>%
#     inner_join(meta_ucell, by = "cell_id") %>%
#     filter(staging != "P0") %>%
#     mutate(
#       system = ifelse(
#         sys %in% c("Neurons", "Other_Brain_spinal_cord"),
#         "Brain_spinal_cord",
#         sys
#       ),
#       celltype_new = as.character(celltype_new),
#       stage_code   = as.numeric(stage_code),
#       shh_score    = as.numeric(SHH_UCell_score)
#     )

#   rm(meta_stage, meta_ucell)
#   gc()

#   ## ---- aggregate ----
#   df_age <- meta %>%
#     group_by(celltype_new) %>%
#     summarise(
#       total_cells = n(),
#       age_mean = mean(stage_code, na.rm = TRUE),
#       age_shh_weighted =
#         sum(stage_code * shh_score, na.rm = TRUE) /
#         sum(shh_score, na.rm = TRUE),
#       .groups = "drop"
#     ) %>%
#     mutate(system = unique(meta$system)) %>%
#     select(system, celltype_new, total_cells, age_mean, age_shh_weighted)

#   rm(meta)
#   gc()

#   ## ---- write incrementally to disk ----
#   write_tsv(
#     df_age,
#     out_file,
#     append = !first_write
#   )
#   first_write <- FALSE

#   rm(df_age)
#   gc()
# }

# cat("DONE. Wrote ", out_file, "\n")

# ===========================================================================================

#!/usr/bin/env Rscript: mean age by celltype by system

suppressPackageStartupMessages({
  library(zellkonverter)
  library(SingleCellExperiment)
  library(dplyr)
  library(readr)
})

## =========================
## Paths
## =========================
STAGING_ROOT <- "/project/imoskowitz/xyang2/chrislowzhengxi/results/raw_added_with_staging"

systems <- c(
  "Blood",
  "Gut",
  "Notochord",
  "Endothelium",
  "Epithelial_cells",
  "Eye",
  "Renal",
  "Mesoderm",
  "Lateral_plate_mesoderm",
  "PNS_neurons",
  "PNS_glia",
  "Neurons",
  "Other_Brain_spinal_cord"
)

# systems <- "Notochord"

out_file <- "age_mean_by_celltype_by_system.tsv"
first_write <- TRUE

## =========================
## Loop over systems
## =========================
for (sys in systems) {

  staging_h5ad <- file.path(
    STAGING_ROOT,
    paste0(sys, "_adata_scale_with_staging.h5ad")
  )

  if (!file.exists(staging_h5ad)) {
    message("[SKIP] ", sys)
    next
  }

  message("[READ] ", sys)

  ## ---- read metadata only ----
  sce <- readH5AD(staging_h5ad)

  meta <- as.data.frame(colData(sce)) %>%
    select(cell_id, staging, stage_code, celltype_new)

  rm(sce)
  gc()

  ## ---- clean + compute mean age ----
  df_age <- meta %>%
    filter(
      staging != "P0",
      !is.na(stage_code)
    ) %>%
    mutate(
      system = ifelse(
        sys %in% c("Neurons", "Other_Brain_spinal_cord"),
        "Brain_spinal_cord",
        sys
      ),
      celltype_new = as.character(celltype_new),
      stage_code   = as.numeric(stage_code)
    ) %>%
    group_by(system, celltype_new) %>%
    summarise(
      total_cells = n(),
      age_mean    = mean(stage_code, na.rm = TRUE),
      .groups = "drop"
    )

  rm(meta)
  gc()

  ## ---- write incrementally ----
  write_tsv(
    df_age,
    out_file,
    append = !first_write
  )
  first_write <- FALSE

  rm(df_age)
  gc()
}

cat("DONE. Wrote ", out_file, "\n")
