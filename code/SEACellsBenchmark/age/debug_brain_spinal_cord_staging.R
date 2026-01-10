suppressPackageStartupMessages({
  library(zellkonverter)
  library(SingleCellExperiment)
  library(dplyr)
  library(readr)
})

BRAIN_FILES <- c(
  "/project/imoskowitz/xyang2/chrislowzhengxi/results/raw_added_with_staging/Neurons_adata_scale_with_staging.h5ad",
  "/project/imoskowitz/xyang2/chrislowzhengxi/results/raw_added_with_staging/Other_Brain_spinal_cord_adata_scale_with_staging.h5ad"
)

meta_brain <- lapply(BRAIN_FILES, function(f) {
  message("Reading ", basename(f))
  sce <- zellkonverter::readH5AD(f)
  as.data.frame(colData(sce)) %>%
    select(meta_group, celltype_new, staging) %>%
    mutate(
      meta_group = as.character(meta_group),
      celltype_new = as.character(celltype_new),
      staging = as.character(staging)
    )
}) %>% bind_rows()

# basic sanity
print(head(meta_brain))
print(table(is.na(meta_brain$meta_group)))
print(table(is.na(meta_brain$celltype_new)))

# write raw counts
brain_counts <- meta_brain %>%
  filter(staging != "P0") %>%
  count(meta_group, celltype_new, staging, name = "n_cells") %>%
  arrange(desc(n_cells))

write_tsv(
  brain_counts,
  "brain_spinal_cord_staging_counts_debug.tsv"
)

cat("Wrote brain_spinal_cord_staging_counts_debug.tsv\n")
