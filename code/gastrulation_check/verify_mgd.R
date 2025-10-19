#!/usr/bin/env Rscript
project_dir <- "/project/imoskowitz/xyang2/chrislowzhengxi"
infile <- file.path(project_dir, "results", "MouseGastrulation_processed.rds")
preview_out <- file.path(project_dir, "results", "gastrulation_ids_preview.csv")

if(!file.exists(infile)){
  cat("Saved RDS not found at:", infile, "\n")
  quit(status = 2)
}

sce <- readRDS(infile)
cat("Loaded RDS. Object class:", class(sce), "\n")
if(requireNamespace("SingleCellExperiment", quietly = TRUE) && inherits(sce, "SingleCellExperiment")){
  cat("Dimensions:", paste(dim(sce), collapse = " x "), "\n")
  md <- as.data.frame(colData(sce))
} else if(is.data.frame(sce)){
  md <- sce
} else {
  cat("Unknown object type; attempting to coerce to data.frame\n")
  md <- try(as.data.frame(sce), silent = TRUE)
  if(inherits(md, "try-error")){
    cat("Could not coerce object to data.frame.\n")
    quit(status = 3)
  }
}

first200 <- head(md, 200)
if(!("celltype.mapped" %in% colnames(first200))){
  cat("Warning: 'celltype.mapped' not found in metadata columns. Columns available:\n")
  print(colnames(first200))
}
if(!("stage" %in% colnames(first200))){
  cat("Warning: 'stage' not found in metadata columns.\n")
}

write.csv(first200[, intersect(c("cell_id","celltype.mapped","stage"), colnames(first200)), drop = FALSE], preview_out, row.names = FALSE)
cat("Wrote preview CSV to:", preview_out, "\n")
cat("Done.\n")
