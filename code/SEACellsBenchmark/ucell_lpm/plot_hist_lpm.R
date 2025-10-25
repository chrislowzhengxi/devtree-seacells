# 0) Reload the per-cell table
obs <- read.table("/project/imoskowitz/xyang2/chrislowzhengxi/results/ucell/Lateral_plate_mesoderm/qc/Lateral_plate_mesoderm_obs_with_SHH_UCell_score.txt",
                  sep = "\t", header = TRUE, check.names = FALSE, stringsAsFactors = FALSE)

# 1) Set paths and columns
out_dir <- "/project/imoskowitz/xyang2/chrislowzhengxi/results/ucell/Lateral_plate_mesoderm/qc/histograms"
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

score_col <- "SHH_UCell_score"
label_col <- "celltype_new"

# 2) Ensure numeric score
obs[[score_col]] <- suppressWarnings(as.numeric(obs[[score_col]]))
obs <- obs[is.finite(obs[[score_col]]), , drop = FALSE]

# 3) Per celltype_new histograms (save PNGs)
for (lab in sort(unique(obs[[label_col]]))) {
  sub <- obs[obs[[label_col]] == lab, , drop = FALSE]
  if (!nrow(sub)) next
  png(file.path(out_dir, paste0("hist_", gsub("[^A-Za-z0-9]+","_", lab), ".png")),
      width = 1200, height = 800, res = 150)
  hist(sub[[score_col]], breaks = 100, xlim = c(0, 1),
       main = paste("LPM", "—", lab, "(n=", nrow(sub), ")"),
       xlab = score_col, ylab = "Count")
  dev.off()
}

# 4) Overall histogram (save PNG)
png(file.path(out_dir, "hist_LPM_ALL.png"), width = 1200, height = 800, res = 150)
hist(obs[[score_col]], breaks = 100, xlim = c(0, 1),
     main = paste("LPM", "—", "All cells (n=", nrow(obs), ")"),
     xlab = score_col, ylab = "Count")
dev.off()
