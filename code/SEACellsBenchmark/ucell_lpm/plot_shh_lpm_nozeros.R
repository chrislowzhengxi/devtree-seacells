# assumes 'obs' is already loaded and cleaned as before
out_dir <- "/project/imoskowitz/xyang2/chrislowzhengxi/results/ucell/Lateral_plate_mesoderm/qc/histograms"
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

score_col <- "SHH_UCell_score"
label_col <- "celltype_new"

safe <- function(x) gsub("[^A-Za-z0-9._-]+","_", x)

write_hist <- function(vals, title_str, file_stem, out_dir) {
  png(file.path(out_dir, paste0(file_stem, ".png")), width=1200, height=800, res=150)
  hist(vals, breaks=100, xlim=c(0,1), main=title_str, xlab=score_col, ylab="Count")
  dev.off()
  pdf(file.path(out_dir, paste0(file_stem, ".pdf")), width=8, height=5.5)
  hist(vals, breaks=100, xlim=c(0,1), main=title_str, xlab=score_col, ylab="Count")
  dev.off()
}

# summary CSV for zero fractions
summ <- data.frame(label=character(), n=integer(), n_zero=integer(), frac_zero=double(), stringsAsFactors=FALSE)

# 1) per celltype_new
for (lab in sort(unique(obs[[label_col]]))) {
  sub <- obs[obs[[label_col]] == lab, , drop=FALSE]
  if (!nrow(sub)) next
  vals <- sub[[score_col]]
  n_all  <- length(vals)
  n_zero <- sum(vals == 0, na.rm=TRUE)
  nz     <- vals[vals > 0]
  frac0  <- if (n_all > 0) n_zero / n_all else NA_real_

  # save both
  write_hist(vals, sprintf("LPM — %s (all, n=%d, zero=%.1f%%)", lab, n_all, 100*frac0),
             paste0("hist_LPM_", safe(lab), "_ALL"), out_dir)
  if (length(nz) > 0) {
    write_hist(nz,  sprintf("LPM — %s (no-zeros, n=%d)", lab, length(nz)),
               paste0("hist_LPM_", safe(lab), "_nozeros"), out_dir)
  }

  summ <- rbind(summ, data.frame(label=lab, n=n_all, n_zero=n_zero, frac_zero=frac0, stringsAsFactors=FALSE))
}

# 2) overall
vals_all <- obs[[score_col]]
n_all  <- length(vals_all)
n_zero <- sum(vals_all == 0, na.rm=TRUE)
nz_all <- vals_all[vals_all > 0]
frac0  <- if (n_all > 0) n_zero / n_all else NA_real_

write_hist(vals_all, sprintf("LPM — All cells (all, n=%d, zero=%.1f%%)", n_all, 100*frac0),
           "hist_LPM_ALL", out_dir)
if (length(nz_all) > 0) {
  write_hist(nz_all,  sprintf("LPM — All cells (no-zeros, n=%d)", length(nz_all)),
             "hist_LPM_ALL_nozeros", out_dir)
}

# save zero-fraction summary
write.csv(summ, file.path(out_dir, "LPM_zero_fraction_summary.csv"), row.names=FALSE)
