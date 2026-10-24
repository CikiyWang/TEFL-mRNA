#!/usr/bin/env Rscript
# Fig. S2B: consistency of the TE class assigned to each transcript across the
# 47 individuals. Classes use the thresholds of the published label definition:
#   TE-low 0 < TE < 0.25 | TE-intermediate 0.25 <= TE <= 1 | TE-high TE > 1
# TE == 0 is left unclassified, as in the published pipeline, but still counts
# in the denominator of measured individuals.

# resolve FigS2/ from this script's location so the package runs from any directory
BASE_DIR <- local({
  a <- commandArgs(trailingOnly = FALSE); f <- sub("^--file=", "", a[grepl("^--file=", a)])
  if (length(f)) dirname(dirname(normalizePath(f))) else normalizePath(".")
})
INPUT  <- file.path(BASE_DIR, "input")
OUTPUT <- file.path(BASE_DIR, "output")

lo_cut <- 0.25; hi_cut <- 1

te  <- as.matrix(read.delim(file.path(INPUT, "NM_isoform_filter_na0.5_count10_cor0.7_na5.TE.txt"),
                            row.names = 1, check.names = FALSE))
cls <- ifelse(is.na(te), NA_integer_, ifelse(te < lo_cut, 0L, ifelse(te > hi_cut, 2L, 1L)))
cls[!is.na(te) & te == 0] <- 3L                 # measured but unclassified

n_valid <- rowSums(!is.na(cls))                 # all measured individuals
tab <- t(apply(cls, 1, function(v) tabulate(factor(v, levels = c(0, 1, 2)), nbins = 3)))
colnames(tab) <- c("n_low", "n_int", "n_high")

d <- data.frame(seq_id = rownames(te), n_indiv = n_valid, tab,
                top_label = c("TE-low", "TE-intermediate", "TE-high")[apply(tab, 1, which.max)],
                agree = apply(tab, 1, max) / n_valid)
d$variability <- ifelse(d$agree == 1, "All same TE label",
                 ifelse(tab[, "n_low"] > 0 & tab[, "n_high"] > 0, "TE-low <-> TE-high", "Others"))

write.csv(d, file.path(OUTPUT, "FigS2B_per_transcript_agreement.csv"), row.names = FALSE)

cat(sprintf("transcripts %d | TE==0 %.1f%% of measured values | median agreement %.1f%%\n",
            nrow(d), 100 * sum(te == 0, na.rm = TRUE) / sum(!is.na(te)), 100 * median(d$agree)))
cat(sprintf("same class in all individuals %.1f%% (n=%d) | >=75%% %.1f%% (n=%d) | >=90%% %.1f%% (n=%d)\n",
            100 * mean(d$agree == 1), sum(d$agree == 1),
            100 * mean(d$agree >= 0.75), sum(d$agree >= 0.75),
            100 * mean(d$agree >= 0.90), sum(d$agree >= 0.90)))
