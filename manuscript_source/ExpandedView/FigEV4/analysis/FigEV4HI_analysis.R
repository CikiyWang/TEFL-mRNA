#!/usr/bin/env Rscript
# Figure EV4H/I analysis: effect of 5'UTR architecture on the predicted
# translational-efficiency change of common 5'UTR variants.
# Writes per-variant source data, binned summaries and Spearman statistics.

suppressMessages(library(data.table))
source(file.path(dirname(sub("^--file=", "", commandArgs(FALSE)[grep("^--file=", commandArgs(FALSE))])), "common.R"))

cov <- fread(file.path(INPUT_DIR, "dbSNP_common_5UTR_variant_covariates.tsv"), showProgress = FALSE)
dl  <- fread(file.path(INPUT_DIR,
       "dbSNP_common_5UTR_SNVs_dpTE_no_ClinVar_GWAS.tsv"),
       select = c("transcript_refid_var", "ref", "alt", "pred_0_all", "delta_score_high"),
       showProgress = FALSE)

u5 <- merge(dl[ref != alt], cov, by = "transcript_refid_var")
u5[, `:=`(delta_pTE = delta_score_high, abs_delta_pTE = abs(delta_score_high))]

src <- rbindlist(lapply(names(PANELS), function(p) data.table(
  panel = p, covariate = PANELS[[p]]$var,
  transcript_refid_var = u5$transcript_refid_var,
  x = u5[[PANELS[[p]]$var]], delta_pTE = u5$delta_pTE, abs_delta_pTE = u5$abs_delta_pTE)))

# Spearman is taken against the absolute effect size: the covariates act on the
# spread of delta pTE, not on its centre.
stats <- src[, .(spearman_rho_vs_abs_delta_pTE = cor(x, abs_delta_pTE, method = "spearman"),
                 n = .N), by = .(panel, covariate)]

# Binned summary: median and 5-95% range of signed delta pTE in equal-count x bins
trend <- src[, {
  br <- unique(quantile(x, seq(0, 1, length.out = N_TREND + 1)))
  b  <- cut(x, br, include.lowest = TRUE, labels = FALSE)
  .(x  = as.numeric(tapply(x, b, median)),
    y  = as.numeric(tapply(delta_pTE, b, median)),
    lo = as.numeric(tapply(delta_pTE, b, function(v) quantile(v, .05, names = FALSE))),
    hi = as.numeric(tapply(delta_pTE, b, function(v) quantile(v, .95, names = FALSE))),
    nb = as.integer(tapply(x, b, length)))
}, by = panel][nb >= MIN_BIN]

fwrite(src,   file.path(OUTPUT_DIR, "FigEV4HI_source_data.tsv"), sep = "\t")
fwrite(trend, file.path(OUTPUT_DIR, "FigEV4HI_binned_summary.tsv"), sep = "\t")
fwrite(stats, file.path(OUTPUT_DIR, "FigEV4HI_statistics.csv"))
