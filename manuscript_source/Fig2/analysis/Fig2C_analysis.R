#!/usr/bin/env Rscript
# Fig. 2C - predicted 5'UTR variant effect size vs gnomAD allele frequency.
# One observation per SNV: delta pTE is averaged across the RefSeq isoforms the
# SNV falls in, and |mean delta pTE| is regressed on log10(AF).
#
# Emits the regression statistics and the plotting-ready data consumed by
# plot/Fig2C_plot.R. The input is part of the data archive (see README.md).

suppressMessages({ library(data.table); library(hexbin) })

# --- portable paths -------------------------------------------------------
args     <- commandArgs(trailingOnly = FALSE)
file_arg <- grep("^--file=", args, value = TRUE)
SCRIPT   <- if (length(file_arg)) normalizePath(sub("^--file=", "", file_arg[1])) else NA
BASE_DIR <- if (!is.na(SCRIPT)) dirname(dirname(SCRIPT)) else normalizePath(getwd())
SHARED_DIR <- file.path(dirname(BASE_DIR), "shared", "input")
OUTPUT_DIR <- file.path(BASE_DIR, "output")
dir.create(OUTPUT_DIR, showWarnings = FALSE, recursive = TRUE)

IN_FILE <- file.path(SHARED_DIR, "gnomAD_v4_5UTR_SNVs_dpTE.tsv.gz")

# --- parameters -----------------------------------------------------------
COL      <- list(af = "gnomade_AF", delta = "delta_score_high",
                 snv = "variant_id")
AF_FLOOR <- 1e-6     # allele frequencies are floored before log10
HEX_BINS <- 60       # must match the value used in plot/Fig2C_plot.R

# --- load -----------------------------------------------------------------
need <- unname(unlist(COL))
hdr  <- names(fread(IN_FILE, nrows = 0))
if (!all(need %in% hdr))
  stop("missing column(s): ", paste(setdiff(need, hdr), collapse = ", "))

d <- fread(IN_FILE, select = need)
setnames(d, need, c("AF", "delta", "snv"))
d[, AF := as.numeric(AF)][, delta := as.numeric(delta)]
d <- d[is.finite(AF) & is.finite(delta) & AF > 0]
n_pairs <- nrow(d)
# one point per SNV; its AF is written at different precision on different
# isoform rows, so the first row's value is used
d <- d[, .(AF = AF[1], delta = mean(delta)), by = snv]
d[, abs_delta := abs(delta)]
d[, log10AF   := log10(pmax(AF, AF_FLOOR))]
N <- nrow(d)
cat(sprintf("%s SNVs from %s transcript-variant pairs\n",
            format(N, big.mark = ","), format(n_pairs, big.mark = ",")))

# --- regression of |dpTE| on log10(AF) ------------------------------------
fit <- lm(abs_delta ~ log10AF, data = d); sf <- summary(fit)
stats_tab <- data.table(
  n          = N,
  slope      = coef(fit)[["log10AF"]],
  intercept  = coef(fit)[["(Intercept)"]],
  slope_lo   = confint(fit, "log10AF")[1],
  slope_hi   = confint(fit, "log10AF")[2],
  r_squared  = sf$r.squared,
  P_slope    = sf$coefficients["log10AF", 4],
  spearman_rho = suppressWarnings(cor(d$log10AF, d$abs_delta,
                                      method = "spearman")))
fwrite(stats_tab, file.path(OUTPUT_DIR, "Fig2C_source_data_stats.tsv"),
       sep = "\t")
print(stats_tab)

# --- hexagonal binning of the scatter, as source data ---------------------
# Cell counts are the quantity actually rendered in the panel; exported here so
# the plotted surface is inspectable without the full variant table.
hb  <- hexbin(d$log10AF, d$abs_delta, xbins = HEX_BINS)
xy  <- hcell2xy(hb)
fwrite(data.table(log10AF = xy$x, abs_delta = xy$y, count = hb@count),
       file.path(OUTPUT_DIR, "Fig2C_source_data_hexbin.tsv"), sep = "\t")

# --- plotting-ready data --------------------------------------------------
# geom_hex() re-bins from the observations, so the two columns it needs are
# written in full.
saveRDS(d[, .(log10AF, abs_delta)],
        file.path(OUTPUT_DIR, "Fig2C_plotdata.rds"), compress = "xz")

cat(sprintf("wrote Fig2C_source_data_stats.tsv, Fig2C_source_data_hexbin.tsv (%d cells), Fig2C_plotdata.rds\n",
            length(hb@count)))
