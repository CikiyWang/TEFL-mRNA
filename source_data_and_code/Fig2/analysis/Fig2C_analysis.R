#!/usr/bin/env Rscript
# Fig. 2C - predicted 5'UTR variant effect size vs gnomAD allele frequency.
# Transcript-variant resolution: one observation per transcript-variant pair,
# no collapsing across isoforms.
#
# Emits the regression statistics and the plotting-ready data consumed by
# plot/Fig2C_plot.R.

suppressMessages({ library(data.table); library(hexbin) })

# --- portable paths -------------------------------------------------------
args     <- commandArgs(trailingOnly = FALSE)
file_arg <- grep("^--file=", args, value = TRUE)
SCRIPT   <- if (length(file_arg)) normalizePath(sub("^--file=", "", file_arg[1])) else NA
BASE_DIR <- if (!is.na(SCRIPT)) dirname(dirname(SCRIPT)) else normalizePath(getwd())
INPUT_DIR  <- file.path(BASE_DIR, "input")
OUTPUT_DIR <- file.path(BASE_DIR, "output")
dir.create(OUTPUT_DIR, showWarnings = FALSE, recursive = TRUE)

IN_FILE <- file.path(INPUT_DIR, paste0(
  "Pred_All.gnomad.exomes.v4.0.sites.PASS_pc_SNV_NM.5_prime_UTR_variant.",
  "filter_cor0.7_l0.25_h1.TPE_opt.delta.filter.sel_cols.v4.tsv"))

# --- parameters -----------------------------------------------------------
COL      <- list(af = "gnomade_AF", delta = "delta_score_high",
                 tid = "transcript_refid_var")
AF_FLOOR <- 1e-6     # allele frequencies are floored before log10
HEX_BINS <- 60       # must match the value used in plot/Fig2C_plot.R

# --- load -----------------------------------------------------------------
need <- unname(unlist(COL))
hdr  <- names(fread(IN_FILE, nrows = 0))
if (!all(need %in% hdr))
  stop("missing column(s): ", paste(setdiff(need, hdr), collapse = ", "))

d <- fread(IN_FILE, select = need)
setnames(d, need, c("AF", "delta", "key"))
d[, AF := as.numeric(AF)][, delta := as.numeric(delta)]
d <- d[is.finite(AF) & is.finite(delta) & AF > 0]
d[, abs_delta := abs(delta)]
d[, log10AF   := log10(pmax(AF, AF_FLOOR))]
N <- nrow(d)
cat(sprintf("%s transcript-variant pairs (%s distinct keys)\n",
            format(N, big.mark = ","), format(uniqueN(d$key), big.mark = ",")))

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
# written in full; this keeps the rendered panel identical to the original.
saveRDS(d[, .(log10AF, abs_delta)],
        file.path(OUTPUT_DIR, "Fig2C_plotdata.rds"), compress = "xz")

cat(sprintf("wrote Fig2C_source_data_stats.tsv, Fig2C_source_data_hexbin.tsv (%d cells), Fig2C_plotdata.rds\n",
            length(hb@count)))
