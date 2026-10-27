#!/usr/bin/env Rscript
# Fig. 2C - renders the panel from the output of analysis/Fig2C_analysis.R.
# Hexbin density of |dpTE| against gnomAD allele frequency with the fitted
# regression line, a dashed line at AF = 0.01 and the fit statistics.

suppressMessages({
  library(data.table); library(ggplot2); library(ggthemes); library(hexbin)
})

# --- portable paths -------------------------------------------------------
args     <- commandArgs(trailingOnly = FALSE)
file_arg <- grep("^--file=", args, value = TRUE)
SCRIPT   <- if (length(file_arg)) normalizePath(sub("^--file=", "", file_arg[1])) else NA
BASE_DIR <- if (!is.na(SCRIPT)) dirname(dirname(SCRIPT)) else normalizePath(getwd())
OUTPUT_DIR <- file.path(BASE_DIR, "output")
source(file.path(dirname(BASE_DIR), "shared", "scripts", "pdf_device.R"))

plotdata <- file.path(OUTPUT_DIR, "Fig2C_plotdata.rds")
statsfile <- file.path(OUTPUT_DIR, "Fig2C_source_data_stats.tsv")
if (!file.exists(plotdata) || !file.exists(statsfile))
  stop("run analysis/Fig2C_analysis.R first")

d  <- as.data.table(readRDS(plotdata))
st <- fread(statsfile)

# --- style ----------------------------------------------------------------
STYLE <- list(font = "Arial", base_size = 44, tick_size = 40, annot_size = 12.0,
              aspect = 1, line_col = "#B2182B", hex_bins = 60)

theme_pub <- function(s = STYLE) {
  theme_few(base_family = s$font, base_size = s$base_size) +
    theme(panel.background = element_rect(fill = NA, colour = NA),
          aspect.ratio = s$aspect,
          axis.text  = element_text(size = s$tick_size, colour = "black"),
          axis.title = element_text(size = s$base_size, colour = "black"),
          legend.position = "none",
          plot.margin  = margin(6, 10, 6, 6))
}

# statistics in the "m x 10^e" form of the published panel, one line each
sci <- function(x, digits) {
  e <- floor(log10(abs(x))); m <- formatC(x / 10^e, format = "f", digits = digits)
  sprintf('"%s" ~ "x" ~ 10^%d', m, e)
}
stat_labels <- c(paste("slope ==", sci(st$slope, 2)), paste("R^2 ==", sci(st$r_squared, 1)),
                 sprintf('rho == "%.3f"', st$spearman_rho))

# --- panel ----------------------------------------------------------------
# the line is predicted from the fitted coefficients rather than refitted here
xg      <- seq(min(d$log10AF), max(d$log10AF), length.out = 100)
fitline <- data.table(x = xg, y = st$intercept + st$slope * xg)

p <- ggplot(d, aes(log10AF, abs_delta)) +
  geom_hex(bins = STYLE$hex_bins) +
  scale_fill_gradientn(colours = rev(hcl.colors(9, "Inferno")), trans = "log10") +
  geom_vline(xintercept = -2, linetype = "dashed", colour = "grey60", linewidth = 1.2) +
  geom_line(data = fitline, aes(x, y), colour = STYLE$line_col,
            linewidth = 1.2, inherit.aes = FALSE) +
  annotate("text", x = -3.2, y = max(d$abs_delta) - 0.045 * (seq_along(stat_labels) - 1), label = stat_labels,
           parse = TRUE, hjust = 0, vjust = 1, size = STYLE$annot_size, family = STYLE$font) +
  scale_x_continuous(breaks = c(-6, -4, -2, 0),
                     labels = c(expression(10^-6), expression(10^-4),
                                expression(10^-2), expression(1))) +
  labs(x = "gnomAD allele frequency", y = expression("|"*Delta*"pTE|")) +
  theme_pub()

ggsave(file.path(OUTPUT_DIR, "Fig2C.pdf"), p, width = 14, height = 13.5,
       device = pdf_device())
cat("wrote output/Fig2C.pdf\n")
