#!/usr/bin/env Rscript
# Fig. 2C - renders the panel from the output of analysis/Fig2C_analysis.R.
# Hexbin density of |dpTE| against gnomAD allele frequency with the fitted
# regression line; statistics are placed beside the data rather than over it.

suppressMessages({
  library(data.table); library(ggplot2); library(ggthemes)
  library(patchwork);  library(scales);  library(hexbin)
})

# --- portable paths -------------------------------------------------------
args     <- commandArgs(trailingOnly = FALSE)
file_arg <- grep("^--file=", args, value = TRUE)
SCRIPT   <- if (length(file_arg)) normalizePath(sub("^--file=", "", file_arg[1])) else NA
BASE_DIR <- if (!is.na(SCRIPT)) dirname(dirname(SCRIPT)) else normalizePath(getwd())
OUTPUT_DIR <- file.path(BASE_DIR, "output")

plotdata <- file.path(OUTPUT_DIR, "Fig2C_plotdata.rds")
statsfile <- file.path(OUTPUT_DIR, "Fig2C_source_data_stats.tsv")
if (!file.exists(plotdata) || !file.exists(statsfile))
  stop("run analysis/Fig2C_analysis.R first")

d  <- as.data.table(readRDS(plotdata))
st <- fread(statsfile)

# --- style ----------------------------------------------------------------
STYLE <- list(font = "Arial", base_size = 44, tick_size = 40, annot_size = 12.0,
              aspect = 1, dpi = 400, line_col = "#B2182B", hex_bins = 60)

theme_pub <- function(s = STYLE) {
  theme_few(base_family = s$font, base_size = s$base_size) +
    theme(panel.background = element_rect(fill = NA, colour = NA),
          aspect.ratio = s$aspect,
          axis.text  = element_text(size = s$tick_size, colour = "black"),
          axis.title = element_text(size = s$base_size, colour = "black"),
          plot.title = element_text(size = s$base_size, colour = "black",
                                    hjust = 0, face = "plain"),
          legend.text  = element_text(size = s$tick_size - 3, family = s$font),
          legend.title = element_blank(),
          plot.margin  = margin(6, 10, 6, 6))
}

# --- panel ----------------------------------------------------------------
# the line is predicted from the fitted coefficients rather than refitted here
xg      <- seq(min(d$log10AF), max(d$log10AF), length.out = 100)
fitline <- data.table(x = xg, y = st$intercept + st$slope * xg)

p_a <- ggplot(d, aes(log10AF, abs_delta)) +
  geom_hex(bins = STYLE$hex_bins) +
  scale_fill_gradientn(colours = rev(hcl.colors(9, "Inferno")),
                       trans = "log10", name = "variants",
                       labels = label_log()) +
  geom_line(data = fitline, aes(x, y), colour = STYLE$line_col,
            linewidth = 1.2, inherit.aes = FALSE) +
  scale_x_continuous(breaks = c(-6, -4, -2, 0),
                     labels = c(expression(10^-6), expression(10^-4),
                                expression(10^-2), expression(1))) +
  labs(x = "gnomAD allele frequency", y = expression("|"*Delta*"pTE|"),
       title = expression("No meaningful AF dependence ("*R^2 < 0.001*")")) +
  theme_pub() +
  theme(legend.position = "right",
        legend.key.height = unit(1.05, "cm"),
        legend.key.width  = unit(0.45, "cm"),
        legend.title = element_text(size = STYLE$tick_size - 3,
                                    family = STYLE$font))

stat_lines <- c(sprintf("n = %s", format(st$n, big.mark = ",")),
                "transcript-variant", "pairs", "",
                sprintf("slope = %.2e", st$slope),
                sprintf("R2 = %.1e", st$r_squared),
                sprintf("rho = %.3f", st$spearman_rho),
                "P < 2.2e-16")
p_stats <- ggplot() +
  geom_text(data = data.table(y = rev(seq_along(stat_lines)), lab = stat_lines),
            aes(0, y, label = lab), hjust = 0, size = STYLE$annot_size,
            family = STYLE$font) +
  scale_x_continuous(limits = c(-0.05, 2.6)) +
  scale_y_continuous(limits = c(0.2, length(stat_lines) + 1.0)) +
  theme_void() +
  theme(plot.margin = margin(6, 4, 6, 0))

fig <- (p_a | p_stats) + plot_layout(widths = c(1.55, 0.62)) &
  theme(text = element_text(family = STYLE$font))

ggsave(file.path(OUTPUT_DIR, "Fig2C.pdf"), fig, width = 19.5, height = 13.5,
       device = cairo_pdf)
ggsave(file.path(OUTPUT_DIR, "Fig2C.png"), fig, width = 19.5, height = 13.5,
       dpi = STYLE$dpi, bg = "white")
cat("wrote output/Fig2C.pdf and output/Fig2C.png\n")
