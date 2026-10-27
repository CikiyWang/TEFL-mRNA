# The y axis is broken between 0.78 and 2.45 as in the published panel: two stacked panels cut with
# coord_cartesian, so violins, boxes and outliers are computed on all mRNAs and only the window
# differs. mRNAs with log10(TE+1) inside the break are not visible; the script prints how many.
suppressPackageStartupMessages({
  library(ggplot2)
  library(patchwork)
})

file_arg <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
plot_dir <- if (length(file_arg)) dirname(normalizePath(sub("^--file=", "", file_arg))) else file.path(getwd(), "plot")
output_dir <- file.path(dirname(plot_dir), "output")

STYLE <- list(
  dev = "pdf", dpi = 300, font = "Arial",
  w = 4.6, h = 5.0, aspect = NULL,
  base_size = 18, title_size = 18, subtitle_size = 16, axis_title = 18, axis_text = 15, legend_text = 14,
  low  = c(0, 0.78),  low_breaks  = c(0, 0.25, 0.5, 0.75),
  high = c(2.45, 3.3), high_breaks = c(2.5, 2.75, 3),
  heights = c(1, 3.4),
  # ColorBrewer Paired
  box_cols = c("#A6CEE3", "#1F78B4", "#B2DF8A", "#33A02C", "#FB9A99", "#E31A1C", "#FDBF6F", "#FF7F00", "#CAB2D6", "#6A3D9A")
)
source(file.path(dirname(dirname(plot_dir)), "shared", "scripts", "fig_style.R"))

d <- read.delim(file.path(output_dir, "Fig1E_source_data.tsv"), stringsAsFactors = FALSE)
med <- read.delim(file.path(output_dir, "Fig1E_bin_medians.tsv"), stringsAsFactors = FALSE)
stats <- read.delim(file.path(output_dir, "Fig1E_stats.tsv"))
edges <- seq(0, 1, 0.1)
lab <- sprintf("%s–%s", sub("^0.0$", "0", format(edges[-11], nsmall = 1)), sub("^1.0$", "1", format(edges[-1], nsmall = 1)))
d$bin <- factor(d$bin, levels = lab)
med$bin <- factor(med$bin, levels = lab)

hidden <- sum(d$log10_TE_plus1 > STYLE$low[2] & d$log10_TE_plus1 < STYLE$high[1])
cat(sprintf("axis break %.2f -> %.2f hides %d of %d mRNAs (%.1f%%); %d are above the break\n", STYLE$low[2], STYLE$high[1],
            hidden, nrow(d), 100 * hidden / nrow(d), sum(d$log10_TE_plus1 >= STYLE$high[1])))

rho_label <- bquote("Spearman" ~ rho == .(sprintf("%.2f", stats$spearman_rho)) * "," ~~ italic(n) == .(format(stats$n, big.mark = ",")))
two_dec <- function(b) formatC(b, format = "f", digits = 2)

layers <- function(family) list(
  geom_violin(fill = "white", colour = "grey20", linewidth = 0.4, scale = "width", width = 0.85),
  geom_boxplot(aes(fill = bin), width = 0.26, linewidth = 0.4, outlier.shape = 15, outlier.size = 1.1, outlier.colour = "black"),
  scale_fill_manual(values = setNames(STYLE$box_cols, levels(d$bin)), guide = "none"),
  scale_x_discrete(drop = FALSE),
  theme_pub(family)
)

make_plot <- function(family) {
  x_text <- element_text(angle = 60, hjust = 1, vjust = 1, size = STYLE$axis_text - 2, colour = "black")
  base <- ggplot(d, aes(bin, log10_TE_plus1)) + layers(family)
  upper <- base + coord_cartesian(ylim = STYLE$high) +
    scale_y_continuous(breaks = STYLE$high_breaks, labels = two_dec) +
    labs(x = NULL, y = "log10(TE+1)", subtitle = rho_label) +
    theme(axis.text.x = element_blank(), axis.ticks.x = element_blank(), axis.line.x = element_blank(),
          plot.margin = margin(8, 14, 3, 8))
  lower <- base + geom_line(data = med, aes(group = 1), linewidth = 1) +
    coord_cartesian(ylim = STYLE$low) +
    scale_y_continuous(breaks = STYLE$low_breaks, labels = two_dec, expand = expansion(c(0.02, 0))) +
    labs(x = "pTE categories", y = "log10(TE+1)") +
    theme(axis.text.x = x_text, plot.margin = margin(3, 14, 6, 8))
  upper / lower + plot_layout(heights = STYLE$heights, axis_titles = "collect")
}
save_fig(file.path(output_dir, "Fig1E"), make_plot)
