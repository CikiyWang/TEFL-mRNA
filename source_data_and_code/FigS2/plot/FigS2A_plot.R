#!/usr/bin/env Rscript
# Fig. S2A: correlation heatmap.
suppressMessages(library(ggplot2))

# resolve FigS2/ from this script's location so the package runs from any directory
BASE_DIR <- local({
  a <- commandArgs(trailingOnly = FALSE); f <- sub("^--file=", "", a[grepl("^--file=", a)])
  if (length(f)) dirname(dirname(normalizePath(f))) else normalizePath(".")
})
INPUT  <- file.path(BASE_DIR, "input")
OUTPUT <- file.path(BASE_DIR, "output")

cm <- as.matrix(read.csv(file.path(OUTPUT, "FigS2A_pairwise_spearman_matrix.csv"),
                         row.names = 1, check.names = FALSE))
pw <- cm[upper.tri(cm)]
lv <- colnames(cm)                              # clustered order from the analysis step

cm_plot <- cm; diag(cm_plot) <- NA               # self-correlations masked so the colour
                                                 # scale spans only real sample pairs
long <- data.frame(
  a   = factor(rep(colnames(cm), times = ncol(cm)), levels = lv),
  b   = factor(rep(colnames(cm), each  = ncol(cm)), levels = lv),
  rho = as.vector(cm_plot))

p <- ggplot(long, aes(x = a, y = b, fill = rho)) +
  geom_raster() +
  scale_fill_gradientn(
    colours = c("#F7FBFF", "#C6DBEF", "#6BAED6", "#2171B5", "#08306B"),
    limits  = range(pw), name = "Spearman\nrho",
    breaks  = c(0.60, 0.65, 0.70, 0.75, 0.80), na.value = "grey88") +
  coord_fixed(expand = FALSE) +
  labs(x = NULL, y = NULL,
       subtitle = sprintf("Median pairwise rho = %.2f (range %.2f-%.2f, %s pairs)",
                          median(pw), min(pw), max(pw), format(length(pw), big.mark = ","))) +
  theme_classic(base_size = 15) +
  theme(plot.subtitle = element_text(size = 14, colour = "grey20", margin = margin(b = 6)),
        axis.text.x = element_text(size = 7, angle = 90, hjust = 1, vjust = 0.5, colour = "black"),
        axis.text.y = element_text(size = 7, colour = "black"),
        axis.ticks  = element_line(linewidth = 0.25),
        axis.line   = element_blank(),
        legend.title = element_text(size = 13, lineheight = 1.05),
        legend.text  = element_text(size = 12),
        legend.key.height = unit(1.05, "cm"),
        legend.key.width  = unit(0.42, "cm"),
        plot.margin = margin(6, 6, 2, 4))

ggsave(file.path(OUTPUT, "FigS2A.pdf"), p, width = 6.2, height = 5.6, device = cairo_pdf)
