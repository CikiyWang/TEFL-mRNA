#!/usr/bin/env Rscript
# Fig. EV4A - per-class bar plot of cumulative dPTE tail fractions in dbSNP common variants.

suppressPackageStartupMessages({
  library(data.table); library(ggplot2); library(scales)
})

args <- commandArgs(trailingOnly = FALSE)
sf <- sub("^--file=", "", args[grep("^--file=", args)])
BASE_DIR <- if (length(sf)) normalizePath(file.path(dirname(sf), "..")) else normalizePath("..")
OUTPUT_DIR <- file.path(BASE_DIR, "output")

CLS  <- c("5'UTR", "Synonymous", "Missense", "3'UTR")
COL  <- c("5'UTR" = "#B2182B", "Synonymous" = "#D1761C",
          "Missense" = "#6A3D9A", "3'UTR" = "#2B6CA3")
LABS <- c("< -0.2", "< -0.1", "< -0.05", "< -0.02", "< -0.01",
          "> 0.01", "> 0.02", "> 0.05", "> 0.1", "> 0.2")
FONT <- if (nrow(systemfonts::system_fonts()[
              systemfonts::system_fonts()$family == "Arial", ]) > 0) "Arial" else "sans"

A <- fread(file.path(OUTPUT_DIR, "FigEV4A_source_data.tsv"))
A[, `:=`(class = factor(class, CLS), bin = factor(bin, LABS))]
A[, lab := ifelse(percent >= 0.01, sprintf("%.2f%%", percent), sprintf("%.3f%%", percent))]

th_p <- function(family, base = 18) theme_classic(base_size = base, base_family = family) +
  theme(axis.text  = element_text(colour = "black", size = base - 1),
        axis.title = element_text(colour = "black", size = base + 1),
        axis.ticks = element_line(colour = "black"),
        strip.background = element_blank(),
        strip.text = element_text(size = base, face = "bold"),
        panel.grid = element_blank(), legend.position = "none")

build <- function(family) ggplot(A[n_beyond > 0], aes(bin, percent, fill = class)) +
  geom_col(width = 0.78, colour = "grey25", linewidth = 0.3) +
  geom_text(aes(label = lab), angle = 90, hjust = -0.12, size = 4.0, family = family) +
  facet_wrap(~class, nrow = 1, scales = "free") +
  scale_fill_manual(values = COL) +
  scale_y_continuous(expand = expansion(mult = c(0, 0.42)),
                     labels = label_number(drop0trailing = TRUE)) +
  labs(x = expression(paste("p", TE[ALT], " - p", TE[REF])),
       y = "Percent of common\nvariants (%)") +
  th_p(family) + theme(axis.text.x = element_text(angle = 45, hjust = 1, size = 13))

# cairo_pdf needs a loadable cairo library (on macOS, XQuartz), so it is tested by drawing with
# the font; otherwise the quartz PDF device (macOS, also embeds Arial), else pdf() with Helvetica
pdf_ok <- function(dev) isTRUE(tryCatch({
  f <- tempfile(fileext = ".pdf"); dev(f, width = 2, height = 2)
  grid::grid.text("a", gp = grid::gpar(fontfamily = FONT)); invisible(grDevices::dev.off()); TRUE
}, error = function(e) { try(grDevices::dev.off(), silent = TRUE); FALSE },
   warning = function(w) { try(grDevices::dev.off(), silent = TRUE); FALSE }))
quartz_pdf <- function(filename, width, height, ...) grDevices::quartz(file = filename, type = "pdf", width = width, height = height)
if (pdf_ok(grDevices::cairo_pdf)) {
  pdf_dev <- grDevices::cairo_pdf; pdf_family <- FONT
} else if (isTRUE(capabilities("aqua")) && pdf_ok(quartz_pdf)) {
  pdf_dev <- quartz_pdf; pdf_family <- FONT
} else {
  pdf_dev <- grDevices::pdf; pdf_family <- "Helvetica"
}

ggsave(file.path(OUTPUT_DIR, "FigEV4A.pdf"), build(pdf_family), width = 12.6, height = 4.6, device = pdf_dev)
ggsave(file.path(OUTPUT_DIR, "FigEV4A.png"), build(FONT), width = 12.6, height = 4.6, dpi = 300, bg = "white")
