#!/usr/bin/env Rscript
# Fig. S2B: distribution of per-transcript TE-class agreement across individuals.
suppressMessages(library(ggplot2))

# resolve FigS2/ from this script's location so the package runs from any directory
BASE_DIR <- local({
  a <- commandArgs(trailingOnly = FALSE); f <- sub("^--file=", "", a[grepl("^--file=", a)])
  if (length(f)) dirname(dirname(normalizePath(f))) else normalizePath(".")
})
INPUT  <- file.path(BASE_DIR, "input")
OUTPUT <- file.path(BASE_DIR, "output")

pal <- c("All same TE label"  = "#2F6DB5",
         "TE-low <-> TE-high" = "#C0504D",
         "Others"             = "#9FBFE0")

d <- read.csv(file.path(OUTPUT, "FigS2B_per_transcript_agreement.csv"))
d$variability <- factor(d$variability, levels = names(pal))

p <- ggplot(d, aes(x = 100 * agree, fill = variability)) +
  geom_histogram(binwidth = 100 / 47, colour = "white", linewidth = 0.12) +
  scale_fill_manual(values = pal, name = NULL, drop = FALSE) +
  scale_x_continuous(breaks = seq(0, 100, 20), labels = function(x) paste0(x, "%")) +
  scale_y_continuous(expand = expansion(mult = c(0, 0.06))) +
  labs(x = "Individuals sharing the top TE label", y = "Number of transcripts") +
  theme_classic(base_size = 16) +
  theme(axis.text  = element_text(colour = "black", size = 15),
        axis.title = element_text(size = 16),
        legend.position = "bottom",
        legend.key.size = unit(0.42, "cm"),
        legend.text = element_text(size = 14),
        legend.margin = margin(0, 0, 0, 0),
        plot.margin = margin(6, 14, 6, 6))

# every transcript must enter the histogram
stopifnot(sum(ggplot_build(p)$data[[1]]$count) == nrow(d))

ggsave(file.path(OUTPUT, "FigS2B.pdf"), p, width = 5.6, height = 4.6, device = cairo_pdf)
