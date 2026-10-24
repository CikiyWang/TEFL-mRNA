#!/usr/bin/env Rscript
# Fig. S2C: mean LCL TE versus RiboNN LCL TE, high-confidence label set.
suppressMessages(library(ggplot2))

# resolve FigS2/ from this script's location so the package runs from any directory
BASE_DIR <- local({
  a <- commandArgs(trailingOnly = FALSE); f <- sub("^--file=", "", a[grepl("^--file=", a)])
  if (length(f)) dirname(dirname(normalizePath(f))) else normalizePath(".")
})
INPUT  <- file.path(BASE_DIR, "input")
OUTPUT <- file.path(BASE_DIR, "output")

cmp   <- read.csv(file.path(OUTPUT, "FigS2C_LCL_TE_vs_RiboNN.csv"))
r_pub <- with(cmp[cmp$in_published, ], cor(our, TE_LCL, method = "spearman"))
cmp   <- cmp[cmp$in_published, ]                # only high-confidence transcripts are plotted

p <- ggplot(cmp, aes(x = our, y = TE_LCL)) +
  geom_point(size = 1.15, alpha = 0.85, stroke = 0, colour = "#2F6DB5") +
  geom_smooth(method = "lm", formula = y ~ x, se = FALSE,
              colour = "#1F4E88", linewidth = 0.9) +
  scale_x_log10(breaks = c(1e-3, 1e-2, 1e-1, 1, 10, 100),
                labels = c("0.001", "0.01", "0.1", "1", "10", "100")) +
  labs(x = "Mean LCL TE (n = 47)", y = "RiboNN LCL TE (log)",
       subtitle = sprintf("rho = %.2f  (n = %s)", r_pub, format(nrow(cmp), big.mark = ","))) +
  theme_classic(base_size = 20) +
  theme(aspect.ratio = 1,
        plot.subtitle = element_text(size = 17, colour = "grey20", lineheight = 1.15,
                                     margin = margin(b = 8)),
        axis.text  = element_text(colour = "black", size = 19),
        axis.title = element_text(size = 20),
        legend.position = "none",
        plot.margin = margin(6, 10, 2, 6))

ggsave(file.path(OUTPUT, "FigS2C.pdf"), p, width = 4.7, height = 5.0, device = cairo_pdf)
