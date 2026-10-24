#!/usr/bin/env Rscript
# Fig. S5C: TEFL-mRNA (Reg) vs RiboNN transfer predicted TE, one panel per cell line; points
# coloured by density, dashed y = x, orange = OLS fit. Each panel has its own range, made
# square (same range on x and y) so y = x is a true diagonal.

suppressPackageStartupMessages({
  library(ggplot2)
  library(dplyr)
  library(tidyr)
})

args <- commandArgs(trailingOnly = FALSE)
SCRIPT_DIR <- dirname(normalizePath(sub("^--file=", "", args[grep("^--file=", args)][1])))
BASE_DIR   <- normalizePath(file.path(SCRIPT_DIR, ".."))
OUTPUT_DIR <- file.path(BASE_DIR, "output")

BASE <- 13
pts  <- read.delim(file.path(OUTPUT_DIR, "FigS5C_predictions.tsv"), stringsAsFactors = FALSE)
cors <- read.delim(file.path(OUTPUT_DIR, "FigS5C_correlations.tsv"), stringsAsFactors = FALSE)
CELLS <- unique(cors$cell_line)
dat <- pts %>% transmute(cell_line = factor(cell_line, levels = CELLS),
                         x = pred_TEFL_mRNA_Reg, y = pred_RiboNN_transfer, dens = density)
cors$cell_line <- factor(cors$cell_line, levels = CELLS)

# common x/y range per panel (union of both axes, 3% padding)
sq <- dat %>% group_by(cell_line) %>%
  summarise(lo = min(c(x, y)), hi = max(c(x, y)), .groups = "drop") %>%
  mutate(pad = (hi - lo) * .03, lo = lo - pad, hi = hi + pad)
sq_pts <- bind_rows(transmute(sq, cell_line, x = lo, y = lo),
                    transmute(sq, cell_line, x = hi, y = hi))

lab <- cors %>%
  transmute(cell_line,
            l1 = sprintf('italic(r)==%.3f', pearson_r),
            l2 = sprintf('rho==%.3f', spearman_rho)) %>%
  pivot_longer(l1:l2, names_to = "row", values_to = "txt") %>%
  left_join(sq %>% transmute(cell_line,
                             lx = lo + (hi - lo) * .03,
                             ly = hi - (hi - lo) * .02,
                             dy = (hi - lo) * .075 * (BASE / 9)), by = "cell_line") %>%
  mutate(ly = ly - dy * (as.integer(sub("l", "", row)) - 1))

theme_scatter <- function() theme_classic(base_size = BASE) +
  theme(axis.text        = element_text(color = "black", size = BASE),
        legend.key.size  = unit(3, "mm"),
        strip.background = element_blank(),
        axis.title       = element_text(size = BASE + 1),
        strip.text       = element_text(face = "bold", size = BASE + 2, margin = margin(b = 3)),
        panel.grid.major = element_line(color = "grey92", linewidth = .25),
        panel.spacing    = unit(4, "mm"),
        aspect.ratio     = 1,
        plot.margin      = margin(4, 10, 3, 4))

p <- ggplot(dat, aes(x, y)) +
  geom_blank(data = sq_pts, aes(x = x, y = y), inherit.aes = FALSE) +
  geom_abline(slope = 1, intercept = 0, linetype = "22", linewidth = .45, color = "grey45") +
  geom_point(aes(color = dens), shape = 16, size = .55, alpha = .55) +
  scale_color_viridis_c(option = "mako", direction = -1, begin = .12, end = .95, guide = "none") +
  geom_smooth(method = "lm", formula = y ~ x, se = FALSE, color = "#D55E00", linewidth = .7) +
  geom_text(data = lab, aes(x = lx, y = ly, label = txt), parse = TRUE,
            hjust = 0, vjust = 1, size = BASE * .34, color = "grey10", inherit.aes = FALSE) +
  labs(x = "TEFL-mRNA (Reg), predicted TE", y = "RiboNN transfer, predicted TE") +
  facet_wrap(~ cell_line, nrow = 1, scales = "free") +
  theme_scatter()
ggsave(file.path(OUTPUT_DIR, "FigS5C.pdf"), p, width = 178, height = 86, units = "mm", device = pdf)
