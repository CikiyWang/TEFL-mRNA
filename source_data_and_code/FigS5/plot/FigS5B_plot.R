#!/usr/bin/env Rscript
# Fig. S5B: Pearson R per cell line, bars = mean +- s.d.; grey lines join the three models'
# values for the same fold.

suppressPackageStartupMessages(library(ggplot2))

args <- commandArgs(trailingOnly = FALSE)
SCRIPT_DIR <- dirname(normalizePath(sub("^--file=", "", args[grep("^--file=", args)][1])))
BASE_DIR   <- normalizePath(file.path(SCRIPT_DIR, ".."))
OUTPUT_DIR <- file.path(BASE_DIR, "output")

Y_TEXT <- 9
PAL <- c("TEFL-mRNA (Reg)" = "#0072B2", "RiboNN transfer" = "#D55E00", "RiboNN multi-task" = "#009E73")

s  <- read.delim(file.path(OUTPUT_DIR, "FigS5B_pearson_summary.tsv"), stringsAsFactors = FALSE)
df <- read.delim(file.path(OUTPUT_DIR, "FigS5B_pearson_by_fold.tsv"), stringsAsFactors = FALSE)
CELLS  <- unique(s$cell_line)
MODELS <- unique(s$model)
for (v in c("s", "df")) {
  d <- get(v)
  d$cell_line <- factor(d$cell_line, levels = CELLS)
  d$model <- factor(d$model, levels = MODELS)
  d$xn <- as.numeric(d$model)
  assign(v, d)
}

theme_pub <- function(base = 7) theme_classic(base_size = base) +
  theme(axis.text        = element_text(color = "black"),
        legend.key.size  = unit(3, "mm"),
        strip.background = element_blank(),
        plot.title       = element_text(face = "bold", size = base + 1))

p <- ggplot() +
  geom_rect(data = s, aes(xmin = xn - .33, xmax = xn + .33, ymin = bar_floor, ymax = mean,
                          fill = model), alpha = .85) +
  geom_errorbar(data = s, aes(x = xn, ymin = mean - sd, ymax = mean + sd),
                width = .16, linewidth = .35, color = "grey20") +
  geom_line(data = df, aes(x = xn, y = Pearson_R, group = fold),
            color = "grey45", linewidth = .22, alpha = .6) +
  geom_point(data = df, aes(x = xn, y = Pearson_R), shape = 21, size = 1.1,
             stroke = .2, fill = "white", color = "grey25", alpha = .9) +
  # anchor each free-y panel at its own bar floor
  geom_blank(data = unique(s[, c("cell_line", "bar_floor")]), aes(x = 1, y = bar_floor)) +
  scale_fill_manual(values = PAL, guide = "none") +
  scale_x_continuous(breaks = seq_along(MODELS), labels = MODELS,
                     expand = expansion(add = .55)) +
  scale_y_continuous(expand = expansion(mult = c(0, .06)), n.breaks = 5) +
  theme_pub() +
  theme(axis.text.y    = element_text(size = Y_TEXT),
        axis.text.x    = element_text(angle = 35, hjust = 1),
        panel.spacing  = unit(2.5, "mm")) +
  facet_wrap(~ cell_line, nrow = 1, scales = "free_y") +
  labs(x = NULL, y = "Pearson R")
ggsave(file.path(OUTPUT_DIR, "FigS5B.pdf"), p, width = min(178, 40 + 34 * length(CELLS)), height = 68,
       units = "mm", device = pdf)
