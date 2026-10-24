#!/usr/bin/env Rscript
# Fig. S5B: Pearson R between predicted and measured TE on RiboNN's own 10 cross-validation
# folds, per cell line, for TEFL-mRNA (regression), RiboNN multi-task and RiboNN transfer
# (RiboNN's single-task models). Bars = mean +- s.d. over folds.

suppressPackageStartupMessages({
  library(dplyr)
  library(readr)
})

args <- commandArgs(trailingOnly = FALSE)
SCRIPT_DIR <- dirname(normalizePath(sub("^--file=", "", args[grep("^--file=", args)][1])))
BASE_DIR   <- normalizePath(file.path(SCRIPT_DIR, ".."))
INPUT_DIR  <- file.path(BASE_DIR, "input")
OUTPUT_DIR <- file.path(BASE_DIR, "output")

METRIC    <- "Pearson_R"
BAR_FLOOR <- 0.5
# model names in the input tables -> names in the figure
MODELS <- c("Our model" = "TEFL-mRNA (Reg)", "RiboNN multi" = "RiboNN multi-task", "RiboNN single" = "RiboNN transfer")

tsvs <- Sys.glob(file.path(INPUT_DIR, "a2a_comparison", "*", "metrics_by_fold.tsv"))
dat <- bind_rows(lapply(tsvs, read_tsv, show_col_types = FALSE)) %>% filter(metric == METRIC)

# cell lines ordered by TEFL-mRNA's mean, best first
ord <- dat %>% filter(model == "Our model") %>%
  group_by(cell_line) %>% summarise(m = mean(value), .groups = "drop") %>%
  arrange(desc(m)) %>% pull(cell_line)
dat <- dat %>%
  mutate(model = factor(MODELS[model], levels = MODELS), cell_line = factor(cell_line, levels = ord)) %>%
  arrange(cell_line, model, fold)

# bars start at BAR_FLOOR in a panel only if every mean - sd there stays above it, else at 0
bars <- dat %>% group_by(cell_line, model) %>%
  summarise(mean = mean(value), sd = sd(value), .groups = "drop") %>%
  group_by(cell_line) %>%
  mutate(bar_floor = { lo <- min(mean - sd, na.rm = TRUE)
                       f <- if (lo > BAR_FLOOR) BAR_FLOOR else 0
                       if (lo < f) 0 else f }) %>%
  ungroup()

dir.create(OUTPUT_DIR, showWarnings = FALSE)
write_tsv(dat %>% select(cell_line, model, fold, n, Pearson_R = value), file.path(OUTPUT_DIR, "FigS5B_pearson_by_fold.tsv"))
write_tsv(bars, file.path(OUTPUT_DIR, "FigS5B_pearson_summary.tsv"))
print(as.data.frame(bars), digits = 4)
