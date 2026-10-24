#!/usr/bin/env Rscript
# Fig. S5C: out-of-fold predicted TE of TEFL-mRNA (regression) vs RiboNN transfer (RiboNN's
# single-task models), one point per transcript, per cell line: Pearson r, Spearman rho and the
# local 2-D point density used to colour the points.

suppressPackageStartupMessages({
  library(dplyr)
  library(readr)
})

args <- commandArgs(trailingOnly = FALSE)
SCRIPT_DIR <- dirname(normalizePath(sub("^--file=", "", args[grep("^--file=", args)][1])))
BASE_DIR   <- normalizePath(file.path(SCRIPT_DIR, ".."))
INPUT_DIR  <- file.path(BASE_DIR, "input")
OUTPUT_DIR <- file.path(BASE_DIR, "output")

CELLS   <- c("HMECs", "T47D", "OSCC")
X_MODEL <- "Our model"       # TEFL-mRNA (Reg)
Y_MODEL <- "RiboNN single"   # RiboNN transfer

read_cell <- function(cell) {
  d <- read_tsv(file.path(INPUT_DIR, "a2a_comparison", cell, "predictions_merged.tsv"), show_col_types = FALSE)
  tibble(cell_line = cell, tx_id = d$tx_id, fold = d$ribonn_fold, true_y = d$true_y,
         x = d[[X_MODEL]], y = d[[Y_MODEL]]) %>%
    filter(is.finite(x), is.finite(y), is.finite(true_y))
}
dat <- bind_rows(lapply(CELLS, read_cell)) %>%
  mutate(cell_line = factor(cell_line, levels = CELLS))
stopifnot(nrow(dat %>% count(cell_line, tx_id) %>% filter(n > 1)) == 0)

stats_one <- function(d) {
  p <- suppressWarnings(cor.test(d$x, d$y, method = "pearson"))
  s <- suppressWarnings(cor.test(d$x, d$y, method = "spearman", exact = FALSE))
  fit <- lm(y ~ x, data = d)
  tibble(n = nrow(d),
         pearson_r = unname(p$estimate), pearson_p = p$p.value,
         pearson_lo = p$conf.int[1], pearson_hi = p$conf.int[2],
         spearman_rho = unname(s$estimate), spearman_p = s$p.value,
         ols_slope = unname(coef(fit)[2]), ols_intercept = unname(coef(fit)[1]))
}
cors <- dat %>% group_by(cell_line) %>% group_modify(~ stats_one(.x)) %>% ungroup()

# local 2-D density at each point, scaled to 0-1 within each cell line
dens_at_points <- function(x, y, n = 220) {
  k  <- MASS::kde2d(x, y, n = n)
  ix <- pmin(pmax(findInterval(x, k$x), 1L), length(k$x))
  iy <- pmin(pmax(findInterval(y, k$y), 1L), length(k$y))
  z  <- k$z[cbind(ix, iy)]
  z / max(z)
}
dat <- dat %>% group_by(cell_line) %>%
  mutate(density = dens_at_points(x, y)) %>%
  arrange(density, .by_group = TRUE) %>%   # row order = drawing order, densest on top
  ungroup()

dir.create(OUTPUT_DIR, showWarnings = FALSE)
write_tsv(dat %>% select(cell_line, tx_id, fold, true_y,
                         pred_TEFL_mRNA_Reg = x, pred_RiboNN_transfer = y, density),
          file.path(OUTPUT_DIR, "FigS5C_predictions.tsv"))
write_tsv(cors, file.path(OUTPUT_DIR, "FigS5C_correlations.tsv"))
print(as.data.frame(cors %>% select(cell_line, n, pearson_r, spearman_rho, ols_slope)), digits = 4)
