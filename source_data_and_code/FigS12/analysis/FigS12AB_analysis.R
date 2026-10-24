## Fig. S12A-B: cross-cell-line variance relative to within-cell-line fold
## variance, by transcript region and by directional concordance.
##
## H = MS(between cell lines) / MS(within cell line, folds) is the F statistic
## of a one-way ANOVA of dpTE across the nine cell lines with the ten folds as
## replicates, tested against F(8, 81).
##
## Usage (from FigS12/):  Rscript analysis/FigS12AB_analysis.R

suppressPackageStartupMessages(library(data.table))

INPUT_DIR  <- "input"
OUTPUT_DIR <- "output"
stopifnot(dir.exists(INPUT_DIR), dir.exists(OUTPUT_DIR))

CELLS_N <- 9L
N_FOLD  <- 10L
REG     <- c("5'UTR", "CDS_missense", "CDS_synonymous", "3'UTR")

PV <- fread(file.path(INPUT_DIR, "per_variant_cross_cellline_stats.tsv.gz"))
PV[, region := factor(region, levels = REG)]

## per-variant F test against F(C - 1, C * (n - 1)) = F(8, 81), corrected
## within each region
PV[, p_F := pf(H, CELLS_N - 1L, CELLS_N * (N_FOLD - 1L), lower.tail = FALSE)]
PV[, q_F := p.adjust(p_F, "BH"), by = region]

fwrite(PV[, .(region, variant, H, q_F, n_cell_pos)],
       file.path(OUTPUT_DIR, "FigS12AB_per_variant_H_q.tsv.gz"), sep = "\t")

## Panel A: distribution of H by region, with the BH pass rate
A <- PV[, .(n = .N, median_H = median(H), pct_H_gt1 = 100 * mean(H > 1),
            pct_q_lt_05 = 100 * mean(q_F < 0.05)),
        by = region][order(match(region, REG))]
fwrite(A, file.path(OUTPUT_DIR, "FigS12A_H_by_region.tsv"), sep = "\t")
print(A, digits = 3)

## Panel B: pass rate across the directional spectrum. Variants are stratified
## by the number of cell lines in which the effect is positive (0-9), so fully
## concordant variants lie at the two ends and discordant variants in the middle.
B <- PV[, .(n = .N, median_H = median(H), frac_pass = mean(q_F < 0.05)),
        by = .(region, n_cell_pos)][order(match(region, REG), n_cell_pos)]
fwrite(B, file.path(OUTPUT_DIR, "FigS12B_Fpass_by_cells_positive.tsv"), sep = "\t")

## null reference, reported in the legend
cat(sprintf("\nnull F(%d, %d): median %.3f, P(H > 1) = %.3f\n",
            CELLS_N - 1L, CELLS_N * (N_FOLD - 1L),
            qf(0.5, CELLS_N - 1L, CELLS_N * (N_FOLD - 1L)),
            pf(1, CELLS_N - 1L, CELLS_N * (N_FOLD - 1L), lower.tail = FALSE)))
