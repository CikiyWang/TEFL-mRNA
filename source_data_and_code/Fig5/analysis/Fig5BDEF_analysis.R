# Fig. 5B, D, E, F: ExAC 5'UTR SNVs scored in 9 cell lines (delta pTE = pTE(ALT) - pTE(REF)).
#   B  pairwise Pearson correlation and direction agreement between cell lines, over SNVs with
#      |delta pTE| > 0.02 in at least one cell line
#   D  ALT-base composition of SNVs shared by >= 3 cell lines at each cutoff
#   E  distance to the start codon of those shared SNVs (< 1,000 nt), with pairwise KS tests
#   F  gnomAD allele-frequency classes of those shared SNVs
suppressPackageStartupMessages(library(data.table))

file_arg <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
base_dir <- if (length(file_arg)) dirname(dirname(normalizePath(sub("^--file=", "", file_arg)))) else getwd()
input_dir <- file.path(base_dir, "input")
output_dir <- file.path(base_dir, "output")
dir.create(output_dir, showWarnings = FALSE)

CELLS <- c("HEK293T", "HeLa", "HepG2", "HMECs", "LCL", "PC3", "PC9", "T47D", "U2OS")

utr5 <- fread(file.path(input_dir, "ExAC_utr5_SNVs_delta_pTE.9_cell_lines.tsv.gz"), sep = "\t")
dpte <- as.matrix(utr5[, ..CELLS])
alt <- vapply(strsplit(utr5$variant_transcript_id, "_"), `[`, "", 4)
af <- suppressWarnings(as.numeric(utr5$gnomade_AF))

# Number of cell lines in which an SNV raises (up) or lowers (down) pTE beyond a cutoff
n_up   <- function(cutoff) rowSums(dpte > cutoff)
n_down <- function(cutoff) rowSums(dpte < -cutoff)

## ---- Fig. 5B ----
# SNVs passing both the up and the down filter enter twice, as in the original analysis
sel_idx <- c(which(n_up(0.02) >= 1), which(n_down(0.02) >= 1))
sel <- dpte[sel_idx, ]
cor_matrix <- cor(sel, use = "complete.obs")
n_obs <- nrow(na.omit(sel))
t_stat <- cor_matrix * sqrt(n_obs - 2) / sqrt(1 - cor_matrix^2)
p_matrix <- 2 * pt(-abs(t_stat), n_obs - 2)

pairs <- combn(CELLS, 2, simplify = FALSE)
fig5b_cor <- rbindlist(lapply(pairs, function(p) data.table(
  cell_line_1 = p[1], cell_line_2 = p[2], n = n_obs,
  pearson_r = cor_matrix[p[1], p[2]], p_value = p_matrix[p[1], p[2]])))

fig5b_dir <- rbindlist(lapply(pairs, function(p) {
  x <- sel[, p[1]]; y <- sel[, p[2]]
  ok <- !is.na(x) & !is.na(y); x <- x[ok]; y <- y[ok]
  both_positive <- sum(x > 0 & y > 0)
  both_negative <- sum(x < 0 & y < 0)
  opposite <- sum((x > 0 & y < 0) | (x < 0 & y > 0))
  total <- length(x)
  data.table(cell_line_1 = p[1], cell_line_2 = p[2],
             both_positive, both_negative, opposite, total,
             pct_both_positive = round(100 * both_positive / total, 1),
             pct_both_negative = round(100 * both_negative / total, 1),
             pct_opposite = round(100 * opposite / total, 1))
}))
cat(sprintf("Fig. 5B: %d SNV rows (%d unique SNVs)\n", nrow(sel), length(unique(sel_idx))))
print(fig5b_cor[, .(min_r = min(pearson_r), max_r = max(pearson_r), max_p = max(p_value))])

## ---- Fig. 5D ----
fig5d <- rbindlist(lapply(c(0.01, 0.02, 0.03, 0.04, 0.05), function(cut) {
  rbind(data.table(TE = "TE-increasing", cutoff = format(cut), idx = which(n_up(cut) >= 3)),
        data.table(TE = "TE-decreasing", cutoff = format(-cut), idx = which(n_down(cut) >= 3)))
}))
fig5d[, alt := alt[idx]]
fig5d <- fig5d[, .(n = .N), by = .(TE, cutoff, alt)]
fig5d[, n_snvs := sum(n), by = .(TE, cutoff)]
fig5d[, freq := n / n_snvs]
setorder(fig5d, TE, cutoff, alt)
print(dcast(fig5d, TE + cutoff + n_snvs ~ alt, value.var = "freq"), digits = 3)

## ---- Fig. 5E ----
fig5e <- rbind(
  data.table(cutoff = "0", pos2start = utr5$pos2start),
  rbindlist(lapply(c(0.01, 0.02, 0.03, 0.04, 0.05), function(cut) data.table(
    cutoff = format(cut),
    pos2start = utr5$pos2start[c(which(n_up(cut) >= 3), which(n_down(cut) >= 3))]))))
fig5e <- fig5e[pos2start < 1000]
groups <- unique(fig5e$cutoff)
fig5e_ks <- rbindlist(lapply(combn(groups, 2, simplify = FALSE), function(g) {
  k <- suppressWarnings(ks.test(fig5e[cutoff == g[1], pos2start], fig5e[cutoff == g[2], pos2start]))
  data.table(cutoff_1 = g[1], cutoff_2 = g[2],
             n_1 = sum(fig5e$cutoff == g[1]), n_2 = sum(fig5e$cutoff == g[2]),
             D = unname(k$statistic), p_value = k$p.value)
}))
print(fig5e_ks, digits = 3)

## ---- Fig. 5F ----
af_class <- function(x) {
  ifelse(x <= 0.00001, "AF (<=0.0001)",
         ifelse(x > 0.00001 & x < 0.0001, "AF (<=0.0001)",
                ifelse(x > 0.0001 & x < 0.001, "AF (0.0001-0.01)",
                       ifelse(x > 0.001 & x < 0.01, "AF (0.0001-0.01)", "AF (>=0.01)"))))
}
AF_LEVELS <- c("AF (<=0.0001)", "AF (0.0001-0.01)", "AF (>=0.01)")
fig5f <- rbindlist(lapply(c(0.03, 0.04, 0.05, 0.06), function(cut) {
  idx <- c(which(n_up(cut) >= 3), which(n_down(cut) >= 3))
  tab <- table(factor(af_class(af[idx]), levels = AF_LEVELS))
  data.table(Cutoffs = paste0(">", cut), AF_class = names(tab), Var_counts = as.integer(tab),
             Percents_allvar = as.integer(tab) / sum(tab))
}))
print(fig5f, digits = 3)

fwrite(fig5b_cor, file.path(output_dir, "Fig5B_correlation.tsv"), sep = "\t")
fwrite(fig5b_dir, file.path(output_dir, "Fig5B_direction_agreement.tsv"), sep = "\t")
fwrite(fig5d, file.path(output_dir, "Fig5D_source_data.tsv"), sep = "\t")
fwrite(fig5e, file.path(output_dir, "Fig5E_source_data.tsv.gz"), sep = "\t", compress = "gzip")
fwrite(fig5e_ks, file.path(output_dir, "Fig5E_ks_tests.tsv"), sep = "\t")
fwrite(fig5f, file.path(output_dir, "Fig5F_source_data.tsv"), sep = "\t")
