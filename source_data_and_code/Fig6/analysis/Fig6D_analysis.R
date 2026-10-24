#!/usr/bin/env Rscript
# Fig. 6D: Spearman correlation between measured TE and five transcript features (AMFE = MFE /
# length, CAI, estimated half-life, GC ratio, full length) in each of 78 cell lines; the panel
# shows nine. Bar colour encodes -log10(P) scaled within each feature across all 78 cell lines.

suppressPackageStartupMessages({
  library(data.table)
  library(dplyr)
})

args <- commandArgs(trailingOnly = FALSE)
SCRIPT_DIR <- dirname(normalizePath(sub("^--file=", "", args[grep("^--file=", args)][1])))
BASE_DIR   <- normalizePath(file.path(SCRIPT_DIR, ".."))
INPUT_DIR  <- file.path(BASE_DIR, "input")
OUTPUT_DIR <- file.path(BASE_DIR, "output")

te <- fread(file.path(INPUT_DIR, "RiboNN_2025_NBT_human_cell_lines_TE.TE_by_cell_line.tsv"))
te[, `:=`(seq_id = sub("\\..*$", "", tx_id), transcript_id = tx_id)]
feat <- fread(file.path(INPUT_DIR, "RiboNN_2025_NBT_human_cell_lines_TE.FL_scores.tsv"))
cai <- fread(file.path(INPUT_DIR, "FL_regions.2025_NBT_celllines_q25q75_mean_te_scoreCAI.tsv"))
te[, CAI := cai$scoreCAI[match(transcript_id, cai$transcript_id)]]
scores <- merge(te, feat, by = "seq_id")

CELL_COLS <- grep("^TE_", names(te), value = TRUE)
SEL_COLS  <- c("CAI", "MFE", "Length", "Unpaired_ratio", "Harpin_counts", "Degscore", "Est_half_life", "GcRatio")
FEATURES  <- c("CAI", "AMFE", "Est_half_life", "GcRatio", "Length")

# TE columns hold log10 TE; Spearman rho is unchanged by the back-transform
cors <- rbindlist(lapply(CELL_COLS, function(cl) {
  d <- na.omit(cbind(data.table(TE = 10^scores[[cl]]), scores[, SEL_COLS, with = FALSE]))
  d[, AMFE := MFE / Length]
  rbindlist(lapply(FEATURES, function(f) {
    ct <- suppressWarnings(cor.test(d$TE, d[[f]], method = "spearman"))
    data.table(Cell_line = sub("^TE_", "", cl), Feature = f, n = nrow(d), cors_TE = unname(ct$estimate), pvals_TE = ct$p.value)
  }))
}))

# x order: cell lines by decreasing rho within features (AMFE first), as one shared order
ordered <- cors %>% group_by(Feature) %>% arrange(Feature, desc(cors_TE)) %>%
  mutate(Cell_line_ordered = factor(Cell_line, levels = unique(Cell_line))) %>% ungroup()
# P below the double-precision range is version-dependent: R 4.0.3, which drew the published
# panel, returned tiny non-zero values (-log10 P of about 300-320), whereas newer R returns 0.
# P = 0 is set to .Machine$double.xmin / 2 (-log10 P = 307.95); other P values are used unchanged.
ordered$minus_log10_pvals_TE <- -log10(ifelse(ordered$pvals_TE == 0, .Machine$double.xmin / 2, ordered$pvals_TE))

PALETTES <- list(AMFE = c("#f7fbff", "#08519c"), CAI = c("#fff5f0", "#a50f15"), Est_half_life = c("#f7fcf5", "#00441b"),
                 GcRatio = c("#fcf2f0", "#8b0058"), Length = c("#fffdf7", "#f47100"))
feature_colors <- function(feature, values) {
  norm <- (values - min(values, na.rm = TRUE)) / (max(values, na.rm = TRUE) - min(values, na.rm = TRUE))
  colorRampPalette(PALETTES[[feature]])(100)[pmax(1, pmin(100, round(norm * 99) + 1))]
}
ordered <- ordered %>% group_by(Feature) %>%
  mutate(custom_color = feature_colors(as.character(Feature[1]), minus_log10_pvals_TE)) %>% ungroup()

CELLS <- c("HEK293T", "HeLa", "HepG2", "HMECs", "LCL", "PC3", "PC9", "T47D", "U2OS")
sel <- ordered %>% filter(Cell_line %in% CELLS) %>%
  mutate(significance = case_when(pvals_TE < 0.001 ~ "***", pvals_TE < 0.01 ~ "**", pvals_TE < 0.05 ~ "*",
                                  pvals_TE < 0.1 ~ ".", TRUE ~ ""),
         label_y_end = ifelse(cors_TE >= 0, cors_TE + 0.02, cors_TE - 0.04),
         x_order = as.integer(droplevels(Cell_line_ordered)))

dir.create(OUTPUT_DIR, showWarnings = FALSE)
fwrite(cors, file.path(OUTPUT_DIR, "Fig6D_spearman_TE_vs_features.78_cell_lines.tsv"), sep = "\t")
fwrite(sel %>% select(Feature, Cell_line, x_order, n, cors_TE, pvals_TE, significance, label_y_end, custom_color),
       file.path(OUTPUT_DIR, "Fig6D_plot_data.tsv"), sep = "\t")
print(dcast(as.data.table(sel), Cell_line ~ Feature, value.var = "cors_TE"), digits = 3)
