#!/usr/bin/env Rscript
# Fig. 3A: one-row heatmap of mean delta pTE per synonymous codon substitution, columns split
# by codon-usage direction and clustered within each split.

suppressPackageStartupMessages({
  library(data.table)
  library(ComplexHeatmap)
  library(circlize)
  library(RColorBrewer)
})
# C locale, as the published panel was rendered; the base pdf() device cannot encode U+0394
# in any locale, so each delta prints as ".."
invisible(Sys.setlocale("LC_CTYPE", "C"))

args <- commandArgs(trailingOnly = FALSE)
SCRIPT_DIR <- dirname(normalizePath(sub("^--file=", "", args[grep("^--file=", args)][1])))
BASE_DIR   <- normalizePath(file.path(SCRIPT_DIR, ".."))
OUTPUT_DIR <- file.path(BASE_DIR, "output")

dpTE_thresh <- 0.05
summary_dt <- fread(file.path(OUTPUT_DIR, "Fig3A_codon_substitution_dpTE.tsv"))

codon_labels <- summary_dt$ref_alt_codon
codon_class  <- summary_dt$delta_codon_var_freq_class
codon_aa     <- summary_dt$aa_label
mat_mean <- matrix(summary_dt$mean_dpTE, nrow = 1, dimnames = list("LCL", codon_labels))

aa_colors <- setNames(
  colorRampPalette(brewer.pal(12, "Set3"))(length(unique(codon_aa))),
  sort(unique(codon_aa))
)
col_ha <- HeatmapAnnotation(
  `CAI direction` = codon_class,
  `Amino acid`    = codon_aa,
  col = list(
    `CAI direction` = c("more optimal" = "#E64B35", "less optimal" = "#4DBBD5"),
    `Amino acid`    = aa_colors
  ),
  annotation_name_gp   = gpar(fontsize = 8),
  simple_anno_size     = unit(4, "mm"),
  annotation_name_side = "left"
)

ht <- Heatmap(mat_mean,
  name                  = "Mean ΔpTE\n(|ΔpTE|>0.05)",
  col                   = colorRamp2(c(-0.1, 0, 0.1), c("navy", "white", "salmon")),
  na_col                = "white",
  top_annotation        = col_ha,
  cluster_columns       = TRUE,
  cluster_column_slices = FALSE,
  column_split          = codon_class,
  column_gap            = unit(3, "mm"),
  cluster_rows          = FALSE,
  row_names_gp          = gpar(fontsize = 10),
  column_names_gp       = gpar(fontsize = 6),
  column_title_gp       = gpar(fontsize = 10, fontface = "bold"),
  rect_gp               = gpar(col = "white", lwd = 0.1),
  heatmap_legend_param  = list(title_gp = gpar(fontsize = 9), labels_gp = gpar(fontsize = 8)),
  column_title          = c("less optimal", "more optimal")
)

pdf(file.path(OUTPUT_DIR, "Fig3A.pdf"), width = 18, height = 2.5)
draw(ht,
  column_title    = paste0("gnomAD synonymous variant ΔpTE (LCL, |ΔpTE| > ", dpTE_thresh, ")  — mean_raw"),
  column_title_gp = gpar(fontsize = 12, fontface = "bold"))
invisible(dev.off())
