#!/usr/bin/env Rscript
# Fig. 6A: heatmap of median delta pTE per synonymous codon substitution (columns, split by
# codon-usage direction and clustered within each split) and cell line (rows, clustered).

suppressPackageStartupMessages({
  library(data.table)
  library(ComplexHeatmap)
  library(circlize)
  library(RColorBrewer)
})
# Rscript may start in the C locale, where the Greek delta is drawn as dots
invisible(suppressWarnings(Sys.setlocale("LC_CTYPE", "en_US.UTF-8")))

args <- commandArgs(trailingOnly = FALSE)
SCRIPT_DIR <- dirname(normalizePath(sub("^--file=", "", args[grep("^--file=", args)][1])))
BASE_DIR   <- normalizePath(file.path(SCRIPT_DIR, ".."))
OUTPUT_DIR <- file.path(BASE_DIR, "output")

mat_wide <- fread(file.path(OUTPUT_DIR, "Fig6A_median_dpTE_matrix.tsv"))
codon_class <- mat_wide$delta_codon_var_freq_class
codon_aa    <- mat_wide$aa_var
mat <- as.matrix(mat_wide[, !c("ref_alt_codon", "delta_codon_var_freq_class", "aa_var"), with = FALSE])
rownames(mat) <- mat_wide$ref_alt_codon
mat_t <- t(mat)                      # cell lines = rows, codon substitutions = columns

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

ht <- Heatmap(mat_t,
  name                  = "Median ΔpTE\n(|ΔpTE|>0.02)",
  col                   = colorRamp2(c(-0.06, 0, 0.06), c("navy", "white", "salmon")),
  na_col                = "white",
  top_annotation        = col_ha,
  cluster_columns       = TRUE,
  cluster_column_slices = FALSE,
  column_split          = codon_class,
  column_gap            = unit(3, "mm"),
  cluster_rows          = TRUE,
  row_names_gp          = gpar(fontsize = 10),
  column_names_gp       = gpar(fontsize = 6),
  column_title_gp       = gpar(fontsize = 10, fontface = "bold"),
  rect_gp               = gpar(col = "white", lwd = 0.1),
  heatmap_legend_param  = list(title_gp = gpar(fontsize = 9), labels_gp = gpar(fontsize = 8)),
  column_title          = c("less optimal", "more optimal")
)

# the base pdf() device cannot encode the Greek delta; use cairo_pdf or the macOS quartz PDF device
pdf_ok <- function(dev) isTRUE(tryCatch({
  f <- tempfile(fileext = ".pdf"); dev(f, width = 2, height = 2)
  grid::grid.text("Δ"); invisible(grDevices::dev.off()); TRUE
}, error = function(e) { try(grDevices::dev.off(), silent = TRUE); FALSE },
   warning = function(w) { try(grDevices::dev.off(), silent = TRUE); FALSE }))
out <- file.path(OUTPUT_DIR, "Fig6A.pdf")
if (pdf_ok(grDevices::cairo_pdf)) {
  grDevices::cairo_pdf(out, width = 17, height = 3.5)
} else if (isTRUE(capabilities("aqua"))) {
  grDevices::quartz(file = out, type = "pdf", width = 17, height = 3.5)
} else {
  grDevices::pdf(out, width = 17, height = 3.5)
}
draw(ht)
invisible(dev.off())
