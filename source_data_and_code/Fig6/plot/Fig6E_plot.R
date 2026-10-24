#!/usr/bin/env Rscript
# Fig. 6E: heatmap of Fisher odds ratios (z-scored within each trait) for disease-trait enrichment
# among TE-altering GWAS SNVs; rows = cell lines grouped by region, columns = traits (clustered);
# stars give P (* < 0.05, ** < 0.01, *** < 0.001). Colours and clustering follow pheatmap
# (scale = "column", breaks -2 to 2 in 100 bins, Euclidean distance, complete linkage).

suppressPackageStartupMessages({
  library(data.table)
  library(ComplexHeatmap)
  library(circlize)
})

args <- commandArgs(trailingOnly = FALSE)
SCRIPT_DIR <- dirname(normalizePath(sub("^--file=", "", args[grep("^--file=", args)][1])))
BASE_DIR   <- normalizePath(file.path(SCRIPT_DIR, ".."))
OUTPUT_DIR <- file.path(BASE_DIR, "output")

res <- fread(file.path(OUTPUT_DIR, "Fig6E_disease_trait_enrichment.tsv"), sep = "\t")
traits  <- unique(res$trait)
columns <- unique(res$column)                       # 5'UTR columns first, then CDS
odds <- as.matrix(dcast(res, trait ~ column, value.var = "odds_ratio")[match(traits, trait), ..columns])
pval <- as.matrix(dcast(res, trait ~ column, value.var = "fisher_p")[match(traits, trait), ..columns])
rownames(odds) <- rownames(pval) <- traits

# cell lines as rows, traits as columns; each trait z-scored across the 17 region x cell-line rows
z <- scale(t(odds))
sig <- t(ifelse(pval < 0.001, "***", ifelse(pval < 0.01, "**", ifelse(pval < 0.05, "*", ""))))
breaks  <- seq(-2, 2, length.out = 101)
palette <- colorRampPalette(c("#2166AC", "#F7F7F7", "#B2182B"))(100)
fill <- matrix(palette[findInterval(z, breaks, all.inside = TRUE)], nrow(z))
fill[is.na(z)] <- "white"
trait_tree <- hclust(dist(t(z)), method = "complete")

to_sentence <- function(x) { x <- tolower(as.character(x)); paste0(toupper(substr(x, 1, 1)), substr(x, 2, nchar(x))) }
region <- factor(ifelse(grepl("_UTR5$", columns), "5'UTR", "CDS"), levels = c("5'UTR", "CDS"))

ht <- Heatmap(z,
  name                 = "Odds ratio z-score",
  col                  = colorRamp2(seq(-2, 2, length.out = 100), palette),
  rect_gp              = gpar(type = "none"),
  cell_fun             = function(j, i, x, y, w, h, f) {
    grid.rect(x, y, w, h, gp = gpar(fill = fill[i, j], col = "white"))
    grid.text(sig[i, j], x, y, gp = gpar(fontsize = 8, col = "black"))
  },
  cluster_rows         = FALSE,
  cluster_columns      = trait_tree,
  column_dend_reorder  = FALSE,
  column_dend_height   = unit(50, "bigpts"),
  row_split            = region,
  row_gap              = unit(4, "bigpts"),
  row_title_rot        = 0,
  row_title_gp         = gpar(fontsize = 14),
  row_labels           = sub("_UTR5$|_CDS$", "", columns),
  column_labels        = to_sentence(traits),
  row_names_gp         = gpar(fontsize = 13),
  column_names_gp      = gpar(fontsize = 13),
  left_annotation      = rowAnnotation(Region = region, col = list(Region = c("5'UTR" = "#1F77B4", "CDS" = "#FF7F0E")),
                                       show_legend = FALSE, annotation_name_side = "bottom",
                                       annotation_name_gp = gpar(fontsize = 10, fontface = "bold")),
  heatmap_legend_param = list(at = -2:2, title_position = "leftcenter-rot", legend_height = unit(4, "cm")),
  width                = ncol(z) * unit(14, "bigpts"),
  height               = nrow(z) * unit(13, "bigpts"))

pdf(file.path(OUTPUT_DIR, "Fig6E.pdf"), width = 15, height = 7.5)
draw(ht)
invisible(dev.off())
