#!/usr/bin/env Rscript
# Fig. EV1A: pairwise Spearman correlation of transcript-level TE profiles
# between the 47 retained LCL individuals. Samples ordered by hierarchical
# clustering (average linkage on 1 - rho).

# resolve FigEV1/ from this script's location so the package runs from any directory
BASE_DIR <- local({
  a <- commandArgs(trailingOnly = FALSE); f <- sub("^--file=", "", a[grepl("^--file=", a)])
  if (length(f)) dirname(dirname(normalizePath(f))) else normalizePath(".")
})
SHARED <- file.path(dirname(dirname(BASE_DIR)), "shared", "input")
OUTPUT <- file.path(BASE_DIR, "output")

te <- as.matrix(read.delim(file.path(SHARED, "LCL_TE_47_individuals.tsv"),
                           row.names = 1, check.names = FALSE))
colnames(te) <- sub("^X", "", colnames(te))

cm  <- cor(te, method = "spearman", use = "pairwise.complete.obs")
pw  <- cm[upper.tri(cm)]                      # 1,081 unique sample pairs
ord <- hclust(as.dist(1 - cm), method = "average")$order

# written in clustered order; the plotting script takes the heatmap order from this file
write.csv(round(cm[ord, ord], 6), file.path(OUTPUT, "FigEV1A_pairwise_spearman_matrix.csv"))

cat(sprintf("samples %d | pairs %d | median rho %.3f | range %.3f-%.3f\n",
            ncol(cm), length(pw), median(pw), min(pw), max(pw)))
