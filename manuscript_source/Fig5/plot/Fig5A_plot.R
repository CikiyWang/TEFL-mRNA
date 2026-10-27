suppressPackageStartupMessages({
  library(data.table)
  library(ggplot2)
  library(ggsci)
})

file_arg <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
base_dir <- if (length(file_arg)) dirname(dirname(normalizePath(sub("^--file=", "", file_arg)))) else getwd()
output_dir <- file.path(base_dir, "output")

curves <- fread(file.path(output_dir, "Fig5A_source_data.tsv.gz"), sep = "\t")
aucs <- fread(file.path(output_dir, "Fig5A_auc.tsv"), sep = "\t")

# One panel per cell line, as arranged in Fig. 5A
for (cl in unique(aucs$cell_line)) {
  lab <- aucs[cell_line == cl, setNames(paste0("Fold-", fold, " (AUC = ", round(auc, 3), ")"), fold)]
  d <- curves[cell_line == cl]
  d[, Fold := lab[as.character(fold)]]
  p <- ggplot(d, aes(x = specificity, y = sensitivity, colour = Fold, group = Fold)) +
    geom_path() +
    scale_colour_d3() +
    labs(title = cl, x = "Specificity", y = "Sensitivity", colour = NULL) +
    theme_bw() +
    theme(aspect.ratio = 1,
          plot.title = element_text(size = 13, hjust = 0.5, colour = "black"),
          axis.text = element_text(size = 12, colour = "black"),
          axis.title = element_text(size = 12, colour = "black"),
          legend.text = element_text(size = 11, colour = "black"))
  ggsave(file.path(output_dir, paste0("Fig5A_", cl, ".pdf")), p, width = 5, height = 5)
}
