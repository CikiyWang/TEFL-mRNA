suppressPackageStartupMessages({
  library(data.table)
  library(ggplot2)
  library(ggsci)
})

file_arg <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
base_dir <- if (length(file_arg)) dirname(dirname(normalizePath(sub("^--file=", "", file_arg)))) else getwd()
output_dir <- file.path(base_dir, "output")

curves <- fread(file.path(output_dir, "Fig1D_source_data.tsv.gz"), sep = "\t")
aucs <- fread(file.path(output_dir, "Fig1D_auc.tsv"), sep = "\t")
lab <- aucs[, setNames(paste0("Fold-", fold, " (AUC = ", round(auc, 3), ")"), fold)]
curves[, Fold := lab[as.character(fold)]]

p <- ggplot(curves, aes(x = specificity, y = sensitivity, color = Fold)) +
  geom_line() +
  theme_bw() +
  scale_color_d3() +
  xlab("Specificities") +
  ylab("Sensitivities") +
  theme(aspect.ratio = 1,
        legend.position = "right",
        axis.text = element_text(size = 12, angle = 0, colour = "black", hjust = 0.5, vjust = 0.5),
        plot.title = element_text(lineheight = 15, size = 10, hjust = 0.5),
        axis.title = element_text(size = 12, color = "black"),
        legend.text = element_text(size = 11, color = "black"))
ggsave(file.path(output_dir, "Fig1D.pdf"), p, height = 5, width = 5.5)
