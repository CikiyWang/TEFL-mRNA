suppressPackageStartupMessages({
  library(ggplot2)
  library(ggthemes)
  library(enrichplot)
})

file_arg <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
base_dir <- if (length(file_arg)) dirname(dirname(normalizePath(sub("^--file=", "", file_arg)))) else getwd()
output_dir <- file.path(base_dir, "output")

DO_var <- readRDS(file.path(output_dir, "Fig4B_DO_enrichment.rds"))

# fixed seed for the network layout
set.seed(1)
# default node scaling (cex_category = 1 in enrichplot < 1.24, renamed size_category later)
p <- emapplot(DO_var) + scale_color_continuous_tableau() + theme_classic() + theme(aspect.ratio = 1)
ggsave(file.path(output_dir, "Fig4B.pdf"), p, height = 7.5, width = 7.5)
