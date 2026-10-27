#!/usr/bin/env Rscript
# Fig. EV3A: ROC curves, TE-high vs others, TEFL-mRNA vs RiboNN.

suppressPackageStartupMessages({
  library(ggplot2)
  library(ggsci)
})

args <- commandArgs(trailingOnly = FALSE)
SCRIPT_DIR <- dirname(normalizePath(sub("^--file=", "", args[grep("^--file=", args)][1])))
BASE_DIR   <- normalizePath(file.path(SCRIPT_DIR, ".."))
OUTPUT_DIR <- file.path(BASE_DIR, "output")

curves  <- read.delim(file.path(OUTPUT_DIR, "FigEV3A_roc_curves.tsv"), stringsAsFactors = FALSE)
auc_tab <- read.delim(file.path(OUTPUT_DIR, "FigEV3A_auc.tsv"), stringsAsFactors = FALSE)
auc_of  <- setNames(auc_tab$AUC, auc_tab$model)
legend_lab <- c("TEFL-mRNA" = sprintf("TEFL-mRNA (AUC = %.3f)", auc_of[["TEFL-mRNA"]]),
                "RiboNN"    = sprintf("RiboNN    (AUC = %.3f)", auc_of[["RiboNN"]]))
curves$Model <- legend_lab[curves$model]

p <- ggplot(curves, aes(x = FPR, y = TPR, color = Model)) +
  geom_line(linewidth = 0.75) +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "grey60") +
  scale_color_jco() +
  labs(title = "ROC: TE-high vs Others",
       x = "False Positive Rate", y = "True Positive Rate",
       color = NULL) +
  coord_equal() +
  theme_bw() +
  theme(aspect.ratio = 1, axis.title = element_text(size = 11), axis.text = element_text(size = 11),
        legend.text = element_text(size = 11),
        legend.position = "bottom", legend.direction = "vertical",
        legend.justification = "left", legend.margin = margin(0, 0, 0, 0),
        plot.title = element_text(size = 12, hjust = 0.5))
ggsave(file.path(OUTPUT_DIR, "FigEV3A.pdf"), p, width = 3.6, height = 4.4, device = pdf)
