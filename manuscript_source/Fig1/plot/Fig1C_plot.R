suppressPackageStartupMessages({
  library(data.table)
  library(ggplot2)
  library(ggpubr)
})

file_arg <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
base_dir <- if (length(file_arg)) dirname(dirname(normalizePath(sub("^--file=", "", file_arg)))) else getwd()
output_dir <- file.path(base_dir, "output")

CRITERIA <- c("AUC", "PRAUC", "Precision", "Recall", "Accuracy", "F1-score")
df <- as.data.frame(fread(file.path(output_dir, "Fig1C_source_data.tsv"), sep = "\t"))
df$Criteria <- factor(df$Criteria, levels = CRITERIA)

# fixed seed for the jittered fold points
set.seed(1)

p <- ggbarplot(df, x = "Criteria", y = "perf", fill = "Criteria", merge = TRUE, add = c("mean", "jitter")) +
  geom_errorbar(aes(group = Criteria, ymax = upper_ci, ymin = lower_ci),
                position = position_dodge(width = 0.8), width = 0.25) +
  scale_fill_manual(values = c("#30312C", "#B6A98A", "#6FAB97", "#AF8344", "#FFEB97", "#527590")) +
  scale_x_discrete(limits = rev(CRITERIA)) +
  ylim(c(0, 1)) +
  coord_flip() +
  xlab("") +
  ylab("") +
  facet_wrap(~class, nrow = 2, scale = "free") +
  theme(aspect.ratio = 0.5,
        legend.position = "right",
        axis.text = element_text(size = 11, angle = 0, colour = "black", hjust = 0.5, vjust = 0.5),
        plot.title = element_text(lineheight = 15, size = 10, hjust = 0.5),
        axis.title = element_text(size = 11, color = "black"),
        legend.text = element_text(size = 11, color = "black"))
ggsave(file.path(output_dir, "Fig1C.pdf"), p, height = 5.5, width = 5.5)
