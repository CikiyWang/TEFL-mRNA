suppressPackageStartupMessages({
  library(data.table)
  library(ggplot2)
})

file_arg <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
base_dir <- if (length(file_arg)) dirname(dirname(normalizePath(sub("^--file=", "", file_arg)))) else getwd()
output_dir <- file.path(base_dir, "output")

plot_data <- fread(file.path(output_dir, "FigEV5C_source_data.tsv"), sep = "\t")
plot_data$PolyP_Length_Factor <- as.factor(plot_data$Mutated_PolyP_Length)
summary_stats <- fread(file.path(output_dir, "FigEV5C_summary.tsv"), sep = "\t")

n_groups <- length(unique(plot_data$Mutated_PolyP_Length))
colors <- c("#2E86AB", "#A23B72", "#F18F01", "#C73E1D", "#8E44AD", "#27AE60", "#E67E22")
polyp_colors <- colors[1:min(n_groups, length(colors))]

# fixed seed for the jittered points
set.seed(1)

p <- ggplot(plot_data, aes(x = PolyP_Length_Factor, y = delta_score_high, fill = PolyP_Length_Factor)) +
  geom_violin(alpha = 0.2) +
  geom_boxplot(alpha = 1, outlier.shape = 16, outlier.alpha = 0.6, color = "black", linewidth = 0.5, width = 0.1) +
  geom_jitter(width = 0.2, alpha = 0.4, size = 0.2, color = "black") +
  scale_fill_manual(values = polyp_colors, name = "PolyP Length") +
  scale_y_continuous(expand = c(0.005, 0.015)) +
  labs(
    title = "Delta Score High by Mutated PolyP Length",
    subtitle = paste0("Distribution across ", nrow(plot_data), " variants with Pro-negative outcomes"),
    x = "Mutated PolyP Length (Number of consecutive P's)",
    y = "Delta Score High",
    caption = "White diamonds indicate group means; dots show individual data points"
  ) +
  theme_bw() +
  theme(
    plot.title = element_text(size = 14, hjust = 0.5, color = "black"),
    plot.subtitle = element_text(size = 12, hjust = 0.5, color = "black"),
    plot.caption = element_text(size = 10, hjust = 0.5, color = "black"),
    axis.text = element_text(size = 11, color = "black"),
    axis.title = element_text(size = 12, color = "black"),
    legend.title = element_text(size = 11, color = "black"),
    legend.text = element_text(size = 10, color = "black"),
    panel.grid.minor = element_blank(),
    aspect.ratio = 1.5
  ) +
  geom_text(data = summary_stats,
            aes(x = as.factor(Mutated_PolyP_Length), y = Inf, label = paste0("n=", n)),
            vjust = 1.5, hjust = 0.5, size = 3.5, color = "black", inherit.aes = FALSE)

ggsave(file.path(output_dir, "FigEV5C.pdf"), p, width = 6, height = 4)
