suppressPackageStartupMessages({
  library(data.table)
  library(ggplot2)
})

file_arg <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
base_dir <- if (length(file_arg)) dirname(dirname(normalizePath(sub("^--file=", "", file_arg)))) else getwd()
output_dir <- file.path(base_dir, "output")

plot_data <- fread(file.path(output_dir, "FigS8D_source_data.tsv"), sep = "\t")
fisher_stats <- fread(file.path(output_dir, "FigS8D_fisher_test.tsv"), sep = "\t")

neg_color <- "#8DD3C7"
pos_color <- "#BEBADA"

p <- ggplot(plot_data, aes(x = Group, y = Proportion, fill = Outcome)) +
  geom_bar(stat = "identity", position = "stack", alpha = 0.9, color = "black", linewidth = 0.3) +
  geom_text(aes(label = paste0(round(Proportion * 100, 1), "%\n(n=", Count, ")")),
            position = position_stack(vjust = 0.5), size = 3.2, color = "black") +
  scale_fill_manual(values = c("TE_decreasing" = neg_color, "TE_increasing" = pos_color)) +
  labs(title = "Proportion Comparison: Pro vs All Groups",
       subtitle = paste0("Fisher's p-value: ", sprintf("%.2e", fisher_stats$p_value),
                         " | Fold enrichment: ", sprintf("%.2fx", fisher_stats$fold_enrichment)),
       x = "Group", y = "Proportion", fill = "Outcome") +
  theme_bw() +
  theme(plot.title = element_text(size = 12, hjust = 0.5, color = "black"),
        plot.subtitle = element_text(size = 10, hjust = 0.5, color = "black"),
        axis.text = element_text(size = 10, color = "black"),
        axis.title = element_text(size = 11, color = "black"),
        legend.title = element_text(size = 11, color = "black"),
        legend.text = element_text(size = 10, color = "black"),
        panel.grid.minor = element_blank(),
        aspect.ratio = 1.5) +
  ylim(0, 1)

# Panel A of the original 2 x 2 layout (10 x 6.5 in); one cell is 5 x 3.13 in
ggsave(file.path(output_dir, "FigS8D.pdf"), p, width = 5, height = 3.13)
