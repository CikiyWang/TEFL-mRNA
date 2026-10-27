suppressPackageStartupMessages({
  library(data.table)
  library(ggplot2)
  library(ggsci)
})

file_arg <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
base_dir <- if (length(file_arg)) dirname(dirname(normalizePath(sub("^--file=", "", file_arg)))) else getwd()
output_dir <- file.path(base_dir, "output")

prop_df <- fread(file.path(output_dir, "Fig5C_source_data.tsv"), sep = "\t")
prop_df[, category := factor(category, levels = c("Shared", "Specific"))]

p <- ggplot(prop_df, aes(x = region, y = proportion, fill = category)) +
  geom_bar(stat = "identity", alpha = 0.7, width = 0.8) +
  scale_fill_futurama(labels = c(Specific = "Others", Shared = "Concordance score > 0.8")) +
  geom_text(aes(label = paste0("n=", count, "\n(", round(proportion * 100, 1), "%)")),
            hjust = 0, size = 3.5, color = "black") +
  labs(x = NULL, y = "Fraction of SNVs", fill = NULL) +
  theme_bw() +
  scale_x_discrete(limits = c("3'UTR", "CDS synonymous", "CDS missense", "5'UTR")) +
  scale_y_continuous(limits = c(0, 1.2), breaks = seq(0, 1.2, 0.3)) +
  coord_flip() +
  theme(
    axis.text.x = element_text(size = 12, color = "black"),
    axis.text.y = element_text(size = 12, color = "black"),
    axis.title = element_text(size = 12, color = "black"),
    legend.text = element_text(size = 11, color = "black"),
    legend.position = "bottom",
    legend.direction = "vertical",
    aspect.ratio = 0.55
  )
ggsave(file.path(output_dir, "Fig5C.pdf"), p, width = 6.5, height = 3.5)
