suppressPackageStartupMessages(library(ggplot2))

file_arg <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
base_dir <- if (length(file_arg)) dirname(dirname(normalizePath(sub("^--file=", "", file_arg)))) else getwd()
output_dir <- file.path(base_dir, "output")

AF_CLASSES <- c("AF (<=0.00001)", "AF (0.00001-0.0001)", "AF (0.0001-0.001)", "AF (0.001-0.01)", "AF (>=0.01)")
AF_COLORS  <- setNames(c("#A50F15", "#DE2D26", "#FB6A4A", "#FCAE91", "#FEE5D9"), AF_CLASSES)  # Brewer Reds, darkest = rarest
df_plot <- read.delim(file.path(output_dir, "Fig4E_source_data.tsv"), stringsAsFactors = FALSE)
df_plot$Region <- factor(df_plot$Region, levels = c("5'UTR", "CDS", "3'UTR"))
df_plot$AF_class <- factor(df_plot$AF_class, levels = AF_CLASSES)

# variants listed alphabetically within each region; points give the gnomAD exome AF class
p <- ggplot(df_plot, aes(x = Name, y = delta_score, fill = Region)) +
  geom_bar(stat = "identity", position = position_dodge2(0.75), color = "black", width = 0.75) +
  geom_point(aes(color = AF_class), size = 2) +
  geom_hline(yintercept = 0, linetype = "dashed") +
  scale_fill_manual(values = c("#DDC36F", "#91C4C0", "#917E78")) +
  scale_color_manual(values = AF_COLORS) +
  theme_bw() +
  xlab("") +
  ylab("Delta Predicted TE probability") +
  theme(aspect.ratio = 0.75,
        strip.text = element_text(size = 10, color = "black"),
        axis.text = element_text(size = 10, angle = 0, colour = "black", hjust = 0.5, vjust = 0.5),
        plot.title = element_text(lineheight = 15, size = 11, hjust = 0.5),
        axis.title = element_text(size = 11, color = "black"),
        legend.text = element_text(size = 11, color = "black")) +
  facet_wrap(~Region, scales = "free_y", nrow = 3) +
  coord_flip()
ggsave(file.path(output_dir, "Fig4E.pdf"), p, width = 7.5, height = 6.5)
