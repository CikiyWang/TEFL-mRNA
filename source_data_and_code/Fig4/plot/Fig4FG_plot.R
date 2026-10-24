suppressPackageStartupMessages({
  library(data.table)
  library(ggplot2)
  library(ggpubr)
})

file_arg <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
base_dir <- if (length(file_arg)) dirname(dirname(normalizePath(sub("^--file=", "", file_arg)))) else getwd()
output_dir <- file.path(base_dir, "output")

PANELS <- list(F = list(genes = c("SPPL2A", "SMC2", "IRF3"), alternative = "less", col = "#C05850", width = 6),
               G = list(genes = c("APOA5", "ELP1"), alternative = "greater", col = "#3A4C71", width = 4.4))

# p-value labels printed as in the published panels (0.000065, 6.9e-06)
options(scipen = 1)

# fixed seed for the jittered replicate points
set.seed(1)

for (panel in names(PANELS)) {
  cfg <- PANELS[[panel]]
  d <- as.data.frame(fread(file.path(output_dir, sprintf("Fig4%s_source_data.tsv", panel)), sep = "\t"))
  d$label <- factor(d$label, levels = c("REF", "MUT"))
  p <- ggplot(d, aes(x = label, y = mean, fill = label)) +
    geom_bar(stat = "identity", width = 0.6, position = position_dodge(.9), color = "grey20") +
    geom_jitter(aes(x = label, y = value), width = 0.2) +
    geom_errorbar(aes(ymin = mean - sd, ymax = mean + sd), linewidth = 0.25, width = .15, position = position_dodge(.9)) +
    facet_wrap(~factor(gene, levels = cfg$genes), scales = "free", nrow = 1) +
    scale_fill_manual(values = c("grey", cfg$col)) +
    scale_x_discrete(labels = c(REF = "REF", MUT = "ALT")) +
    scale_y_continuous(expand = expansion(mult = c(0.05, 0.15))) +
    stat_compare_means(aes(x = label, y = value), method = "t.test", label = "..p..",
                       method.args = list(var.equal = TRUE, alternative = cfg$alternative)) +
    labs(x = NULL, y = "Relative expression level\n(rel. to Fluc)") +
    theme_bw() +
    theme(axis.text = element_text(size = 10, angle = 0, colour = "black", hjust = 0.5, vjust = 0.5),
          aspect.ratio = 1.5,
          plot.title = element_text(lineheight = 20, size = 13, hjust = 0.5),
          axis.title = element_text(size = 11, color = "black"),
          legend.position = "none")
  ggsave(file.path(output_dir, sprintf("Fig4%s.pdf", panel)), p, width = cfg$width, height = 2.5)
}
