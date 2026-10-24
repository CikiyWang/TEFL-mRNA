suppressPackageStartupMessages({
  library(ggplot2)
  library(ggsci)
  library(rrvgo)
})

file_arg <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
base_dir <- if (length(file_arg)) dirname(dirname(normalizePath(sub("^--file=", "", file_arg)))) else getwd()
output_dir <- file.path(base_dir, "output")

rr <- readRDS(file.path(output_dir, "Fig4A_rrvgo.rds"))
df <- read.delim(file.path(output_dir, "Fig4A_source_data.tsv"), row.names = 1)

# fixed seed for the repelled parent-term labels
set.seed(1)
options(ggrepel.max.overlaps = Inf, ggrepel.max.iter = Inf, ggrepel.box.padding = 0.1)

p <- scatterPlot(rr$simMatrix, rr$reducedTerms, labelSize = 3, addLabel = FALSE, size = "size") +
  theme_bw() +
  ggrepel::geom_label_repel(aes(label = parentTerm),
                            data = subset(df, parent == rownames(df)),
                            max.overlaps = Inf, force = 1, max.iter = Inf, alpha = 0.9,
                            min.segment.length = .2, box.padding = grid::unit(0.5, "cm"), size = 3) +
  scale_size_continuous(range = c(2, 15)) +
  scale_color_manual(values = c(pal_aaas()(5), pal_uchicago()(5), pal_nejm()(5), pal_futurama()(5), pal_lancet()(5),
                                pal_jama()(5), pal_rickandmorty()(5), pal_npg()(5), pal_jco()(5))) +
  theme(aspect.ratio = 1,
        axis.text = element_text(size = 13, angle = 0, colour = "black", hjust = 0.5, vjust = 0.5),
        plot.title = element_text(lineheight = 15, size = 12, hjust = 0.5),
        axis.title = element_text(size = 13, color = "black"),
        legend.text = element_text(size = 10.5, color = "black"))
ggsave(file.path(output_dir, "Fig4A.pdf"), p, height = 6, width = 10)
