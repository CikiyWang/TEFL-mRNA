# Fig. 1F: fraction of mRNAs with and without a 3'UTR miRNA binding site in each 0.1-wide pTE bin, as
# horizontal stacked bars with the N of each group and the Fisher's exact test at pTE 0.5, as in the
# published panel. Drawn with the default pdf() device (Helvetica), which prints "-" as a minus sign,
# so the label uses a soft hyphen (U+00AD).
suppressPackageStartupMessages({
  library(data.table)
  library(ggplot2)
})

file_arg <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
base_dir <- if (length(file_arg)) dirname(dirname(normalizePath(sub("^--file=", "", file_arg)))) else getwd()
output_dir <- file.path(base_dir, "output")
invisible(suppressWarnings(Sys.setlocale("LC_CTYPE", "en_US.UTF-8")))   # soft hyphen and multiplication sign

cutoffs <- c(0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
BIN_LABELS <- paste(cutoffs, "-", cutoffs + 0.1, sep = "")
GROUPS <- c("without miRNA BS", "with miRNA BS")   # second level drawn on top

miBS_bins <- fread(file.path(output_dir, "Fig1F_bin_fractions.tsv"), sep = "\t")
miBS_bins[, `:=`(Class = factor(Class, levels = rev(BIN_LABELS)), if_miBS = factor(if_miBS, levels = GROUPS))]
n <- miBS_bins[, .(n = sum(count)), by = if_miBS]
group_labels <- setNames(paste0("N=", format(n$n, big.mark = ",", trim = TRUE), "\n", n$if_miBS), n$if_miBS)

tests <- fread(file.path(output_dir, "Fig1F_tests.tsv"), sep = "\t")
p_value <- tests[test == "Fisher" & cutoff == 0.5, p_value]
e <- floor(log10(p_value)); m <- signif(p_value / 10^e, 2)
fisher_lab <- parse(text = sprintf('"Fisher\'s Exact test:  p­value = %s" %%*%% 10^%d', m, e))

p <- ggplot(miBS_bins, aes(x = if_miBS, y = Frac, fill = Class)) +
  geom_bar(stat = "identity", position = "stack", linewidth = 0.5, width = 0.7, color = "black") +
  scale_fill_brewer(palette = "Spectral", name = "pTE categories") +   # 0.9-1 dark red, 0-0.1 purple
  scale_x_discrete(labels = group_labels) +
  scale_y_continuous(position = "right", breaks = c(0, 0.25, 0.5, 0.75, 1), labels = c("0", "0.25", "0.5", "0.75", "1")) +
  coord_flip() +
  theme_classic() +
  guides(y = guide_axis(cap = "both")) +
  labs(x = NULL, y = "Fractions of mRNAs in pTE category", caption = fisher_lab) +
  theme(aspect.ratio = 0.4,
        axis.line.y = element_blank(),
        axis.ticks.y = element_blank(),
        axis.ticks.x = element_line(color = "grey20"),
        axis.text.y = element_text(size = 12, color = "black"),
        axis.text.x = element_text(size = 12, color = "black"),
        axis.title = element_text(size = 12, color = "black"),
        legend.title = element_text(size = 12, color = "black"),
        legend.text = element_text(size = 10, color = "black"),
        legend.key.size = unit(3.5, "mm"),
        plot.caption = element_text(size = 12, color = "black", hjust = 0.5))
ggsave(file.path(output_dir, "Fig1F.pdf"), p, width = 6.5, height = 2.75)
