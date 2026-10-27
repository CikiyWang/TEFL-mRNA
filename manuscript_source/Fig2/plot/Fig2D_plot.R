#!/usr/bin/env Rscript
# Fig. 2D: stacked bars of the fraction of uAUG-creating SNV-mRNA pairs and of RefSeq mRNAs in
# each 0.1-wide pTE bin, with the Fisher's exact test P in the title. Drawn with the default pdf()
# device (Helvetica), which prints "-" as a minus sign, so labels use a soft hyphen (U+00AD).

suppressPackageStartupMessages({
  library(data.table)
  library(ggplot2)
})

args <- commandArgs(trailingOnly = FALSE)
SCRIPT_DIR <- dirname(normalizePath(sub("^--file=", "", args[grep("^--file=", args)][1])))
BASE_DIR   <- normalizePath(file.path(SCRIPT_DIR, ".."))
OUTPUT_DIR <- file.path(BASE_DIR, "output")
invisible(suppressWarnings(Sys.setlocale("LC_CTYPE", "en_US.UTF-8")))   # soft hyphen and multiplication sign

uAUG_count_bins <- as.data.frame(fread(file.path(OUTPUT_DIR, "Fig2D_source_data.tsv")))
ft <- fread(file.path(OUTPUT_DIR, "Fig2D_fisher_test.tsv"))
p_value <- ft$p_value[1]
e <- floor(log10(p_value)); m <- signif(p_value / 10^e, 1)
fisher_lab <- parse(text = sprintf('"Fisher\'s Exact test:  p­value = %s" %%*%% 10^%d', m, e))

n_snvs <- format(uAUG_count_bins$n_unique_snvs[uAUG_count_bins$if_uAUG == "uAUG_creating"][1],
                 big.mark = ",", trim = TRUE)
p <- ggplot(uAUG_count_bins, aes(x = if_uAUG, y = Frac, fill = Class)) +
  geom_bar(stat = "identity", position = "stack", linewidth = 0.5, width = 0.7, color = "black") +
  scale_fill_brewer(palette = "Spectral", direction = -1) +          # 0.9-1 dark red, as published
  scale_x_discrete(labels = c(Reference = "gnomAD\n5'UTR variants",                # published label
                              uAUG_creating = paste0("uAUG­creating\nN=", n_snvs))) +
  theme_classic() +
  # fraction axis stops at 0 and 1, as in the published panel
  guides(x = guide_axis(cap = "both"), fill = guide_legend(reverse = TRUE)) +
  labs(x = NULL, y = "Fractions of mRNAs with SNVs in pTE category", title = fisher_lab) +
  theme(panel.background = element_rect(fill = NA),
        aspect.ratio = 0.4,
        legend.position = "right",
        axis.ticks = element_line(unit(0.1, "cm"), color = "grey20"),
        axis.text.y = element_text(size = 12, color = "black"),
        axis.text.x = element_text(size = 12, color = "black"),
        legend.title = element_blank(),
        legend.text = element_text(size = 10, color = "black"),
        legend.key.size = unit(3.5, "mm"),
        axis.title = element_text(size = 12, color = "black"),
        plot.title = element_text(color = "black", size = 12, hjust = 0.5, vjust = 0.5)) +
  coord_flip()
ggsave(file.path(OUTPUT_DIR, "Fig2D.pdf"), p, width = 6.5, height = 2.75)
