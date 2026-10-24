# Fig. S8C: delta pTE of missense SNVs grouped by the reference amino acid.
suppressPackageStartupMessages(library(data.table))

file_arg <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
base_dir <- if (length(file_arg)) dirname(dirname(normalizePath(sub("^--file=", "", file_arg)))) else getwd()
input_dir <- file.path(base_dir, "input")
output_dir <- file.path(base_dir, "output")
dir.create(output_dir, showWarnings = FALSE)

mis <- fread(file.path(input_dir, "gnomad_v4_missense_SNVs_delta_pTE_filter_0.05.tsv"), sep = "\t")

# x-axis order: reference amino acids by decreasing median delta pTE
aa_order <- levels(reorder(mis$aa_novar, -mis$delta_score_high, FUN = median))

summary_aa <- mis[, .(n = .N, median_delta_score_high = median(delta_score_high)), by = aa_novar]
summary_aa <- summary_aa[match(aa_order, aa_novar)]
print(summary_aa)

fwrite(mis[, .(transcript_refid_var, variant_id, aa_novar, aa_var, delta_score_high)],
       file.path(output_dir, "FigS8C_source_data.tsv"), sep = "\t")
fwrite(summary_aa, file.path(output_dir, "FigS8C_summary.tsv"), sep = "\t")
