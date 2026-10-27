# Fig. EV5A: delta pTE of missense SNVs (|delta pTE| > 0.05) grouped by the amino acid they introduce.
suppressPackageStartupMessages(library(data.table))

file_arg <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
base_dir <- if (length(file_arg)) dirname(dirname(normalizePath(sub("^--file=", "", file_arg)))) else getwd()
shared_dir <- file.path(dirname(dirname(base_dir)), "shared", "input")
output_dir <- file.path(base_dir, "output")
dir.create(output_dir, showWarnings = FALSE)

mis <- fread(file.path(shared_dir, "gnomAD_v4_missense_SNVs_dpTE_abs_gt_0.05.tsv.gz"), sep = "\t")

# x-axis order: introduced amino acids by decreasing median delta pTE
aa_order <- levels(reorder(mis$aa_var, -mis$delta_score_high, FUN = median))

summary_aa <- mis[, .(n = .N, median_delta_score_high = median(delta_score_high)), by = aa_var]
summary_aa <- summary_aa[match(aa_order, aa_var)]
print(summary_aa)

fwrite(mis[, .(transcript_refid_var, variant_id, aa_novar, aa_var, delta_score_high)],
       file.path(output_dir, "FigEV5A_source_data.tsv"), sep = "\t")
fwrite(summary_aa, file.path(output_dir, "FigEV5A_summary.tsv"), sep = "\t")
