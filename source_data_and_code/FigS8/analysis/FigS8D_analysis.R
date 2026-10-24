# Fig. S8D: fraction of TE-decreasing (delta pTE < 0) SNVs among proline-creating
# missense SNVs (ALT amino acid = Pro) compared with all missense SNVs.
suppressPackageStartupMessages(library(data.table))

file_arg <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
base_dir <- if (length(file_arg)) dirname(dirname(normalizePath(sub("^--file=", "", file_arg)))) else getwd()
input_dir <- file.path(base_dir, "input")
output_dir <- file.path(base_dir, "output")
dir.create(output_dir, showWarnings = FALSE)

mis <- fread(file.path(input_dir, "gnomad_v4_missense_SNVs_delta_pTE_filter_0.05.tsv"), sep = "\t")
mis_pro <- mis[aa_var == "Pro"]

pro_total_count <- nrow(mis_pro)
pro_neg_count <- sum(mis_pro$delta_score_high < 0)
pro_pos_count <- pro_total_count - pro_neg_count
all_total_count <- nrow(mis)
all_neg_count <- sum(mis$delta_score_high < 0)
all_pos_count <- all_total_count - all_neg_count

contingency_table <- matrix(c(pro_neg_count, pro_pos_count,
                              all_neg_count, all_pos_count),
                            nrow = 2, byrow = TRUE,
                            dimnames = list(c("Pro_SNVs", "All_SNVs"), c("TE_decreasing", "TE_increasing")))
print(contingency_table)
fisher_result <- fisher.test(contingency_table)

plot_data <- data.frame(
  Group = c("Pro_SNVs", "All_SNVs", "Pro_SNVs", "All_SNVs"),
  Outcome = c("TE_decreasing", "TE_decreasing", "TE_increasing", "TE_increasing"),
  Count = c(pro_neg_count, all_neg_count, pro_pos_count, all_pos_count),
  Proportion = c(pro_neg_count / pro_total_count, all_neg_count / all_total_count,
                 pro_pos_count / pro_total_count, all_pos_count / all_total_count),
  Total = c(pro_total_count, all_total_count, pro_total_count, all_total_count)
)

fisher_stats <- data.frame(
  fold_enrichment = (pro_neg_count / pro_total_count) / (all_neg_count / all_total_count),
  odds_ratio = unname(fisher_result$estimate),
  conf_int_low = fisher_result$conf.int[1],
  conf_int_high = fisher_result$conf.int[2],
  p_value = fisher_result$p.value
)
print(fisher_stats)

fwrite(plot_data, file.path(output_dir, "FigS8D_source_data.tsv"), sep = "\t")
fwrite(fisher_stats, file.path(output_dir, "FigS8D_fisher_test.tsv"), sep = "\t")
