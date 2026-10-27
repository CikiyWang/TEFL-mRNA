# Fig. 4E: the 10 immune-related ClinVar (2024-05-14) SNVs with the largest predicted TE change
# (|delta pTE|) in the 5'UTR and among synonymous CDS variants. The input tables hold the immune-related
# ClinVar SNV-transcript pairs scored with the LCL TEFL-mRNA model (analysis/Fig4E_prepare_input.R).
suppressPackageStartupMessages(library(data.table))

file_arg <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
base_dir <- if (length(file_arg)) dirname(dirname(normalizePath(sub("^--file=", "", file_arg)))) else getwd()
input_dir <- file.path(base_dir, "input")
output_dir <- file.path(base_dir, "output")
dir.create(output_dir, showWarnings = FALSE)

utr5 <- fread(file.path(input_dir, "ClinVar_immune_5UTR_SNVs_dpTE.tsv"))
syn  <- fread(file.path(input_dir, "ClinVar_immune_synonymous_SNVs_dpTE.tsv"))

top10 <- function(d, region) {
  d <- d[order(-abs(delta_score_high))][1:10]
  data.table(delta_score = d$delta_score_high, Region = region, Var_id = d$transcript_refid_var,
             seq_id = d$seq_id, pos = d$pos, Gene = d$gene, ref2alt = paste(d$ref, ">", d$alt),
             clinvar_id = d$clinvar_id, ClinVar_CLNDN = d$ClinVar_CLNDN)
}
df_plot <- rbind(top10(utr5, "5'UTR"), top10(syn, "CDS"))
df_plot[, Name := paste(Gene, "@", seq_id, "@", ref2alt, "@", pos, sep = "")]
print(df_plot[, .(Region, Name, delta_score)])

fwrite(df_plot, file.path(output_dir, "Fig4E_source_data.tsv"), sep = "\t")
