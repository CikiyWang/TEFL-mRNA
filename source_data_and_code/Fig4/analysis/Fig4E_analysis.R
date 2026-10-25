# Fig. 4E: the 10 immune-related ClinVar (2024-05-14) SNVs with the largest predicted TE change
# (|delta pTE|) in the 5'UTR and among synonymous CDS variants. Both input tables hold ExAC
# SNV-transcript pairs that are in the ClinVar immune disease tables (analysis/Fig4E_input_subset.R).
suppressPackageStartupMessages(library(data.table))

file_arg <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
base_dir <- if (length(file_arg)) dirname(dirname(normalizePath(sub("^--file=", "", file_arg)))) else getwd()
input_dir <- file.path(base_dir, "input")
output_dir <- file.path(base_dir, "output")
dir.create(output_dir, showWarnings = FALSE)

utr5 <- fread(file.path(input_dir, "ClinVar_UTR5_merge_delta.sel.tsv"))
syn  <- fread(file.path(input_dir, "ClinVar_Syn_var_delta.sel.tsv"))
# in the merged 5'UTR table, gene.x and gnomade_AF.x come from the ExAC prediction table
setnames(utr5, c("gene.x", "gnomade_AF.x"), c("gene", "gnomade_AF"))

# gnomAD exome allele-frequency classes of the points
AF_CLASSES <- c("AF (<=0.00001)", "AF (0.00001-0.0001)", "AF (0.0001-0.001)", "AF (0.001-0.01)", "AF (>=0.01)")
af_class <- function(af) {
  factor(ifelse(af <= 1e-5, AF_CLASSES[1], ifelse(af < 1e-4, AF_CLASSES[2], ifelse(af < 1e-3, AF_CLASSES[3],
         ifelse(af < 1e-2, AF_CLASSES[4], AF_CLASSES[5])))), levels = AF_CLASSES)
}

top10 <- function(d, region) {
  d <- d[order(-abs(delta_score_high))][1:10]
  data.table(delta_score = d$delta_score_high, Region = region, Var_id = d$transcript_refid_var,
             seq_id = d$seq_id, pos = d$pos, Gene = d$gene, AF = as.numeric(d$gnomade_AF), ref2alt = d$ref_alt)
}
df_plot <- rbind(top10(utr5, "5'UTR"), top10(syn, "CDS"))
df_plot[, AF_class := af_class(AF)]
df_plot[, Name := paste(Gene, "@", seq_id, "@", ref2alt, "@", pos, sep = "")]
print(df_plot[, .(Region, Name, delta_score, AF, AF_class)])

fwrite(df_plot, file.path(output_dir, "Fig4E_source_data.tsv"), sep = "\t")
