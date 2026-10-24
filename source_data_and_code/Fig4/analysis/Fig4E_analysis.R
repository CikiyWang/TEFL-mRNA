# Fig. 4E: ClinVar (2024-05-14) SNVs in immune disease genes with the largest predicted TE change
# (|delta pTE|, top 10 per region) in the 5'UTR and among synonymous CDS variants.
suppressPackageStartupMessages(library(data.table))

file_arg <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
base_dir <- if (length(file_arg)) dirname(dirname(normalizePath(sub("^--file=", "", file_arg)))) else getwd()
input_dir <- file.path(base_dir, "input")
output_dir <- file.path(base_dir, "output")
dir.create(output_dir, showWarnings = FALSE)

# The first column of the 5'UTR delta table is an unnamed row index
fastread <- function(file, sep) {
  df <- as.data.frame(fread(file, sep = sep, stringsAsFactors = FALSE))
  rownames(df) <- df$V1
  df[, -1]
}

UTR5_var_delta <- fastread(file.path(input_dir, "Pred_All.ExAC_vep_protein_coding_NM.5_prime_UTR_variant.SNV.filter_cor0.7_l0.25_h1.tpe_opt.delta.filter.tsv"), sep = "\t")
Syn_var_delta <- as.data.frame(fread(file.path(input_dir, "Pred_All.ExAC_vep_protein_coding_NM.synonymous_variant.SNV.filter_cor0.7_l0.25_h1.tpe_opt.delta.filter.tsv")))

ClinVar_UTR5 <- read.table(file.path(input_dir, "clinvar_20240514_immun.hg38.sorted.refseq_vep.fork.NM_known.5_prime_UTR_variant.SNV.multiprocessing.FL_regions-correct.tsv"),
                           sep = "\t", header = TRUE)
ClinVar_Syn <- read.table(file.path(input_dir, "clinvar_20240514_immun.hg38.sorted.refseq_vep.fork.NM_known.synonymous_variant.SNV.multiprocessing.FL_regions-correct.tsv"),
                          sep = "\t", header = TRUE)
ClinVar_UTR5$transcript_refid_var <- ClinVar_UTR5$transcript_refid
ClinVar_Syn$transcript_refid_var <- ClinVar_Syn$transcript_refid

top10 <- function(delta, clinvar, region) {
  m <- merge(delta, clinvar, by = "transcript_refid_var")
  m <- m[order(abs(m$delta_score_high), decreasing = TRUE), ][1:10, ]
  data.frame(delta_score = m$delta_score_high, Region = region, Var_id = m$transcript_refid_var,
             seq_id = m$seq_id, pos = m$pos, Gene = m$gene.x, AF = as.numeric(m$gnomade_AF.x), ref2alt = m$ref_alt)
}
df_plot <- rbind(top10(UTR5_var_delta, ClinVar_UTR5, "5'UTR"), top10(Syn_var_delta, ClinVar_Syn, "CDS"))
df_plot$Name <- paste(df_plot$Gene, "@", df_plot$seq_id, "@", df_plot$ref2alt, "@", df_plot$pos, sep = "")
print(df_plot[, c("Region", "Name", "delta_score")])

write.table(df_plot, file.path(output_dir, "Fig4E_source_data.tsv"), sep = "\t", quote = FALSE, row.names = FALSE)
