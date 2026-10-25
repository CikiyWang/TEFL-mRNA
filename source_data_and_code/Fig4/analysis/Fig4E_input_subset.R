#!/usr/bin/env Rscript
# Builds the Fig. 4E inputs: ExAC 5'UTR and synonymous SNV-transcript pairs (delta pTE of the LCL model)
# that are in the ClinVar (2024-05-14) immune disease tables. Not needed to reproduce the figure;
# documents how input/ was made (taken from disease_var/ClinVar_immuno_pred_TE.R in the Translation_DL
# project). Sequence columns are dropped.
#
#   Rscript analysis/Fig4E_input_subset.R \
#     <Pred_All.ExAC_vep_protein_coding_NM.5_prime_UTR_variant.SNV.filter_cor0.7_l0.25_h1.tpe_opt.delta.filter.tsv> \
#     <Pred_All.ExAC_vep_protein_coding_NM.synonymous_variant.SNV.filter_cor0.7_l0.25_h1.tpe_opt.delta.filter.tsv> \
#     <clinvar_20240514_immun.hg38.sorted.refseq_vep.fork.NM_known.5_prime_UTR_variant.SNV.multiprocessing.FL_regions-correct.tsv> \
#     <clinvar_20240514_immun.hg38.sorted.refseq_vep.fork.NM_known.synonymous_variant.SNV.multiprocessing.FL_regions-correct.tsv>

suppressPackageStartupMessages(library(data.table))

args <- commandArgs(trailingOnly = FALSE)
SCRIPT_DIR <- dirname(normalizePath(sub("^--file=", "", args[grep("^--file=", args)][1])))
INPUT_DIR  <- normalizePath(file.path(SCRIPT_DIR, "..", "input"))
a <- commandArgs(trailingOnly = TRUE)
stopifnot(length(a) == 4, all(file.exists(a)))

# the 5'UTR delta table starts with an unnamed row index, which is dropped; the synonymous table keeps it as V1
fastread <- function(file) {
  df <- as.data.frame(fread(file, sep = "\t", stringsAsFactors = FALSE))
  rownames(df) <- df$V1
  df[, -1]
}
UTR5_var_delta <- fastread(a[1])
Syn_var_delta  <- as.data.frame(fread(a[2]))
ClinVar_UTR5 <- read.table(a[3], sep = "\t", header = TRUE)
ClinVar_Syn  <- read.table(a[4], sep = "\t", header = TRUE)
ClinVar_UTR5$transcript_refid_var <- ClinVar_UTR5$transcript_refid
ClinVar_UTR5$Type <- "5'UTR"

# 5'UTR: ExAC pairs joined to their immune ClinVar records (duplicated columns get .x/.y)
utr5 <- merge(UTR5_var_delta, ClinVar_UTR5, by = "transcript_refid_var")
utr5 <- utr5[order(abs(utr5$delta_score_high), decreasing = TRUE), ]
# synonymous: ExAC pairs that are in the immune ClinVar table
syn <- Syn_var_delta[Syn_var_delta$transcript_refid_var %in% ClinVar_Syn$transcript_refid, ]
syn <- syn[order(abs(syn$delta_score_high), decreasing = TRUE), ]

drop_seq <- function(d) d[, !grepl("^Sequences_", colnames(d))]
write.table(drop_seq(utr5), file.path(INPUT_DIR, "ClinVar_UTR5_merge_delta.sel.tsv"),
            sep = "\t", quote = FALSE, row.names = FALSE, col.names = TRUE)
write.table(drop_seq(syn), file.path(INPUT_DIR, "ClinVar_Syn_var_delta.sel.tsv"),
            sep = "\t", quote = FALSE, row.names = FALSE, col.names = TRUE)
