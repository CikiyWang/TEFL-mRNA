#!/usr/bin/env Rscript
# Builds the Fig. 1F inputs from the saved intermediate table of miRNA binding sites and the RefSeq
# mRNA predictions. Not needed to reproduce the figure; documents how input/ was made.
# analysis/Fig1F_preprocess_miRNA.R builds the same site table from the downloaded databases.
#
#   Rscript analysis/Fig1F_input_subset.R <pred_score_all_miRNA_merge_exp.tsv> \
#                                         <pred_ncbi_mRNA.filter_cor0.7_l0.25_h1.tpe_opt.tsv>
#
# pred_score_all_miRNA_merge_exp.tsv: miRWalk sites (binding probability 1) of miRNAs expressed in
# naive B cells on RefSeq mRNAs, one row per site, with the miRNA expression; its prediction columns
# come from another model version and are not used (pTE is taken from the second table).

suppressPackageStartupMessages(library(data.table))

file_arg <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
input_dir <- normalizePath(file.path(dirname(dirname(normalizePath(sub("^--file=", "", file_arg)))), "input"))
a <- commandArgs(trailingOnly = TRUE)
stopifnot(length(a) == 2, all(file.exists(a)))

SITE_COLS <- c("seq_id", "miRNA", "Genesymbol", "duplex", "binding_probability", "region",
               "Name", "Accession", "RPM", "Chromosome", "Start", "End", "Strand", "Confidence", "log2RPM")
# values are read as text so they are written out unchanged (surrounding spaces removed)
sites <- fread(a[1], sep = "\t", colClasses = "character", select = SITE_COLS)
fwrite(sites, file.path(input_dir, "miRWalk_prob1_sites.B_naive_miRNAs.tsv"), sep = "\t")

pte <- fread(a[2], sep = "\t", colClasses = "character", select = c("transcript_refid", "pred_0"))
fwrite(pte, file.path(input_dir, "pred_ncbi_mRNA.filter_cor0.7_l0.25_h1.tpe_opt.pTE.tsv"), sep = "\t")
cat(nrow(sites), "sites on", uniqueN(sites$seq_id), "mRNAs;", nrow(pte), "mRNAs with pTE\n")
