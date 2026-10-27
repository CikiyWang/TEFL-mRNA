#!/usr/bin/env Rscript
# Builds the Fig. 1F site table from the saved intermediate table of miRNA binding sites. Not needed
# to reproduce the figure; documents how input/ was made. analysis/Fig1F_prepare_input.R builds the
# same table from the downloaded databases.
#
#   Rscript analysis/Fig1F_prepare_input_from_intermediate.R <pred_score_all_miRNA_merge_exp.tsv>
#
# pred_score_all_miRNA_merge_exp.tsv: miRWalk sites (binding probability 1) of miRNAs expressed in
# naive B cells on RefSeq mRNAs, one row per site, with the miRNA expression; its prediction columns
# come from another model version and are not used.

suppressPackageStartupMessages(library(data.table))

file_arg <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
input_dir <- normalizePath(file.path(dirname(dirname(normalizePath(sub("^--file=", "", file_arg)))), "input"))
a <- commandArgs(trailingOnly = TRUE)
stopifnot(length(a) == 1, file.exists(a))

SITE_COLS <- c("seq_id", "Genesymbol", "miRNA", "Name", "duplex", "binding_probability", "region", "RPM")
# values are read as text so they are written out unchanged (surrounding spaces removed)
sites <- fread(a[1], sep = "\t", colClasses = "character", select = SITE_COLS)
sites <- sites[region == "3'UTR"]
fwrite(sites, file.path(input_dir, "miRWalk_3UTR_sites_B_naive_miRNAs.tsv"), sep = "\t")
cat(nrow(sites), "3'UTR sites on", uniqueN(sites$seq_id), "mRNAs\n")
