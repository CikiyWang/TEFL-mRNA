#!/usr/bin/env Rscript
# Builds the Fig. 2G inputs: the qPCR table of the GM12878 reporter validation (copied unchanged)
# and the six validated gnomAD 5'UTR variants, with each REF base checked against the RefSeq mRNA
# sequence and the predicted delta pTE of each variant taken from the gnomAD 5'UTR table. Not
# needed to reproduce the figure; documents how input/ was made.
#
#   Rscript analysis/Fig2G_input_subset.R <Experimental_validation_GM_qPCR.tsv> \
#     <pred_ncbi_mRNA.filter_cor0.7_l0.25_h1.tpe_opt.tsv> \
#     <Pred_All.gnomad.exomes.v4.0.sites.PASS_pc_SNV_NM.5_prime_UTR_variant.filter_cor0.7_l0.25_h1.TPE_opt.delta.filter.sel_cols.for_mod.tsv>

suppressPackageStartupMessages(library(data.table))

args <- commandArgs(trailingOnly = FALSE)
SCRIPT_DIR <- dirname(normalizePath(sub("^--file=", "", args[grep("^--file=", args)][1])))
INPUT_DIR  <- normalizePath(file.path(SCRIPT_DIR, "..", "input"))
a <- commandArgs(trailingOnly = TRUE)
stopifnot(length(a) == 3, all(file.exists(a)))

stopifnot(file.copy(a[1], file.path(INPUT_DIR, basename(a[1])), overwrite = TRUE))

# validated variants: position in the full-length mRNA (all in the 5'UTR), REF and ALT as DNA bases;
# left group = predicted to increase pTE, right group = predicted to decrease it
variants <- data.table(
  group         = c("left", "left", "left", "right", "right", "right"),
  gene          = c("CCL14", "BCL6", "MLH1", "PIK3C3", "H3C4", "HLA-G"),
  transcript_id = c("NM_032963", "NM_001706", "NM_001258271", "NM_001308020", "NM_001376937", "NM_001363567"),
  position      = c(10L, 10L, 9L, 14L, 12L, 29L),
  ref           = c("C", "G", "C", "T", "A", "T"),
  alt           = c("A", "A", "T", "C", "C", "A"))
seqs <- fread(a[2], select = c("transcript_refid", "Sequences_UTR5", "Sequences_FL"))
variants[seqs, on = c(transcript_id = "transcript_refid"),
         `:=`(ref_check = substring(i.Sequences_FL, position, position), utr5_len = nchar(i.Sequences_UTR5))]
stopifnot(variants[, all(ref_check == ref & position <= utr5_len)])

# delta pTE (ALT - REF) of each variant, read as text so it is written out unchanged
variants[, transcript_refid_var := paste(transcript_id, ref, alt, position, sep = "_")]
gnomad <- fread(a[3], select = c("transcript_refid_var", "variant_id", "delta_score_high"), colClasses = "character")
variants[gnomad, on = "transcript_refid_var", `:=`(gnomad_variant_id = i.variant_id, delta_pTE = i.delta_score_high)]
stopifnot(!anyNA(variants$delta_pTE))
fwrite(variants[, .(group, gene, transcript_id, position, ref, alt, gnomad_variant_id, delta_pTE)],
       file.path(INPUT_DIR, "Fig2G_validated_variants.tsv"), sep = "\t")
