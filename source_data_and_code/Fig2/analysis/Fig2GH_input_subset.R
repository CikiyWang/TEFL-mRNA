#!/usr/bin/env Rscript
# Builds the Fig. 2G and 2H inputs: the qPCR table of the GM12878 reporter validation (copied
# unchanged) and the six validated 5'UTR variants, with each REF base checked against the RefSeq
# mRNA sequence. Not needed to reproduce the figure; documents how input/ was made.
#
#   Rscript analysis/Fig2GH_input_subset.R <Experimental_validation_GM_qPCR.tsv> \
#                                          <pred_ncbi_mRNA.filter_cor0.7_l0.25_h1.tpe_opt.tsv>

suppressPackageStartupMessages(library(data.table))

args <- commandArgs(trailingOnly = FALSE)
SCRIPT_DIR <- dirname(normalizePath(sub("^--file=", "", args[grep("^--file=", args)][1])))
INPUT_DIR  <- normalizePath(file.path(SCRIPT_DIR, "..", "input"))
a <- commandArgs(trailingOnly = TRUE)
stopifnot(length(a) == 2, all(file.exists(a)))

stopifnot(file.copy(a[1], file.path(INPUT_DIR, basename(a[1])), overwrite = TRUE))

# validated variants: position in the full-length mRNA (all in the 5'UTR), REF and ALT as DNA bases;
# predicted direction of the TE change: up (Fig. 2G) or down (Fig. 2H)
variants <- data.table(
  panel         = c("G", "G", "G", "H", "H", "H"),
  gene          = c("CCL14", "BCL6", "MLH1", "PIK3C3", "H3C4", "HLA-G"),
  transcript_id = c("NM_032963", "NM_001706", "NM_001258271", "NM_001308020", "NM_001376937", "NM_001363567"),
  position      = c(10L, 10L, 9L, 14L, 12L, 29L),
  ref           = c("C", "G", "C", "T", "A", "T"),
  alt           = c("A", "A", "T", "C", "C", "A"))
seqs <- fread(a[2], select = c("transcript_refid", "Sequences_UTR5", "Sequences_FL"))
variants[seqs, on = c(transcript_id = "transcript_refid"),
         `:=`(ref_check = substring(i.Sequences_FL, position, position), utr5_len = nchar(i.Sequences_UTR5))]
stopifnot(variants[, all(ref_check == ref & position <= utr5_len)])
fwrite(variants[, .(panel, gene, transcript_id, position, ref, alt)],
       file.path(INPUT_DIR, "Fig2GH_validated_variants.tsv"), sep = "\t")
