#!/usr/bin/env Rscript
# Derive per-variant 5'UTR covariates from the full sequence/prediction table.
# The source table (~250 MB) is not distributed with this package; its product,
# input/dbSNP_common_5UTR_variant_covariates.tsv, is supplied instead. Re-run only to
# regenerate that file:
#   Rscript analysis/FigEV4HI_prepare_input.R <sequence_table.tsv>

suppressMessages(library(data.table))
source(file.path(dirname(sub("^--file=", "", commandArgs(FALSE)[grep("^--file=", commandArgs(FALSE))])), "common.R"))

seq_file <- commandArgs(trailingOnly = TRUE)[1]
if (is.na(seq_file)) stop("usage: Rscript analysis/FigEV4HI_prepare_input.R <sequence_table.tsv>")

sq <- fread(seq_file, select = c("transcript_refid", "Sequences_UTR5", "Sequences_CDS"),
            showProgress = FALSE)
setnames(sq, "transcript_refid", "transcript_refid_var")

# Variant id encodes ..._<ref>_<alt>_<position in 5'UTR, 1-based>
parts <- tstrsplit(sq$transcript_refid_var, "_", fixed = TRUE); np <- length(parts)
sq[, `:=`(pos = as.integer(parts[[np]]), alt = parts[[np - 1]], ref = parts[[np - 2]])]
sq[, utr5_len := nchar(Sequences_UTR5)]
sq <- sq[pos >= 1 & pos <= utr5_len]

# GC window may run into the CDS for variants near the start codon
sq[, seq5cds := paste0(Sequences_UTR5, Sequences_CDS)]
sq[, `:=`(w_lo = pmax(1L, pos - GC_FLANK), w_hi = pmin(nchar(seq5cds), pos + GC_FLANK))]
sq[, win := substr(seq5cds, w_lo, w_hi)]
sq[, `:=`(win_len = w_hi - w_lo + 1L, gc_n = nchar(gsub("[^GCgc]", "", win)))]

# Sequences carry the alternate allele; swap it back to report reference GC
sq[, gc_ref := (gc_n - (alt %in% c("G", "C")) + (ref %in% c("G", "C"))) / win_len]
sq[, d2start := utr5_len - pos]

fwrite(sq[, .(transcript_refid_var, utr5_len, d2start, gc_ref)],
       file.path(INPUT_DIR, "dbSNP_common_5UTR_variant_covariates.tsv"), sep = "\t")
