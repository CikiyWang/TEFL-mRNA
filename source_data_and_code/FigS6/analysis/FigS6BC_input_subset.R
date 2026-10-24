#!/usr/bin/env Rscript
# Builds the Fig. S6B-C inputs from the genome-wide gnomAD v4 exome SNV tables (delta pTE per
# variant and RefSeq transcript) and the reference RefSeq mRNA sequences.
# Not needed to reproduce the figure; documents how input/ was made.
#
#   Rscript analysis/FigS6BC_input_subset.R <dir with the gnomAD *.sel_cols.{v4,for_mod}.tsv tables> \
#                                           <pred_ncbi_mRNA.filter_cor0.7_l0.25_h1.tpe_opt.tsv>
#
# Distances (nt), with transcript_refid_var = <RefSeq id>_<REF>_<ALT>_<position in the mRNA>:
#   5'UTR       to start: UTR5_len - pos + 1        (1 = base immediately 5' of the AUG)
#   CDS_start   from start: pos - UTR5_len          (1 = A of the AUG)
#   CDS_stop    to stop: UTR5_len + CDS_len - pos   (0 = last base of the stop codon)
#   3'UTR       from stop: pos - UTR5_len - CDS_len (1 = base immediately 3' of the stop codon)

suppressPackageStartupMessages(library(data.table))

args <- commandArgs(trailingOnly = FALSE)
SCRIPT_DIR <- dirname(normalizePath(sub("^--file=", "", args[grep("^--file=", args)][1])))
INPUT_DIR  <- normalizePath(file.path(SCRIPT_DIR, "..", "input"))
a <- commandArgs(trailingOnly = TRUE)
stopifnot(length(a) == 2, dir.exists(a[1]), file.exists(a[2]))
GNOMAD <- function(cls, ver) file.path(a[1], sprintf(
  "Pred_All.gnomad.exomes.v4.0.sites.PASS_pc_SNV_NM.%s.filter_cor0.7_l0.25_h1.TPE_opt.delta.filter.sel_cols.%s.tsv", cls, ver))
WINDOW <- 1000

parse_id <- function(d) d[, `:=`(seq_id = sub("^([^_]+_[^_]+)_.*$", "\\1", transcript_refid_var),
                                 alt = sub("^[^_]+_[^_]+_[ACGT]_([ACGT])_[0-9]+$", "\\1", transcript_refid_var),
                                 pos = as.integer(sub("^.*_([0-9]+)$", "\\1", transcript_refid_var)))]

# region lengths of the reference transcripts the variants were scored against
ref <- fread(a[2], sep = "\t", colClasses = "character", select = c("transcript_refid", "Sequences_UTR5", "Sequences_CDS"))
ref <- ref[, .(seq_id = transcript_refid, UTR5_length = nchar(Sequences_UTR5), CDS_length = nchar(Sequences_CDS))]
stopifnot(!anyDuplicated(ref$seq_id))

# delta_score_high is kept as text so the values are written out unchanged
u5 <- fread(GNOMAD("5_prime_UTR_variant", "v4"), colClasses = c(delta_score_high = "character"),
            select = c("transcript_refid_var", "UTR5_len", "delta_score_high"))[!is.na(transcript_refid_var)]
syn <- fread(GNOMAD("synonymous_variant", "for_mod"), colClasses = c(delta_score_high = "character"),
             select = c("transcript_refid_var", "delta_score_high"))[!is.na(transcript_refid_var)]
u3 <- fread(GNOMAD("3_prime_UTR_variant", "for_mod"), colClasses = c(delta_score_high = "character"),
            select = c("transcript_refid_var", "delta_score_high"))[!is.na(transcript_refid_var)]
for (d in list(u5, syn, u3)) parse_id(d)
syn <- merge(syn, ref, by = "seq_id", sort = FALSE)
u3  <- merge(u3,  ref, by = "seq_id", sort = FALSE)

win <- rbind(
  u5[,  .(region = "5'UTR",     transcript_refid_var, alt, distance_nt = UTR5_len - pos + 1L, delta_score_high)],
  syn[, .(region = "CDS_start", transcript_refid_var, alt, distance_nt = pos - UTR5_length, delta_score_high)],
  syn[, .(region = "CDS_stop",  transcript_refid_var, alt, distance_nt = UTR5_length + CDS_length - pos, delta_score_high)],
  u3[,  .(region = "3'UTR",     transcript_refid_var, alt, distance_nt = pos - UTR5_length - CDS_length, delta_score_high)])
stopifnot(all(win$distance_nt >= 0))
win <- win[distance_nt <= WINDOW]
fwrite(win, file.path(INPUT_DIR, "gnomad_v4_SNVs_delta_pTE.start_stop_1kb.tsv.gz"), sep = "\t", compress = "gzip")
print(win[, .N, by = region])

# Fig. S6C uses only 5'UTR SNVs at or beyond the smallest cutoff (|delta pTE| >= 0.1)
c5 <- u5[abs(as.numeric(delta_score_high)) >= 0.1, .(transcript_refid_var, delta_score_high)]
fwrite(c5, file.path(INPUT_DIR, "gnomad_v4_5UTR_SNVs_delta_pTE.abs_ge_0.1.tsv"), sep = "\t")
cat("5'UTR SNVs:", nrow(u5), "in total,", nrow(c5), "with |delta pTE| >= 0.1\n")
