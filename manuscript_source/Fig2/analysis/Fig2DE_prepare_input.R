#!/usr/bin/env Rscript
# Builds the Fig. 2D and 2E input: pTE of all possible uAUG-creating 5'UTR SNVs (scored on the
# mutated sequence) with the reference sequence around each variant site; pTE of the RefSeq mRNAs
# is shared/input/RefSeq_mRNA_pTE.tsv. The sequence context replaces the full-length
# sequences (about 0.5 GB) for the check that the ALT allele forms an ATG. Not needed to reproduce
# the figure; documents how input/ was made.
#
#   Rscript analysis/Fig2DE_prepare_input.R \
#     <Pred_All.uAUG-creating_all_possible_annotated.refseq_vep.5_prime_UTR_variant.SNV.filter_cor0.7_l0.25_h1.tpe_opt.tsv> \
#     <pred_ncbi_mRNA.filter_cor0.7_l0.25_h1.tpe_opt.tsv>

suppressPackageStartupMessages(library(data.table))

args <- commandArgs(trailingOnly = FALSE)
SCRIPT_DIR <- dirname(normalizePath(sub("^--file=", "", args[grep("^--file=", args)][1])))
INPUT_DIR  <- normalizePath(file.path(SCRIPT_DIR, "..", "input"))
a <- commandArgs(trailingOnly = TRUE)
stopifnot(length(a) == 2, all(file.exists(a)))

# pTE values are read as text so they are written out unchanged
pred_var <- fread(a[1], select = c("transcript_refid", "variant_id", "pred_0"), colClasses = "character")
pred_all <- fread(a[2], select = c("transcript_refid", "Sequences_FL", "pred_0"), colClasses = "character")

# transcript_refid of a variant = <RefSeq ID>_<REF>_<ALT>_<position in the full-length mRNA>
id <- tstrsplit(pred_var$transcript_refid, "_", fixed = TRUE)
pred_var[, `:=`(seq_id = paste(id[[1]], id[[2]], sep = "_"), pos = as.integer(id[[5]]))]
pred_var[pred_all, on = c(seq_id = "transcript_refid"), seq := i.Sequences_FL]
pred_var[, `:=`(ref_seq_available = !is.na(seq),
                ref_left  = substring(seq, pos - 2, pos - 1),   # two bases 5' of the variant
                ref_base  = substring(seq, pos, pos),
                ref_right = substring(seq, pos + 1, pos + 2))]  # two bases 3' of the variant

fwrite(pred_var[, .(transcript_refid, variant_id, pred_0, ref_seq_available, ref_left, ref_base, ref_right)],
       file.path(INPUT_DIR, "uAUG_creating_5UTR_SNVs_pTE.tsv.gz"), sep = "\t", na = "NA", quote = FALSE)
cat(nrow(pred_var), "uAUG-creating variant-transcript pairs,", sum(!pred_var$ref_seq_available),
    "without a reference sequence;", nrow(pred_all), "RefSeq mRNAs\n")
