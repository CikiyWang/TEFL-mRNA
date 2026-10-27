#!/usr/bin/env Rscript
# Builds the Fig. 4E inputs: ClinVar (2024-05-14 release) 5'UTR and synonymous SNV-transcript pairs of
# immune-related records, scored with the LCL TEFL-mRNA model. A ClinVar record counts as immune-related
# when its CLNDN (condition name) contains "immun" (immunodeficiency, autoimmune, ...); the match is
# case-sensitive, as for the ClinVar immune tables used for the published panel. delta pTE = pTE of the
# variant mRNA - pTE of the RefSeq mRNA. Not needed to reproduce the figure; documents how input/ was made.
#
#   Rscript analysis/Fig4E_prepare_input.R \
#     <Pred_All.clinvar_20240514.vep_protein_coding_NM.5_prime_UTR_variant.SNV.filter_cor0.7_l0.25_h1.tpe_opt.tsv> \
#     <Pred_All.clinvar_20240514.vep_protein_coding_NM.synonymous_variant.SNV.filter_cor0.7_l0.25_h1.tpe_opt.tsv> \
#     <clinvar_20240514.vcf.gz>
#
# clinvar_20240514.vcf.gz: https://ftp.ncbi.nlm.nih.gov/pub/clinvar/vcf_GRCh38/archive_2.0/2024/clinvar_20240514.vcf.gz
# (MD5 e730a29f8bfd7fcd832e065b2dce4f1b). The two prediction tables hold the TEFL-mRNA predictions (pred_0,
# P(TE-high)) of every ClinVar SNV-transcript pair in RefSeq 5'UTRs and synonymous codons; variant_id is
# <ClinVar variation ID>@<chrom>:<pos>_<REF>/<ALT>.

suppressPackageStartupMessages(library(data.table))

args <- commandArgs(trailingOnly = FALSE)
SCRIPT_DIR <- dirname(normalizePath(sub("^--file=", "", args[grep("^--file=", args)][1])))
INPUT_DIR  <- normalizePath(file.path(SCRIPT_DIR, "..", "input"))
SHARED_DIR <- normalizePath(file.path(SCRIPT_DIR, "..", "..", "shared", "input"))
a <- commandArgs(trailingOnly = TRUE)
stopifnot(length(a) == 3, all(file.exists(a)))

# immune-related ClinVar records (VCF ID = ClinVar variation ID), with condition and clinical significance
vcf <- fread(cmd = paste("gzip -dc", shQuote(a[3]), "| grep -v '^##'"), sep = "\t", select = c("ID", "INFO"),
             colClasses = "character")
vcf[, CLNDN := sub("^(.*;)?CLNDN=([^;]*).*$", "\\2", INFO)][!grepl("(^|;)CLNDN=", INFO), CLNDN := ""]
vcf[, CLNSIG := sub("^(.*;)?CLNSIG=([^;]*).*$", "\\2", INFO)][!grepl("(^|;)CLNSIG=", INFO), CLNSIG := ""]
immune <- vcf[grepl("immun", CLNDN, fixed = TRUE), .(clinvar_id = ID, ClinVar_CLNSIG = CLNSIG, ClinVar_CLNDN = CLNDN)]
rm(vcf); invisible(gc())
cat(nrow(immune), "immune-related ClinVar records\n")

# pTE of the RefSeq (reference) mRNAs
pte <- fread(file.path(SHARED_DIR, "RefSeq_mRNA_pTE.tsv"), sep = "\t")
pte[, seq_id := gsub("\\s+", "", transcript_refid)]

build <- function(pred_file, out_name) {
  # numeric columns are read as text so the values are written out unchanged
  d <- fread(pred_file, sep = "\t", select = c("transcript_refid", "variant_id", "gene", "gnomade_AF", "pred_0"),
             colClasses = "character", na.strings = c("", "NA"))
  setnames(d, c("transcript_refid", "pred_0"), c("transcript_refid_var", "pred_0_var"))
  d[, clinvar_id := sub("@.*$", "", variant_id)]
  d <- merge(d, immune, by = "clinvar_id")
  id <- tstrsplit(d$transcript_refid_var, "_", fixed = TRUE)        # NM_xxxxxx_REF_ALT_pos
  d[, `:=`(seq_id = paste(id[[1]], id[[2]], sep = "_"), ref = id[[3]], alt = id[[4]], pos = as.integer(id[[5]]))]
  d[pte, on = "seq_id", pred_0_all := i.pred_0]
  stopifnot(!anyNA(d$pred_0_all))
  d[, delta_score_high := as.numeric(pred_0_var) - pred_0_all]
  d <- d[order(-abs(delta_score_high))]
  out <- d[, .(transcript_refid_var, seq_id, gene, variant_id, clinvar_id, ref, alt, pos, gnomade_AF,
               pred_0_all, pred_0_var, delta_score_high, ClinVar_CLNSIG, ClinVar_CLNDN)]
  fwrite(out, file.path(INPUT_DIR, out_name), sep = "\t")
  cat(out_name, ":", nrow(out), "SNV-transcript pairs of", uniqueN(out$clinvar_id), "ClinVar records\n")
}
build(a[1], "ClinVar_immune_5UTR_SNVs_dpTE.tsv")
build(a[2], "ClinVar_immune_synonymous_SNVs_dpTE.tsv")
