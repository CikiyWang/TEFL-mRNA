#!/usr/bin/env Rscript
# Builds shared/input/gnomAD_v4_SNVs_dpTE_abs_gt_0.1.tsv (Fig. 2B, 4A, 4B): gnomAD v4.0 exome
# SNV-transcript pairs with |delta pTE| > 0.1 in the 5'UTR, 3'UTR, synonymous and missense tables, with
# the variant ID and the gene symbol of each RefSeq transcript. Not needed to reproduce the figures;
# documents how the table was made.
#
# Gene symbols are the gnomAD VEP symbols carried by the 5'UTR table and by the missense table of
# pairs with |delta pTE| > 0.02 (one symbol per transcript); transcripts in neither table (10
# synonymous transcripts) are mapped with org.Hs.eg.db (REFSEQ -> SYMBOL).
#
#   Rscript analysis/Fig4AB_prepare_input.R <directory with the gnomAD tables>
#   Pred_All.gnomad.exomes.v4.0.sites.PASS_pc_SNV_NM.5_prime_UTR_variant.filter_cor0.7_l0.25_h1.TPE_opt.delta.filter.sel_cols.v3.tsv
#   Pred_All.gnomad.exomes.v4.0.sites.PASS_pc_SNV_NM.synonymous_variant.filter_cor0.7_l0.25_h1.TPE_opt.delta.filter.sel_cols.v3.tsv
#   Pred_All.gnomad.exomes.v4.0.sites.PASS_pc_SNV_NM.missense_variant.filter_cor0.7_l0.25_h1.TPE_opt.delta.filter.sel_cols.v3.tsv
#   Pred_All.gnomad.exomes.v4.0.sites.PASS_pc_SNV_NM.<class>.filter_cor0.7_l0.25_h1.TPE_opt.delta.filter.sel_cols.for_mod.tsv
#     (3'UTR pairs, and the variant ID of every class)
#   Pred_All.gnomad.exomes.v4.0.sites.PASS_pc_SNV_NM.missense_variant.filter_cor0.7_l0.25_h1.TPE_opt.delta.filter_0.02_translated.tsv (gene symbols)

suppressPackageStartupMessages({
  library(data.table)
  library(org.Hs.eg.db)
})

args <- commandArgs(trailingOnly = FALSE)
SCRIPT_DIR <- dirname(normalizePath(sub("^--file=", "", args[grep("^--file=", args)][1])))
SHARED_DIR <- normalizePath(file.path(SCRIPT_DIR, "..", "..", "shared", "input"))
a <- commandArgs(trailingOnly = TRUE)
stopifnot(length(a) == 1, dir.exists(a))

PREFIX <- file.path(a, "Pred_All.gnomad.exomes.v4.0.sites.PASS_pc_SNV_NM.")
TABLES <- c("5'UTR"      = "5_prime_UTR_variant.filter_cor0.7_l0.25_h1.TPE_opt.delta.filter.sel_cols.v3.tsv",
            "3'UTR"      = "3_prime_UTR_variant.filter_cor0.7_l0.25_h1.TPE_opt.delta.filter.sel_cols.for_mod.tsv",
            "synonymous" = "synonymous_variant.filter_cor0.7_l0.25_h1.TPE_opt.delta.filter.sel_cols.v3.tsv",
            "missense"   = "missense_variant.filter_cor0.7_l0.25_h1.TPE_opt.delta.filter.sel_cols.v3.tsv")
tx_of <- function(x) { s <- tstrsplit(x, "_", fixed = TRUE); paste(s[[1]], s[[2]], sep = "_") }

# delta pTE is read as text so the selected values are written out unchanged
pairs <- rbindlist(lapply(names(TABLES), function(k) {
  d <- fread(paste0(PREFIX, TABLES[[k]]), select = c("transcript_refid_var", "delta_score_high"), colClasses = "character")
  d[abs(as.numeric(delta_score_high)) > 0.1, .(snv_class = k, transcript_refid_var, delta_score_high)]
}))
# variant IDs: the v3 tables have none, the for_mod tables of the same classes do
ids <- rbindlist(lapply(sub("v3[.]tsv$", "for_mod.tsv", TABLES), function(f)
  fread(paste0(PREFIX, f), select = c("transcript_refid_var", "variant_id"))))
pairs[ids, on = "transcript_refid_var", variant_id := i.variant_id]
stopifnot(!anyNA(pairs$variant_id))

# transcript -> gene symbol
utr5 <- fread(paste0(PREFIX, TABLES[["5'UTR"]]), select = c("transcript_refid_var", "gene"))
mis  <- suppressWarnings(fread(paste0(PREFIX, "missense_variant.filter_cor0.7_l0.25_h1.TPE_opt.delta.filter_0.02_translated.tsv"),
                               select = c("transcript_refid_var", "gene")))
vep <- unique(rbind(utr5[, .(transcript_id = tx_of(transcript_refid_var), gene)],
                    mis[, .(transcript_id = tx_of(transcript_refid_var), gene)]))
stopifnot(!anyDuplicated(vep$transcript_id))
pairs[, transcript_id := tx_of(transcript_refid_var)]
pairs[vep, on = "transcript_id", `:=`(gene = i.gene, gene_source = "gnomAD VEP")]
rest <- unique(pairs[is.na(gene), transcript_id])
orgdb <- as.data.table(AnnotationDbi::select(org.Hs.eg.db, keys = rest, keytype = "REFSEQ", columns = "SYMBOL"))
pairs[orgdb, on = c(transcript_id = "REFSEQ"), `:=`(gene = i.SYMBOL, gene_source = "org.Hs.eg.db")]
stopifnot(!anyNA(pairs$gene))

fwrite(pairs[, .(snv_class, transcript_refid_var, variant_id, transcript_id, gene, gene_source, delta_score_high)],
       file.path(SHARED_DIR, "gnomAD_v4_SNVs_dpTE_abs_gt_0.1.tsv"), sep = "\t")
print(pairs[, .(pairs = .N, transcripts = uniqueN(transcript_id), genes = uniqueN(gene)), by = snv_class])
cat("genes in total:", uniqueN(pairs$gene), "; mapped with org.Hs.eg.db:", length(rest), "transcripts\n")
