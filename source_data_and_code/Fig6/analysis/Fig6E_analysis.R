#!/usr/bin/env Rscript
# Fig. 6E: enrichment of GWAS disease traits among TE-altering SNVs (|delta pTE| > 0.02) in
# 5'UTRs and CDSs (missense + synonymous) per cell line. For each trait, a one-sided Fisher's
# exact test compares the SNV-transcript pairs annotated with the trait among TE-altering pairs
# with all pairs of that region. Traits with P < 0.05 in at least two region x cell-line columns
# are kept.

suppressPackageStartupMessages({
  library(data.table)
  library(dplyr)
})

args <- commandArgs(trailingOnly = FALSE)
SCRIPT_DIR <- dirname(normalizePath(sub("^--file=", "", args[grep("^--file=", args)][1])))
BASE_DIR   <- normalizePath(file.path(SCRIPT_DIR, ".."))
INPUT_DIR  <- file.path(BASE_DIR, "input")
OUTPUT_DIR <- file.path(BASE_DIR, "output")

CELLS <- c("HEK293T", "HeLa", "HepG2", "HMECs", "LCL", "PC3", "PC9", "T47D", "U2OS")
gwas <- fread(file.path(INPUT_DIR, "GWAS_5UTR_CDS_vars_delta_pTE.uniq.correct.v2.9_cell_lines.tsv"), sep = "\t")
gwas_list <- setNames(lapply(CELLS, function(cl) as.data.frame(gwas[cell_line == cl] %>%
  group_by(chrom_pos_s, seq_id, transcript_refid_var, region, cell_line) %>%
  summarise(delta_score_high = mean(delta_score_high), .groups = "drop"))), CELLS)
traits <- as.data.frame(fread(file.path(INPUT_DIR, "GWAS_catalog_traits.5UTR_CDS_vars.tsv"), sep = "\t"))
disease_traits <- fread(file.path(INPUT_DIR, "GWAS_disease_traits.EFO_0000408_child.tsv"), sep = "\t")$MAPPED_TRAIT

# counts of SNV-transcript pairs per trait: TE-altering (filter) and all (all) pairs of the region
cal_all_traits <- function(sel_ids, merge_df) {
  df_iso <- unique(merge_df[merge_df$chrom_pos_s %in% sel_ids, c("transcript_refid_var", "MAPPED_TRAIT")])
  iso_filter <- as.matrix(table(df_iso$MAPPED_TRAIT))
  iso_all    <- as.matrix(table(unique(merge_df[, c("transcript_refid_var", "MAPPED_TRAIT")])$MAPPED_TRAIT))
  merged_iso <- merge(iso_filter, iso_all, by = "row.names")
  data.frame(iso_filter_counts            = as.numeric(merged_iso[, 2]),
             iso_all_counts               = as.numeric(merged_iso[, 3]),
             iso_filter_all_traits_counts = length(unique(df_iso$transcript_refid_var)),
             iso_all_all_traits_counts    = length(unique(merge_df$transcript_refid_var)),
             row.names = merged_iso$Row.names)
}

calc_fisher <- function(df) {
  t(apply(df, 1, function(x) {
    ft <- fisher.test(matrix(c(x["iso_filter_counts"], x["iso_filter_all_traits_counts"] - x["iso_filter_counts"],
                               x["iso_all_counts"], x["iso_all_all_traits_counts"] - x["iso_all_counts"]),
                             nrow = 2, byrow = FALSE), alternative = "greater")
    c(pval = ft$p.value, odds_ratio = unname(ft$estimate))
  }))
}

build_region_tables <- function(region_filter, delta_cut = 0.02) {
  pval_list <- list(); odds_list <- list()
  for (cl in CELLS) {
    df_cl <- gwas_list[[cl]]
    df_cl <- df_cl[df_cl$region %in% region_filter, ]
    if (nrow(df_cl) == 0) next
    merge_df <- merge(df_cl, traits, by = "chrom_pos_s")
    sel_ids <- unique(df_cl[abs(df_cl$delta_score_high) > delta_cut, "chrom_pos_s"])
    if (length(sel_ids) == 0) next
    fres <- calc_fisher(cal_all_traits(sel_ids, merge_df))
    pval_list[[cl]] <- fres[, "pval"]
    odds_list[[cl]] <- fres[, "odds_ratio"]
  }
  all_traits <- unique(unlist(lapply(pval_list, names)))
  pval_mat <- sapply(pval_list, function(v) as.numeric(v[all_traits]))
  odds_mat <- sapply(odds_list, function(v) as.numeric(v[all_traits]))
  rownames(pval_mat) <- rownames(odds_mat) <- all_traits
  list(pval = as.data.frame(pval_mat), odds = as.data.frame(odds_mat))
}

filter_disease <- function(tab) {
  keep <- rownames(tab$pval) %in% disease_traits & !grepl(",", rownames(tab$pval))
  list(pval = tab$pval[keep, , drop = FALSE], odds = tab$odds[keep, , drop = FALSE])
}

utr5 <- filter_disease(build_region_tables("5'UTR"))
cds  <- filter_disease(build_region_tables(c("CDS_missense", "CDS_synonymous")))

suffix <- function(df, s) { colnames(df) <- paste0(colnames(df), s); df }
all_traits <- union(rownames(utr5$pval), rownames(cds$pval))
pad <- function(df) { out <- as.data.frame(matrix(NA, length(all_traits), ncol(df), dimnames = list(all_traits, colnames(df)))); out[rownames(df), ] <- df; out }
pval_c <- cbind(pad(suffix(utr5$pval, "_UTR5")), pad(suffix(cds$pval, "_CDS")))
odds_c <- cbind(pad(suffix(utr5$odds, "_UTR5")), pad(suffix(cds$odds, "_CDS")))
pval_c[is.na(pval_c)] <- 1
odds_c[is.na(odds_c)] <- 0

keep <- which(apply(pval_c, 1, function(x) sum(x < 0.05)) >= 2)
pval_plot <- pval_c[keep, , drop = FALSE]
odds_plot <- odds_c[keep, , drop = FALSE]

to_long <- function(m, value) {
  d <- data.table(trait = rownames(m), as.data.table(m))
  melt(d, id.vars = "trait", variable.name = "column", value.name = value)
}
res <- merge(to_long(odds_plot, "odds_ratio"), to_long(pval_plot, "fisher_p"), by = c("trait", "column"), sort = FALSE)
res[, `:=`(region = ifelse(grepl("_UTR5$", column), "5'UTR", "CDS"), cell_line = sub("_UTR5$|_CDS$", "", column))]

dir.create(OUTPUT_DIR, showWarnings = FALSE)
fwrite(res[, .(trait, region, cell_line, column, odds_ratio, fisher_p)], file.path(OUTPUT_DIR, "Fig6E_disease_trait_enrichment.tsv"), sep = "\t")
cat("Fig. 6E:", nrow(pval_plot), "disease traits x", ncol(pval_plot), "region-cell line columns\n")
