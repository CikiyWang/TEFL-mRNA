# Fig. 4C, D: GWAS traits whose 5'UTR risk SNVs are enriched for TE-altering variants (|delta pTE| > 0.1).
# For each mapped trait, the number of variant-transcript pairs with |delta pTE| > 0.1 is compared with
# all scored 5'UTR GWAS pairs of that trait (one-sided Fisher's exact test, isoform level).
#   C  traits under disease (EFO:0000408 and its children), P < 0.05
#   D  the 10 traits with most TE-altering pairs among the remaining (non-disease) traits, P < 0.05
suppressPackageStartupMessages(library(data.table))

file_arg <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
base_dir <- if (length(file_arg)) dirname(dirname(normalizePath(sub("^--file=", "", file_arg)))) else getwd()
input_dir <- file.path(base_dir, "input")
output_dir <- file.path(base_dir, "output")
shared_dir <- file.path(dirname(base_dir), "shared", "input")
dir.create(output_dir, showWarnings = FALSE)

# GWAS Catalog mapped traits of the 5'UTR and CDS SNVs (chrom_pos_s = "chrN:position") and the disease
# traits (EFO:0000408 and its children)
gwas_catalog_update_nona <- as.data.frame(fread(file.path(shared_dir, "GWAS_catalog_traits_5UTR_CDS_SNVs.tsv"), sep = "\t"))
disease_traits <- fread(file.path(shared_dir, "GWAS_disease_traits_EFO_0000408.tsv"), sep = "\t")$MAPPED_TRAIT
gwas_catalog_update_nona_disease <- gwas_catalog_update_nona[gwas_catalog_update_nona$MAPPED_TRAIT %in% disease_traits, ]

# delta pTE of GWAS 5'UTR SNVs on each RefSeq transcript
pred_score_var_utr5 <- as.data.frame(fread(file.path(input_dir, "GWAS_5UTR_SNVs_dpTE.tsv"), sep = "\t",
                                           select = c("chrom_pos_s", "transcript_refid_var", "delta_score_high")))
pred_score_var_utr5_filter <- pred_score_var_utr5[abs(pred_score_var_utr5$delta_score_high) > 0.1, ]

trait_enrichment <- function(catalog) {
  merged_filter <- merge(pred_score_var_utr5_filter, catalog, by = "chrom_pos_s")
  merged <- merge(pred_score_var_utr5, catalog, by = "chrom_pos_s")
  n_iso_filter <- length(unique(merged_filter$transcript_refid_var))
  n_iso <- length(unique(merged$transcript_refid_var))
  n_var_filter <- length(unique(merged_filter$chrom_pos_s))
  n_var <- length(unique(merged$chrom_pos_s))
  res <- rbindlist(lapply(unique(merged_filter$MAPPED_TRAIT), function(trait) {
    count_trait_filter_iso <- length(unique(merged_filter[merged_filter$MAPPED_TRAIT == trait, "transcript_refid_var"]))
    count_trait_filter_var <- length(unique(merged_filter[merged_filter$MAPPED_TRAIT == trait, "chrom_pos_s"]))
    count_trait_iso <- length(unique(merged[merged$MAPPED_TRAIT == trait, "transcript_refid_var"]))
    count_trait_var <- length(unique(merged[merged$MAPPED_TRAIT == trait, "chrom_pos_s"]))
    ft_iso <- fisher.test(matrix(c(count_trait_filter_iso, n_iso_filter - count_trait_filter_iso,
                                   count_trait_iso, n_iso - count_trait_iso), c(2, 2), byrow = FALSE), alternative = "greater")
    ft_var <- fisher.test(matrix(c(count_trait_filter_var, n_var_filter - count_trait_filter_var,
                                   count_trait_var, n_var - count_trait_var), c(2, 2), byrow = FALSE), alternative = "greater")
    data.table(Traits = trait,
               Counts_iso = count_trait_filter_iso, Counts_traits_iso = count_trait_iso,
               Fracs_iso = count_trait_filter_iso / count_trait_iso,
               Pval_iso = ft_iso$p.value, Odd_iso = unname(ft_iso$estimate),
               Counts_var = count_trait_filter_var, Counts_traits_var = count_trait_var,
               Fracs_var = count_trait_filter_var / count_trait_var,
               Pval_var = ft_var$p.value, Odd_var = unname(ft_var$estimate))
  }))
  res[order(res$Counts_iso, decreasing = TRUE)]
}

capitalize_first <- function(x) gsub("^(\\w)", "\\U\\1", x, perl = TRUE)

## ---- Fig. 4C ----
disease <- trait_enrichment(gwas_catalog_update_nona_disease)
disease[, Traits_upper := capitalize_first(Traits)]
disease[Traits_upper == "Attention deficit hyperactivity disorder, substance abuse, antisocial behaviour measurement",
        Traits_upper := "Attention deficit hyperactivity disorder"]
disease_sig <- disease[Pval_iso < 0.05]
print(disease_sig[, .(Traits_upper, Counts_iso, Odd_iso, Pval_iso)], digits = 3)

## ---- Fig. 4D ----
all_traits <- trait_enrichment(gwas_catalog_update_nona)
nondisease <- all_traits[!(Traits %in% gwas_catalog_update_nona_disease$MAPPED_TRAIT)]
nondisease[, Traits_upper := capitalize_first(Traits)]
nondisease_sig <- nondisease[Pval_iso < 0.05]
print(nondisease_sig[1:10, .(Traits_upper, Counts_iso, Odd_iso, Pval_iso)], digits = 3)

fwrite(disease, file.path(output_dir, "Fig4C_trait_enrichment.tsv"), sep = "\t")
fwrite(disease_sig, file.path(output_dir, "Fig4C_source_data.tsv"), sep = "\t")
fwrite(nondisease, file.path(output_dir, "Fig4D_trait_enrichment.tsv"), sep = "\t")
fwrite(nondisease_sig[1:10], file.path(output_dir, "Fig4D_source_data.tsv"), sep = "\t")
