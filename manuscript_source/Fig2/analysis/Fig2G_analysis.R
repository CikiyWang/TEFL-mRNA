#!/usr/bin/env Rscript
# Fig. 2G: reporter validation in GM12878 cells. Relative expression level (relative to Fluc) of
# REF and MUT 5'UTR reporters, three replicates each; bars show the mean and error bars the SD.
# One-sided two-sample t-test with equal variances: MUT > REF for variants predicted to increase
# pTE (left group), MUT < REF for those predicted to decrease it (right group). Predicted pTE:
# REF = pTE of the RefSeq mRNA, MUT = REF pTE + delta pTE of the variant.

suppressPackageStartupMessages(library(data.table))

args <- commandArgs(trailingOnly = FALSE)
SCRIPT_DIR <- dirname(normalizePath(sub("^--file=", "", args[grep("^--file=", args)][1])))
BASE_DIR   <- normalizePath(file.path(SCRIPT_DIR, ".."))
INPUT_DIR  <- file.path(BASE_DIR, "input")
OUTPUT_DIR <- file.path(BASE_DIR, "output")
SHARED_DIR <- file.path(dirname(BASE_DIR), "shared", "input")

reporter <- fread(file.path(INPUT_DIR, "reporter_assay_gnomAD_SNVs.tsv"))
variants <- fread(file.path(INPUT_DIR, "validated_gnomAD_SNVs.tsv"))
pte <- fread(file.path(SHARED_DIR, "RefSeq_mRNA_pTE.tsv"))
variants[, ref_pTE := pte$pred_0[match(transcript_id, pte$transcript_refid)]]
variants[, mut_pTE := ref_pTE + delta_pTE]

# replicate-level values of the six validated variants
reps <- melt(reporter[gene %in% variants$gene], id.vars = c("name", "label", "gene"),
             measure.vars = c("rep1", "rep2", "rep3"), variable.name = "replicate", value.name = "value")
reps[, allele := ifelse(label == "ref", "REF", "MUT")]
reps <- merge(variants[, .(group, gene)], reps, by = "gene", sort = FALSE)
reps <- reps[order(match(gene, variants$gene), match(allele, c("REF", "MUT")), replicate),
             .(group, gene, allele, replicate, value)]

tests <- rbindlist(lapply(seq_len(nrow(variants)), function(i) {
  v <- variants[i]
  alternative <- if (v$group == "left") "greater" else "less"
  ref <- reps[gene == v$gene & allele == "REF", value]; mut <- reps[gene == v$gene & allele == "MUT", value]
  tt <- t.test(mut, ref, var.equal = TRUE, alternative = alternative)
  cbind(v, data.table(mean_REF = mean(ref), sd_REF = sd(ref), mean_MUT = mean(mut), sd_MUT = sd(mut),
                      alternative = paste("MUT", if (alternative == "greater") ">" else "<", "REF"),
                      t = unname(tt$statistic), df = unname(tt$parameter), p_value = tt$p.value))
}))

dir.create(OUTPUT_DIR, showWarnings = FALSE)
fwrite(reps, file.path(OUTPUT_DIR, "Fig2G_source_data.tsv"), sep = "\t")
fwrite(tests, file.path(OUTPUT_DIR, "Fig2G_ttest.tsv"), sep = "\t")
print(tests[, .(group, gene, transcript_id, position, ref, alt, ref_pTE = round(ref_pTE, 3), mut_pTE = round(mut_pTE, 3),
                alternative, p_value = signif(p_value, 2))])
