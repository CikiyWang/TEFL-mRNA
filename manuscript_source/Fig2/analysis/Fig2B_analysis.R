#!/usr/bin/env Rscript
# Fig. 2B: predicted delta pTE of gnomAD v4.0 exome SNVs by transcript region (CDS = synonymous +
# missense), counted over variant-transcript pairs, as in the published panel.
#   left : fraction of pairs with |delta pTE| above each cutoff (0.01, 0.05, 0.1, 0.15, 0.2, 0.3)
#   right: |delta pTE| of the pairs with |delta pTE| > 0.1, with the N printed above each box

suppressPackageStartupMessages(library(data.table))

args <- commandArgs(trailingOnly = FALSE)
SCRIPT_DIR <- dirname(normalizePath(sub("^--file=", "", args[grep("^--file=", args)][1])))
BASE_DIR   <- normalizePath(file.path(SCRIPT_DIR, ".."))
INPUT_DIR  <- file.path(BASE_DIR, "input")
OUTPUT_DIR <- file.path(BASE_DIR, "output")
SHARED_DIR <- file.path(dirname(BASE_DIR), "shared", "input")

REGIONS   <- c("5'UTR", "CDS", "3'UTR")
REGION_OF <- c("5'UTR" = "5'UTR", "3'UTR" = "3'UTR", synonymous = "CDS", missense = "CDS")

# left: as in the published panel, pairs without a delta pTE count in the total and above every
# cutoff (the original data.frame subsetting kept them as NA rows)
counts <- fread(file.path(INPUT_DIR, "gnomAD_v4_SNVs_n_above_dpTE_cutoffs.tsv"))
counts[, region := unname(REGION_OF[snv_class])]
left <- counts[, .(n_above_cutoff = sum(n_above_cutoff + n_missing), n_total = sum(n_total + n_missing)),
               by = .(region, cutoff)][, fraction := n_above_cutoff / n_total]
left <- left[order(match(region, REGIONS), cutoff)]

# right: pairs with |delta pTE| > 0.1
pairs <- fread(file.path(SHARED_DIR, "gnomAD_v4_SNVs_dpTE_abs_gt_0.1.tsv"))
right <- pairs[, .(region = unname(REGION_OF[snv_class]), snv_class, transcript_refid_var, variant_id,
                   delta_pTE = delta_score_high, abs_delta_pTE = abs(delta_score_high))]

# N above each box, counted as in the published panel: unique SNV IDs among the pairs above 0.1,
# where pairs without a delta pTE add one NA ID, and the CDS N is the missense count taken twice.
# n_unique_snvs is the number of unique SNVs.
has_na <- counts[cutoff == min(cutoff), setNames(n_missing > 0, snv_class)]
n_ids  <- function(cls) right[snv_class == cls, uniqueN(variant_id)] + has_na[[cls]]
n_tab <- right[, .(n_pairs = .N, n_unique_snvs = uniqueN(variant_id)), by = region]
n_tab[, n_label := fcase(region == "5'UTR", n_ids("5'UTR"), region == "3'UTR", n_ids("3'UTR"),
                         region == "CDS", 2L * n_ids("missense"))]
n_tab <- n_tab[order(match(region, REGIONS))]

dir.create(OUTPUT_DIR, showWarnings = FALSE)
fwrite(left, file.path(OUTPUT_DIR, "Fig2B_left_source_data.tsv"), sep = "\t")
fwrite(right, file.path(OUTPUT_DIR, "Fig2B_right_source_data.tsv"), sep = "\t")
fwrite(n_tab, file.path(OUTPUT_DIR, "Fig2B_right_N.tsv"), sep = "\t")
print(dcast(left, region ~ cutoff, value.var = "fraction"), digits = 3)
print(n_tab)
