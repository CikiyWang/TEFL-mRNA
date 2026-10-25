#!/usr/bin/env Rscript
# Builds the Fig. 1F miRNA binding-site table from the downloaded databases: miRWalk sites with
# binding probability 1 on RefSeq mRNAs, kept for miRNAs expressed in naive B cells, with the
# miRNA expression. Writes input/miRWalk_prob1_sites.B_naive_miRNAs.tsv (the same table that
# analysis/Fig1F_input_subset.R extracts from the saved intermediate) and the pTE table.
#
#   Rscript analysis/Fig1F_preprocess_miRNA.R <miRWalk directory> <PMD22221297_B_naive_high_confidence.tsv> \
#                                             <pred_ncbi_mRNA.filter_TE_less.opt-correct.tsv>
#
# miRWalk directory: hsa_miRWalk_5UTR.prob_0.9_NM.info.txt, hsa_miRWalk_CDS.prob_0.9_NM.info.txt and
# hsa_miRWalk_3UTR.prob_0.9_NM.info.txt, the miRWalk human target predictions of each region kept for
# RefSeq NM_ transcripts with binding probability >= 0.9; tab-separated, no header, columns miRNA,
# RefSeq ID, gene symbol, duplex position (start,end) and binding probability.
# Expression table: miRBase high-confidence hairpins with reads per million (RPM) in the naive B-cell
# sample of PMID 22221297; tab-separated with header Name, Accession, RPM, Chromosome, Start, End,
# Strand, Confidence.
#
# Expressed miRNAs are matched to miRWalk by name, with "mir" in the hairpin name replaced by "miR".
# This matches only mature miRNAs named without a -5p/-3p arm (8 miRNAs here).

suppressPackageStartupMessages(library(data.table))

file_arg <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
input_dir <- normalizePath(file.path(dirname(dirname(normalizePath(sub("^--file=", "", file_arg)))), "input"))
a <- commandArgs(trailingOnly = TRUE)
stopifnot(length(a) == 3, dir.exists(a[1]), file.exists(a[2]), file.exists(a[3]))

# miRNAs expressed in naive B cells (values kept as text so they are written out unchanged)
mir_exp <- fread(a[2], sep = "\t", colClasses = "character")
mir_exp[, miRNA := gsub("mir", "miR", Name)]

# miRWalk sites with binding probability 1 of the expressed miRNAs
REGIONS <- c("5'UTR" = "5UTR", "CDS" = "CDS", "3'UTR" = "3UTR")
mir_mRNA <- rbindlist(lapply(names(REGIONS), function(r) {
  f <- file.path(a[1], sprintf("hsa_miRWalk_%s.prob_0.9_NM.info.txt", REGIONS[[r]]))
  d <- fread(f, sep = "\t", header = FALSE, colClasses = "character", strip.white = TRUE,
             col.names = c("miRNA", "mRNA", "Genesymbol", "duplex", "binding_probability"))
  d <- d[miRNA %in% mir_exp$miRNA & as.numeric(binding_probability) == 1]
  d[, `:=`(binding_probability = as.character(as.numeric(binding_probability)), region = r, seq_id = mRNA)]
  message(r, ": ", nrow(d), " sites")
  d
}))
# base-R merges as in the original analysis, which fix the row order of the output
sites <- merge(as.data.frame(mir_mRNA), as.data.frame(mir_exp), by = "miRNA")

# RefSeq mRNAs with a prediction
pte <- fread(a[3], sep = "\t", colClasses = "character", select = c("transcript_refid", "pred_0"))
sites <- as.data.table(merge(data.frame(seq_id = gsub("\\s+", "", pte$transcript_refid)), sites, by = "seq_id"))
sites[, log2RPM := as.character(log2(as.numeric(RPM) + 1))]

SITE_COLS <- c("seq_id", "miRNA", "Genesymbol", "duplex", "binding_probability", "region",
               "Name", "Accession", "RPM", "Chromosome", "Start", "End", "Strand", "Confidence", "log2RPM")
fwrite(sites[, ..SITE_COLS], file.path(input_dir, "miRWalk_prob1_sites.B_naive_miRNAs.tsv"), sep = "\t")
fwrite(pte, file.path(input_dir, "pred_ncbi_mRNA.filter_TE_less.opt-correct.pTE.tsv"), sep = "\t")
cat(nrow(sites), "sites of", uniqueN(sites$miRNA), "miRNAs on", uniqueN(sites$seq_id), "mRNAs\n")
