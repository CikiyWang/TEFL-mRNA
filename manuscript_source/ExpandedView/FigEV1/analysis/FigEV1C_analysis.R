#!/usr/bin/env Rscript
# Fig. EV1C: comparison of mean LCL TE across the 47 individuals with the LCL TE
# compiled by Zheng et al. for RiboNN.
# RefSeq (NM) and Ensembl (ENST) accessions do not cross-map directly, so
# transcripts are matched by exact sequence identity: full-length transcript
# sequence first, then coding sequence (RefSeq and Ensembl annotate UTR ends
# differently while the CDS is identical).

# resolve FigEV1/ from this script's location so the package runs from any directory
BASE_DIR <- local({
  a <- commandArgs(trailingOnly = FALSE); f <- sub("^--file=", "", a[grepl("^--file=", a)])
  if (length(f)) dirname(dirname(normalizePath(f))) else normalizePath(".")
})
INPUT  <- file.path(BASE_DIR, "input")
SHARED <- file.path(dirname(dirname(BASE_DIR)), "shared", "input")
OUTPUT <- file.path(BASE_DIR, "output")

te <- as.matrix(read.delim(file.path(SHARED, "LCL_TE_47_individuals.tsv"),
                           row.names = 1, check.names = FALSE))
fl <- read.delim(file.path(INPUT, "LCL_mRNA_sequences.tsv.gz"),
                 check.names = FALSE)
rn <- read.delim(file.path(SHARED, "RiboNN_TE_by_cell_line.tsv.gz"), check.names = FALSE)
# consensus TE label set used to train the model
pub <- read.delim(file.path(SHARED, "LCL_consensus_TE_classes.tsv"))
pub <- pub[pub$in_training_set, ]

rn$fl_key  <- toupper(rn$tx_sequence)
rn$cds_key <- toupper(substr(rn$tx_sequence, rn$utr5_size + 1, rn$utr5_size + rn$cds_size))

a <- merge(data.frame(seq_id = fl$Row.names, k = toupper(fl$Sequences_FL)),
           data.frame(rn_tx = rn$tx_id, TE_LCL = rn$TE_LCL, k = rn$fl_key), by = "k")
b <- merge(data.frame(seq_id = fl$Row.names, k = toupper(fl$Sequences_CDS)),
           data.frame(rn_tx = rn$tx_id, TE_LCL = rn$TE_LCL, k = rn$cds_key), by = "k")
pairs <- unique(rbind(a[, c("seq_id", "rn_tx", "TE_LCL")], b[, c("seq_id", "rn_tx", "TE_LCL")]))
stopifnot(all(table(pairs$seq_id) == 1))        # matching must be one-to-one

our <- data.frame(seq_id = sub("\\..*$", "", rownames(te)), our = rowMeans(te, na.rm = TRUE))
cmp <- merge(pairs[, c("seq_id", "TE_LCL")], our, by = "seq_id")
cmp <- cmp[is.finite(cmp$TE_LCL) & cmp$our > 0, ]
cmp$in_published <- cmp$seq_id %in% sub("\\..*$", "", pub$seq_id)

write.csv(cmp, file.path(OUTPUT, "FigEV1C_LCL_TE_vs_RiboNN.csv"), row.names = FALSE)

cat(sprintf("matched %d transcripts (%d in the published label set)\n",
            nrow(cmp), sum(cmp$in_published)))
cat(sprintf("Spearman rho all matched %.3f | published label set %.3f\n",
            cor(cmp$our, cmp$TE_LCL, method = "spearman"),
            with(cmp[cmp$in_published, ], cor(our, TE_LCL, method = "spearman"))))
