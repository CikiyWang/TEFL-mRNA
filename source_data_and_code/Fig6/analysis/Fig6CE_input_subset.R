#!/usr/bin/env Rscript
# Builds the Fig. 6C and 6E inputs: delta pTE of GWAS catalog SNVs in 5'UTRs and CDSs per cell
# line (cell-line TE models), the GWAS catalog traits of these SNVs, and the disease traits
# (EFO_0000408 descendants). Not needed to reproduce the figure; documents how input/ was made.
#
#   Rscript analysis/Fig6CE_input_subset.R <dir with GWAS_all_region_vars_delta_pTE.uniq.<cell>.correct.v2.tsv> \
#                                          <GWAS_vars_update.mod.correct.tsv> <GWAS_chrpos_update.mod.correct.bed> \
#                                          <GWAS_vars_update.mod.correct.EFO_0000408_child.tsv>

suppressPackageStartupMessages(library(data.table))

args <- commandArgs(trailingOnly = FALSE)
SCRIPT_DIR <- dirname(normalizePath(sub("^--file=", "", args[grep("^--file=", args)][1])))
INPUT_DIR  <- normalizePath(file.path(SCRIPT_DIR, "..", "input"))
a <- commandArgs(trailingOnly = TRUE)
stopifnot(length(a) == 4, dir.exists(a[1]), all(file.exists(a[2:4])))

CELLS <- c("HEK293T", "HeLa", "HepG2", "HMECs", "LCL", "PC3", "PC9", "T47D", "U2OS")

# delta_score_high is kept as text so the values are written out unchanged
gwas <- rbindlist(lapply(CELLS, function(cl)
  fread(file.path(a[1], paste0("GWAS_all_region_vars_delta_pTE.uniq.", cl, ".correct.v2.tsv")),
        select = c("chrom_pos_s", "seq_id", "transcript_refid_var", "region", "cell_line", "delta_score_high"),
        colClasses = c(delta_score_high = "character"))))
gwas <- gwas[region %in% c("5'UTR", "CDS_missense", "CDS_synonymous")]
fwrite(gwas, file.path(INPUT_DIR, "GWAS_5UTR_CDS_vars_delta_pTE.uniq.correct.v2.9_cell_lines.tsv"), sep = "\t")
print(gwas[, .N, by = .(cell_line, region)][order(cell_line, region)])

# trait annotation, read as in the original analysis (catalog rows and bed rows are parallel)
catalog <- cbind(read.delim(a[2], quote = "", fill = TRUE), read.delim(a[3], header = FALSE, sep = "\t", quote = "", fill = TRUE))
catalog$chrom_pos_s <- gsub("@", ":", catalog$chrom_pos)
catalog <- catalog[!is.na(catalog$MAPPED_TRAIT), ]
traits <- unique(as.data.table(catalog)[chrom_pos_s %in% gwas$chrom_pos_s, .(chrom_pos_s, MAPPED_TRAIT)])
fwrite(traits, file.path(INPUT_DIR, "GWAS_catalog_traits.5UTR_CDS_vars.tsv"), sep = "\t")

disease <- read.delim(a[4], row.names = 1, header = TRUE, sep = "\t")
fwrite(data.table(MAPPED_TRAIT = sort(unique(disease$MAPPED_TRAIT))), file.path(INPUT_DIR, "GWAS_disease_traits.EFO_0000408_child.tsv"), sep = "\t")
cat(nrow(traits), "SNV-trait pairs;", length(unique(disease$MAPPED_TRAIT)), "disease traits\n")
