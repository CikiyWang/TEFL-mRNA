###############################################################################
# Fig. S9A-C -- analysis
#   Coding-region variant effects on predicted translation efficiency, controlled
#   for sequence composition and codon position.
#
#   pTE is the model's predicted probability of high translation efficiency.
#   dpTE = pTE(ALT) - pTE(REF) for the same transcript. The counting unit is a
#   variant-transcript pair, one SNV scored on each RefSeq isoform it falls in.
#
#   Writes three source-data tables to input/:
#     FigS9A_source_data.tsv  per-substitution mean dpTE, unadjusted and adjusted
#     FigS9B_source_data.tsv  per-substitution mean dpTE within GC deciles
#     FigS9C_source_data.tsv  nested-model R2 values and GC coefficients
#
#   Rscript analysis/FigS9ABC_analysis.R
#
#   Requires the gnomAD prediction tables and RefSeq sequence tables, which are
#   external to this directory. Set TEFL_ROOT to the repository root if this
#   directory has been moved out of Data_and_code/.
###############################################################################
suppressMessages({
  library(data.table)
})
setDTthreads(0)

base_dir <- local({
  f <- sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE))
  normalizePath(file.path(if (length(f)) dirname(f) else getwd(), ".."))
})
input_dir <- file.path(base_dir, "input")
root <- Sys.getenv("TEFL_ROOT", unset = normalizePath(file.path(base_dir, "..", ".."),
                                                     mustWork = FALSE))

seq_ribonn  <- file.path(root, "multi_cellline/ncbi_isoform/RiboNN_pred",
                         "pred_NCBI_RefSeq_GRCh38.mRNA_pc.FL_regions-correct.ribonn_output.tsv")
seq_isoform <- file.path(root, "ncbi_isoform",
                         "pred_ncbi_mRNA.filter_cor0.7_l0.25_h1.tpe_opt.tsv")
var_file <- function(cls)
  file.path(root, "gnomad_var/data",
            sprintf(paste0("Pred_All.gnomad.exomes.v4.0.sites.PASS_pc_SNV_NM.%s_variant.",
                           "filter_cor0.7_l0.25_h1.TPE_opt.delta.filter.sel_cols.for_mod.tsv"), cls))

for (f in c(seq_ribonn, seq_isoform, var_file("synonymous"), var_file("missense")))
  if (!file.exists(f)) stop("missing input: ", f)

cols <- c("transcript_refid_var", "delta_score_high", "codon_pos",
          "codon_num", "codon_novar", "codon_var")
rd <- function(p) {
  d <- fread(p, select = cols, showProgress = FALSE)
  d[, tx_id := sub("_[ACGT]_[ACGT]_[0-9]+$", "", transcript_refid_var)][, transcript_refid_var := NULL]
  d[!is.na(delta_score_high) & codon_pos %in% 1:3 &
      grepl("^[ACGT]{3}$", codon_novar) & grepl("^[ACGT]{3}$", codon_var)]
}
syn_raw <- rd(var_file("synonymous"))
mis_raw <- rd(var_file("missense"))

# Sequence table: the RiboNN prediction table, extended with the transcripts that
# the variant tables reference but it does not contain (1,482 accessions).
sq <- fread(seq_ribonn, select = c("tx_id", "tx_sequence", "tx_size", "utr5_size", "cds_size"),
            showProgress = FALSE)
sq <- sq[!is.na(tx_size) & tx_size == nchar(tx_sequence) & cds_size %% 3 == 0]
miss_tx <- setdiff(unique(c(syn_raw$tx_id, mis_raw$tx_id)), sq$tx_id)

add <- fread(seq_isoform, select = c("transcript_refid", "Sequences_UTR5", "Sequences_CDS",
                                     "Sequences_UTR3", "Sequences_FL"), showProgress = FALSE)
setnames(add, "transcript_refid", "tx_id")
add <- add[tx_id %chin% miss_tx]
add[, `:=`(utr5_size = nchar(Sequences_UTR5), cds_size = nchar(Sequences_CDS),
           utr3_size = nchar(Sequences_UTR3), tx_size = nchar(Sequences_FL))]
add <- add[tx_size == utr5_size + cds_size + utr3_size & cds_size %% 3 == 0 &
             Sequences_FL == paste0(Sequences_UTR5, Sequences_CDS, Sequences_UTR3)]
sq2 <- rbind(sq[, .(tx_id, tx_sequence, tx_size, utr5_size, cds_size)],
             add[, .(tx_id, tx_sequence = Sequences_FL, tx_size, utr5_size, cds_size)])
rm(sq, add); invisible(gc())

# Cumulative G+C over the concatenated transcriptome, for O(1) window queries.
rv  <- charToRaw(paste0(sq2$tx_sequence, collapse = ""))
cum <- c(0L, cumsum((rv == charToRaw("G")) | (rv == charToRaw("C")))); rm(rv); invisible(gc())
sq2[, off := cumsum(c(0, head(as.numeric(tx_size), -1)))]
sq2[, `:=`(tx_gc = (cum[off + tx_size + 1] - cum[off + 1]) / tx_size, ncod = cds_size / 3L)]
setkey(sq2, tx_id)

W <- 25L   # local GC window half-width, nt, on the reference mature mRNA
feat <- function(d) {
  d <- sq2[, .(tx_id, off, tx_size, utr5_size, ncod, tx_gc)][d, on = "tx_id", nomatch = 0]
  d <- d[codon_num >= 1 & codon_num <= ncod]
  d[, pos := utr5_size + (codon_num - 1L) * 3L + codon_pos]
  a <- pmax(1L, d$pos - W); b <- pmin(d$tx_size, d$pos + W)
  d[, local_gc := (cum[off + b + 1] - cum[off + a]) / (b - a + 1)]
  d[, `:=`(rel_pos = codon_num / ncod, logL = log10(ncod),
           subst = factor(paste0(substr(codon_novar, codon_pos, codon_pos), ">",
                                 substr(codon_var, codon_pos, codon_pos))),
           cpos = factor(codon_pos))]
  d[, c("off", "codon_novar", "codon_var", "codon_pos") := NULL][]
}
syn <- feat(syn_raw); mis <- feat(mis_raw); rm(syn_raw, mis_raw); invisible(gc())

# Adjusted effects are estimated marginal means at the dataset mean of the
# continuous covariates and the modal codon position.
fitone <- function(d, lab) {
  m0 <- lm(delta_score_high ~ subst, data = d)
  m1 <- lm(delta_score_high ~ subst + local_gc + tx_gc + cpos + rel_pos + logL, data = d)
  mg <- lm(delta_score_high ~ local_gc + tx_gc, data = d)
  modal <- names(sort(table(d$cpos), decreasing = TRUE))[1]
  nd <- data.table(subst = factor(levels(d$subst), levels = levels(d$subst)),
                   local_gc = mean(d$local_gc), tx_gc = mean(d$tx_gc),
                   cpos = factor(modal, levels = levels(d$cpos)),
                   rel_pos = mean(d$rel_pos), logL = mean(d$logL))
  out <- merge(d[, .(unadjusted = mean(delta_score_high), n = .N), by = subst],
               data.table(subst = nd$subst, adjusted = as.numeric(predict(m1, nd))),
               by = "subst")[, class := lab]
  co <- summary(m1)$coefficients
  list(eff = out,
       gc  = data.table(class = lab, term = c("local_gc", "tx_gc"),
                        est = co[c("local_gc", "tx_gc"), 1], se = co[c("local_gc", "tx_gc"), 2]),
       r2  = data.table(class = lab, subst_only = summary(m0)$r.squared,
                        gc_only = summary(mg)$r.squared, full = summary(m1)$r.squared, n = nrow(d)))
}
rs  <- fitone(syn, "Synonymous"); invisible(gc())
rm_ <- fitone(mis, "Missense");   invisible(gc())
eff <- rbind(rs$eff, rm_$eff)

# Panel B is unadjusted; the decile stratification is itself the control.
dec <- function(d, lab) {
  d[, gcdec := cut(tx_gc, quantile(tx_gc, 0:10 / 10), include.lowest = TRUE, labels = 1:10)]
  d[, .(mu = mean(delta_score_high), n = .N), by = .(gcdec, subst)][, class := lab][]
}
dc <- rbind(dec(syn, "Synonymous"), dec(mis, "Missense"))

U <- function(s) gsub("T", "U", s)
eff[, subst_u := U(as.character(subst))][, alt := substr(subst_u, 3, 3)]
dc[,  subst_u := U(as.character(subst))]
r2  <- rbind(rs$r2, rm_$r2)
gcc <- rbind(rs$gc, rm_$gc)

fwrite(eff[, .(class, substitution = subst_u, n, unadjusted, adjusted, allele_introduced = alt)],
       file.path(input_dir, "FigS9A_source_data.tsv"), sep = "\t")
fwrite(dc[, .(class, gc_decile = gcdec, substitution = subst_u, mean_dpTE = mu, n)],
       file.path(input_dir, "FigS9B_source_data.tsv"), sep = "\t")
fwrite(rbind(r2[, .(class, metric = "R2_subst_only", value = subst_only)],
             r2[, .(class, metric = "R2_gc_only",    value = gc_only)],
             r2[, .(class, metric = "R2_full",       value = full)],
             r2[, .(class, metric = "n_variants",    value = as.numeric(n))],
             gcc[, .(class, metric = paste0("beta_", term), value = est)]),
       file.path(input_dir, "FigS9C_source_data.tsv"), sep = "\t")
