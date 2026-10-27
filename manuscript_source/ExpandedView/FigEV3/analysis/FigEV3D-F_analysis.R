# Fig. EV3D-F: four variant-effect scores on the same GWAS 5'UTR variants in 9 cell lines.
#   dpte    = delta P(high-TE class)          TEFL classifier
#   ordinal = delta [P(high) - P(low)]        ordinal-expected class score
#   our_reg = delta predicted TE              TEFL regression model
#   ribonn  = delta predicted TE              RiboNN (pretrained)
# Each score is averaged over folds per variant and cell line.
suppressPackageStartupMessages(library(data.table))

file_arg <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
base_dir <- if (length(file_arg)) dirname(dirname(normalizePath(sub("^--file=", "", file_arg)))) else getwd()
input_dir <- file.path(base_dir, "input")
output_dir <- file.path(base_dir, "output")
dir.create(output_dir, showWarnings = FALSE)

CELLS <- c("HEK293T", "HMECs", "HeLa", "HepG2", "LCL", "PC3", "PC9", "T47D", "U2OS")
METS  <- c("dpte", "ordinal", "our_reg", "ribonn")

clf <- rbindlist(lapply(CELLS, function(cl) {
  f <- file.path(input_dir, sprintf("GWAS_5UTR_SNVs_classifier_folds_%s.tsv.gz", cl))
  d <- fread(f, select = c("transcript_refid_var", "delta_high", "delta_low"))
  d[, .(dpte    = mean(delta_high),
        ordinal = mean(delta_high) - mean(delta_low)),
    by = transcript_refid_var][, cell_line := cl]
}))

reg <- fread(file.path(input_dir, "GWAS_5UTR_SNVs_regression_folds.tsv.gz"),
             select = c("cell_line", "transcript_refid_var", "delta_TE"))
reg <- reg[cell_line %in% CELLS,
           .(our_reg = mean(delta_TE)), by = .(transcript_refid_var, cell_line)]

rib <- fread(file.path(input_dir, "GWAS_5UTR_SNVs_RiboNN.tsv.gz"))
rib <- melt(rib[, c("transcript_refid_var", paste0("delta_", CELLS)), with = FALSE],
            id.vars = "transcript_refid_var",
            variable.name = "cell_line", value.name = "ribonn")
rib[, cell_line := sub("^delta_", "", cell_line)]

m <- Reduce(function(a, b) merge(a, b, by = c("transcript_refid_var", "cell_line")),
            list(clf, reg, rib))
m <- m[complete.cases(m)]
m[, cell_line := factor(cell_line, levels = CELLS)]
cat(sprintf("Merged %d variant-cell pairs (%d variants x %d cell lines)\n",
            nrow(m), nrow(m) / length(CELLS), length(CELLS)))

# Sign concordance per cell line among variants that both scores call large
# (|score| above its own upper tertile in that cell line)
strong_sign <- function(a, b) m[, {
  ta <- quantile(abs(get(a)), 2/3); tb <- quantile(abs(get(b)), 2/3)
  k  <- abs(get(a)) > ta & abs(get(b)) > tb
  .(sc = mean(sign(get(a)[k]) == sign(get(b)[k])))
}, by = cell_line]

pairwise <- rbindlist(lapply(combn(METS, 2, simplify = FALSE), function(p) {
  a <- p[1]; b <- p[2]; ss <- strong_sign(a, b)$sc
  data.table(pair = paste(a, "vs", b),
             pearson         = cor(m[[a]], m[[b]]),
             spearman        = cor(m[[a]], m[[b]], method = "spearman"),
             sign_all        = mean(sign(m[[a]]) == sign(m[[b]])),
             sign_strong_min = min(ss), sign_strong_max = max(ss))
}))
print(pairwise, digits = 3)

# Fig. EV3E: Spearman matrix
cm <- CJ(a = METS, b = METS)[, rho := mapply(function(x, y)
          cor(m[[x]], m[[y]], method = "spearman"), a, b)]

# Fig. EV3F: TEFL (Class) against each alternative score
sc_cell <- rbindlist(lapply(c("ordinal", "our_reg", "ribonn"), function(b) {
  strong_sign("dpte", b)[, .(alt = b, cell_line, sign_concordance = sc)]
}))
sc <- sc_cell[, .(mean_sc = mean(sign_concordance), lo = min(sign_concordance), hi = max(sign_concordance)),
              by = alt]
print(sc, digits = 3)

fwrite(m, file.path(output_dir, "FigEV3D-F_variant_scores.tsv.gz"), sep = "\t", compress = "gzip")
fwrite(pairwise, file.path(output_dir, "FigEV3D-F_pairwise_stats.tsv"), sep = "\t")
fwrite(cm, file.path(output_dir, "FigEV3E_source_data.tsv"), sep = "\t")
fwrite(sc_cell, file.path(output_dir, "FigEV3F_source_data.tsv"), sep = "\t")
fwrite(sc, file.path(output_dir, "FigEV3F_summary.tsv"), sep = "\t")
