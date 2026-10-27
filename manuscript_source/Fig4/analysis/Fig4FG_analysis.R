# Fig. 4F, G: dual-luciferase reporter assays in GM12878 cells (4 replicates) for 5'UTR GWAS SNVs
# predicted to raise (F) or lower (G) TE. Values are reporter expression relative to Fluc.
# REF and ALT are compared by one-sided Student's t-test (equal variances) in the predicted direction.
# The predicted pTE of the REF and ALT alleles and the GWAS Catalog traits of each SNV are written too.
suppressPackageStartupMessages({
  library(data.table)
  library(dplyr)
})

file_arg <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
base_dir <- if (length(file_arg)) dirname(dirname(normalizePath(sub("^--file=", "", file_arg)))) else getwd()
input_dir <- file.path(base_dir, "input")
output_dir <- file.path(base_dir, "output")
shared_dir <- file.path(dirname(base_dir), "shared", "input")
dir.create(output_dir, showWarnings = FALSE)

PANELS <- list(F = list(genes = c("SPPL2A", "SMC2", "IRF3"), alternative = "less"),
               G = list(genes = c("APOA5", "ELP1"), alternative = "greater"))
# tested SNV of each gene: RefSeq transcript_REF_ALT_position in the mRNA
SNVS <- c(SPPL2A = "NM_032802_G_A_36", SMC2 = "NM_001042550_C_A_3", IRF3 = "NM_001197127_A_G_95",
          APOA5 = "NM_001166598_G_A_47", ELP1 = "NM_003640_C_A_16")

data <- read.table(file.path(input_dir, "reporter_assay_GWAS_SNVs_GM12878.tsv"), sep = "\t", header = TRUE)
df_data <- data.frame(gene = vapply(strsplit(data$Sample, "-"), `[`, "", 1),
                      value = data$Value,
                      rep = data$Replicates,
                      label = toupper(data$Genotype))
df_data <- df_data %>% group_by(gene, label) %>% mutate(mean = mean(value), sd = sd(value)) %>% ungroup() %>% as.data.frame()

for (panel in names(PANELS)) {
  cfg <- PANELS[[panel]]
  d <- df_data[df_data$gene %in% cfg$genes, ]
  # t.test(REF, MUT): "less" tests REF < MUT (predicted increase), "greater" REF > MUT (predicted decrease)
  tests <- rbindlist(lapply(cfg$genes, function(g) {
    ref <- d$value[d$gene == g & d$label == "REF"]
    mut <- d$value[d$gene == g & d$label == "MUT"]
    tt <- t.test(ref, mut, var.equal = TRUE, alternative = cfg$alternative)
    data.table(gene = g, mean_REF = mean(ref), mean_ALT = mean(mut), alternative = cfg$alternative,
               t = unname(tt$statistic), df = unname(tt$parameter), p_value = tt$p.value)
  }))
  print(tests, digits = 3)
  fwrite(d, file.path(output_dir, sprintf("Fig4%s_source_data.tsv", panel)), sep = "\t")
  fwrite(tests, file.path(output_dir, sprintf("Fig4%s_ttests.tsv", panel)), sep = "\t")
}

# predicted pTE (TEFL-mRNA, LCL) of the REF and ALT mRNA and the GWAS Catalog traits of each SNV
gwas <- unique(fread(file.path(input_dir, "GWAS_5UTR_SNVs_dpTE.tsv"), sep = "\t")[transcript_refid_var %in% SNVS,
  .(transcript_refid_var, chrom_pos_s, ref, alt, pos, pTE_REF = pred_0_all, pTE_ALT = pred_0_var, delta_pTE = delta_score_high)])
stopifnot(nrow(gwas) == length(SNVS))
traits <- fread(file.path(shared_dir, "GWAS_catalog_traits_5UTR_CDS_SNVs.tsv"), sep = "\t")
gwas[, GWAS_traits := vapply(chrom_pos_s, function(p) paste(sort(unique(traits[chrom_pos_s == p, MAPPED_TRAIT])), collapse = "; "), "")]
for (panel in names(PANELS)) {
  genes <- PANELS[[panel]]$genes
  pte <- cbind(gene = genes, gwas[match(SNVS[genes], gwas$transcript_refid_var)])
  print(pte[, .(gene, ref, alt, pTE_REF = round(pTE_REF, 3), pTE_ALT = round(pTE_ALT, 3), GWAS_traits)])
  fwrite(pte, file.path(output_dir, sprintf("Fig4%s_pTE.tsv", panel)), sep = "\t")
}
