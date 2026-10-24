# Fig. 4F, G: dual-luciferase reporter assays in GM12878 cells (4 replicates) for 5'UTR GWAS SNVs
# predicted to raise (F) or lower (G) TE. Values are reporter expression relative to Fluc.
# REF and ALT are compared by one-sided Student's t-test (equal variances) in the predicted direction.
suppressPackageStartupMessages({
  library(data.table)
  library(dplyr)
})

file_arg <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
base_dir <- if (length(file_arg)) dirname(dirname(normalizePath(sub("^--file=", "", file_arg)))) else getwd()
input_dir <- file.path(base_dir, "input")
output_dir <- file.path(base_dir, "output")
dir.create(output_dir, showWarnings = FALSE)

PANELS <- list(F = list(genes = c("SPPL2A", "SMC2", "IRF3"), alternative = "less"),
               G = list(genes = c("APOA5", "ELP1"), alternative = "greater"))

data <- read.table(file.path(input_dir, "20250625_GM12878_norm_rep4.tsv"), sep = "\t", header = TRUE)
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
