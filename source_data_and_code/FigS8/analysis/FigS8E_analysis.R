# Fig. S8E: delta pTE of TE-decreasing proline-creating missense SNVs by the length of the
# proline run they form in the mutant 5-residue peptide centred on the variant.
suppressPackageStartupMessages(library(data.table))

file_arg <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
base_dir <- if (length(file_arg)) dirname(dirname(normalizePath(sub("^--file=", "", file_arg)))) else getwd()
input_dir <- file.path(base_dir, "input")
output_dir <- file.path(base_dir, "output")
dir.create(output_dir, showWarnings = FALSE)

codonTable <- c("TTT"="F","TTC"="F","TTA"="L","TTG"="L","TAT"="Y","TAC"="Y","TAA"="*","TAG"="*",
                "CTT"="L","CTC"="L","CTA"="L","CTG"="L","CAT"="H","CAC"="H","CAA"="Q","CAG"="Q",
                "ATT"="I","ATC"="I","ATA"="I","ATG"="M","AAT"="N","AAC"="N","AAA"="K","AAG"="K",
                "GTT"="V","GTC"="V","GTA"="V","GTG"="V","GAT"="D","GAC"="D","GAA"="E","GAG"="E",
                "TCT"="S","TCC"="S","TCA"="S","TCG"="S","TGT"="C","TGC"="C","TGA"="*","TGG"="W",
                "CCT"="P","CCC"="P","CCA"="P","CCG"="P","CGT"="R","CGC"="R","CGA"="R","CGG"="R",
                "ACT"="T","ACC"="T","ACA"="T","ACG"="T","AGT"="S","AGC"="S","AGA"="R","AGG"="R",
                "GCT"="A","GCC"="A","GCA"="A","GCG"="A","GGT"="G","GGC"="G","GGA"="G","GGG"="G")

translate_dna_to_protein <- function(dna_seq, codon_table = codonTable) {
  dna_seq <- toupper(gsub("\\s", "", dna_seq))
  if (nchar(dna_seq) %% 3 != 0) {
    warning("DNA sequence length is not divisible by 3. Truncating to nearest multiple of 3.")
    dna_seq <- substr(dna_seq, 1, floor(nchar(dna_seq) / 3) * 3)
  }
  codons <- substring(dna_seq, seq(1, nchar(dna_seq), 3), seq(3, nchar(dna_seq), 3))
  amino_acids <- sapply(codons, function(codon) {
    if (codon %in% names(codon_table)) codon_table[codon] else "X"
  })
  paste(amino_acids, collapse = "")
}

# variant_pos is the 1-based position of the SNV in the CDS
generate_mutated_protein <- function(original_cds, variant_pos, ref_allele, alt_allele, codon_table = codonTable) {
  original_cds <- toupper(gsub("\\s", "", original_cds))
  ref_allele <- toupper(ref_allele)
  alt_allele <- toupper(alt_allele)
  if (variant_pos < 1 || variant_pos > nchar(original_cds)) {
    warning(paste("Variant position", variant_pos, "is out of range for sequence of length", nchar(original_cds)))
    return(NA)
  }
  actual_ref <- substr(original_cds, variant_pos, variant_pos)
  if (actual_ref != ref_allele) {
    warning(paste("Reference allele mismatch at position", variant_pos, ": expected", ref_allele, "but found", actual_ref))
  }
  mutated_cds <- paste0(substr(original_cds, 1, variant_pos - 1),
                        alt_allele,
                        substr(original_cds, variant_pos + 1, nchar(original_cds)))
  translate_dna_to_protein(mutated_cds, codon_table)
}

# 5-residue window: the variant residue and 2 residues on each side
extract_flanking_peptides <- function(original_protein, mutated_protein, variant_pos_cds) {
  if (is.na(mutated_protein) || is.na(original_protein)) {
    return(list(Original_Flanking = NA, Mutated_Flanking = NA))
  }
  protein_pos <- ceiling(variant_pos_cds / 3)
  flank_size <- 2
  start_pos <- max(1, protein_pos - flank_size)
  end_pos <- min(nchar(original_protein), protein_pos + flank_size)
  original_flanking <- substr(original_protein, start_pos, end_pos)
  mutated_flanking <- substr(mutated_protein, start_pos, min(nchar(mutated_protein), end_pos))
  position_info <- paste0("(", start_pos, "-", end_pos, ")")
  list(Original_Flanking = paste0(original_flanking, " ", position_info),
       Mutated_Flanking = paste0(mutated_flanking, " ", position_info))
}

# Longest run of consecutive prolines (>= min_length) in a peptide
check_polyP <- function(sequence, min_length = 2) {
  if (is.na(sequence) || sequence == "") {
    return(list(Has_PolyP = FALSE, PolyP_Length = 0, PolyP_Position = NA, PolyP_Sequence = NA))
  }
  clean_seq <- gsub("\\s*\\([^)]*\\)", "", sequence)
  polyP_matches <- gregexpr("P{2,}", clean_seq)[[1]]
  if (polyP_matches[1] == -1) {
    return(list(Has_PolyP = FALSE, PolyP_Length = 0, PolyP_Position = NA, PolyP_Sequence = NA))
  }
  polyP_lengths <- attr(polyP_matches, "match.length")
  max_length_idx <- which.max(polyP_lengths)
  max_length <- polyP_lengths[max_length_idx]
  max_position <- polyP_matches[max_length_idx]
  list(Has_PolyP = max_length >= min_length,
       PolyP_Length = max_length,
       PolyP_Position = max_position,
       PolyP_Sequence = substr(clean_seq, max_position, max_position + max_length - 1))
}

mis <- fread(file.path(input_dir, "gnomad_v4_missense_SNVs_delta_pTE_filter_0.05.tsv"), sep = "\t")
pro_neg <- mis[aa_var == "Pro" & delta_score_high < 0]

original_protein <- sapply(pro_neg$Sequences_CDS, translate_dna_to_protein, USE.NAMES = FALSE)
mutated_protein <- mapply(generate_mutated_protein,
                          original_cds = pro_neg$Sequences_CDS,
                          variant_pos = pro_neg$pos2start_cds,
                          ref_allele = pro_neg$ref,
                          alt_allele = pro_neg$alt,
                          USE.NAMES = FALSE)
flanking <- mapply(extract_flanking_peptides,
                   original_protein = original_protein,
                   mutated_protein = mutated_protein,
                   variant_pos_cds = pro_neg$pos2start_cds,
                   SIMPLIFY = FALSE)
pro_neg$Original_Flanking_5mer <- sapply(flanking, function(x) x$Original_Flanking)
pro_neg$Mutated_Flanking_5mer <- sapply(flanking, function(x) x$Mutated_Flanking)

# A gained proline without a neighbouring proline is counted as a run of length 1
pro_neg$Mutated_PolyP_Length <- sapply(pro_neg$Mutated_Flanking_5mer, function(flanking_seq) {
  polyP_info <- check_polyP(flanking_seq)
  if (polyP_info$Has_PolyP) polyP_info$PolyP_Length else 0
}, USE.NAMES = FALSE)
pro_neg$Mutated_PolyP_Length <- ifelse(pro_neg$Mutated_PolyP_Length == 0, 1, pro_neg$Mutated_PolyP_Length)

plot_data <- pro_neg[!is.na(delta_score_high) & !is.na(Mutated_PolyP_Length)]
plot_data[, PolyP_Length_Factor := as.factor(Mutated_PolyP_Length)]

summary_stats <- plot_data[, .(n = .N,
                               mean = mean(delta_score_high),
                               median = median(delta_score_high),
                               sd = sd(delta_score_high),
                               q25 = quantile(delta_score_high, 0.25),
                               q75 = quantile(delta_score_high, 0.75)),
                           keyby = Mutated_PolyP_Length]
summary_stats[, se := sd / sqrt(n)]
print(summary_stats)

anova_1to3 <- summary(aov(delta_score_high ~ PolyP_Length_Factor, data = plot_data[Mutated_PolyP_Length %in% c(1, 2, 3)]))[[1]]
anova_all_fit <- aov(delta_score_high ~ PolyP_Length_Factor, data = plot_data)
anova_all <- summary(anova_all_fit)[[1]]
kw_result <- kruskal.test(delta_score_high ~ PolyP_Length_Factor, data = plot_data)
cor_test <- cor.test(plot_data$Mutated_PolyP_Length, plot_data$delta_score_high, method = "spearman")

tests <- data.table(
  test = c("ANOVA", "ANOVA", "Kruskal-Wallis", "Spearman"),
  groups = c("1,2,3", "1,2,3,4", "1,2,3,4", "1,2,3,4"),
  statistic_name = c("F", "F", "chi-squared", "rho"),
  statistic = c(anova_1to3[1, "F value"], anova_all[1, "F value"], unname(kw_result$statistic), unname(cor_test$estimate)),
  p_value = c(anova_1to3[1, "Pr(>F)"], anova_all[1, "Pr(>F)"], kw_result$p.value, cor_test$p.value)
)
if (anova_all[1, "Pr(>F)"] < 0.05) {
  tukey <- TukeyHSD(anova_all_fit)$PolyP_Length_Factor
  tests <- rbind(tests, data.table(test = "Tukey HSD", groups = rownames(tukey), statistic_name = "difference",
                                   statistic = tukey[, "diff"], p_value = tukey[, "p adj"]))
}
print(tests)

fwrite(plot_data[, .(transcript_refid_var, variant_id, gene, HGVSc_protein,
                     Original_Flanking_5mer, Mutated_Flanking_5mer, Mutated_PolyP_Length, delta_score_high)],
       file.path(output_dir, "FigS8E_source_data.tsv"), sep = "\t")
fwrite(summary_stats, file.path(output_dir, "FigS8E_summary.tsv"), sep = "\t")
fwrite(tests, file.path(output_dir, "FigS8E_tests.tsv"), sep = "\t")
