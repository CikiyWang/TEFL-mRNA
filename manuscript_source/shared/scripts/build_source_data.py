#!/usr/bin/env python3
"""Build the two Source Data workbooks from the figure output tables.

    python3 shared/scripts/build_source_data.py

Writes journal_source_data/SourceData_Figure_main.xlsx and journal_source_data/SourceData_Figures_EV_Appendix.xlsx.
Each workbook has a README sheet (panel titles and column definitions) and one sheet per figure panel.
Run it after the analysis scripts. Needs Python 3 with pandas and openpyxl.
"""
import math
from pathlib import Path

import openpyxl.cell._writer as cell_writer
import pandas as pd
from openpyxl import Workbook
from openpyxl.cell import WriteOnlyCell
from openpyxl.styles import Font, PatternFill
from openpyxl.utils import get_column_letter

# openpyxl writes numbers as "%.16g", which drops the 17th significant digit that some doubles need;
# repr() is the shortest string that reads back as the same double.
_safe_string = cell_writer.safe_string


def exact_number(v):
    if type(v) is float and math.isfinite(v):
        return repr(v)
    if type(v) is int:
        return str(v)
    return _safe_string(v)


cell_writer.safe_string = exact_number

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "journal_source_data"
EXCEL_MAX_ROWS = 1_048_576

# ---------------------------------------------------------------- column definitions
COLS = {
    "seq_id": "RefSeq transcript", "transcript_id": "RefSeq transcript", "transcript_refid": "RefSeq transcript",
    "tx_id": "Ensembl transcript", "transcript_refid_var": "SNV-transcript pair: RefSeq transcript_REF_ALT_position in the mRNA",
    "transcript_refid_all": "SNV-transcript pair: RefSeq transcript_REF_ALT_position in the mRNA",
    "variant_id": "variant: chromosome@position_REF/ALT (hg38)", "chrom_pos_s": "variant position, chromosome:position (hg38)",
    "delta_score_high": "ΔpTE = pTE(ALT) − pTE(REF)", "delta_pTE": "ΔpTE = pTE(ALT) − pTE(REF)", "abs_delta_pTE": "|ΔpTE|",
    "pred_0": "pTE, predicted probability of the TE-high class", "pTE": "pTE, predicted probability of the TE-high class",
    "cell_line": "cell line of the TE model", "Cell_line": "cell line", "Cell_Line": "cell line", "fold": "cross-validation fold",
    "n": "number of observations", "p_value": "P value", "pearson_r": "Pearson correlation coefficient",
    "spearman_rho": "Spearman's rank correlation coefficient", "auc": "area under the ROC curve", "AUC": "area under the ROC curve",
    "specificity": "specificity (x axis)", "sensitivity": "sensitivity (y axis)", "gene": "gene symbol", "Gene": "gene symbol",
    "odds_ratio": "odds ratio", "mean": "mean", "median": "median", "sd": "standard deviation", "ref": "REF base", "alt": "ALT base",
    "region": "transcript region", "aa_novar": "reference amino acid", "aa_var": "amino acid introduced by the SNV",
    "Feature": "transcript feature", "feature": "transcript feature", "bin": "bin number",
}
BINS = {"Feature": "transcript feature", "bin": "bin number (mRNAs sorted by the feature, 20 per bin)",
        "mean_pred_bins": "mean pTE of the bin (y)", "sd_pred_bins": "standard deviation of pTE in the bin",
        "mean_value_bins": "mean feature value of the bin (x)", "sd_value_bins": "standard deviation of the feature in the bin"}
BIN_CORR = {"panel": "panel", "Feature": "transcript feature", "x_scale": "x axis scale of the correlation",
            "n_bins": "bins", "pearson_r": "Pearson R of the bin means", "p_value": "P value"}
BIN_CORR_LINEAR = {k: v for k, v in BIN_CORR.items() if k != "x_scale"}
FOLD_ROC = {"fold": "cross-validation fold", "specificity": "specificity (x axis)", "sensitivity": "sensitivity (y axis)"}
AUC = {"fold": "cross-validation fold", "n": "held-out mRNAs", "n_high": "TE-high mRNAs among them", "auc": "area under the ROC curve"}
PTE_BINS = {"Class": "pTE bin", "count": "mRNAs in the bin", "Frac": "fraction of the group (bar segment)"}
PTE_TESTS = {"test": "one-sided Kolmogorov-Smirnov test (with > without), or Fisher's exact test of pTE above the cutoff",
             "statistic": "KS D, or Fisher odds ratio", "p_value": "P value", "cutoff": "pTE cutoff of the Fisher test",
             "n_without_low": "mRNAs without, pTE ≤ cutoff", "n_without_high": "mRNAs without, pTE > cutoff",
             "n_with_low": "mRNAs with, pTE ≤ cutoff", "n_with_high": "mRNAs with, pTE > cutoff"}
TRAITS = {"Traits": "GWAS Catalog mapped trait", "Traits_upper": "trait as printed",
          "Counts_iso": "transcripts with a TE-altering 5′UTR SNV (|ΔpTE| > 0.1) of the trait (bar)",
          "Counts_traits_iso": "transcripts with a 5′UTR SNV of the trait", "Fracs_iso": "Counts_iso / Counts_traits_iso (bar colour)",
          "Pval_iso": "one-sided Fisher's exact P", "Odd_iso": "odds ratio"}
ENRICH = {"ID": "term ID", "Description": "term name", "GeneRatio": "genes in the term / genes tested",
          "BgRatio": "genes in the term / background genes", "pvalue": "hypergeometric P", "p.adjust": "FDR-adjusted P",
          "qvalue": "q value", "geneID": "genes in the term (/-separated)", "Count": "genes in the term"}
REPORTER = {"gene": "gene symbol", "value": "luciferase ratio (Nluc relative to Fluc)", "rep": "biological replicate",
            "label": "REF or MUT (ALT) reporter", "mean": "mean of the replicates (bar)", "sd": "standard deviation (error bar)"}
TTEST = {"gene": "gene symbol", "mean_REF": "mean, REF reporter", "mean_ALT": "mean, ALT reporter",
         "alternative": "alternative hypothesis of t.test(REF, ALT)", "t": "t statistic", "df": "degrees of freedom",
         "p_value": "one-sided P"}
PTE_ALLELES = {"gene": "gene symbol", "transcript_refid_var": COLS["transcript_refid_var"], "chrom_pos_s": COLS["chrom_pos_s"],
               "ref": "REF base", "alt": "ALT base", "pos": "position in the mRNA", "pTE_REF": "pTE of the REF mRNA",
               "pTE_ALT": "pTE of the ALT mRNA", "delta_pTE": COLS["delta_pTE"], "GWAS_traits": "GWAS Catalog mapped traits of the SNV"}
EV1_STATS = {"panel": "panels the row belongs to", "estimate": "TE estimate",
             "spearman_vs_te_ratio": "Spearman's ρ with TPM-ratio TE (printed on the panel)",
             "consensus_class_agreement_pct": "percentage of transcripts given their consensus class",
             "highlow_reversal_pct": "percentage of consensus TE-high or TE-low transcripts placed in the opposite extreme class",
             "highlow_reversals": "the same as a count", "tertile_class_agreement_pct": "class agreement when both TE values are split into tertiles",
             "spearman_vs_rna_tpm": "Spearman's ρ with mean RNA TPM"}
EV1_CONC = {"consensus_class": "consensus TE class (row)", "assigned_class": "rank-matched class of the alternative estimate (column)",
            "n": "transcripts", "percent_of_consensus_class": "percentage of the row (cell colour)"}


def keep(cols):
    return lambda d: d[cols]


def rows(col, values, cols=None):
    def f(d):
        d = d[d[col].isin(values)]
        return d[cols] if cols else d
    return f


def drop_rows(col, values, cols=None):
    def f(d):
        d = d[~d[col].isin(values)]
        return d[cols] if cols else d
    return f


class T:
    """One table on a panel sheet: output file, label, column definitions (order = sheet order), transform.
    readme_wildcard=False leaves the columns covered only by "*" out of the README."""
    def __init__(self, file, label, cols, transform=None, readme_wildcard=True):
        self.file, self.label, self.cols, self.transform = file, label, cols, transform
        self.readme_wildcard = readme_wildcard


class P:
    """One panel sheet."""
    def __init__(self, sheet, title, tables, note=None):
        self.sheet, self.title, self.tables, self.note = sheet, title, tables, note


def cols_of(t):
    return [c for c in t.cols if c != "*"]


EV = "ExpandedView/"
AP = "Appendix/"
MAIN = [
    P("Fig1C", "Classifier performance for the TE-high class in 10 cross-validation folds", [
        T("Fig1/output/Fig1C_source_data.tsv", None, {
            "fold": "cross-validation fold", "Criteria": "metric", "perf": "metric value in the fold (point)",
            "class": "class scored", "mean": "mean over the 10 folds (bar)", "upper_ci": "mean + SD (error bar)",
            "lower_ci": "mean − SD (error bar)"})]),
    P("Fig1D", "ROC curves of the 10 cross-validation folds", [
        T("Fig1/output/Fig1D_auc.tsv", "AUC per fold", AUC),
        T("Fig1/output/Fig1D_source_data.tsv.gz", "ROC curve points", FOLD_ROC)]),
    P("Fig1E", "Measured TE of mRNAs by pTE bin", [
        T("Fig1/output/Fig1E_stats.tsv", "Correlation over all mRNAs", {
            "n": "mRNAs", "spearman_rho": "Spearman's ρ of pTE and TE", "spearman_p": "P",
            "pearson_r_log10_TE_plus1": "Pearson r of pTE and log10(TE + 1)"}),
        T("Fig1/output/Fig1E_bin_medians.tsv", "Median per pTE bin (line)", {"bin": "pTE bin", "log10_TE_plus1": "median log10(TE + 1)"}),
        T("Fig1/output/Fig1E_source_data.tsv", "Per-mRNA values", {
            "transcript_id": "RefSeq transcript", "pTE": COLS["pTE"], "TE": "measured TE, median across the 47 individuals",
            "log10_TE_plus1": "log10(TE + 1) (y)", "bin": "pTE bin (x)"})]),
    P("Fig1F", "pTE of mRNAs with and without a 3′UTR miRNA binding site", [
        T("Fig1/output/Fig1F_bin_fractions.tsv", "Fraction of mRNAs per pTE bin",
          {"if_miBS": "with or without a 3′UTR binding site of a miRNA expressed in naive B cells", **PTE_BINS}),
        T("Fig1/output/Fig1F_tests.tsv", "Tests", PTE_TESTS),
        T("Fig1/output/Fig1F_source_data.tsv", "Per-mRNA values", {
            "seq_id": "RefSeq transcript", "pred_0": COLS["pred_0"], "n_UTR3": "3′UTR binding sites",
            "if_miBS": "with or without a site", "Class": "pTE bin"})]),
    P("Fig1G", "pTE of mRNAs with and without a strong uORF", [
        T("Fig1/output/Fig1G_bin_fractions.tsv", "Fraction of mRNAs per pTE bin",
          {"if_uORF": "with (at least one strong uORF in uORFdb) or without", **PTE_BINS}),
        T("Fig1/output/Fig1G_tests.tsv", "Tests", PTE_TESTS),
        T("Fig1/output/Fig1G_source_data.tsv", "Per-mRNA values", {
            "transcript_refid": "RefSeq transcript", "pred_0": COLS["pred_0"], "count_n": "strong uORFs",
            "if_uORF": "with or without a strong uORF", "Class": "pTE bin"})]),
    P("Fig2B", "|ΔpTE| of gnomAD v4.0 exome SNVs by transcript region", [
        T("Fig2/output/Fig2B_left_source_data.tsv", "Left: SNV-transcript pairs above each |ΔpTE| cutoff", {
            "region": "transcript region", "cutoff": "|ΔpTE| cutoff", "n_above_cutoff": "pairs above the cutoff",
            "n_total": "pairs in the region", "fraction": "n_above_cutoff / n_total (bar segment)"}),
        T("Fig2/output/Fig2B_right_N.tsv", "Right: N above each box", {
            "region": "transcript region", "n_pairs": "pairs with |ΔpTE| > 0.1", "n_unique_snvs": "unique SNVs among them",
            "n_label": "N printed on the panel"}),
        T("Fig2/output/Fig2B_right_source_data.tsv", "Right: pairs with |ΔpTE| > 0.1 (box plots)", {
            "region": "transcript region (x)", "snv_class": "variant class", "transcript_refid_var": COLS["transcript_refid_var"],
            "variant_id": COLS["variant_id"], "delta_pTE": COLS["delta_pTE"], "abs_delta_pTE": "|ΔpTE| (y)"})]),
    P("Fig2C", "|ΔpTE| of gnomAD v4.0 exome 5′UTR SNVs against allele frequency", [
        T("Fig2/output/Fig2C_source_data_stats.tsv", "Least-squares fit of |ΔpTE| on log10(AF), one point per SNV", {
            "n": "SNVs", "slope": "slope", "intercept": "intercept", "slope_lo": "95% CI of the slope, lower",
            "slope_hi": "95% CI of the slope, upper", "r_squared": "R²", "P_slope": "P of the slope", "spearman_rho": "Spearman's ρ"}),
        T("Fig2/output/Fig2C_source_data_hexbin.tsv", "Hexagonal bins (60 across)", {
            "log10AF": "bin centre, log10 gnomAD exome allele frequency (floored at 1e-6)",
            "abs_delta": "bin centre, |ΔpTE| averaged over the isoforms of the SNV", "count": "SNVs in the bin (colour)"})],
      note="Per-SNV values (2,620,802 SNVs) come from shared/input/gnomAD_v4_5UTR_SNVs_dpTE.tsv.gz of the data archive."),
    P("Fig2D", "pTE of mRNAs with uAUG-creating 5′UTR SNVs and of all RefSeq mRNAs", [
        T("Fig2/output/Fig2D_source_data.tsv", "Fraction per pTE bin", {
            "if_uAUG": "uAUG_creating (SNV-mRNA pairs, pTE of the mutated mRNA) or Reference (RefSeq mRNAs)",
            "Class": "pTE bin", "count": "number in the bin", "Frac": "fraction of the group (bar segment)",
            "n_unique_snvs": "unique uAUG-creating SNVs (N on the panel)"}),
        T("Fig2/output/Fig2D_fisher_test.tsv", "Fisher's exact test", {
            "group": "group", "n_high_pTE": "number in the bins above pTE 0.5", "n_low_pTE": "number in the bins 0-0.2 and 0.3-0.5",
            "odds_ratio": "odds ratio", "p_value": "P value"})]),
    P("Fig2E", "Pairs with ΔpTE < 0 among uAUG-creating SNVs by reference-allele pTE", [
        T("Fig2/output/Fig2E_source_data.tsv", None, {
            "cutoff": "|ΔpTE| cutoff (0 = all pairs)", "ref_bin": "bin of reference-allele pTE", "n": "pairs with ΔpTE ≠ 0",
            "n_neg": "pairs with ΔpTE < 0", "n_pos": "pairs with ΔpTE > 0", "pct_neg": "percentage with ΔpTE < 0 (bar)",
            "pct_pos": "percentage with ΔpTE > 0", "ci_lo": "95% Clopper-Pearson interval of pct_neg, lower", "ci_hi": "upper",
            "log10_p": "log10 P, two-sided exact sign test against 50%"})]),
    P("Fig2G", "Luciferase reporter assays of six gnomAD 5′UTR SNVs in GM12878 cells", [
        T("Fig2/output/Fig2G_source_data.tsv", "Reporter measurements", {
            "group": "left (predicted TE increase) or right (decrease)", "gene": "gene symbol", "allele": "REF or MUT reporter",
            "replicate": "biological replicate", "value": "luciferase ratio (Nluc relative to Fluc)"}),
        T("Fig2/output/Fig2G_ttest.tsv", "One-tailed t-tests and predicted pTE", {
            "group": "panel side", "gene": "gene symbol", "transcript_id": "RefSeq transcript", "position": "variant position in the mRNA",
            "ref": "REF base", "alt": "ALT base", "gnomad_variant_id": "gnomAD variant", "delta_pTE": COLS["delta_pTE"],
            "ref_pTE": "pTE of the REF mRNA", "mut_pTE": "pTE of the MUT mRNA (REF pTE + ΔpTE)", "mean_REF": "mean, REF",
            "sd_REF": "SD, REF", "mean_MUT": "mean, MUT", "sd_MUT": "SD, MUT", "alternative": "alternative hypothesis",
            "t": "t statistic", "df": "degrees of freedom", "p_value": "one-tailed P"})]),
    P("Fig3A", "ΔpTE of synonymous SNVs (|ΔpTE| > 0.05) by codon substitution", [
        T("Fig3/output/Fig3A_codon_substitution_dpTE.tsv", None, {
            "ref_alt_codon": "REF > ALT codon", "delta_codon_var_freq_class": "ALT codon more or less optimal than REF (Kazusa codon usage)",
            "aa_label": "amino acid", "median_dpTE": "median ΔpTE", "mean_dpTE": "mean ΔpTE (heatmap colour)",
            "n_variants": "SNV-transcript pairs"})]),
    P("Fig3B", "TE-increasing and TE-decreasing synonymous SNVs by ALT-codon optimality", [
        T("Fig3/output/Fig3B_counts.tsv", "Counts", {
            "count": "synonymous SNV-transcript pairs (bar)", "delta_TE_class": "up- or down-regulated (sign of ΔpTE)",
            "SNVs_codon_class": "optimal or non-optimal ALT codon"}),
        T("Fig3/output/Fig3B_fisher_test.tsv", "Fisher's exact test", {"test": "test", "odds_ratio": "odds ratio", "p_value": "P value"})]),
    P("Fig3C", "ΔpTE of synonymous SNVs (|ΔpTE| > 0.05) by amino acid and G/C bases in the ALT codon", [
        T("Fig3/output/Fig3C_group_medians.tsv", "Group medians (dashed lines)", {
            "codon_var_class_aa": "amino acid", "codon_var_class_GC": "G/C bases in the ALT codon", "median_value": "median ΔpTE",
            "n": "SNV-transcript pairs"}, drop_rows("codon_var_class_aa", ["Gln", "Glu"])),
        T("Fig3/output/Fig3C_variants.tsv", "SNV-transcript pairs", {
            "transcript_refid_all": COLS["transcript_refid_all"], "codon_var_class_aa": "amino acid",
            "codon_var_class_GC": "G/C bases in the ALT codon", "delta_score_high": "ΔpTE (y)"},
          drop_rows("codon_var_class_aa", ["Gln", "Glu"]))]),
    P("Fig3D", "ΔpTE of synonymous SNVs (|ΔpTE| > 0.05) in Gln and Glu codons by G/C bases in the ALT codon", [
        T("Fig3/output/Fig3C_group_medians.tsv", "Group medians (dashed lines)", {
            "codon_var_class_aa": "amino acid", "codon_var_class_GC": "G/C bases in the ALT codon", "median_value": "median ΔpTE",
            "n": "SNV-transcript pairs"}, rows("codon_var_class_aa", ["Gln", "Glu"])),
        T("Fig3/output/Fig3C_variants.tsv", "SNV-transcript pairs", {
            "transcript_refid_all": COLS["transcript_refid_all"], "codon_var_class_aa": "amino acid",
            "codon_var_class_GC": "G/C bases in the ALT codon", "delta_score_high": "ΔpTE (y)"},
          rows("codon_var_class_aa", ["Gln", "Glu"]))]),
    P("Fig3E", "ΔpTE of missense SNVs (|ΔpTE| > 0.05) introducing Phe or Leu", [
        T("Fig3/output/Fig3EF_aa_medians.tsv", "Median per amino acid", {
            "aa_var": "amino acid introduced", "median_delta_score_high": "median ΔpTE", "n": "SNV-transcript pairs"},
          rows("aa_var", ["Phe", "Leu"])),
        T("Fig3/output/Fig3EF_variants.tsv", "SNV-transcript pairs", {
            "transcript_refid_var": COLS["transcript_refid_var"], "variant_id": COLS["variant_id"], "aa_novar": "reference amino acid",
            "aa_var": "amino acid introduced", "delta_score_high": "ΔpTE (y)"}, rows("aa_var", ["Phe", "Leu"]))]),
    P("Fig3F", "ΔpTE of missense SNVs (|ΔpTE| > 0.05) introducing Gly, Thr or Pro", [
        T("Fig3/output/Fig3EF_aa_medians.tsv", "Median per amino acid", {
            "aa_var": "amino acid introduced", "median_delta_score_high": "median ΔpTE", "n": "SNV-transcript pairs"},
          rows("aa_var", ["Gly", "Thr", "Pro"])),
        T("Fig3/output/Fig3EF_variants.tsv", "SNV-transcript pairs", {
            "transcript_refid_var": COLS["transcript_refid_var"], "variant_id": COLS["variant_id"], "aa_novar": "reference amino acid",
            "aa_var": "amino acid introduced", "delta_score_high": "ΔpTE (y)"}, rows("aa_var", ["Gly", "Thr", "Pro"]))]),
    P("Fig4A", "GO biological-process enrichment of genes with TE-altering gnomAD SNVs (|ΔpTE| > 0.1)", [
        T("Fig4/output/Fig4A_source_data.tsv", "GO terms placed by semantic similarity (points)", {
            "go": "GO term", "V1": "first PCoA component of 1 − semantic similarity (x)", "V2": "second PCoA component (y)",
            "term": "GO term name", "parent": "parent GO term (rrvgo, threshold 0.9)", "parentTerm": "parent term name (colour)",
            "size": "genes annotated to the GO term (point size)"}),
        T("Fig4/output/Fig4A_GO_enrichment.tsv", "Enrichment of the biological-process terms (FDR < 0.05)", ENRICH,
          rows("ONTOLOGY", ["BP"], list(ENRICH)))]),
    P("Fig4B", "Disease Ontology enrichment of the same genes", [
        T("Fig4/output/Fig4B_source_data.tsv", None, {**ENRICH, "Count": "genes in the term (point size)",
                                                     "p.adjust": "FDR-adjusted P (colour)"})]),
    P("Fig4C", "GWAS disease traits enriched for TE-altering 5′UTR SNVs", [
        T("Fig4/output/Fig4C_source_data.tsv", None, TRAITS, keep(list(TRAITS)))]),
    P("Fig4D", "GWAS non-disease traits enriched for TE-altering 5′UTR SNVs", [
        T("Fig4/output/Fig4D_source_data.tsv", None, TRAITS, keep(list(TRAITS)))]),
    P("Fig4E", "Top 10 immune-related ClinVar SNVs by |ΔpTE| in the 5′UTR and CDS", [
        T("Fig4/output/Fig4E_source_data.tsv", None, {
            "delta_score": "ΔpTE (bar)", "Region": "5′UTR, or CDS (synonymous variants)", "Var_id": COLS["transcript_refid_var"],
            "seq_id": "RefSeq transcript", "pos": "variant position in the mRNA", "Gene": "gene symbol",
            "ref2alt": "REF > ALT base", "clinvar_id": "ClinVar variation ID",
            "ClinVar_CLNDN": "ClinVar condition(s), 2024-05-14 release", "Name": "label: gene@transcript@REF > ALT@position"})]),
    P("Fig4F", "Luciferase reporter assays of GWAS 5′UTR SNVs predicted to increase TE (GM12878)", [
        T("Fig4/output/Fig4F_source_data.tsv", "Reporter measurements", REPORTER),
        T("Fig4/output/Fig4F_ttests.tsv", "One-tailed t-tests", TTEST),
        T("Fig4/output/Fig4F_pTE.tsv", "Predicted pTE and GWAS traits", PTE_ALLELES)]),
    P("Fig4G", "Luciferase reporter assays of GWAS 5′UTR SNVs predicted to decrease TE (GM12878)", [
        T("Fig4/output/Fig4G_source_data.tsv", "Reporter measurements", REPORTER),
        T("Fig4/output/Fig4G_ttests.tsv", "One-tailed t-tests", TTEST),
        T("Fig4/output/Fig4G_pTE.tsv", "Predicted pTE and GWAS traits", PTE_ALLELES)]),
    P("Fig5A", "ROC curves of the TE classifiers of nine cell lines (10 folds each)", [
        T("Fig5/output/Fig5A_auc.tsv", "AUC per cell line and fold", {"cell_line": "cell line", **AUC}),
        T("Fig5/output/Fig5A_source_data.tsv.gz", "ROC curve points", {"cell_line": "cell line", **FOLD_ROC})]),
    P("Fig5B", "Agreement of ExAC 5′UTR SNV ΔpTE between cell lines", [
        T("Fig5/output/Fig5B_correlation.tsv", "Pearson correlation (tiles)", {
            "cell_line_1": "cell line", "cell_line_2": "cell line", "n": "SNV-transcript pairs with |ΔpTE| > 0.02 in at least one cell line",
            "pearson_r": "Pearson r", "p_value": "two-sided P"}),
        T("Fig5/output/Fig5B_direction_agreement.tsv", "Sign agreement (pie charts)", {
            "cell_line_1": "cell line", "cell_line_2": "cell line", "both_positive": "pairs with ΔpTE > 0 in both",
            "both_negative": "pairs with ΔpTE < 0 in both", "opposite": "pairs with opposite signs", "total": "pairs",
            "pct_both_positive": "percentage", "pct_both_negative": "percentage", "pct_opposite": "percentage"})]),
    P("Fig5C", "Cross-cell-line concordance of ExAC SNV ΔpTE direction by region", [
        T("Fig5/output/Fig5C_source_data.tsv", None, {
            "region": "transcript region", "category": "Shared (concordance score > 0.8 or < 0.2) or Specific (others)",
            "count": "SNV-transcript pairs with |ΔpTE| > 0.02 in more than one cell line", "proportion": "fraction of the region (bar)"})]),
    P("Fig5D", "ALT-allele composition of ExAC 5′UTR SNVs passing ΔpTE cutoffs in at least 3 cell lines", [
        T("Fig5/output/Fig5D_source_data.tsv", None, {
            "TE": "TE-increasing or TE-decreasing", "cutoff": "ΔpTE cutoff", "alt": "ALT base (T = U)",
            "n": "pairs with this ALT base", "n_snvs": "pairs passing the cutoff", "freq": "n / n_snvs (bar segment)"})]),
    P("Fig5E", "Distance to the start codon of ExAC 5′UTR SNVs passing |ΔpTE| cutoffs in at least 3 cell lines", [
        T("Fig5/output/Fig5E_ecdf.tsv", "Cumulative distributions (lines)", {
            "cutoff": "|ΔpTE| cutoff (0 = all 5′UTR pairs)", "distance_to_start_codon_nt": "distance to the start codon (nt, x)",
            "n_pairs_at_distance": "pairs at this distance", "cumulative_fraction": "fraction of pairs at or below this distance (y)"}),
        T("Fig5/output/Fig5E_ks_tests.tsv", "Pairwise two-sample Kolmogorov-Smirnov tests", {
            "cutoff_1": "cutoff", "cutoff_2": "cutoff", "n_1": "pairs", "n_2": "pairs", "D": "KS statistic", "p_value": "P value"})]),
    P("Fig5F", "Allele-frequency classes of ExAC 5′UTR SNVs passing |ΔpTE| cutoffs in at least 3 cell lines", [
        T("Fig5/output/Fig5F_source_data.tsv", None, {
            "Cutoffs": "|ΔpTE| cutoff passed in the same direction in at least 3 cell lines",
            "AF_class": "gnomAD allele-frequency class", "Var_counts": "SNV-transcript pairs in the class",
            "Percents_allvar": "fraction of the pairs passing the cutoff (bar, plotted as a percentage)"})]),
    P("Fig6A", "Median ΔpTE of synonymous ExAC SNVs (|ΔpTE| > 0.02) by codon substitution and cell line", [
        T("Fig6/output/Fig6A_median_dpTE_matrix.tsv", "Median ΔpTE (heatmap)", {
            "ref_alt_codon": "REF > ALT codon", "delta_codon_var_freq_class": "ALT codon more or less optimal", "aa_var": "amino acid",
            "*": "median ΔpTE in the cell line named in the column header"}),
        T("Fig6/output/Fig6A_codon_substitution_dpTE_by_cell_line.tsv", "SNV-transcript pairs per heatmap cell", {
            "ref_alt_codon": "REF > ALT codon", "cell_line": "cell line", "n_variants": "SNV-transcript pairs"},
          keep(["ref_alt_codon", "cell_line", "n_variants"]))]),
    P("Fig6B", "ΔpTE of proline-creating missense ExAC SNVs (|ΔpTE| > 0.02) by cell line", [
        T("Fig6/output/Fig6B_summary.tsv", "Summary per cell line", {
            "cell_line": "cell line (ordered by mean ΔpTE)", "n_SNVs": "SNV-transcript pairs", "mean_delta_pTE": "mean ΔpTE",
            "median_delta_pTE": "median ΔpTE", "frac_delta_pTE_negative": "fraction with ΔpTE < 0"}),
        T("Fig6/output/Fig6B_proline_SNVs_dpTE.tsv", "SNV-transcript pairs (violins)", {
            "cell_line": "cell line", "transcript_refid_var": COLS["transcript_refid_var"], "delta_score_high": "ΔpTE (y)"})]),
    P("Fig6C", "TE-increasing and TE-decreasing missense GWAS SNVs by cell line", [
        T("Fig6/output/Fig6C_source_data.tsv", None, {
            "Cell_Line": "cell line", "Regulation": "TE-increasing (ΔpTE > 0.02) or TE-decreasing (ΔpTE < −0.02)",
            "Fraction": "fraction of the SNV-transcript pairs with |ΔpTE| > 0.02 (bar)", "Counts": "number (printed above the bar)",
            "Ratio": "increasing / decreasing (x-axis order)"}, keep(["Cell_Line", "Regulation", "Fraction", "Counts", "Ratio"]))]),
    P("Fig6D", "Spearman correlation of transcript features with measured TE in nine cell lines", [
        T("Fig6/output/Fig6D_plot_data.tsv", None, {
            "Feature": "transcript feature (AMFE = MFE / length)", "Cell_line": "cell line", "n": "transcripts",
            "cors_TE": "Spearman's ρ with measured TE (bar)", "pvals_TE": "P value", "significance": "stars",
            "custom_color": "bar colour (−log10 P scaled within the feature across 78 cell lines)"},
          keep(["Feature", "Cell_line", "n", "cors_TE", "pvals_TE", "significance", "custom_color"]))]),
    P("Fig6E", "Disease-trait enrichment of TE-altering GWAS 5′UTR and CDS SNVs by cell line", [
        T("Fig6/output/Fig6E_disease_trait_enrichment.tsv", None, {
            "trait": "GWAS disease trait", "region": "5′UTR or CDS", "cell_line": "cell line", "odds_ratio": "odds ratio",
            "odds_ratio_z": "odds ratio z-scored within the trait (heatmap colour)", "fisher_p": "one-sided Fisher's exact P"},
          keep(["trait", "region", "cell_line", "odds_ratio", "odds_ratio_z", "fisher_p"]))]),
]

EV_APPENDIX = [
    P("FigEV1A", "Pairwise Spearman correlation of transcript TE between the 47 individuals", [
        T(EV + "FigEV1/output/FigEV1A_pairwise_spearman_matrix.csv", None,
          {"individual": "individual (row)", "*": "Spearman's ρ with the individual named in the column header"},
          lambda d: d.rename(columns={d.columns[0]: "individual"}), readme_wildcard=False)]),
    P("FigEV1B", "Agreement of TE classes across the 47 individuals", [
        T(EV + "FigEV1/output/FigEV1B_per_transcript_agreement.csv", None, {
            "seq_id": "RefSeq transcript", "n_indiv": "individuals with a TE value", "n_low": "individuals calling TE-low",
            "n_int": "individuals calling TE-intermediate", "n_high": "individuals calling TE-high", "top_label": "most frequent class",
            "agree": "fraction of individuals calling top_label (histogram)", "variability": "which classes occur (bar colour)"})]),
    P("FigEV1C", "Mean LCL TE against RiboNN LCL TE (consensus TE label set)", [
        T(EV + "FigEV1/output/FigEV1C_LCL_TE_vs_RiboNN.csv", None, {
            "seq_id": "RefSeq transcript", "TE_LCL": "RiboNN LCL TE, log (Zheng et al. 2025) (y)",
            "our": "mean TE across the 47 individuals (x)"}, rows("in_published", [True], ["seq_id", "TE_LCL", "our"]))]),
    P("FigEV1D", "Compositional against TPM-ratio TE", [
        T(EV + "FigEV1/output/FigEV1D-H_statistics.tsv", "Correlation", EV1_STATS, rows("panel", ["D, E"], list(EV1_STATS))),
        T(EV + "FigEV1/output/FigEV1D-H_source_data.tsv", "Per-transcript TE", {
            "seq_id": "RefSeq transcript", "consensus_class": "consensus TE class (colour)", "te_ratio": "TPM-ratio TE",
            "te_ratio_log2": "log2 TPM-ratio TE (x)", "te_compositional_log2": "compositional TE, ΔCLR in log2 units (y)"},
          keep(["seq_id", "consensus_class", "te_ratio", "te_ratio_log2", "te_compositional_log2"]))]),
    P("FigEV1E", "Consensus against compositional TE classes", [
        T(EV + "FigEV1/output/FigEV1EG_class_concordance.tsv", None, EV1_CONC, rows("panel", ["E"], list(EV1_CONC)))]),
    P("FigEV1F", "Regression-residual against TPM-ratio TE", [
        T(EV + "FigEV1/output/FigEV1D-H_statistics.tsv", "Correlation", EV1_STATS, rows("panel", ["F, G"], list(EV1_STATS))),
        T(EV + "FigEV1/output/FigEV1D-H_source_data.tsv", "Per-transcript TE", {
            "seq_id": "RefSeq transcript", "consensus_class": "consensus TE class (colour)", "te_ratio": "TPM-ratio TE",
            "te_ratio_log2": "log2 TPM-ratio TE (x)", "te_regression_log2": "regression-residual TE, log2 units (y)"},
          keep(["seq_id", "consensus_class", "te_ratio", "te_ratio_log2", "te_regression_log2"]))]),
    P("FigEV1G", "Consensus against regression-residual TE classes", [
        T(EV + "FigEV1/output/FigEV1EG_class_concordance.tsv", None, EV1_CONC, rows("panel", ["G"], list(EV1_CONC)))]),
    P("FigEV1H_top", "Consensus TE classes by RNA-abundance quartile", [
        T(EV + "FigEV1/output/FigEV1H_quartile_summary.tsv", "Class composition per quartile", {
            "quartile": "quartile of mean RNA TPM", "n": "transcripts", "median_rna_tpm": "median RNA TPM",
            "pct_low": "percentage TE-low", "pct_intermediate": "percentage TE-intermediate", "pct_high": "percentage TE-high",
            "chi2": "chi-square statistic", "chi2_dof": "degrees of freedom", "chi2_p": "chi-square P"},
          keep(["quartile", "n", "median_rna_tpm", "pct_low", "pct_intermediate", "pct_high", "chi2", "chi2_dof", "chi2_p"])),
        T(EV + "FigEV1/output/FigEV1D-H_source_data.tsv", "Per-transcript quartile and class", {
            "seq_id": "RefSeq transcript", "rna_mean_tpm": "mean RNA TPM", "quartile_rna_tpm": "quartile",
            "consensus_class": "consensus TE class"}, keep(["seq_id", "rna_mean_tpm", "quartile_rna_tpm", "consensus_class"]))]),
    P("FigEV1H_bottom", "TPM-ratio TE against mean RNA abundance", [
        T(EV + "FigEV1/output/FigEV1D-H_statistics.tsv", "Correlation", {
            "panel": "panels the row belongs to", "spearman_vs_rna_tpm": "Spearman's ρ of TPM-ratio TE and mean RNA TPM"},
          rows("panel", ["H"], ["panel", "spearman_vs_rna_tpm"])),
        T(EV + "FigEV1/output/FigEV1D-H_source_data.tsv", "Per-transcript values", {
            "seq_id": "RefSeq transcript", "consensus_class": "consensus TE class (colour)", "rna_mean_tpm": "mean RNA TPM (x)",
            "te_ratio": "TPM-ratio TE", "te_ratio_log2": "log2 TPM-ratio TE (y)"},
          keep(["seq_id", "consensus_class", "rna_mean_tpm", "te_ratio", "te_ratio_log2"]))]),
    P("FigEV1I", "AUC of classifiers trained with and without consensus filtering", [
        T(EV + "FigEV1/output/FigEV1I_auc_summary.tsv", "Summary and test", {
            "filter_mean": "mean AUC, Filtered (bar)", "filter_sd": "SD, Filtered (error bar)", "nofilter_mean": "mean AUC, Unfiltered (bar)",
            "nofilter_sd": "SD, Unfiltered (error bar)", "folds_filter_better": "folds where Filtered is higher",
            "n_transcripts": "held-out consensus mRNAs", "wilcoxon_p": "two-sided Wilcoxon signed-rank P"}),
        T(EV + "FigEV1/output/FigEV1I_auc_per_fold.tsv", "AUC per fold (points)", {
            "model": "filter = Filtered, nofilter = Unfiltered", "fold": "cross-validation fold", "AUC": "AUC"})]),
    P("FigEV2A", "pTE against estimated mRNA half-life (bins of 20 mRNAs)", [
        T(EV + "FigEV2/output/FigEV2AB_correlations.tsv", "Correlation of the bin means", BIN_CORR, rows("panel", ["EV2A"])),
        T(EV + "FigEV2/output/FigEV2A_binned_half_life_vs_pTE.tsv", "Bins", BINS)]),
    P("FigEV2B", "pTE against 5′UTR, CDS and 3′UTR length (bins of 20 mRNAs)", [
        T(EV + "FigEV2/output/FigEV2AB_correlations.tsv", "Correlation of the bin means (log10 length)", BIN_CORR, rows("panel", ["EV2B"])),
        T(EV + "FigEV2/output/FigEV2B_binned_length_vs_pTE.tsv", "Bins", BINS)]),
    P("FigEV2C", "pTE against GC ratio (bins of 20 mRNAs)", [
        T(EV + "FigEV2/output/FigEV2CD_correlations.tsv", "Correlation of the bin means", BIN_CORR_LINEAR, rows("panel", ["EV2C"])),
        T(EV + "FigEV2/output/FigEV2C_binned_GC_ratio_vs_pTE.tsv", "Bins", BINS)]),
    P("FigEV2D", "pTE against CAI (bins of 20 mRNAs)", [
        T(EV + "FigEV2/output/FigEV2CD_correlations.tsv", "Correlation of the bin means", BIN_CORR_LINEAR, rows("panel", ["EV2D"])),
        T(EV + "FigEV2/output/FigEV2D_binned_CAI_vs_pTE.tsv", "Bins", BINS)]),
    P("FigEV3A", "ROC curves of TEFL-mRNA and RiboNN on the held-out LCL test set", [
        T(EV + "FigEV3/output/FigEV3A_auc.tsv", "AUC", {"model": "model", "AUC": "AUC", "n_TE_high": "TE-high mRNAs", "n_other": "other mRNAs"}),
        T(EV + "FigEV3/output/FigEV3A_roc_curves.tsv", "ROC curve points", {"model": "model", "FPR": "false positive rate (x)",
                                                                       "TPR": "true positive rate (y)"})]),
    P("FigEV3B", "Pearson r of TEFL-mRNA (Reg), multitask RiboNN and RiboNN transfer per fold", [
        T(EV + "FigEV3/output/FigEV3B_pearson_summary.tsv", "Mean and SD over folds", {
            "cell_line": "cell line", "model": "model", "mean": "mean r (bar)", "sd": "SD (error bar)"},
          keep(["cell_line", "model", "mean", "sd"])),
        T(EV + "FigEV3/output/FigEV3B_pearson_by_fold.tsv", "Pearson r per fold (points and lines)", {
            "cell_line": "cell line", "model": "model", "fold": "RiboNN cross-validation fold", "n": "held-out transcripts",
            "Pearson_R": "Pearson r of predicted and measured TE"})]),
    P("FigEV3C", "TEFL-mRNA (Reg) against RiboNN transfer predictions", [
        T(EV + "FigEV3/output/FigEV3C_correlations.tsv", "Correlations and fit", {
            "cell_line": "cell line", "n": "transcripts", "pearson_r": "Pearson r", "pearson_p": "P", "pearson_lo": "95% CI, lower",
            "pearson_hi": "95% CI, upper", "spearman_rho": "Spearman's ρ", "spearman_p": "P", "ols_slope": "least-squares slope",
            "ols_intercept": "least-squares intercept"}),
        T(EV + "FigEV3/output/FigEV3C_predictions.tsv", "Out-of-fold predictions", {
            "cell_line": "cell line", "tx_id": "Ensembl transcript", "fold": "RiboNN fold", "true_y": "measured TE",
            "pred_TEFL_mRNA_Reg": "TEFL-mRNA (Reg) prediction (x)", "pred_RiboNN_transfer": "RiboNN transfer prediction (y)",
            "density": "local point density (colour)"})]),
    P("FigEV3D", "ΔpTE against three alternative variant-effect scores (GWAS 5′UTR variants, nine cell lines)", [
        T(EV + "FigEV3/output/FigEV3D-F_pairwise_stats.tsv", "Correlations between scores", {
            "pair": "score pair", "pearson": "Pearson r", "spearman": "Spearman's ρ", "sign_all": "sign agreement over all pairs",
            "sign_strong_min": "sign agreement among large effects, lowest cell line", "sign_strong_max": "highest cell line"}),
        T(EV + "FigEV3/output/FigEV3D-F_variant_scores.tsv.gz", "Scores per variant and cell line (points)", {
            "transcript_refid_var": COLS["transcript_refid_var"], "cell_line": "cell line", "dpte": "TEFL (Class) ΔpTE",
            "ordinal": "ordinal score change", "our_reg": "TEFL (Reg) ΔTE", "ribonn": "RiboNN ΔTE"})]),
    P("FigEV3E", "Pairwise Spearman correlation of the four variant-effect scores", [
        T(EV + "FigEV3/output/FigEV3E_source_data.tsv", None, {"a": "score", "b": "score", "rho": "Spearman's ρ (tile)"})]),
    P("FigEV3F", "Directional concordance of ΔpTE with the alternative scores among large-effect variants", [
        T(EV + "FigEV3/output/FigEV3F_summary.tsv", "Mean over cell lines (bars) and range (error bars)", {
            "alt": "alternative score", "mean_sc": "mean sign concordance", "lo": "lowest cell line", "hi": "highest cell line"}),
        T(EV + "FigEV3/output/FigEV3F_source_data.tsv", "Sign concordance per cell line", {
            "alt": "alternative score", "cell_line": "cell line", "sign_concordance": "fraction with the same sign as ΔpTE"})]),
    P("FigEV4A", "ΔpTE of dbSNP common variants (allele frequency 0.1 to 0.9) by annotation class", [
        T(EV + "FigEV4/output/FigEV4A_source_data.tsv", None, {
            "class": "annotation class", "bin": "ΔpTE cutoff (cumulative tail)", "n_pairs": "SNV-transcript pairs in the class",
            "n_beyond": "pairs beyond the cutoff", "percent": "percentage (bar)"})]),
    P("FigEV4B", "ΔpTE of gnomAD v4.0 SNVs within 1 kb of the start and stop codons", [
        T(EV + "FigEV4/output/FigEV4B_summary.tsv", None, {
            "region": "5′UTR and CDS around the start codon, CDS and 3′UTR around the stop codon", "alt": "ALT base (colour)",
            "n_SNVs": "SNV-transcript pairs (points)", "n_outside_plot_range": "pairs outside the plotted ΔpTE range (−0.6 to 0.6)",
            "median_delta_pTE": "median ΔpTE", "frac_delta_pTE_gt_0.1": "fraction with ΔpTE > 0.1",
            "frac_delta_pTE_lt_minus_0.1": "fraction with ΔpTE < −0.1"})],
      note="The panel plots 33,413,533 SNV-transcript pairs, more than an Excel sheet holds; the per-pair values are in "
           "ExpandedView/FigEV4/input/gnomAD_v4_SNVs_dpTE_start_stop_1kb.tsv.gz of the data archive."),
    P("FigEV4C", "ALT-allele composition of gnomAD v4.0 5′UTR SNVs beyond ΔpTE cutoffs", [
        T(EV + "FigEV4/output/FigEV4C_source_data.tsv", None, {
            "Class": "TE_up or TE_down", "Cutoff": "ΔpTE cutoff (cumulative)", "Base": "ALT base (T = U)",
            "Count": "SNV-transcript pairs", "Frac": "fraction (bar segment)"})]),
    P("FigEV4G", "Unpaired probability of the variant base in the 5′UTR (ViennaRNA)", [
        T(EV + "FigEV4/output/FigEV4G_source_data.csv", None, {
            "gene": "gene symbol", "hgvs": "HGVS c. notation (axis label)", "hgvs_full": "HGVS with the RefSeq accession",
            "refseq": "RefSeq transcript with version", "utr5_position": "position in the 5′UTR (1-based)",
            "ref_allele": "reference base", "alt_allele": "variant base", "utr5_length": "5′UTR length (nt)",
            "pUnpaired_variant_ref": "P(unpaired), reference allele (open circle)",
            "pUnpaired_variant_alt": "P(unpaired), variant allele (filled circle)",
            "luc_direction": "direction of the measured reporter effect (colour)", "delta_pUnpaired": "variant minus reference"})]),
]
for x, title in [("H", "ΔpTE of dbSNP common 5′UTR variants against local GC content (±25 nt)"),
                 ("I", "ΔpTE of dbSNP common 5′UTR variants against distance to the start codon")]:
    EV_APPENDIX.append(P(f"FigEV4{x}", title, [
        T(EV + "FigEV4/output/FigEV4HI_statistics.csv", "Correlation with |ΔpTE|", {
            "covariate": "covariate", "spearman_rho_vs_abs_delta_pTE": "Spearman's ρ with |ΔpTE|", "n": "variant-transcript pairs"},
          rows("panel", [x], ["covariate", "spearman_rho_vs_abs_delta_pTE", "n"])),
        T(EV + "FigEV4/output/FigEV4HI_binned_summary.tsv", "Equal-count bins (orange line and band)", {
            "x": "bin median of the covariate (x)", "y": "median signed ΔpTE", "lo": "5th percentile", "hi": "95th percentile",
            "nb": "pairs in the bin"}, rows("panel", [x], ["x", "y", "lo", "hi", "nb"])),
        T(EV + "FigEV4/output/FigEV4HI_source_data.tsv", "Variant-transcript pairs (grey points)", {
            "transcript_refid_var": COLS["transcript_refid_var"], "x": "covariate value (x)", "delta_pTE": "ΔpTE (y)",
            "abs_delta_pTE": "|ΔpTE|"}, rows("panel", [x], ["transcript_refid_var", "x", "delta_pTE", "abs_delta_pTE"]))]))
EV_APPENDIX += [
    P("FigEV5A", "ΔpTE of missense SNVs (|ΔpTE| > 0.05) by amino acid introduced", [
        T(EV + "FigEV5/output/FigEV5A_summary.tsv", "Median per amino acid (x-axis order)", {
            "aa_var": "amino acid introduced", "n": "SNV-transcript pairs", "median_delta_score_high": "median ΔpTE"}),
        T(EV + "FigEV5/output/FigEV5A_source_data.tsv", "SNV-transcript pairs (violins)", {
            "transcript_refid_var": COLS["transcript_refid_var"], "variant_id": COLS["variant_id"], "aa_novar": "reference amino acid",
            "aa_var": "amino acid introduced (x)", "delta_score_high": "ΔpTE (y)"})]),
    P("FigEV5B", "TE-increasing and TE-decreasing proline-creating and all missense SNVs", [
        T(EV + "FigEV5/output/FigEV5B_source_data.tsv", "Counts", {
            "Group": "All_SNVs or Pro_SNVs (amino acid introduced is proline)", "Outcome": "TE_increasing or TE_decreasing",
            "Count": "SNV-transcript pairs", "Proportion": "fraction of the group (bar)", "Total": "pairs in the group"}),
        T(EV + "FigEV5/output/FigEV5B_fisher_test.tsv", "Fisher's exact test", {
            "fold_enrichment": "fold enrichment of TE-decreasing", "odds_ratio": "odds ratio", "conf_int_low": "95% CI, lower",
            "conf_int_high": "95% CI, upper", "p_value": "P value"})]),
    P("FigEV5C", "ΔpTE of TE-decreasing proline-creating missense SNVs by poly-proline length", [
        T(EV + "FigEV5/output/FigEV5C_summary.tsv", "Summary per poly-proline length", {
            "Mutated_PolyP_Length": "longest proline run in the mutant 5-residue window", "n": "SNV-transcript pairs (n above)",
            "mean": "mean ΔpTE", "median": "median ΔpTE", "sd": "standard deviation", "q25": "25th percentile",
            "q75": "75th percentile", "se": "standard error"}),
        T(EV + "FigEV5/output/FigEV5C_tests.tsv", "Tests", {
            "test": "test", "groups": "poly-proline lengths compared", "statistic_name": "statistic", "statistic": "value",
            "p_value": "P value"}),
        T(EV + "FigEV5/output/FigEV5C_source_data.tsv", "SNV-transcript pairs", {
            "transcript_refid_var": COLS["transcript_refid_var"], "variant_id": COLS["variant_id"], "gene": "gene symbol",
            "HGVSc_protein": "HGVS protein change", "Original_Flanking_5mer": "reference 5-residue window",
            "Mutated_Flanking_5mer": "mutant 5-residue window", "Mutated_PolyP_Length": "longest proline run (x)",
            "delta_score_high": "ΔpTE (y)"})]),
    P("FigEV5D", "Mean ΔpTE per nucleotide substitution of gnomAD v4.0 coding SNVs", [
        T(EV + "FigEV5/output/FigEV5D_source_data.tsv", None, {
            "class": "Synonymous or Missense", "substitution": "REF>ALT (U for T)", "n": "SNV-transcript pairs",
            "unadjusted": "mean ΔpTE (filled circle)", "adjusted": "covariate-adjusted marginal mean (open circle)",
            "allele_introduced": "ALT base"})]),
    P("FigEV5E", "Mean ΔpTE per nucleotide substitution across transcript GC deciles", [
        T(EV + "FigEV5/output/FigEV5E_source_data.tsv", None, {
            "class": "Synonymous or Missense", "gc_decile": "decile of transcript-wide GC content (x)",
            "substitution": "REF>ALT (U for T)", "mean_dpTE": "mean ΔpTE (y)", "n": "SNV-transcript pairs"})]),
    P("FigEV5F", "Variance in ΔpTE explained by substitution type and GC content", [
        T(EV + "FigEV5/output/FigEV5F_source_data.tsv", None, {
            "class": "Synonymous or Missense", "metric": "R2_subst_only, R2_gc_only, R2_full (bars, plotted as %), n and GC coefficients",
            "value": "value"})]),
    P("FigEV5G_left", "Mean ΔpTE of codon pairs against change in codon stability coefficient (HeLa model)", [
        T(EV + "FigEV5/input/ExAC_HeLa_codon_pairs_dpTE_CSC.tsv", None, {
            "cell": "cell line", "codon_ref": "reference codon", "codon_alt": "alternative codon", "amino_acid": "amino acid",
            "mean_delta_pTE": "mean ΔpTE", "sem_delta_pTE": "standard error", "mean_delta_pTE_in_SD_units": "mean ΔpTE / SD (y)",
            "n_transcripts": "transcripts", "cell_line_SD_of_delta_pTE": "SD of ΔpTE in the cell line",
            "delta_GC_of_codon": "change in codon G+C", "GC_neutral": "G+C unchanged (filled point)",
            "delta_CSC_HeLa": "ΔCSC, alternative − reference (x)"})]),
    P("FigEV5G_right", "Mean ΔpTE of amino-acid pairs against change in amino-acid stability coefficient (HeLa model)", [
        T(EV + "FigEV5/input/ExAC_HeLa_aminoacid_pairs_dpTE_AASC.tsv", None, {
            "cell": "cell line", "aa_ref": "reference amino acid", "aa_alt": "alternative amino acid", "mean_delta_pTE": "mean ΔpTE",
            "sem_delta_pTE": "standard error", "mean_delta_pTE_in_SD_units": "mean ΔpTE / SD (y)", "n_transcripts": "transcripts",
            "cell_line_SD_of_delta_pTE": "SD of ΔpTE in the cell line", "delta_AASC_HeLa": "ΔAASC, alternative − reference (x)",
            "delta_GC_of_aminoacid": "change in amino-acid GC content", "GC_matched": "|ΔGC| ≤ 0.05 (filled point)"})]),
    P("FigEV5H", "Cross-cell-line variation relative to model uncertainty by variant class", [
        T(EV + "FigEV5/output/FigEV5H_H_by_region.tsv", "Summary per variant class", {
            "region": "variant class", "n": "variants", "median_H": "median H", "pct_H_gt1": "percentage with H > 1",
            "pct_q_lt_05": "percentage with BH-adjusted F-test q < 0.05 (printed)"}),
        T(EV + "FigEV5/output/FigEV5HI_per_variant_H_q.tsv.gz", "Per-variant H (violins)", {
            "region": "variant class", "variant": "SNV-transcript pair", "H": "between-cell-line / within-cell-line fold mean square",
            "q_F": "BH-adjusted F-test q"}, keep(["region", "variant", "H", "q_F"]))]),
    P("FigEV5I", "Variants with significant cross-cell-line variation by directional concordance", [
        T(EV + "FigEV5/output/FigEV5I_Fpass_by_cells_positive.tsv", None, {
            "region": "variant class", "n_cell_pos": "cell lines with a positive mean ΔpTE (x)", "n": "variants",
            "median_H": "median H", "frac_pass": "fraction with q < 0.05 (y)"})]),
    P("FigEV5J", "Cross-cell-line variation of effect-size-matched 5′UTR and coding variants", [
        T(EV + "FigEV5/output/FigEV5J_H_summary.tsv", "Summary of the matched groups", {
            "quantity": "quantity", "5UTR": "5′UTR variants", "Coding": "coding variants", "ratio_5UTR_over_coding": "5′UTR / coding"}),
        T(EV + "FigEV5/output/FigEV5H-K_matching_balance.tsv", "Balance of mean |ΔpTE| after matching", {
            "grp": "group", "q25": "25th percentile", "median": "median", "q75": "75th percentile", "n": "variants",
            "ks_p": "Kolmogorov-Smirnov P", "smd": "standardized mean difference"}),
        T(EV + "FigEV5/output/FigEV5JK_matched_pairs.tsv.gz", "Matched pairs (violins)", {
            "pair": "matched pair", "grp": "5′UTR or Coding", "H": "H", "mean_abs": "mean |ΔpTE| over cell lines",
            "ms_cell": "between-cell-line mean square", "ms_err": "within-cell-line mean square",
            "switch8": "confident direction switch, ≥ 8 of 10 folds", "switch9": "≥ 9 of 10 folds"})]),
    P("FigEV5K", "Confident direction switches among effect-size-matched variants", [
        T(EV + "FigEV5/output/FigEV5K_reversal_pct_with_CI.tsv", None, {
            "threshold": "switch8 (≥ 8 of 10 folds) or switch9 (≥ 9 of 10)", "grp": "5′UTR or Coding",
            "pct": "percentage of matched variants (bar)", "lo": "95% bootstrap CI, lower", "hi": "upper"})]),
    P("AppendixFigS2", "Transcript features against measured TE in nine cell lines (bins of 10 transcripts)", [
        T(AP + "AppendixFigS2/output/AppendixFigS2_correlations.tsv", "Correlations", {
            "cell_line": "cell line", "feature": "transcript feature", "n_bins": "bins", "pearson_r": "Pearson R", "p_value": "P value"}),
        T(AP + "AppendixFigS2/output/AppendixFigS2_source_data.tsv", "Bins", {
            "cell_line": "cell line", "feature": "transcript feature", "bin": "bin number",
            "mean_value_bins": "mean feature value (x)", "mean_log10_TE_bins": "mean log10 measured TE (y)"})]),
]

WORKBOOKS = [("SourceData_Figure_main.xlsx", "Source Data for Figures 1-6", MAIN),
             ("SourceData_Figures_EV_Appendix.xlsx", "Source Data for Figures EV1–EV5 and Appendix Figure S2", EV_APPENDIX)]


# ---------------------------------------------------------------- reading and writing
def read_table(path):
    sep = "," if ".csv" in path.name else "\t"
    return pd.read_csv(path, sep=sep, low_memory=False, float_precision="round_trip")


def load(t):
    d = read_table(ROOT / t.file)
    if t.transform:
        d = t.transform(d)
    d = d.reset_index(drop=True)
    named = cols_of(t)
    if "*" not in t.cols:
        assert list(d.columns) == named or set(d.columns) == set(named), f"{t.file}: columns {list(d.columns)} vs {named}"
        d = d[named]
    return d


# every cell is set in Arial; panel titles in bold italics on a grey band, column headers and table labels in bold
PLAIN = Font(name="Arial")
BOLD = Font(name="Arial", bold=True)
TITLE = Font(name="Arial", bold=True, italic=True)
GREY = PatternFill(fill_type="solid", start_color="FFD9D9D9", end_color="FFD9D9D9")


def cell(ws, v, font=PLAIN):
    c = WriteOnlyCell(ws, value=v)
    c.font = font
    return c


def value(ws, v):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return None
    if hasattr(v, "item"):                       # numpy scalar
        v = v.item()
    c = cell(ws, v)
    if isinstance(v, str) and v.startswith("="):
        c.data_type = "s"
    return c


def bold(ws, v):
    return cell(ws, v, BOLD)


def title_row(ws, text, n_cols):
    """Panel title in the first cell, on a grey band n_cols cells wide."""
    cells = [cell(ws, text, TITLE)] + [WriteOnlyCell(ws) for _ in range(n_cols - 1)]
    for c in cells:
        c.fill = GREY
    return cells


def build(name, title, panels):
    tables = [(p, [(t, load(t)) for t in p.tables]) for p in panels]
    wb = Workbook(write_only=True)
    readme = wb.create_sheet("README")
    readme.column_dimensions["A"].width = 32
    readme.column_dimensions["B"].width = 90
    readme.append([bold(readme, title)])
    for p, tabs in tables:
        readme.append([])
        readme.append(title_row(readme, f"{p.sheet}. {p.title}", 2))
        for t, d in tabs:
            if len(tabs) > 1:
                readme.append([bold(readme, t.label)])
            readme.append([bold(readme, "Column"), bold(readme, "Description")])
            for c in d.columns:
                if c not in t.cols and not t.readme_wildcard:
                    continue
                desc = t.cols.get(c, t.cols.get("*", COLS.get(c, "")))
                assert desc, f"{p.sheet}: no description for column {c}"
                readme.append([cell(readme, str(c)), cell(readme, desc)])
    for p, tabs in tables:
        n_rows = 2 + bool(p.note) + sum(len(d) + 2 + (len(tabs) > 1) for _, d in tabs)
        assert n_rows <= EXCEL_MAX_ROWS, f"{p.sheet}: {n_rows} rows exceed the Excel limit"
        ws = wb.create_sheet(p.sheet)
        width = max(len(d.columns) for _, d in tabs)
        for j in range(width):
            ws.column_dimensions[get_column_letter(j + 1)].width = 16
        text = f"{p.sheet}. {p.title}"
        # the band covers the table, or the title text if wider (columns are 16 characters wide)
        ws.append(title_row(ws, text, max(width, math.ceil(len(text) * 1.2 / 16))))
        if p.note:
            ws.append([cell(ws, p.note)])
        for t, d in tabs:
            ws.append([])
            if len(tabs) > 1:
                ws.append([bold(ws, t.label)])
            ws.append([bold(ws, str(c)) for c in d.columns])
            for row in d.itertuples(index=False):
                ws.append([value(ws, v) for v in row])
    OUT.mkdir(exist_ok=True)
    wb.save(OUT / name)
    print(f"{name}: {len(panels)} panel sheets, {sum(len(d) for _, tabs in tables for _, d in tabs):,} data rows")


if __name__ == "__main__":
    for name, title, panels in WORKBOOKS:
        build(name, title, panels)
