# Figure S9

Commands run from the `FigS9/` directory. Panels S9A–S9C share one analysis script and one
plot script. The analysis script writes the source-data tables into `input/`; the plot script
reads them and writes the panel PDFs into `output/`. The committed source data lets the panels
be reproduced without the large prediction tables, so the plot script can be run on its own.

pTE is the model's predicted probability of high translation efficiency. ΔpTE = pTE(ALT) −
pTE(REF) for the same transcript. The counting unit is a variant–transcript pair, one SNV
scored on each RefSeq isoform it falls in. Substitutions are directed (reference > alternate)
at the variant position and written with U for thymine, giving twelve classes. Effects are in
units of 10⁻³ ΔpTE.

## Fig. S9A

Analysis: Mean ΔpTE for each of the twelve directed nucleotide substitutions in synonymous and
missense variants, before and after adjustment for local GC content, transcript-wide GC content,
codon position, relative CDS position and CDS length. Adjusted values are estimated marginal
means from `ΔpTE ~ subst + local_gc + tx_gc + codon_pos + rel_pos + logL`.

Input: `input/FigS9A_source_data.tsv`
Analysis script: `analysis/FigS9ABC_analysis.R`
Plot script: `plot/FigS9ABC_plot.R`
Output: `output/FigS9A.pdf`
Run:
```
Rscript analysis/FigS9ABC_analysis.R
Rscript plot/FigS9ABC_plot.R
```

## Fig. S9B

Analysis: Unadjusted mean ΔpTE for each substitution within deciles of transcript-wide GC
content, spanning GC 0.31 to 0.77. No model is fitted; the decile stratification is itself the
control.

Input: `input/FigS9B_source_data.tsv`
Analysis script: `analysis/FigS9ABC_analysis.R`
Plot script: `plot/FigS9ABC_plot.R`
Output: `output/FigS9B.pdf`
Run:
```
Rscript analysis/FigS9ABC_analysis.R
Rscript plot/FigS9ABC_plot.R
```

## Fig. S9C

Analysis: Variance of ΔpTE explained by substitution type alone, by local and transcript-wide
GC content alone, and by the full covariate set, from three nested ordinary-least-squares fits.

Input: `input/FigS9C_source_data.tsv`
Analysis script: `analysis/FigS9ABC_analysis.R`
Plot script: `plot/FigS9ABC_plot.R`
Output: `output/FigS9C.pdf`
Run:
```
Rscript analysis/FigS9ABC_analysis.R
Rscript plot/FigS9ABC_plot.R
```

## Fig. S9D

Analysis: Mean ΔpTE per synonymous codon substitution pair in HeLa, against the change in codon
stability coefficient (ΔCSC, alternate − reference; Forrest et al. 2020). Filled points are the
34 GC-neutral pairs, in which the codon's GC count is unchanged and no GC signal is available to
be read; open points are the remaining pairs. Spearman's ρ is reported over all pairs and over
the GC-neutral subset.

Input: `input/FigS9D_source_data.tsv`
Analysis script: upstream, see "External inputs"
Plot script: `plot/FigS9DE_plot.R`
Output: `output/FigS9D.pdf`
Run:
```
Rscript plot/FigS9DE_plot.R
```

## Fig. S9E

Analysis: Mean ΔpTE per amino-acid substitution in HeLa, against the change in amino-acid
stability coefficient (ΔAASC, alternate − reference; Forrest et al. 2020). Filled points are the
30 GC-matched substitutions, for which the amino-acid GC content differs by less than 0.05.

Input: `input/FigS9E_source_data.tsv`
Analysis script: upstream, see "External inputs"
Plot script: `plot/FigS9DE_plot.R`
Output: `output/FigS9E.pdf`
Run:
```
Rscript plot/FigS9DE_plot.R
```

## Source data columns

`FigS9A_source_data.tsv` — one row per variant class × substitution (24 rows). `class`
(`Synonymous`, `Missense`), `substitution`, `n` variant–transcript pairs, `unadjusted` raw mean
ΔpTE, `adjusted` marginal mean from the full model, `allele_introduced` the alternate base.

`FigS9B_source_data.tsv` — one row per class × GC decile × substitution (240 rows).
`gc_decile` 1–10 cut at deciles of the per-pair transcript-wide GC distribution, `mean_dpTE`
the unadjusted mean within that cell, `n` its pairs.

`FigS9C_source_data.tsv` — one row per class × metric (12 rows). `R2_subst_only`, `R2_gc_only`
and `R2_full` are plotted as percentages; `beta_local_gc` and `beta_tx_gc` are the full model's
GC coefficients in ΔpTE per unit GC fraction; `n_variants` the pairs entering each fit
(16,512,884 synonymous, 18,656,909 missense). Substitution type explains 4.79% and 4.14% of the
variance in ΔpTE, GC content alone 0.031% and 0.079%, and the full covariate set 5.00% and 4.18%.

`FigS9D_source_data.tsv` — one row per synonymous codon substitution pair in HeLa (134 rows).
`codon_ref`, `codon_alt` and `amino_acid` identify the pair; `mean_delta_pTE` and `sem_delta_pTE`
are the transcript-clustered mean effect and its standard error; `mean_delta_pTE_in_SD_units`
(plotted on y) divides that mean by `cell_line_SD_of_delta_pTE`; `n_transcripts` the transcripts
contributing; `delta_GC_of_codon` the change in the codon's GC count and `GC_neutral` whether it
is zero (34 pairs); `delta_CSC_HeLa` the change in codon stability coefficient (plotted on x).
Because the input is SNVs, the 134 ordered pairs are 67 reciprocal duplicates, so the effective
number of independent observations is 67 and 17 for the GC-neutral subset.

`FigS9E_source_data.tsv` — one row per amino-acid substitution in HeLa (150 rows). `aa_ref` and
`aa_alt` identify the substitution; the effect columns are as above; `delta_AASC_HeLa` is the
change in amino-acid stability coefficient (plotted on x), `delta_GC_of_aminoacid` the change in
amino-acid GC content and `GC_matched` whether it is below 0.05 (30 substitutions).

## External inputs

`analysis/FigS9ABC_analysis.R` reads four tables that are too large to include here, resolved
relative to the repository root (`Translation_DL/`). If this directory is moved out of
`Data_and_code/`, set `TEFL_ROOT` to the repository root.

- `gnomad_var/data/Pred_All.gnomad.exomes.v4.0.sites.PASS_pc_SNV_NM.{synonymous,missense}_variant.filter_cor0.7_l0.25_h1.TPE_opt.delta.filter.sel_cols.for_mod.tsv`
  — about 5 GB together; columns `transcript_refid_var`, `delta_score_high`, `codon_pos`,
  `codon_num`, `codon_novar`, `codon_var`.
- `multi_cellline/ncbi_isoform/RiboNN_pred/pred_NCBI_RefSeq_GRCh38.mRNA_pc.FL_regions-correct.ribonn_output.tsv`
  — transcript sequences and region sizes.
- `ncbi_isoform/pred_ncbi_mRNA.filter_cor0.7_l0.25_h1.tpe_opt.tsv` — sequences for the 1,482
  transcripts the variant tables reference but the table above does not contain. The two are
  combined as a union; without it, 5.4% of synonymous and 5.7% of missense pairs are dropped,
  the loss falling preferentially on the longest coding sequences.

The analysis step needs roughly 16 GB of memory and about 15 minutes.

Panels S9D and S9E have no analysis script here. Their source-data tables are the plotting-ready
output of the ExAC codon- and amino-acid-metric analysis, which scores ExAC variants with the
HEK293T and HeLa models and joins them to the metrics of Forrest et al. 2020 (Table S2,
`gnomad_var/coding_region_bg_control/data/forrest2020_S2_codon_aa_metrics.tsv`). That analysis
also produces the ΔCAI and ΔtAI panels belonging to another figure; the tables shipped here are
restricted to the HeLa rows and the CSC and AASC columns those two panels use. Add the upstream
script reference before submission — it is not reproduced in this directory.

## Checks

Variants are restricted to pairs with well-formed codon fields whose codon number falls inside
the annotated CDS. Every pair in both tables satisfies the CDS-range condition; the only
exclusions are 18 synonymous and 17 missense pairs with a missing or malformed codon field.
Codon identity was validated against the transcript sequences for every pair: the codon implied
by `codon_num` and the mature-mRNA sequence matches `codon_novar` throughout, and substituting
the alternate base reproduces `codon_var`. Local GC is computed on the reference sequence, so it
is identical for both alleles of a pair and cannot absorb the effect being estimated.

## Dependencies

R 4.5.3; data.table 1.17.8, ggplot2 4.0.3. Panels are set in Arial through `cairo_pdf` and
require a UTF-8 locale.
