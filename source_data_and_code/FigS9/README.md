# Supplementary Figure 9. Coding-variant effects on translation efficiency in relation to nucleotide composition and amino-acid properties

Run the commands from the `FigS9/` folder. The source data for every panel are already in `input/FigS9{A..E}_source_data.tsv`, so the plot scripts run on their own. You need R with data.table and ggplot2, Arial through `cairo_pdf`, and a UTF-8 locale.

ΔpTE = pTE(ALT) − pTE(REF) on the same transcript, counted over SNV–transcript pairs. Substitutions are directed (reference > alternate), written with U for T, which gives twelve classes. Effects are in units of 10⁻³ ΔpTE.

## Panels

### Supplementary Fig. 9A
Nucleotide substitution-specific effects on predicted TE. Mean ΔpTE for the 12 nucleotide substitutions among synonymous (n = 16,512,884) and missense (n = 18,656,909) gnomAD v4.0 SNVs. Filled and open circles indicate unadjusted and covariate-adjusted effects, respectively.

The adjusted values are estimated marginal means from `ΔpTE ~ subst + local_gc + tx_gc + codon_pos + rel_pos + logL` (local and transcript-wide GC, codon position, relative CDS position and log CDS length).

### Supplementary Fig. 9B
Substitution-specific effects across transcript GC content. Mean ΔpTE for each nucleotide substitution across deciles of transcript-wide GC content.

No model is fitted here; the deciles span GC 0.31 to 0.77.

### Supplementary Fig. 9C
Variance in predicted variant effects explained by substitution identity and GC content. Proportion of variance in ΔpTE (R²) explained by substitution type, GC content, or the full covariate model.

The three values come from nested least-squares fits. Substitution type explains 4.79% (synonymous) and 4.14% (missense) of the variance, GC content alone 0.031% and 0.079%, and the full model 5.00% and 4.18%.

Panels A to C: `analysis/FigS9ABC_analysis.R` writes `input/FigS9{A,B,C}_source_data.tsv`, and `plot/FigS9ABC_plot.R` draws `output/FigS9A.pdf` to `output/FigS9C.pdf`.

### Supplementary Fig. 9D-E
Correlation between predicted variant effects and codon stability coefficients (D) and amino-acid stability coefficients (E). Scatter plot of ΔpTE versus changes in codon stability coefficients (ΔCSC) and amino-acid stability coefficients (ΔAASC). The stability coefficients were obtained from Forrest et al. (2020, Supplementary Table 2), based on HeLa mRNA half-life measurements. Each data point represents a codon pair (D) or amino-acid pair (E). The y-axis shows the mean ΔpTE, rescaled by the standard deviation (SD) across cell lines. The x-axis represents the change in stability coefficient between alternative and reference alleles (alternative - reference). Filled points indicate GC-neutral subsets: codon pairs with unchanged G+C content (D) or amino-acid pairs with similar GC content (|ΔGC| ≤ 0.05; E). Lines indicate linear regression fits across all pairs; Spearman’s ρ is calculated for all pairs and the GC-neutral subsets.

The points are ExAC variants scored with the HeLa model. `plot/FigS9DE_plot.R` draws `output/FigS9D.pdf` and `output/FigS9E.pdf` from `input/FigS9D_source_data.tsv` and `input/FigS9E_source_data.tsv`. These two tables come from a separate analysis that joins the ExAC predictions to Forrest et al. (2020) Table S2; that script is not in this folder. The 134 codon pairs in 9D are 67 reciprocal pairs, so 67 (17 GC-neutral) are independent.

## Rebuilding the source data for A to C

`analysis/FigS9ABC_analysis.R` reads four large tables from the `Translation_DL/` project folder (set `TEFL_ROOT` if this folder is moved): the gnomAD synonymous and missense prediction tables in `gnomad_var/data/` (about 5 GB) and the transcript sequences in `multi_cellline/ncbi_isoform/RiboNN_pred/` and `ncbi_isoform/`. It needs about 16 GB of memory and 15 minutes. Pairs with a missing or malformed codon field are dropped (18 synonymous, 17 missense).

## Run

```bash
Rscript analysis/FigS9ABC_analysis.R   # optional: rebuilds input/FigS9{A,B,C}_source_data.tsv
Rscript plot/FigS9ABC_plot.R
Rscript plot/FigS9DE_plot.R
```
