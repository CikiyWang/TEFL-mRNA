# Figure EV4. The distribution of TE effects by SNVs across mRNA regions, complete 5′UTR sequences and underlying mechanisms of validated TE-altering 5′UTR variants, and population-genetic controls for translation-altering variant predictions

This folder has the code for panels A–C and G–I. Panels D–F (the 5′UTR sequences and structures, the HLA-G uORF and the H3C4 hairpin) have no script.

Run the commands from `ExpandedView/FigEV4/`. Analysis scripts write tables to `output/`; plot scripts draw the panels. Requirements: R 4.4.2 with the packages in `../../shared/environment/R_packages.tsv`, and Python 3.12 with the packages in `../../shared/environment/python_requirements.txt` for panel G; the ViennaRNA Python package is needed only for the panel G analysis step. The panel H–I scripts source `analysis/common.R`.

ΔpTE = pTE(ALT) − pTE(REF) on the same RefSeq transcript; counts are SNV–transcript pairs. In panels H–I the variants are dbSNP common 5′UTR variants (allele frequency 0.1 to 0.9) after removing ClinVar and GWAS Catalog entries: 27,156 variant–transcript pairs in both panels; ΔpTE is the variant minus the reference prediction.

## Panels

| Panel | Content | Scripts | Outputs (`output/`) |
|---|---|---|---|
| EV4A | ΔpTE of dbSNP common variants (allele frequency 0.1–0.9) by annotation class | `analysis/FigEV4A_analysis.R`, `plot/FigEV4A_plot.R` | `FigEV4A.pdf`, `FigEV4A_source_data.tsv` |
| EV4B | ΔpTE of gnomAD v4.0 SNVs within 1 kb of the start and stop codons | `analysis/FigEV4B_analysis.R`, `plot/FigEV4B_plot.R` | `FigEV4B.pdf`, `FigEV4B_summary.tsv` |
| EV4C | ALT-allele composition of gnomAD 5′UTR SNVs beyond ΔpTE cutoffs | `analysis/FigEV4C_analysis.R`, `plot/FigEV4C_plot.R` | `FigEV4C.pdf`, `FigEV4C_source_data.tsv` |
| EV4G | Probability that the variant base is unpaired in the 5′UTR (ViennaRNA), reference against variant allele | `analysis/FigEV4G_analysis.py`, `plot/FigEV4G_plot.py` (uses `plot/fig_style.py`) | `FigEV4G.pdf`, `FigEV4G_source_data.csv` |
| EV4H | ΔpTE against local GC content (±25 nt) | `analysis/FigEV4HI_analysis.R`, `plot/FigEV4HI_plot.R` | `FigEV4H.pdf` |
| EV4I | ΔpTE against distance to the start codon | as EV4H | `FigEV4I.pdf` |

EV4H and EV4I share `FigEV4HI_source_data.tsv` (pairs), `FigEV4HI_binned_summary.tsv` (bins) and `FigEV4HI_statistics.csv` (correlations).

## Inputs

| File | Content | Panels |
|---|---|---|
| `input/dbSNP_common_{5UTR,synonymous,missense,3UTR}_SNVs_dpTE.tsv.gz` | ΔpTE of dbSNP common variants (allele frequency 0.1 to 0.9) by annotation class | EV4A |
| `input/gnomAD_v4_SNVs_dpTE_start_stop_1kb.tsv.gz` (data archive) | ΔpTE of the 33,413,533 gnomAD v4.0 exome pairs in the 5′UTR and CDS within 1 kb of the start codon and in the CDS and 3′UTR within 1 kb of the stop codon; the CDS pairs are synonymous | EV4B |
| `input/gnomAD_v4_5UTR_SNVs_dpTE_abs_ge_0.1.tsv.gz` | the 147,532 of 6,239,195 gnomAD 5′UTR pairs with \|ΔpTE\| ≥ 0.1 | EV4C |
| `input/validated_5UTR_variants_sequences.json` | reference and variant 5′UTR, CDS and 3′UTR sequences and RefSeq versions of the six validated variants | EV4G |
| `input/dbSNP_common_5UTR_SNVs_dpTE_no_ClinVar_GWAS.tsv` | ΔpTE of the variant–transcript pairs | EV4H–I |
| `input/dbSNP_common_5UTR_variant_covariates.tsv` | 5′UTR length, distance to the start codon and reference GC fraction within ±25 nt of each variant | EV4H–I |

`analysis/FigEV4BC_prepare_input.R` shows how the EV4B and EV4C inputs were cut from the genome-wide gnomAD tables; it is not needed to reproduce the figure.

`Rscript analysis/FigEV4HI_prepare_input.R <sequence_table.tsv>` rebuilds the covariate table from a RefSeq sequence table (250 MB, not included); it is not needed to reproduce the figure.

## Run

```bash
for p in A B C; do
  Rscript analysis/FigEV4${p}_analysis.R
  Rscript plot/FigEV4${p}_plot.R
done
python3 analysis/FigEV4G_analysis.py
python3 plot/FigEV4G_plot.py
Rscript analysis/FigEV4HI_analysis.R
Rscript plot/FigEV4HI_plot.R
```

## Notes

- EV4A: the percentages are cumulative tails beyond each cutoff, not disjoint bins.
- EV4B: the plot script reads the input table directly. The panel shows ΔpTE between −0.6 and 0.6 (one 5′UTR pair lies outside), and the points are rasterised at 300 dpi.
- EV4C: the cutoffs run from 0.1 to 0.5 in each direction and are cumulative.
- EV4G: ViennaRNA computes the structure ensemble at 37 °C with its default Turner 2004 parameters; the unpaired probability of a base is 1 minus the sum of its base-pair probabilities. Each 5′UTR is folded on its own, because in the reporter the native CDS is replaced by the Nluc ORF. In the source data, `pUnpaired_variant_ref` and `pUnpaired_variant_alt` are the unpaired probabilities for the reference and variant alleles, `delta_pUnpaired` is variant minus reference, `utr5_position` is the 1-based position in the 5′UTR and `luc_direction` is the direction of the measured reporter effect.
- EV4H–I: grey points are single pairs; orange points are the median signed ΔpTE in 25 equal-count bins (bins with fewer than 100 pairs dropped), with the 5–95% range shaded. Spearman's ρ is computed against \|ΔpTE\|, not the signed values on the y axis. Local GC is computed on the reference allele and extends into the CDS for variants within 25 nt of the start codon.
