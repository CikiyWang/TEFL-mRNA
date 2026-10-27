## Build the per-variant source-data table for Fig. EV5H-K from the raw per-fold
## predictions.
##
## The raw predictions (9 cell lines x 10 cross-validation folds per variant,
## ~0.6 GB) are not distributed with this package; they are available from the
## authors. This script documents how input/per_variant_cross_cellline_stats.tsv.gz
## was derived from them, and is not required to reproduce the figure.
##
## Usage (from FigEV5/):
##   Rscript analysis/FigEV5H-K_prepare_input.R <per_fold_dir>

suppressPackageStartupMessages(library(data.table))

## paths are relative to FigEV5/; run all commands from that directory
args        <- commandArgs(trailingOnly = TRUE)
PERFOLD_DIR <- if (length(args) >= 1) args[1] else "perfold"
INPUT_DIR   <- "input"
stopifnot(dir.exists(INPUT_DIR))

CELLS  <- c("HEK293T", "HMECs", "HeLa", "HepG2", "LCL", "PC3", "PC9", "T47D", "U2OS")
REG    <- c("5'UTR", "CDS_missense", "CDS_synonymous", "3'UTR")
VARMAP <- c("5'UTR"          = "5_prime_UTR_variant",
            "CDS_missense"   = "missense_variant",
            "CDS_synonymous" = "synonymous_variant",
            "3'UTR"          = "3_prime_UTR_variant")
N_FOLD <- 10L

load_one <- function(reg, cell) {
  f <- file.path(PERFOLD_DIR, sprintf("GWAS_delta_pTE.%s.%s.folds.tsv.gz",
                                      VARMAP[[reg]], cell))
  if (!file.exists(f)) stop("missing per-fold file: ", f)
  d <- fread(f, select = c("transcript_refid_var", "Fold", "delta_high"))
  ## average over duplicate transcript records within a (variant, fold)
  d <- d[, .(delta = mean(delta_high)), by = .(transcript_refid_var, Fold)]
  d[, cell_line := cell][]
}

per_region <- function(reg) {
  FL <- rbindlist(lapply(CELLS, load_one, reg = reg))
  ## keep only variants with a complete cell line x fold design
  nf <- FL[, .N, by = .(transcript_refid_var, cell_line)]
  ok <- nf[, .(ok = all(N == N_FOLD) && .N == length(CELLS)),
           by = transcript_refid_var][ok == TRUE, transcript_refid_var]
  FL <- FL[transcript_refid_var %in% ok]

  ## per (variant, cell line): fold ensemble mean, fold variance, fold sign vote
  PC <- FL[, .(ens = mean(delta), v = var(delta), n_pos = sum(delta > 0)),
           by = .(transcript_refid_var, cell_line)]

  PC[, .(
    ## between-cell-line mean square; the N_FOLD factor puts the cell-line means
    ## on the scale of a single fold prediction
    ms_cell    = N_FOLD * sum((ens - mean(ens))^2) / (length(CELLS) - 1),
    ms_err     = mean(v),                 # within-cell-line (fold) mean square
    mean_abs   = mean(abs(ens)),          # magnitude used for matching
    abs_eff    = abs(mean(ens)),          # alternative magnitude (sensitivity)
    n_cell_pos = sum(ens > 0),            # cell lines with a positive effect
    ## cell lines whose direction is certified by >=8 or >=9 of 10 folds
    rpos8 = sum(n_pos >= 8L), rneg8 = sum(n_pos <= N_FOLD - 8L),
    rpos9 = sum(n_pos >= 9L), rneg9 = sum(n_pos <= N_FOLD - 9L)
  ), by = transcript_refid_var][, region := reg][]
}

PV <- rbindlist(lapply(REG, per_region))
setnames(PV, "transcript_refid_var", "variant")
PV[, H := ms_cell / ms_err]
setcolorder(PV, c("region", "variant", "ms_cell", "ms_err", "H",
                  "mean_abs", "abs_eff", "n_cell_pos",
                  "rpos8", "rneg8", "rpos9", "rneg9"))
fwrite(PV, file.path(INPUT_DIR, "per_variant_cross_cellline_stats.tsv.gz"),
       sep = "\t")
cat(sprintf("wrote %s variants to input/per_variant_cross_cellline_stats.tsv.gz\n",
            format(nrow(PV), big.mark = ",")))
