## Fig. EV5J-K: 5'UTR and coding variants compared at matched predicted effect
## size, by H (panel J) and by fold-validated direction switches (panel K).
##
## Magnitude is mean |dpTE|, the mean across cell lines of the per-cell-line
## absolute effect, rather than |mean dpTE|, which is reduced by cross-cell-line
## sign disagreement and is therefore partly a function of the outcome tested.
##
## Usage:  Rscript analysis/FigEV5JK_analysis.R

suppressPackageStartupMessages(library(data.table))

file_arg <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
BASE_DIR <- if (length(file_arg)) dirname(dirname(normalizePath(sub("^--file=", "", file_arg)))) else getwd()
INPUT_DIR  <- file.path(BASE_DIR, "input")
OUTPUT_DIR <- file.path(BASE_DIR, "output")
dir.create(OUTPUT_DIR, showWarnings = FALSE)

FOCAL   <- "5'UTR"
CODING  <- c("CDS_missense", "CDS_synonymous")
CALIPER <- 0.01      # 1:1 matching tolerance, log10 units (~2.3% relative)
NBOOT   <- 2000
SEED    <- 1

PV <- fread(file.path(INPUT_DIR, "per_variant_cross_cellline_stats.tsv.gz"))
PV[, grp := fifelse(region == FOCAL, "5'UTR",
             fifelse(region %in% CODING, "Coding", NA_character_))]
D <- PV[!is.na(grp)]
D[, lmag := log10(mean_abs)]

## restrict both groups to their overlapping magnitude range
lo <- D[, max(min(mean_abs)), by = grp][, max(V1)]
hi <- D[, min(max(mean_abs)), by = grp][, min(V1)]
Dr <- D[mean_abs >= lo & mean_abs <= hi]

## 1:1 greedy nearest-neighbour matching on log10 magnitude, without
## replacement, within CALIPER. 5'UTR is the smaller group and supplies cases.
set.seed(SEED)
cases <- Dr[grp == "5'UTR"][sample(.N)]
ctrl  <- Dr[grp == "Coding"][order(lmag)]
cl    <- ctrl$lmag
used  <- logical(length(cl))
pick  <- integer(nrow(cases))
for (i in seq_len(nrow(cases))) {
  j <- findInterval(cases$lmag[i], cl, all.inside = TRUE)
  best <- NA_integer_; bestd <- Inf
  for (step in 0:400) {
    for (k in c(j - step, j + step + 1L)) {
      if (k >= 1L && k <= length(cl) && !used[k]) {
        d <- abs(cl[k] - cases$lmag[i])
        if (d < bestd) { bestd <- d; best <- k }
      }
    }
    if (!is.na(best) && bestd <= abs(cl[max(min(j - step, length(cl)), 1L)] -
                                     cases$lmag[i])) break
  }
  if (!is.na(best) && bestd <= CALIPER) { used[best] <- TRUE; pick[i] <- best }
}
keep <- pick > 0L
MC <- cases[keep]                # matched 5'UTR
MK <- ctrl[pick[keep]]           # matched coding
np <- nrow(MC)
cat(sprintf("1:1 matched pairs: %s of %s eligible 5'UTR variants\n",
            format(np, big.mark = ","), format(nrow(cases), big.mark = ",")))

## a confident direction switch: certified positive in >=1 cell line and
## certified negative in >=1 other, where a cell line is certified when at
## least k of the 10 folds agree on the sign
for (k in c(8L, 9L)) {
  for (dt in list(MC, MK))
    dt[, paste0("switch", k) := get(paste0("rpos", k)) > 0 &
                               get(paste0("rneg", k)) > 0]
}

PAIRS <- rbind(
  data.table(pair = seq_len(np), grp = "5'UTR",  H = MC$H, mean_abs = MC$mean_abs,
             ms_cell = MC$ms_cell, ms_err = MC$ms_err,
             switch8 = MC$switch8, switch9 = MC$switch9),
  data.table(pair = seq_len(np), grp = "Coding", H = MK$H, mean_abs = MK$mean_abs,
             ms_cell = MK$ms_cell, ms_err = MK$ms_err,
             switch8 = MK$switch8, switch9 = MK$switch9))
fwrite(PAIRS, file.path(OUTPUT_DIR, "FigEV5JK_matched_pairs.tsv.gz"), sep = "\t")

## matching balance
BAL <- PAIRS[, .(q25 = quantile(mean_abs, .25), median = median(mean_abs),
                 q75 = quantile(mean_abs, .75), n = .N), by = grp]
BAL[, ks_p := ks.test(MC$lmag, MK$lmag)$p.value]
BAL[, smd  := (mean(MC$lmag) - mean(MK$lmag)) /
              sqrt((var(MC$lmag) + var(MK$lmag)) / 2)]
fwrite(BAL, file.path(OUTPUT_DIR, "FigEV5H-K_matching_balance.tsv"), sep = "\t")
print(BAL, digits = 3)

## Panel J: H and its two components on the matched pairs
HS <- data.table(
  quantity = c("median mean|dpTE|", "median MS_between", "median MS_within",
               "median H"),
  `5UTR`  = c(median(MC$mean_abs), median(MC$ms_cell), median(MC$ms_err),
              median(MC$H)),
  Coding  = c(median(MK$mean_abs), median(MK$ms_cell), median(MK$ms_err),
              median(MK$H)))
HS[, ratio_5UTR_over_coding := `5UTR` / Coding]
fwrite(HS, file.path(OUTPUT_DIR, "FigEV5J_H_summary.tsv"), sep = "\t")
print(HS, digits = 4)

## bootstrap over the matched-pair index, so both groups are resampled together
RBL <- rbindlist(lapply(c("switch8", "switch9"), function(cn) {
  set.seed(SEED)
  bC <- replicate(NBOOT, mean(MC[[cn]][sample.int(np, np, TRUE)]))
  bK <- replicate(NBOOT, mean(MK[[cn]][sample.int(np, np, TRUE)]))
  rbind(data.table(threshold = cn, grp = "5'UTR", pct = 100 * mean(MC[[cn]]),
                   lo = 100 * quantile(bC, .025), hi = 100 * quantile(bC, .975)),
        data.table(threshold = cn, grp = "Coding", pct = 100 * mean(MK[[cn]]),
                   lo = 100 * quantile(bK, .025), hi = 100 * quantile(bK, .975)))
}))
fwrite(RBL, file.path(OUTPUT_DIR, "FigEV5K_reversal_pct_with_CI.tsv"), sep = "\t")
print(RBL, digits = 3)
