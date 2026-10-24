# Shared paths and plotting constants for Figure S10 (panels A and B).

fig_base_dir <- function() {
  a <- commandArgs(trailingOnly = FALSE)
  f <- sub("^--file=", "", a[grep("^--file=", a)])
  if (length(f) == 1L && nzchar(f)) normalizePath(file.path(dirname(f), "..")) else normalizePath(".")
}

BASE_DIR   <- fig_base_dir()
INPUT_DIR  <- file.path(BASE_DIR, "input")
OUTPUT_DIR <- file.path(BASE_DIR, "output")

GC_FLANK  <- 25L   # half-width (nt) of the local GC window centred on the variant
N_TREND   <- 25L   # equal-count x bins for the binned summary
MIN_BIN   <- 100L  # bins with fewer points are dropped
BASE_SIZE <- 14
PT_COL    <- "grey55"
TREND_COL <- "#D55E00"
RHO_COL   <- "#8B0000"

PANELS <- list(
  A = list(var = "gc_ref",  title = sprintf("local GC (+/-%d nt)", GC_FLANK),
           xlab = sprintf("GC fraction, +/-%d nt around variant", GC_FLANK)),
  B = list(var = "d2start", title = "start-codon distance",
           xlab = "distance to start codon (nt)")
)
