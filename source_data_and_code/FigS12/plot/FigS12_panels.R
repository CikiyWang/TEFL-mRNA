## Shared panel definitions and style for Fig. S12. Sourced by FigS12_plot.R.

suppressPackageStartupMessages({
  library(data.table); library(ggplot2); library(scales)
})

OUTPUT_DIR <- "output"

REG <- c("5'UTR", "CDS_missense", "CDS_synonymous", "3'UTR")
LAB <- c("5'UTR" = "5'UTR", "CDS_missense" = "missense",
         "CDS_synonymous" = "synonymous", "3'UTR" = "3'UTR")

S <- list(dpi = 400, base = 24, axis_text = 21, title = 22, annot = 18,
          legend = 18, aspect = 0.88, lw = 1.0, pt = 3.2,
          col_region = c("5'UTR" = "#D6604D", "CDS_missense" = "#4393C3",
                         "CDS_synonymous" = "#92C5DE", "3'UTR" = "grey70"),
          col_grp    = c("5'UTR" = "#D6604D", "Coding" = "#4393C3"))

FONT <- if (requireNamespace("systemfonts", quietly = TRUE) &&
            "Arial" %in% systemfonts::system_fonts()$family) "Arial" else "sans"

theme_big <- function(s = S) theme_classic(base_size = s$base, base_family = FONT) +
  theme(aspect.ratio = s$aspect,
        axis.title  = element_text(size = s$base, colour = "black"),
        axis.text   = element_text(size = s$axis_text, colour = "black"),
        axis.line   = element_line(linewidth = 0.8, colour = "black"),
        axis.ticks  = element_line(linewidth = 0.8, colour = "black"),
        axis.ticks.length = unit(0.18, "lines"),
        plot.title  = element_text(size = s$title, hjust = 0, face = "plain",
                                   margin = margin(b = 7)),
        plot.tag    = element_text(size = s$base + 7, face = "bold", family = FONT),
        legend.title = element_blank(),
        legend.text = element_text(size = s$legend, family = FONT),
        legend.key.size = unit(1.05, "lines"),
        legend.background = element_rect(fill = alpha("white", 0.75), colour = NA),
        plot.margin = margin(10, 12, 8, 10))

save_panel <- function(name, p, w, h) {
  dev <- if (capabilities("cairo")) cairo_pdf else "pdf"
  ggsave(file.path(OUTPUT_DIR, paste0(name, ".pdf")), p, width = w, height = h,
         dpi = S$dpi, device = dev)
}

## ---- A: per-variant H by region, all variants ------------------------------
panel_A <- function() {
  PV <- fread(file.path(OUTPUT_DIR, "FigS12AB_per_variant_H_q.tsv.gz"))
  RS <- fread(file.path(OUTPUT_DIR, "FigS12A_H_by_region.tsv"))
  PV[, rlab := factor(LAB[region], levels = LAB)]
  RS[, rlab := factor(LAB[region], levels = LAB)]
  ggplot(PV, aes(rlab, H, fill = region)) +
    geom_hline(yintercept = 1, linetype = "dashed", colour = "black") +
    geom_violin(colour = "grey35", linewidth = 0.5, scale = "width", width = 0.85) +
    geom_boxplot(width = 0.15, fill = "white", colour = "black", linewidth = 0.55,
                 outlier.shape = NA, median.linewidth = 0.95) +
    geom_text(data = RS, aes(rlab, y = Inf, label = sprintf("%.0f%%", pct_q_lt_05)),
              inherit.aes = FALSE, vjust = 1.5, size = S$annot / .pt, family = FONT) +
    scale_fill_manual(values = S$col_region, guide = "none") +
    scale_y_log10(breaks = 10^(-2:2), labels = label_log()) +
    labs(x = NULL, y = "H  (cell MS /\nfold-error MS)",
         title = "Cross-cell-line variation exceeds\nfold noise (% BH q < 0.05)") +
    theme_big() + theme(axis.text.x = element_text(angle = 25, hjust = 1))
}

## ---- B: F-test pass rate across the directional spectrum -------------------
panel_B <- function() {
  PS <- fread(file.path(OUTPUT_DIR, "FigS12B_Fpass_by_cells_positive.tsv"))
  PS[, region := factor(region, levels = REG)]
  ggplot(PS, aes(n_cell_pos, frac_pass, colour = region, group = region)) +
    geom_hline(yintercept = 0.5, linetype = "dashed", colour = "grey40") +
    geom_line(linewidth = S$lw) + geom_point(size = S$pt) +
    scale_colour_manual(values = S$col_region, labels = LAB, name = NULL) +
    scale_x_continuous(breaks = 0:9) +
    scale_y_continuous(labels = percent_format(accuracy = 1), limits = c(0, 1)) +
    labs(x = "cell lines with predicted\n\u0394pTE > 0 (of 9)",
         y = "variants with\nF test q < 0.05",
         title = "Discordance costs 8-11 points for\ncoding, 30 for 3'UTR") +
    theme_big() +
    theme(legend.position = c(0.98, 0.02), legend.justification = c(1, 0))
}

## ---- C: H at matched effect size -------------------------------------------
panel_C <- function() {
  MM <- fread(file.path(OUTPUT_DIR, "FigS12CD_matched_pairs.tsv.gz"))
  MM[, grp := factor(grp, levels = c("5'UTR", "Coding"))]
  np <- MM[grp == "5'UTR", .N]
  ggplot(MM, aes(grp, H, fill = grp)) +
    geom_hline(yintercept = 1, linetype = "dashed", colour = "black") +
    geom_violin(colour = "grey35", linewidth = 0.5, scale = "width", width = 0.82) +
    geom_boxplot(width = 0.16, fill = "white", colour = "black", linewidth = 0.55,
                 outlier.shape = NA, median.linewidth = 0.95) +
    scale_fill_manual(values = S$col_grp, guide = "none") +
    scale_y_log10(breaks = 10^(-1:2), labels = label_log()) +
    labs(x = NULL, y = "H  (cell MS /\nfold-error MS)",
         title = sprintf("Matched on effect size\n(%s pairs)",
                         format(np, big.mark = ","))) +
    theme_big()
}

## ---- D: confident direction switches at matched effect size ----------------
panel_D <- function() {
  RBL <- fread(file.path(OUTPUT_DIR, "FigS12D_reversal_pct_with_CI.tsv"))
  RBL[, threshold := factor(threshold, levels = c("switch8", "switch9"),
                            labels = c("\u2265 8/10", "\u2265 9/10"))]
  RBL[, grp := factor(grp, levels = c("5'UTR", "Coding"))]
  ggplot(RBL, aes(threshold, pct, fill = grp)) +
    geom_col(position = position_dodge(0.78), width = 0.7) +
    geom_errorbar(aes(ymin = lo, ymax = hi), position = position_dodge(0.78),
                  width = 0.18, linewidth = 0.6) +
    geom_text(aes(label = sprintf("%.1f%%", pct)), position = position_dodge(0.78),
              vjust = -1.1, size = S$annot / .pt, family = FONT) +
    scale_fill_manual(values = S$col_grp, name = NULL) +
    scale_y_continuous(limits = c(0, 70), breaks = seq(0, 60, 20),
                       expand = expansion(c(0, 0.04))) +
    labs(x = "folds agreeing on sign", y = "confident reversal (%)",
         title = "Fold-validated sign reversal,\nmatched") +
    theme_big() + theme(legend.position = c(0.98, 0.98),
                        legend.justification = c(1, 1))
}
