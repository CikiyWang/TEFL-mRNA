###############################################################################
# Fig. S9D-E -- plotting
#   S9D  Mean dpTE per synonymous codon substitution pair against the change in
#        codon stability coefficient (CSC), HeLa. Filled points are the
#        GC-neutral pairs (codon GC count unchanged), open points the remainder.
#   S9E  Mean dpTE per amino-acid substitution against the change in amino-acid
#        stability coefficient (AASC), HeLa. Filled points are the GC-matched
#        substitutions.
#
#   Rscript plot/FigS9DE_plot.R
#
#   Reads input/FigS9{D,E}_source_data.tsv, writes output/FigS9{D,E}.pdf.
#   dpTE is expressed in units of the cell line's SD of dpTE. rho is Spearman's,
#   over all pairs and over the controlled subset.
#
#   Panel construction is unchanged from the original six-panel figure; only the
#   panel tags differ (E, F there; D, E here).
###############################################################################
suppressPackageStartupMessages({
  library(data.table)
  library(ggplot2)
})

base_dir <- local({
  f <- sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE))
  normalizePath(file.path(if (length(f)) dirname(f) else getwd(), ".."))
})
input_dir  <- file.path(base_dir, "input")
output_dir <- file.path(base_dir, "output")
dir.create(output_dir, showWarnings = FALSE)

S <- list(base = 20, title = 22, axis_title = 21, axis_text = 18, annot = 15,
          tag = 26, stab = "#E08A1E", grey = "#8A8A8A", ink = "#1A1A1A")

FONT <- tryCatch(
  if (requireNamespace("systemfonts", quietly = TRUE) &&
      any(systemfonts::system_fonts()$family == "Arial")) "Arial" else "sans",
  error = function(e) "sans")
if (FONT != "Arial") message("Arial not found; using '", FONT, "'.")

theme_panel <- function() {
  theme_classic(base_size = S$base, base_family = FONT) +
    theme(axis.text  = element_text(colour = S$ink, size = S$axis_text),
          axis.title = element_text(colour = S$ink, size = S$axis_title),
          axis.line  = element_line(colour = S$ink, linewidth = 0.6),
          axis.ticks = element_line(colour = S$ink, linewidth = 0.6),
          plot.title = element_text(size = S$title, hjust = 0.5, face = "bold",
                                    colour = S$ink, margin = margin(b = 6)),
          plot.tag   = element_text(size = S$tag, face = "bold", colour = S$ink),
          plot.tag.position = c(0, 1),
          panel.grid = element_blank(),
          legend.position = "none",
          plot.margin = margin(10, 12, 8, 10))
}

D <- "\u0394"

make_panel <- function(dat, colour, tag, title, xlab, sub_label) {
  r_all <- cor(dat$x, dat$y, method = "spearman")
  r_sub <- cor(dat$x[dat$controlled], dat$y[dat$controlled], method = "spearman")
  lab <- sprintf("all  rho = %+.2f\n%s  rho = %+.2f", r_all, sub_label, r_sub)
  ggplot(dat, aes(x, y)) +
    geom_hline(yintercept = 0, colour = S$grey, linewidth = 0.4) +
    geom_vline(xintercept = 0, colour = S$grey, linewidth = 0.4) +
    geom_point(aes(shape = controlled, alpha = controlled), colour = colour,
               size = 2.4, stroke = 0.7) +
    geom_smooth(method = "lm", formula = y ~ x, se = FALSE, colour = colour,
                linewidth = 1.1) +
    annotate("text", x = -Inf, y = Inf, label = lab, hjust = -0.07, vjust = 1.25,
             size = S$annot/.pt, family = FONT, colour = S$ink, lineheight = 0.95) +
    scale_shape_manual(values = c("TRUE" = 16, "FALSE" = 1)) +
    scale_alpha_manual(values = c("TRUE" = 0.95, "FALSE" = 0.55)) +
    scale_x_continuous(expand = expansion(mult = 0.14), n.breaks = 5) +
    scale_y_continuous(limits = c(-1.25, 1.65), breaks = seq(-1.0, 1.5, 0.5)) +
    labs(tag = tag, title = title, x = xlab, y = paste0(D, "pTE (SD rescaled)")) +
    theme_panel() + theme(aspect.ratio = 1)
}

COD <- fread(file.path(input_dir, "FigS9D_source_data.tsv"))
AA  <- fread(file.path(input_dir, "FigS9E_source_data.tsv"))

pD <- make_panel(COD[, .(x = delta_CSC_HeLa, y = mean_delta_pTE_in_SD_units,
                         controlled = GC_neutral)],
                 S$stab, "D", "HeLa", paste0(D, "CSC (ALT-REF)"), "GC-neutral")
pE <- make_panel(AA[, .(x = delta_AASC_HeLa, y = mean_delta_pTE_in_SD_units,
                        controlled = GC_matched)],
                 S$stab, "E", "HeLa", paste0(D, "AASC (ALT-REF)"), "GC-matched")

ggsave(file.path(output_dir, "FigS9D.pdf"), pD, width = 5.5, height = 5.5,
       device = cairo_pdf, bg = "white")
ggsave(file.path(output_dir, "FigS9E.pdf"), pE, width = 5.5, height = 5.5,
       device = cairo_pdf, bg = "white")
