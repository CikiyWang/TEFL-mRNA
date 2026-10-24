###############################################################################
# Fig. S9A-C -- plotting
#   S9A  Mean dpTE for each of the 12 directed nucleotide substitutions, before
#        (filled) and after (open) adjustment for local GC, transcript-wide GC,
#        codon position, relative CDS position and CDS length.
#   S9B  Unadjusted mean dpTE per substitution within deciles of transcript-wide
#        GC content.
#   S9C  Variance of dpTE explained by substitution type alone, by the two GC
#        terms alone, and by the full covariate set.
#
#   Rscript plot/FigS9ABC_plot.R
#
#   Reads input/FigS9{A,B,C}_source_data.tsv, writes output/FigS9{A,B,C}.pdf.
#   Set in Arial through cairo_pdf; needs a UTF-8 locale.
###############################################################################
suppressMessages({
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

eff <- fread(file.path(input_dir, "FigS9A_source_data.tsv"))
dc  <- fread(file.path(input_dir, "FigS9B_source_data.tsv"))
ms  <- fread(file.path(input_dir, "FigS9C_source_data.tsv"))

setnames(eff, c("substitution", "allele_introduced"), c("subst_u", "alt"))
setnames(dc,  c("substitution", "mean_dpTE"), c("subst_u", "mu"))

# Substitutions ordered by mean effect across both variant classes, so panels A
# and B share a y-axis and colour ordering.
ordr <- eff[, .(m = mean(unadjusted)), by = subst_u][order(m), subst_u]
for (D in list(eff, dc)) D[, `:=`(subst_u = factor(subst_u, levels = ordr),
                                  class   = factor(class, levels = c("Synonymous", "Missense")))]
dc[, alt := substr(as.character(subst_u), 3, 3)]

r2  <- dcast(ms[metric %chin% c("R2_subst_only", "R2_gc_only", "R2_full")],
             class ~ metric, value.var = "value")
r2l <- melt(r2[, .(class, `Substitution type` = R2_subst_only, `GC only` = R2_gc_only,
                   `Full model` = R2_full)],
            id.vars = "class", variable.name = "model", value.name = "R2")
r2l[, class := factor(class, levels = c("Synonymous", "Missense"))]

PAL <- c(A = "#E69F00", C = "#0072B2", G = "#009E73", U = "#D55E00")
th <- theme_classic(base_size = 19, base_family = "Arial") +
  theme(strip.background = element_blank(), strip.text = element_text(size = 19, face = "bold"),
        axis.text = element_text(colour = "black", size = 18), axis.title = element_text(size = 20),
        legend.position = "top", legend.title = element_text(size = 17),
        legend.text = element_text(size = 17), plot.title = element_text(size = 18, hjust = 0),
        panel.grid.major.y = element_line(colour = "grey92", linewidth = .4))

pA <- ggplot(eff, aes(y = subst_u)) +
  geom_vline(xintercept = 0, colour = "grey55", linewidth = .5) +
  geom_segment(aes(x = unadjusted * 1e3, xend = adjusted * 1e3, yend = subst_u),
               colour = "grey45", linewidth = 1.1) +
  geom_point(aes(x = unadjusted * 1e3, colour = alt), size = 4.2) +
  geom_point(aes(x = adjusted * 1e3), shape = 21, fill = "white", stroke = 1.2, size = 3.0) +
  facet_wrap(~class) + scale_colour_manual(values = PAL, name = "Allele introduced") +
  labs(x = expression(Mean ~ Delta * pTE ~ (x10^-3)), y = NULL, title = "A") +
  th + theme(aspect.ratio = 1)

pB <- ggplot(dc, aes(as.integer(gc_decile), mu * 1e3, colour = alt, group = subst_u)) +
  geom_hline(yintercept = 0, colour = "grey55", linewidth = .5) +
  geom_line(linewidth = 1.0, alpha = .85) + geom_point(size = 2.2) +
  facet_wrap(~class) + scale_colour_manual(values = PAL, guide = "none") +
  scale_x_continuous(breaks = c(1, 3, 5, 7, 10)) +
  labs(x = "Transcript-wide GC decile  (1 = GC 0.31, 10 = GC 0.77)",
       y = expression(Mean ~ Delta * pTE ~ (x10^-3)), title = "B") +
  th + theme(aspect.ratio = 1)

pC <- ggplot(r2l, aes(model, R2 * 100, fill = model)) + geom_col(width = 0.7) +
  geom_text(aes(label = sprintf("%.2f%%", R2 * 100)), vjust = -0.35, size = 6.0, family = "Arial") +
  facet_wrap(~class) + scale_fill_manual(values = c("#334155", "#94A3B8", "#64748B"), guide = "none") +
  scale_y_continuous(expand = expansion(mult = c(0, .22))) +
  labs(x = NULL, y = expression(Variance ~ of ~ Delta * pTE ~ explained ~ ("%")), title = "C") +
  th + theme(aspect.ratio = 1.25, axis.text.x = element_text(angle = 25, hjust = 1))

panel_pdf <- function(p, file, w, h) {
  cairo_pdf(file.path(output_dir, file), width = w, height = h, family = "Arial")
  print(p); invisible(dev.off())
}
panel_pdf(pA, "FigS9A.pdf", 11, 6.0)
panel_pdf(pB, "FigS9B.pdf", 11, 6.0)
panel_pdf(pC, "FigS9C.pdf", 11, 6.8)
