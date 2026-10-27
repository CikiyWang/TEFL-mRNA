suppressPackageStartupMessages({
  library(data.table); library(ggplot2); library(patchwork)
})

file_arg <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
base_dir <- if (length(file_arg)) dirname(dirname(normalizePath(sub("^--file=", "", file_arg)))) else getwd()
output_dir <- file.path(base_dir, "output")

STYLE <- list(
  base_size    = 13,
  title_size   = 12,
  axis_title   = 12,
  axis_text    = 11,
  annot_size   = 4.0,
  face_title   = "plain",
  hex_w        = 3.4,
  hex_palette  = "inferno",
  hex_bins     = 50,
  heat_low     = "#F7F7F7",
  heat_mid     = "#FDDBC7",
  heat_high    = "#B2182B",
  heat_limits  = c(0, 1),
  bar_cols     = c(ordinal = "#1B7837", our_reg = "#4393C3", ribonn = "#B2182B"),
  zero_line    = "grey85",
  diag_col     = "#2166AC",
  lab_dpte     = expression(Delta*"TE: TEFL (Class)"),
  labs_alt     = c(ordinal = "Ordinal-expected",
                   our_reg = "TEFL (Reg)",
                   ribonn  = "RiboNN")
)

theme_pub <- function(s = STYLE) {
  theme_classic(base_size = s$base_size) +
    theme(axis.text   = element_text(colour = "black", size = s$axis_text),
          axis.title  = element_text(size = s$axis_title),
          plot.title  = element_text(size = s$title_size, face = s$face_title, hjust = 0),
          legend.position = "none",
          panel.grid  = element_blank())
}

m <- fread(file.path(output_dir, "FigEV3D-F_variant_scores.tsv.gz"), sep = "\t")
pairwise <- fread(file.path(output_dir, "FigEV3D-F_pairwise_stats.tsv"), sep = "\t")

## Fig. EV3D: TEFL (Class) delta pTE against each alternative score
mk_hex <- function(col, s = STYLE) {
  st <- pairwise[pair == paste("dpte vs", col)]
  ggplot(m, aes(dpte, .data[[col]])) +
    geom_hline(yintercept = 0, colour = s$zero_line, linewidth = 0.3) +
    geom_vline(xintercept = 0, colour = s$zero_line, linewidth = 0.3) +
    geom_hex(bins = s$hex_bins) +
    scale_fill_viridis_c(option = s$hex_palette, transform = "log10") +
    labs(x = s$lab_dpte, y = bquote(Delta*"TE: "*.(s$labs_alt[[col]])),
         title = sprintf("r = %.2f,  rho = %.2f", st$pearson, st$spearman)) +
    theme_pub(s) + theme(aspect.ratio = 1) +
    geom_abline(slope = 1, intercept = 0, linetype = "dashed", colour = s$diag_col, linewidth = 0.4)
}
f1 <- mk_hex("ordinal") + mk_hex("our_reg") + mk_hex("ribonn") + plot_layout(nrow = 1)
ggsave(file.path(output_dir, "FigEV3D.pdf"), f1, width = 3 * STYLE$hex_w, height = 3)

## Fig. EV3E: Spearman correlation matrix
METS <- c("dpte", "ordinal", "our_reg", "ribonn")
cm <- fread(file.path(output_dir, "FigEV3E_source_data.tsv"), sep = "\t")
cm[, `:=`(a = factor(a, levels = METS), b = factor(b, levels = rev(METS)))]
lab_map <- c(dpte = "TEFL (Class)", ordinal = "Ordinal",
             our_reg = "TEFL (Reg)", ribonn = "RiboNN")
f2 <- ggplot(cm, aes(a, b, fill = rho)) +
  geom_tile(colour = "white", linewidth = 1.1) +
  geom_text(aes(label = sprintf("%.2f", rho)), size = STYLE$annot_size) +
  scale_fill_gradient2(low = STYLE$heat_low, mid = STYLE$heat_mid,
                       high = STYLE$heat_high, midpoint = 0.5,
                       limits = STYLE$heat_limits, name = "Spearman") +
  scale_x_discrete(labels = lab_map) + scale_y_discrete(labels = lab_map) +
  labs(x = NULL, y = NULL) +
  theme_pub() + theme(aspect.ratio = 1,
                      axis.text.x = element_text(angle = 25, hjust = 1),
                      axis.line = element_blank(), axis.ticks = element_blank())
ggsave(file.path(output_dir, "FigEV3E.pdf"), f2, width = 5.2, height = 3.5)

## Fig. EV3F: sign concordance among large-effect variants (bar: mean, error bar: range over cell lines)
sc <- fread(file.path(output_dir, "FigEV3F_summary.tsv"), sep = "\t")
sc[, alt := factor(alt, levels = c("ribonn", "our_reg", "ordinal"))]
f3 <- ggplot(sc, aes(mean_sc, alt, fill = alt)) +
  geom_col(width = 0.62) +
  geom_errorbar(aes(xmin = lo, xmax = hi), width = 0.14, linewidth = 0.5, orientation = "y") +
  geom_text(aes(label = sprintf("%.0f%%", 100 * mean_sc)),
            hjust = 1.25, colour = "white", size = STYLE$annot_size,
            fontface = "bold") +
  scale_fill_manual(values = STYLE$bar_cols) +
  scale_x_continuous(limits = c(0, 1.02), labels = scales::percent,
                     expand = expansion(c(0, 0.02))) +
  scale_y_discrete(labels = c(ordinal = "vs Ordinal",
                              our_reg = "vs TEFL Reg",
                              ribonn  = "vs RiboNN")) +
  labs(x = "Sign concordance, large-effect variants", y = NULL,
       title = "Direction agreement") +
  theme_pub() + theme(aspect.ratio = 0.5)
ggsave(file.path(output_dir, "FigEV3F.pdf"), f3, width = 6, height = 2.25)
