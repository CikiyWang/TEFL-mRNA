# Shared by the plot scripts: font / PDF-device probing, theme_pub(), save_fig().
# Source it after defining STYLE (needs STYLE$font, $w, $h and the size fields).

suppressPackageStartupMessages(library(ggplot2))
# Rscript starts in the C locale, where non-ASCII text (en dashes, Greek letters) is drawn as dots.
invisible(suppressWarnings(Sys.setlocale("LC_CTYPE", "en_US.UTF-8")))

HAS_FONT <- tryCatch(requireNamespace("systemfonts", quietly = TRUE) &&
  nrow(systemfonts::system_fonts()[systemfonts::system_fonts()$family == STYLE$font, ]) > 0, error = function(e) FALSE)
# capabilities("cairo") can say TRUE while the cairo library cannot load (macOS without XQuartz), so
# cairo_pdf is probed by actually drawing with the font. Fallback: the base pdf() device, which
# only knows the PostScript families -> Helvetica (metric-compatible with Arial).
CAIRO_OK <- HAS_FONT && isTRUE(tryCatch({
  tf <- tempfile(fileext = ".pdf"); grDevices::cairo_pdf(tf, 2, 2)
  grid::grid.text("a", gp = grid::gpar(fontfamily = STYLE$font)); invisible(dev.off()); TRUE
}, error = function(e) { try(dev.off(), silent = TRUE); FALSE }, warning = function(w) { try(dev.off(), silent = TRUE); FALSE }))
# Second choice on macOS: the quartz PDF device, which embeds the system font and needs no X11.
QUARTZ_OK <- !CAIRO_OK && HAS_FONT && isTRUE(capabilities("aqua")) && isTRUE(tryCatch({
  tf <- tempfile(fileext = ".pdf"); grDevices::quartz(file = tf, type = "pdf", width = 2, height = 2)
  grid::grid.text("a", gp = grid::gpar(fontfamily = STYLE$font)); invisible(dev.off()); TRUE
}, error = function(e) { try(dev.off(), silent = TRUE); FALSE }, warning = function(w) { try(dev.off(), silent = TRUE); FALSE }))
quartz_pdf <- function(filename, width, height, ...) grDevices::quartz(file = filename, type = "pdf", width = width, height = height)
message("PDF device: ", if (CAIRO_OK) paste("cairo_pdf,", STYLE$font) else if (QUARTZ_OK) paste("quartz,", STYLE$font)
        else "pdf(), Helvetica")

theme_pub <- function(family, s = STYLE) theme_classic(base_size = s$base_size, base_family = family) +
  theme(aspect.ratio = s$aspect,
        axis.text  = element_text(colour = "black", size = s$axis_text),
        axis.title = element_text(size = s$axis_title),
        plot.title = element_text(size = s$title_size, hjust = 0),
        plot.subtitle = element_text(size = s$subtitle_size, hjust = 0, colour = "grey25"),
        legend.text = element_text(size = s$legend_text),
        legend.title = element_blank(),
        legend.position = "bottom", legend.direction = "vertical",
        legend.justification = "left", legend.box.margin = margin(0, 0, 0, 0),
        legend.location = "plot",   # ggplot2 >= 3.5
        legend.key.width = unit(1.2, "cm"), legend.key.height = unit(0.62, "cm"),
        legend.background = element_blank(), panel.grid = element_blank(),
        plot.margin = margin(8, 14, 6, 8))

# `build(family)` returns the plot, because the usable font family depends on the device
save_fig <- function(f, build, s = STYLE) {
  if (CAIRO_OK) { dev <- grDevices::cairo_pdf; fam <- s$font }
  else if (QUARTZ_OK) { dev <- quartz_pdf; fam <- s$font }
  else { dev <- "pdf"; fam <- "Helvetica" }
  ggsave(paste0(f, ".pdf"), build(fam), width = s$w, height = s$h, device = dev)
}
