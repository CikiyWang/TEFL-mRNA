# PDF device for the plot scripts set in Arial: cairo_pdf when the cairo library loads (on macOS it
# needs XQuartz), otherwise the macOS quartz PDF device; both embed the font. Use as
# ggsave(device = pdf_device()) or call the returned device like cairo_pdf(). The choice is made
# once per R session.
.pdf_device_cache <- new.env()
pdf_device <- function(family = "Arial") {
  if (!is.null(.pdf_device_cache[[family]])) return(.pdf_device_cache[[family]])
  works <- function(dev) {
    tf <- tempfile(fileext = ".pdf"); on.exit(unlink(tf))
    isTRUE(tryCatch({
      dev(tf, width = 2, height = 2)
      grid::grid.text("a", gp = grid::gpar(fontfamily = family)); invisible(grDevices::dev.off()); TRUE
    }, error = function(e) { try(grDevices::dev.off(), silent = TRUE); FALSE },
       warning = function(w) { try(grDevices::dev.off(), silent = TRUE); FALSE }))
  }
  quartz_pdf <- function(filename, width, height, family = "Arial", bg = "white", ...)
    grDevices::quartz(file = filename, type = "pdf", width = width, height = height, family = family, bg = bg)
  dev <- if (works(grDevices::cairo_pdf)) grDevices::cairo_pdf
         else if (isTRUE(capabilities("aqua")) && works(quartz_pdf)) quartz_pdf
         else grDevices::cairo_pdf   # as before: cairo_pdf with the platform's font substitution
  .pdf_device_cache[[family]] <- dev
  dev
}
