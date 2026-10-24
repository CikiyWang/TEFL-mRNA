suppressPackageStartupMessages(library(data.table))

file_arg <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
base_dir <- if (length(file_arg)) dirname(dirname(normalizePath(sub("^--file=", "", file_arg)))) else getwd()
output_dir <- file.path(base_dir, "output")

CELLS <- c("HEK293T", "HeLa", "HepG2", "HMECs", "LCL", "PC3", "PC9", "T47D", "U2OS")
cors <- fread(file.path(output_dir, "Fig5B_correlation.tsv"), sep = "\t")
dirs <- fread(file.path(output_dir, "Fig5B_direction_agreement.tsv"), sep = "\t")
setkey(cors, cell_line_1, cell_line_2)
setkey(dirs, cell_line_1, cell_line_2)

n_vars <- length(CELLS)
bg_gradient_neg <- colorRampPalette(c("#F0F0F0", "#E0E0E0", "#D1E5F0",
                                      "#92C5DE", "#4393C3", "#2166AC", "#053061"))(100)
text_gradient <- colorRampPalette(c("#666666", "#4d4d4d", "#333333", "#2d2d2d",
                                    "#ffffff", "#ffffff", "#ffffff", "#ffffff"))(100)
bg_gradient <- colorRampPalette(c("#FFF5F0", "#FEE0D2", "#FDDBC7",
                                  "#F4A582", "#D6604D", "#B2182B", "#67001F"))(100)

pdf(file.path(output_dir, "Fig5B.pdf"), width = 10, height = 6.5)
par(mfrow = c(1, 1), mar = c(2, 2, 2, 2), pty = "s")
plot(1, 1, xlim = c(0.5, n_vars + 0.5), ylim = c(0.5, n_vars + 0.5),
     type = "n", axes = FALSE, xlab = "", ylab = "", asp = 1, main = "")

# Upper triangle: Pearson r, coloured by |r|, with significance stars
for (i in 1:(n_vars - 1)) {
  for (j in (i + 1):n_vars) {
    st <- cors[.(CELLS[i], CELLS[j])]
    cor_val <- st$pearson_r
    abs_cor <- abs(cor_val)
    gradient_pos <- pmax(1, pmin(100, round(abs_cor * 100))) * 1.5
    if (cor_val >= 0) {
      bg_color <- bg_gradient[gradient_pos]
    } else {
      bg_color <- bg_gradient_neg[gradient_pos]
    }
    text_color <- text_gradient[gradient_pos]
    if (abs_cor < 0.05) {
      bg_color <- "#F8F8F8"
      text_color <- "#999999"
    }
    rect(j - 0.4, n_vars - i + 1 - 0.4, j + 0.4, n_vars - i + 1 + 0.4,
         col = bg_color, border = "white", lwd = 2)
    text(j, n_vars - i + 1, sprintf("%.3f", cor_val), cex = 0.8, font = 3, col = text_color)
    stars <- ifelse(st$p_value < 0.001, "***", ifelse(st$p_value < 0.01, "**", ifelse(st$p_value < 0.05, "*", "")))
    text(j + 0.08, n_vars - i + 1 - 0.27, stars, cex = 0.9, col = "black")
  }
}

for (i in 1:n_vars) {
  rect(i - 0.4, n_vars - i + 1 - 0.4, i + 0.4, n_vars - i + 1 + 0.4,
       col = "white", border = "white", lwd = 2)
  text(i, n_vars - i + 1, CELLS[i], cex = 0.8, font = 2, col = "black")
}

# Lower triangle: % of SNVs whose delta pTE is positive in both (blue), negative in both
# (red) or opposite in sign (light red) between the two cell lines
draw_slice <- function(cx, cy, from, to, col, radius = 0.45) {
  theta <- seq(from, to, length.out = 100)
  polygon(c(cx, cx + radius * cos(theta), cx), c(cy, cy + radius * sin(theta), cy),
          col = col, border = "white", lwd = 2)
}
for (i in 2:n_vars) {
  for (j in 1:(i - 1)) {
    a <- dirs[.(CELLS[j], CELLS[i])]
    cx <- j
    cy <- n_vars - i + 1
    pos_angle <- 2 * pi * a$both_positive / a$total
    neg_angle <- 2 * pi * a$both_negative / a$total
    opp_angle <- 2 * pi * a$opposite / a$total
    if (a$both_positive > 0) draw_slice(cx, cy, 0, pos_angle, "#2166AC")
    if (a$both_negative > 0) draw_slice(cx, cy, pos_angle, pos_angle + neg_angle, "#B2182B")
    if (a$opposite > 0) draw_slice(cx, cy, pos_angle + neg_angle, 2 * pi, "#F4A582")
    if (a$pct_both_positive > 15) {
      angle <- pos_angle / 2
      text(cx + 0.25 * cos(angle), cy + 0.25 * sin(angle), a$pct_both_positive, cex = 0.55, font = 3, col = "white")
    }
    if (a$pct_both_negative > 15) {
      angle <- pos_angle + neg_angle / 2
      text(cx + 0.25 * cos(angle), cy + 0.25 * sin(angle), a$pct_both_negative, cex = 0.55, font = 3, col = "white")
    }
    if (a$pct_opposite > 15) {
      angle <- pos_angle + neg_angle + opp_angle / 2
      text(cx + 0.25 * cos(angle), cy + 0.25 * sin(angle), a$pct_opposite, cex = 0.55, font = 3, col = "white")
    }
  }
}
invisible(dev.off())
