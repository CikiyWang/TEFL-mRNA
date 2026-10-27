# Fig. 5A: ROC curves of the TE-high class (pred_0 vs true_y_bi_0) for each of the 10
# cross-validation folds in each cell line.
suppressPackageStartupMessages({
  library(data.table)
  library(pROC)
})

file_arg <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
base_dir <- if (length(file_arg)) dirname(dirname(normalizePath(sub("^--file=", "", file_arg)))) else getwd()
input_dir <- file.path(base_dir, "input")
output_dir <- file.path(base_dir, "output")
dir.create(output_dir, showWarnings = FALSE)

CELLS <- c("HEK293T", "HeLa", "HepG2", "HMECs", "LCL", "PC3", "PC9", "T47D", "U2OS")

curves <- list()
aucs <- list()
for (cl in CELLS) {
  for (f in 1:10) {
    df <- fread(file.path(input_dir, "cell_line_classifier_CV_predictions", cl, paste0("Model_Prediction.fold_", f, ".tsv")), sep = "\t")
    roc_obj <- roc(df$true_y_bi_0, df$pred_0, quiet = TRUE)
    auc_val <- as.numeric(auc(roc_obj))
    curves[[length(curves) + 1]] <- data.table(cell_line = cl, fold = f,
                                               specificity = roc_obj$specificities,
                                               sensitivity = roc_obj$sensitivities)
    aucs[[length(aucs) + 1]] <- data.table(cell_line = cl, fold = f, n = nrow(df),
                                           n_high = sum(df$true_y_bi_0 == 1), auc = auc_val)
  }
}
curves <- rbindlist(curves)
aucs <- rbindlist(aucs)
print(aucs[, .(mean_auc = mean(auc), sd_auc = sd(auc)), by = cell_line][order(-mean_auc)], digits = 3)

fwrite(curves, file.path(output_dir, "Fig5A_source_data.tsv.gz"), sep = "\t", compress = "gzip")
fwrite(aucs, file.path(output_dir, "Fig5A_auc.tsv"), sep = "\t")
