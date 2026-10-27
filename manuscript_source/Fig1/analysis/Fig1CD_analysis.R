# Fig. 1C, D: performance of the LCL TE classifier for the TE-high class (pred_0 vs true_y_bi_0)
# in each of the 10 cross-validation folds. Threshold-based metrics use pred_0 >= 0.5.
suppressPackageStartupMessages({
  library(data.table)
  library(MLmetrics)
  library(pROC)
})

file_arg <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
base_dir <- if (length(file_arg)) dirname(dirname(normalizePath(sub("^--file=", "", file_arg)))) else getwd()
input_dir <- file.path(base_dir, "input")
output_dir <- file.path(base_dir, "output")
dir.create(output_dir, showWarnings = FALSE)

CRITERIA <- c("AUC", "PRAUC", "Precision", "Recall", "Accuracy", "F1-score")

metrics <- list()
curves <- list()
aucs <- list()
for (i in 1:10) {
  res <- read.table(file.path(input_dir, "LCL_classifier_CV_predictions", paste0("Model_Prediction.fold_", i, ".tsv")), sep = "\t", header = TRUE)
  pred_label <- ifelse(res$pred_0 >= .5, 1, 0)
  metrics[[i]] <- data.table(fold = i, Criteria = CRITERIA,
                             perf = c(AUC(res$pred_0, res$true_y_bi_0),
                                      PRAUC(res$pred_0, res$true_y_bi_0),
                                      Precision(res$true_y_bi_0, pred_label, positive = 1),
                                      Recall(res$true_y_bi_0, pred_label, positive = 1),
                                      Accuracy(pred_label, res$true_y_bi_0),
                                      F1_Score(res$true_y_bi_0, pred_label, positive = 1)))
  roc_obj <- roc(res$true_y_bi_0, res$pred_0, quiet = TRUE)
  curves[[i]] <- data.table(fold = i, specificity = roc_obj$specificities, sensitivity = roc_obj$sensitivities)
  aucs[[i]] <- data.table(fold = i, n = nrow(res), n_high = sum(res$true_y_bi_0 == 1), auc = as.numeric(roc_obj$auc))
}

# Fig. 1C: one row per fold and metric, with the mean +/- SD drawn as the bar and error bar
fig1c <- rbindlist(metrics)
fig1c[, Criteria := factor(Criteria, levels = CRITERIA)]
setorder(fig1c, Criteria, fold)
fig1c[, class := "High"]
fig1c[, `:=`(mean = mean(perf), upper_ci = mean(perf) + sd(perf), lower_ci = mean(perf) - sd(perf)), by = Criteria]
print(unique(fig1c[, .(Criteria, mean, sd = upper_ci - mean)]), digits = 3)

fwrite(fig1c, file.path(output_dir, "Fig1C_source_data.tsv"), sep = "\t")
fwrite(rbindlist(curves), file.path(output_dir, "Fig1D_source_data.tsv.gz"), sep = "\t", compress = "gzip")
fwrite(rbindlist(aucs), file.path(output_dir, "Fig1D_auc.tsv"), sep = "\t")
