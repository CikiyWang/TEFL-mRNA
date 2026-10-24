# Fig. S3G: per-fold AUC for TE-high vs all other mRNAs, classifier trained on the consensus mRNAs
# (filter) vs on all mRNAs (nofilter). Both models share the 10-fold split and predict identical
# held-out transcripts, so the folds are paired. Scored on the held-out consensus transcripts
# (in_filtered) against their published class; P = two-sided Wilcoxon signed-rank test on the
# 10 paired per-fold AUCs.

from pathlib import Path

import pandas as pd
from scipy.stats import wilcoxon
from sklearn.metrics import roc_auc_score

BASE_DIR = Path(__file__).resolve().parents[1]
INPUT_DIR = BASE_DIR / "input"
OUTPUT_DIR = BASE_DIR / "output"

RUNS = {"filter": INPUT_DIR / "FL_regions.filter_cor0.7_l0.25_h1.TPE_opt.same_split_as_nofilter",
        "nofilter": INPUT_DIR / "FL_regions.nofilter_cor0.7_na5.TPE_opt"}
FOLDS = range(1, 11)

pred = {m: {k: pd.read_csv(d / f"Model_Prediction.fold_{k}.tsv", sep="\t") for k in FOLDS} for m, d in RUNS.items()}
for k in FOLDS:
    assert (pred["filter"][k].transcript_id.values == pred["nofilter"][k].transcript_id.values).all(), \
        f"fold {k}: models are not paired"

# pred_0 = P(high)
rows = []
for m in RUNS:
    for k in FOLDS:
        d = pred[m][k][pred[m][k].in_filtered]
        rows.append(dict(model=m, fold=k, AUC=roc_auc_score((d.published_Class == "high").astype(int), d.pred_0)))
auc = pd.DataFrame(rows)

w = auc.pivot(index="fold", columns="model", values="AUC")
summary = pd.DataFrame([dict(
    filter_mean=w["filter"].mean(), filter_sd=w["filter"].std(ddof=1),
    nofilter_mean=w["nofilter"].mean(), nofilter_sd=w["nofilter"].std(ddof=1),
    folds_filter_better=int((w["filter"] > w["nofilter"]).sum()),
    n_transcripts=int(sum(pred["filter"][k].in_filtered.sum() for k in FOLDS)),
    wilcoxon_p=wilcoxon(w["filter"], w["nofilter"]).pvalue)])

OUTPUT_DIR.mkdir(exist_ok=True)
auc.to_csv(OUTPUT_DIR / "FigS3G_auc_per_fold.tsv", sep="\t", index=False)
summary.to_csv(OUTPUT_DIR / "FigS3G_auc_summary.tsv", sep="\t", index=False)
print(w.round(4).to_string())
print(summary.round(4).to_string(index=False))
