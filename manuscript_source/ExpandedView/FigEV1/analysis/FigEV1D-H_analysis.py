# Fig. EV1D-H: sensitivity of the TE estimate and of the consensus TE classes.
#   D, E  compositional TE  = CLR(RPF) - CLR(RNA), averaged over samples
#   F, G  regression TE     = residual of per-sample OLS of CLR(RPF) on CLR(RNA)
#   H     top: consensus TE class composition across RNA-abundance quartiles
#   H     bottom: TPM-ratio TE against mean RNA abundance
# Reference TE is the per-sample RPF/RNA TPM ratio averaged over samples (TE_mean in the
# class table). CLR is taken over the filtered mRNA set; zeros are replaced by the
# smallest positive TPM before the log, as the CLR is undefined at zero.

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, spearmanr

BASE_DIR = Path(__file__).resolve().parents[1]
INPUT_DIR = BASE_DIR / "input"
OUTPUT_DIR = BASE_DIR / "output"
SHARED_DIR = BASE_DIR.parents[1] / "shared" / "input"

RPF_TPM = INPUT_DIR / "LCL_RiboSeq_isoform_TPM_GSE61742.tsv.gz"
RNA_TPM = INPUT_DIR / "LCL_RNAseq_isoform_TPM_GSE19480.tsv.gz"
SAMPLES = INPUT_DIR / "LCL_samples_47.txt"
TE_CLASS = SHARED_DIR / "LCL_consensus_TE_classes.tsv"
FILTERED = SHARED_DIR / "LCL_TE_47_individuals.tsv"

LEVELS = ["low", "intermediate", "high"]


def clr(matrix):
    """Centred log-ratio over transcripts, within each sample (natural log)."""
    m = matrix.copy()
    m[m <= 0] = m[m > 0].min()
    log_m = np.log(m)
    return log_m - log_m.mean(axis=0, keepdims=True)


def tertile_class(x):
    lo, hi = np.quantile(x, [1 / 3, 2 / 3])
    return np.where(x <= lo, "low", np.where(x >= hi, "high", "intermediate"))


def rank_matched_class(x, reference):
    """Partition x by rank into classes of the same sizes as `reference`."""
    n_high = (reference == "high").sum()
    n_low = (reference == "low").sum()
    order = pd.Series(-np.asarray(x)).rank(method="first").values
    out = np.full(len(x), "intermediate", dtype=object)
    out[order <= n_high] = "high"
    out[order > len(x) - n_low] = "low"
    return out


def concordance(reference, assigned):
    extreme = np.isin(reference, ["high", "low"])
    flips = ((reference[extreme] == "high") & (assigned[extreme] == "low")) | \
            ((reference[extreme] == "low") & (assigned[extreme] == "high"))
    return 100 * np.mean(reference == assigned), 100 * flips.mean(), int(flips.sum()), int(extreme.sum())


samples = {line.strip() for line in SAMPLES.read_text().split("\n") if line.strip()}
rpf = pd.read_csv(RPF_TPM, sep="\t", index_col=0)
rna = pd.read_csv(RNA_TPM, sep="\t", index_col=0)
rpf.columns = [c[2:] if c.startswith("GM") else c for c in rpf.columns]
rna.columns = [c[2:] if c.startswith("NA") else c for c in rna.columns]
shared_samples = [c for c in rpf.columns if c in set(rna.columns) & samples]

training = pd.read_csv(TE_CLASS, sep="\t").set_index("seq_id")
filtered_ids = set(pd.read_csv(FILTERED, sep="\t", index_col=0).index)

transcripts = rpf.index.intersection(rna.index)
in_reference = transcripts.isin(filtered_ids)
R = rpf.loc[transcripts, shared_samples].to_numpy(float)[in_reference]
N = rna.loc[transcripts, shared_samples].to_numpy(float)[in_reference]
reference_ids = transcripts[in_reference]

clr_rpf, clr_rna = clr(R), clr(N)
te_compositional = pd.Series((clr_rpf - clr_rna).mean(axis=1), index=reference_ids)

# CLR coordinates are mean-centred, so the per-sample fit has zero intercept
slope = (clr_rna * clr_rpf).sum(axis=0) / (clr_rna ** 2).sum(axis=0)
te_regression = pd.Series((clr_rpf - slope * clr_rna).mean(axis=1), index=reference_ids)

rna_mean_tpm = rna.loc[transcripts, shared_samples].mean(axis=1)

d = pd.DataFrame({
    "seq_id": training.index,
    "consensus_class": pd.Categorical(training["Class"].values, LEVELS),
    "te_ratio": training["TE_mean"].values,
    "te_compositional": te_compositional.reindex(training.index).values,
    "te_regression": te_regression.reindex(training.index).values,
    "rna_mean_tpm": rna_mean_tpm.reindex(training.index).values,
}).set_index("seq_id")
assert d[["te_compositional", "te_regression", "rna_mean_tpm"]].notna().all().all()

d["te_compositional_log2"] = d.te_compositional / np.log(2)
d["te_regression_log2"] = d.te_regression / np.log(2)
d["te_ratio_log2"] = np.log2(d.te_ratio)
reference_class = d.consensus_class.astype(str).to_numpy()
d["class_compositional"] = rank_matched_class(d.te_compositional.to_numpy(), reference_class)
d["class_regression"] = rank_matched_class(d.te_regression.to_numpy(), reference_class)
d["quartile_rna_tpm"] = pd.qcut(d.rna_mean_tpm, 4, labels=[f"Q{i}" for i in range(1, 5)])

OUTPUT_DIR.mkdir(exist_ok=True)
d.reset_index().to_csv(OUTPUT_DIR / "FigEV1D-H_source_data.tsv", sep="\t", index=False)

stats = []
for panel, estimate, assigned in [("D, E", "te_compositional", "class_compositional"),
                                  ("F, G", "te_regression", "class_regression")]:
    agree, flip_pct, n_flip, n_extreme = concordance(reference_class, d[assigned].to_numpy())
    stats.append(dict(
        panel=panel, estimate=estimate,
        spearman_vs_te_ratio=spearmanr(d.te_ratio, d[estimate]).statistic,
        consensus_class_agreement_pct=agree, highlow_reversal_pct=flip_pct,
        highlow_reversals=f"{n_flip} of {n_extreme}",
        tertile_class_agreement_pct=100 * np.mean(tertile_class(d.te_ratio.to_numpy()) == tertile_class(d[estimate].to_numpy())),
        spearman_vs_rna_tpm=spearmanr(d[estimate], d.rna_mean_tpm).statistic))
stats.append(dict(panel="H", estimate="te_ratio", spearman_vs_te_ratio=1.0,
                  spearman_vs_rna_tpm=spearmanr(d.te_ratio, d.rna_mean_tpm).statistic))
pd.DataFrame(stats).to_csv(OUTPUT_DIR / "FigEV1D-H_statistics.tsv", sep="\t", index=False)

quartile_rows = []
for panel, column in [("H", "quartile_rna_tpm")]:
    table = pd.crosstab(d[column], d.consensus_class)
    chi2, p, dof, _ = chi2_contingency(table.values)
    for quartile in table.index:
        group = d[d[column] == quartile]
        log2_te = np.log2(group.te_ratio)
        quartile_rows.append(dict(
            panel=panel, quartile=quartile, n=len(group),
            median_rna_tpm=group.rna_mean_tpm.median(),
            pct_low=100 * (group.consensus_class == "low").mean(),
            pct_intermediate=100 * (group.consensus_class == "intermediate").mean(),
            pct_high=100 * (group.consensus_class == "high").mean(),
            median_log2_te=log2_te.median(), iqr_log2_te=log2_te.quantile(.75) - log2_te.quantile(.25),
            chi2=chi2, chi2_dof=dof, chi2_p=p))
pd.DataFrame(quartile_rows).to_csv(OUTPUT_DIR / "FigEV1H_quartile_summary.tsv", sep="\t", index=False)

# E, G: consensus class (rows) against the rank-matched class of the alternative estimate
concordance_rows = []
for panel, assigned in [("E", "class_compositional"), ("G", "class_regression")]:
    counts = pd.crosstab(d.consensus_class, pd.Categorical(d[assigned], LEVELS)).reindex(
        index=LEVELS, columns=LEVELS).fillna(0).astype(int)
    for consensus in LEVELS:
        for alt in LEVELS:
            concordance_rows.append(dict(panel=panel, consensus_class=consensus, assigned_class=alt,
                                         n=counts.loc[consensus, alt],
                                         percent_of_consensus_class=100 * counts.loc[consensus, alt] / counts.loc[consensus].sum()))
pd.DataFrame(concordance_rows).to_csv(OUTPUT_DIR / "FigEV1EG_class_concordance.tsv", sep="\t", index=False)

print(pd.DataFrame(stats).round(4).to_string(index=False))
print(pd.DataFrame(quartile_rows).round(3).to_string(index=False))
