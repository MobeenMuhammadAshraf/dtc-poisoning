"""
Generate Tables T1-T4 and figures from results/full_results_dedup.csv.
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon
from statsmodels.stats.multitest import multipletests


os.makedirs("results", exist_ok=True)
res = pd.read_csv("results/full_results_dedup.csv")


# --- T1: clean baseline ---
clean = res[(res.attack == "clean") & (res.condition == "no_defence")]
t1 = (clean.groupby("model")[["Accuracy", "Recall", "Precision", "F1", "AUC", "FN", "FP"]]
        .agg(["mean", "std"]).round(4))
t1.to_csv("results/T1_clean_baseline.csv")
print("T1 saved")


# --- T2: poisoning grid ---
nd = res[res.condition == "no_defence"].copy()
t2 = nd.groupby(["model", "attack", "rate"]).agg(
        Accuracy=("Accuracy", "mean"),
        Recall=("Recall", "mean"),
        Recall_std=("Recall", "std"),
        Precision=("Precision", "mean"),
        F1=("F1", "mean"),
        AUC=("AUC", "mean"),
        FN=("FN", "mean"),
        FP=("FP", "mean"),
    ).reset_index()
t2["FN_per_108"] = (t2["FN"] * 5).round(1)
t2["FP_per_364"] = (t2["FP"] * 5).round(1)
t2.to_csv("results/T2_poisoning.csv", index=False)
print("T2 saved")


# --- T3: defence summary ---
d = res[res.condition.str.startswith("defence")].copy()
d["defence"] = d["condition"].str.extract(r"(defence\d)")
t3 = d.groupby(["model", "attack", "rate", "condition"]).agg(
        Recall=("Recall", "mean"),
        Recall_std=("Recall", "std"),
        Precision=("Precision", "mean"),
        AUC=("AUC", "mean"),
        FN=("FN", "mean"),
        FP=("FP", "mean"),
        n_flipped=("n_flipped", "mean"),
        n_dropped=("n_dropped", "mean"),
        caught=("caught", "mean"),
        detection_recall=("detection_recall", "mean"),
        detection_precision=("detection_precision", "mean"),
    ).reset_index()
t3.to_csv("results/T3_defences.csv", index=False)


nd_key = nd.groupby(["model", "attack", "rate"]).agg(
    Recall_base=("Recall", "mean"), FN_base=("FN", "mean")
).reset_index()
agg = d.groupby(["model", "attack", "rate", "defence"]).agg(
    Recall=("Recall", "mean"),
    FN=("FN", "mean"),
    AUC=("AUC", "mean"),
    Precision=("Precision", "mean"),
    detection_recall=("detection_recall", "mean"),
    detection_precision=("detection_precision", "mean"),
).reset_index()
agg = agg.merge(nd_key, on=["model", "attack", "rate"])
agg["delta_recall"] = (agg["Recall"] - agg["Recall_base"]).round(3)
agg["delta_FN"] = (agg["FN"] - agg["FN_base"]).round(2)
agg.to_csv("results/T3_summary.csv", index=False)
print("T3 saved")


# --- T3: Wilcoxon + FDR ---
rows = []
for (m, a, r, dfn), grp in d.groupby(["model", "attack", "rate", "defence"]):
    base = nd[(nd.model == m) & (nd.attack == a) & (nd.rate == r)]
    merged = base.merge(grp, on=["model", "attack", "rate", "seed", "fold"],
                        suffixes=("_base", "_def"))
    if len(merged) < 5:
        continue
    diff = merged["Recall_def"] - merged["Recall_base"]
    try:
        _, p = wilcoxon(diff)
    except ValueError:
        p = 1.0
    rows.append(dict(model=m, attack=a, rate=r, defence=dfn,
                     delta=diff.mean(), p=p, n=len(diff)))
w = pd.DataFrame(rows)
rej, padj, _, _ = multipletests(w["p"], alpha=0.05, method="fdr_bh")
w["p_adj_BH"] = padj
w["significant_BH"] = rej
w.to_csv("results/T3_wilcoxon_fdr.csv", index=False)
print("T3 Wilcoxon saved")


# --- T4: threshold retuning ---
th = res[res.condition.isin(["clean_default", "clean_tuned",
                              "poisoned_default", "poisoned_tuned",
                              "poisoned_oracle"])].copy()
t4 = th.groupby(["model", "rate", "condition"]).agg(
        Recall=("Recall", "mean"),
        Recall_std=("Recall", "std"),
        Precision=("Precision", "mean"),
        F1=("F1", "mean"),
        AUC=("AUC", "mean"),
        FN=("FN", "mean"),
        FP=("FP", "mean"),
    ).reset_index()
t4.to_csv("results/T4_threshold.csv", index=False)
print("T4 saved")


print("\nAll tables generated.")