"""
Full poisoning + defence + threshold pipeline.

Output: results/full_results_dedup.csv
"""
import os
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (accuracy_score, recall_score, f1_score,
                             precision_score, roc_auc_score, confusion_matrix)
from xgboost import XGBClassifier

os.makedirs("results", exist_ok=True)

X = pd.read_csv("data/X_prepared_dedup.csv")
y = pd.read_csv("data/y_prepared_dedup.csv").squeeze()

SEEDS = [42, 7, 2024]
RATES = [0.10, 0.20, 0.30]
DISCARD = [0.10, 0.20]
MODELS = ["LogReg", "RandomForest", "XGBoost", "MLP"]


def make_model(name, seed):
    if name == "LogReg":
        return Pipeline([("s", StandardScaler()),
                         ("c", LogisticRegression(max_iter=2000,
                                                  class_weight="balanced",
                                                  random_state=seed))])
    if name == "RandomForest":
        return RandomForestClassifier(n_estimators=400,
                                      class_weight="balanced",
                                      random_state=seed, n_jobs=-1)
    if name == "XGBoost":
        return XGBClassifier(n_estimators=400, max_depth=4, learning_rate=0.05,
                             subsample=0.9, colsample_bytree=0.9,
                             scale_pos_weight=(y == 0).sum() / (y == 1).sum(),
                             eval_metric="logloss",
                             random_state=seed, n_jobs=-1)
    if name == "MLP":
        return Pipeline([("s", StandardScaler()),
                         ("c", MLPClassifier(hidden_layer_sizes=(64, 32),
                                             max_iter=1000,
                                             random_state=seed))])


def random_flip(y_tr, rate, rng):
    y_np = y_tr.reset_index(drop=True).to_numpy().copy()
    n = int(round(rate * len(y_np)))
    idx = rng.choice(len(y_np), size=n, replace=False)
    y_np[idx] = 1 - y_np[idx]
    return pd.Series(y_np), set(idx.tolist())


def targeted_flip(y_tr, rate, rng):
    y_np = y_tr.reset_index(drop=True).to_numpy().copy()
    pos = np.where(y_np == 1)[0]
    n = int(round(rate * len(pos)))
    chosen = set(rng.choice(pos, size=n, replace=False).tolist()) if n > 0 else set()
    for i in chosen:
        y_np[i] = 0
    return pd.Series(y_np), chosen


def def1_keep(X_tr, y_poison, discard, seed=None):
    Xs = StandardScaler().fit_transform(X_tr)
    y_np = y_poison.to_numpy()
    dist = np.zeros(len(y_np))
    for cls in [0, 1]:
        mask = (y_np == cls)
        if mask.sum() == 0:
            continue
        c = Xs[mask].mean(axis=0)
        dist[mask] = np.linalg.norm(Xs[mask] - c, axis=1)
    n_drop = int(round(discard * len(y_np)))
    drop = set(np.argsort(dist)[-n_drop:].tolist())
    return np.array(sorted(set(range(len(y_np))) - drop))


def def2_keep(X_tr, y_poison, discard, seed=42):
    n = len(y_poison)
    oof = np.zeros(n)
    inner = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
    for tri, tei in inner.split(X_tr, y_poison):
        m = Pipeline([("s", StandardScaler()),
                      ("c", LogisticRegression(max_iter=2000,
                                               class_weight="balanced",
                                               random_state=seed))])
        m.fit(X_tr.iloc[tri], y_poison.iloc[tri])
        oof[tei] = m.predict_proba(X_tr.iloc[tei])[:, 1]
    susp = np.abs(oof - y_poison.to_numpy())
    n_drop = int(round(discard * n))
    drop = set(np.argsort(susp)[-n_drop:].tolist())
    return np.array(sorted(set(range(n)) - drop))


def eval_at(model, X_te, y_te, thr=0.5):
    prob = model.predict_proba(X_te)[:, 1]
    pred = (prob >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_te, pred, labels=[0, 1]).ravel()
    return dict(Accuracy=accuracy_score(y_te, pred),
                Recall=recall_score(y_te, pred, zero_division=0),
                Precision=precision_score(y_te, pred, zero_division=0),
                F1=f1_score(y_te, pred, zero_division=0),
                AUC=roc_auc_score(y_te, prob),
                FN=int(fn), TP=int(tp), FP=int(fp), TN=int(tn))


def tune_threshold(model, X_val, y_val):
    prob = model.predict_proba(X_val)[:, 1]
    best_t, best_f1 = 0.5, -1
    for t in np.linspace(0.05, 0.95, 91):
        pred = (prob >= t).astype(int)
        f1 = f1_score(y_val, pred, zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, t
    return best_t


def main():
    t0 = time.time()
    all_rows = []

    for seed in SEEDS:
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        for model_name in MODELS:
            print(f"seed={seed} model={model_name} elapsed={time.time() - t0:.0f}s")

            for attack in ["clean", "random", "targeted"]:
                rate_list = [0.0] if attack == "clean" else RATES
                for rate in rate_list:
                    for fold, (tr_idx, te_idx) in enumerate(skf.split(X, y), 1):
                        rng = np.random.default_rng(seed * 1000 + fold)
                        X_tr = X.iloc[tr_idx].reset_index(drop=True)
                        y_tr_clean = y.iloc[tr_idx].reset_index(drop=True)
                        X_te, y_te = X.iloc[te_idx], y.iloc[te_idx]

                        if attack == "clean":
                            y_tr_poison = y_tr_clean.copy()
                            flipped = set()
                        elif attack == "random":
                            y_tr_poison, flipped = random_flip(y_tr_clean, rate, rng)
                        else:
                            y_tr_poison, flipped = targeted_flip(y_tr_clean, rate, rng)

                        m = make_model(model_name, seed)
                        m.fit(X_tr, y_tr_poison)
                        r = eval_at(m, X_te, y_te)
                        r.update(dict(model=model_name, attack=attack, rate=rate,
                                      condition="no_defence", seed=seed, fold=fold,
                                      n_flipped=len(flipped)))
                        all_rows.append(r)

                        if attack != "clean":
                            for dname, dkeep in [("defence1", def1_keep),
                                               ("defence2", def2_keep)]:
                                for disc in DISCARD:
                                    keep = dkeep(X_tr, y_tr_poison, disc, seed)
                                    X_f = X_tr.iloc[keep].reset_index(drop=True)
                                    y_f = y_tr_poison.iloc[keep].reset_index(drop=True)
                                    m = make_model(model_name, seed)
                                    m.fit(X_f, y_f)
                                    r = eval_at(m, X_te, y_te)
                                    dropped = set(range(len(y_tr_poison))) - set(keep.tolist())
                                    caught = len(dropped & flipped)
                                    r.update(dict(
                                        model=model_name, attack=attack, rate=rate,
                                        condition=f"{dname}_disc{disc}",
                                        seed=seed, fold=fold,
                                        n_flipped=len(flipped),
                                        n_dropped=len(dropped),
                                        caught=caught,
                                        detection_recall=caught / len(flipped) if flipped else np.nan,
                                        detection_precision=caught / len(dropped) if dropped else np.nan))
                                    all_rows.append(r)

            for rate in RATES:
                for fold, (tr_idx, te_idx) in enumerate(skf.split(X, y), 1):
                    rng = np.random.default_rng(seed * 1000 + fold)
                    X_tr_pool = X.iloc[tr_idx].reset_index(drop=True)
                    y_tr_pool = y.iloc[tr_idx].reset_index(drop=True)
                    X_te, y_te = X.iloc[te_idx], y.iloc[te_idx]

                    n_pool = len(X_tr_pool)
                    n_trusted = int(round(0.20 * n_pool))
                    perm = rng.permutation(n_pool)
                    tr_idx2, val_idx2 = perm[n_trusted:], perm[:n_trusted]

                    X_train = X_tr_pool.iloc[tr_idx2].reset_index(drop=True)
                    y_train_clean = y_tr_pool.iloc[tr_idx2].reset_index(drop=True)
                    X_val = X_tr_pool.iloc[val_idx2].reset_index(drop=True)
                    y_val = y_tr_pool.iloc[val_idx2].reset_index(drop=True)

                    m_clean = make_model(model_name, seed)
                    m_clean.fit(X_train, y_train_clean)

                    r = eval_at(m_clean, X_te, y_te, 0.5)
                    r.update(dict(model=model_name, rate=0.0,
                                  condition="clean_default", seed=seed, fold=fold))
                    all_rows.append(r)

                    t_clean = tune_threshold(m_clean, X_val, y_val)
                    r = eval_at(m_clean, X_te, y_te, t_clean)
                    r.update(dict(model=model_name, rate=0.0,
                                  condition="clean_tuned", seed=seed, fold=fold,
                                  threshold=t_clean))
                    all_rows.append(r)

                    y_train_poison, _ = targeted_flip(y_train_clean, rate, rng)
                    m = make_model(model_name, seed)
                    m.fit(X_train, y_train_poison)

                    r = eval_at(m, X_te, y_te, 0.5)
                    r.update(dict(model=model_name, rate=rate,
                                  condition="poisoned_default", seed=seed, fold=fold))
                    all_rows.append(r)

                    t_c = tune_threshold(m, X_val, y_val)
                    r = eval_at(m, X_te, y_te, t_c)
                    r.update(dict(model=model_name, rate=rate,
                                  condition="poisoned_tuned", seed=seed, fold=fold,
                                  threshold=t_c))
                    all_rows.append(r)

                    t_o = tune_threshold(m, X_te, y_te)
                    r = eval_at(m, X_te, y_te, t_o)
                    r.update(dict(model=model_name, rate=rate,
                                  condition="poisoned_oracle", seed=seed, fold=fold,
                                  threshold=t_o))
                    all_rows.append(r)

    res = pd.DataFrame(all_rows)
    res.to_csv("results/full_results_dedup.csv", index=False)
    print(f"\nDONE. {len(res)} rows saved. Total time: {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
