"""Step 6: train the final model, evaluate it on a hold-out set and save it.

The model choice comes from the models experiment (best mean F1 in CV).
"""
from dataclasses import replace

import joblib
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import classification_report, confusion_matrix, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

from src import lake
from src.clean import iqr_fences, usable_iqr_columns
from src.config import MODELS, SEED, TABLES, TARGET
from src.experiments import FINAL, prepare_fold, split_dev_test
from src.features import NULL_COLUMNS

MODEL_PATH = MODELS / "churn_model.joblib"


def best_model_name():
    try:
        models = pd.read_csv(TABLES / "exp_models.csv")
        return models.sort_values("f1", ascending=False)["model"].iloc[0]
    except FileNotFoundError:
        return "hist_gb"


def holdout_report(df, recipe):
    # Same split as the experiments: train on the development set, grade once on the
    # locked test set that no experiment has seen.
    train, test = split_dev_test(df)
    pipe, X_tr, y_tr, X_te, y_te = prepare_fold(train, test, recipe)
    pipe.fit(X_tr, y_tr)
    pred = pipe.predict(X_te)
    prob = pipe.predict_proba(X_te)[:, 1]

    print(classification_report(y_te, pred, target_names=["stayed", "churned"]))
    tn, fp, fn, tp = confusion_matrix(y_te, pred).ravel()
    metrics = pd.DataFrame([{"model": recipe.model, "f1": f1_score(y_te, pred),
                             "roc_auc": roc_auc_score(y_te, prob), "tp": tp, "fp": fp,
                             "fn": fn, "tn": tn, "test_rows": len(y_te)}]).round(4)
    metrics.to_csv(TABLES / "final_model_metrics.csv", index=False)

    # Permutation importance: how much F1 drops when I shuffle one input column.
    imp = permutation_importance(pipe, X_te, y_te, scoring="f1", n_repeats=5,
                                 random_state=SEED, n_jobs=-1)
    importance = (pd.DataFrame({"feature": X_te.columns, "f1_drop": imp.importances_mean})
                  .sort_values("f1_drop", ascending=False).round(4))
    importance.to_csv(TABLES / "final_model_importance.csv", index=False)
    print(importance.head(10).to_string(index=False))
    return metrics


def out_of_fold_scores(df, recipe):
    """Churn probability for every customer from a model that never saw that customer."""
    scores = pd.Series(np.nan, index=df.index)
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    for train_idx, test_idx in splitter.split(df, df[TARGET]):
        pipe, X_tr, y_tr, X_te, _ = prepare_fold(df.iloc[train_idx], df.iloc[test_idx], recipe)
        pipe.fit(X_tr, y_tr)
        scores.iloc[test_idx] = pipe.predict_proba(X_te)[:, 1]
    return scores


def run(df=None):
    if df is None:
        df = lake.read("silver", "customers_integrated")
    recipe = replace(FINAL, model=best_model_name())
    print(f"final recipe: {recipe}")

    metrics = holdout_report(df, recipe)

    # I refit on every customer for the saved model.
    pipe, X, y, _, _ = prepare_fold(df, df, recipe)
    pipe.fit(X, y)

    # predict.py needs the same cleaning values the training data used.
    numeric_fill = {c: float(df[c].median()) for c in NULL_COLUMNS
                    if pd.api.types.is_numeric_dtype(df[c])}
    fences = {c: [float(v) for v in iqr_fences(df[c])[2:]] for c in usable_iqr_columns(df)}
    joblib.dump({"pipeline": pipe, "recipe": recipe, "columns": list(X.columns),
                 "numeric_fill": numeric_fill, "fences": fences}, MODEL_PATH)
    print(f"saved {MODEL_PATH.name}")

    # The dashboard shows these risk scores, so I store them in the gold layer too.
    gold = lake.read("gold", "customer_features")
    oof = out_of_fold_scores(df, recipe)
    gold["churn_probability"] = gold["customer_id"].map(
        pd.Series(oof.to_numpy(), index=df["customer_id"])).round(4)
    lake.write(gold, "gold", "customer_features")
    lake.to_warehouse(gold, "customer_features")
    return metrics


if __name__ == "__main__":
    run()
