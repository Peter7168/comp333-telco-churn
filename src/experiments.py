"""Step 5: measure what each preparation step is worth.

The prof asked for model performance with and without data cleaning (DC) and
data transformation (DT). I describe every setup as a Recipe and score it
with the same 5-fold stratified cross-validation, so every number in the
results tables is directly comparable.

Each experiment changes ONE setting of the final recipe and keeps the rest.
"""
import warnings
from dataclasses import asdict, dataclass, replace

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline

from src import lake
from src.clean import NULL_STRATEGIES, OUTLIER_STRATEGIES, handle_nulls, handle_outliers
from src.config import SEED, TABLES, TARGET
from src.features import SKEWED, add_features, columns, present
from src.transform import ENCODERS, NORMALIZERS, SCALERS, build_preprocessor

# Cities that only appear in a test fold are expected (they become all zeros),
# and unscaled logistic regression not converging is one of the results we report.
warnings.filterwarnings("ignore", message="Found unknown categories")
warnings.filterwarnings("ignore", category=ConvergenceWarning)


@dataclass(frozen=True)
class Recipe:
    use_source2: bool = True
    use_engineered: bool = True
    nulls: str = "cross_source"
    outliers: str = "cap_iqr"
    normalizer: str = "yeo_johnson"
    scaler: str = "standard"
    encoder: str = "onehot"
    model: str = "logreg"


# The setup we present as our final pipeline. Experiments vary one field at a time.
FINAL = Recipe()

# What "no cleaning, no transformation" means for us: drop rows with nulls,
# keep outliers, raw numbers, and categories turned into integer codes
# (the least we can do so the model accepts text columns).
RAW = Recipe(use_source2=False, use_engineered=False, nulls="drop_rows", outliers="keep",
             normalizer="none", scaler="none", encoder="ordinal")


def make_model(name):
    if name == "logreg":
        return LogisticRegression(max_iter=3000, class_weight="balanced")
    if name == "knn":
        return KNeighborsClassifier(n_neighbors=25)
    if name == "random_forest":
        return RandomForestClassifier(n_estimators=300, min_samples_leaf=3,
                                      class_weight="balanced", n_jobs=-1, random_state=SEED)
    if name == "hist_gb":
        return HistGradientBoostingClassifier(class_weight="balanced", random_state=SEED)
    raise ValueError(name)


def prepare_fold(train, test, recipe):
    """Apply one recipe to a train/test split. Returns a ready pipeline and the data."""
    binary, numeric, categorical = columns(recipe.use_source2, recipe.use_engineered)
    raw_cols = [c for c in binary + numeric + categorical if c in train]
    # I keep the source 2 copies of total_charges and tenure for the cross_source null fill.
    keep = list(dict.fromkeys(raw_cols + ["total_charges_s2", "tenure_s2", TARGET]))
    train, test = train[keep], test[keep]

    train, test = handle_nulls(train, test, recipe.nulls)
    train, test = handle_outliers(train, test, recipe.outliers)
    if recipe.use_engineered:
        train, test = add_features(train), add_features(test)

    binary, numeric, categorical = (present(binary, train), present(numeric, train),
                                    present(categorical, train))
    pre = build_preprocessor(binary, numeric, categorical, SKEWED,
                             scaler=recipe.scaler, normalizer=recipe.normalizer,
                             encoder=recipe.encoder)
    pipe = Pipeline([("prep", pre), ("model", make_model(recipe.model))])
    features = binary + numeric + categorical
    return pipe, train[features], train[TARGET], test[features], test[TARGET]


def evaluate(df, recipe, folds=5):
    """Mean and std of the main metrics over stratified k-fold CV."""
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=SEED)
    scores = []
    for train_idx, test_idx in splitter.split(df, df[TARGET]):
        pipe, X_tr, y_tr, X_te, y_te = prepare_fold(df.iloc[train_idx], df.iloc[test_idx], recipe)
        pipe.fit(X_tr, y_tr)
        pred = pipe.predict(X_te)
        prob = pipe.predict_proba(X_te)[:, 1]
        scores.append({
            "f1": f1_score(y_te, pred), "accuracy": accuracy_score(y_te, pred),
            "precision": precision_score(y_te, pred), "recall": recall_score(y_te, pred),
            "roc_auc": roc_auc_score(y_te, prob),
            "train_rows": len(X_tr), "test_rows": len(X_te),
            "n_features": pipe.named_steps["prep"].transform(X_te.head(1)).shape[1],
        })
    s = pd.DataFrame(scores)
    out = {m: float(round(s[m].mean(), 4)) for m in ["f1", "accuracy", "precision", "recall", "roc_auc"]}
    out["f1_std"] = float(round(s["f1"].std(), 4))
    out["train_rows"] = int(s["train_rows"].mean())
    out["test_rows"] = int(s["test_rows"].mean())
    out["n_features"] = int(s["n_features"].iloc[0])
    return out


def run_grid(df, name, label_field, recipes):
    """Score a list of (label, recipe) pairs and save them as one results table."""
    rows = []
    for label, recipe in recipes:
        result = evaluate(df, recipe)
        rows.append({"experiment": name, label_field: label, **result, **asdict(recipe)})
        print(f"  {name:<14} {label:<28} F1={result['f1']:.4f}  AUC={result['roc_auc']:.4f}")
    table = pd.DataFrame(rows)
    table.to_csv(TABLES / f"exp_{name}.csv", index=False)
    return table


def stage_recipes():
    """The cumulative story for the slides: add one stage of the project at a time."""
    cleaned = replace(RAW, nulls=FINAL.nulls, outliers=FINAL.outliers)
    transformed = replace(cleaned, normalizer=FINAL.normalizer, scaler=FINAL.scaler,
                          encoder=FINAL.encoder)
    integrated = replace(transformed, use_source2=True)
    engineered = replace(integrated, use_engineered=True)
    return [
        ("1. raw source 1", RAW),
        ("2. + cleaning", cleaned),
        ("3. + transformation", transformed),
        ("4. + integration (source 2)", integrated),
        ("5. + ETL features", engineered),
    ]


STRESS_NUMERIC = ["tenure", "monthly_charges", "total_charges", "age", "avg_monthly_gb_download"]


def corrupt(df, null_rate=0.10, error_rate=0.03, seed=SEED):
    """Simulate dirty data: blank out 10% of some numeric cells and multiply 3%
    of them by 10 (a typing error with an extra zero). The source 2 backups stay
    untouched, like a second system that kept a correct copy."""
    rng = np.random.default_rng(seed)
    dirty = df.copy()
    for col in STRESS_NUMERIC:
        dirty[col] = dirty[col].astype(float)
        typo = rng.random(len(dirty)) < error_rate
        dirty.loc[typo, col] = dirty.loc[typo, col] * 10
        blank = rng.random(len(dirty)) < null_rate
        dirty.loc[blank, col] = np.nan
    return dirty


def run_stress(df):
    """Same recipes on the corrupted copy, so the value of cleaning is visible.

    I use source 1 features only here: with source 2 the model has enough other
    signal that it barely notices the damage, which hides the point of the test.
    """
    dirty = corrupt(df)
    base = replace(FINAL, use_source2=False, use_engineered=False, normalizer="none")
    recipes = [("clean data (reference)", base, df)]
    recipes += [(f"nulls: {s}", replace(base, nulls=s, outliers="keep"), dirty)
                for s in NULL_STRATEGIES]
    recipes += [(f"outliers: {s}", replace(base, nulls="cross_source", outliers=s), dirty)
                for s in OUTLIER_STRATEGIES]
    rows = []
    for label, recipe, data in recipes:
        result = evaluate(data, recipe)
        rows.append({"experiment": "stress", "setup": label, **result, **asdict(recipe)})
        print(f"  stress         {label:<28} F1={result['f1']:.4f}  AUC={result['roc_auc']:.4f}"
              f"  test_rows={result['test_rows']}")
    table = pd.DataFrame(rows)
    table.to_csv(TABLES / "exp_stress.csv", index=False)
    return table


def run(df=None):
    if df is None:
        df = lake.read("silver", "customers_integrated")

    results = {}
    print("stages")
    results["stages"] = run_grid(df, "stages", "stage", stage_recipes())

    print("nulls")
    results["nulls"] = run_grid(df, "nulls", "strategy",
                                [(s, replace(FINAL, nulls=s)) for s in NULL_STRATEGIES])
    print("outliers")
    results["outliers"] = run_grid(df, "outliers", "strategy",
                                   [(s, replace(FINAL, outliers=s)) for s in OUTLIER_STRATEGIES])
    print("normalizers")
    results["normalizers"] = run_grid(df, "normalizers", "normalizer",
                                      [(n, replace(FINAL, normalizer=n)) for n in NORMALIZERS])
    print("scalers")
    # Logistic regression and KNN both depend on distances / magnitudes, so
    # scaling should matter for them. I test both.
    results["scalers"] = run_grid(df, "scalers", "setup",
                                  [(f"{s} | {m}", replace(FINAL, scaler=s, model=m))
                                   for m in ("logreg", "knn") for s in SCALERS])
    print("encoders")
    results["encoders"] = run_grid(df, "encoders", "encoder",
                                   [(e, replace(FINAL, encoder=e)) for e in ENCODERS])
    print("models")
    results["models"] = run_grid(df, "models", "model",
                                 [(m, replace(FINAL, model=m))
                                  for m in ("logreg", "knn", "random_forest", "hist_gb")])

    print("stress test (simulated dirty data)")
    results["stress"] = run_stress(df)

    summary = pd.concat(results.values(), ignore_index=True)
    summary.to_csv(TABLES / "all_experiments.csv", index=False)
    return results


if __name__ == "__main__":
    run()
