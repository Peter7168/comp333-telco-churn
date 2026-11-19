"""Experiments: model performance with and without each cleaning and transformation step.

Every setup is a Recipe scored with the same 5-fold stratified CV.
Each experiment changes one setting of the final recipe.
"""
import warnings
from dataclasses import asdict, dataclass, replace

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline

from src import lake
from src.clean import NULL_STRATEGIES, OUTLIER_STRATEGIES, handle_nulls, handle_outliers
from src.config import SEED, TABLES, TARGET
from src.features import SKEWED, add_features, columns, present, rank_features
from src.transform import ENCODERS, NORMALIZERS, SCALERS, build_preprocessor

# Expected warnings: unseen cities in a fold, and unscaled logistic regression.
warnings.filterwarnings("ignore", message="Found unknown categories")
warnings.filterwarnings("ignore", category=ConvergenceWarning)


@dataclass(frozen=True)
class Recipe:
    use_source2: bool = True
    use_engineered: bool = True
    nulls: str = "constant"
    outliers: str = "cap_iqr"
    normalizer: str = "yeo_johnson"
    scaler: str = "standard"
    encoder: str = "onehot"
    model: str = "logreg"
    top_k: int = 0          # keep only the k best-ranked features (0 = keep all)
    drop_correlated: bool = False   # drop repeated numeric features (Feature-engine)


# Final pipeline settings. Each experiment changes one field.
FINAL = Recipe()

# Baseline with minimal preparation: drop null rows, keep outliers, no scaling,
# categories as integer codes.
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
    # Source 2 backups for the cross_source null strategy.
    keep = list(dict.fromkeys(raw_cols + ["total_charges_s2", "tenure_s2", TARGET]))
    train, test = train[keep], test[keep]

    train, test = handle_nulls(train, test, recipe.nulls)
    train, test = handle_outliers(train, test, recipe.outliers)
    if recipe.use_engineered:
        train, test = add_features(train), add_features(test)

    binary, numeric, categorical = (present(binary, train), present(numeric, train),
                                    present(categorical, train))
    if recipe.top_k:
        # Ranked on training rows only (no leakage).
        best = set(rank_features(train, binary, numeric, categorical, TARGET).index[:recipe.top_k])
        binary, numeric, categorical = ([c for c in binary if c in best],
                                        [c for c in numeric if c in best],
                                        [c for c in categorical if c in best])
    if recipe.drop_correlated:
        # Drop numeric features with |r| > 0.9, fit on training rows only.
        from feature_engine.selection import DropCorrelatedFeatures
        finder = DropCorrelatedFeatures(threshold=0.9, variables=numeric)
        finder.fit(train[numeric].astype(float))
        numeric = [c for c in numeric if c not in finder.features_to_drop_]
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
    """Add one pipeline stage at a time."""
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
    """Simulated dirty data: 10% blanks and 3% values x10 in some numeric columns."""
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
    """Cleaning strategies on the corrupted copy (source 1 features only)."""
    dirty = corrupt(df)
    base = replace(FINAL, use_source2=False, use_engineered=False, normalizer="none")
    recipes = [("clean data (reference)", base, df)]
    recipes += [(f"nulls: {s}", replace(base, nulls=s, outliers="keep"), dirty)
                for s in NULL_STRATEGIES]
    recipes += [(f"outliers: {s}", replace(base, outliers=s), dirty)
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


def split_dev_test(df):
    """80/20 split: development set for experiments, test set for the final model only."""
    return train_test_split(df, test_size=0.2, stratify=df[TARGET], random_state=SEED)


def run(df=None):
    if df is None:
        df = lake.read("silver", "customers_integrated")
    # Experiments use the development set only.
    df, _ = split_dev_test(df)
    df = df.reset_index(drop=True)
    print(f"development set: {len(df)} customers (the 20% test set stays locked away)")

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
    # Scaling is tested on two scale-sensitive models.
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

    print("feature selection")
    results["feature_selection"] = run_grid(
        df, "feature_selection", "features",
        [(f"top {k}", replace(FINAL, top_k=k)) for k in (5, 10, 15, 20, 30)] + [("all", FINAL)])

    print("stress test (simulated dirty data)")
    results["stress"] = run_stress(df)

    summary = pd.concat(results.values(), ignore_index=True)
    summary.to_csv(TABLES / "all_experiments.csv", index=False)
    return results


if __name__ == "__main__":
    run()
