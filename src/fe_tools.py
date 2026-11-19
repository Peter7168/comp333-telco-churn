"""Feature-engine transformers compared with the pipeline's own steps.

Winsorizer (IQR capping), RareLabelEncoder (rare categories -> "Rare"),
DropCorrelatedFeatures (redundant numeric features).
"""
from dataclasses import asdict, replace

import pandas as pd

from src import lake
from src.clean import handle_nulls, iqr_fences, usable_iqr_columns
from src.config import TABLES
from src.experiments import FINAL, evaluate, split_dev_test
from src.features import columns, present

try:
    from feature_engine.outliers import Winsorizer
    from feature_engine.selection import DropCorrelatedFeatures
    AVAILABLE = True
except ImportError:
    AVAILABLE = False


def winsorizer_check(df):
    """Fit Feature-engine's Winsorizer and compare it with the pipeline's IQR fences."""
    # Winsorizer needs no nulls.
    df, _ = handle_nulls(df, df, FINAL.nulls)
    cols = usable_iqr_columns(df)
    data = df[cols].astype(float)
    capper = Winsorizer(capping_method="iqr", tail="both", fold=1.5, variables=cols).fit(data)
    capped = capper.transform(data)
    rows = []
    for col in cols:
        _, _, low, high = iqr_fences(data[col])
        ours = data[col].clip(low, high)
        rows.append({"column": col,
                     "values_capped": int((capped[col] != data[col]).sum()),
                     "upper_cap_feature_engine": round(capper.right_tail_caps_[col], 2),
                     "upper_cap_ours": round(high, 2),
                     "max_difference_vs_ours": round(float((capped[col] - ours).abs().max()), 6)})
    return pd.DataFrame(rows)


def correlated_groups(gold, threshold=0.9):
    """Groups of numeric features with |Pearson r| above the threshold."""
    _, numeric, _ = columns(use_source2=True, use_engineered=True)
    numeric = present(numeric, gold)
    finder = DropCorrelatedFeatures(threshold=threshold, method="pearson", variables=numeric)
    finder.fit(gold[numeric].astype(float))
    rows = []
    for group in finder.correlated_feature_sets_:
        group = sorted(group)
        kept = [f for f in group if f not in finder.features_to_drop_]
        rows.append({"correlated_features": ", ".join(group),
                     "kept": ", ".join(kept),
                     "dropped": ", ".join(f for f in group if f not in kept)})
    return pd.DataFrame(rows)


def run(df=None):
    if not AVAILABLE:
        print("feature-engine is not installed, skipping (pip install feature-engine)")
        return None
    if df is None:
        df = lake.read("silver", "customers_integrated")

    check = winsorizer_check(lake.read("silver", "customers_integrated"))
    check.to_csv(TABLES / "feature_engine_winsorizer.csv", index=False)
    print(check.to_string(index=False))

    groups = correlated_groups(lake.read("gold", "customer_features"))
    groups.to_csv(TABLES / "feature_engine_correlated.csv", index=False)
    print()
    print(groups.to_string(index=False))

    # Score each option on the development set.
    dev, _ = split_dev_test(df)
    dev = dev.reset_index(drop=True)
    setups = [
        ("our IQR capping (final)", FINAL),
        ("Feature-engine Winsorizer", replace(FINAL, outliers="winsorizer_fe")),
        ("Feature-engine RareLabelEncoder + one-hot", replace(FINAL, encoder="rare_onehot")),
        ("Feature-engine DropCorrelatedFeatures", replace(FINAL, drop_correlated=True)),
    ]
    rows = []
    for label, recipe in setups:
        result = evaluate(dev, recipe)
        rows.append({"experiment": "feature_engine", "setup": label, **result, **asdict(recipe)})
        print(f"  {label:<44} F1={result['f1']:.4f}  features={result['n_features']}")
    table = pd.DataFrame(rows)
    table.to_csv(TABLES / "exp_feature_engine.csv", index=False)
    return table


if __name__ == "__main__":
    run()
