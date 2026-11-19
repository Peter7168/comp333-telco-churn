"""Step 3: data cleaning. Null handling and outlier handling.

Every strategy is "fit" on the training rows only and then applied to the test
rows, so the test fold never leaks into the cleaning decisions. Rows are only
ever removed from training data, never from the data we evaluate on.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.impute import KNNImputer

from src.config import TARGET
from src.features import NULL_COLUMNS, OUTLIER_COLUMNS

NULL_STRATEGIES = ["drop_rows", "drop_columns", "constant", "statistical",
                   "cross_source", "knn"]
OUTLIER_STRATEGIES = ["keep", "remove_iqr", "cap_iqr", "remove_zscore", "isolation_forest"]


# ---------- nulls ----------

def null_report(df):
    rows = []
    for col in NULL_COLUMNS:
        rows.append({"column": col, "nulls": int(df[col].isna().sum()),
                     "pct": round(100 * df[col].isna().mean(), 2)})
    return pd.DataFrame(rows)


# Backup copies from source 2 that cross_source can fill from.
BACKUPS = {"total_charges": "total_charges_s2", "tenure": "tenure_s2"}


def handle_nulls(train, test, strategy):
    """Return cleaned (train, test). drop_rows also shrinks the test set (see note in report)."""
    train, test = train.copy(), test.copy()
    backups = [c for c in BACKUPS.values() if c in train]
    has_null = [c for c in train.columns
                if c not in backups and (train[c].isna().any() or test[c].isna().any())]

    if strategy == "drop_rows":
        # The naive fix: throw away any row with a null. Offer is null for 55% of
        # customers (they never got an offer), so this loses a lot of data.
        return train.dropna(subset=has_null), test.dropna(subset=has_null)

    if strategy == "drop_columns":
        return train.drop(columns=has_null), test.drop(columns=has_null)

    # For offer and internet_type a null means "no offer" / "no internet",
    # so "None" is the correct value, not a guess. I use it in every strategy below.
    for df in (train, test):
        for col in ("offer", "internet_type"):
            if col in df:
                df[col] = df[col].fillna("None")

    numeric = [c for c in has_null if pd.api.types.is_numeric_dtype(train[c])]
    text = [c for c in has_null if c not in numeric and c not in ("offer", "internet_type")]

    if strategy == "cross_source":
        # Source 2 has values where source 1 is blank (e.g. the 11 new customers).
        # Anything still missing after that falls back to the median.
        # We compare this option but do not use it in the final pipeline: for those 11
        # customers source 2 also says tenure 10 while source 1 says 0, so copying only
        # the total would leave rows with tenure 0 and 10 months of charges.
        for col, backup in BACKUPS.items():
            if col in train and backup in train:
                train[col] = train[col].fillna(train[backup])
                test[col] = test[col].fillna(test[backup])
        strategy = "statistical"

    if strategy == "constant":
        # Phase 1 rule and our final choice: the 11 blank totals are customers with
        # tenure 0 who have not been billed yet, so their total is 0.
        fills = {c: 0.0 for c in numeric}
    elif strategy == "statistical":
        fills = {c: train[c].median() for c in numeric}
    elif strategy == "knn":
        if numeric:
            # I let KNN look at all numeric columns, but only write back the ones with nulls.
            feats = [c for c in train.columns if pd.api.types.is_numeric_dtype(train[c])
                     and c not in backups and c != TARGET]
            imputer = KNNImputer(n_neighbors=5).fit(train[feats])
            train[feats] = imputer.transform(train[feats])
            test[feats] = imputer.transform(test[feats])
        fills = {}
    else:
        raise ValueError(f"unknown null strategy {strategy}")

    # Text columns get their most common training value.
    fills.update({c: train[c].mode()[0] for c in text})
    return train.fillna(fills), test.fillna(fills)


# ---------- outliers ----------

def iqr_fences(series, k=1.5):
    q1, q3 = series.quantile([0.25, 0.75])
    iqr = q3 - q1
    return q1, q3, q1 - k * iqr, q3 + k * iqr


def usable_iqr_columns(df):
    # I skip columns where Q1 == Q3 (mostly zeros), since IQR would flag every non-zero value.
    cols = []
    for col in OUTLIER_COLUMNS:
        if col in df:
            q1, q3, _, _ = iqr_fences(df[col].dropna())
            if q3 > q1:
                cols.append(col)
    return cols


def outlier_report(df):
    """Compare three detectors on the full table: IQR, z-score and Isolation Forest."""
    rows = []
    for col in OUTLIER_COLUMNS:
        s = df[col].dropna()
        q1, q3, low, high = iqr_fences(s)
        z = (s - s.mean()) / s.std()
        rows.append({
            "column": col, "q1": round(q1, 2), "q3": round(q3, 2),
            "lower_fence": round(low, 2), "upper_fence": round(high, 2),
            "min": round(s.min(), 2), "max": round(s.max(), 2),
            "iqr_outliers": int(((s < low) | (s > high)).sum()) if q3 > q1 else None,
            "zscore_outliers": int((z.abs() > 3).sum()),
            "skew": round(s.skew(), 2),
            "note": "zero-inflated, IQR not usable" if q3 == q1 else "",
        })
    report = pd.DataFrame(rows)

    numeric = df[OUTLIER_COLUMNS].fillna(0)
    forest = IsolationForest(contamination=0.02, random_state=42).fit(numeric)
    flagged = int((forest.predict(numeric) == -1).sum())
    return report, flagged


def handle_outliers(train, test, strategy):
    train, test = train.copy(), test.copy()
    cols = [c for c in OUTLIER_COLUMNS if c in train]

    if strategy == "keep":
        return train, test

    if strategy in ("remove_iqr", "cap_iqr"):
        fences = {c: iqr_fences(train[c])[2:] for c in usable_iqr_columns(train)}
        if strategy == "cap_iqr":
            # I clip both sets with the fences learned on train (winsorizing).
            for c, (low, high) in fences.items():
                train[c] = train[c].clip(low, high)
                test[c] = test[c].clip(low, high)
            return train, test
        keep = np.ones(len(train), dtype=bool)
        for c, (low, high) in fences.items():
            keep &= train[c].between(low, high).to_numpy()
        return train[keep], test

    if strategy == "winsorizer_fe":
        # Same rule as cap_iqr, done with Feature-engine's Winsorizer.
        from feature_engine.outliers import Winsorizer
        usable = usable_iqr_columns(train)
        train[usable] = train[usable].astype(float)
        test[usable] = test[usable].astype(float)
        capper = Winsorizer(capping_method="iqr", tail="both", fold=1.5, variables=usable)
        capper.fit(train[usable])
        train[usable] = capper.transform(train[usable])
        test[usable] = capper.transform(test[usable])
        return train, test

    if strategy == "remove_zscore":
        z = (train[cols] - train[cols].mean()) / train[cols].std()
        return train[(z.abs() <= 3).all(axis=1)], test

    if strategy == "isolation_forest":
        forest = IsolationForest(contamination=0.02, random_state=42).fit(train[cols])
        return train[forest.predict(train[cols]) == 1], test

    raise ValueError(f"unknown outlier strategy {strategy}")
