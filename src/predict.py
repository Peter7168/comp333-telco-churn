"""Load the saved model and score new customers (used by the dashboard)."""
import joblib
import pandas as pd

from src.features import add_features
from src.train import MODEL_PATH


def load():
    return joblib.load(MODEL_PATH)


def predict(customers, bundle=None):
    """Churn probability for each row (same columns as the integrated table)."""
    bundle = bundle or load()
    df = customers.copy()

    # Same cleaning as training, using the saved values.
    for col in ("offer", "internet_type"):
        if col in df:
            df[col] = df[col].fillna("None")
    df = df.fillna(bundle["numeric_fill"])
    for col, (low, high) in bundle["fences"].items():
        if col in df:
            df[col] = df[col].clip(low, high)

    if bundle["recipe"].use_engineered:
        df = add_features(df)
    return bundle["pipeline"].predict_proba(df[bundle["columns"]])[:, 1]


if __name__ == "__main__":
    from src import lake
    sample = lake.read("silver", "customers_integrated").head(5)
    print(pd.DataFrame({"customer_id": sample["customer_id"], "actual": sample["churn"],
                        "churn_probability": predict(sample).round(3)}))
