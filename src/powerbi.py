"""Step 8: export the gold table for Power BI as a small star schema.

  fact_customers      one row per customer: measures + keys
  dim_contract        contract type
  dim_payment         payment method
  dim_internet        internet service and type
  dim_location        zip code, city, coordinates, population
  dim_tenure_band     tenure band with a sort order
  model_experiments   F1 with and without each cleaning / transformation step

Power BI reads these CSVs (Get data -> Text/CSV); results/powerbi/POWERBI_GUIDE.md says how
to link them and which visuals to build.
"""
import pandas as pd

from src import lake
from src.config import ROOT, TABLES

OUT = ROOT / "results" / "powerbi"

FACT_COLUMNS = [
    "customer_id", "contract", "payment_method", "internet_key", "zip_code", "tenure_band",
    "churn", "churn_probability", "risk_level", "gender", "senior_citizen", "partner",
    "dependents", "number_of_dependents", "age", "tenure", "monthly_charges", "total_charges",
    "total_revenue", "total_refunds", "number_of_referrals", "offer", "addon_count",
    "charge_per_service", "avg_monthly_gb_download", "paperless_billing", "auto_pay",
]
EXPERIMENTS = ["stages", "nulls", "outliers", "normalizers", "scalers", "encoders",
               "feature_selection", "models", "stress", "feature_engine"]


def dimension(df, key, columns, sort=None):
    dim = df[[key] + columns].drop_duplicates(key).sort_values(sort or key)
    return dim.reset_index(drop=True)


def experiments_table():
    """One long table: experiment, option, scores. Good for a slicer in Power BI."""
    frames = []
    for name in EXPERIMENTS:
        path = TABLES / f"exp_{name}.csv"
        if not path.exists():
            continue
        t = pd.read_csv(path)
        label = t.columns[1]   # the column that names each option
        frames.append(pd.DataFrame({"experiment": name, "option": t[label], "f1": t["f1"],
                                    "roc_auc": t["roc_auc"], "accuracy": t["accuracy"],
                                    "rows_graded": t["test_rows"]}))
    return pd.concat(frames, ignore_index=True)


def run(gold=None):
    if gold is None:
        gold = lake.read("gold", "customer_features")
    OUT.mkdir(parents=True, exist_ok=True)
    df = gold.copy()

    # I combine service and type into one key, e.g. "Fiber optic | Fiber Optic".
    df["internet_type"] = df["internet_type"].fillna("None")
    df["internet_key"] = df["internet_service"] + " | " + df["internet_type"]
    df["risk_level"] = pd.cut(df["churn_probability"], [-0.01, 0.3, 0.6, 1.0],
                              labels=["Low", "Medium", "High"]).astype(str)

    tables = {
        "fact_customers": df[FACT_COLUMNS],
        "dim_contract": dimension(df.assign(contract_months=df["contract"].map(
            {"Month-to-month": 1, "One year": 12, "Two year": 24})),
            "contract", ["contract_months"], sort="contract_months"),
        "dim_payment": dimension(df.assign(is_automatic=df["auto_pay"]), "payment_method",
                                 ["is_automatic"]),
        "dim_internet": dimension(df, "internet_key", ["internet_service", "internet_type"]),
        "dim_location": dimension(df, "zip_code", ["city", "latitude", "longitude", "population"]),
        "dim_tenure_band": pd.DataFrame({"tenure_band": ["0-1y", "1-2y", "2-4y", "4y+"],
                                         "sort_order": [1, 2, 3, 4],
                                         "months": ["0-12", "13-24", "25-48", "49-72"]}),
        "model_experiments": experiments_table(),
    }
    for name, table in tables.items():
        table.to_csv(OUT / f"{name}.csv", index=False)
        print(f"{name:<18} {len(table):>5} rows  {table.shape[1]:>2} columns")
    return tables


if __name__ == "__main__":
    run()
