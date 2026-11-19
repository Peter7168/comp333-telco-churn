"""Step 1: load every raw file into the bronze layer and record dataset stats."""
import pandas as pd

from src import lake
from src.config import COGNOS_DIR, GITHUB_DIR, TABLES

GITHUB_FILES = ["customer_profiles.csv", "service_subscriptions.csv", "account_billing.csv"]
COGNOS_FILES = [
    "Telco_customer_churn_demographics.xlsx",
    "Telco_customer_churn_location.xlsx",
    "Telco_customer_churn_population.xlsx",
    "Telco_customer_churn_services.xlsx",
    "Telco_customer_churn_status.xlsx",
]


def load_sources():
    """Return {bronze_table_name: dataframe} for all 8 raw files."""
    tables = {}

    for f in GITHUB_FILES:
        # I keep text as text here; type fixes happen in integration.
        df = pd.read_csv(GITHUB_DIR / f, dtype=str, keep_default_na=False)
        tables["github_" + f.removesuffix(".csv")] = (df, GITHUB_DIR / f, "IBM GitHub (CSV)")

    for f in COGNOS_FILES:
        df = pd.read_excel(COGNOS_DIR / f)
        short = f.removeprefix("Telco_customer_churn_").removesuffix(".xlsx")
        tables["cognos_" + short] = (df, COGNOS_DIR / f, "IBM Cognos 11.1.3 (XLSX)")

    return tables


def count_nulls(df):
    # I count real NaNs plus text cells that are only spaces (like TotalCharges).
    blanks = 0
    for col in df.columns:
        if not pd.api.types.is_numeric_dtype(df[col]):
            blanks += df[col].fillna("x").astype(str).str.strip().eq("").sum()
    return int(df.isna().sum().sum() + blanks)


def run():
    stats = []
    for name, (df, path, source) in load_sources().items():
        lake.write(df, "bronze", name)
        stats.append({
            "source": source,
            "file": path.name,
            "bronze_table": name,
            "rows": len(df),
            "columns": df.shape[1],
            "null_cells": count_nulls(df),
            "size_mb": round(path.stat().st_size / 1_000_000, 3),
        })

    stats = pd.DataFrame(stats)
    stats.to_csv(TABLES / "dataset_stats.csv", index=False)
    # I print the short view; the CSV keeps every column.
    short = stats.assign(file=stats["file"].str.replace("Telco_customer_churn_", ""))
    print(short[["file", "rows", "columns", "null_cells", "size_mb"]].to_string(index=False))
    print(f"\n{len(stats)} files, {stats['size_mb'].sum():.2f} MB total")
    return stats


if __name__ == "__main__":
    run()
