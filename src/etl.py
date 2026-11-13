"""Step 3b: run the final cleaning + feature ETL on the whole table.

  silver/customers_integrated  ->  silver/customers_clean  ->  gold/customer_features

The experiments clean inside each fold; this step produces the tables the
dashboard and the report read. It also saves the null and outlier reports.
"""
import pandas as pd
from sklearn.feature_selection import mutual_info_classif

from src import lake
from src.clean import handle_nulls, handle_outliers, null_report, outlier_report
from src.config import LEAKY_COLUMNS, TABLES, TARGET
from src.experiments import FINAL
from src.features import ENG_BINARY, ENG_CATEGORICAL, ENG_NUMERIC, add_features, columns


def clean_full(df):
    # I pass the same frame as "test" so the fit and the apply use all rows.
    clean, _ = handle_nulls(df, df, FINAL.nulls)
    clean, _ = handle_outliers(clean, clean, FINAL.outliers)
    return clean


def feature_ranking(gold):
    """Rank every model feature by mutual information with churn (a filter method)."""
    binary, numeric, categorical = columns(use_source2=True, use_engineered=True)
    X = gold[binary + numeric + categorical].copy()
    for col in categorical:
        X[col] = X[col].astype("category").cat.codes
    discrete = [c in binary or c in categorical for c in X.columns]
    mi = mutual_info_classif(X, gold[TARGET], discrete_features=discrete, random_state=42)
    engineered = set(ENG_BINARY + ENG_NUMERIC + ENG_CATEGORICAL)
    ranking = pd.DataFrame({"feature": X.columns, "mutual_info": mi.round(4)})
    ranking["origin"] = ranking["feature"].map(
        lambda f: "ETL (engineered)" if f in engineered else "source 1" if f in
        columns(False, False)[0] + columns(False, False)[1] + columns(False, False)[2]
        else "source 2")
    return ranking.sort_values("mutual_info", ascending=False)


def run(df=None):
    if df is None:
        df = lake.read("silver", "customers_integrated")

    nulls = null_report(df)
    outliers, forest_flagged = outlier_report(df)
    nulls.to_csv(TABLES / "null_report.csv", index=False)
    outliers.to_csv(TABLES / "outlier_report.csv", index=False)
    print(nulls.to_string(index=False))
    print(outliers.to_string(index=False))
    print(f"isolation forest flags {forest_flagged} customers (2% contamination)")

    # Three of the 11 blank totals, before and after the cross-source fill.
    clean = clean_full(df)
    blanks = df.loc[df["total_charges"].isna(), "customer_id"].head(3)
    examples = (df[df["customer_id"].isin(blanks)][["customer_id", "tenure", "monthly_charges",
                                                     "total_charges", "total_charges_s2"]]
                .merge(clean[["customer_id", "total_charges"]], on="customer_id",
                       suffixes=("_before", "_after")))
    examples.to_csv(TABLES / "null_fill_examples.csv", index=False)

    # I drop the leaky status columns before anything downstream can use them.
    clean = clean.drop(columns=[c for c in LEAKY_COLUMNS if c in clean])
    lake.write(clean, "silver", "customers_clean")

    gold = add_features(clean)
    lake.write(gold, "gold", "customer_features")

    ranking = feature_ranking(gold)
    ranking.to_csv(TABLES / "feature_ranking.csv", index=False)
    print("\ntop features by mutual information:")
    print(ranking.head(15).to_string(index=False))

    lake.to_warehouse(gold, "customer_features")
    return gold


if __name__ == "__main__":
    run()
