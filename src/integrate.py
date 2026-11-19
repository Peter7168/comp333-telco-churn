"""Schema integration of source 1 (IBM CSV) and source 2 (IBM Cognos XLSX).

- One snake_case name per column (SCHEMA_MAP)
- Yes/No -> 1/0
- Source 1 wins on shared attributes; source 2 adds new ones
- Conflicts logged to results/tables/integration_conflicts.csv
"""
import pandas as pd

from src import lake
from src.config import TABLES

# source table -> {original column: unified column}
SCHEMA_MAP = {
    "github_customer_profiles": {
        "customerID": "customer_id", "gender": "gender", "SeniorCitizen": "senior_citizen",
        "Partner": "partner", "Dependents": "dependents", "tenure": "tenure",
    },
    "github_service_subscriptions": {
        "customerID": "customer_id", "PhoneService": "phone_service",
        "MultipleLines": "multiple_lines", "InternetService": "internet_service",
        "OnlineSecurity": "online_security", "OnlineBackup": "online_backup",
        "DeviceProtection": "device_protection", "TechSupport": "tech_support",
        "StreamingTV": "streaming_tv", "StreamingMovies": "streaming_movies",
    },
    "github_account_billing": {
        "customerID": "customer_id", "Contract": "contract",
        "PaperlessBilling": "paperless_billing", "PaymentMethod": "payment_method",
        "MonthlyCharges": "monthly_charges", "TotalCharges": "total_charges", "Churn": "churn",
    },
    "cognos_demographics": {
        "Customer ID": "customer_id", "Gender": "gender", "Age": "age", "Under 30": "under_30",
        "Senior Citizen": "senior_citizen", "Married": "partner", "Dependents": "dependents",
        "Number of Dependents": "number_of_dependents",
    },
    "cognos_location": {
        "Customer ID": "customer_id", "City": "city", "Zip Code": "zip_code",
        "Latitude": "latitude", "Longitude": "longitude",
    },
    "cognos_population": {"Zip Code": "zip_code", "Population": "population"},
    "cognos_services": {
        "Customer ID": "customer_id", "Referred a Friend": "referred_a_friend",
        "Number of Referrals": "number_of_referrals", "Tenure in Months": "tenure",
        "Offer": "offer", "Phone Service": "phone_service",
        "Avg Monthly Long Distance Charges": "avg_monthly_long_distance_charges",
        "Multiple Lines": "multiple_lines", "Internet Service": "has_internet",
        "Internet Type": "internet_type", "Avg Monthly GB Download": "avg_monthly_gb_download",
        "Online Security": "online_security", "Online Backup": "online_backup",
        "Device Protection Plan": "device_protection", "Premium Tech Support": "tech_support",
        "Streaming TV": "streaming_tv", "Streaming Movies": "streaming_movies",
        "Streaming Music": "streaming_music", "Unlimited Data": "unlimited_data",
        "Contract": "contract", "Paperless Billing": "paperless_billing",
        "Payment Method": "payment_method", "Monthly Charge": "monthly_charges",
        "Total Charges": "total_charges", "Total Refunds": "total_refunds",
        "Total Extra Data Charges": "total_extra_data_charges",
        "Total Long Distance Charges": "total_long_distance_charges",
        "Total Revenue": "total_revenue",
    },
    "cognos_status": {
        "Customer ID": "customer_id", "Satisfaction Score": "satisfaction_score",
        "Customer Status": "customer_status", "Churn Label": "churn", "Churn Score": "churn_score",
        "CLTV": "cltv", "Churn Category": "churn_category", "Churn Reason": "churn_reason",
    },
}
# Dropped: Count, Country, State, Lat Long, Quarter, Churn Value and the population ID
# (constant, duplicated or not a real key).

BINARY = ["senior_citizen", "partner", "dependents", "phone_service", "paperless_billing",
          "churn", "under_30", "referred_a_friend", "has_internet", "streaming_music",
          "unlimited_data"]
YES_NO = {"Yes": 1, "No": 0, "1": 1, "0": 0, 1: 1, 0: 0}

# Source 2 contract spellings -> source 1 spellings.
CONTRACT_FIX = {"Month-to-Month": "Month-to-month", "One Year": "One year", "Two Year": "Two year"}

# Attributes present in both sources (compared for conflicts).
SHARED = ["gender", "senior_citizen", "partner", "dependents", "tenure", "phone_service",
          "multiple_lines", "online_security", "online_backup", "device_protection",
          "tech_support", "streaming_tv", "streaming_movies", "contract", "paperless_billing",
          "payment_method", "monthly_charges", "total_charges", "churn"]


def load_renamed(name):
    df = lake.read("bronze", name)
    mapping = SCHEMA_MAP[name]
    return df[list(mapping)].rename(columns=mapping)


def harmonize(df):
    """Bring values to one format: 1/0 binaries, numbers as numbers, same spellings."""
    df = df.copy()
    for col in BINARY:
        if col in df:
            df[col] = df[col].map(YES_NO).astype("Int64")
    if "contract" in df:
        df["contract"] = df["contract"].replace(CONTRACT_FIX)
    if "total_charges" in df:
        # Blank strings become NaN; filled later in cleaning.
        df["total_charges"] = pd.to_numeric(df["total_charges"], errors="coerce")
    for col in ("tenure", "monthly_charges"):
        if col in df:
            df[col] = pd.to_numeric(df[col])
    return df


def build_source1():
    s1 = load_renamed("github_customer_profiles")
    for name in ("github_service_subscriptions", "github_account_billing"):
        s1 = s1.merge(load_renamed(name), on="customer_id", how="inner", validate="one_to_one")
    return harmonize(s1)


def build_source2():
    s2 = load_renamed("cognos_demographics")
    for name in ("cognos_location", "cognos_services", "cognos_status"):
        s2 = s2.merge(load_renamed(name), on="customer_id", how="inner", validate="one_to_one")
    # Location -> population is many-to-one on zip code.
    s2 = s2.merge(load_renamed("cognos_population"), on="zip_code", how="left", validate="many_to_one")
    return harmonize(s2)


def compare_shared(s1, s2):
    """For each shared attribute, count how many customers the sources disagree on."""
    both = s1.merge(s2, on="customer_id", suffixes=("_s1", "_s2"))
    summary, examples = [], []

    for col in SHARED:
        a, b = both[f"{col}_s1"], both[f"{col}_s2"]
        if col in ("multiple_lines", "online_security", "online_backup", "device_protection",
                   "tech_support", "streaming_tv", "streaming_movies"):
            # "No internet service" in source 1 is "No" in source 2.
            a = a.replace({"No internet service": "No", "No phone service": "No"})
        if pd.api.types.is_float_dtype(a) or pd.api.types.is_float_dtype(b):
            same = (a - b).abs().lt(0.01) | (a.isna() & b.isna())
        else:
            same = a.astype(str).eq(b.astype(str))

        summary.append({"attribute": col, "conflicts": int((~same).sum()),
                        "agreement_pct": round(100 * same.mean(), 2)})
        for _, row in both.loc[~same].head(3).iterrows():
            examples.append({"attribute": col, "customer_id": row["customer_id"],
                             "source1_value": row[f"{col}_s1"], "source2_value": row[f"{col}_s2"]})

    return pd.DataFrame(summary), pd.DataFrame(examples)


def mapping_table():
    rows = []
    for table, mapping in SCHEMA_MAP.items():
        for original, unified in mapping.items():
            rows.append({"source_table": table, "original_column": original,
                         "unified_column": unified})
    return pd.DataFrame(rows)


def run():
    s1 = build_source1()
    s2 = build_source2()

    # Match rates: no customer lost or duplicated.
    ids1, ids2 = set(s1["customer_id"]), set(s2["customer_id"])
    print(f"source 1: {len(s1)} customers, source 2: {len(s2)} customers, "
          f"matched: {len(ids1 & ids2)}")
    print(f"zip codes with no population row: {s2['population'].isna().sum()}")

    conflicts, examples = compare_shared(s1, s2)
    print("\nconflicts between sources:")
    disagree = conflicts[conflicts["conflicts"] > 0]
    print(disagree.to_string(index=False))
    print(f"the other {len(conflicts) - len(disagree)} shared attributes agree for every customer")

    # Source 1 wins on shared attributes; source 2 adds new columns.
    # Source 2 tenure/total_charges kept as backups for the cross_source null strategy.
    extra = [c for c in s2.columns if c not in SHARED or c == "customer_id"]
    backup = s2[["customer_id", "tenure", "total_charges"]].rename(
        columns={"tenure": "tenure_s2", "total_charges": "total_charges_s2"})
    integrated = (s1.merge(s2[extra], on="customer_id", how="inner", validate="one_to_one")
                    .merge(backup, on="customer_id", validate="one_to_one"))

    print(f"\nintegrated table: {integrated.shape[0]} rows x {integrated.shape[1]} columns")

    # Crosstabs show payment method has no one-to-one mapping between sources.
    both = s1[["customer_id", "payment_method", "contract"]].merge(
        s2[["customer_id", "payment_method", "contract"]], on="customer_id", suffixes=("_s1", "_s2"))
    pd.crosstab(both["payment_method_s1"], both["payment_method_s2"]).to_csv(
        TABLES / "payment_method_crosswalk.csv")
    pd.crosstab(both["contract_s1"], both["contract_s2"]).to_csv(TABLES / "contract_crosswalk.csv")

    mapping_table().to_csv(TABLES / "schema_mapping.csv", index=False)
    conflicts.to_csv(TABLES / "integration_conflicts.csv", index=False)
    examples.to_csv(TABLES / "integration_conflict_examples.csv", index=False)
    lake.write(integrated, "silver", "customers_integrated")
    return integrated


if __name__ == "__main__":
    run()
