"""Column groups used by the model, and the ETL step that builds new features."""
import numpy as np
import pandas as pd

# Source 1 columns (what we had in Phase 1).
S1_BINARY = ["senior_citizen", "partner", "dependents", "phone_service", "paperless_billing"]
S1_NUMERIC = ["tenure", "monthly_charges", "total_charges"]
S1_CATEGORICAL = ["gender", "multiple_lines", "internet_service", "online_security",
                  "online_backup", "device_protection", "tech_support", "streaming_tv",
                  "streaming_movies", "contract", "payment_method"]

# Columns that only source 2 gives us.
S2_BINARY = ["referred_a_friend", "streaming_music", "unlimited_data"]
S2_NUMERIC = ["age", "number_of_dependents", "number_of_referrals",
              "avg_monthly_long_distance_charges", "avg_monthly_gb_download",
              "total_refunds", "total_extra_data_charges", "total_long_distance_charges",
              "total_revenue", "population", "latitude", "longitude"]
S2_CATEGORICAL = ["offer", "internet_type", "city"]
# I leave out under_30, has_internet and zip_code: the dependency checks showed
# they are fully determined by age, internet_service and city/lat/long.

# Columns the ETL step creates.
ENG_NUMERIC = ["addon_count", "avg_charge_per_month", "charge_per_service",
               "long_distance_share", "revenue_per_month"]
ENG_BINARY = ["is_month_to_month", "is_fiber", "auto_pay", "is_new_customer"]
ENG_CATEGORICAL = ["tenure_band"]

# Columns with real nulls, and columns I run outlier checks on.
NULL_COLUMNS = ["total_charges", "offer", "internet_type"]
OUTLIER_COLUMNS = ["tenure", "monthly_charges", "total_charges", "age",
                   "number_of_referrals", "avg_monthly_gb_download",
                   "avg_monthly_long_distance_charges", "total_long_distance_charges",
                   "total_revenue", "total_refunds", "total_extra_data_charges", "population"]
# Right-skewed columns where a log / Yeo-Johnson normalization makes sense.
SKEWED = ["total_charges", "total_revenue", "total_long_distance_charges",
          "avg_monthly_gb_download", "number_of_referrals", "total_refunds",
          "total_extra_data_charges", "population"]

ADDONS = ["online_security", "online_backup", "device_protection", "tech_support",
          "streaming_tv", "streaming_movies"]


def columns(use_source2=True, use_engineered=True):
    """Return (binary, numeric, categorical) feature lists for a given setup."""
    binary, numeric, categorical = list(S1_BINARY), list(S1_NUMERIC), list(S1_CATEGORICAL)
    if use_source2:
        binary += S2_BINARY
        numeric += S2_NUMERIC
        categorical += S2_CATEGORICAL
    if use_engineered:
        binary += ENG_BINARY
        numeric += ENG_NUMERIC
        categorical += ENG_CATEGORICAL
    return binary, numeric, categorical


def present(cols, df):
    """Keep only the columns that still exist (some cleaning options drop columns)."""
    return [c for c in cols if c in df.columns]


def add_features(df):
    """Build new features row by row. Nothing here is fit on data, so no leakage.
    Run this after null handling. If a cleaning option dropped an input column,
    I just skip the features that need it."""
    df = df.copy()

    def has(*cols):
        return all(c in df for c in cols)

    # I count how many paid add-ons each customer has.
    df["addon_count"] = df[ADDONS].eq("Yes").sum(axis=1) + df["streaming_music"].fillna(0)

    # I use tenure + 1 so brand new customers (tenure 0) do not divide by zero.
    if has("total_charges", "tenure"):
        df["avg_charge_per_month"] = df["total_charges"] / (df["tenure"] + 1)
    if has("total_revenue", "tenure"):
        df["revenue_per_month"] = df["total_revenue"] / (df["tenure"] + 1)
    df["long_distance_share"] = (df["total_long_distance_charges"]
                                 / df["total_revenue"].replace(0, np.nan)).fillna(0)

    # I divide the bill by the number of services to get a price per service.
    if has("monthly_charges"):
        services = (df["phone_service"] + (df["internet_service"] != "No").astype(int)
                    + df["addon_count"])
        df["charge_per_service"] = df["monthly_charges"] / services.clip(lower=1)

    df["is_month_to_month"] = (df["contract"] == "Month-to-month").astype(int)
    df["is_fiber"] = (df["internet_service"] == "Fiber optic").astype(int)
    df["auto_pay"] = df["payment_method"].str.contains("automatic").astype(int)

    if has("tenure"):
        df["is_new_customer"] = (df["tenure"] <= 6).astype(int)
        # The last band is open-ended so a bad tenure value still lands somewhere.
        df["tenure_band"] = pd.cut(df["tenure"], bins=[-np.inf, 12, 24, 48, np.inf],
                                   labels=["0-1y", "1-2y", "2-4y", "4y+"]).astype(str)
    return df
