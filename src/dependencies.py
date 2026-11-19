"""Functional dependency checks: X -> Y holds if each X value maps to one Y value."""
import pandas as pd

from src import lake
from src.config import TABLES

# (left side, right side, note)
CANDIDATES = [
    (["customer_id"], "contract", "key determines every attribute"),
    (["zip_code"], "city", "a zip code belongs to one city"),
    (["zip_code"], "population", "population is stored per zip code"),
    (["zip_code"], "latitude", "location of a zip code"),
    (["age"], "senior_citizen", "senior means age 65 or more"),
    (["age"], "under_30", "derived from age"),
    (["number_of_referrals"], "referred_a_friend", "referred means at least one referral"),
    (["internet_service"], "internet_type", "fails: source 2 splits DSL and Fiber into finer types incl. Cable"),
    (["internet_service"], "has_internet", "same fact stored twice across sources"),
    (["phone_service"], "multiple_lines", "fails overall, holds when phone_service = 0 (see conditional)"),
    (["number_of_dependents"], "dependents", "fails: source 2 count vs source 1 flag disagree on 551 customers"),
    (["city"], "zip_code", "expected to fail: a city has many zip codes"),
]


def check(df, lhs, rhs):
    counts = df.groupby(lhs, dropna=False)[rhs].nunique(dropna=False)
    broken = counts[counts > 1]
    rows_in_broken = df[df.set_index(lhs).index.isin(broken.index)].shape[0]
    return len(counts), len(broken), rows_in_broken


def conditional_checks(df):
    """Dependencies that only hold for part of the table."""
    out = []
    no_net = df[df["internet_service"] == "No"]
    for col in ["online_security", "online_backup", "device_protection", "tech_support",
                "streaming_tv", "streaming_movies"]:
        ok = no_net[col].eq("No internet service").all()
        out.append({"dependency": f"internet_service='No' -> {col}='No internet service'",
                    "rows": len(no_net), "holds": ok})
    out.append({"dependency": "internet_service='No' -> avg_monthly_gb_download=0",
                "rows": len(no_net), "holds": no_net["avg_monthly_gb_download"].eq(0).all()})
    no_phone = df[df["phone_service"] == 0]
    out.append({"dependency": "phone_service=0 -> avg_monthly_long_distance_charges=0",
                "rows": len(no_phone),
                "holds": no_phone["avg_monthly_long_distance_charges"].eq(0).all()})

    # Arithmetic dependency inside source 2.
    revenue = (df["total_charges_s2"] + df["total_extra_data_charges"]
               + df["total_long_distance_charges"] - df["total_refunds"])
    out.append({"dependency": "total_revenue = total_charges + extra_data + long_distance - refunds",
                "rows": len(df), "holds": (revenue - df["total_revenue"]).abs().lt(0.01).all()})
    return pd.DataFrame(out)


def run(df=None):
    if df is None:
        df = lake.read("silver", "customers_integrated")

    rows = []
    for lhs, rhs, note in CANDIDATES:
        groups, broken, affected = check(df, lhs, rhs)
        rows.append({"dependency": f"{', '.join(lhs)} -> {rhs}", "lhs_groups": groups,
                     "violating_groups": broken, "rows_affected": affected,
                     "holds": broken == 0, "note": note})
    fds = pd.DataFrame(rows)
    cond = conditional_checks(df)

    print(fds.to_string(index=False))
    print()
    print(cond.to_string(index=False))
    fds.to_csv(TABLES / "functional_dependencies.csv", index=False)
    cond.to_csv(TABLES / "conditional_dependencies.csv", index=False)
    return fds, cond


if __name__ == "__main__":
    run()
