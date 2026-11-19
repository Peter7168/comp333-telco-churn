"""Split the IBM CSV into customer_profiles, service_subscriptions and account_billing."""
import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.config import GITHUB_DIR

PROFILE_COLS = ["customerID", "gender", "SeniorCitizen", "Partner", "Dependents", "tenure"]
SERVICE_COLS = ["customerID", "PhoneService", "MultipleLines", "InternetService",
                "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport",
                "StreamingTV", "StreamingMovies"]
BILLING_COLS = ["customerID", "Contract", "PaperlessBilling", "PaymentMethod",
                "MonthlyCharges", "TotalCharges", "Churn"]


def main():
    # Read as text to keep TotalCharges blanks.
    flat = pd.read_csv(GITHUB_DIR / "Telco-Customer-Churn.csv", dtype=str, keep_default_na=False)

    tables = {
        "customer_profiles.csv": PROFILE_COLS,
        "service_subscriptions.csv": SERVICE_COLS,
        "account_billing.csv": BILLING_COLS,
    }
    for name, cols in tables.items():
        out = GITHUB_DIR / name
        flat[cols].to_csv(out, index=False)
        print(f"wrote {name}: {len(flat)} rows, {len(cols)} columns, "
              f"{out.stat().st_size / 1_000_000:.3f} MB")


if __name__ == "__main__":
    main()
