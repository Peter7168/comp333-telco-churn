"""Paths and constants shared by every step of the pipeline."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

RAW = ROOT / "data" / "raw"
GITHUB_DIR = RAW / "ibm_github"      # source 1: IBM sample on GitHub (CSV)
COGNOS_DIR = RAW / "ibm_cognos"      # source 2: IBM Cognos 11.1.3 module (XLSX)

LAKE = ROOT / "data" / "lake"        # bronze / silver / gold layers live here
WAREHOUSE = ROOT / "data" / "warehouse.db"   # SQLite copy the dashboard reads

FIGURES = ROOT / "reports" / "figures"
TABLES = ROOT / "reports" / "tables"
MODELS = ROOT / "models"

TARGET = "churn"
SEED = 42

# I keep these out of the features because they are computed after the
# customer already churned (or they literally encode the answer).
LEAKY_COLUMNS = [
    "satisfaction_score",   # 1-2 is always churn, 4-5 is never churn
    "churn_score",
    "cltv",
    "customer_status",
    "churn_category",
    "churn_reason",
]

for folder in (LAKE, FIGURES, TABLES, MODELS):
    folder.mkdir(parents=True, exist_ok=True)
