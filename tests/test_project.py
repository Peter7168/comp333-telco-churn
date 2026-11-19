"""Checks that the built project is sound. Run after the pipeline:  pytest -q tests

On GitHub (CI=true) the Delta Lake and Streamlit tests must pass. On a machine
without those packages they are skipped, since the pipeline falls back to CSV.
"""
import os
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))

from src import lake  # noqa: E402
from src.config import LEAKY_COLUMNS, TABLES  # noqa: E402
from src.predict import load, predict  # noqa: E402

ON_CI = os.environ.get("CI") == "true"


def test_lake_is_delta_with_history_and_time_travel():
    if not lake.HAS_DELTA and not ON_CI:
        pytest.skip("deltalake not installed here")
    assert lake.HAS_DELTA, "the lake must be Delta Lake, not the CSV fallback"
    # train.py rewrites the gold table with risk scores, so it has at least 2 versions.
    history = lake.history("gold", "customer_features")
    assert len(history) >= 2
    first = lake.read("gold", "customer_features", version=0)
    latest = lake.read("gold", "customer_features")
    assert "churn_probability" not in first.columns
    assert "churn_probability" in latest.columns


def test_every_layer_has_its_tables():
    for name in ["github_customer_profiles", "github_service_subscriptions",
                 "github_account_billing", "cognos_demographics", "cognos_location",
                 "cognos_population", "cognos_services", "cognos_status"]:
        assert len(lake.read("bronze", name)) > 0
    assert len(lake.read("silver", "customers_integrated")) == 7043
    assert len(lake.read("silver", "customers_clean")) == 7043


def test_gold_table_is_one_clean_row_per_customer():
    gold = lake.read("gold", "customer_features")
    assert len(gold) == 7043
    assert gold["customer_id"].is_unique
    assert not gold.isna().any().any(), "gold must have no missing values"
    assert not set(LEAKY_COLUMNS) & set(gold.columns), "leaky columns must not reach gold"


def test_new_customers_are_consistent():
    # The 11 customers with a blank total have tenure 0, so their total must be 0.
    gold = lake.read("gold", "customer_features")
    new = gold[gold["tenure"] == 0]
    assert len(new) == 11
    assert (new["total_charges"] == 0).all()


def test_model_does_not_use_leaky_columns_and_predicts():
    bundle = load()
    assert not set(LEAKY_COLUMNS) & set(bundle["columns"])
    sample = lake.read("silver", "customers_integrated").head(50)
    probs = predict(sample, bundle)
    assert len(probs) == 50
    assert ((probs >= 0) & (probs <= 1)).all()


def test_result_tables_exist():
    for name in ["dataset_stats", "schema_mapping", "integration_conflicts", "entity_resolution",
                 "functional_dependencies", "null_report", "outlier_report", "feature_ranking",
                 "exp_stages", "exp_nulls", "exp_outliers", "exp_scalers", "exp_normalizers",
                 "exp_encoders", "exp_feature_selection", "exp_models", "exp_stress",
                 "final_model_metrics"]:
        assert len(pd.read_csv(TABLES / f"{name}.csv")) > 0, name


def test_feature_engine_results():
    if not ON_CI:
        pytest.importorskip("feature_engine")
    capped = pd.read_csv(TABLES / "feature_engine_winsorizer.csv")
    # Feature-engine's IQR caps must be the same as ours.
    assert capped["max_difference_vs_ours"].abs().max() < 1e-6
    assert len(pd.read_csv(TABLES / "exp_feature_engine.csv")) == 4


def test_powerbi_export_links_up():
    out = ROOT / "results" / "powerbi"
    fact = pd.read_csv(out / "fact_customers.csv", keep_default_na=False)
    assert len(fact) == 7043
    for dim, key in [("dim_contract", "contract"), ("dim_payment", "payment_method"),
                     ("dim_internet", "internet_key"), ("dim_location", "zip_code"),
                     ("dim_tenure_band", "tenure_band")]:
        table = pd.read_csv(out / f"{dim}.csv", keep_default_na=False)
        assert table[key].is_unique, dim
        assert fact[key].isin(table[key]).all(), dim


def test_dashboard_loads_and_predicts():
    if ON_CI:
        from streamlit.testing.v1 import AppTest
    else:
        AppTest = pytest.importorskip("streamlit.testing.v1").AppTest
    app = AppTest.from_file(str(ROOT / "dashboard" / "app.py"), default_timeout=180).run()
    assert not app.exception, [e.value for e in app.exception]

    # Pressing the form button runs the deployed model on the default customer.
    submit = next(b for b in app.button if b.label == "Predict churn")
    app = submit.click().run()
    assert not app.exception, [e.value for e in app.exception]
    assert any(m.label == "Churn probability" for m in app.metric)
