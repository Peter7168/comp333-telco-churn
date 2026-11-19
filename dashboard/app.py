"""Churn dashboard. Run: streamlit run dashboard/app.py"""
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))

from src import lake  # noqa: E402
from src.config import TABLES  # noqa: E402
from src.predict import load, predict  # noqa: E402

st.set_page_config(page_title="Telco churn | Stats and Furious", layout="wide")


@st.cache_data
def customers():
    return lake.query("SELECT * FROM customer_features")


@st.cache_data
def table(name):
    return pd.read_csv(TABLES / f"{name}.csv")


@st.cache_resource
def model():
    return load()


df = customers()
st.title("Telco customer churn")
st.caption("7,043 customers from 2 IBM sources, 8 files")

# Filters apply to every tab.
f1, f2, f3 = st.columns(3)
contracts = f1.multiselect("Contract", sorted(df["contract"].unique()))
internet = f2.multiselect("Internet type", sorted(df["internet_type"].unique()))
tenure = f3.multiselect("Tenure band", sorted(df["tenure_band"].unique()))
view = df
if contracts:
    view = view[view["contract"].isin(contracts)]
if internet:
    view = view[view["internet_type"].isin(internet)]
if tenure:
    view = view[view["tenure_band"].isin(tenure)]

overview, drivers, quality, results, live = st.tabs(
    ["Overview", "Churn drivers", "Data quality", "Model results", "Predict a customer"])

with overview:
    a, b, c, d = st.columns(4)
    a.metric("Customers", f"{len(view):,}")
    b.metric("Churn rate", f"{view['churn'].mean():.1%}")
    c.metric("Avg monthly charge", f"${view['monthly_charges'].mean():.2f}")
    at_risk = view[(view["churn"] == 0) & (view["churn_probability"] >= 0.5)]
    d.metric("Still active, high risk", f"{len(at_risk):,}",
             help="Customers who have not left yet but the model scores at 50% or more")

    st.subheader("Where the customers are")
    # Red = churned, blue = stayed.
    points = view[["latitude", "longitude", "churn"]].copy()
    points["color"] = points["churn"].map({1: "#e34948", 0: "#2a78d6"})
    st.map(points, latitude="latitude", longitude="longitude", color="color", size=40)
    st.caption("Red: churned · Blue: stayed")

    st.subheader("Highest-risk active customers")
    st.dataframe(at_risk.sort_values("churn_probability", ascending=False)
                 [["customer_id", "city", "contract", "tenure", "monthly_charges",
                   "number_of_referrals", "churn_probability"]].head(25),
                 hide_index=True)

with drivers:
    st.subheader("Churn rate by group")
    pick = st.selectbox("Group by", ["contract", "internet_type", "tenure_band",
                                     "payment_method", "offer", "addon_count",
                                     "number_of_referrals", "senior_citizen"])
    rates = (view.groupby(pick)["churn"].agg(["mean", "size"])
             .rename(columns={"mean": "churn_rate", "size": "customers"}))
    rates["churn_rate"] = (rates["churn_rate"] * 100).round(1)
    left, right = st.columns([2, 1])
    left.bar_chart(rates["churn_rate"], y_label="Churn rate (%)")
    right.dataframe(rates)

    st.subheader("Monthly charge vs tenure")
    st.scatter_chart(view.sample(min(len(view), 2000), random_state=1),
                     x="tenure", y="monthly_charges", color="churn")

    st.subheader("Feature ranking (mutual information with churn)")
    st.dataframe(table("feature_ranking"), hide_index=True)

with quality:
    st.subheader("Datasets")
    st.dataframe(table("dataset_stats"), hide_index=True)
    c1, c2 = st.columns(2)
    c1.subheader("Conflicts between the two sources")
    c1.dataframe(table("integration_conflicts"), hide_index=True)
    c2.subheader("Payment method: no one-to-one mapping")
    c2.dataframe(table("payment_method_crosswalk"), hide_index=True)
    st.subheader("Entity resolution: matching the sources without the customer ID")
    st.dataframe(table("entity_resolution"), hide_index=True)
    st.subheader("Functional dependencies")
    st.dataframe(table("functional_dependencies"), hide_index=True)
    st.dataframe(table("conditional_dependencies"), hide_index=True)
    c3, c4 = st.columns(2)
    c3.subheader("Nulls")
    c3.dataframe(table("null_report"), hide_index=True)
    c3.dataframe(table("null_fill_examples"), hide_index=True)
    c4.subheader("Outliers (IQR and z-score)")
    c4.dataframe(table("outlier_report"), hide_index=True)

with results:
    st.subheader("F1 as each pipeline stage is added")
    stages = table("exp_stages").set_index("stage")
    st.bar_chart(stages["f1"], y_label="F1", horizontal=True)

    name = st.selectbox("Experiment", ["nulls", "outliers", "normalizers", "scalers", "encoders",
                                       "feature_selection", "models", "stress"])
    exp = table(f"exp_{name}")
    label = exp.columns[1]
    st.bar_chart(exp.set_index(label)["f1"], y_label="F1", horizontal=True)
    st.dataframe(exp[[label, "f1", "f1_std", "accuracy", "precision", "recall", "roc_auc",
                      "train_rows", "test_rows", "n_features"]],
                 hide_index=True)
    if name == "nulls":
        st.info("drop_rows looks best only because it also deletes about two thirds of the "
                "rows it is graded on (every customer without an offer). It cannot score "
                "those customers at all.")

    st.subheader("Final model on a 20% hold-out set")
    st.dataframe(table("final_model_metrics"), hide_index=True)
    st.bar_chart(table("final_model_importance").head(12).set_index("feature")["f1_drop"],
                 horizontal=True, y_label="Drop in F1 when shuffled")

with live:
    st.subheader("Score a customer with the deployed model")
    st.caption("Fields you don't set keep the most common value in the data.")

    # Start from a typical customer; the form overrides fields.
    template = lake.read("silver", "customers_integrated")
    base = {c: (template[c].median() if pd.api.types.is_numeric_dtype(template[c])
                else template[c].mode()[0]) for c in template.columns}

    with st.form("customer"):
        c1, c2, c3 = st.columns(3)
        base["contract"] = c1.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        base["tenure"] = c1.slider("Tenure (months)", 0, 72, 12)
        base["monthly_charges"] = c1.slider("Monthly charge ($)", 18.0, 120.0, 70.0)
        base["internet_service"] = c2.selectbox("Internet", ["Fiber optic", "DSL", "No"])
        base["payment_method"] = c2.selectbox("Payment", sorted(template["payment_method"].unique()))
        base["number_of_referrals"] = c2.slider("Referrals", 0, 11, 0)
        base["age"] = c3.slider("Age", 19, 80, 45)
        base["number_of_dependents"] = c3.slider("Dependents", 0, 9, 0)
        base["dependents"] = int(base["number_of_dependents"] > 0)
        submitted = st.form_submit_button("Predict churn")

    if submitted:
        # Keep dependent fields consistent.
        no_net = base["internet_service"] == "No"
        base["internet_type"] = None if no_net else ("Fiber Optic" if base["internet_service"] == "Fiber optic" else "DSL")
        if no_net:
            for col in ["online_security", "online_backup", "device_protection", "tech_support",
                        "streaming_tv", "streaming_movies"]:
                base[col] = "No internet service"
            base["avg_monthly_gb_download"] = 0
        base["referred_a_friend"] = int(base["number_of_referrals"] > 0)
        base["total_charges"] = base["monthly_charges"] * base["tenure"]
        prob = predict(pd.DataFrame([base]), model())[0]
        st.metric("Churn probability", f"{prob:.0%}")
        st.progress(float(prob))
        st.write("High risk: consider a retention offer." if prob >= 0.5 else "Low risk.")
