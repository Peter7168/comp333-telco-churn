# COMP 333 – Telco Customer Churn (Stats and Furious)

We predict which telecom customers will cancel (churn), a **binary classification** task.
The project covers data collection, integration of two sources, cleaning,
transformation, ETL feature building, a Delta Lake store, a model and a dashboard.

## Data

| Source | Format | Files | Rows | Notes |
|---|---|---|---|---|
| IBM Telco sample on GitHub ([IBM/telco-customer-churn-on-icp4d](https://github.com/IBM/telco-customer-churn-on-icp4d)) | CSV | 3 (our Phase 1 split) | 7,043 each | customer profile, services, billing + churn label |
| IBM Cognos Analytics 11.1.3 "Telco customer churn" module ([IBM docs](https://community.ibm.com/community/user/businessanalytics/blogs/steven-macko/2019/07/11/telco-customer-churn-1113)) | XLSX | 5 | 7,043 (population: 1,671 zip codes) | demographics, location, zip population, services usage, status |

8 files, 3.69 MB in total. All 7,043 customer IDs match across both sources.
`reports/tables/dataset_stats.csv` has the per-file counts.

## How to run

```bash
pip install -r requirements.txt
python scripts/split_phase1.py        # rebuilds the 3 Phase 1 CSVs (already in the repo)
python -m src.pipeline                # full run, about 8 minutes
python -m src.pipeline --skip-experiments   # quick run, under 1 minute
streamlit run dashboard/app.py        # the dashboard
```

If `deltalake` is not installed, the lake falls back to CSV files in the same
folders and everything else still works.

## Pipeline

```
raw files ──ingest──> bronze (8 tables, as-is)
          ──integrate──> silver/customers_integrated (schema mapping, harmonized values, joins)
          ──etl──> silver/customers_clean ──> gold/customer_features ──> SQLite warehouse ──> dashboard
          ──experiments──> F1 with / without every cleaning and transformation step
          ──train──> models/churn_model.joblib
```

| File | What it does |
|---|---|
| `src/ingest.py` | Loads the 3 CSV + 5 XLSX files into the bronze layer, writes dataset stats |
| `src/integrate.py` | Schema mapping (74 columns -> one unified schema), value harmonization, joins, conflict report |
| `src/dependencies.py` | Checks 12 candidate functional dependencies + 9 conditional ones |
| `src/clean.py` | 6 null strategies, 5 outlier strategies, IQR / z-score / Isolation Forest detection |
| `src/transform.py` | 6 scalers (incl. decimal scaling), 4 normalizers, 5 encoders |
| `src/features.py` | Column groups and the ETL features (add-on count, charge per service, tenure band, ...) |
| `src/etl.py` | Builds the clean silver and gold tables, mutual information feature ranking |
| `src/experiments.py` | 5-fold CV for every setup; one setting changed at a time |
| `src/train.py` / `src/predict.py` | Final model, hold-out evaluation, saved model, scoring |
| `src/lake.py` | Delta Lake bronze / silver / gold layers with time travel, SQLite warehouse |
| `src/figures.py` | Slide charts in `reports/figures` |
| `dashboard/app.py` | Streamlit dashboard with filters, map, data quality, results, live prediction |

## Main results (5-fold CV, churn-class F1)

| Stage | F1 | ROC-AUC |
|---|---|---|
| 1. Raw source 1 (drop null rows, no scaling, integer codes) | 0.626 | 0.844 |
| 2. + cleaning | 0.626 | 0.844 |
| 3. + transformation | 0.632 | 0.849 |
| 4. + integration of source 2 | 0.679 | 0.891 |
| 5. + ETL features | 0.680 | 0.890 |
| Final model: gradient boosting on stage 5 | **0.703** | **0.902** |

What the experiments show:

- **Integration** gives the biggest gain. Source 2 adds referrals, dependants count, age and location.
- **Cleaning** barely changes F1 on the real data: only 11 cells were blank. On a copy with
  simulated damage (10% blanks, 3% typos), F1 falls from 0.626 to 0.609. Imputing and IQR
  capping bring it back to 0.619. Dropping rows instead throws away 26% of the test customers.
- **Dropping rows with nulls** looks best on the real data (0.701), but only because it deletes
  65% of the test set (every customer without an offer). Those customers can't be scored at all.
- **Scaling** barely matters for logistic regression but matters a lot for KNN (0.504 -> 0.612).
- **One-hot encoding** beats ordinal, frequency and target encoding, and all of them beat
  dropping the categorical columns.

All numbers come from `reports/tables/`. The pipeline rewrites them on every run.

## Integration findings

- Payment method uses 4 categories in source 1 and 3 in source 2, and they don't map one to one.
  For example, "Mailed check" in source 1 is "Credit Card" in source 2 for 1,227 customers. We keep source 1.
- Contract disagrees for 265 customers and dependents for 551. Source 1 wins for shared attributes.
- The 11 blank `TotalCharges` in source 1 have values in source 2, which our `cross_source` null strategy uses.
- `Satisfaction Score`, `Churn Score`, `Customer Status`, `Churn Category` and `Churn Reason` leak the
  label (e.g. satisfaction 1–2 is always churn), so they are excluded from the features.
