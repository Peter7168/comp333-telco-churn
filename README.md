# Telco Customer Churn

**COMP 333 · Team Stats and Furious**
Mohammed Al-Nabulsi, Mehran Bordbar, Marc El Haddad, Samer Hasna, Peter Samaha, Boudy Joe Samaha, Georges Ojeil

We predict whether a telecom customer will cancel their service (**binary classification**),
using two IBM data sources about the same 7,043 customers.

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Peter7168/comp333-telco-churn/blob/main/notebooks/COMP333_Telco_Churn_Colab.ipynb)

## Repository

| Folder | Contents |
|---|---|
| `presentation/` | Slides |
| `notebooks/` | Google Colab notebook that runs everything step by step |
| `src/` | Pipeline code, one file per step |
| `dashboard/` | Streamlit dashboard with live predictions |
| `data/raw/` | The 8 source files (3 CSV + 5 Excel) |
| `results/tables/` | All results as CSV |
| `results/figures/` | Charts |
| `tests/` | Automated checks, run on GitHub on every push |

**Tools:** Google Colab, pandas, scikit-learn, Feature-engine, Delta Lake, SQLite, Streamlit.

## How to run

### Option 1: Google Colab (nothing to install)

Click **Open in Colab** above, then **Runtime → Run all**.
The repository is private, so the first cell asks for a GitHub token with read access.

### Option 2: Your computer

Requires Python 3.11+ and Git.

```bash
git clone https://github.com/Peter7168/comp333-telco-churn.git
cd comp333-telco-churn
python -m venv .venv
```

Activate the environment:

* Windows: `.venv\Scripts\activate`
* Mac / Linux: `source .venv/bin/activate`

Then:

```bash
pip install -r requirements.txt
python -m src.pipeline           # runs everything (about 6 min)
streamlit run dashboard/app.py   # opens the dashboard at http://localhost:8501
```

`python -m src.pipeline --skip-experiments` is a 1-minute version that reuses the saved
experiment results.

## Pipeline

| Step | File | Output |
|---|---|---|
| Load the 8 files | `src/ingest.py` | Delta Lake bronze layer |
| Integrate both sources | `src/integrate.py`, `src/entity_resolution.py`, `src/dependencies.py` | silver layer |
| Clean | `src/clean.py` | nulls filled, outliers capped |
| Transform | `src/transform.py`, `src/fe_tools.py` | normalization, scaling, encoding |
| Build features (ETL) | `src/features.py`, `src/etl.py` | gold layer |
| Compare every option | `src/experiments.py` | F1 with and without each step |
| Train and save the model | `src/train.py`, `src/predict.py` | `models/churn_model.joblib` |
| Charts | `src/figures.py` | `results/figures/` |

## Results

| Stage (logistic regression, 5-fold CV) | F1 |
|---|---|
| Raw source 1 | 0.626 |
| + cleaning | 0.626 |
| + transformation | 0.632 |
| + second source | 0.682 |
| + engineered features | 0.679 |

**Final model:** gradient boosting. On 1,409 test customers held out from the start, it
catches 288 of the 374 who left (77%). F1 0.695, ROC-AUC 0.90.

## Data sources

* IBM Telco Customer Churn (CSV): https://github.com/IBM/telco-customer-churn-on-icp4d
* IBM Cognos Analytics 11.1.3 Telco customer churn module (Excel):
  https://community.ibm.com/community/user/businessanalytics/blogs/steven-macko/2019/07/11/telco-customer-churn-1113

## Troubleshooting

* **`python` not recognized (Windows):** reinstall Python with "Add Python to PATH" ticked.
* **"running scripts is disabled" (Windows):** run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`, then activate again.
* **First line says `CSV fallback`:** run `pip install deltalake pyarrow`.
