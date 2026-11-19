# COMP 333 Project: Telco Customer Churn

**Team Stats and Furious:** Mohammed Al-Nabulsi, Mehran Bordbar, Marc El Haddad, Samer Hasna,
Peter Samaha, Boudy Joe Samaha, Georges Ojeil

We predict which telecom customers will cancel their service (binary classification). The
project integrates two data sources, cleans and transforms the data, stores it in a Delta
Lake, trains a model and serves it in a dashboard.

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Peter7168/comp333-telco-churn/blob/main/notebooks/COMP333_Telco_Churn_Colab.ipynb)

**Tools:** Google Colab (data science platform), pandas, scikit-learn and Feature-engine (data
preparation and ETL), Delta Lake (data lake), SQLite, Streamlit (dashboard and deployed model),
Power BI (report on the exported star schema), matplotlib.

**Start with the report:** [`report/COMP333_Final_Report.docx`](report/COMP333_Final_Report.docx).

## What is in this repository

| Folder | Content |
|---|---|
| `report/` | The final report (Word) |
| `data/raw/ibm_github/` | Source 1: the IBM Telco CSV and the three tables we split it into |
| `data/raw/ibm_cognos/` | Source 2: the five Excel files of the IBM Cognos 11.1.3 Telco module |
| `src/` | The pipeline, one Python file per step |
| `notebooks/` | The Google Colab notebook that runs the whole project step by step |
| `dashboard/` | The Streamlit dashboard, with a form that scores a new customer |
| `results/tables/` | Every table of results as CSV (mapping, conflicts, experiments, ...) |
| `results/figures/` | The figures used in the report |
| `results/powerbi/` | Star-schema CSVs for Power BI and a guide to build the report |
| `tests/` | Checks on the lake, the model and the dashboard |
| `scripts/` | Rebuilds the three Phase 1 tables from the original CSV |

## Running on Google Colab (no installation)

Click the **Open in Colab** badge above, then **Runtime → Run all**. The first cell downloads
the repository; because the repository is private, it asks for a GitHub token with read
access (GitHub → Settings → Developer settings → Personal access tokens). The full run takes
about 10 minutes; set `RUN_EXPERIMENTS = False` in the first cell to reuse our saved experiment
results (about 2 minutes).

## Running on your own computer

### Installation

You need **Python 3.11 or newer** and **Git**.

```bash
git clone https://github.com/Peter7168/comp333-telco-churn.git
cd comp333-telco-churn
```

Create a virtual environment and install the libraries.

Windows (PowerShell):

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Mac or Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Running

**1. Run the pipeline** (about 6 minutes):

```bash
python -m src.pipeline
```

The first line printed should be `storage: Delta Lake`. The pipeline then:

- loads the 8 raw files into the bronze layer of the lake (`data/lake/`)
- integrates the two sources and writes the silver tables
- checks entity resolution and functional dependencies
- cleans the data, builds features and writes the gold table
- runs every experiment (with and without each cleaning and transformation step)
- trains the final model and saves it to `models/churn_model.joblib`
- compares three Feature-engine transformers with our own steps
- rewrites `results/tables/` and `results/figures/`, and exports the Power BI tables

For a quick run that skips the experiments (about 1 minute) and uses the experiment tables
already in `results/tables/`:

```bash
python -m src.pipeline --skip-experiments
```

**2. Open the dashboard:**

```bash
streamlit run dashboard/app.py
```

It opens in the browser at http://localhost:8501. Stop it with Ctrl+C.

**3. Power BI** (Windows): open Power BI Desktop and follow
[`results/powerbi/POWERBI_GUIDE.md`](results/powerbi/POWERBI_GUIDE.md) to load the exported
tables and build the report.

**4. Run the checks** (optional):

```bash
pip install pytest
pytest -q tests
```

## The pipeline, file by file

| File | What it does |
|---|---|
| `src/ingest.py` | Loads the 3 CSV and 5 XLSX files into the bronze layer |
| `src/integrate.py` | Schema mapping, value mapping, joins, conflict report |
| `src/entity_resolution.py` | Matches the two sources without the customer ID, as a check on the join |
| `src/dependencies.py` | Functional dependency checks |
| `src/clean.py` | Null strategies and outlier strategies |
| `src/transform.py` | Normalization, scaling and encoding |
| `src/fe_tools.py` | Feature-engine: Winsorizer, RareLabelEncoder, DropCorrelatedFeatures |
| `src/features.py`, `src/etl.py` | Engineered features, clean silver table, gold table, feature ranking |
| `src/experiments.py` | Locks a 20% test set, then 5-fold cross-validation for every option |
| `src/train.py`, `src/predict.py` | Final model, test-set evaluation, saved model, scoring |
| `src/lake.py` | Delta Lake layers (bronze, silver, gold) and the SQLite copy for the dashboard |
| `src/figures.py` | Figures for the report |
| `src/powerbi.py` | Exports the gold table as a star schema for Power BI |
| `src/pipeline.py` | Runs all of the above in order |

## Main result

On a test set of 1,409 customers that no experiment used, the final model (gradient boosting)
catches 288 of the 374 customers who left. F1 is 0.695 and ROC-AUC is 0.900.

| Stage (logistic regression, 5-fold CV) | F1 |
|---|---|
| 1. Source 1 only, minimal preparation | 0.626 |
| 2. + cleaning | 0.626 |
| 3. + transformation | 0.632 |
| 4. + integration of source 2 | 0.682 |
| 5. + engineered features | 0.679 |

The report explains each step and discusses these numbers.

## Data sources

- **Source 1:** IBM Telco Customer Churn sample,
  https://github.com/IBM/telco-customer-churn-on-icp4d
- **Source 2:** IBM Cognos Analytics 11.1.3 "Telco customer churn" sample module,
  https://community.ibm.com/community/user/businessanalytics/blogs/steven-macko/2019/07/11/telco-customer-churn-1113

## Troubleshooting

- **`python` is not recognized (Windows):** reinstall Python and tick "Add Python to PATH".
- **"running scripts is disabled" when activating (Windows):** run
  `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once, then activate again.
- **The first line says `CSV fallback`:** the `deltalake` package did not install. Run
  `pip install deltalake pyarrow`. The pipeline still works without it, storing CSV files.
