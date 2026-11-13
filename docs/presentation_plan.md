# Presentation plan (10 minutes, PDF due Dec 17)

The slides follow the prof's order. Each line says which rubric item the slide covers,
which figure or table to use, and roughly how long to spend on it.

| # | Section | Content | Evidence | Rubric | Time |
|---|---|---|---|---|---|
| 1 | Title | Team, topic | | | 0:15 |
| 2 | Topic & ML task | Binary classification of churn, 26.5% positive class, F1 + ROC-AUC because of the imbalance | | ML task (5) | 0:45 |
| 3 | Data collection | 2 IBM sources, 8 files (3 CSV + 5 XLSX), 3.69 MB, stats table | `dataset_stats.csv` | 3+ files (5) | 0:45 |
| 4 | Storage | Delta Lake bronze / silver / gold + SQLite warehouse, show Delta version history | `src/lake.py` | storage systems (5) | 0:30 |
| 5 | Integration: schema | Mapping table excerpt (e.g. `Tenure in Months` -> `tenure`, Yes/No -> 1/0), ERD with the N:1 zip join | `schema_mapping.csv`, `docs/ERD.md` | schema integration (5), mapping & ER (10) | 0:50 |
| 6 | Integration: mapping & ER | Payment-method crosstab (no 1:1 mapping), contract 265 conflicts, source 1 wins; entity resolution without the ID (6,931 of 7,043 matched, 100% precision, 112 look-alikes); functional dependencies | `payment_method_crosswalk.csv`, `entity_resolution.csv`, `functional_dependencies.csv` | mapping & ER (10) | 0:50 |
| 7 | Cleaning: nulls | 3 null columns; structural nulls -> "None"; 11 blank totals -> 0 (tenure 0, not billed; source 2's value rejected because it assumes tenure 10); 6 strategies compared; why drop_rows is misleading | `02_nulls.png`, `null_fill_examples.csv` | DC nulls (10) | 0:50 |
| 8 | Cleaning: outliers | Boxplots, IQR vs z-score vs Isolation Forest, zero-inflated columns; stress test shows cleaning recovering F1 | `06_outlier_boxplots.png`, `04_stress_test.png` | DC outliers (10) | 0:50 |
| 9 | Transformation | Normalization (log / Yeo-Johnson / quantile), 6 scalers (KNN chart), 5 encoders | `03_scalers.png`, `02_normalizers.png`, `02_encoders.png` | DT (20) | 1:00 |
| 10 | ETL & features | Engineered features + mutual information ranking + feature selection (top 10 = 0.63, all 47 = 0.68) + permutation importance | `feature_ranking.csv`, `02_feature_selection.png`, `07_feature_importance.png` | ETL to find features (8) | 0:40 |
| 11 | Results | Stage-by-stage F1 chart, final model F1 0.70 in CV, 0.695 on the locked test set (AUC 0.900) | `01_stages_f1.png`, `02_models.png` | with/without DC & DT | 0:40 |
| 12 | Dashboard demo | Filters -> map -> churn drivers -> predict a customer live | `streamlit run dashboard/app.py` | dashboard (10), tools (2) | 1:30 |
| 13 | Tools & wrap-up | Python, pandas, scikit-learn, Delta Lake, SQLite, Streamlit, matplotlib, GitHub | | tools (2) | 0:15 |

Total: about 9:50.

## Demo script (90 seconds)

1. Overview tab: 7,043 customers, 26.5% churn, map of California (red = churned).
2. Filter Contract = Month-to-month: the churn rate jumps to about 43%.
3. Model results tab: pick "nulls", point out that drop_rows was scored on fewer test rows.
4. Predict tab: month-to-month, 2 months, fiber, 0 referrals -> high risk (about 80%). Change it to a
   two-year contract with 3 referrals -> the probability falls sharply.

Keep screenshots of each step in the slides in case the live demo fails.
