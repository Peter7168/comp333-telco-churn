# Project review against the prof's rubric

Date: Nov 13, 2026. This is our own estimate of where we stand, not the prof's grade.
Scores are for the state of the repo **before** the fixes in the to-do list below.

## What the prof asked for

- Define the topic and the ML task.
- Datasets from different data sources, at least three files.
- A data lake platform (Delta Lake, AWS Glue Data Catalog, Azure Data Catalog).
- Solutions for: data integration, data cleaning and transformation, visualization and ETL,
  modelling and deploying the model with an application.
- Show model performance **with and without** data cleaning (DC) and data transformation (DT).
- 10-minute presentation with a demo, in this order: topic and ML task, data collection,
  integration, cleaning, transformation, visualization and ETL. At least two presenters.
  Slides PDF due Dec 17.

## Section-by-section grade

| Rubric item | Points | Our estimate | Why |
|---|---|---|---|
| ML task defined | 5 | 5 | Binary churn classification, metric choice justified |
| At least three files | 5 | 5 | 8 files (3 CSV + 5 XLSX). Risk: both sources are IBM and describe the same customers, see Q&A |
| Storage systems | 5 | 3 | Delta Lake code exists but has never run with Delta installed. Only the CSV fallback was tested |
| Schema integration | 5 | 5 | 74 columns mapped to one schema, value harmonization, conflict report |
| Data mapping and ER | 10 | 6 | Mapping is solid. Entity resolution is only an exact join on customer ID; nothing shows ER as a technique |
| DC on null values | 10 | 8 | 6 strategies with F1 with/without. But the chosen fill creates 11 inconsistent rows (see to-do 1) |
| DC for outliers | 10 | 9 | IQR, z-score, Isolation Forest; 5 strategies with F1 with/without; stress test |
| Data normalization | 5 | 5 | Min-max, z-score, decimal scaling, log, Yeo-Johnson, quantile |
| Different scaling functions | 5 | 5 | 6 scalers on 2 models, KNN shows the effect clearly |
| Data encoding | 10 | 10 | Binary, one-hot, ordinal, frequency, target; F1 for each |
| Use of different tools | 2 | 2 | pandas, scikit-learn, Delta Lake, SQLite, Streamlit, matplotlib |
| ETL to find features | 8 | 6 | Features are built and ranked, but nothing tests which features the model needs |
| Dashboard | 10 | 6 | Written but never opened in Streamlit. One call we use is deprecated and may crash |
| Presentation (structure, clarity, time) | 10 | not started | Plan and figures exist, no slides yet |
| **Technical total** | **90** | **75** | Target after the to-do list: 86 to 88 |

## Things that are right and should not change

- The 20% test set is locked before any experiment. All choices use 5-fold CV on the other 80%.
- Cleaning and transformation are fit on training folds only, then applied to the held-out fold.
- Label-leaking columns from source 2 (satisfaction score, churn score, churn reason, ...) are excluded.
- "Drop rows with nulls" is flagged as misleading rather than presented as the winner.
- Cleaning shows almost no gain on the real data. That is the honest result; the stress test
  shows what cleaning is worth when data is dirty.

## To-do list

Fixes Claude can do in the repo:

- [ ] **1. Null fill makes 11 inconsistent rows.** For the 11 customers with a blank total,
  source 1 says tenure 0 and source 2 says tenure 10 with a total of 10 months of charges.
  We kept source 1's tenure (0) but copied source 2's total, e.g. customer 4472-LVYGI ends
  up with tenure 0 and total charges 525.50, and `avg_charge_per_month` = 525.50. Fix: use
  the Phase 1 rule (0.0, not billed yet) as the final choice, which matches tenure 0 and the
  Phase 1 report. Keep the cross-source fill as a compared strategy and explain why we rejected it.
- [ ] **2. Entity resolution (10-point item).** Add an ER step that links the two sources
  *without* the customer ID (blocking + attribute similarity), scores it against the true
  IDs, and checks for duplicate customers inside each source.
- [ ] **3. Feature selection experiment (ETL to find features).** Rank features on the
  training fold and measure F1 with only the top 5, 10, 15, 20, 30 features vs all.
- [ ] **4. Dashboard robustness.** Remove the deprecated `use_container_width` argument.
- [ ] **5. Prove Delta Lake and the dashboard run.** Add a GitHub Actions check that installs
  the real requirements, runs the pipeline, asserts the lake is Delta (with version history
  and time travel) and loads the Streamlit app headlessly, including a prediction.
- [ ] **6. Refresh numbers** in README, explainer PDF and presentation plan after the rerun.

Needs the team:

- [ ] **7. Slides.** Build the deck from `docs/presentation_plan.md`, decide who presents (at least two).
- [ ] **8. Rehearse to 10 minutes** with the live demo, and keep screenshots as a backup.
- [ ] **9. Ask the prof** whether "ER" in the rubric means entity resolution (we assume yes)
  and whether a written Phase 2 report is due with the slides.
- [ ] **10. Optional: a third, independent source** (for example US Census income by zip code)
  would make "different data sources" harder to challenge. Not required by the rubric.
- [ ] **11. Optional: host the dashboard** on Streamlit Community Cloud so the demo does not
  depend on one laptop.
