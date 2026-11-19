# Building the Power BI report

The CSV files in this folder are produced by `python -m src.powerbi` (the full pipeline runs it
too). They form a star schema: one fact table with one row per customer, and small dimension
tables that describe it. Power BI Desktop is free and runs on Windows.

## 1. Load the data

1. Open Power BI Desktop → **Get data** → **Text/CSV**.
2. Load each of the seven files in this folder: `fact_customers`, `dim_contract`,
   `dim_payment`, `dim_internet`, `dim_location`, `dim_tenure_band`, `model_experiments`.
   Click **Load** each time (no transformation needed).
3. In **Model view**, check these relationships (Power BI usually creates them itself; drag
   the column from the dimension onto the fact table if one is missing). All are
   one-to-many, from the dimension (one) to `fact_customers` (many):

| Dimension column | Fact column |
|---|---|
| `dim_contract[contract]` | `fact_customers[contract]` |
| `dim_payment[payment_method]` | `fact_customers[payment_method]` |
| `dim_internet[internet_key]` | `fact_customers[internet_key]` |
| `dim_location[zip_code]` | `fact_customers[zip_code]` |
| `dim_tenure_band[tenure_band]` | `fact_customers[tenure_band]` |

`model_experiments` stays on its own (no relationship).

4. Two settings:
   * Select `dim_tenure_band[tenure_band]` → **Column tools** → **Sort by column** →
     `sort_order`, so bands appear in order.
   * Select `dim_contract[contract]` → **Sort by column** → `contract_months`.
   * Select `dim_location[latitude]` and set **Data category** to Latitude;
     `dim_location[longitude]` to Longitude.

## 2. Measures

In `fact_customers`, **New measure**, one at a time:

```DAX
Customers = COUNTROWS ( fact_customers )

Churned = SUM ( fact_customers[churn] )

Churn Rate = DIVIDE ( [Churned], [Customers] )

Avg Monthly Charge = AVERAGE ( fact_customers[monthly_charges] )

Monthly Revenue Lost = CALCULATE ( SUM ( fact_customers[monthly_charges] ), fact_customers[churn] = 1 )

Active High Risk =
CALCULATE (
    [Customers],
    fact_customers[churn] = 0,
    fact_customers[risk_level] = "High"
)

Avg Predicted Risk = AVERAGE ( fact_customers[churn_probability] )
```

Format `Churn Rate` and `Avg Predicted Risk` as **Percentage** and the two charge measures as
**Currency**.

## 3. Report pages

**Page 1: Overview**

* Four **Card** visuals: `Customers`, `Churn Rate`, `Monthly Revenue Lost`, `Active High Risk`.
* **Slicers** across the top: `dim_contract[contract]`, `dim_internet[internet_service]`,
  `dim_tenure_band[tenure_band]`.
* **Map**: Latitude and Longitude from `dim_location`, Bubble size = `Customers`,
  Tooltips = `Churn Rate`.

**Page 2: Churn drivers**

* **Clustered bar chart**: Y-axis `dim_contract[contract]`, X-axis `Churn Rate`.
* **Clustered bar chart**: Y-axis `dim_payment[payment_method]`, X-axis `Churn Rate`.
* **Clustered column chart**: X-axis `dim_tenure_band[tenure_band]`, Y-axis `Churn Rate`.
* **Clustered column chart**: X-axis `fact_customers[number_of_referrals]`, Y-axis `Churn Rate`.
* **Scatter chart**: X `fact_customers[tenure]`, Y `fact_customers[monthly_charges]`,
  Legend `fact_customers[churn]` (set **Values** to "Don't summarize" for both axes, or put
  `customer_id` in **Values**).

**Page 3: Model**

* **Slicer**: `model_experiments[experiment]` (single select).
* **Clustered bar chart**: Y-axis `model_experiments[option]`, X-axis `model_experiments[f1]`
  (Sum). Choosing "stages" shows performance with and without cleaning, transformation,
  integration and features; "nulls", "outliers", "scalers" and "encoders" show each option.
* **Table**: `fact_customers[customer_id]`, `dim_location[city]`, `fact_customers[contract]`,
  `fact_customers[tenure]`, `fact_customers[churn_probability]`, filtered to
  `churn = 0` and `risk_level = High`, sorted by probability. These are the customers to call.

## 4. Save

**File → Save as** `results/powerbi/COMP333_Churn.pbix`. For the slides, **File → Export →
Export to PDF** gives one page per report page.

## Column notes

* `churn`: 1 = left, 0 = stayed.
* `churn_probability`: the model's risk score from a model that never saw that customer
  (5-fold out-of-fold prediction).
* `risk_level`: Low below 30%, Medium 30–60%, High above 60%.
* `internet_key`: internet service and type together, because source 2 adds Cable.
