# Questions to be ready for

Short answers the whole team should be able to give. Numbers come from `reports/tables/`
(run of Nov 13). If the pipeline is rerun, check them again.

## Topic and ML task

**1. What exactly are you predicting, and is it classification or regression?**
Whether a telecom customer cancels (churn: yes or no). Binary classification, because the
answer is one of two categories. Predicting a number, like next year's revenue, would be regression.

**2. Why F1 and not accuracy?**
Only 26.5% of customers churn. A model that always says "stays" is 73.5% accurate and catches
nobody. F1 only goes up if we catch leavers (recall) and our warnings are right (precision).

**3. How good is the final model?**
On 1,409 customers it never saw: it catches 288 of the 374 who left (recall 77%), and 63% of
its warnings are correct (precision). F1 0.695, ROC-AUC 0.900.

**4. Why is precision only 63%?**
We weight the churn class more so the model catches more leavers. That raises recall and
lowers precision. For a retention offer, missing a leaver costs more than one extra discount.
The 0.5 threshold can be moved if the business prefers fewer false alarms.

**5. Which model, and why that one?**
Gradient boosting (HistGradientBoostingClassifier): a chain of small decision trees where each
one corrects the previous ones. It had the best cross-validated F1 of the four we tried:
gradient boosting 0.696, logistic regression 0.679, random forest 0.657, KNN 0.608.

**6. Did you use a validation set? How do you know you are not overfitting?**
20% of customers were locked away as a test set before any experiment. Every choice was made
with 5-fold cross-validation on the other 80%, where each fold takes a turn as the validation
set. The test set was used once. Cross-validation F1 (0.696) and test F1 (0.695) agree.

## Data collection and storage

**7. Your two sources are both from IBM and describe the same customers. Are they really different sources?**
They are two separate releases: a CSV on IBM's GitHub and a five-file Excel module shipped
with Cognos Analytics 11.1.3. They have different schemas, different category values and they
disagree on some customers. Same entities in separately maintained systems is the normal
integration case in a company (billing vs CRM). It also gives us ground truth to measure the
integration against.

**8. Source 1 is one file that you split into three. Isn't that artificial?**
Yes, the split is ours, from Phase 1, to model three subjects (profile, services, billing).
The real integration work is between source 1 and the five source 2 files.

**9. How much data is it?**
8 files, 3.69 MB: 3 CSV and 5 XLSX. 7,043 customers, plus 1,671 zip codes in the population file.

**10. Which data lake did you use and what does it give you?**
Delta Lake, through the `deltalake` Python package (no Spark needed). Three layers: bronze
(raw files as-is), silver (integrated, then cleaned), gold (model-ready features). Every run
writes a new version, and we can read an old one back (time travel), e.g. the gold table
before risk scores were added. A SQLite copy of gold feeds the dashboard with SQL.

**11. Why not Spark or Databricks?**
The data is 3.69 MB. A cluster would add setup time and nothing else. The same Delta tables
could be read by Spark later without changes.

## Data integration

**12. How did you integrate the schemas?**
A mapping table of 74 source columns to one snake_case schema (`reports/tables/schema_mapping.csv`).
Example: `Tenure in Months` and `tenure` both become `tenure`; `Married` and `Partner` both
become `partner`. Values are harmonized too: Yes/No becomes 1/0, `Month-to-Month` becomes
`Month-to-month`.

**13. What conflicts did you find between the sources and how did you resolve them?**
Contract disagrees for 265 customers, dependents for 551, tenure and total charges for 11, and
payment method uses different categories entirely (4 vs 3, no one-to-one mapping). Rule: source 1
wins for attributes both sources have, because the label and Phase 1 are built on it. Source 2
contributes the attributes source 1 lacks. Every conflict is logged.

**14. What did you do for entity resolution?**
Both sources have a customer ID, and all 7,043 IDs match with no duplicates. To check the link
and show the technique, we also matched the sources without the ID: blocking on six stable
attributes cut 49.6 million possible pairs to 15,241, then we scored pairs on 11 attributes.
That links 6,931 customers with 100% precision (recall 98.4%). The other 112 tie with a
look-alike customer who has the same bill and services, which shows why a real key matters.

**15. What kinds of joins are there?**
One-to-one on customer ID for the seven customer tables. Many-to-one from location to
population on zip code: 1,626 zip codes are shared by 7,043 customers.

**16. Give examples of functional dependencies.**
Hold: `zip_code -> city`, `zip_code -> population`, `age -> senior_citizen`,
`number_of_referrals -> referred_a_friend`, and the arithmetic one
`total_revenue = total_charges + extra data + long distance - refunds`.
Conditional: `internet_service = 'No'` forces all six add-ons to "No internet service".
Fail: `city -> zip_code` (156 cities have several zip codes), and
`number_of_dependents -> dependents`, which exposes the 551 conflicts.

**17. Why did you drop some source 2 columns?**
Satisfaction score, churn score, customer status, churn category and churn reason are recorded
after or because of the churn. Satisfaction 1 or 2 is always churn; 4 or 5 never is. Using
them would be label leakage: great scores in the lab, useless on a current customer. CLTV is
dropped too: it is a value IBM predicted from the same data, not something we observe.

## Data cleaning

**18. What nulls did you have and what did you do?**
Three columns. `offer` (3,877 blanks) and `internet_type` (1,526 blanks) are structural: the
customer has no offer or no internet, so we fill "None". `TotalCharges` has 11 blanks, all
customers with tenure 0 who have not been billed, so we fill 0.

**19. Source 2 has a total for those 11 customers. Why not use it?**
Source 2 also says those customers have tenure 10 and a total of 10 months of charges, while
source 1 says tenure 0. Copying only the total would give rows with tenure 0 and $525 in
charges. We keep source 1 consistently and compare the cross-source fill as an experiment.

**20. Cleaning did not change your F1. So is it useless?**
On this data, yes, almost: 11 bad cells out of 7,043 rows cannot move the score (0.626 before
and after). To measure what cleaning is worth, we damaged a copy on purpose (10% blanks, 3%
values multiplied by 10). F1 fell from 0.629 to 0.615, and IQR capping brought it back to 0.622.

**21. "Drop rows" has the highest F1 in your null chart. Why not use it?**
It deletes every customer without an offer, 64% of the rows, including the rows it is graded
on. Its score is for an easier, smaller problem, and it cannot predict for most customers.

**22. How did you detect outliers, and what did you find?**
IQR fences (1.5 x IQR), z-score above 3, and Isolation Forest. Source 1 columns have none.
Source 2 has them: referrals 676, GB download 362, long distance charges 196, population 57
by IQR. Refunds and extra data charges are mostly zero, so IQR is not usable there (Q1 = Q3)
and we rely on z-score.

**23. Why cap outliers and not remove them?**
They are real heavy users, not errors, so deleting them loses real customers. Capping at the
fence keeps the row and limits its pull. On real data all five strategies are within 0.003 F1.
In the stress test, capping was the best (0.622 vs 0.615 for keeping).

**24. How do you avoid leaking test data during cleaning?**
Fences, medians and imputers are computed on training folds only and then applied to the
held-out fold. Rows are only ever removed from training data.

## Data transformation

**25. What is the difference between normalization and scaling in your project?**
Scaling changes the range (z-score, min-max, robust, max-abs, decimal scaling). Normalization
changes the shape of a skewed column so it is closer to a bell curve (log, Yeo-Johnson, quantile).

**26. Scaling made no difference for logistic regression. Why show it?**
That is the finding: with regularized logistic regression F1 is 0.679 to 0.681 for every
scaler. KNN depends on distances, and there scaling lifts F1 from 0.514 to 0.608. Tree models
don't care about scale at all. Whether to scale depends on the model.

**27. Which encodings did you compare?**
Binary 1/0 for yes/no columns. For the others: one-hot (0.679), target (0.678), ordinal (0.676),
frequency (0.675), and dropping them (0.670). One-hot is the safest for unordered categories
such as payment method.

**28. City has over 1,100 values. Isn't one-hot a problem?**
It makes about 1,100 sparse columns. Regularization and tree splitting handle it, and unknown
cities at prediction time become all zeros. Frequency and target encoding are the compact
alternatives and score about the same.

## Visualization and ETL

**29. What features did you build?**
Add-on count, price per service, average charge per month, revenue per month, long-distance
share, tenure band, and flags for month-to-month, fiber, auto-pay and new customer.

**30. Did the engineered features help?**
Not for logistic regression: 0.682 without, 0.679 with, which is inside the noise. But
`is_month_to_month` is the second most important input of the final model (shuffling it costs
0.097 F1), so the model does use them.

**31. How did you find which features matter?**
Three ways. Mutual information ranking (filter). A feature selection experiment: top 10
features give 0.629, top 30 give 0.642, all 47 give 0.679. And permutation importance on the
final model: referrals, month-to-month, number of dependants, dependents, tenure.

**32. Why does the model need all 47 features if the top 10 are the "best"?**
The ranking scores one feature at a time. Some features are weak alone but strong in
combination. Number of dependants ranks 27th by mutual information yet is the third most
important feature in the model.

**33. What does the dashboard show, and where is the model deployed?**
Streamlit app with five tabs: overview with KPIs and a map, churn drivers, data quality,
experiment results, and a form that scores a new customer with the saved model
(`models/churn_model.joblib`). Filters at the top apply to every tab.

**34. What tools did you use?**
Python, pandas, scikit-learn, Delta Lake (delta-rs), SQLite, Streamlit, matplotlib, GitHub
and GitHub Actions for an automatic check on every push.

## Harder questions

**35. Source 1 says no dependents, source 2 says 2 dependants, and you use both columns. Isn't that inconsistent?**
We keep source 1's flag by our conflict rule and add source 2's count as a new attribute.
They disagree for 551 customers and we report that. The model finds both useful. A stricter
option is to keep only one; we left both in because that is what the data says.

**36. The dataset is synthetic. Does the result mean anything?**
The numbers would differ on real customers. The method carries over: integrate, check
conflicts, clean inside folds, test every choice, hold out a test set.

**37. What would you do with more time?**
Add an independent source (for example census income by zip code), tune the model's
hyperparameters, choose the decision threshold from retention costs, and host the dashboard online.

**38. What changed since Phase 1?**
A second source with five files, a data lake, conflict handling, entity resolution, more
cleaning and transformation options with measured effects, engineered features, a model and
a dashboard. The Phase 1 decisions still stand: the 11 blank totals are filled with 0, and
source 1 columns have no IQR outliers.
