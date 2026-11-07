# LogicMojo AI/ML Program — Coursework & Assignments

Notebooks, exercises and assignments from the **LogicMojo AI/ML program (June 2026 cohort)**, worked through between **21 June and 10 September 2026** (29 commits). The path runs from Python fundamentals through NumPy/Pandas and data visualization to exploratory data analysis and supervised machine learning.

Assignment problem statements and datasets come from the program. Results quoted below are taken from the executed notebooks in this repo.

## Highlights

### 1. Bank Loan Default Risk — EDA Case Study
`Assignment/EDA_Bank_Case_Study/Bank_loan_Assignment.ipynb`

Exploratory analysis of a consumer-lending dataset to find the applicant and loan attributes that drive default, framed around the two decision risks: rejecting a good borrower vs. approving a defaulter.

- **Scale:** 307,511 applications × 122 columns, joined with 1,670,214 previous-application records; the merged analysis frame has ~1.41M rows.
- **Cleaning:** removed sparse and redundant columns (122 → 46 on the application table, 37 → 22 on the history table), standardized `DAYS_*` fields, and rescaled amount columns.
- **Missing-value strategy chosen per variable, not blanket-filled:**
  - median for right-skewed amounts (`AMT_ANNUITY`, `AMT_GOODS_PRICE`)
  - zero for zero-inflated social-circle counts
  - mode for near-complete categoricals (`NAME_TYPE_SUITE`, 0.4% missing)
  - an explicit `Unknown` category for `OCCUPATION_TYPE` (31.3% missing)
  - credit-bureau columns share an identical 41,519 missing count, read as "applicant has no bureau record"
- **Outliers:** IQR-based upper-fence capping on income, credit, annuity, goods price and dependents.
- **Feature engineering:** binned age, income, credit and employment duration.
- **Class imbalance:** 91.9% repaid vs. 8.1% defaulted (~11.4 : 1), so accuracy alone is not a meaningful metric here.
- **Analysis:** univariate and bivariate comparison of defaulters vs. repayers, plus a merged analysis linking past loan behavior to current default.
- **Findings:** income level, loan burden (credit / annuity), employment stability, regional rating, occupation and organization type are the main differentiators. Supporting flowchart: `loan_eda_flow_chart.png`.

### 2. Imbalanced Classification — Purchase Intent (Logistic Regression)
`Assignment/02_logistic_regression/online_shoppers_logistic_regression_solution.ipynb`

Predicting whether an e-commerce session ends in a purchase (12,330 sessions, 13 features, 15.5% positive class). Four approaches compared on the same stratified 80/20 split (9,864 train / 2,466 test):

| Method | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| Baseline (threshold 0.5) | 0.880 | 0.739 | 0.348 | 0.473 |
| Threshold tuning (0.3) | 0.890 | 0.687 | 0.534 | 0.601 |
| Class-weighted (`balanced`) | 0.851 | 0.514 | 0.741 | 0.607 |
| SMOTE (training data only) | 0.846 | 0.502 | 0.757 | 0.603 |

The notebook works through why `stratify=y` is used, why SMOTE must be applied to the training split only (to avoid leakage into evaluation), and how the precision–recall trade-off maps to business cost (discount spend vs. missed buyers).

### 3. IMDb Movie Analysis
`Assignment/Movie_Assignment/IMDb+Movie+Assignment.ipynb`

Pandas + Seaborn analysis of the 100 top-rated movies of 2010–2016: profit analysis, critic-vs-audience scores, most popular cast trios, runtime distribution, genre aggregation across three genre columns, and demographic heatmaps (gender × age × genre, US vs. non-US voters, top-1000 voters).

### 4. K-Nearest Neighbors — Wine Classification
`Assignment/03_k_nearest_neighbors/wine_knn_assignment.ipynb`

Multiclass KNN on the scikit-learn Wine dataset: effect of feature scaling (test accuracy 0.806 → 0.972), choosing K with stratified cross-validation, distance-metric and vote-weighting comparison, final evaluation with a classification report and confusion matrix, and KNN-based imputation of missing values.

### 5. Decision Trees and Random Forests — Dry Bean Classification
`Assignment/04_decision_tree_random_forest/`

Seven-class classification on the UCI Dry Bean dataset (13,611 samples, 16 shape features): stratified train/validation/test split, underfitting vs overfitting via `max_depth` and `min_samples_leaf`, Gini vs entropy vs log-loss criteria, number of trees and OOB score, `max_features`, impurity-based feature importance, and validation-based model selection with a final classification report and confusion matrix.

## Repository Layout

```
Assignment/
  01_linear_regression/      Course reference solution (medical insurance regression), kept for study
  02_logistic_regression/    Imbalanced classification (purchase intent)
  03_k_nearest_neighbors/    Wine classification, KNN imputation
  04_decision_tree_random_forest/  Dry Bean classification
  EDA_Bank_Case_Study/       Bank loan default risk EDA
  Movie_Assignment/          IMDb movie analysis
  Bike_Example/              Pandas data manipulation walkthrough
  tips_Example/              Matplotlib / Seaborn / Plotly visualization walkthrough
Data/                        Datasets used by the notebooks (see notes below)
week1/ … week4/              Class notebooks and practice exercises
```

## Learning Log

| When | What |
|---|---|
| Week 1 | Python basics, list-slicing exercises |
| Week 2 | Slicing, control flow, loops, error handling; tuples, sets, dictionaries; linked lists |
| Week 3 | Functional programming (pure functions, immutability, higher-order functions, recursion); Python data types practice |
| Week 4 | NumPy (array creation, reshaping, indexing, boolean indexing, arithmetic); Pandas (Series/DataFrame, selection, filtering, adding/dropping columns, missing data); bike-rental data manipulation |
| Mid-July | IMDb movie analysis |
| 19 Jul – 4 Aug | Bank loan default-risk EDA, built up step by step: data loading, filtering, standardization, null imputation, univariate/bivariate plots, conclusions |
| 30 Aug | Logistic regression on imbalanced data |
| 10 Sep | KNN classification and imputation |
| Later | Decision trees and random forests |

## Tech Stack

Python · NumPy · Pandas · Matplotlib · Seaborn · Plotly · scikit-learn · imbalanced-learn (SMOTE) · Jupyter

## Running the Notebooks

```bash
pip install numpy pandas matplotlib seaborn plotly scikit-learn imbalanced-learn jupyter
```

- The **Bank Loan** notebook expects `application_data.csv` and `previous_application.csv`, which are **not included** in this repo because of their size. Place them under `Data/Bank_loan EDA/` and update the path in the loading cell.
- The **Linear Regression** notebook downloads its dataset automatically on first run.
- The **KNN** notebook uses the Wine dataset bundled with scikit-learn.
- Other notebooks read from `Data/`.
