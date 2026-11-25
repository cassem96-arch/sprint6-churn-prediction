# Customer Churn Prediction Analysis

**TripleTen Sprint 6 Project — Introduction to Machine Learning using Python**

## 1. Project Overview
An e-commerce company has experienced rising customer churn and wants to determine whether machine learning models can proactively identify customers likely to leave the platform. In this project, we explore customer demographic, satisfaction, and behavioral data, build and compare **Logistic Regression** and **Decision Tree** classification models, evaluate their performance on a held-out test set, and translate the findings into actionable recommendations for the customer retention team.

---

## 2. Dataset
- **Source:** `https://practicum-content.s3.amazonaws.com/datasets/ecommerce_churn.csv`
- **Size:** **1,896 rows × 10 columns** (`0` duplicate rows; `1,896` unique `CustomerID` values)
- **Target Variable (`Churn`):** Perfectly balanced — **948 Active (`0`, 50.0%)** and **948 Churned (`1`, 50.0%)**
- **Missing Values:**
  - `Tenure`: 125 missing (**6.59%**)
  - `DaySinceLastOrder`: 99 missing (**5.22%**)
  - `OrderCount`: 63 missing (**3.32%**)

| Column | Description |
|---|---|
| `CustomerID` | Unique customer identifier (dropped prior to modeling) |
| `Churn` | Target variable (`1` = Churned, `0` = Active) |
| `Tenure` | Months the customer has been with the company |
| `Gender` | Customer gender (`Female`, `Male`) |
| `SatisfactionScore` | Customer satisfaction rating (`1` to `5`) |
| `MaritalStatus` | Marital status (`Single`, `Married`, `Divorced`) |
| `NumberOfAddress` | Number of shipping addresses saved on the account |
| `Complain` | Whether the customer submitted a complaint (`1` = Yes, `0` = No) |
| `OrderCount` | Total number of orders placed |
| `DaySinceLastOrder` | Days since the customer's last order |

---

## 3. Workflow & Methodology
All steps are implemented and executed in [`churn_prediction.ipynb`](./churn_prediction.ipynb):

1. **Import Libraries:** `pandas`, `numpy`, `matplotlib`, `seaborn`, and `scikit-learn`.
2. **Data Exploration:** Inspected shape, types, summary statistics, missing values, duplicates, target distribution, churn-rate breakdowns, numerical distributions, and feature correlations.
3. **Data Preprocessing:** Dropped `CustomerID`; one-hot encoded `Gender` and `MaritalStatus` with `drop_first=True` (yielding 9 features); verified right-skewness (`Tenure` skew 1.14, `OrderCount` skew 2.33, `DaySinceLastOrder` skew 1.82) to justify **median imputation**.
4. **Train-Test Split:** 80/20 stratified split (`stratify=y`, `random_state=42`) producing **1,516 training rows** (50.00% churn) and **380 test rows** (50.00% churn). Fitted both **median imputation** (`Tenure = 4.0`, `OrderCount = 2.0`, `DaySinceLastOrder = 3.0`) and **`StandardScaler`** (on `Tenure`, `SatisfactionScore`, `NumberOfAddress`, `OrderCount`, `DaySinceLastOrder`) strictly on the training set to avoid data leakage.
5. **Model Training:** Trained `LogisticRegression(random_state=42, max_iter=1000)` and compared `DecisionTreeClassifier(random_state=42)` across `max_depth` values (`2, 3, 4, 5, 6, None`), selecting `max_depth=4` to prevent overfitting (`max_depth=None` reached 99.08% train accuracy vs 86.05% test accuracy and lower ROC-AUC 0.8706).
6. **Model Evaluation:** Evaluated Accuracy, Precision, Recall, F1-Score, ROC-AUC, classification reports, confusion matrices, ROC curves, Logistic Regression coefficients/odds ratios, and Decision Tree feature importances.
7. **Conclusion & Recommendations:** Selected the best model and outlined business actions.

---

## 4. Key Exploratory Findings
- **Early-Tenure Drop-off Is Massive (`r = -0.49`):** Churned customers have a median tenure of **1.0 month** (mean **3.38 months**) versus **10.0 months** (mean **11.70 months**) for active customers. Customers with `Tenure <= 1` month churned at **84.03%** (`n = 739`), compared to **23.84%** for `Tenure > 1` month (`n = 1,032`).
- **Complaints Double Churn Risk (`r = +0.32`):** Customers who filed a complaint (`Complain = 1`) churned at **70.36%** (`n = 722`), nearly double the **37.48%** churn rate among non-complainers (`n = 1,174`).
- **Marital Status & Gender Differences:** **Single** customers churned at **64.09%** (`n = 749`), compared with **45.26%** for Divorced (`n = 274`) and **39.40%** for Married (`n = 873`). Male customers churned at **52.36%** vs **46.40%** for Female customers.
- **Satisfaction Score Anomaly:** Self-reported `SatisfactionScore` rose slightly with churn (**39.30%** at score 1 up to **61.97%** at score 5; `r = +0.14`), showing that satisfaction ratings alone cannot be used to rule out churn risk.

---

## 5. Model Comparison (Test Set, `n = 380`)

| Model | Train Accuracy | Test Accuracy | Precision | Recall | F1-Score | ROC-AUC | Test Confusion Matrix (`TN / FP / FN / TP`) |
|---|---:|---:|---:|---:|---:|---:|---|
| **Logistic Regression** | 0.8094 | **0.8263** | 0.8069 | **0.8579** | **0.8316** | **0.8786** | `151 / 39 / 27 / 163` |
| **Decision Tree (`max_depth=4`)** | 0.8120 | 0.8237 | **0.8090** | 0.8474 | 0.8278 | 0.8762 | `152 / 38 / 29 / 161` |

### Top Churn Drivers Across Both Models
- **Logistic Regression (Standardized Coefficients / Odds Ratios):**
  - `Complain`: **+1.7309** (Odds Ratio: **5.65×**)
  - `Tenure`: **-1.5777** (Odds Ratio: **0.21×** per SD increase)
  - `MaritalStatus_Single`: **+0.5981** (Odds Ratio: **1.82×**)
  - `NumberOfAddress`: **+0.5027** (Odds Ratio: **1.65×**)
  - `OrderCount`: **+0.4726** (Odds Ratio: **1.60×**)
- **Decision Tree (`max_depth=4`, Feature Importance):**
  - `Tenure`: **72.29%**, `Complain`: **13.53%**, `DaySinceLastOrder`: **4.81%**, `SatisfactionScore`: **4.73%**

---

## 6. Selected Model & Why
**Logistic Regression** is selected as the recommended production model:
1. **Strongest Performance on Retention Metrics:** It outperforms the regularized Decision Tree on **Accuracy (82.63% vs 82.37%)**, **Recall (85.79% vs 84.74%)**, **F1-Score (83.16% vs 82.78%)**, and **ROC-AUC (0.8786 vs 0.8762)**.
2. **Higher Recall Reduces Costly Missed Churners:** In retention, a False Negative (missing a customer who leaves) loses future revenue, whereas a False Positive only incurs a low-cost outreach touchpoint. Logistic Regression catches **163 of 190** churners on the test set (missing only 27).
3. **No Overfitting:** Test accuracy (**82.63%**) tracks closely with training accuracy (**80.94%**), confirming strong generalization.
4. **Continuous Probability Scoring:** `predict_proba` produces smooth risk scores that allow the retention team to rank customers and tune thresholds for different campaign budgets, while the Decision Tree (`max_depth=4`) remains useful as an interpretable rule summary for stakeholders.

---

## 7. Business Recommendations
1. **Prioritize First-30-Day Onboarding (`Tenure <= 1` month):** With **84.03%** churn among customers at `Tenure <= 1` month, introduce automated onboarding sequences, proactive support check-ins, and a second-order incentive within the first 14–21 days to carry new buyers past the month-1 drop-off.
2. **Treat Complaints as Immediate Retention Escalations (`Complain = 1`):** Because complaints carry a **70.36%** churn rate (**5.65×** churn odds), route every complaint into a fast-track SLA with post-resolution follow-up credits or outreach.
3. **Segment Campaigns for Single Shoppers (`MaritalStatus = Single`):** Target the **64.09%** churn segment with personalized promotions and multi-item convenience bundles.
4. **Use Model Probability Tiers Instead of Survey Scores Alone:** Score active customers weekly with the Logistic Regression model — reserve high-touch retention offers for High Risk (`p >= 0.70`) and automated lifecycle emails for Medium Risk (`0.40 <= p < 0.70`) — rather than relying on `SatisfactionScore` alone.

---

## 8. How to Run
1. Install Python 3.10+ and required libraries:
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn jupyter nbconvert
   ```
2. Execute the notebook from the terminal (the notebook loads `ecommerce_churn.csv` directly from its public URL):
   ```bash
   jupyter nbconvert --to notebook --execute --inplace churn_prediction.ipynb
   ```
   Or open it interactively with:
   ```bash
   jupyter notebook churn_prediction.ipynb
   ```
