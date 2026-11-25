# Customer Churn Prediction Project

**TripleTen Sprint 6 Project — Introduction to Machine Learning using Python**

## 1. Project Overview
An e-commerce company wants to identify customers who are likely to churn so the retention team can intervene proactively. In this project, we explore customer behavior and satisfaction data, preprocess missing values and categorical features, train and compare **Logistic Regression** and **Decision Tree** classification models, evaluate generalization and classification metrics, and translate the findings into actionable business recommendations.

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
| `CustomerID` | Unique customer identifier (excluded from features) |
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
All steps follow the project template and are executed in [`churn_prediction.ipynb`](./churn_prediction.ipynb):

1. **Step 0 — Import Libraries:** `pandas`, `numpy`, `matplotlib`, `seaborn`, and `scikit-learn` (`train_test_split`, `StandardScaler`, `LogisticRegression`, `DecisionTreeClassifier`, `accuracy_score`, `precision_score`, `recall_score`, `confusion_matrix`).
2. **Task 1 — Data Exploration:** Inspected sample rows (`head()`), schema (`info()`), missing values (`isnull().sum()`), target balance (`value_counts()`), summary statistics (`describe()`), duplicate checks, and churn-rate breakdowns by `Complain`, `SatisfactionScore`, `MaritalStatus`, `Gender`, and `Tenure`.
3. **Task 2 — Data Preprocessing:** Dropped any rows missing `Churn`, applied **median imputation** to numerical columns (`Tenure`, `SatisfactionScore`, `NumberOfAddress`, `OrderCount`, `DaySinceLastOrder`), applied one-hot encoding (`pd.get_dummies(..., drop_first=True)`) to `Gender` and `MaritalStatus`, and defined `X` (`(1896, 9)`, excluding `CustomerID` and `Churn`) and `y` (`(1896,)`).
4. **Task 3 — Train-Test Split & Feature Scaling:** Performed an 80/20 split (`test_size=0.20, random_state=42`) producing **1,516 training rows** and **380 test rows** (200 churned, 180 active), then applied `StandardScaler` fit on `X_train[num_cols]` and transformed `X_test[num_cols]`.
5. **Task 4 — Train Classification Models:** Trained `LogisticRegression(max_iter=1000, random_state=42)` and tuned `DecisionTreeClassifier(random_state=42)` across `max_depth` candidates (`2, 3, 4, 5, 6, None`), selecting **`max_depth=4`** for the highest constrained Recall (`0.8450`) and controlled train–test gap (`0.0252`).
6. **Task 5 — Model Evaluation:** Evaluated Accuracy, Precision, Recall, Confusion Matrices, and train–test overfitting checks, then compared both models on Recall and generalization.
7. **Conclusion:** Summarized the selected model, key churn drivers, and retention strategy recommendations.

---

## 4. Key Exploratory Findings
- **Early-Tenure Drop-Off:** Churned customers have a median tenure of **1.0 month** (mean **3.38 months**) versus **10.0 months** (mean **11.70 months**) for active customers. Customers with `Tenure <= 1` month churned at **84.03%** (`n = 739`), compared to **23.84%** for `Tenure > 1` month (`n = 1,032`).
- **Complaints Strongly Signal Churn:** Customers who filed a complaint (`Complain = 1`) churned at **70.36%** (`n = 722`), nearly double the **37.48%** churn rate among non-complainers (`n = 1,174`).
- **Marital Status & Gender Differences:** `Single` customers churned at **64.09%** (`n = 749`), compared with **45.26%** for `Divorced` (`n = 274`) and **39.40%** for `Married` (`n = 873`). Male customers churned at **52.36%** vs. **46.40%** for Female customers.
- **Satisfaction Score:** Self-reported `SatisfactionScore` increased slightly with churn (**39.30%** at score 1 up to **61.97%** at score 5), showing that survey ratings alone cannot rule out churn risk.

---

## 5. Decision Tree `max_depth` Experiment

| `max_depth` | Train Accuracy | Test Accuracy | Train–Test Gap | Precision | Recall |
|---|---:|---:|---:|---:|---:|
| `2` | 0.7658 | 0.7632 | +0.0027 | 0.8438 | 0.6750 |
| `3` | 0.8015 | 0.8132 | -0.0117 | 0.8377 | 0.8000 |
| **`4` (Selected)** | **0.8226** | **0.7974** | **+0.0252** | **0.7860** | **0.8450** |
| `5` | 0.8417 | 0.8026 | +0.0391 | 0.8453 | 0.7650 |
| `6` | 0.8542 | 0.8263 | +0.0279 | 0.8454 | 0.8200 |
| `None` | 0.9901 | 0.8947 | +0.0954 | 0.8670 | 0.9450 |

`max_depth=4` achieves the highest Recall (**0.8450**) among all constrained depths while keeping the train–test gap moderate (`0.0252`), whereas `max_depth=None` severely overfits (`0.0954` gap, `> 0.05`).

---

## 6. Model Comparison (Test Set, `n = 380`)

| Model | Train Accuracy | Test Accuracy | Train–Test Gap | Precision | Recall | Test Confusion Matrix (`[[TN, FP], [FN, TP]]`) |
|---|---:|---:|---:|---:|---:|---|
| **Logistic Regression** | 0.8140 | **0.8053** | **0.0087** (Small) | **0.7944** | **0.8500** | `[[136, 44], [30, 170]]` |
| **Decision Tree (`max_depth=4`)** | 0.8226 | 0.7974 | 0.0252 (Moderate) | 0.7860 | 0.8450 | `[[134, 46], [31, 169]]` |

### Top Churn Drivers Across Both Models
- **Logistic Regression (Standardized Coefficients / Odds Ratios):**
  - `Complain`: **+1.6705** (Odds Ratio: **5.31×**)
  - `Tenure`: **-1.5835** (Odds Ratio: **0.21×** per SD increase)
  - `NumberOfAddress`: **+0.4988** (Odds Ratio: **1.65×**)
  - `MaritalStatus_Married`: **-0.4677** (Odds Ratio: **0.63×**)
  - `OrderCount`: **+0.4499** (Odds Ratio: **1.57×**)
  - `MaritalStatus_Single`: **+0.4327** (Odds Ratio: **1.54×**)
  - `DaySinceLastOrder`: **-0.4256** (Odds Ratio: **0.65×**)
- **Decision Tree (`max_depth=4`, Feature Importance):**
  - `Tenure`: **71.90%**, `Complain`: **11.98%**, `NumberOfAddress`: **7.42%**, `DaySinceLastOrder`: **4.08%**

---

## 7. Selected Model & Why
**Logistic Regression** is selected as the best model for this project:
- **Highest Recall (Primary Metric):** Achieves **0.8500** vs. **0.8450** for the Decision Tree, catching **170 of 200** churned customers on the test set (missing 30 vs. 31).
- **Higher Accuracy & Precision (Lower False-Positive Cost):** Outperforms the Decision Tree on Accuracy (**0.8053 vs. 0.7974**) and Precision (**0.7944 vs. 0.7860**), generating fewer false alarms (**44 vs. 46** False Positives).
- **Stronger Generalization:** Exhibits a small train–test gap of **0.0087** (`< 0.02`), compared with **0.0252** (`0.02–0.05`, moderate) for the Decision Tree.

---

## 8. Business Recommendations
1. **Prioritize First-30-Day Onboarding (`Tenure <= 1` month):** With **84.03%** churn in the first month, launch automated onboarding sequences, proactive support check-ins, and a second-order incentive within days 14–21.
2. **Escalate Complaints Immediately (`Complain = 1`):** Route every complaint (**70.36%** churn rate, **5.31×** churn odds) into a fast-track resolution SLA with post-resolution follow-up credits.
3. **Target Single Shoppers (`MaritalStatus = Single`):** Engage the **64.09%** churn segment with personalized promotions and multi-item bundles.
4. **Score Active Accounts Weekly with Model Probabilities:** Use Logistic Regression risk probabilities rather than satisfaction survey scores alone to focus high-touch retention offers on the highest-risk accounts.

---

## 9. How to Run
1. Install Python 3.10+ and required libraries:
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn jupyter nbconvert
   ```
2. Execute the notebook from the terminal:
   ```bash
   jupyter nbconvert --to notebook --execute --inplace churn_prediction.ipynb
   ```
