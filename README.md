# Bank Term Deposit Prediction

Predicts whether a bank customer will subscribe to a term deposit after a marketing campaign.

## Problem Statement
Using customer demographics and campaign details, classify whether the customer subscribes (`y` = yes) or not (`y` = no). This helps banks target the right customers and reduce wasted calls.

## Dataset
- 4,521 rows, 17 columns (`bank_test.csv`)
- Target `y`: 4,000 "no" vs 521 "yes", so the data is **imbalanced**
- No missing values
- File not included in this repo; add your own copy

## Approach
1. **EDA**: statistics, correlation heatmap, pairplot, boxplots, countplots for job, marital status, education, loans, etc.
2. **Feature selection**: dropped `pdays`, `day`, `contact`, `previous`, `poutcome`
3. **Preprocessing**:
   - Converted `duration` from seconds to minutes
   - `LabelEncoder` for categorical columns
   - `StandardScaler` on `balance`
4. **Split**: 70/30 stratified train-test split (`random_state=1`)
5. **Models**: Random Forest and XGBoost, both tuned with `GridSearchCV`

## Results (test set, 1,357 samples)
| Model | Accuracy | Recall (class "yes") | F1 (class "yes") |
|---|---|---|---|
| Random Forest | 88% | 0.21 | 0.29 |
| XGBoost | 88% | 0.26 | 0.32 |

**Key takeaway:** overall accuracy looks good, but it is driven by the majority class. Both models miss most actual subscribers, so accuracy alone is misleading for this problem.

## Tech Stack
Python, pandas, seaborn, matplotlib, scikit-learn, XGBoost

## How to Run
1. Place `bank_test.csv` in your working directory and update the path in the notebook
2. `pip install xgboost`
3. Open `Bank_TermDeposit.ipynb` and run all cells

## Possible Improvements
- Handle class imbalance (undersampling, SMOTE, `class_weight`, `scale_pos_weight`)
- Optimise for recall / F1 / ROC-AUC instead of accuracy
- Tune the decision threshold
