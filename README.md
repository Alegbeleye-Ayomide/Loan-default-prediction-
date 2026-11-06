<<<<<<< HEAD
# Loan Default Prediction

Predicting whether a borrower will experience serious financial distress (90+ days past due) within two years, using the Kaggle **Give Me Some Credit** dataset. The project compares logistic regression, random forest and XGBoost, tunes them with cross-validation, chooses a decision threshold from a recall target, and evaluates the final model once on a held-out test set.

## Results

Final model: **XGBoost**, with the decision threshold set so that about 80% of defaulters are caught (threshold ≈ 0.254, chosen on out-of-fold training predictions).

| Metric (test set) | Value |
|---|---|
| Recall | 0.816 |
| Precision | 0.1942 |
| F1 | 0.3137 |
| ROC AUC | 0.8615 |

Confusion matrix: 21,207 (true negatives), 1,636 (true positives), 6,788 (false positives) and 369 (false negatives).

### Model comparison

| Model | Baseline CV ROC AUC | Notes |
|---|---|---|
| Logistic regression | 0.8522 | Highest recall at the default 0.5 threshold, lowest precision |
| XGBoost | 0.8501 | Selected: highest precision at the 80% recall target |
| Random forest | 0.8317 | Gained the most from tuning |

After tuning, XGBoost (0.8579) and random forest (0.8573) are effectively tied on CV ROC AUC. At the same 80% recall, all three models are within about 0.01 of each other in precision, so the final choice rests on a small margin.

## Dataset

- 150,000 borrowers, 10 raw features, binary target `SeriousDlqin2yrs`.
- Heavily imbalanced: 6.684% defaulters.

## Approach

1. **Split first.** Stratified 80/20 train/test split. Everything learned from data (medians, caps, scaler, SMOTE, threshold) comes from the training set only.
2. **Cleaning.** Median imputation for missing values, IQR capping for continuous columns, fixed domain-based caps for age and the delinquency counts.
3. **Feature engineering.** `DelinquencyScore` (sum of the three past-due counts) and `TotalLoans`. Correlated features were removed (`TotalLoans` and the three raw past-due columns).
4. **Skewness.** A log transform is applied to `RevolvingUtilizationOfUnsecuredLines` and `DebtRatio`. It worsened `MonthlyIncome`, so that column was restored to its original scale.
5. **Class imbalance.** SMOTE inside an `imblearn` pipeline, so synthetic rows are created only from the training part of each fold and never touch validation or test data.
6. **Model selection.** Stratified 5-fold cross-validation on the training set, scored on recall, precision, F1 and ROC AUC.
7. **Tuning.** `GridSearchCV` for logistic regression, `RandomizedSearchCV` for random forest and XGBoost, scored on ROC AUC.
8. **Threshold.** Chosen from out-of-fold predicted probabilities as the highest threshold that still reaches 80% recall. The 80% target is an assumption that missed defaulters cost more than false alarms.
9. **Final evaluation.** Scored once on the test set with the fixed threshold.

## Key findings

- `DelinquencyScore` and `RevolvingUtilizationOfUnsecuredLines` are the most influential features for the XGBoost model.
- Tuning barely helped logistic regression (it plateaus for `C` above about 10) but helped random forest a lot, since the default unlimited depth overfits.
- Comparing models at a fixed 0.5 threshold is misleading, because each model's probabilities sit in a different range after SMOTE. Comparing precision at the same recall is fairer.

## Repository contents

| File | Description |
|---|---|
| `loan_default_prediction.ipynb` | Full analysis: EDA, cleaning, modelling, tuning, threshold selection, evaluation |
| `loan_model.pkl` | Saved XGBoost pipeline, decision threshold and feature columns |
| `README.md` | This file |

## How to run

```bash
pip install -r requirements.txt
```

Download the dataset from Kaggle (Give Me Some Credit) and update the data path in the first cells of the notebook.

## Using the saved model

```python
import joblib
import pandas as pd

saved = joblib.load('loan_model.pkl')

# new_data must have the same columns as training and be cleaned the same way
proba = saved['model'].predict_proba(new_data[saved['feature_columns']])[:, 1]
flagged = (proba >= saved['threshold']).astype(int)
```



## Limitations

- **Cleaning is outside the pipeline.** The saved model contains the scaler, SMOTE and XGBoost, but not the cleaning and feature steps, so new data must be cleaned the same way before prediction.
- **The recall target is assumed.** A real deployment would choose the threshold from the actual costs of missed defaulters and wrongly rejected borrowers.
- **Single train/test split and seed.** Differences of a few thousandths in ROC AUC between models are within noise.

## Tech stack

python, pandas, NumPy, scikit-learn, imbalanced-learn, XGBoost, matplotlib, seaborn, joblib.

## Author

**Alegbeleye Ayomide**: Computer Engineering student & ML engineer
[GitHub](https://github.com/Alegbeleye-Ayomide) · [LinkedIn](https://www.linkedin.com/in/alegbeleye-ayomide-b74a13394/)
=======
# Loan-default-prediction-
This is a machine learning pipeline of a loan default prediction project
>>>>>>> ca3d806be67eb2e7fc0e210b9f92d66be294adea
