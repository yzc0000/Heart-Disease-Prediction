# Heart disease prediction research demo

This project cleans a historical heart-disease dataset, compares three classifiers, and serves a saved model in Streamlit. The positive label is `num > 0` (disease presence in the [UCI Heart Disease dataset](https://uci-ics-mlr-prod.aws.uci.edu/dataset/45/heart%2Bdisease)); `num = 0` is negative. The app is a research demonstration, **not a diagnostic service or a validated personal risk calculator**. [MODEL_CARD.md](MODEL_CARD.md) records the full methodology and limitations.

## Run locally

From this directory, with Python 3.14 or a compatible supported version:

```powershell
python -m pip install -r requirements.txt
python tune.py
python transfer_ablation.py
python -m streamlit run app.py
```

`python train.py` reproduces the original fixed-configuration baseline. It writes to the same primary artifact paths, so run `python tune.py` afterward to restore the tuned app model. The previous baseline results are preserved in `artifacts/baseline_metrics.json` and `artifacts/baseline_model_comparison.csv`.

## Data and preprocessing

`heart_disease_uci.csv` is the sole training source and is included in this repository. Two additional files supplied during the audit are excluded from the repository because they are not needed to reproduce the model and their redistribution terms are unclear. `heart.csv` has 302 distinct profiles, all found in Cleveland, and `dataset_` has 918 distinct profiles compatible with the 918 distinct profiles in the original. Appending either file would duplicate patients. `dataset_` fills gaps present in the original, but the origin of those values is not documented in the supplied file, so we do not treat them as observed measurements.

Data attribution: [UCI Heart Disease, DOI 10.24432/C52P4X](https://doi.org/10.24432/C52P4X), attributed to Andras Janosi, William Steinbrunn, Matthias Pfisterer, and Robert Detrano under the repository's [CC BY 4.0 license](https://uci-ics-mlr-prod.aws.uci.edu/dataset/45/heart%2Bdisease).

The original has 920 rows; two exact duplicate profiles apart from `id` are removed. The remaining 918 rows contain 508 positive and 410 negative labels. Zero cholesterol in 172 raw rows and one zero blood pressure are changed to missing. Imputation happens inside each training fold. Hospital `dataset` is used to stratify and audit the split; it and `id` are not prediction features.

The feature search compares:

| Feature set | Inputs |
|---|---|
| `core_10` | Age, resting blood pressure, cholesterol, maximum heart rate, oldpeak, sex, chest-pain type, fasting blood sugar category, resting ECG, exercise-induced angina |
| `plus_slope_11` | `core_10` plus exercise ST slope |
| `all_13` | `plus_slope_11` plus number of major vessels (`ca`) and thallium scan category (`thal`) |

The two missing-value treatments are `median_mode` (numeric median, categorical mode) and `missing_indicator` (numeric median plus missingness flags, categorical `Missing` level). Numeric inputs are standardized for logistic regression; categorical inputs are one-hot encoded. Tree models use unscaled numeric inputs. The source's `slope`, `ca`, and `thal` fields are missing in 309, 611, and 486 of the 920 raw rows, respectively. [Scikit-learn explains why preprocessing must be fitted within validation folds](https://scikit-learn.org/stable/modules/compose.html).

## Search and evaluation

The same seed-42 80/20 split is used throughout: 734 training rows (406 positive) and 184 holdout rows (102 positive), stratified jointly by source hospital and label. Each of logistic regression, random forest, and XGBoost searches **18 combinations**: three feature sets × two missing-value treatments × three parameter settings. The settings are declared in [tune.py](tune.py).

Within training, a three-fold outer / three-fold inner nested cross-validation compares the search procedures using F2 at a fixed 0.50 threshold. A five-fold search on all training rows chooses each family's final settings. The family with the highest mean outer F2 is saved. A separate leave-one-hospital-out check tests whether the selected settings transfer to an unseen source. `transfer_ablation.py` compares all six feature/missingness pairs with each model family's selected parameter setting. The holdout is used for descriptive comparison and does not choose settings or the model family. [Nested CV reduces selection optimism](https://scikit-learn.org/stable/auto_examples/model_selection/plot_nested_cross_validation_iris.html).

### Current search results

| Model | Outer CV F2 | Holdout precision | Holdout recall | False negatives | False positives |
|---|---:|---:|---:|---:|---:|
| Logistic regression | 0.844 | 0.844 | 0.902 | 10 | 17 |
| Random forest | **0.855** | 0.836 | 0.902 | 10 | 18 |
| XGBoost | 0.836 | 0.835 | 0.892 | 11 | 18 |

All three final searches selected `all_13` with `missing_indicator`. The selected random forest has 300 trees, maximum depth 6, and a minimum of 5 samples per leaf. Its holdout confusion matrix is `[[64, 18], [10, 92]]`, with rows actual negative/positive and columns predicted negative/positive.

The mixed-hospital cross-validation result is only part of the picture. With random forest parameters held fixed, the leave-one-hospital-out check favored `plus_slope_11` with `missing_indicator` over `all_13` on mean site F2 (0.834 versus 0.829), worst-site recall (0.814 versus 0.772), and total false positives (97 versus 125). Some individual sites still favored `all_13`. This is an exploratory transfer comparison, so the default app model remains the nested-CV winner. The site result is shown in the app and [MODEL_CARD.md](MODEL_CARD.md).

The 184-row holdout was already inspected during the baseline stage. It is a **development holdout**, so changes guided by its numbers can overfit it. A genuinely independent cohort representative of intended users is needed before making a final performance claim. That principle is reflected in [FDA good machine learning practice guidance](https://www.fda.gov/medical-devices/artificial-intelligence-enabled-medical-devices/good-machine-learning-practice-medical-device-development-guiding-principles).

## Outputs

| File | Purpose |
|---|---|
| `artifacts/heart_model.joblib` | Saved selected pipeline used by Streamlit |
| `artifacts/metrics.json` | Cleaning, split, settings, nested CV, holdout, and hospital metrics |
| `artifacts/model_comparison.csv` | One-row tuned model comparison |
| `artifacts/candidate_search.csv` | Five-fold training scores for all 54 candidates |
| `artifacts/nested_cv_folds.csv` | Outer-fold choices and scores |
| `artifacts/transfer_ablation_summary.csv` | Hospital-transfer comparison of feature and missing-value choices |
| `artifacts/transfer_ablation_by_site.csv` | Per-hospital transfer results |

The raw input files are never modified. The app displays the selected configuration, model scores, confusion matrices, and transfer limitations. Its percentage is an **uncalibrated model score**, not a validated patient probability.
