# Heart disease prediction research demo

This project cleans a historical heart-disease dataset, compares three classifiers, and serves a saved model in Streamlit. The positive label is `num > 0` (disease presence in the [UCI Heart Disease dataset](https://uci-ics-mlr-prod.aws.uci.edu/dataset/45/heart%2Bdisease)); `num = 0` is negative. The app is a research demonstration, **not a diagnostic service or a validated personal risk calculator**. It does not diagnose disease or predict a future heart attack.

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

`heart_disease_uci.csv` is the sole training source and is included in this repository. It has 920 rows from Cleveland, Hungary, Switzerland, and VA Long Beach. Two additional files supplied during the audit are excluded because they are not needed to reproduce the model and their redistribution terms are unclear:

- `heart.csv` has 1,025 rows but only 302 distinct profiles; all 302 match Cleveland profiles in the training source. Its target coding is reversed relative to our binary target.
- `dataset_` has 1,190 rows but only 918 distinct profiles. After harmonization, all 918 are compatible with the distinct profiles in the training source. Its filled-in values where the original has gaps have unverified provenance.

Neither file provides an independent cohort. We do not append their repeated profiles or treat the filled-in values as observed measurements.

Data attribution: [UCI Heart Disease, DOI 10.24432/C52P4X](https://doi.org/10.24432/C52P4X), attributed to Andras Janosi, William Steinbrunn, Matthias Pfisterer, and Robert Detrano under the repository's [CC BY 4.0 license](https://uci-ics-mlr-prod.aws.uci.edu/dataset/45/heart%2Bdisease).

Two exact duplicate profiles apart from `id` are removed. The remaining 918 rows contain 508 positive and 410 negative labels. Zero cholesterol in 172 raw rows and one zero resting blood pressure are changed to missing. The source's `slope`, `ca`, and `thal` fields are missing in 309, 611, and 486 of the 920 raw rows, respectively. Missingness varies by hospital. The cleaned data retains missing values; imputation is learned inside each training fold. Hospital `dataset` is used to stratify and audit the split; it and `id` are not prediction features.

The feature search compares:

| Feature set | Inputs |
|---|---|
| `core_10` | Age, resting blood pressure, cholesterol, maximum heart rate, oldpeak, sex, chest-pain type, fasting blood sugar category, resting ECG, exercise-induced angina |
| `plus_slope_11` | `core_10` plus exercise ST slope |
| `all_13` | `plus_slope_11` plus number of major vessels (`ca`) and thallium scan category (`thal`) |

Only 660 of 918 cleaned rows have every `core_10` input observed, 462 have every `plus_slope_11` input, and 299 have every `all_13` input. Dropping incomplete rows would remove much of the cohort, especially for the larger feature sets.

The two missing-value treatments are `median_mode` (training-fold numeric median and categorical most frequent value) and `missing_indicator` (training-fold numeric median plus missingness flags, categorical `Missing` level). Numeric inputs are standardized for logistic regression; categorical inputs are one-hot encoded. Tree models use unscaled numeric inputs. No preprocessing is fitted before a validation split. [Scikit-learn explains why preprocessing must be fitted within validation folds](https://scikit-learn.org/stable/modules/compose.html).

## Search and evaluation

The same seed-42 80/20 split is used throughout: 734 training rows (406 positive) and 184 holdout rows (102 positive), stratified jointly by source hospital and label. Each of logistic regression, random forest, and XGBoost searches **18 combinations**: three feature sets × two missing-value treatments × three parameter settings. The settings are declared in [tune.py](tune.py).

Within training, three outer stratified folds each evaluate a full three-fold inner search per model family using F2 at a fixed 0.50 decision threshold. F2 gives recall extra weight. The family with the highest mean outer F2 is saved; lower fold variation breaks a tie. A five-fold search on all 734 training rows chooses each family's final feature set, missing-value treatment, and parameters. A separate leave-one-hospital-out check tests whether the selected settings transfer to an unseen source. `transfer_ablation.py` compares all six feature/missingness pairs with each model family's selected parameter setting. The holdout is used for descriptive comparison and does not choose settings or the model family. [Nested CV reduces selection optimism](https://scikit-learn.org/stable/auto_examples/model_selection/plot_nested_cross_validation_iris.html).

### Current search results

| Model | Outer CV F2 mean ± SD | Holdout precision | Holdout recall | Holdout specificity | Holdout FN | Holdout FP |
|---|---:|---:|---:|---:|---:|---:|
| Logistic regression | 0.844 ± 0.042 | 0.844 | 0.902 | 0.793 | 10 | 17 |
| Random forest | **0.855 ± 0.024** | 0.836 | 0.902 | 0.780 | 10 | 18 |
| XGBoost | 0.836 ± 0.031 | 0.835 | 0.892 | 0.780 | 11 | 18 |

All three final searches selected `all_13` with `missing_indicator`. Selected parameters were logistic regression `C=0.1`; random forest 300 trees, maximum depth 6, and minimum of 5 samples per leaf; and XGBoost 150 trees, depth 4, learning rate 0.05, minimum child weight 2, and L2 regularization 5. Random forest had the highest mean outer F2 and is saved for the app. Its holdout confusion matrix is `[[64, 18], [10, 92]]`, with rows actual negative/positive and columns predicted negative/positive. Its original fixed-configuration baseline had 12 false negatives and 21 false positives on the same holdout. That comparison is descriptive because the holdout was already viewed before tuning.

### Hospital transfer check

Leaving each hospital out of training in turn gives the following results for the selected random forest configuration:

| Held-out hospital | Rows | Positive labels | Recall | False negatives | False positives |
|---|---:|---:|---:|---:|---:|
| Cleveland | 243 | 111 | 0.901 | 11 | 53 |
| Hungary | 234 | 85 | 0.941 | 5 | 53 |
| Switzerland | 98 | 92 | 0.772 | 21 | 2 |
| VA Long Beach | 159 | 118 | 0.822 | 21 | 17 |

With forest parameters held fixed, the feature and missing-value ablation gives:

| Forest features and missingness | Mean hospital F2 | Worst hospital recall | Total FN | Total FP |
|---|---:|---:|---:|---:|
| `all_13` + `missing_indicator` | 0.829 | 0.772 | **58** | 125 |
| `plus_slope_11` + `missing_indicator` | **0.834** | **0.814** | 62 | **97** |

These are four folds on the 734 training records, not 734 additional patients. The 13-feature choice finds four more positive labels overall while producing 28 more false positives and lower worst-hospital recall. It also requires two sparsely recorded tests. Some individual sites favored `all_13`. This is an exploratory transfer comparison, so the app retains the predeclared nested-CV winner and displays this limitation.

The 184-row holdout was already inspected during the baseline stage. It is a **development holdout**, so changes guided by its numbers can overfit it. A genuinely independent cohort representative of intended users is needed before making a final performance claim. That principle is reflected in [FDA good machine learning practice guidance](https://www.fda.gov/medical-devices/artificial-intelligence-enabled-medical-devices/good-machine-learning-practice-medical-device-development-guiding-principles).

## Known limitations

- These historical hospital datasets do not establish performance for people entering their own values into a public website. Several inputs require medical tests.
- Label prevalence differs sharply by source: approximately 36% positive in Hungary and 94% in Switzerland. Missingness also differs by source, so random within-source testing can overstate transfer to a new setting.
- The app's percentage is an uncalibrated model score, not a validated patient risk. Probability calibration and a decision threshold tied to a defined clinical use remain future work.
- The cohort is small for comparing many choices. Fold variation and subgroup sample sizes limit how strongly small differences can be interpreted.

## Outputs

| File | Purpose |
|---|---|
| `artifacts/heart_model.joblib` | Saved selected pipeline used by Streamlit |
| `artifacts/heart_disease_cleaned.csv` | Cleaned records before fold-specific imputation |
| `artifacts/metrics.json` | Cleaning, split, settings, nested CV, holdout, and hospital metrics |
| `artifacts/model_comparison.csv` | One-row tuned model comparison |
| `artifacts/candidate_search.csv` | Five-fold training scores for all 54 candidates |
| `artifacts/nested_cv_folds.csv` | Outer-fold choices and scores |
| `artifacts/transfer_ablation_summary.csv` | Hospital-transfer comparison of feature and missing-value choices |
| `artifacts/transfer_ablation_by_site.csv` | Per-hospital transfer results |
| `artifacts/baseline_metrics.json` and `artifacts/baseline_model_comparison.csv` | Original fixed-configuration benchmark |

The raw input file is never modified. The app displays the selected configuration, model scores, confusion matrices, and transfer limitations.
