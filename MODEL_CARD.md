# Heart disease classifier: model and data record

## Intended use

This is a research demonstration of binary classification on the historical [UCI Heart Disease dataset](https://uci-ics-mlr-prod.aws.uci.edu/dataset/45/heart%2Bdisease). The label is `num > 0` (disease present) versus `num = 0` (no disease in the dataset). The Streamlit form shows a model output for an entered profile. It does not diagnose disease, predict a future heart attack, or provide a clinically validated individual risk.

## Data lineage and quality

- Source for training: `heart_disease_uci.csv`, with 920 rows from Cleveland, Hungary, Switzerland, and VA Long Beach. It is the only raw dataset distributed with this repository.
- Two exact duplicate profiles apart from `id` are removed, leaving 918 rows: 508 positive and 410 negative.
- `heart.csv` contains 1,025 rows but only 302 distinct profiles; all 302 match Cleveland profiles in the training source. Its target coding is reversed relative to our binary target.
- `dataset_` contains 1,190 rows but only 918 distinct profiles. After harmonization, all 918 are compatible with the 918 distinct training-source profiles; it provides no independent cohort. Its completed values where the original has gaps have unverified provenance.
- We do not append either additional file, count their repetitions as new patients, or use their filled-in values as observed measurements. Those files are excluded from the repository because they are unnecessary for reproduction and their redistribution terms are unclear.

The source has substantial and hospital-dependent missingness. In the raw file, `slope` is missing in 309 rows, `ca` in 611, and `thal` in 486. Cholesterol values of zero (172 rows) and one zero resting blood pressure are treated as missing. The cleaned data retains missing values; imputation is learned inside each training fold. `id` and hospital `dataset` are not prediction features. Hospital is retained for splitting and transfer checks.

After cleaning, only 660 of 918 rows have every `core_10` input observed, 462 have every `plus_slope_11` input, and 299 have every `all_13` input. Dropping incomplete rows would therefore remove much of the available data, especially for the larger feature sets.

## Feature and missing-value experiments

The search evaluates three feature sets:

| Name | Inputs |
|---|---|
| `core_10` | Age, resting blood pressure, cholesterol, maximum heart rate, oldpeak, sex, chest-pain type, fasting blood sugar category, resting ECG, exercise-induced angina |
| `plus_slope_11` | `core_10` plus exercise ST slope |
| `all_13` | `plus_slope_11` plus number of major vessels (`ca`) and thallium scan category (`thal`) |

Every feature set is evaluated with two treatments:

1. `median_mode`: training-fold median for numeric gaps; training-fold most frequent value for categorical gaps.
2. `missing_indicator`: training-fold median plus a missingness flag for numeric gaps; a separate `Missing` category for categorical gaps.

Numeric values are standardized for logistic regression and left on their original scales for tree models. Categorical values are one-hot encoded. No preprocessing is fitted before a validation split. [Scikit-learn documents why preprocessing belongs inside the pipeline](https://scikit-learn.org/stable/modules/compose.html).

## Selection and evaluation protocol

The cleaned records are split 80/20 once with seed 42 and stratification by hospital and binary label. This gives 734 training rows (406 positive) and 184 holdout rows (102 positive). Each of logistic regression, random forest, and XGBoost receives 18 predeclared combinations: three feature sets × two missing-value treatments × three parameter settings. The exact settings are in `tune.py` and the saved `metrics.json`.

Within the training split, three outer stratified folds evaluate a full three-fold inner search per model family. The family with the highest mean outer F2 is selected; lower fold variation breaks a tie. A final five-fold search on all 734 training records selects the saved feature, missingness, and parameter settings for each family. The decision threshold stays fixed at 0.50. F2 gives recall extra weight. A separate leave-one-hospital-out test on the training records measures transfer to an unseen source. The fitted models are compared on the same 184-row holdout, but none of its outcomes select the model or settings.

The holdout was already inspected during baseline development. It is therefore a **development holdout**, not a fresh final test after this search. Repeated decisions informed by its results can overfit it; an independent patient cohort is needed for a final performance estimate. Independent test data representative of the intended users is also emphasized in [FDA good machine learning practice principles](https://www.fda.gov/medical-devices/artificial-intelligence-enabled-medical-devices/good-machine-learning-practice-medical-device-development-guiding-principles).

## Results

The final search on all training rows selected `all_13` with `missing_indicator` for all three model families. Selected parameter settings were logistic regression `C=0.1`; random forest 300 trees, maximum depth 6, and minimum leaf size 5; and XGBoost 150 trees, depth 4, learning rate 0.05, minimum child weight 2, and L2 regularization 5. Random forest had the highest mean **outer nested-CV F2** and is saved for the app.

| Model | Outer CV F2 mean ± SD | Holdout precision | Holdout recall | Holdout specificity | Holdout FN | Holdout FP |
|---|---:|---:|---:|---:|---:|---:|
| Logistic regression | 0.844 ± 0.042 | 0.844 | 0.902 | 0.793 | 10 | 17 |
| Random forest | **0.855 ± 0.024** | 0.836 | 0.902 | 0.780 | 10 | 18 |
| XGBoost | 0.836 ± 0.031 | 0.835 | 0.892 | 0.780 | 11 | 18 |

The selected forest's holdout confusion matrix is `[[64, 18], [10, 92]]`, with rows actual negative/positive and columns predicted negative/positive. Its fixed-configuration baseline had 12 false negatives and 21 false positives on the same holdout. This comparison is descriptive because the holdout was already viewed before tuning.

When each hospital is left out of training, the selected forest configuration gives:

| Held-out hospital | Rows | Positive labels | Recall | False negatives | False positives |
|---|---:|---:|---:|---:|---:|
| Cleveland | 243 | 111 | 0.901 | 11 | 53 |
| Hungary | 234 | 85 | 0.941 | 5 | 53 |
| Switzerland | 98 | 92 | 0.772 | 21 | 2 |
| VA Long Beach | 159 | 118 | 0.822 | 21 | 17 |

The transfer ablation holds forest parameters fixed and varies features and missingness. It shows a material tradeoff between the final search winner and the simpler 11-feature alternative:

| Forest features and missingness | Mean hospital F2 | Worst hospital recall | Total FN | Total FP |
|---|---:|---:|---:|---:|
| `all_13` + `missing_indicator` | 0.829 | 0.772 | **58** | 125 |
| `plus_slope_11` + `missing_indicator` | **0.834** | **0.814** | 62 | **97** |

These figures cover the four held-out hospital folds on the 734 training records; they are not an additional 734 independent patients. The 13-feature choice finds four more positive labels overall while producing 28 more false positives and a lower worst-hospital recall. It also requires two sparsely recorded tests. The app retains the predeclared nested-CV winner and exposes these transfer results for review.

Results from the completed search are saved in:

- `artifacts/metrics.json`: selected settings, nested cross-validation, holdout metrics, confusion matrices, and hospital-specific results.
- `artifacts/candidate_search.csv`: all 54 final-training candidate scores.
- `artifacts/nested_cv_folds.csv`: outer-fold results and selected settings.
- `artifacts/model_comparison.csv`: one-row summary per tuned model.
- `artifacts/transfer_ablation_summary.csv` and `artifacts/transfer_ablation_by_site.csv`: transfer checks for each feature/missingness treatment.
- `artifacts/baseline_metrics.json` and `artifacts/baseline_model_comparison.csv`: the original fixed-configuration benchmark.

## Known limitations

- These historical hospital datasets do not establish performance on people entering their own values into a public website. Several inputs require medical tests.
- Label prevalence differs sharply by source: approximately 36% positive in Hungary and 94% in Switzerland. Missingness also differs by source. Random within-source testing can overstate transfer to a new setting.
- The percentage shown in the app is an uncalibrated model score, not a validated patient risk. Probability calibration and a decision threshold tied to a defined clinical use remain future work.
- The cohort is small for comparing many choices. Fold-to-fold variation and subgroup sample sizes limit how strongly any small difference can be interpreted.

## Reproduction

From the project directory, install `requirements.txt`, run `python tune.py`, and launch `python -m streamlit run app.py`. `python train.py` reproduces the original fixed-configuration baseline and overwrites the primary model artifacts, so run `python tune.py` afterward to restore the tuned model. The raw CSV files stay unchanged.
