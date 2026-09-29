"""Streamlit research demo for the saved UCI heart-disease classifier."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

from heart_model import ALL_NUMERIC_FEATURES


ARTIFACTS = Path(__file__).resolve().parent / "artifacts"


@st.cache_resource
def load_model(modified_ns: int):
    return joblib.load(ARTIFACTS / "heart_model.joblib")


@st.cache_data
def load_report(modified_ns: int):
    return json.loads((ARTIFACTS / "metrics.json").read_text(encoding="utf-8"))


def optional_choice(label: str, options: list[str], help: str | None = None):
    return st.selectbox(label, options, index=None, placeholder="Select or leave unknown", help=help)


def optional_number(
    label: str,
    *,
    min_value: float,
    max_value: float | None = None,
    step: float = 1.0,
    help: str | None = None,
):
    return st.number_input(
        label,
        min_value=min_value,
        max_value=max_value,
        value=None,
        step=step,
        placeholder="Leave blank if unknown",
        help=help,
    )


def show_evaluation(report: dict) -> None:
    st.subheader("Model comparison")
    with st.expander("Data split and preprocessing"):
        cleaning, split = report["cleaning"], report["split"]
        st.write(
            f"{cleaning['raw_rows']} source rows → {cleaning['cleaned_rows']} "
            f"after removing {cleaning['exact_duplicate_rows_removed']} duplicates. "
            f"Training: {split['train_rows']} rows ({split['train_positive']} positive). "
            f"Holdout: {split['test_rows']} rows ({split['test_positive']} positive)."
        )
        st.write("Selected model inputs: " + ", ".join(report["features"]))
        st.write(
            "Zeros in cholesterol and resting blood pressure were treated as "
            "missing. Imputation and categorical encoding are fitted within "
            "training folds; numeric scaling is used for logistic regression."
        )
        missing = pd.DataFrame(
            [
                {"Feature": name, "Missing after cleaning": count}
                for name, count in cleaning["missing_after_cleaning"].items()
            ]
        )
        st.dataframe(missing.set_index("Feature"), width="stretch")
        st.caption(
            "The two additional files overlap the original patients and are not "
            "appended to training. Source: UCI Heart Disease dataset."
        )
    if "search" in report:
        st.caption(
            "Positive means the original UCI label was 1–4. Results use a 0.50 "
            "threshold. Model family was chosen by outer training cross-validation; "
            "features, missing-value treatment, and parameters were chosen inside "
            "training folds. The 20% holdout was inspected in earlier development "
            "and is a descriptive comparison, not a fresh final test."
        )
    else:
        st.caption(
            "Positive means the original UCI label was 1–4. All test results use a "
            "0.50 threshold and the same 20% holdout. The model was chosen "
            "using only five-fold training cross-validation F2."
        )
    rows = []
    for name, item in report["models"].items():
        holdout = item["holdout"]
        rows.append(
            {
                "Model": name,
                **(
                    {
                        "Features": item["config"]["feature_set"],
                        "Missing values": item["config"]["missing_strategy"],
                    }
                    if "config" in item
                    else {}
                ),
                "CV F2 mean": item["cross_validation"]["f2"]["mean"],
                "Precision": holdout["precision"],
                "Recall": holdout["recall"],
                "Specificity": holdout["specificity"],
                "F2": holdout["f2"],
                "ROC AUC": holdout["roc_auc"],
                "PR AUC": holdout["average_precision"],
                "False negatives": holdout["false_negatives"],
                "False positives": holdout["false_positives"],
            }
        )
    st.dataframe(pd.DataFrame(rows).set_index("Model").round(3), width="stretch")

    selected = report["selected_model"]
    result = report["models"][selected]["holdout"]
    st.subheader(f"Confusion matrix — {selected}")
    st.dataframe(
        pd.DataFrame(
            result["confusion_matrix"],
            index=["Actual: no disease", "Actual: disease"],
            columns=["Predicted: no disease", "Predicted: disease"],
        ),
        width="stretch",
    )
    st.caption(
        "A false negative is a disease-labeled record predicted as no disease. "
        "A false positive is a no-disease record predicted as disease."
    )

    with st.expander("Confusion matrices for all three models"):
        for name, item in report["models"].items():
            st.write(name)
            st.dataframe(
                pd.DataFrame(
                    item["holdout"]["confusion_matrix"],
                    index=["Actual: no disease", "Actual: disease"],
                    columns=["Predicted: no disease", "Predicted: disease"],
                ),
                width="stretch",
            )

    with st.expander("Performance by source dataset"):
        site_rows = []
        for site, metrics in report["models"][selected]["holdout_by_site"].items():
            site_rows.append(
                {
                    "Dataset": site,
                    "Rows": metrics["n"],
                    "Positive cases": metrics["positive_cases"],
                    "Recall": metrics["recall"],
                    "Precision": metrics["precision"],
                    "False negatives": metrics["false_negatives"],
                    "False positives": metrics["false_positives"],
                }
            )
        st.dataframe(pd.DataFrame(site_rows).set_index("Dataset").round(3), width="stretch")
        st.caption("These are small slices of the same holdout, so estimates are uncertain.")

    with st.expander("Unseen source validation on training data"):
        st.write(
            "For each source, a separate copy of the selected model was trained on "
            "the other three sources in the training split. These results show how "
            "performance changes when an entire source is unseen during fitting."
        )
        unseen_rows = []
        for site, metrics in report["models"][selected]["unseen_site_validation"].items():
            unseen_rows.append(
                {
                    "Held-out dataset": site,
                    "Rows": metrics["n"],
                    "Positive cases": metrics["positive_cases"],
                    "Precision": metrics["precision"],
                    "Recall": metrics["recall"],
                    "False negatives": metrics["false_negatives"],
                    "False positives": metrics["false_positives"],
                }
            )
        st.dataframe(pd.DataFrame(unseen_rows).set_index("Held-out dataset").round(3), width="stretch")

    if "search" in report:
        with st.expander("Feature, missing-value, and parameter experiments"):
            search = report["search"]
            st.write(
                f"Compared {len(search['feature_sets'])} feature sets, "
                f"{len(search['missing_strategies'])} missing-value treatments, "
                f"and three parameter settings per model. Each model has "
                f"{search['candidate_count_per_model']} combinations."
            )
            st.write(
                "Median/mode fills missing numeric values with training medians and "
                "categories with training modes. Missing indicator also adds numeric "
                "missingness flags and treats a missing category as its own category."
            )
            candidate_file = ARTIFACTS / "candidate_search.csv"
            if candidate_file.exists():
                candidates = pd.read_csv(candidate_file)
                best_by_preprocessing = (
                    candidates.sort_values("cv_f2_mean", ascending=False)
                    .drop_duplicates(["model", "feature_set", "missing_strategy"])
                    [["model", "feature_set", "missing_strategy", "cv_f2_mean"]]
                    .sort_values(["model", "cv_f2_mean"], ascending=[True, False])
                )
                st.dataframe(best_by_preprocessing.round(3), width="stretch")
                st.caption(
                    "Each number is the best five-fold training CV F2 among the "
                    "three parameter settings for that feature and missing-value choice."
                )
            transfer_file = ARTIFACTS / "transfer_ablation_summary.csv"
            if transfer_file.exists():
                st.write("Hospital transfer check (training records only):")
                transfer = pd.read_csv(transfer_file)
                st.dataframe(
                    transfer[
                        [
                            "model", "feature_set", "missing_strategy",
                            "mean_site_f2", "worst_site_recall",
                            "total_false_negatives", "total_false_positives",
                        ]
                    ].round(3),
                    width="stretch",
                )
                st.caption(
                    "Each row holds out one hospital at a time while keeping the "
                    "model family's selected parameter setting fixed. The best "
                    "within-hospital choice may differ from the best transfer choice."
                )
            st.write("Selected settings by model:")
            for name, item in report["models"].items():
                st.write(f"**{name}:** {item['config']['model_params']}")


def main() -> None:
    st.set_page_config(page_title="Heart disease model demo", page_icon="❤️", layout="wide")
    st.title("Heart disease model demo")
    st.warning(
        "Research demonstration only. This model was trained on historical UCI data "
        "and has not been clinically validated. It cannot diagnose or rule out heart "
        "disease. Seek medical care for symptoms or personal health decisions."
    )

    if not (ARTIFACTS / "heart_model.joblib").exists() or not (ARTIFACTS / "metrics.json").exists():
        st.error("Model files are missing. Run `python train.py` before starting the app.")
        st.stop()
    model = load_model((ARTIFACTS / "heart_model.joblib").stat().st_mtime_ns)
    report = load_report((ARTIFACTS / "metrics.json").stat().st_mtime_ns)
    model_caption = f"Selected model: {report['selected_model']} · "
    if "feature_set" in report:
        model_caption += f"Features: {report['feature_set']} · "
    model_caption += (
        f"Decision threshold: {report['decision_threshold']:.2f} · "
        f"Holdout: {report['split']['test_rows']} records"
    )
    st.caption(model_caption)

    predict_tab, evaluation_tab = st.tabs(["Try a profile", "Model evaluation"])
    with predict_tab:
        st.write(
            "Enter known measurements from an adult profile. Leave unavailable test "
            "results blank; the saved training pipeline handles missing inputs. "
            "Age, sex, and chest-pain category are required."
        )
        with st.form("profile"):
            left, right = st.columns(2)
            with left:
                age = optional_number("Age (years) *", min_value=18.0)
                sex = optional_choice("Sex in dataset *", ["Female", "Male"])
                cp = optional_choice(
                    "Chest-pain category *",
                    ["typical angina", "atypical angina", "non-anginal", "asymptomatic"],
                )
                trestbps = optional_number("Resting blood pressure (mm Hg)", min_value=1.0)
                chol = optional_number("Serum cholesterol (mg/dL)", min_value=1.0)
            with right:
                fbs = optional_choice("Fasting blood sugar >120 mg/dL", ["No", "Yes"])
                restecg = optional_choice(
                    "Resting ECG result",
                    ["normal", "st-t abnormality", "lv hypertrophy"],
                )
                thalch = optional_number("Maximum heart rate achieved (beats/min)", min_value=1.0)
                exang = optional_choice("Exercise-induced angina", ["No", "Yes"])
                oldpeak = optional_number(
                    "ST depression after exercise (oldpeak)", min_value=-3.0, step=0.1
                )
                slope = (
                    optional_choice(
                        "Exercise ST slope",
                        ["upsloping", "flat", "downsloping"],
                    )
                    if "slope" in report["features"]
                    else None
                )
                ca = (
                    optional_number(
                        "Number of major vessels (ca)", min_value=0.0, max_value=3.0
                    )
                    if "ca" in report["features"]
                    else None
                )
                thal = (
                    optional_choice(
                        "Thallium scan result (thal)",
                        ["normal", "fixed defect", "reversible defect"],
                    )
                    if "thal" in report["features"]
                    else None
                )
            submitted = st.form_submit_button("Get model estimate", type="primary")

        if submitted:
            if age is None or sex is None or cp is None:
                st.error("Enter age, sex, and chest-pain category before submitting.")
            else:
                values = {
                    "age": age,
                    "trestbps": trestbps,
                    "chol": chol,
                    "thalch": thalch,
                    "oldpeak": oldpeak,
                    "sex": sex,
                    "cp": cp,
                    "fbs": fbs,
                    "restecg": restecg,
                    "exang": exang,
                }
                if "slope" in report["features"]:
                    values["slope"] = slope
                if "ca" in report["features"]:
                    values["ca"] = ca
                if "thal" in report["features"]:
                    values["thal"] = (
                        "reversable defect" if thal == "reversible defect" else thal
                    )
                profile = pd.DataFrame(
                    [{key: np.nan if value is None else value for key, value in values.items()}],
                    columns=report["features"],
                )
                probability = float(model.predict_proba(profile)[0, 1])
                prediction = int(probability >= report["decision_threshold"])
                if prediction:
                    st.error("Model output: disease label predicted")
                else:
                    st.info("Model output: no-disease label predicted")
                st.metric("Model score for disease label", f"{probability:.1%}")
                st.caption(
                    "This is the model's score for the dataset label. It is not a "
                    "validated personal risk estimate or a medical diagnosis."
                )
                missing_count = int(profile.isna().sum(axis=1).iloc[0])
                if missing_count:
                    st.warning(
                        f"{missing_count} input(s) were unknown and handled by the "
                        "training pipeline. Estimates with missing tests need extra caution."
                    )
                outside = [
                    column
                    for column in ALL_NUMERIC_FEATURES
                    if column in report["features"]
                    if pd.notna(profile.at[0, column])
                    and (
                        profile.at[0, column] < report["training_feature_ranges"][column]["min"]
                        or profile.at[0, column] > report["training_feature_ranges"][column]["max"]
                    )
                ]
                if outside:
                    st.warning(
                        "Outside the observed training range: " + ", ".join(outside)
                        + ". This estimate may be especially unreliable."
                    )

    with evaluation_tab:
        show_evaluation(report)


if __name__ == "__main__":
    main()
