"""Data cleaning and leakage-safe model pipelines for the UCI heart dataset."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier


NUMERIC_FEATURES = ["age", "trestbps", "chol", "thalch", "oldpeak"]
CATEGORICAL_FEATURES = ["sex", "cp", "fbs", "restecg", "exang"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
ALL_NUMERIC_FEATURES = NUMERIC_FEATURES + ["ca"]
ALL_CATEGORICAL_FEATURES = CATEGORICAL_FEATURES + ["slope", "thal"]
FEATURE_SETS = {
    "core_10": FEATURES,
    "plus_slope_11": FEATURES + ["slope"],
    "all_13": FEATURES + ["slope", "ca", "thal"],
}
REQUIRED_COLUMNS = ["id", "dataset", "num", *FEATURES, "slope", "ca", "thal"]


def clean_data(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Return an auditable cleaned table; learned imputation happens in the pipeline."""
    missing_columns = sorted(set(REQUIRED_COLUMNS) - set(raw.columns))
    if missing_columns:
        raise ValueError(f"CSV is missing columns: {missing_columns}")

    data = raw.copy()
    raw_rows = len(data)
    data = data.drop_duplicates(subset=[c for c in data.columns if c != "id"]).copy()
    duplicate_rows = raw_rows - len(data)

    for column in ALL_NUMERIC_FEATURES:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    for column in ["trestbps", "chol", "thalch"]:
        data.loc[data[column] <= 0, column] = float("nan")

    for column in ["fbs", "exang"]:
        # read_csv parses TRUE/FALSE as booleans even in columns with gaps.
        data[column] = data[column].map(
            {True: "Yes", False: "No", "TRUE": "Yes", "FALSE": "No"}
        )
    for column in ["sex", "cp", "restecg", "slope", "thal"]:
        data[column] = data[column].astype("string").str.strip().replace("", pd.NA)
        data[column] = data[column].astype(object).where(data[column].notna(), np.nan)

    target = pd.to_numeric(data["num"], errors="coerce")
    if target.isna().any() or not target.isin([0, 1, 2, 3, 4]).all():
        raise ValueError("Target 'num' must contain only 0, 1, 2, 3, or 4.")
    data["heart_disease"] = (target > 0).astype(int)

    cleaned = data[["dataset", *FEATURE_SETS["all_13"], "heart_disease"]].reset_index(drop=True)
    if cleaned[["age", "sex", "cp"]].isna().any().any():
        raise ValueError("Required age, sex, or chest-pain type is missing.")
    report = {
        "raw_rows": raw_rows,
        "cleaned_rows": len(cleaned),
        "exact_duplicate_rows_removed": duplicate_rows,
        "zero_cholesterol_as_missing": int(raw["chol"].eq(0).sum()),
        "zero_blood_pressure_as_missing": int(raw["trestbps"].eq(0).sum()),
        "missing_after_cleaning": cleaned[FEATURE_SETS["all_13"]].isna().sum().to_dict(),
        "excluded_feature_missing_raw": raw[["slope", "ca", "thal"]].isna().sum().to_dict(),
        "excluded_feature_missing_after_dedup": data[["slope", "ca", "thal"]].isna().sum().to_dict(),
        "positive_rows": int(cleaned["heart_disease"].sum()),
        "positive_label": "num > 0",
        "non_predictor_columns": ["id", "dataset", "num"],
    }
    return cleaned, report


def make_pipeline(
    model_name: str,
    feature_set: str = "core_10",
    missing_strategy: str = "median_mode",
    model_params: dict | None = None,
) -> Pipeline:
    """Build a pipeline for one fully specified experiment candidate."""
    if feature_set not in FEATURE_SETS:
        raise ValueError(f"Unknown feature set: {feature_set}")
    if missing_strategy not in {"median_mode", "missing_indicator"}:
        raise ValueError(f"Unknown missing strategy: {missing_strategy}")
    selected = FEATURE_SETS[feature_set]
    numeric = [column for column in ALL_NUMERIC_FEATURES if column in selected]
    categorical = [column for column in ALL_CATEGORICAL_FEATURES if column in selected]
    numeric_steps = [
        (
            "impute",
            SimpleImputer(
                strategy="median", add_indicator=missing_strategy == "missing_indicator"
            ),
        )
    ]
    if model_name == "Logistic regression":
        numeric_steps.append(("scale", StandardScaler()))
    categorical_imputer = (
        SimpleImputer(strategy="most_frequent")
        if missing_strategy == "median_mode"
        else SimpleImputer(strategy="constant", fill_value="Missing")
    )
    preprocessing = ColumnTransformer(
        transformers=[
            ("numeric", Pipeline(numeric_steps), numeric),
            (
                "categorical",
                Pipeline(
                    [
                        ("impute", categorical_imputer),
                        ("one_hot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical,
            ),
        ]
    )
    models = {
        "Logistic regression": LogisticRegression(max_iter=2000, random_state=42),
        "Random forest": RandomForestClassifier(
            n_estimators=300, min_samples_leaf=3, random_state=42, n_jobs=1
        ),
        "XGBoost": XGBClassifier(
            n_estimators=200,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=5,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=42,
            n_jobs=1,
        ),
    }
    if model_name not in models:
        raise ValueError(f"Unknown model: {model_name}")
    model = models[model_name]
    if model_params:
        model.set_params(**model_params)
    return Pipeline([("preprocess", preprocessing), ("model", model)])
