"""Train, compare, and save heart-disease classifiers.

Usage: python train.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    fbeta_score,
    precision_score,
    recall_score,
    roc_auc_score,
    make_scorer,
)
from sklearn.model_selection import (
    LeaveOneGroupOut,
    StratifiedKFold,
    cross_validate,
    train_test_split,
)

from heart_model import FEATURES, clean_data, make_pipeline


MODEL_NAMES = ["Logistic regression", "Random forest", "XGBoost"]


def evaluate(y_true, probabilities) -> dict:
    predictions = (np.asarray(probabilities) >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, predictions, labels=[0, 1]).ravel()
    return {
        "n": int(len(y_true)),
        "positive_cases": int(np.sum(y_true)),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "recall": float(recall_score(y_true, predictions, zero_division=0)),
        "specificity": float(tn / (tn + fp)) if tn + fp else None,
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
        "f2": float(fbeta_score(y_true, predictions, beta=2, zero_division=0)),
        "accuracy": float(accuracy_score(y_true, predictions)),
        "roc_auc": float(roc_auc_score(y_true, probabilities))
        if len(np.unique(y_true)) == 2
        else None,
        "average_precision": float(average_precision_score(y_true, probabilities))
        if len(np.unique(y_true)) == 2
        else None,
        "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("heart_disease_uci.csv"))
    parser.add_argument("--output", type=Path, default=Path("artifacts"))
    args = parser.parse_args()

    cleaned, cleaning_report = clean_data(pd.read_csv(args.data, sep=";"))
    args.output.mkdir(parents=True, exist_ok=True)
    cleaned.to_csv(args.output / "heart_disease_cleaned.csv", index=False)

    X = cleaned[FEATURES]
    y = cleaned["heart_disease"]
    sites = cleaned["dataset"]
    # Preserve both site and outcome proportions in the untouched holdout.
    strata = sites.astype(str) + "_" + y.astype(str)
    X_train, X_test, y_train, y_test, site_train, site_test = train_test_split(
        X, y, sites, test_size=0.2, random_state=42, stratify=strata
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scoring = {
        "f2": make_scorer(fbeta_score, beta=2, zero_division=0),
        "precision": make_scorer(precision_score, zero_division=0),
        "recall": make_scorer(recall_score, zero_division=0),
        "specificity": make_scorer(recall_score, pos_label=0, zero_division=0),
        "roc_auc": "roc_auc",
        "average_precision": "average_precision",
    }
    results = {}
    fitted = {}
    for name in MODEL_NAMES:
        pipeline = make_pipeline(name)
        scores = cross_validate(
            pipeline, X_train, y_train, scoring=scoring, cv=cv, n_jobs=1
        )
        cv_scores = {
            metric: {
                "mean": float(np.mean(scores[f"test_{metric}"])),
                "std": float(np.std(scores[f"test_{metric}"], ddof=1)),
            }
            for metric in scoring
        }
        # Stress test transfer to a source hospital absent from fitting. These
        # diagnostic models use only the training split and do not choose the model.
        unseen_site = {}
        for fit_idx, validation_idx in LeaveOneGroupOut().split(
            X_train, y_train, groups=site_train
        ):
            diagnostic_model = make_pipeline(name)
            diagnostic_model.fit(X_train.iloc[fit_idx], y_train.iloc[fit_idx])
            site = site_train.iloc[validation_idx[0]]
            site_probabilities = diagnostic_model.predict_proba(
                X_train.iloc[validation_idx]
            )[:, 1]
            unseen_site[site] = evaluate(
                y_train.iloc[validation_idx], site_probabilities
            )
        pipeline.fit(X_train, y_train)
        probabilities = pipeline.predict_proba(X_test)[:, 1]
        holdout = evaluate(y_test, probabilities)
        by_site = {}
        for site in sorted(site_test.unique()):
            mask = site_test.eq(site).to_numpy()
            by_site[site] = evaluate(y_test.to_numpy()[mask], probabilities[mask])
        results[name] = {
            "cross_validation": cv_scores,
            "unseen_site_validation": unseen_site,
            "holdout": holdout,
            "holdout_by_site": by_site,
        }
        fitted[name] = pipeline

    # Select without consulting holdout performance. F2 weights recall more heavily.
    selected = max(
        MODEL_NAMES,
        key=lambda name: (
            results[name]["cross_validation"]["f2"]["mean"],
            results[name]["cross_validation"]["recall"]["mean"],
        ),
    )
    joblib.dump(fitted[selected], args.output / "heart_model.joblib")

    report = {
        "data_file": str(args.data),
        "cleaning": cleaning_report,
        "features": FEATURES,
        "training_feature_ranges": {
            column: {
                "min": float(X_train[column].min()),
                "max": float(X_train[column].max()),
            }
            for column in ["age", "trestbps", "chol", "thalch", "oldpeak"]
        },
        "split": {
            "train_rows": len(X_train),
            "test_rows": len(X_test),
            "train_positive": int(y_train.sum()),
            "test_positive": int(y_test.sum()),
            "test_fraction": 0.2,
            "random_state": 42,
            "stratified_by": "dataset and binary target",
        },
        "decision_threshold": 0.5,
        "selection_rule": "Highest 5-fold training CV F2 at threshold 0.5; recall breaks ties",
        "selected_model": selected,
        "models": results,
    }
    (args.output / "metrics.json").write_text(
        json.dumps(report, indent=2, allow_nan=False), encoding="utf-8"
    )
    comparison = pd.DataFrame(
        [
            {
                "model": name,
                "cv_f2_mean": results[name]["cross_validation"]["f2"]["mean"],
                "cv_f2_std": results[name]["cross_validation"]["f2"]["std"],
                **{
                    key: results[name]["holdout"][key]
                    for key in [
                        "precision", "recall", "specificity", "f1", "f2",
                        "roc_auc", "average_precision", "false_negatives",
                        "false_positives", "true_positives", "true_negatives",
                    ]
                },
            }
            for name in MODEL_NAMES
        ]
    )
    comparison.to_csv(args.output / "model_comparison.csv", index=False)
    print(f"Cleaned {cleaning_report['raw_rows']} rows to {len(cleaned)} rows")
    print(f"Training: {len(X_train)} | holdout: {len(X_test)}")
    print(comparison.round(3).to_string(index=False))
    print(f"Selected: {selected} (training CV F2)")
    print(f"Saved artifacts to {args.output.resolve()}")


if __name__ == "__main__":
    main()
