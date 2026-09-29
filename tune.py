"""Search feature, imputation, and model settings without using the holdout.

Run from the project directory: python tune.py
"""

from __future__ import annotations

import argparse
import json
from itertools import product
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import fbeta_score, make_scorer
from sklearn.model_selection import (
    LeaveOneGroupOut,
    StratifiedKFold,
    cross_val_score,
    train_test_split,
)

from heart_model import (
    ALL_NUMERIC_FEATURES,
    FEATURE_SETS,
    clean_data,
    make_pipeline,
)
from train import MODEL_NAMES, evaluate


MISSING_STRATEGIES = ["median_mode", "missing_indicator"]
PARAMETER_SETTINGS = {
    "Logistic regression": [
        {"C": 0.1},
        {"C": 1.0},
        {"C": 10.0},
    ],
    "Random forest": [
        {"n_estimators": 300, "min_samples_leaf": 3, "max_depth": None},
        {"n_estimators": 300, "min_samples_leaf": 5, "max_depth": 6},
        {"n_estimators": 300, "min_samples_leaf": 1, "max_depth": 10},
    ],
    "XGBoost": [
        {
            "n_estimators": 200,
            "max_depth": 3,
            "learning_rate": 0.05,
            "min_child_weight": 1,
            "reg_lambda": 5,
        },
        {
            "n_estimators": 300,
            "max_depth": 2,
            "learning_rate": 0.04,
            "min_child_weight": 3,
            "reg_lambda": 10,
        },
        {
            "n_estimators": 150,
            "max_depth": 4,
            "learning_rate": 0.05,
            "min_child_weight": 2,
            "reg_lambda": 5,
        },
    ],
}


def candidates_for(model_name: str) -> list[dict]:
    candidates = []
    for feature_set, missing_strategy, model_params in product(
        FEATURE_SETS, MISSING_STRATEGIES, PARAMETER_SETTINGS[model_name]
    ):
        candidates.append(
            {
                "model": model_name,
                "feature_set": feature_set,
                "missing_strategy": missing_strategy,
                "model_params": model_params,
                "pipeline": make_pipeline(
                    model_name, feature_set, missing_strategy, model_params
                ),
            }
        )
    return candidates


def search_candidates(
    candidates: list[dict], X: pd.DataFrame, y: pd.Series, cv: StratifiedKFold
) -> tuple[dict, list[dict]]:
    scorer = make_scorer(fbeta_score, beta=2, zero_division=0)
    rows = []
    for candidate in candidates:
        scores = cross_val_score(
            candidate["pipeline"], X, y, cv=cv, scoring=scorer, n_jobs=1
        )
        rows.append(
            {
                "model": candidate["model"],
                "feature_set": candidate["feature_set"],
                "missing_strategy": candidate["missing_strategy"],
                "model_params": candidate["model_params"],
                "cv_f2_mean": float(np.mean(scores)),
                "cv_f2_std": float(np.std(scores, ddof=1)),
            }
        )
    best_index = max(
        range(len(rows)),
        key=lambda index: (rows[index]["cv_f2_mean"], -rows[index]["cv_f2_std"]),
    )
    return candidates[best_index], rows


def site_validation(
    candidate: dict, X: pd.DataFrame, y: pd.Series, sites: pd.Series
) -> dict:
    by_site = {}
    for fit_idx, validation_idx in LeaveOneGroupOut().split(X, y, groups=sites):
        model = clone(candidate["pipeline"])
        model.fit(X.iloc[fit_idx], y.iloc[fit_idx])
        probabilities = model.predict_proba(X.iloc[validation_idx])[:, 1]
        by_site[sites.iloc[validation_idx[0]]] = evaluate(
            y.iloc[validation_idx], probabilities
        )
    return by_site


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("heart_disease_uci.csv"))
    parser.add_argument("--output", type=Path, default=Path("artifacts"))
    args = parser.parse_args()

    cleaned, cleaning_report = clean_data(pd.read_csv(args.data, sep=";"))
    X = cleaned[FEATURE_SETS["all_13"]]
    y = cleaned["heart_disease"]
    sites = cleaned["dataset"]
    strata = sites.astype(str) + "_" + y.astype(str)
    X_train, X_test, y_train, y_test, site_train, site_test = train_test_split(
        X, y, sites, test_size=0.2, random_state=42, stratify=strata
    )

    outer_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=17)
    inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=29)
    final_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    model_results = {}
    candidate_rows = []
    outer_rows = []
    final_models = {}

    for model_name in MODEL_NAMES:
        candidates = candidates_for(model_name)
        print(f"{model_name}: {len(candidates)} combinations", flush=True)
        outer_metrics = []
        for fold, (fit_idx, validation_idx) in enumerate(
            outer_cv.split(X_train, y_train), start=1
        ):
            X_fit, y_fit = X_train.iloc[fit_idx], y_train.iloc[fit_idx]
            best, _ = search_candidates(candidates, X_fit, y_fit, inner_cv)
            model = clone(best["pipeline"])
            model.fit(X_fit, y_fit)
            metrics = evaluate(
                y_train.iloc[validation_idx],
                model.predict_proba(X_train.iloc[validation_idx])[:, 1],
            )
            outer_metrics.append(metrics)
            outer_rows.append(
                {
                    "model": model_name,
                    "fold": fold,
                    "feature_set": best["feature_set"],
                    "missing_strategy": best["missing_strategy"],
                    "model_params": json.dumps(best["model_params"], sort_keys=True),
                    **{key: metrics[key] for key in ["n", "f2", "precision", "recall", "false_negatives", "false_positives"]},
                }
            )
            print(f"  outer fold {fold}: F2={metrics['f2']:.3f}", flush=True)

        best, rows = search_candidates(candidates, X_train, y_train, final_cv)
        candidate_rows.extend(rows)
        fitted = clone(best["pipeline"])
        fitted.fit(X_train, y_train)
        probabilities = fitted.predict_proba(X_test)[:, 1]
        holdout = evaluate(y_test, probabilities)
        by_site = {}
        for site in sorted(site_test.unique()):
            mask = site_test.eq(site).to_numpy()
            by_site[site] = evaluate(y_test.to_numpy()[mask], probabilities[mask])

        selected_row = next(
            row
            for row in rows
            if row["feature_set"] == best["feature_set"]
            and row["missing_strategy"] == best["missing_strategy"]
            and row["model_params"] == best["model_params"]
        )
        outer_f2 = [metrics["f2"] for metrics in outer_metrics]
        nested_summary = {
            "f2": {
                "mean": float(np.mean(outer_f2)),
                "std": float(np.std(outer_f2, ddof=1)),
            },
            "folds": outer_metrics,
        }
        model_results[model_name] = {
            "config": {
                "feature_set": best["feature_set"],
                "features": FEATURE_SETS[best["feature_set"]],
                "missing_strategy": best["missing_strategy"],
                "model_params": best["model_params"],
            },
            "nested_cv": nested_summary,
            "cross_validation": {"f2": nested_summary["f2"]},
            "final_search_cv_f2": {
                "mean": selected_row["cv_f2_mean"],
                "std": selected_row["cv_f2_std"],
            },
            "holdout": holdout,
            "holdout_by_site": by_site,
            "unseen_site_validation": site_validation(
                best, X_train, y_train, site_train
            ),
        }
        final_models[model_name] = fitted
        print(
            f"  final: {best['feature_set']}, {best['missing_strategy']}, "
            f"nested F2={nested_summary['f2']['mean']:.3f}, "
            f"holdout FN={holdout['false_negatives']}",
            flush=True,
        )

    # Model family is chosen by outer CV. The holdout does not choose features,
    # imputation, parameters, model family, or the decision threshold.
    selected = max(
        MODEL_NAMES,
        key=lambda name: (
            model_results[name]["nested_cv"]["f2"]["mean"],
            -model_results[name]["nested_cv"]["f2"]["std"],
        ),
    )
    selected_config = model_results[selected]["config"]
    args.output.mkdir(parents=True, exist_ok=True)
    joblib.dump(final_models[selected], args.output / "heart_model.joblib")
    cleaned.to_csv(args.output / "heart_disease_cleaned.csv", index=False)

    candidate_table = pd.DataFrame(candidate_rows)
    candidate_table["model_params"] = candidate_table["model_params"].map(
        lambda value: json.dumps(value, sort_keys=True)
    )
    candidate_table.sort_values(
        ["model", "cv_f2_mean"], ascending=[True, False]
    ).to_csv(args.output / "candidate_search.csv", index=False)
    pd.DataFrame(outer_rows).to_csv(args.output / "nested_cv_folds.csv", index=False)

    comparison = pd.DataFrame(
        [
            {
                "model": name,
                "feature_set": model_results[name]["config"]["feature_set"],
                "missing_strategy": model_results[name]["config"]["missing_strategy"],
                "nested_cv_f2_mean": model_results[name]["nested_cv"]["f2"]["mean"],
                "nested_cv_f2_std": model_results[name]["nested_cv"]["f2"]["std"],
                "final_search_cv_f2_mean": model_results[name]["final_search_cv_f2"]["mean"],
                **{
                    key: model_results[name]["holdout"][key]
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

    report = {
        "data_file": str(args.data),
        "cleaning": cleaning_report,
        "features": selected_config["features"],
        "feature_set": selected_config["feature_set"],
        "missing_strategy": selected_config["missing_strategy"],
        "training_feature_ranges": {
            column: {
                "min": float(X_train[column].min()),
                "max": float(X_train[column].max()),
            }
            for column in selected_config["features"]
            if column in ALL_NUMERIC_FEATURES
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
        "search": {
            "candidate_count_per_model": len(candidates_for(MODEL_NAMES[0])),
            "feature_sets": FEATURE_SETS,
            "missing_strategies": MISSING_STRATEGIES,
            "parameter_settings": PARAMETER_SETTINGS,
            "outer_folds": 3,
            "inner_folds": 3,
            "final_search_folds": 5,
            "scoring": "F2 at decision threshold 0.5",
        },
        "decision_threshold": 0.5,
        "selection_rule": "Highest mean outer nested-CV F2; lower fold SD breaks ties",
        "holdout_status": "Previously inspected during baseline development; descriptive comparison, not a fresh final test",
        "selected_model": selected,
        "models": model_results,
    }
    (args.output / "metrics.json").write_text(
        json.dumps(report, indent=2, allow_nan=False), encoding="utf-8"
    )
    print("\nTuned model comparison:")
    print(comparison.round(3).to_string(index=False))
    print(f"Selected: {selected} | {selected_config}")
    print(f"Saved artifacts to {args.output.resolve()}")


if __name__ == "__main__":
    main()
