"""Compare feature and missing-value choices when each hospital is unseen.

Run after tune.py: python transfer_ablation.py
"""

from __future__ import annotations

import argparse
import json
from itertools import product
from pathlib import Path

import pandas as pd
from sklearn.model_selection import LeaveOneGroupOut, train_test_split

from heart_model import FEATURE_SETS, clean_data, make_pipeline
from train import MODEL_NAMES, evaluate
from tune import MISSING_STRATEGIES


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("heart_disease_uci.csv"))
    parser.add_argument("--output", type=Path, default=Path("artifacts"))
    args = parser.parse_args()
    report = json.loads((args.output / "metrics.json").read_text(encoding="utf-8"))
    if "search" not in report:
        raise ValueError("Run python tune.py before this transfer check.")
    cleaned, _ = clean_data(pd.read_csv(args.data, sep=";"))
    X = cleaned[FEATURE_SETS["all_13"]]
    y = cleaned["heart_disease"]
    sites = cleaned["dataset"]
    strata = sites.astype(str) + "_" + y.astype(str)
    X_train, _, y_train, _, site_train, _ = train_test_split(
        X, y, sites, test_size=0.2, random_state=42, stratify=strata
    )

    rows = []
    for model_name in MODEL_NAMES:
        parameters = report["models"][model_name]["config"]["model_params"]
        for feature_set, missing_strategy in product(
            FEATURE_SETS, MISSING_STRATEGIES
        ):
            for fit_idx, validation_idx in LeaveOneGroupOut().split(
                X_train, y_train, groups=site_train
            ):
                model = make_pipeline(
                    model_name, feature_set, missing_strategy, parameters
                )
                model.fit(X_train.iloc[fit_idx], y_train.iloc[fit_idx])
                probabilities = model.predict_proba(
                    X_train.iloc[validation_idx]
                )[:, 1]
                metrics = evaluate(y_train.iloc[validation_idx], probabilities)
                rows.append(
                    {
                        "model": model_name,
                        "feature_set": feature_set,
                        "missing_strategy": missing_strategy,
                        "held_out_site": site_train.iloc[validation_idx[0]],
                        **{
                            key: metrics[key]
                            for key in [
                                "n", "positive_cases", "precision", "recall",
                                "specificity", "f2", "false_negatives",
                                "false_positives", "true_positives", "true_negatives",
                            ]
                        },
                    }
                )
    table = pd.DataFrame(rows)
    table.to_csv(args.output / "transfer_ablation_by_site.csv", index=False)
    summary = (
        table.groupby(["model", "feature_set", "missing_strategy"], as_index=False)
        .agg(
            mean_site_f2=("f2", "mean"),
            mean_site_recall=("recall", "mean"),
            worst_site_recall=("recall", "min"),
            mean_site_specificity=("specificity", "mean"),
            total_false_negatives=("false_negatives", "sum"),
            total_false_positives=("false_positives", "sum"),
        )
    )
    summary.to_csv(args.output / "transfer_ablation_summary.csv", index=False)
    print(summary.round(3).to_string(index=False))
    print(f"Saved transfer checks to {args.output.resolve()}")


if __name__ == "__main__":
    main()
