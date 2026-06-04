from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV

# Menghapus MLFLOW_RUN_ID bawaan dari mlflow run agar tidak bentrok di GitHub Actions
os.environ.pop("MLFLOW_RUN_ID", None)

TARGET_COLUMN = "target"
OUTPUT_DIR = Path("run_outputs")
PIP_REQUIREMENTS = ["mlflow==2.19.0", "pandas", "numpy", "scikit-learn==1.5.2"]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, default="breast_cancer_preprocessing/train.csv")
    parser.add_argument("--test_path", type=str, default="breast_cancer_preprocessing/test.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment("SMSML Workflow CI Retraining")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    train_df = pd.read_csv(args.dataset_path)
    test_df = pd.read_csv(args.test_path)

    X_train = train_df.drop(columns=[TARGET_COLUMN])
    y_train = train_df[TARGET_COLUMN].astype(int)
    X_test = test_df.drop(columns=[TARGET_COLUMN])
    y_test = test_df[TARGET_COLUMN].astype(int)

    grid = GridSearchCV(
        RandomForestClassifier(random_state=42, class_weight="balanced"),
        {
            "n_estimators": [50, 100],
            "max_depth": [4, 8],
            "min_samples_split": [2, 5],
        },
        cv=3,
        scoring="f1",
        n_jobs=1,
    )

    grid.fit(X_train, y_train)

    model = grid.best_estimator_
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred)),
        "recall": float(recall_score(y_test, y_pred)),
        "f1_score": float(f1_score(y_test, y_pred)),
        "roc_auc": float(roc_auc_score(y_test, y_prob)),
        "best_cv_f1": float(grid.best_score_),
    }

    (OUTPUT_DIR / "metrics.json").write_text(
        json.dumps(metrics, indent=2),
        encoding="utf-8",
    )

    (OUTPUT_DIR / "classification_report.json").write_text(
        json.dumps(classification_report(y_test, y_pred, output_dict=True), indent=2),
        encoding="utf-8",
    )

    with mlflow.start_run(run_name="ci_random_forest_retraining") as run:
        mlflow.log_param("model_name", "RandomForestClassifier")
        mlflow.log_params(grid.best_params_)

        for key, value in metrics.items():
            mlflow.log_metric(key, value)

        mlflow.log_artifacts(str(OUTPUT_DIR), artifact_path="reports")

        mlflow.sklearn.log_model(
            model,
            artifact_path="model",
            pip_requirements=PIP_REQUIREMENTS,
        )

        (OUTPUT_DIR / "run_id.txt").write_text(run.info.run_id, encoding="utf-8")

        print("CI retraining completed successfully")
        print(f"run_id={run.info.run_id}")
        print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
