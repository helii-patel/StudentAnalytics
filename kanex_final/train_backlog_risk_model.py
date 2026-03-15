from __future__ import annotations

from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
GOLD_PATH = DATA_DIR / "gold" / "ml_features_dataset.csv"
MODELS_DIR = BASE_DIR / "models"
PREDICTIONS_DIR = DATA_DIR / "predictions"


BASE_FEATURE_COLUMNS = [
    "cgpa",
    "gpa_1",
    "gpa_2",
    "gpa_3",
    "gpa_4",
    "gpa_5",
    "gpa_6",
    "marks_12th",
    "thispresent",
    "prispresent",
    "daily_study_time",
    "stress_level",
    "social_media_time",
    "travelling_time_",
]

ENGINEERED_FEATURE_COLUMNS = [
    "academic_score",
    "attendance_score",
    "stress_risk",
    "productivity_score",
    "avg_gpa",
    "gpa_variance",
    "gpa_trend",
    "academic_momentum",
    "credit_efficiency",
    "subject_load",
    "wellbeing_balance",
    "employability_signal",
]

FEATURE_COLUMNS = BASE_FEATURE_COLUMNS + ENGINEERED_FEATURE_COLUMNS
ID_COLUMNS = ["student_id", "semester", "subjectid", "noofbacklog"]


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()
    numeric_cols = [col for col in X.columns if col not in categorical_cols]

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_cols),
            ("cat", categorical_pipeline, categorical_cols),
        ]
    )


def pick_best_threshold(
    probabilities: np.ndarray, y_true: pd.Series
) -> tuple[float, dict[str, float]]:
    best_threshold = 0.5
    best_metrics = {"f1": -1.0, "precision": 0.0, "recall": 0.0}

    for threshold in np.arange(0.2, 0.81, 0.02):
        predictions = (probabilities >= threshold).astype(int)
        f1 = f1_score(y_true, predictions, zero_division=0)
        precision = precision_score(y_true, predictions, zero_division=0)
        recall = recall_score(y_true, predictions, zero_division=0)

        if f1 > best_metrics["f1"]:
            best_threshold = float(round(threshold, 2))
            best_metrics = {
                "f1": float(f1),
                "precision": float(precision),
                "recall": float(recall),
            }

    return best_threshold, best_metrics


def evaluate_model(
    model: Pipeline, X_test: pd.DataFrame, y_test: pd.Series, threshold: float
) -> dict[str, object]:
    probabilities = model.predict_proba(X_test)[:, 1]
    predictions = (probabilities >= threshold).astype(int)

    return {
        "threshold": threshold,
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(y_test, predictions, zero_division=0),
        "recall": recall_score(y_test, predictions, zero_division=0),
        "f1": f1_score(y_test, predictions, zero_division=0),
        "roc_auc": roc_auc_score(y_test, probabilities),
        "confusion_matrix": confusion_matrix(y_test, predictions).tolist(),
        "classification_report": classification_report(
            y_test, predictions, output_dict=True, zero_division=0
        ),
    }


def risk_band(probability: float) -> str:
    if probability >= 0.7:
        return "High Risk"
    if probability >= 0.4:
        return "Medium Risk"
    return "No Risk"


def main() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(GOLD_PATH)

    missing_columns = [col for col in FEATURE_COLUMNS + ID_COLUMNS if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    modeling_df = df[ID_COLUMNS + FEATURE_COLUMNS].copy()
    modeling_df["backlog_risk"] = (modeling_df["noofbacklog"] > 0).astype(int)

    X = modeling_df[FEATURE_COLUMNS]
    y = modeling_df["backlog_risk"]
    meta = modeling_df[ID_COLUMNS].copy()

    X_train_full, X_test, y_train_full, y_test, meta_train_full, meta_test = train_test_split(
        X,
        y,
        meta,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    X_train, X_valid, y_train, y_valid, meta_train, meta_valid = train_test_split(
        X_train_full,
        y_train_full,
        meta_train_full,
        test_size=0.25,
        random_state=42,
        stratify=y_train_full,
    )

    preprocessor = build_preprocessor(X)
    candidate_models = {
        "logistic_regression": LogisticRegression(
            max_iter=3000,
            class_weight="balanced",
            random_state=42,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=500,
            random_state=42,
            class_weight="balanced_subsample",
            min_samples_leaf=2,
            n_jobs=1,
        ),
    }

    trained_models: dict[str, Pipeline] = {}
    thresholds: dict[str, float] = {}
    validation_metrics: dict[str, dict[str, float]] = {}
    test_metrics: dict[str, dict[str, object]] = {}

    for model_name, estimator in candidate_models.items():
        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", estimator),
            ]
        )
        pipeline.fit(X_train, y_train)
        trained_models[model_name] = pipeline

        valid_probabilities = pipeline.predict_proba(X_valid)[:, 1]
        threshold, threshold_metrics = pick_best_threshold(valid_probabilities, y_valid)
        thresholds[model_name] = threshold
        validation_metrics[model_name] = {
            "threshold": threshold,
            "f1": threshold_metrics["f1"],
            "precision": threshold_metrics["precision"],
            "recall": threshold_metrics["recall"],
            "roc_auc": roc_auc_score(y_valid, valid_probabilities),
        }
        test_metrics[model_name] = evaluate_model(pipeline, X_test, y_test, threshold)

    best_model_name = max(
        test_metrics,
        key=lambda name: (
            test_metrics[name]["roc_auc"],
            test_metrics[name]["f1"],
        ),
    )
    best_model = trained_models[best_model_name]
    best_threshold = thresholds[best_model_name]

    for model_name, pipeline in trained_models.items():
        joblib.dump(pipeline, MODELS_DIR / f"{model_name}_backlog_risk.joblib")

    all_probabilities = {
        model_name: pipeline.predict_proba(X)[:, 1]
        for model_name, pipeline in trained_models.items()
    }
    best_probabilities = all_probabilities[best_model_name]
    best_predictions = (best_probabilities >= best_threshold).astype(int)

    scored_students = meta.copy()
    scored_students["backlog_risk_actual"] = y
    scored_students["backlog_risk_probability"] = best_probabilities
    scored_students["backlog_risk_prediction"] = best_predictions
    scored_students["risk_label"] = scored_students["backlog_risk_probability"].apply(risk_band)
    scored_students["recommended_intervention"] = np.where(
        scored_students["backlog_risk_probability"] >= 0.7,
        "Immediate mentor follow-up",
        np.where(
            scored_students["backlog_risk_probability"] >= 0.4,
            "Monitor and counsel",
            "Routine monitoring",
        ),
    )
    scored_students["logistic_regression_probability"] = all_probabilities[
        "logistic_regression"
    ]
    scored_students["random_forest_probability"] = all_probabilities["random_forest"]
    scored_students.to_csv(
        PREDICTIONS_DIR / "backlog_risk_predictions.csv", index=False
    )

    test_probabilities = best_model.predict_proba(X_test)[:, 1]
    test_predictions = (test_probabilities >= best_threshold).astype(int)
    test_output = meta_test.copy()
    test_output["backlog_risk_actual"] = y_test.values
    test_output["backlog_risk_probability"] = test_probabilities
    test_output["backlog_risk_prediction"] = test_predictions
    test_output["risk_label"] = test_output["backlog_risk_probability"].apply(risk_band)
    test_output.to_csv(
        PREDICTIONS_DIR / "backlog_risk_test_predictions.csv", index=False
    )

    dashboard_ready = scored_students[
        [
            "student_id",
            "semester",
            "subjectid",
            "backlog_risk_probability",
            "risk_label",
            "recommended_intervention",
        ]
    ].copy()
    dashboard_ready.to_csv(
        PREDICTIONS_DIR / "backlog_risk_dashboard_output.csv", index=False
    )

    summary = {
        "target_definition": "backlog_risk = 1 if noofbacklog > 0 else 0",
        "feature_columns": FEATURE_COLUMNS,
        "split_rows": {
            "train": int(len(X_train)),
            "validation": int(len(X_valid)),
            "test": int(len(X_test)),
        },
        "class_distribution": {
            "no_risk": int((y == 0).sum()),
            "high_risk": int((y == 1).sum()),
        },
        "best_model": best_model_name,
        "best_threshold": best_threshold,
        "validation_metrics": validation_metrics,
        "test_metrics": test_metrics,
        "prediction_outputs": {
            "full": str(PREDICTIONS_DIR / "backlog_risk_predictions.csv"),
            "test": str(PREDICTIONS_DIR / "backlog_risk_test_predictions.csv"),
            "dashboard": str(PREDICTIONS_DIR / "backlog_risk_dashboard_output.csv"),
        },
    }
    with open(MODELS_DIR / "backlog_risk_metrics.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    print(f"Best model: {best_model_name}")
    print(
        "Test metrics: "
        + ", ".join(
            f"{name} roc_auc={test_metrics[name]['roc_auc']:.4f} "
            f"f1={test_metrics[name]['f1']:.4f} "
            f"threshold={test_metrics[name]['threshold']:.2f}"
            for name in test_metrics
        )
    )
    print(f"Saved predictions: {PREDICTIONS_DIR / 'backlog_risk_predictions.csv'}")


if __name__ == "__main__":
    main()
