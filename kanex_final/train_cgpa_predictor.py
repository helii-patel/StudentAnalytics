from __future__ import annotations

from pathlib import Path
import json

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "gold" / "final_dataset.csv"
MODELS_DIR = BASE_DIR / "models"

FEATURE_COLUMNS = [
    "marks_10th",
    "marks_12th",
    "current_back",
    "ever_back",
    "gender",
    "branch",
    "category",
    "study_hours_per_day",
    "attendance_percentage",
    "sleep_hours",
    "parental_support_level",
    "motivation_level",
    "exam_anxiety_score",
    "mental_health_rating",
    "social_media_hours",
    "screen_time",
    "exercise_frequency",
    "diet_quality",
]
TARGET_COLUMN = "cgpa"


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


def main() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(DATA_PATH)
    missing = [col for col in FEATURE_COLUMNS + [TARGET_COLUMN] if col not in df.columns]
    if missing:
        raise ValueError(f"Missing columns for CGPA training: {missing}")

    X = df[FEATURE_COLUMNS].copy()
    y = pd.to_numeric(df[TARGET_COLUMN], errors="coerce")

    valid_mask = y.notna()
    X = X.loc[valid_mask].copy()
    y = y.loc[valid_mask].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", build_preprocessor(X)),
            (
                "model",
                RandomForestRegressor(
                    n_estimators=160,
                    random_state=42,
                    n_jobs=1,
                    min_samples_leaf=2,
                ),
            ),
        ]
    )
    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_test)
    rmse = mean_squared_error(y_test, predictions) ** 0.5
    metrics = {
        "model_type": "RandomForestRegressor",
        "features": FEATURE_COLUMNS,
        "target": TARGET_COLUMN,
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "mae": float(mean_absolute_error(y_test, predictions)),
        "rmse": float(rmse),
        "r2": float(r2_score(y_test, predictions)),
    }

    joblib.dump(pipeline, MODELS_DIR / "cgpa_predictor.joblib")
    with open(MODELS_DIR / "cgpa_predictor_metrics.json", "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2)

    print(
        f"CGPA predictor trained. R2={metrics['r2']:.4f}, "
        f"MAE={metrics['mae']:.4f}, RMSE={metrics['rmse']:.4f}"
    )


if __name__ == "__main__":
    main()
