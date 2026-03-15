from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, jsonify, request

app = Flask(__name__)

@app.route("/")
def home():
    return "Flask API is running"
BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
MODEL_PATH = PROJECT_ROOT / "CGPApredictionModels" / "c1.pkl"
FEATURES_PATH = PROJECT_ROOT / "CGPApredictionModels" / "f1.pkl"
IMPUTER_PATH = PROJECT_ROOT / "CGPApredictionModels" / "i1.pkl"
SCALER_PATH = PROJECT_ROOT / "CGPApredictionModels" / "s1.pkl"
ENCODERS_PATH = PROJECT_ROOT / "CGPApredictionModels" / "l1.pkl"

INPUT_FIELDS = [
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

NUMERIC_INPUT_FIELDS = {
    "marks_10th",
    "marks_12th",
    "current_back",
    "ever_back",
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
}

model = None
feature_order = None
imputer = None
scaler = None
saved_encoders = None


def load_model():
    global model
    if model is None:
        model = joblib.load(MODEL_PATH)
    return model


def load_feature_order() -> list[str]:
    global feature_order
    if feature_order is None:
        feature_order = joblib.load(FEATURES_PATH)
    return feature_order


def load_imputer():
    global imputer
    if imputer is None:
        imputer = joblib.load(IMPUTER_PATH)
    return imputer


def load_scaler():
    global scaler
    if scaler is None:
        scaler = joblib.load(SCALER_PATH)
    return scaler


def load_saved_encoders() -> dict:
    global saved_encoders
    if saved_encoders is None:
        saved_encoders = joblib.load(ENCODERS_PATH)
    return saved_encoders


def prepare_model_input(payload: dict[str, object]) -> pd.DataFrame:
    # Backward-compatible aliases for older frontends/datasets.
    field_aliases = {
        "current_back": ["current_back", "noofbacklog", "total_backlogs"],
        "ever_back": ["ever_back", "history_of_backlogs", "backlog_history"],
    }
    normalized_payload = dict(payload)
    for canonical, aliases in field_aliases.items():
        if canonical in normalized_payload:
            continue
        for alias in aliases:
            if alias in normalized_payload:
                normalized_payload[canonical] = normalized_payload[alias]
                break

    missing = [field for field in INPUT_FIELDS if field not in normalized_payload]
    if missing:
        raise KeyError(f"Missing fields: {missing}")

    row = {}
    for field in load_feature_order():
        value = normalized_payload[field]
        row[field] = float(value) if field in NUMERIC_INPUT_FIELDS else str(value)

    input_df = pd.DataFrame([row], columns=load_feature_order())

    encoders = load_saved_encoders()
    for col, encoder in encoders.items():
        try:
            input_df[col] = encoder.transform(input_df[col].astype(str))
        except ValueError as exc:
            allowed = list(getattr(encoder, "classes_", []))
            raise ValueError(
                f"Invalid value '{input_df.at[0, col]}' for {col}. Allowed values: {allowed}"
            ) from exc

    numeric_cols = list(load_scaler().feature_names_in_)
    input_df[numeric_cols] = load_scaler().transform(input_df[numeric_cols])
    return input_df


@app.get("/health")
def health():
    return jsonify(
        {
            "status": "ok",
            "model_path": str(MODEL_PATH),
            "features_path": str(FEATURES_PATH),
            "imputer_path": str(IMPUTER_PATH),
            "scaler_path": str(SCALER_PATH),
            "encoders_path": str(ENCODERS_PATH),
            "required_input_fields": INPUT_FIELDS,
            "model_feature_count": len(load_model().feature_names_in_),
            "frontend_input_feature_count": len(load_feature_order()),
        }
    )


@app.get("/metadata")
def metadata():
    encoders = load_saved_encoders()
    categorical_options = {
        column: list(getattr(encoder, "classes_", []))
        for column, encoder in encoders.items()
    }
    return jsonify(
        {
            "required_input_fields": INPUT_FIELDS,
            "numeric_input_fields": sorted(NUMERIC_INPUT_FIELDS),
            "categorical_options": categorical_options,
        }
    )


@app.post("/predict/cgpa")
def predict_cgpa():
    payload = request.get_json(silent=True) or {}
    try:
        input_df = prepare_model_input(payload)
    except KeyError as exc:
        message = str(exc)
        return jsonify({"error": "Missing fields", "details": message}), 400
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    prediction = float(load_model().predict(input_df)[0])
    return jsonify(
        {
            "predicted_cgpa": round(prediction, 4),
            "input_features": {key: payload[key] for key in INPUT_FIELDS},
            "model_used": str(MODEL_PATH),
        }
    )


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
