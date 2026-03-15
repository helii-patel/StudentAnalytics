from __future__ import annotations

from pathlib import Path
import io
import json
import subprocess

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go


BASE_DIR = Path(__file__).resolve().parent
RAW_DIR = BASE_DIR / "data" / "raw"
DATA_PATH = BASE_DIR / "data" / "gold" / "ml_features_dataset.csv"
ETL_SCRIPT = BASE_DIR / "etl_pipeline.py"
ETL_STATUS_PATH = BASE_DIR / "data" / "dashboard" / "etl_status.json"
PROJECT_ROOT = BASE_DIR.parent
MODEL_PATH = PROJECT_ROOT / "CGPApredictionModels" / "c1.pkl"
FEATURES_PATH = PROJECT_ROOT / "CGPApredictionModels" / "f1.pkl"
IMPUTER_PATH = PROJECT_ROOT / "CGPApredictionModels" / "i1.pkl"
SCALER_PATH = PROJECT_ROOT / "CGPApredictionModels" / "s1.pkl"
ENCODERS_PATH = PROJECT_ROOT / "CGPApredictionModels" / "l1.pkl"
CAREER_MODEL_PATH = PROJECT_ROOT / "CGPApredictionModels" / "random_forest_career_prediction_model.pkl"
CAREER_CLASS_DISPLAY = {
    0: "Higher Education",
    1: "Entrepreneurship",
    2: "Job / Placement",
    3: "Other / Undecided",
}
CACHE_SCHEMA_VERSION = "2026-03-15-v3"

STUDY_ORDER = ["0-30_min", "30-60_min", "1-2_hour", "2+_hour", "other"]
STRESS_ORDER = ["fabulous", "good", "bad", "awful"]
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

sns.set_theme(style="whitegrid")


def _first_existing_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for col in candidates:
        if col in df.columns:
            return col
    return None


def _normalize_schema(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize column names and fill essential defaults so downstream charts are stable."""
    df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]
    lower_to_actual = {str(c).strip().lower(): c for c in df.columns}

    alias_groups = {
        "current_back": ["current_back", "noofbacklog", "current_backlog", "total_backlogs"],
        "noofbacklog": ["noofbacklog", "current_back", "current_backlog", "total_backlogs"],
        "ever_back": ["ever_back", "history_of_backlogs", "backlog_history"],
        "stress_level": ["stress_level", "stresslevel", "stress"],
        "daily_study_time": ["daily_study_time", "study_time"],
        "social_media_time": ["social_media_time", "social_media_usage"],
    }
    for target, candidates in alias_groups.items():
        if target in df.columns:
            continue
        source = _first_existing_column(df, candidates)
        if not source:
            for candidate in candidates:
                source = lower_to_actual.get(candidate.lower())
                if source:
                    break
        if source:
            df[target] = df[source]

    defaults: dict[str, object] = {
        "current_back": 0,
        "noofbacklog": pd.NA,
        "ever_back": 0,
        "attendance_percentage": 0,
        "study_hours_per_day": 0,
        "social_media_hours": 0,
        "exam_score": pd.NA,
        "gpa_1": pd.NA,
        "gpa_6": pd.NA,
        "semester": 1,
        "stress_level": "good",
        "daily_study_time": "other",
        "social_media_time": "other",
        "cgpa": pd.NA,
        "gender": "Unknown",
        "branch": "Unknown",
        "category": "GEN",
        "diet_quality": "average",
        "technical_projects": 0,
        "tech_quiz": 0,
        "certification_course": "no",
        "exam_anxiety_score": 0,
        "mental_health_rating": 0,
        "screen_time": 0,
        "exercise_frequency": 0,
    }
    for col, value in defaults.items():
        if col not in df.columns:
            df[col] = value

    if "noofbacklog" not in df.columns:
        candidate = lower_to_actual.get("no_of_backlog") or lower_to_actual.get("backlogs")
        df["noofbacklog"] = df[candidate] if candidate and candidate in df.columns else 0

    if "current_back" not in df.columns:
        df["current_back"] = df.get("noofbacklog", 0)

    if "ever_back" not in df.columns:
        df["ever_back"] = 0

    return df


def _apply_common_transforms(df: pd.DataFrame) -> pd.DataFrame:
    """Apply light feature engineering for charts and predictions."""
    df = _normalize_schema(df)

    # Effective CGPA: prefer cgpa, then gpa_6, then gpa_1, finally exam_score/10.
    cgpa_candidates = [
        pd.to_numeric(df.get("cgpa"), errors="coerce"),
        pd.to_numeric(df.get("gpa_6"), errors="coerce"),
        pd.to_numeric(df.get("gpa_1"), errors="coerce"),
        pd.to_numeric(df.get("exam_score"), errors="coerce") / 10,
    ]
    effective_cgpa = None
    for series in cgpa_candidates:
        if series is not None:
            effective_cgpa = series if effective_cgpa is None else effective_cgpa.fillna(series)
    if effective_cgpa is None:
        df["effective_cgpa"] = pd.Series([0] * len(df))
    else:
        df["effective_cgpa"] = effective_cgpa.fillna(0)

    numeric_cols = [
        "study_hours_per_day",
        "attendance_percentage",
        "social_media_hours",
        "exam_anxiety_score",
        "mental_health_rating",
        "screen_time",
        "exercise_frequency",
        "current_back",
        "ever_back",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    # Study efficiency normalized to an 8-hour baseline: (marks/100) * (8 / study_hours) * 100.
    # Marks prefers exam_score; fallback to effective_cgpa*10.
    hours = pd.to_numeric(df.get("study_hours_per_day", 0), errors="coerce").replace(0, pd.NA)
    marks = pd.to_numeric(df.get("exam_score"), errors="coerce")
    marks = marks.where(marks.notna(), df["effective_cgpa"] * 10)
    df["study_efficiency"] = ((marks / 100) * (8 / hours) * 100).fillna(0).clip(0, 100)

    # Distraction index: time on social media per study hour.
    sm_hours = pd.to_numeric(df.get("social_media_hours", 0), errors="coerce")
    df["distraction_index"] = (sm_hours / hours).fillna(0)

    # Discipline score: blend attendance (60%) and normalized study hours (40%).
    attendance_norm = pd.to_numeric(df.get("attendance_percentage", 0), errors="coerce").fillna(0).clip(0, 100) / 100
    study_norm = pd.to_numeric(df.get("study_hours_per_day", 0), errors="coerce").fillna(0)
    study_norm = (study_norm / 8).clip(0, 1)
    df["discipline_score"] = (attendance_norm * 0.6 + study_norm * 0.4) * 100

    return df


def _career_model_mtime() -> float | None:
    if CAREER_MODEL_PATH.exists():
        return CAREER_MODEL_PATH.stat().st_mtime
    return None


@st.cache_resource
def load_prediction_assets():
    model = joblib.load(MODEL_PATH)
    features = joblib.load(FEATURES_PATH)
    imputer = joblib.load(IMPUTER_PATH) if IMPUTER_PATH.exists() else None
    scaler = joblib.load(SCALER_PATH)
    encoders = joblib.load(ENCODERS_PATH)
    return {
        "model": model,
        "features": list(features),
        "imputer": imputer,
        "scaler": scaler,
        "encoders": encoders,
    }


@st.cache_data(show_spinner=False)
def load_cgpa_metrics() -> dict[str, float] | None:
    metrics_path = BASE_DIR / "models" / "cgpa_predictor_metrics.json"
    if not metrics_path.exists():
        return None
    try:
        with open(metrics_path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:
        return None


def prepare_model_input(payload: dict[str, object], assets: dict[str, object]) -> pd.DataFrame:
    df = pd.DataFrame([payload])
    encoders: dict[str, object] = assets.get("encoders", {})
    for col, encoder in encoders.items():
        if col in df.columns:
            try:
                df[col] = encoder.transform(df[col].astype(str))
            except Exception:
                fallback = encoder.classes_[0] if hasattr(encoder, "classes_") else df[col].iloc[0]
                df[col] = encoder.transform(pd.Series([fallback]))

    features: list[str] = assets.get("features", [])
    for col in features:
        if col not in df.columns:
            df[col] = 0
    df = df[features].copy()

    numeric_cols = []
    scaler = assets.get("scaler")
    imputer = assets.get("imputer")
    if scaler is not None and getattr(scaler, "feature_names_in_", None) is not None:
        numeric_cols = list(scaler.feature_names_in_)
    elif imputer is not None and getattr(imputer, "feature_names_in_", None) is not None:
        numeric_cols = list(imputer.feature_names_in_)

    if numeric_cols:
        numeric_df = df[numeric_cols].copy()
        if imputer is not None:
            numeric_df = pd.DataFrame(imputer.transform(numeric_df), columns=numeric_cols)
        else:
            numeric_df = numeric_df.fillna(0)
        if scaler is not None:
            numeric_df = pd.DataFrame(scaler.transform(numeric_df), columns=numeric_cols)
        df[numeric_cols] = numeric_df
    else:
        df = df.fillna(0)

    return df[features].values


def get_feature_importances(assets: dict[str, object]) -> pd.DataFrame | None:
    model = assets.get("model")
    features: list[str] = assets.get("features", [])
    if hasattr(model, "feature_importances_"):
        return pd.DataFrame({"feature": features, "importance": model.feature_importances_}).sort_values("importance", ascending=False)
    if hasattr(model, "coef_"):
        coefs = model.coef_[0] if len(getattr(model, "coef_", [])) else model.coef_
        return pd.DataFrame({"feature": features, "importance": coefs}).sort_values("importance", ascending=False)
    return None


@st.cache_resource
def load_career_model(cache_mtime: float | None = None):
    _ = cache_mtime  # tie cache to file modification time
    if not CAREER_MODEL_PATH.exists():
        return None
    return joblib.load(CAREER_MODEL_PATH)


@st.cache_data
def load_data(cache_schema_version: str = CACHE_SCHEMA_VERSION) -> pd.DataFrame:
    # Keep cache key tied to schema/version changes.
    _ = cache_schema_version
    df = pd.read_csv(DATA_PATH)
    # Defensive normalization for mixed-source datasets (prevents KeyError on noofbacklog/backlogs/etc.).
    df.columns = [str(c).strip().lower() for c in df.columns]
    if "noofbacklog" not in df.columns:
        if "no_of_backlog" in df.columns:
            df["noofbacklog"] = df["no_of_backlog"]
        elif "backlogs" in df.columns:
            df["noofbacklog"] = df["backlogs"]
        else:
            df["noofbacklog"] = 0
    if "current_back" not in df.columns:
        df["current_back"] = df["noofbacklog"]
    if "ever_back" not in df.columns:
        df["ever_back"] = 0
    # Ensure frequently used numeric columns always exist to avoid KeyErrors.
    required_defaults = {
        "attendance_percentage": 0,
        "study_hours_per_day": 0,
        "social_media_hours": 0,
        "exam_score": pd.NA,
        "gpa_1": pd.NA,
        "gpa_6": pd.NA,
        "exam_anxiety_score": 0,
        "mental_health_rating": 0,
        "screen_time": 0,
        "exercise_frequency": 0,
    }
    for col, default in required_defaults.items():
        if col not in df.columns:
            df[col] = default
    return _apply_common_transforms(df)


@st.cache_data(show_spinner=False)
def load_uploaded_dataset(uploaded_file, cache_schema_version: str = CACHE_SCHEMA_VERSION) -> pd.DataFrame:
    """Load an uploaded CSV and align its schema to the dashboard expectations."""
    _ = cache_schema_version
    file_bytes = uploaded_file.getvalue()
    df = pd.read_csv(io.BytesIO(file_bytes))
    # Mirror the same safeguards as load_data for uploads.
    df.columns = [str(c).strip().lower() for c in df.columns]
    if "noofbacklog" not in df.columns:
        if "no_of_backlog" in df.columns:
            df["noofbacklog"] = df["no_of_backlog"]
        elif "backlogs" in df.columns:
            df["noofbacklog"] = df["backlogs"]
        else:
            df["noofbacklog"] = 0
    if "current_back" not in df.columns:
        df["current_back"] = df["noofbacklog"]
    if "ever_back" not in df.columns:
        df["ever_back"] = 0
    required_defaults = {
        "attendance_percentage": 0,
        "study_hours_per_day": 0,
        "social_media_hours": 0,
        "exam_score": pd.NA,
        "gpa_1": pd.NA,
        "gpa_6": pd.NA,
        "exam_anxiety_score": 0,
        "mental_health_rating": 0,
        "screen_time": 0,
        "exercise_frequency": 0,
    }
    for col, default in required_defaults.items():
        if col not in df.columns:
            df[col] = default
    return _apply_common_transforms(df)


def raw_data_is_newer() -> bool:
    if not DATA_PATH.exists():
        return True
    raw_files = [p for p in RAW_DIR.iterdir() if p.is_file()]
    if not raw_files:
        return False
    latest_raw_time = max(p.stat().st_mtime for p in raw_files)
    return latest_raw_time > DATA_PATH.stat().st_mtime


def ensure_pipeline_current() -> None:
    if raw_data_is_newer():
        with st.spinner("New raw data detected. Refreshing ETL pipeline..."):
            result = subprocess.run(
                ["python", str(ETL_SCRIPT)],
                cwd=BASE_DIR,
                capture_output=True,
                text=True,
                check=False,
            )
        if result.returncode != 0:
            st.error("ETL pipeline refresh failed.")
            st.code(result.stderr or result.stdout)
            st.stop()
        load_data.clear()


def load_etl_status() -> dict[str, object]:
    if not ETL_STATUS_PATH.exists():
        return {}
    with open(ETL_STATUS_PATH, "r", encoding="utf-8") as fh:
        return json.load(fh)


def render_etl_monitor(status: dict[str, object], df: pd.DataFrame) -> None:
    st.subheader("ETL Monitor")
    if not status:
        st.warning("ETL status file not found yet. Run the pipeline once.")
        return

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Pipeline Status", str(status.get("pipeline_status", "unknown")).upper())
    col2.metric("Last Run", str(status.get("last_run_at", "n/a")).replace("T", " ")[:19])
    col3.metric("Gold Rows", f"{len(df):,}")
    col4.metric(
        "Raw Newer Than Gold",
        "Yes" if raw_data_is_newer() else "No",
    )

    with st.expander("Layer Row Counts", expanded=True):
        rows = []
        for layer_name in ["raw_layer", "bronze_layer", "silver_layer", "gold_layer", "warehouse_layer"]:
            layer = status.get(layer_name, {})
            for file_name, info in layer.items():
                rows.append(
                    {
                        "layer": layer_name.replace("_layer", ""),
                        "table_or_file": file_name,
                        "rows": info.get("rows"),
                        "columns": info.get("columns", ""),
                    }
                )
        if rows:
            layer_df = pd.DataFrame(rows)
            if "columns" in layer_df.columns:
                layer_df["columns"] = layer_df["columns"].astype(str)
            st.dataframe(layer_df, width="stretch")

    with st.expander("Source Coverage Checks", expanded=False):
        coverage = status.get("source_coverage", {})
        coverage_rows = pd.DataFrame(
            {
                "source_flag": list(coverage.keys()),
                "available_rows": list(coverage.values()),
            }
        )
        st.dataframe(coverage_rows, width="stretch")
        st.caption(
            "These counts help confirm whether research, attitude, and performance records are actually present in the merged gold dataset."
        )

    with st.expander("Freshness Check", expanded=False):
        freshness_rows = []
        for file_name, info in status.get("raw_layer", {}).items():
            freshness_rows.append(
                {
                    "raw_file": file_name,
                    "rows": info.get("rows"),
                    "modified_at_epoch": info.get("modified_at"),
                }
            )
        if freshness_rows:
            freshness_df = pd.DataFrame(freshness_rows)
            st.dataframe(freshness_df, width="stretch")
        st.caption(
            "If you add new rows to raw files, reload this page or click 'Refresh From Raw Data'. The ETL monitor should then show updated counts."
        )


def filtered_dataframe(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    st.sidebar.header("Filters")
    working_df = df.copy()
    if "branch" not in working_df:
        working_df["branch"] = "Unknown"
    if "gender" not in working_df:
        working_df["gender"] = "Unknown"
    if "semester" not in working_df:
        working_df["semester"] = 1

    branches = sorted(working_df["branch"].dropna().astype(str).unique().tolist())
    genders = sorted(working_df["gender"].dropna().astype(str).unique().tolist())
    semesters = sorted(working_df["semester"].dropna().astype(int).unique().tolist())

    selected_branches = st.sidebar.multiselect("Branch", branches, default=branches)
    selected_genders = st.sidebar.multiselect("Gender", genders, default=genders)
    selected_semesters = st.sidebar.multiselect(
        "Semester", semesters, default=semesters
    )

    filtered_all = working_df[
        working_df["branch"].astype(str).isin(selected_branches)
        & working_df["gender"].astype(str).isin(selected_genders)
        & working_df["semester"].astype(int).isin(selected_semesters)
    ].copy()
    filtered_profile = filtered_all.copy()

    st.sidebar.caption(f"Profile rows in view: {len(filtered_profile):,}")
    st.sidebar.caption(f"Performance rows in view: {len(filtered_all):,}")
    return filtered_profile, filtered_all


def metric_row(df: pd.DataFrame) -> None:
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Students / Records", f"{len(df):,}")
    col2.metric("Average CGPA", f"{df['effective_cgpa'].mean():.2f}")
    col3.metric("Average Study Hours", f"{df['study_hours_per_day'].mean():.2f}")
    col4.metric("Avg Current Backlogs", f"{df['current_back'].mean():.2f}")


def render_figure(fig: plt.Figure) -> None:
    st.pyplot(fig, clear_figure=True, width="stretch")


def plot_grade_distribution(profile_df: pd.DataFrame, performance_df: pd.DataFrame, interactive: bool) -> None:
    st.subheader("1. Overall Academic Performance")
    grade_options = [m for m in ["effective_cgpa", "gpa_1", "gpa_6"] if m in profile_df or m in performance_df]
    if not grade_options:
        st.info("No grade columns available to plot.")
        return
    grade_metric = st.selectbox("Grade metric", grade_options, index=0)
    source_df = profile_df if grade_metric == "effective_cgpa" else performance_df
    title = "Distribution of CGPA" if grade_metric == "effective_cgpa" else f"Distribution of {grade_metric.upper()}"
    xlabel = "CGPA" if grade_metric == "effective_cgpa" else grade_metric.upper()
    if interactive:
        fig = px.histogram(source_df, x=grade_metric, nbins=30, title=title, color_discrete_sequence=["#0f766e"])
        fig.update_layout(xaxis_title=xlabel, yaxis_title="Student Count", bargap=0.05)
        st.plotly_chart(fig, use_container_width=True)
    else:
        fig, ax = plt.subplots(figsize=(10, 4))
        sns.histplot(source_df[grade_metric], bins=30, color="#0f766e", ax=ax)
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Student Count")
        render_figure(fig)


def plot_study_efficiency(profile_df: pd.DataFrame, interactive: bool) -> None:
    st.subheader("2. Study Efficiency Score")
    if "study_efficiency" not in profile_df.columns:
        st.info("Study efficiency is unavailable for this dataset.")
        return
    display_df = profile_df.copy()
    display_df["study_efficiency"] = pd.to_numeric(display_df["study_efficiency"], errors="coerce")
    if interactive:
        fig = px.scatter(
            display_df,
            x="study_hours_per_day",
            y="study_efficiency",
            color="effective_cgpa",
            color_continuous_scale=px.colors.sequential.Blues,
            hover_data=["effective_cgpa", "branch", "gender"],
            title="Study Efficiency vs Study Hours",
            opacity=0.6,
        )
        fig.update_layout(xaxis_title="Study Hours Per Day", yaxis_title="Study Efficiency (marks/hour)")
        st.plotly_chart(fig, use_container_width=True)
    else:
        fig, ax = plt.subplots(figsize=(10, 4))
        sns.scatterplot(
            data=display_df,
            x="study_hours_per_day",
            y="study_efficiency",
            hue="effective_cgpa",
            palette="Blues",
            alpha=0.4,
            ax=ax,
        )
        ax.set_title("Study Efficiency vs Study Hours")
        ax.set_xlabel("Study Hours Per Day")
        ax.set_ylabel("Study Efficiency (marks/hour)")
        render_figure(fig)


def plot_distraction_index(profile_df: pd.DataFrame, interactive: bool) -> None:
    st.subheader("3. Distraction Index (time wasted vs productive time)")
    if "distraction_index" not in profile_df.columns:
        st.info("Distraction index is unavailable for this dataset.")
        return
    display_df = profile_df.copy()
    display_df["distraction_index"] = pd.to_numeric(display_df["distraction_index"], errors="coerce")
    display_df["study_hours_per_day"] = pd.to_numeric(display_df.get("study_hours_per_day", 0), errors="coerce")
    display_df["social_media_hours"] = pd.to_numeric(display_df.get("social_media_hours", 0), errors="coerce")

    if interactive:
        fig = px.scatter(
            display_df,
            x="study_hours_per_day",
            y="distraction_index",
            size="social_media_hours",
            color="effective_cgpa",
            color_continuous_scale=px.colors.sequential.Oranges,
            hover_data=["effective_cgpa", "branch", "gender", "social_media_hours"],
            title="Higher index indicates more time on social media per study hour",
            opacity=0.6,
        )
        fig.update_layout(
            xaxis_title="Study Hours Per Day",
            yaxis_title="Distraction Index (social_media_hours / study_hours_per_day)",
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        fig, ax = plt.subplots(figsize=(10, 4))
        sns.scatterplot(
            data=display_df,
            x="study_hours_per_day",
            y="distraction_index",
            hue="effective_cgpa",
            size="social_media_hours",
            palette="Oranges",
            alpha=0.4,
            ax=ax,
        )
        ax.set_title("Higher index indicates more time on social media per study hour")
        ax.set_xlabel("Study Hours Per Day")
        ax.set_ylabel("Distraction Index (social_media_hours / study_hours_per_day)")
        render_figure(fig)


def plot_academic_discipline(profile_df: pd.DataFrame, interactive: bool) -> None:
    st.subheader("4. Academic Discipline Score")
    if "discipline_score" not in profile_df.columns:
        st.info("Discipline score is unavailable for this dataset.")
        return
    display_df = profile_df.copy()
    display_df["discipline_score"] = pd.to_numeric(display_df["discipline_score"], errors="coerce")
    display_df["attendance_percentage"] = pd.to_numeric(display_df.get("attendance_percentage", 0), errors="coerce")
    display_df["study_hours_per_day"] = pd.to_numeric(display_df.get("study_hours_per_day", 0), errors="coerce")

    if interactive:
        fig = px.scatter(
            display_df,
            x="attendance_percentage",
            y="discipline_score",
            size="study_hours_per_day",
            color="effective_cgpa",
            color_continuous_scale=px.colors.sequential.Greens,
            hover_data=["effective_cgpa", "branch", "gender", "study_hours_per_day"],
            title="Higher score blends attendance (60%) and normalized study hours (40%)",
            opacity=0.65,
        )
        fig.update_layout(
            xaxis_title="Attendance Percentage",
            yaxis_title="Discipline Score",
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        fig, ax = plt.subplots(figsize=(10, 4))
        sns.scatterplot(
            data=display_df,
            x="attendance_percentage",
            y="discipline_score",
            hue="effective_cgpa",
            size="study_hours_per_day",
            palette="Greens",
            alpha=0.45,
            ax=ax,
        )
        ax.set_title("Higher score blends attendance (60%) and normalized study hours (40%)")
        ax.set_xlabel("Attendance Percentage")
        ax.set_ylabel("Discipline Score")
        render_figure(fig)


def plot_study_vs_performance(profile_df: pd.DataFrame, interactive: bool) -> None:
    st.subheader("5. Study Time vs Academic Performance")
    if "study_hours_per_day" not in profile_df:
        st.info("study_hours_per_day not found in this dataset.")
        return
    performance_metric = st.selectbox("Performance metric", ["effective_cgpa", "exam_score"], index=0)
    ylabel = "CGPA" if performance_metric == "effective_cgpa" else performance_metric.upper()
    if interactive:
        fig = px.scatter(
            profile_df,
            x="study_hours_per_day",
            y=performance_metric,
            opacity=0.5,
            color_discrete_sequence=["#1d4ed8"],
            title=f"Study Hours Per Day vs {ylabel}",
        )
        fig.update_layout(xaxis_title="Study Hours Per Day", yaxis_title=ylabel)
        st.plotly_chart(fig, use_container_width=True)
    else:
        fig, ax = plt.subplots(figsize=(10, 4))
        sns.scatterplot(
            data=profile_df,
            x="study_hours_per_day",
            y=performance_metric,
            alpha=0.35,
            color="#1d4ed8",
            ax=ax,
        )
        ax.set_title(f"Study Hours Per Day vs {ylabel}")
        ax.set_xlabel("Study Hours Per Day")
        ax.set_ylabel(ylabel)
        render_figure(fig)


def plot_social_media_vs_cgpa(profile_df: pd.DataFrame, interactive: bool) -> None:
    st.subheader("6. Social Media Usage vs CGPA")
    if "social_media_hours" not in profile_df:
        st.info("social_media_hours not found in this dataset.")
        return
    if interactive:
        fig = px.scatter(
            profile_df,
            x="social_media_hours",
            y="effective_cgpa",
            opacity=0.5,
            color_discrete_sequence=["#dc2626"],
            title="Social Media Hours vs CGPA",
        )
        fig.update_layout(xaxis_title="Social Media Hours", yaxis_title="CGPA")
        st.plotly_chart(fig, use_container_width=True)
    else:
        fig, ax = plt.subplots(figsize=(10, 4))
        sns.scatterplot(
            data=profile_df,
            x="social_media_hours",
            y="effective_cgpa",
            alpha=0.35,
            color="#dc2626",
            ax=ax,
        )
        ax.set_title("Social Media Hours vs CGPA")
        ax.set_xlabel("Social Media Hours")
        ax.set_ylabel("CGPA")
        render_figure(fig)


def plot_stress_vs_performance(profile_df: pd.DataFrame, interactive: bool) -> None:
    st.subheader("7. Stress Level vs Performance")
    if interactive:
        fig = px.box(
            profile_df.sort_values("stress_level"),
            x="stress_level",
            y="effective_cgpa",
            category_orders={"stress_level": STRESS_ORDER},
            color="stress_level",
            color_discrete_sequence=px.colors.sequential.Aggrnyl,
            title="Stress Level vs CGPA",
        )
        fig.update_layout(xaxis_title="Stress Level", yaxis_title="CGPA", showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    else:
        fig, ax = plt.subplots(figsize=(10, 4))
        sns.boxplot(
            data=profile_df.sort_values("stress_level"),
            x="stress_level",
            y="effective_cgpa",
            order=STRESS_ORDER,
            hue="stress_level",
            palette="crest",
            legend=False,
            ax=ax,
        )
        ax.set_title("Stress Level vs CGPA")
        ax.set_xlabel("Stress Level")
        ax.set_ylabel("CGPA")
        render_figure(fig)


def plot_backlog_risk(performance_df: pd.DataFrame, interactive: bool) -> None:
    st.subheader("5. Risk Probability (ML Feature)")
    risk_col = _first_existing_column(
        performance_df,
        [
            "risk_probability",
            "backlog_risk_probability",
            "risk_probability_model",
        ],
    )
    if not risk_col:
        st.info("Risk probability not available in this dataset.")
        return

    working = performance_df.copy()
    risk_series = pd.to_numeric(working[risk_col], errors="coerce")
    # Scale to 0-100 if provided in 0-1 range; then clamp to avoid extreme spikes.
    if risk_series.max(skipna=True) is not None and risk_series.max(skipna=True) <= 1.1:
        risk_series = risk_series * 100
    working["risk_probability_pct"] = risk_series.clip(lower=0, upper=100)

    avg_risk = working["risk_probability_pct"].mean()
    st.metric("Average Risk Probability", f"{avg_risk:.1f}%")

    if interactive:
        fig = px.histogram(
            working,
            x="risk_probability_pct",
            nbins=30,
            title="Risk Probability Distribution",
            color_discrete_sequence=["#ef4444"],
        )
        fig.update_layout(xaxis_title="Risk Probability (%)", yaxis_title="Students")
        st.plotly_chart(fig, use_container_width=True)
    else:
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.histplot(working["risk_probability_pct"].dropna(), bins=30, color="#ef4444", ax=ax)
        ax.set_title("Risk Probability Distribution")
        ax.set_xlabel("Risk Probability (%)")
        ax.set_ylabel("Students")
        render_figure(fig)

    if "attendance_percentage" in working.columns:
        working["attendance_percentage"] = pd.to_numeric(working["attendance_percentage"], errors="coerce")
        if interactive:
            fig2 = px.scatter(
                working,
                x="attendance_percentage",
                y="risk_probability_pct",
                color="risk_probability_pct",
                color_continuous_scale="Reds",
                opacity=0.55,
                title="Risk vs Attendance",
            )
            fig2.update_layout(xaxis_title="Attendance Percentage", yaxis_title="Risk Probability (%)")
            st.plotly_chart(fig2, use_container_width=True)
        else:
            fig2, ax2 = plt.subplots(figsize=(8, 4))
            sns.scatterplot(
                data=working,
                x="attendance_percentage",
                y="risk_probability_pct",
                hue="risk_probability_pct",
                palette="Reds",
                alpha=0.45,
                ax=ax2,
            )
            ax2.set_title("Risk vs Attendance")
            ax2.set_xlabel("Attendance Percentage")
            ax2.set_ylabel("Risk Probability (%)")
            render_figure(fig2)


def plot_branch_performance(profile_df: pd.DataFrame, interactive: bool) -> None:
    st.subheader("8. Branch / Department Performance")
    branch_summary = (
        profile_df.groupby("branch", dropna=False)["effective_cgpa"].mean().reset_index(name="avg_cgpa")
    ).sort_values("avg_cgpa", ascending=False)
    if interactive:
        fig = px.bar(
            branch_summary,
            x="branch",
            y="avg_cgpa",
            color="branch",
            color_discrete_sequence=px.colors.sequential.Blues,
            title="Average CGPA by Branch",
        )
        fig.update_layout(xaxis_title="Branch", yaxis_title="Average CGPA", showlegend=False)
        fig.update_xaxes(tickangle=25)
        st.plotly_chart(fig, use_container_width=True)
    else:
        fig, ax = plt.subplots(figsize=(10, 4))
        sns.barplot(
            data=branch_summary,
            x="branch",
            y="avg_cgpa",
            hue="branch",
            palette="mako",
            legend=False,
            ax=ax,
        )
        ax.set_title("Average CGPA by Branch")
        ax.set_xlabel("Branch")
        ax.set_ylabel("Average CGPA")
        ax.tick_params(axis="x", rotation=25)
        render_figure(fig)


def plot_gender_performance(profile_df: pd.DataFrame, interactive: bool) -> None:
    st.subheader("7. Gender vs Performance")
    gender_summary = (
        profile_df.groupby("gender", dropna=False)["effective_cgpa"].mean().reset_index(name="avg_cgpa")
    )
    if interactive:
        fig = px.bar(
            gender_summary,
            x="gender",
            y="avg_cgpa",
            color="gender",
            color_discrete_sequence=px.colors.qualitative.Set2,
            title="Average CGPA by Gender",
        )
        fig.update_layout(xaxis_title="Gender", yaxis_title="Average CGPA", showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    else:
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.barplot(
            data=gender_summary,
            x="gender",
            y="avg_cgpa",
            hue="gender",
            palette="Set2",
            legend=False,
            ax=ax,
        )
        ax.set_title("Average CGPA by Gender")
        ax.set_xlabel("Gender")
        ax.set_ylabel("Average CGPA")
        render_figure(fig)


def plot_attendance_vs_cgpa(profile_df: pd.DataFrame, interactive: bool) -> None:
    st.subheader("9. Attendance vs CGPA")
    if interactive:
        fig = px.scatter(
            profile_df,
            x="attendance_percentage",
            y="effective_cgpa",
            color="current_back",
            color_continuous_scale=px.colors.sequential.Viridis,
            opacity=0.5,
            title="Attendance Percentage vs CGPA",
        )
        fig.update_layout(xaxis_title="Attendance Percentage", yaxis_title="CGPA")
        st.plotly_chart(fig, use_container_width=True)
    else:
        fig, ax = plt.subplots(figsize=(10, 4))
        sns.scatterplot(
            data=profile_df,
            x="attendance_percentage",
            y="effective_cgpa",
            hue="current_back",
            alpha=0.35,
            palette="viridis",
            ax=ax,
        )
        ax.set_title("Attendance Percentage vs CGPA")
        ax.set_xlabel("Attendance Percentage")
        ax.set_ylabel("CGPA")
        render_figure(fig)


def plot_technical_activity(profile_df: pd.DataFrame, interactive: bool) -> None:
    st.subheader("10. Technical Activities Participation")
    activity_summary = pd.DataFrame(
        {
            "activity": ["technical_projects", "certification_course", "tech_quiz"],
            "value": [
                float(profile_df["technical_projects"].sum()),
                float((profile_df["certification_course"].astype(str).str.lower() == "yes").sum()),
                float(profile_df["tech_quiz"].sum()),
            ],
        }
    )
    if interactive:
        fig = px.pie(
            activity_summary,
            names="activity",
            values="value",
            title="Technical Activity Participation Mix",
            color_discrete_sequence=px.colors.qualitative.Set2,
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.pie(
            activity_summary["value"],
            labels=activity_summary["activity"],
            autopct="%1.1f%%",
            startangle=90,
            colors=sns.color_palette("Set2", n_colors=len(activity_summary)),
        )
        ax.set_title("Technical Activity Participation Mix")
        render_figure(fig)


def plot_semester_trend(profile_df: pd.DataFrame, interactive: bool) -> None:
    if "semester" not in profile_df.columns:
        return
    st.subheader("11. Semester CGPA Trend")
    trend = (
        profile_df.groupby("semester", dropna=True)["effective_cgpa"]
        .mean()
        .reset_index(name="avg_cgpa")
        .sort_values("semester")
    )
    if interactive:
        fig = px.line(
            trend,
            x="semester",
            y="avg_cgpa",
            markers=True,
            title="Average CGPA by Semester",
            color_discrete_sequence=["#6b21a8"],
        )
        fig.update_layout(xaxis_title="Semester", yaxis_title="Average CGPA", yaxis_range=[0, None])
        st.plotly_chart(fig, use_container_width=True)
    else:
        fig, ax = plt.subplots(figsize=(10, 4))
        sns.lineplot(data=trend, x="semester", y="avg_cgpa", marker="o", color="#6b21a8", ax=ax)
        ax.set_title("Average CGPA by Semester")
        ax.set_xlabel("Semester")
        ax.set_ylabel("Average CGPA")
        ax.set_ylim(bottom=0)
        render_figure(fig)


def plot_correlation_heatmap(profile_df: pd.DataFrame, interactive: bool) -> None:
    st.subheader("12. Feature Correlations")
    numeric_cols = profile_df.select_dtypes(include="number").columns
    if len(numeric_cols) < 2:
        st.info("Need at least two numeric columns to show correlations.")
        return
    corr = profile_df[numeric_cols].corr().round(2)
    if interactive:
        fig = px.imshow(corr, text_auto=True, color_continuous_scale="RdBu_r", origin="lower", title="Correlation Heatmap (numeric features)")
        fig.update_layout(xaxis_title="", yaxis_title="")
        st.plotly_chart(fig, use_container_width=True)
    else:
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5, ax=ax)
        ax.set_title("Correlation Heatmap (numeric features)")
        render_figure(fig)


def plot_anxiety_vs_cgpa(profile_df: pd.DataFrame, interactive: bool) -> None:
    if "exam_anxiety_score" not in profile_df.columns:
        return
    st.subheader("12. Exam Anxiety vs CGPA")
    if interactive:
        fig = px.scatter(
            profile_df,
            x="exam_anxiety_score",
            y="effective_cgpa",
            opacity=0.4,
            color_discrete_sequence=["#ef4444"],
            title="Higher anxiety often correlates with lower CGPA",
        )
        fig.update_layout(xaxis_title="Exam Anxiety Score", yaxis_title="CGPA")
        st.plotly_chart(fig, use_container_width=True)
    else:
        fig, ax = plt.subplots(figsize=(10, 4))
        sns.regplot(
            data=profile_df,
            x="exam_anxiety_score",
            y="effective_cgpa",
            scatter_kws={"alpha": 0.25, "color": "#ef4444"},
            line_kws={"color": "#10b981"},
            ax=ax,
        )
        ax.set_title("Higher anxiety often correlates with lower CGPA")
        ax.set_xlabel("Exam Anxiety Score")
        ax.set_ylabel("CGPA")
        render_figure(fig)


def render_cgpa_prediction_form(df: pd.DataFrame) -> None:
    st.subheader("CGPA Prediction")
    st.caption("This form predicts CGPA directly in Streamlit using local model files.")
    metrics = load_cgpa_metrics()
    if metrics:
        rmse = metrics.get("rmse")
        r2 = metrics.get("r2")
        rse = None
        if r2 is not None:
            rse = 1 - r2
        cols_metrics = st.columns(3)
        cols_metrics[0].metric("R²", f"{r2:.3f}" if r2 is not None else "N/A")
        cols_metrics[1].metric("RMSE", f"{rmse:.3f}" if rmse is not None else "N/A")
        cols_metrics[2].metric("RSE", f"{rse:.3f}" if rse is not None else "N/A")
        st.caption("Model Performance")
        if r2 is not None:
            st.write(f"• R² = {r2:.2f} → The model explains {(r2 * 100):.0f}% of CGPA variation.")
        if rmse is not None:
            st.write(f"• RMSE = {rmse:.3f} → Average prediction error is about {rmse:.3f} CGPA.")
    assets = load_prediction_assets()
    gender_options = sorted(df["gender"].dropna().astype(str).unique().tolist()) or ["Male", "Female"]
    branch_options = sorted(df["branch"].dropna().astype(str).unique().tolist()) or ["CSE"]
    category_options = sorted(df["category"].dropna().astype(str).unique().tolist()) or ["GEN"]
    diet_options = sorted(df["diet_quality"].dropna().astype(str).unique().tolist()) or ["average"]
    options = {
        col: list(getattr(encoder, "classes_", []))
        for col, encoder in assets["encoders"].items()
    }
    gender_options = options.get("gender", gender_options) or gender_options
    branch_options = options.get("branch", branch_options) or branch_options
    category_options = options.get("category", category_options) or category_options
    diet_options = options.get("diet_quality", diet_options) or diet_options

    with st.form("cgpa_prediction_form"):
        col1, col2, col3 = st.columns(3)
        marks_10th = col1.number_input("marks_10th", min_value=0.0, max_value=100.0, value=80.0)
        marks_12th = col2.number_input("marks_12th", min_value=0.0, max_value=100.0, value=75.0)
        current_back = col3.number_input("current_back", min_value=0.0, value=0.0)

        col1, col2, col3 = st.columns(3)
        ever_back = col1.number_input("ever_back", min_value=0.0, value=1.0)
        gender = col2.selectbox("gender", gender_options, index=0)
        branch = col3.selectbox("branch", branch_options, index=0)

        col1, col2, col3 = st.columns(3)
        category = col1.selectbox("category", category_options, index=0)
        study_hours_per_day = col2.number_input("study_hours_per_day", min_value=0.0, max_value=24.0, value=5.0)
        attendance_percentage = col3.number_input("attendance_percentage", min_value=0.0, max_value=100.0, value=80.0)

        col1, col2, col3 = st.columns(3)
        sleep_hours = col1.number_input("sleep_hours", min_value=0.0, max_value=24.0, value=7.0)
        parental_support_level = col2.number_input("parental_support_level", min_value=0.0, max_value=10.0, value=6.0)
        motivation_level = col3.number_input("motivation_level", min_value=0.0, max_value=10.0, value=7.0)

        col1, col2, col3 = st.columns(3)
        exam_anxiety_score = col1.number_input("exam_anxiety_score", min_value=0.0, max_value=10.0, value=5.0)
        mental_health_rating = col2.number_input("mental_health_rating", min_value=0.0, max_value=10.0, value=6.0)
        social_media_hours = col3.number_input("social_media_hours", min_value=0.0, max_value=24.0, value=2.0)

        col1, col2, col3 = st.columns(3)
        screen_time = col1.number_input("screen_time", min_value=0.0, max_value=24.0, value=6.0)
        exercise_frequency = col2.number_input("exercise_frequency", min_value=0.0, max_value=14.0, value=3.0)
        diet_quality = col3.selectbox("diet_quality", diet_options, index=0)

        submitted = st.form_submit_button("Predict CGPA")

    if submitted:
        payload = {
            "marks_10th": marks_10th,
            "marks_12th": marks_12th,
            "current_back": current_back,
            "ever_back": ever_back,
            "gender": gender,
            "branch": branch,
            "category": category,
            "study_hours_per_day": study_hours_per_day,
            "attendance_percentage": attendance_percentage,
            "sleep_hours": sleep_hours,
            "parental_support_level": parental_support_level,
            "motivation_level": motivation_level,
            "exam_anxiety_score": exam_anxiety_score,
            "mental_health_rating": mental_health_rating,
            "social_media_hours": social_media_hours,
            "screen_time": screen_time,
            "exercise_frequency": exercise_frequency,
            "diet_quality": diet_quality,
        }

        try:
            input_df = pd.DataFrame([payload])
            input_df = pd.get_dummies(input_df)

            features = assets["features"]

            for col in features:
                if col not in input_df.columns:
                    input_df[col] = 0

            input_df = input_df[features]

            prediction = float(assets["model"].predict(input_df)[0])

            st.success(f"Predicted CGPA: {round(prediction, 2)}")

            importance_df = get_feature_importances(assets)
            if importance_df is not None and not importance_df.empty:
                fig = px.bar(
                    importance_df.head(15).iloc[::-1],
                    x="importance",
                    y="feature",
                    orientation="h",
                    title="Top Feature Influence",
                    color="importance",
                    color_continuous_scale="Blues",
                )
                fig.update_layout(xaxis_title="Importance", yaxis_title="Feature")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Model does not expose feature importances or coefficients.")

        except Exception as e:
            st.error("Prediction failed.")
            st.code(str(e))


def render_career_prediction_form() -> None:
    st.title("AI Career Path Predictor")
    st.caption("Predict your most likely career direction using machine learning insights.")

    model = load_career_model(_career_model_mtime())
    if model is None:
        st.warning(
            f"Career model not found at {CAREER_MODEL_PATH}. Please place the trained model file to enable predictions."
        )
        return

    model_features = list(getattr(model, "feature_names_in_", []))

    with st.container():
        st.markdown("### Student Profile")
        col1, col2 = st.columns(2)
        future_education_plan = col1.selectbox("Future Education Plan", ["Yes", "No"], index=0)
        entrepreneurship_interest = col2.selectbox(
            "Entrepreneurship Interest", ["High", "Medium", "Low", "None"], index=1
        )
        job_search_status = col1.selectbox(
            "Job Search Status", ["Actively looking", "Passively looking", "Not looking"], index=0
        )
        current_status = col2.selectbox(
            "Current Status", ["Student", "Unemployed", "Employed"], index=0
        )

        predict_clicked = st.button("Predict Career Path", type="primary")

    # Backend-only readiness inputs (hidden from UX)
    cgpa_ready = 7.0
    internship_experience = 1.0
    projects_completed = 3.0
    technical_skills_score = 6.0

    def _readiness_status(score: float) -> str:
        if score >= 70:
            return "Ready"
        if score >= 40:
            return "Moderate"
        return "Needs Improvement"

    if predict_clicked:
        row = {
            "future_education_plan": future_education_plan,
            "entrepreneurship_interest": entrepreneurship_interest,
            "job_search_status": job_search_status,
            "current_status": current_status,
        }

        try:
            if model_features:
                df_raw = pd.DataFrame([row])
                df_encoded = pd.get_dummies(df_raw)
                for col in model_features:
                    if col not in df_encoded.columns:
                        df_encoded[col] = 0
                df_encoded = df_encoded[model_features]
                input_df = df_encoded
            else:
                df_raw = pd.DataFrame([row])
                input_df = df_raw.apply(lambda s: s.astype("category").cat.codes)

            prediction = model.predict(input_df)
            pred_value_raw = prediction[0] if len(prediction) else "Unavailable"
            pred_value = CAREER_CLASS_DISPLAY.get(pred_value_raw, str(pred_value_raw))

            result_container = st.container()
            with result_container:
                st.success(f"Predicted Career Path: {pred_value}")

                proba_df = None
                top_confidence = None
                if hasattr(model, "predict_proba"):
                    proba = model.predict_proba(input_df)
                    if proba is not None and len(proba.shape) == 2:
                        classes = list(getattr(model, "classes_", range(proba.shape[1])))
                        display_classes = [CAREER_CLASS_DISPLAY.get(c, str(c)) for c in classes]
                        proba_df = pd.DataFrame({"career_path": display_classes, "probability": proba[0]})
                        proba_df = proba_df.sort_values("probability", ascending=False)
                        top_confidence = float(proba_df["probability"].max() * 100)

                cols_metrics = st.columns(2)
                with cols_metrics[0]:
                    if top_confidence is not None:
                        st.metric("Prediction Confidence", f"{top_confidence:.0f}%")
                    else:
                        st.metric("Prediction Confidence", "N/A")
                with cols_metrics[1]:
                    st.metric("Career Path", pred_value)

                if proba_df is not None:
                    prob_chart = px.bar(
                        proba_df,
                        x="probability",
                        y="career_path",
                        orientation="h",
                        range_x=[0, 1],
                        color="probability",
                        color_continuous_scale="Blues",
                        title="Career Path Probabilities",
                    )
                    prob_chart.update_layout(height=300, margin=dict(l=10, r=10, t=40, b=10))
                    st.plotly_chart(prob_chart, use_container_width=True)

            insights_container = st.container()
            with insights_container:
                st.markdown("### Career Insights")
                bullets = [
                    "Actively looking for jobs boosts placement likelihood." if job_search_status == "Actively looking" else "A passive job search keeps options open for other paths.",
                    "Entrepreneurship interest leaves room for startup paths." if entrepreneurship_interest in ["High", "Medium"] else "Lower entrepreneurship interest shifts focus toward jobs or further studies.",
                    "Pursuing further education signals higher-education orientation." if future_education_plan == "Yes" else "No immediate further education focus detected.",
                ]
                st.markdown("\n".join([f"- {b}" for b in bullets]))

            readiness_score = (
                (cgpa_ready * 10) * 0.4
                + internship_experience * 0.3 * 10
                + projects_completed * 0.2 * 10
                + technical_skills_score * 0.1 * 10
            )
            readiness_score = max(0, min(100, readiness_score))
            status_text = _readiness_status(readiness_score)

            st.markdown("### Career Readiness Score")
            gauge_fig = go.Figure(
                go.Indicator(
                    mode="gauge+number",
                    value=readiness_score,
                    number={"suffix": "/100"},
                    gauge={
                        "axis": {"range": [0, 100]},
                        "bar": {"color": "#0ea5e9"},
                        "steps": [
                            {"range": [0, 40], "color": "#fecdd3"},
                            {"range": [40, 70], "color": "#fef08a"},
                            {"range": [70, 100], "color": "#bbf7d0"},
                        ],
                        "threshold": {"line": {"color": "#0ea5e9", "width": 4}, "value": readiness_score},
                    },
                )
            )
            gauge_fig.update_layout(height=280, margin=dict(l=20, r=20, t=20, b=20))
            st.plotly_chart(gauge_fig, use_container_width=True)
            st.write(f"Status: {status_text}")

            st.markdown("---")
        except Exception as exc:
            st.error("Career prediction failed. Please check your input values.")
            st.code(str(exc))


def main() -> None:
    st.set_page_config(page_title="Student Analytics Dashboard", layout="wide")
    if st.session_state.get("_cache_schema_version") != CACHE_SCHEMA_VERSION:
        st.cache_data.clear()
        st.session_state["_cache_schema_version"] = CACHE_SCHEMA_VERSION

    st.markdown(
        """
        <style>
        .block-container {padding-top: 1rem;}
        .metric-row {margin-bottom: 0.75rem;}
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.title("Student Analytics Dashboard - kanex_final")
    st.caption("Upload data, run the ETL pipeline, explore insights, and predict CGPA.")
    st.code(f"App folder: {BASE_DIR}")
    st.code(f"Default gold data source: {DATA_PATH}")

    uploaded_file = st.sidebar.file_uploader("Upload dataset (CSV)", type=["csv"])
    data_choice = st.sidebar.radio(
        "Data source",
        ["Gold dataset (ETL output)", "Uploaded CSV (quick analysis)"]
        + ([f"Use uploaded: {uploaded_file.name}"] if uploaded_file else []),
        index=0,
    )

    using_uploaded = uploaded_file is not None and "uploaded" in data_choice.lower()

    if using_uploaded:
        df = load_uploaded_dataset(uploaded_file, CACHE_SCHEMA_VERSION)
        etl_status: dict[str, object] = {}
        st.success(f"Using uploaded dataset: {uploaded_file.name}")
    else:
        ensure_pipeline_current()
        df = load_data(CACHE_SCHEMA_VERSION)
        etl_status = load_etl_status()

    if df.empty:
        st.warning("The selected dataset has no rows after loading. Please upload a valid CSV or run ETL.")
        return

    filtered_profile, filtered_all = filtered_dataframe(df)

    if filtered_profile.empty and filtered_all.empty:
        st.warning("No data matches the selected filters.")
        return

    data_tab, insights_tab, prediction_tab, career_tab = st.tabs([
        "Data & ETL",
        "Insights",
        "CGPA Prediction",
        "Career Choice",
    ])

    with data_tab:
        st.subheader("Data and Pipeline")
        st.info("Choose between the processed gold dataset or a one-off uploaded CSV.")
        col1, col2 = st.columns(2)
        col1.metric("Records loaded", f"{len(df):,}")
        col2.metric("Columns", f"{len(df.columns):,}")
        st.dataframe(df.head(), use_container_width=True)

        if using_uploaded:
            st.caption("Uploaded data is analyzed in-memory without altering the ETL outputs.")
            if st.button("Save upload into raw layer and refresh ETL"):
                raw_target = RAW_DIR / f"uploaded_{uploaded_file.name}"
                raw_target.parent.mkdir(parents=True, exist_ok=True)
                raw_target.write_bytes(uploaded_file.getvalue())
                with st.spinner("Running ETL pipeline on uploaded file..."):
                    result = subprocess.run(
                        ["python", str(ETL_SCRIPT)],
                        cwd=BASE_DIR,
                        capture_output=True,
                        text=True,
                        check=False,
                    )
                if result.returncode != 0:
                    st.error("ETL pipeline refresh failed.")
                    st.code(result.stderr or result.stdout)
                else:
                    load_data.clear()
                    st.success("Upload saved to raw layer and ETL refreshed.")
        else:
            if st.button("Refresh From Raw Data"):
                with st.spinner("Running ETL pipeline from raw data..."):
                    result = subprocess.run(
                        ["python", str(ETL_SCRIPT)],
                        cwd=BASE_DIR,
                        capture_output=True,
                        text=True,
                        check=False,
                    )
                if result.returncode != 0:
                    st.error("ETL pipeline refresh failed.")
                    st.code(result.stderr or result.stdout)
                    st.stop()
                load_data.clear()
                st.success("Pipeline refreshed from raw data.")

            render_etl_monitor(etl_status, df)

    with insights_tab:
        interactive = st.checkbox("Use interactive charts (Plotly)", value=True)
        metric_row(filtered_all if not filtered_all.empty else filtered_profile)
        st.info("Interactive filters are on the left. Charts respond to your slice of the data.")
        if not filtered_profile.empty:
            col1, col2 = st.columns(2)
            with col1:
                plot_grade_distribution(filtered_profile, filtered_all, interactive)
            with col2:
                plot_study_efficiency(filtered_profile, interactive)

            col1, col2 = st.columns(2)
            with col1:
                plot_distraction_index(filtered_profile, interactive)
            with col2:
                plot_academic_discipline(filtered_profile, interactive)

            col1, col2 = st.columns(2)
            with col1:
                plot_study_vs_performance(filtered_profile, interactive)
            with col2:
                plot_social_media_vs_cgpa(filtered_profile, interactive)

            col1, col2 = st.columns(2)
            with col1:
                plot_stress_vs_performance(filtered_profile, interactive)
            with col2:
                plot_branch_performance(filtered_profile, interactive)

            col1, col2 = st.columns(2)
            with col1:
                plot_attendance_vs_cgpa(filtered_profile, interactive)
            with col2:
                plot_technical_activity(filtered_profile, interactive)

            col1, col2 = st.columns(2)
            with col1:
                plot_semester_trend(filtered_profile, interactive)
            with col2:
                plot_correlation_heatmap(filtered_profile, interactive)
        if not filtered_all.empty:
            pass

    with prediction_tab:
        render_cgpa_prediction_form(df)

    with career_tab:
        render_career_prediction_form()


if __name__ == "__main__":
    main()
