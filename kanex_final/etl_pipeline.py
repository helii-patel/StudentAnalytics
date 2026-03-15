from __future__ import annotations

from pathlib import Path
import json
import shutil

import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
BRONZE_DIR = DATA_DIR / "bronze"
SILVER_DIR = DATA_DIR / "silver"
GOLD_DIR = DATA_DIR / "gold"
WAREHOUSE_DIR = DATA_DIR / "warehouse"
DASHBOARD_DIR = DATA_DIR / "dashboard"
ETL_STATUS_PATH = DASHBOARD_DIR / "etl_status.json"

SOURCE_FILES = {
    "research_student__1_.xlsx": Path(
        r"c:\Documents\kanexai_hackathon\kenexhackathon\data\raw\research_student__1_.xlsx"
    ),
    "Student_Attitude_and_Behavior.csv": Path(
        r"c:\Documents\kanexai_hackathon\kenexhackathon\data\raw\Student_Attitude_and_Behavior.csv"
    ),
    "studentPerformance.csv": Path(
        r"c:\Documents\kanexai_hackathon\StudentAnalyticsPlatform_ext\datasets\studentPerformance.csv"
    ),
}

GOLD_FINAL_COLUMNS = [
    "student_id",
    "branch",
    "marks_10th",
    "marks_12th",
    "gender",
    "board_10th",
    "board_12th",
    "category",
    "gpa_1",
    "rank",
    "normalized_rank",
    "cgpa",
    "current_back",
    "ever_back",
    "gpa_2",
    "gpa_3",
    "gpa_4",
    "gpa_5",
    "gpa_6",
    "olympiads_qualified",
    "technical_projects",
    "tech_quiz",
    "engg_coaching",
    "ntse_scholarships",
    "miscellany_tech_events",
    "certification_course",
    "department",
    "heightcm",
    "weightkg",
    "10th_mark",
    "12th_mark",
    "college_mark",
    "hobbies",
    "daily_study_time",
    "prefer_to_study_in",
    "salary_expectation",
    "like_degree",
    "career_willingness",
    "social_media_time",
    "travelling_time_",
    "stress_level",
    "financial_status",
    "parttime_job",
    "age",
    "major",
    "study_hours_per_day",
    "social_media_hours",
    "netflix_hours",
    "part_time_job",
    "attendance_percentage",
    "sleep_hours",
    "diet_quality",
    "exercise_frequency",
    "parental_education_level",
    "internet_quality",
    "mental_health_rating",
    "extracurricular_participation",
    "previous_gpa",
    "semester",
    "dropout_risk",
    "social_activity",
    "screen_time",
    "study_environment",
    "access_to_tutoring",
    "family_income_range",
    "parental_support_level",
    "motivation_level",
    "exam_anxiety_score",
    "learning_style",
    "time_management_score",
    "exam_score",
]

BRONZE_RESEARCH_COLUMNS = [
    "branch",
    "marks_10th",
    "marks_12th",
    "gender",
    "board_10th",
    "board_12th",
    "category",
    "gpa_1",
    "rank",
    "normalized_rank",
    "cgpa",
    "current_back",
    "ever_back",
    "gpa_2",
    "gpa_3",
    "gpa_4",
    "gpa_5",
    "gpa_6",
    "olympiads_qualified",
    "technical_projects",
    "tech_quiz",
    "engg_coaching",
    "ntse_scholarships",
    "miscellany_tech_events",
]

BRONZE_ATTITUDE_COLUMNS = [
    "certification_course",
    "gender",
    "department",
    "heightcm",
    "weightkg",
    "10th_mark",
    "12th_mark",
    "college_mark",
    "hobbies",
    "daily_study_time",
    "prefer_to_study_in",
    "salary_expectation",
    "like_degree",
    "career_willingness",
    "social_media_time",
    "travelling_time_",
    "stress_level",
    "financial_status",
    "parttime_job",
]

BRONZE_PERFORMANCE_COLUMNS = [
    "student_id",
    "age",
    "gender",
    "major",
    "study_hours_per_day",
    "social_media_hours",
    "netflix_hours",
    "part_time_job",
    "attendance_percentage",
    "sleep_hours",
    "diet_quality",
    "exercise_frequency",
    "parental_education_level",
    "internet_quality",
    "mental_health_rating",
    "extracurricular_participation",
    "previous_gpa",
    "semester",
    "stress_level",
    "dropout_risk",
    "social_activity",
    "screen_time",
    "study_environment",
    "access_to_tutoring",
    "family_income_range",
    "parental_support_level",
    "motivation_level",
    "exam_anxiety_score",
    "learning_style",
    "time_management_score",
    "exam_score",
]


def ensure_directories() -> None:
    for directory in [RAW_DIR, BRONZE_DIR, SILVER_DIR, GOLD_DIR, WAREHOUSE_DIR, DASHBOARD_DIR]:
        directory.mkdir(parents=True, exist_ok=True)


def seed_raw_sources() -> dict[str, Path]:
    seeded = {}
    for name, source in SOURCE_FILES.items():
        destination = RAW_DIR / name
        if not destination.exists():
            shutil.copy2(source, destination)
        seeded[name] = destination
    return seeded


def read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    return pd.read_excel(path)


def cap_outliers_iqr(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    for col in columns:
        if col not in df.columns:
            continue
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        df[col] = np.where(df[col] < lower_bound, lower_bound, df[col])
        df[col] = np.where(df[col] > upper_bound, upper_bound, df[col])
    return df


def clean_research_dataset(path: Path) -> pd.DataFrame:
    df = read_table(path)

    df.columns = (
        df.columns.str.lower()
        .str.replace(" ", "_", regex=False)
        .str.replace("[", "_", regex=False)
        .str.replace("]", "", regex=False)
        .str.replace("(", "_", regex=False)
        .str.replace(")", "", regex=False)
        .str.replace(".", "_", regex=False)
        .str.replace("__", "_", regex=False)
        .str.strip("_")
    )
    df = df.drop_duplicates().copy()

    cols_to_convert_numeric = [
        "marks_10th",
        "marks_12th",
        "gpa_1",
        "gpa_2",
        "gpa_3",
        "gpa_4",
        "gpa_5",
        "gpa_6",
        "cgpa",
        "olympiads_qualified",
        "technical_projects",
        "tech_quiz",
        "engg_coaching",
        "ntse_scholarships",
        "miscellany_tech_events",
        "current_back",
        "ever_back",
        "rank",
        "normalized_rank",
    ]

    for col in cols_to_convert_numeric:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    categorical_cols = df.select_dtypes(exclude=[np.number]).columns

    for col in numeric_cols:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].median())

    for col in categorical_cols:
        if df[col].isnull().any():
            mode = df[col].mode()
            df[col] = df[col].fillna(mode.iloc[0] if not mode.empty else "Unknown")

    board_10th_mapping = {
        "BSEB Patna": "BSEB",
        "BSEB": "BSEB",
        "B.S.E.B": "BSEB",
        "BSEB PATNA": "BSEB",
        "B.S.E.B PATNA": "BSEB",
        "BIHAR SCHOOL EXAMINATION BOARD PATNA": "BSEB",
        "B.S.E.B.PATNA": "BSEB",
        "BIHAR SCHOOL EXAMINATION BOARD, PATNA": "BSEB",
        "B.S.E.B(PATNA)": "BSEB",
        "BIHAR SCHOOL EXAMINATION BOARD,PATNA": "BSEB",
        "Bihar School ExaminatiAon Board, Patna": "BSEB",
        "Bihar Shool Examination Board ,Patna": "BSEB",
        "bihar school examination board": "BSEB",
        "Bihar Board": "BSEB",
        "BIHAR BOARD": "BSEB",
        "CBSE": "CBSE",
        "C.B.S.E": "CBSE",
        "C.B.S.E.": "CBSE",
        "cbse": "CBSE",
        "CENTRAL BOARD OF SECONDARY EDUCATION": "CBSE",
        "CBSE BOARD": "CBSE",
        "Central Board of Secondary Education": "CBSE",
        "ICSE": "ICSE",
        "I.C.S.E": "ICSE",
        "ISCE": "ICSE",
        "I.C.S.E.": "ICSE",
        "i.c.s.e": "ICSE",
        "UP BOARD": "UP Board",
        "Uttar Pradesh Board": "UP Board",
        "U.P. BOARD": "UP Board",
        "Rajasthan Board": "RBSE",
        "RBSE": "RBSE",
        "RAJASTHAN BOARD": "RBSE",
        "BOARD OF SECONDARY EDUCATION,ANDHRA PRADESH": "Andhra Pradesh Board",
        "STATE BOARD OF ANDHRA PRADESH": "Andhra Pradesh Board",
        "Board of Secondary Education": "State Board",
        "MAHARASHTRA STATE BOARD OF SECONDARY EDUCATION": "Maharashtra State Board",
        "Maharashtra State Board": "Maharashtra State Board",
        "STATE BOARD": "State Board",
        "Haryana Board": "Haryana Board",
    }
    board_12th_mapping = {
        "BSEB Patna": "BSEB",
        "CBSE": "CBSE",
        "ICSE": "ICSE",
        "C.B.S.E": "CBSE",
        "B.S.E.B": "BSEB",
        "BSEB": "BSEB",
        "I.S.C": "ICSE",
        "BSEB PATNA": "BSEB",
        "UP BOARD": "UP Board",
        "ISC": "ICSE",
        "I Sc": "ICSE",
        "cbse": "CBSE",
        "C.B.S.E.": "CBSE",
        "NIOS BOARD": "NIOS",
        "state": "State Board",
        "BOARD OF INTERMEDIATE EDUCATION,ANDHRA PRADESH": "Andhra Pradesh Board",
        "Central Board": "CBSE",
        "bhopal": "Other",
        "CENTRAL BOARD OF SECONDARY EDUCATION": "CBSE",
        "NIOS": "NIOS",
        "CBSE BOARD": "CBSE",
        "B.S.E.B.PATNA": "BSEB",
        "Board of Intermediate Education Andhra Pradesh": "Andhra Pradesh Board",
        "INTERMEDIATE BOARD OF EDUCATION- ANDHRA PRADESH": "Andhra Pradesh Board",
        "BIHAR SCHOOL EXAMINATION BOARD, PATNA": "BSEB",
        "Boar of Intermediate Education, A.P.": "Andhra Pradesh Board",
        "STATE BOARD": "State Board",
        "STATE BOARD OF ANDHRA PRADESH": "Andhra Pradesh Board",
        "B.S.E.B(H.S.PATNA)": "BSEB",
        "Maharashtra State Board": "Maharashtra State Board",
        "bihar school examination board": "BSEB",
        "BIHAR SCHOOL EXAMINATION BOARD,PATNA": "BSEB",
        "BIHAR BOARD": "BSEB",
        "Bihar School ExaminatiAon Board, Patna": "BSEB",
        "Bihar School Examination Board ,Patna": "BSEB",
        "Central Board of Secondary Education": "CBSE",
        "RBSE": "RBSE",
        "ANDHRA PRADESH BOARD": "Andhra Pradesh Board",
        "ISCE": "ICSE",
        "c.b.s.e": "CBSE",
        "RAJASTHAN BOARD": "RBSE",
        "Goa Board for Secondary and Higher Secondary Education": "Goa Board",
        "U.P. BOARD": "UP Board",
        "Bihar Board": "BSEB",
        "CBSE Board": "CBSE",
        "BOARD OF SECONDARY EDUCATION, RAJASTHAN": "RBSE",
        "H.S.E.B": "State Board",
    }

    if "board_10th" in df.columns:
        df["board_10th"] = df["board_10th"].replace(board_10th_mapping)
    if "board_12th" in df.columns:
        df["board_12th"] = df["board_12th"].replace(board_12th_mapping)

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    df = cap_outliers_iqr(df, numeric_cols)
    return df[[col for col in BRONZE_RESEARCH_COLUMNS if col in df.columns]].copy()


def clean_attitude_dataset(path: Path) -> pd.DataFrame:
    df = read_table(path)

    df.columns = (
        df.columns.str.lower()
        .str.replace(" ", "_", regex=False)
        .str.replace(r"[^a-z0-9_]", "", regex=True)
    )
    df = df.rename(
        columns={
            "daily_studing_time": "daily_study_time",
            "do_you_like_your_degree": "like_degree",
            "willingness_to_pursue_a_career_based_on_their_degree__": "career_willingness",
            "social_medai__video": "social_media_time",
            "stress_level_": "stress_level",
        }
    )
    df = df.drop_duplicates().copy()

    numeric_candidate_cols = [
        "heightcm",
        "weightkg",
        "10th_mark",
        "12th_mark",
        "college_mark",
        "salary_expectation",
    ]

    for col in numeric_candidate_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "career_willingness" in df.columns:
        df["career_willingness"] = (
            df["career_willingness"].astype(str).str.replace("%", "", regex=False)
        )
        df["career_willingness"] = pd.to_numeric(df["career_willingness"], errors="coerce")

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    categorical_cols = df.select_dtypes(exclude=[np.number]).columns

    for col in numeric_cols:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].median())

    for col in categorical_cols:
        if df[col].isnull().any():
            mode = df[col].mode()
            df[col] = df[col].fillna(mode.iloc[0] if not mode.empty else "unknown")

    def clean_text(value: object) -> object:
        if isinstance(value, str):
            return value.lower().strip()
        return value

    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].apply(clean_text)

    for col in ["certification_course", "like_degree", "parttime_job"]:
        if col in df.columns:
            df[col] = df[col].map({"yes": "yes", "no": "no"}).fillna("no")

    def standardize_time_range(value: object) -> str:
        if pd.isna(value):
            return "other"
        value = str(value).lower()
        if "0 - 30" in value or "30 minute" in value:
            return "0-30_min"
        if "30 - 60" in value or "1 - 1.30" in value:
            return "30-60_min"
        if "1 - 2" in value or "1.30 - 2" in value:
            return "1-2_hour"
        if "more than 2" in value or "2+" in value:
            return "2+_hour"
        return "other"

    for col in ["daily_study_time", "social_media_time", "travelling_time_"]:
        if col in df.columns:
            df[col] = df[col].apply(standardize_time_range)

    df = cap_outliers_iqr(
        df, ["heightcm", "weightkg", "salary_expectation", "college_mark"]
    )
    return df[[col for col in BRONZE_ATTITUDE_COLUMNS if col in df.columns]].copy()


def clean_performance_dataset(path: Path) -> pd.DataFrame:
    df = read_table(path)
    df.columns = (
        df.columns.str.lower()
        .str.replace(" ", "_", regex=False)
        .str.replace(r"[^a-z0-9_]", "", regex=True)
    )
    df = df.drop_duplicates().copy()

    numeric_candidate_cols = [
        "student_id",
        "age",
        "study_hours_per_day",
        "social_media_hours",
        "netflix_hours",
        "attendance_percentage",
        "sleep_hours",
        "exercise_frequency",
        "mental_health_rating",
        "previous_gpa",
        "semester",
        "stress_level",
        "social_activity",
        "screen_time",
        "parental_support_level",
        "motivation_level",
        "exam_anxiety_score",
        "time_management_score",
        "exam_score",
    ]

    for col in numeric_candidate_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    categorical_cols = df.select_dtypes(exclude=[np.number]).columns

    for col in numeric_cols:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].median())

    for col in categorical_cols:
        if df[col].isnull().any():
            mode = df[col].mode()
            df[col] = df[col].fillna(mode.iloc[0] if not mode.empty else "unknown")

    if "semester" in df.columns:
        df["semester"] = (
            pd.to_numeric(df["semester"], errors="coerce").fillna(0).round().astype(int)
        )

    return df[[col for col in BRONZE_PERFORMANCE_COLUMNS if col in df.columns]].copy()


def save_bronze_datasets(raw_paths: dict[str, Path]) -> dict[str, Path]:
    research_df = clean_research_dataset(raw_paths["research_student__1_.xlsx"])
    attitude_df = clean_attitude_dataset(raw_paths["Student_Attitude_and_Behavior.csv"])
    performance_df = clean_performance_dataset(raw_paths["studentPerformance.csv"])

    outputs = {
        "research": BRONZE_DIR / "clean_student_dataset1.csv",
        "attitude": BRONZE_DIR / "clean_student_dataset2_final.csv",
        "performance": BRONZE_DIR / "clean_student_dataset3.csv",
    }

    research_df.to_csv(outputs["research"], index=False)
    attitude_df.to_csv(outputs["attitude"], index=False)
    performance_df.to_csv(outputs["performance"], index=False)
    return outputs


def merge_bronze_datasets(bronze_paths: dict[str, Path]) -> pd.DataFrame:
    df1 = pd.read_csv(bronze_paths["research"])
    df2 = pd.read_csv(bronze_paths["attitude"])
    df3 = pd.read_csv(bronze_paths["performance"])

    if "student_id" in df3.columns:
        df3 = df3.drop("student_id", axis=1)

    df1 = df1.reset_index(drop=True)
    df2 = df2.reset_index(drop=True)
    df3 = df3.reset_index(drop=True)

    df1.insert(0, "student_id", range(1, len(df1) + 1))
    df2.insert(0, "student_id", range(1, len(df2) + 1))
    if "student_id" not in df3.columns:
        df3.insert(0, "student_id", range(1, len(df3) + 1))

    merged_df = pd.concat([df1, df2, df3], axis=1)
    merged_df = merged_df.loc[:, ~merged_df.columns.duplicated(keep="first")].copy()

    if "student_id" not in merged_df.columns:
        merged_df.insert(0, "student_id", range(1, len(merged_df) + 1))
    else:
        null_student_ids = merged_df["student_id"].isna().sum()
        if null_student_ids > 0:
            valid_ids = merged_df["student_id"].dropna().max()
            start_id = int(valid_ids) + 1 if not pd.isna(valid_ids) else 1
            null_indices = merged_df[merged_df["student_id"].isna()].index
            merged_df.loc[null_indices, "student_id"] = range(
                start_id, start_id + len(null_indices)
            )

    numeric_cols = merged_df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = merged_df.select_dtypes(include=["object"]).columns.tolist()

    if "student_id" in numeric_cols:
        numeric_cols.remove("student_id")
    for col in numeric_cols:
        if merged_df[col].isnull().sum() > 0:
            merged_df[col] = merged_df[col].fillna(merged_df[col].median())

    for col in categorical_cols:
        if merged_df[col].isnull().sum() > 0:
            mode = merged_df[col].mode()
            fill_value = mode.iloc[0] if not mode.empty else "Unknown"
            merged_df[col] = merged_df[col].fillna(fill_value)

    merged_df["student_id"] = pd.to_numeric(
        merged_df["student_id"], errors="coerce"
    ).fillna(0).astype(int)
    return merged_df


def create_gold_dataset(silver_df: pd.DataFrame) -> pd.DataFrame:
    gold_df = silver_df.copy()
    gold_df = gold_df[[col for col in GOLD_FINAL_COLUMNS if col in gold_df.columns]].copy()

    return gold_df


def create_ml_features(gold_df: pd.DataFrame) -> pd.DataFrame:
    features_df = gold_df.copy()

    if {"technical_projects", "gpa_1", "career_willingness"}.issubset(features_df.columns):
        conditions = [
            features_df["technical_projects"] > 2,
            features_df["gpa_1"] > 8,
            features_df["career_willingness"] > 70,
        ]
        choices = ["technical", "academic", "business"]
        features_df["career_path"] = np.select(conditions, choices, default="balanced")

    time_score_map = {
        "0-30_min": 1,
        "30-60_min": 2,
        "1-2_hour": 3,
        "2+_hour": 4,
        "other": 2,
    }
    stress_score_map = {
        "fabulous": 0,
        "good": 1,
        "bad": 3,
        "awful": 4,
    }
    yes_no_map = {"yes": 1, "no": 0}
    financial_status_map = {
        "awful": 1,
        "bad": 2,
        "good": 3,
        "fabulous": 4,
    }

    if {"marks_10th", "marks_12th", "cgpa"}.issubset(features_df.columns):
        features_df["academic_score"] = (
            features_df["marks_10th"] + features_df["marks_12th"] + features_df["cgpa"]
        ) / 3

    certification_numeric = features_df.get("certification_course", 0)
    if isinstance(certification_numeric, pd.Series):
        certification_numeric = certification_numeric.astype(str).str.lower().map(yes_no_map).fillna(0)

    if {
        "technical_projects",
        "tech_quiz",
        "olympiads_qualified",
    }.issubset(features_df.columns):
        features_df["technical_score"] = (
            features_df["technical_projects"]
            + features_df["tech_quiz"]
            + features_df["olympiads_qualified"]
            + certification_numeric
        )

    gpa_cols = [col for col in ["gpa_1", "gpa_2", "gpa_3", "gpa_4", "gpa_5", "gpa_6"] if col in features_df.columns]
    if gpa_cols:
        features_df["avg_gpa"] = features_df[gpa_cols].mean(axis=1)
        features_df["gpa_variance"] = features_df[gpa_cols].var(axis=1, ddof=0)
        if "gpa_1" in features_df.columns and "gpa_6" in features_df.columns:
            features_df["gpa_trend"] = features_df["gpa_6"] - features_df["gpa_1"]

    if {"thispresent", "prispresent"}.issubset(features_df.columns):
        features_df["attendance_score"] = (
            features_df["thispresent"] + features_df["prispresent"]
        ) / 2

    if {"technical_projects", "miscellany_tech_events"}.issubset(features_df.columns):
        features_df["engagement_score"] = (
            features_df["technical_projects"]
            + features_df["miscellany_tech_events"]
            + certification_numeric
        )

    stress_numeric = features_df.get("stress_level", 0)
    if isinstance(stress_numeric, pd.Series):
        stress_numeric = (
            stress_numeric.astype(str).str.lower().map(stress_score_map).fillna(2)
        )

    social_media_numeric = features_df.get("social_media_time", 0)
    if isinstance(social_media_numeric, pd.Series):
        social_media_numeric = (
            social_media_numeric.astype(str).str.lower().map(time_score_map).fillna(2)
        )

    daily_study_numeric = features_df.get("daily_study_time", 0)
    if isinstance(daily_study_numeric, pd.Series):
        daily_study_numeric = (
            daily_study_numeric.astype(str).str.lower().map(time_score_map).fillna(2)
        )

    features_df["stress_risk"] = stress_numeric + social_media_numeric
    features_df["productivity_score"] = daily_study_numeric - social_media_numeric

    if {"current_back", "ever_back", "noofbacklog"}.issubset(features_df.columns):
        features_df["backlog_pressure"] = (
            features_df["current_back"]
            + features_df["ever_back"]
            + features_df["noofbacklog"]
        )

    if {"theorycredit", "practicalcredit"}.issubset(features_df.columns):
        features_df["subject_load"] = (
            features_df["theorycredit"] + features_df["practicalcredit"]
        )

    if {"sgpa", "prsgpa"}.issubset(features_df.columns):
        features_df["academic_momentum"] = features_df["sgpa"] - features_df["prsgpa"]

    if {
        "theoryagggradepoint",
        "theorycredit",
        "practicalagggradepoint",
        "practicalcredit",
    }.issubset(features_df.columns):
        total_credit = features_df["theorycredit"] + features_df["practicalcredit"]
        weighted_points = (
            features_df["theoryagggradepoint"] * features_df["theorycredit"]
            + features_df["practicalagggradepoint"] * features_df["practicalcredit"]
        )
        features_df["credit_efficiency"] = np.where(
            total_credit > 0,
            weighted_points / total_credit,
            0,
        )

    like_degree_numeric = features_df.get("like_degree", 0)
    if isinstance(like_degree_numeric, pd.Series):
        like_degree_numeric = like_degree_numeric.astype(str).str.lower().map(yes_no_map).fillna(0)

    parttime_numeric = features_df.get("parttime_job", 0)
    if isinstance(parttime_numeric, pd.Series):
        parttime_numeric = parttime_numeric.astype(str).str.lower().map(yes_no_map).fillna(0)

    financial_numeric = features_df.get("financial_status", 0)
    if isinstance(financial_numeric, pd.Series):
        financial_numeric = (
            financial_numeric.astype(str).str.lower().map(financial_status_map).fillna(2)
        )

    features_df["wellbeing_balance"] = features_df["productivity_score"] - features_df["stress_risk"]
    features_df["employability_signal"] = (
        features_df.get("technical_score", 0)
        + features_df.get("engagement_score", 0)
        + like_degree_numeric
    )
    features_df["student_support_need"] = (
        features_df.get("backlog_pressure", 0)
        + features_df["stress_risk"]
        + parttime_numeric
        - financial_numeric
    )

    if {"weightkg", "heightcm"}.issubset(features_df.columns):
        height_m = features_df["heightcm"] / 100
        features_df["bmi"] = np.where(
            height_m > 0,
            features_df["weightkg"] / (height_m ** 2),
            np.nan,
        )
        if pd.isna(features_df["bmi"]).any():
            features_df["bmi"] = features_df["bmi"].fillna(features_df["bmi"].median())

    return features_df


def create_dashboard_outputs(gold_df: pd.DataFrame, features_df: pd.DataFrame) -> dict[str, object]:
    for existing_file in DASHBOARD_DIR.glob("*"):
        if existing_file.is_file():
            existing_file.unlink()

    student_columns = [
        "student_id",
        "branch",
        "department",
        "gender",
        "career_path",
        "cgpa",
        "sgpa",
        "noofbacklog",
        "academic_score",
        "technical_score",
        "engagement_score",
        "stress_risk",
        "productivity_score",
        "bmi",
        "student_support_need",
        "employability_signal",
    ]
    student_overview = features_df[
        [col for col in student_columns if col in features_df.columns]
    ].copy()
    student_overview.to_csv(DASHBOARD_DIR / "student_overview.csv", index=False)

    career_group = features_df.groupby("career_path", dropna=False)
    career_agg = {"student_id": "count"}
    rename_map = {"student_id": "student_count"}
    for col, alias in [
        ("cgpa", "avg_cgpa"),
        ("academic_score", "avg_academic_score"),
        ("technical_score", "avg_technical_score"),
        ("employability_signal", "avg_employability_signal"),
    ]:
        if col in features_df.columns:
            career_agg[col] = "mean"
            rename_map[col] = alias
    career_summary = career_group.agg(career_agg).reset_index().rename(columns=rename_map)
    career_summary.to_csv(DASHBOARD_DIR / "career_summary.csv", index=False)

    semester_group = features_df.groupby("semester", dropna=False)
    semester_agg = {"student_id": "count"}
    semester_rename = {"student_id": "student_count"}
    for col, alias in [
        ("sgpa", "avg_sgpa"),
        ("prsgpa", "avg_prsgpa"),
        ("prcgpa", "avg_prcgpa"),
        ("noofbacklog", "avg_backlog"),
        ("attendance_score", "avg_attendance_score"),
        ("exam_score", "avg_exam_score"),
        ("cgpa", "avg_cgpa"),
    ]:
        if col in features_df.columns:
            semester_agg[col] = "mean"
            semester_rename[col] = alias
    semester_summary = semester_group.agg(semester_agg).reset_index().rename(columns=semester_rename)
    semester_summary.to_csv(DASHBOARD_DIR / "semester_summary.csv", index=False)

    risk_source = "student_support_need" if "student_support_need" in features_df.columns else None
    if risk_source is None and "exam_anxiety_score" in features_df.columns:
        risk_source = "exam_anxiety_score"
    if risk_source is not None:
        risk_band = pd.cut(
            features_df[risk_source],
            bins=[-np.inf, 1, 4, np.inf],
            labels=["low", "medium", "high"],
        )
        risk_summary = (
            pd.DataFrame({"risk_band": risk_band})
            .value_counts(dropna=False)
            .reset_index(name="student_count")
            .sort_values("risk_band")
        )
    else:
        risk_summary = pd.DataFrame({"risk_band": [], "student_count": []})
    risk_summary.to_csv(DASHBOARD_DIR / "risk_summary.csv", index=False)

    branch_group = features_df.groupby("branch", dropna=False)
    branch_agg = {"student_id": "count"}
    branch_rename = {"student_id": "student_count"}
    for col, alias in [
        ("cgpa", "avg_cgpa"),
        ("stress_risk", "avg_stress_risk"),
        ("productivity_score", "avg_productivity"),
        ("student_support_need", "avg_support_need"),
        ("exam_score", "avg_exam_score"),
    ]:
        if col in features_df.columns:
            branch_agg[col] = "mean"
            branch_rename[col] = alias
    branch_summary = branch_group.agg(branch_agg).reset_index().rename(columns=branch_rename)
    branch_summary.to_csv(DASHBOARD_DIR / "branch_summary.csv", index=False)

    dashboard_kpis = {
        "student_count": int(len(features_df)),
        "subject_count": int(features_df["subjectid"].nunique()) if "subjectid" in features_df.columns else 0,
        "avg_cgpa": round(float(features_df["cgpa"].mean()), 4) if "cgpa" in features_df.columns else 0.0,
        "avg_sgpa": round(float(features_df["sgpa"].mean()), 4) if "sgpa" in features_df.columns else 0.0,
        "avg_academic_score": round(float(features_df["academic_score"].mean()), 4) if "academic_score" in features_df.columns else 0.0,
        "avg_technical_score": round(float(features_df["technical_score"].mean()), 4) if "technical_score" in features_df.columns else 0.0,
        "avg_stress_risk": round(float(features_df["stress_risk"].mean()), 4) if "stress_risk" in features_df.columns else 0.0,
        "last_pipeline_stage": "dashboard",
    }
    with open(DASHBOARD_DIR / "dashboard_kpis.json", "w", encoding="utf-8") as fh:
        json.dump(dashboard_kpis, fh, indent=2)

    return {
        "student_overview": student_overview.shape,
        "career_summary": career_summary.shape,
        "semester_summary": semester_summary.shape,
        "risk_summary": risk_summary.shape,
        "branch_summary": branch_summary.shape,
        "dashboard_kpis": dashboard_kpis,
    }


def create_etl_status(
    raw_paths: dict[str, Path],
    bronze_paths: dict[str, Path],
    silver_df: pd.DataFrame,
    gold_df: pd.DataFrame,
    features_df: pd.DataFrame,
    warehouse_tables: dict[str, pd.DataFrame],
) -> dict[str, object]:
    raw_counts = {}
    for name, path in raw_paths.items():
        raw_counts[name] = {
            "rows": int(len(read_table(path))),
            "modified_at": path.stat().st_mtime,
        }

    bronze_counts = {}
    for name, path in bronze_paths.items():
        bronze_counts[path.name] = {
            "rows": int(len(pd.read_csv(path))),
            "modified_at": path.stat().st_mtime,
        }

    status = {
        "pipeline_status": "success",
        "last_run_at": pd.Timestamp.now().isoformat(),
        "raw_layer": raw_counts,
        "bronze_layer": bronze_counts,
        "silver_layer": {
            "merged_student_dataset.csv": {
                "rows": int(len(silver_df)),
                "columns": int(silver_df.shape[1]),
            }
        },
        "gold_layer": {
            "final_dataset.csv": {
                "rows": int(len(gold_df)),
                "columns": int(gold_df.shape[1]),
            },
            "ml_features_dataset.csv": {
                "rows": int(len(features_df)),
                "columns": int(features_df.shape[1]),
            },
        },
        "warehouse_layer": {
            name: {"rows": int(len(table)), "columns": int(table.shape[1])}
            for name, table in warehouse_tables.items()
        },
        "source_coverage": {
            "research_record_available": int(gold_df.get("research_record_available", pd.Series(dtype=int)).sum()),
            "attitude_record_available": int(gold_df.get("attitude_record_available", pd.Series(dtype=int)).sum()),
            "performance_record_available": int(gold_df.get("performance_record_available", pd.Series(dtype=int)).sum()),
        },
    }

    with open(ETL_STATUS_PATH, "w", encoding="utf-8") as fh:
        json.dump(status, fh, indent=2)

    return status


def create_star_schema(gold_df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    def select_distinct(columns: list[str]) -> pd.DataFrame:
        return gold_df[[col for col in columns if col in gold_df.columns]].drop_duplicates().reset_index(drop=True)

    dim_student_cols = [
        "student_id",
        "gender",
        "branch",
        "category",
        "department",
        "heightcm",
        "weightkg",
        "financial_status",
        "parttime_job",
    ]
    dim_academic_history_cols = [
        "student_id",
        "marks_10th",
        "marks_12th",
        "board_10th",
        "board_12th",
        "gpa_1",
        "gpa_2",
        "gpa_3",
        "gpa_4",
        "gpa_5",
        "gpa_6",
        "cgpa",
        "rank",
        "normalized_rank",
    ]
    dim_student_activity_cols = [
        "student_id",
        "technical_projects",
        "tech_quiz",
        "olympiads_qualified",
        "certification_course",
        "ntse_scholarships",
        "miscellany_tech_events",
        "engg_coaching",
    ]
    dim_behavior_cols = [
        "student_id",
        "daily_study_time",
        "prefer_to_study_in",
        "social_media_time",
        "travelling_time_",
        "stress_level",
        "hobbies",
        "like_degree",
        "career_willingness",
        "salary_expectation",
    ]
    dim_subject_cols = [
        "subjectid",
        "subjectname",
        "theorycredit",
        "practicalcredit",
    ]
    fact_student_academic_cols = [
        "student_id",
        "semester",
        "subjectid",
        "theoryagggradepoint",
        "theorycredit",
        "practicalagggradepoint",
        "practicalcredit",
        "sgpa",
        "prsgpa",
        "prcgpa",
        "noofbacklog",
        "thispresent",
        "prispresent",
    ]

    dim_student = select_distinct(dim_student_cols)
    dim_academic_history = select_distinct(dim_academic_history_cols)
    dim_student_activity = select_distinct(dim_student_activity_cols)
    dim_behavior = select_distinct(dim_behavior_cols).rename(
        columns={"travelling_time_": "travelling_time"}
    )
    dim_subject = select_distinct(dim_subject_cols)
    fact_student_academic = gold_df[
        [col for col in fact_student_academic_cols if col in gold_df.columns]
    ].copy()
    fact_student_academic = fact_student_academic.rename(
        columns={"travelling_time_": "travelling_time"}
    )

    warehouse_tables = {
        "dim_student": dim_student,
        "dim_academic_history": dim_academic_history,
        "dim_student_activity": dim_student_activity,
        "dim_behavior": dim_behavior,
        "dim_subject": dim_subject,
        "fact_student_academic": fact_student_academic,
    }

    for existing_csv in WAREHOUSE_DIR.glob("*.csv"):
        existing_csv.unlink()

    for name, table in warehouse_tables.items():
        table.to_csv(WAREHOUSE_DIR / f"{name}.csv", index=False)

    return warehouse_tables


def main() -> None:
    ensure_directories()
    raw_paths = seed_raw_sources()
    bronze_paths = save_bronze_datasets(raw_paths)

    silver_merged = merge_bronze_datasets(bronze_paths)
    silver_merged.to_csv(SILVER_DIR / "merged_student_dataset.csv", index=False)

    gold_df = create_gold_dataset(silver_merged)
    gold_df.to_csv(GOLD_DIR / "final_dataset.csv", index=False)
    features_df = create_ml_features(gold_df)
    features_df.to_csv(GOLD_DIR / "ml_features_dataset.csv", index=False)
    warehouse_tables = create_star_schema(gold_df)
    dashboard_outputs = create_dashboard_outputs(gold_df, features_df)
    etl_status = create_etl_status(
        raw_paths,
        bronze_paths,
        silver_merged,
        gold_df,
        features_df,
        warehouse_tables,
    )

    print("ETL pipeline completed successfully.")
    print(f"Bronze files: {[path.name for path in bronze_paths.values()]}")
    print(f"Silver shape: {silver_merged.shape}")
    print(f"Gold shape: {gold_df.shape}")
    print(f"ML features shape: {features_df.shape}")
    print(
        "Warehouse tables: "
        + ", ".join(f"{name}={table.shape}" for name, table in warehouse_tables.items())
    )
    print(
        "Dashboard outputs: "
        + ", ".join(
            f"{name}={value}"
            for name, value in dashboard_outputs.items()
            if name != "dashboard_kpis"
        )
    )
    print(f"ETL status written to: {ETL_STATUS_PATH}")
    print(f"Total missing values after silver step: {silver_merged.isnull().sum().sum()}")


if __name__ == "__main__":
    main()
