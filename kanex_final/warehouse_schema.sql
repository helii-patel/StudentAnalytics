CREATE TABLE dim_student (
    student_id INTEGER PRIMARY KEY,
    gender TEXT,
    branch TEXT,
    department TEXT,
    category TEXT,
    heightcm REAL,
    weightkg REAL,
    financial_status TEXT,
    parttime_job TEXT
);

CREATE TABLE dim_academic_history (
    student_id INTEGER PRIMARY KEY,
    marks_10th REAL,
    marks_12th REAL,
    board_10th TEXT,
    board_12th TEXT,
    gpa_1 REAL,
    gpa_2 REAL,
    gpa_3 REAL,
    gpa_4 REAL,
    gpa_5 REAL,
    gpa_6 REAL,
    cgpa REAL,
    rank REAL,
    normalized_rank REAL,
    FOREIGN KEY (student_id) REFERENCES dim_student(student_id)
);

CREATE TABLE dim_student_activity (
    student_id INTEGER PRIMARY KEY,
    technical_projects REAL,
    tech_quiz REAL,
    olympiads_qualified REAL,
    certification_course TEXT,
    ntse_scholarships REAL,
    miscellany_tech_events REAL,
    engg_coaching REAL,
    FOREIGN KEY (student_id) REFERENCES dim_student(student_id)
);

CREATE TABLE dim_behavior (
    student_id INTEGER PRIMARY KEY,
    daily_study_time TEXT,
    prefer_to_study_in TEXT,
    social_media_time TEXT,
    travelling_time TEXT,
    stress_level TEXT,
    hobbies TEXT,
    like_degree TEXT,
    career_willingness REAL,
    salary_expectation REAL,
    FOREIGN KEY (student_id) REFERENCES dim_student(student_id)
);

CREATE TABLE dim_subject (
    subjectid REAL PRIMARY KEY,
    subjectname TEXT,
    theorycredit REAL,
    practicalcredit REAL
);

CREATE TABLE fact_student_academic (
    student_id INTEGER,
    semester INTEGER,
    subjectid REAL,
    theoryagggradepoint REAL,
    theorycredit REAL,
    practicalagggradepoint REAL,
    practicalcredit REAL,
    sgpa REAL,
    prsgpa REAL,
    prcgpa REAL,
    noofbacklog REAL,
    thispresent REAL,
    prispresent REAL,
    FOREIGN KEY (student_id) REFERENCES dim_student(student_id),
    FOREIGN KEY (subjectid) REFERENCES dim_subject(subjectid)
);
