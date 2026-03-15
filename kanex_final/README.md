# Kanex Final ETL

This folder contains a fresh bronze/silver/gold ETL pipeline built from your original cleaning and merge logic, plus a star-schema warehouse model generated from the gold layer.

## Files

- `etl_pipeline.py`: copies the three source datasets, cleans each one individually, merges them with synthetic `student_id`, imputes missing values, and writes bronze/silver/gold outputs.
- `train_backlog_risk_model.py`: trains backlog-risk models and saves probability outputs for the next dashboard step.
- `warehouse_schema.sql`: SQL DDL for the star schema created from the gold dataset.

## Output layers

- `data/raw`: copied source files
- `data/bronze`: individually cleaned datasets
- `data/silver`: merged and imputed dataset
- `data/gold`: model-ready final dataset
- `data/gold/ml_features_dataset.csv`: derived ML feature set built from the gold dataset
- `data/warehouse`: star-schema dimension tables and the fact table
- `data/dashboard`: dashboard-ready CSV and JSON outputs refreshed on every pipeline run
- `data/predictions`: model prediction outputs

## Warehouse tables

- `fact_student_academic`
- `dim_student`
- `dim_academic_history`
- `dim_student_activity`
- `dim_behavior`
- `dim_subject`

## Dashboard outputs

- `student_overview.csv`
- `career_summary.csv`
- `semester_summary.csv`
- `risk_summary.csv`
- `branch_summary.csv`
- `dashboard_kpis.json`
## Run

```powershell
python etl_pipeline.py
python train_backlog_risk_model.py
streamlit run app.py
```
