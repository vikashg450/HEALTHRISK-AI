# 03. Feature Engineering

This document details the feature engineering pipeline used to preprocess raw data and construct predictor matrices for both clinical and financial models.

## Raw Data Inputs
1. **Patients**: Cohort demographics (Age, Gender, Race, Insurance Type).
2. **Admissions**: HADM records containing timelines, discharge codes, and ICU flags.
3. **Lab Events**: Time-series logs of lab tests (Creatinine, Hemoglobin, Sodium, WBC, Lactate, BNP).
4. **Diagnoses**: ICD-10 codes associated with patient admissions.
5. **Clinical Notes**: Discharge summary notes written by medical staff.
6. **Hospital Financials**: Operating margin, debt capitalization, beds, days cash, CMS quality metrics.
7. **Pharma Pipeline**: Clinical trial phases, indications, companies, sales estimates, enrollment velocity.

## Engineered Features

### 1. Patient Demographics & Baseline Risk
- **Age Splines**: Normalized continuous age and age category buckets.
- **Charlson Comorbidity Index (CCI)**: Computed using standard ICD-10 weights mapping chronic conditions (e.g., heart failure, diabetes, renal disease) into a single integer score capped at 15.
- **Payer Mix Codes**: One-hot representation of insurance type (Medicare, Medicaid, Private, Uninsured).

### 2. Time-Series Lab Trends (`compute_lab_trends`)
To capture patient trajectory, the pipeline processes time-series lab events within a sliding window (default 72 hours) and extracts:
- **Last Value**: The most recent lab result (e.g., `Lactate_last`).
- **Mean & Std Dev**: Metric central tendency and volatility during the admission.
- **Slope (Rate of Change)**: Polynomial fit modeling the trajectory direction (e.g., worsening kidney function indicated by rising creatinine slope).
- **Extremes**: Min and Max values during the admission.

### 3. Polypharmacy & Drug Risk Metrics
- **Medication Count**: Total unique drugs prescribed during admission.
- **Polypharmacy Flag**: Binary indicator set to 1 if unique medications $\ge 5$.
- **Hyperpolypharmacy Flag**: Binary indicator set to 1 if unique medications $\ge 10$.
- **Drug-Drug Interaction (DDI) Score**: Risk weight computed from NDC/ATC codes.

### 4. Past Healthcare Utilization
- **Prior Admissions Count**: Number of historical admissions within the lookback window.
- **Accumulated Length of Stay (LOS)**: Cumulative days spent hospitalized in the prior 12 months.
- **ICU Admits Count**: Historical frequency of critical care unit admissions.

### 5. Hospital Bond Risk Features
- **Enhanced Debt Service Coverage Ratio (DSCR)**: Adjusted by embedding the hospital's clinical quality metrics (CMS stars, 30-day readmissions) to model covenant breach probability.
- **CMI-Adjusted Days Cash**: Cash reserves normalized against patient Case Mix Index volatility.
