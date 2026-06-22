"""
HealthRisk AI — Synthetic Data Generator (Day 2 Fallback)
Generates realistic synthetic patient, hospital, and pharmaceutical data
when MIMIC-IV access is pending.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import random


def set_seed(seed: int = 42):
    np.random.seed(seed)
    random.seed(seed)


# ──────────────────────────────────────────────────────
# SYNTHETIC PATIENT DATA
# ──────────────────────────────────────────────────────

def generate_patients(n: int = 10_000, seed: int = 42) -> pd.DataFrame:
    """Generate synthetic patient demographics."""
    set_seed(seed)
    ages = np.random.normal(65, 15, n).clip(18, 100).astype(int)
    df = pd.DataFrame({
        "subject_id": range(1, n + 1),
        "age": ages,
        "gender": np.random.choice(["M", "F"], n, p=[0.48, 0.52]),
        "race": np.random.choice(
            ["White", "Black", "Hispanic", "Asian", "Other"],
            n, p=[0.60, 0.13, 0.18, 0.06, 0.03]
        ),
        "insurance": np.random.choice(
            ["Medicare", "Medicaid", "Private", "Uninsured"],
            n, p=[0.45, 0.20, 0.30, 0.05]
        ),
        "num_prior_admissions": np.random.poisson(2.5, n).clip(0, 20),
        "num_chronic_conditions": np.random.poisson(3.5, n).clip(0, 15),
        "charlson_index": np.random.poisson(2.8, n).clip(0, 15),
    })
    return df


def generate_admissions(patients: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """Generate synthetic hospital admissions."""
    set_seed(seed)
    n = len(patients)
    start_date = datetime(2019, 1, 1)

    los = np.random.exponential(5, n).clip(1, 90)
    admission_dates = [
        start_date + timedelta(days=int(np.random.uniform(0, 365 * 3)))
        for _ in range(n)
    ]

    df = pd.DataFrame({
        "hadm_id": range(10001, 10001 + n),
        "subject_id": patients["subject_id"].values,
        "admittime": admission_dates,
        "los_days": los.round(1),
        "admission_type": np.random.choice(
            ["EMERGENCY", "ELECTIVE", "URGENT"], n, p=[0.55, 0.30, 0.15]
        ),
        "discharge_location": np.random.choice(
            ["Home", "SNF", "Rehab", "AMA", "Expired"],
            n, p=[0.55, 0.20, 0.10, 0.05, 0.10]
        ),
        # Outcomes
        "hospital_expire_flag": np.random.binomial(1, 0.08, n),
        "readmission_30d": np.random.binomial(1, 0.15, n),
        "icu_flag": np.random.binomial(1, 0.30, n),
        # Costs
        "total_charges": np.random.lognormal(10.5, 1.2, n).round(2),
    })
    return df


def generate_lab_events(admissions: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """Generate synthetic lab results for each admission."""
    set_seed(seed)
    records = []
    key_labs = {
        "Creatinine":     (1.0,  0.5,  0.5,  5.0),
        "Hemoglobin":     (12.5, 2.0,  6.0,  18.0),
        "WBC":            (8.0,  3.5,  2.0,  30.0),
        "Sodium":         (139,  4.0,  120,  155),
        "Potassium":      (4.0,  0.6,  2.5,  6.5),
        "BNP":            (200,  300,  10,   5000),
        "HbA1c":          (7.0,  1.5,  4.5,  14.0),
        "Lactate":        (1.5,  1.0,  0.5,  15.0),
    }
    for _, row in admissions.iterrows():
        n_draws = np.random.randint(3, 12)
        for lab_name, (mean, std, lo, hi) in key_labs.items():
            for draw in range(n_draws):
                records.append({
                    "hadm_id": row["hadm_id"],
                    "lab_name": lab_name,
                    "value": np.clip(np.random.normal(mean, std), lo, hi).round(2),
                    "draw_offset_hours": draw * np.random.uniform(4, 24),
                })
    return pd.DataFrame(records)


def generate_diagnoses(admissions: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """Generate synthetic ICD-10 diagnoses."""
    set_seed(seed)
    common_icd10 = {
        "I50.9": "Heart failure, unspecified",
        "E11.9": "Type 2 diabetes mellitus without complications",
        "J44.1": "COPD with acute exacerbation",
        "N18.3": "Chronic kidney disease, stage 3",
        "I21.9": "Acute myocardial infarction, unspecified",
        "J18.9": "Pneumonia, unspecified",
        "F32.9": "Major depressive disorder, single episode",
        "I10":   "Essential hypertension",
        "E78.5": "Hyperlipidemia, unspecified",
        "K92.1": "Melena",
        "A41.9": "Sepsis, unspecified organism",
        "G47.33": "Obstructive sleep apnea",
    }
    records = []
    icd_codes = list(common_icd10.keys())
    for _, row in admissions.iterrows():
        n_dx = np.random.randint(2, 8)
        for i, code in enumerate(np.random.choice(icd_codes, n_dx, replace=False)):
            records.append({
                "hadm_id": row["hadm_id"],
                "icd_code": code,
                "icd_version": 10,
                "seq_num": i + 1,
                "long_title": common_icd10[code],
            })
    return pd.DataFrame(records)


def generate_clinical_notes(admissions: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """Generate synthetic discharge summary templates."""
    set_seed(seed)
    templates = [
        "Patient is a {age}-year-old {gender} admitted for {dx}. "
        "History of {hx}. "
        "Physical exam notable for {exam}. "
        "Labs showed {labs}. "
        "Patient was treated with {tx} and discharged in {condition} condition.",
        "This {age}-year-old {gender} presents with {dx}. "
        "Past medical history significant for {hx}. "
        "Assessment: {dx} managed with {tx}. "
        "Disposition: {disposition}.",
    ]
    diagnoses = ["heart failure exacerbation", "COPD exacerbation",
                 "sepsis", "acute MI", "pneumonia", "DKA", "AKI"]
    hx_options = ["hypertension, diabetes", "COPD, CHF", "CKD, anemia",
                  "CAD, hyperlipidemia", "atrial fibrillation, heart failure"]
    exam_options = ["bilateral crackles", "elevated JVP", "wheeze",
                    "diaphoresis", "altered mental status"]
    lab_options = ["elevated creatinine", "leukocytosis", "elevated troponin",
                   "metabolic acidosis", "low hemoglobin"]
    tx_options = ["IV furosemide and oxygen", "broad-spectrum antibiotics",
                  "bronchodilators and steroids", "aspirin and heparin",
                  "insulin infusion and IV fluids"]
    records = []
    for _, row in admissions.iterrows():
        tmpl = random.choice(templates)
        note = tmpl.format(
            age=np.random.randint(45, 90),
            gender=random.choice(["male", "female"]),
            dx=random.choice(diagnoses),
            hx=random.choice(hx_options),
            exam=random.choice(exam_options),
            labs=random.choice(lab_options),
            tx=random.choice(tx_options),
            condition=random.choice(["stable", "improved", "fair"]),
            disposition=random.choice(["home", "skilled nursing facility", "rehab"]),
        )
        records.append({"hadm_id": row["hadm_id"], "note_text": note,
                        "note_type": "Discharge summary"})
    return pd.DataFrame(records)


# ──────────────────────────────────────────────────────
# SYNTHETIC HOSPITAL FINANCIAL DATA
# ──────────────────────────────────────────────────────

def generate_hospital_financials(n_hospitals: int = 200, seed: int = 42) -> pd.DataFrame:
    """Generate synthetic hospital financial and clinical quality data."""
    set_seed(seed)
    np.random.seed(seed)

    df = pd.DataFrame({
        "hospital_id": range(1, n_hospitals + 1),
        "hospital_name": [f"Hospital_{i:03d}" for i in range(1, n_hospitals + 1)],
        "hospital_type": np.random.choice(
            ["Academic", "Community", "Critical Access", "Specialty"],
            n_hospitals, p=[0.15, 0.55, 0.20, 0.10]
        ),
        "beds": np.random.randint(25, 800, n_hospitals),
        # Financial metrics
        "operating_margin": np.random.normal(0.025, 0.04, n_hospitals).round(4),
        "dscr": np.random.normal(2.5, 0.8, n_hospitals).clip(0.5, 8.0).round(2),
        "days_cash_on_hand": np.random.normal(120, 50, n_hospitals).clip(0, 400).round(0),
        "debt_to_capitalization": np.random.normal(0.40, 0.15, n_hospitals).clip(0, 0.95).round(3),
        "revenue_growth_yoy": np.random.normal(0.03, 0.06, n_hospitals).round(4),
        "total_revenue_m": np.random.lognormal(5.0, 1.2, n_hospitals).round(1),
        # Clinical quality metrics
        "readmission_rate_30d": np.random.normal(0.158, 0.03, n_hospitals).clip(0.05, 0.30).round(4),
        "hcahps_star": np.random.choice([1, 2, 3, 4, 5], n_hospitals, p=[0.05, 0.15, 0.35, 0.30, 0.15]),
        "case_mix_index": np.random.normal(1.55, 0.35, n_hospitals).clip(0.8, 3.5).round(3),
        "cmi_trend_yoy": np.random.normal(0.0, 0.05, n_hospitals).round(4),
        "ed_boarding_hours": np.random.normal(4.5, 2.5, n_hospitals).clip(0.5, 15).round(1),
        "cms_star_rating": np.random.choice([1, 2, 3, 4, 5], n_hospitals, p=[0.07, 0.18, 0.35, 0.28, 0.12]),
        # Payer mix
        "medicare_pct": np.random.normal(0.45, 0.12, n_hospitals).clip(0.1, 0.80).round(3),
        "medicaid_pct": np.random.normal(0.22, 0.10, n_hospitals).clip(0.05, 0.55).round(3),
        # Target: 1 = default within 5 years, 0 = performing
        "default_within_5yr": np.random.binomial(1, 0.05, n_hospitals),
    })
    return df


# ──────────────────────────────────────────────────────
# SYNTHETIC PHARMACEUTICAL PIPELINE DATA
# ──────────────────────────────────────────────────────

def generate_pharma_pipeline(n_trials: int = 300, seed: int = 42) -> pd.DataFrame:
    """Generate synthetic clinical trial pipeline data."""
    set_seed(seed)
    indications = ["Oncology", "Cardiovascular", "CNS", "Immunology",
                   "Rare Disease", "Infectious Disease", "Metabolic"]
    mechanisms = ["mAb", "Small molecule", "ADC", "Cell therapy",
                  "Gene therapy", "Bispecific", "SiRNA"]
    endpoints = ["Overall Survival", "Progression-Free Survival",
                 "Complete Remission", "HbA1c reduction", "MACE reduction",
                 "Clinical response rate"]
    companies = [f"BioPharma_{chr(65+i)}" for i in range(20)]

    df = pd.DataFrame({
        "trial_id": [f"NCT{np.random.randint(1000000, 9999999)}" for _ in range(n_trials)],
        "company": np.random.choice(companies, n_trials),
        "indication": np.random.choice(indications, n_trials),
        "mechanism": np.random.choice(mechanisms, n_trials),
        "phase": np.random.choice(["Phase I", "Phase II", "Phase III"], n_trials, p=[0.35, 0.40, 0.25]),
        "primary_endpoint": np.random.choice(endpoints, n_trials),
        "enrollment_target": np.random.randint(50, 1500, n_trials),
        "enrollment_current": None,  # filled below
        "sites": np.random.randint(5, 200, n_trials),
        "countries": np.random.randint(1, 25, n_trials),
        "months_to_interim": np.random.randint(3, 36, n_trials),
        # Financial signals
        "peak_sales_estimate_b": np.random.lognormal(0.5, 1.0, n_trials).round(2),
        "patent_years_remaining": np.random.randint(2, 20, n_trials),
        "market_cap_b": np.random.lognormal(1.5, 1.2, n_trials).round(2),
        # Outcome (for ML training)
        "phase_success": None,  # filled below
    })

    # Fill enrollment (percentage of target enrolled)
    pct_enrolled = np.random.uniform(0.3, 1.0, n_trials)
    df["enrollment_current"] = (df["enrollment_target"] * pct_enrolled).astype(int)
    df["enrollment_pct"] = pct_enrolled.round(3)
    df["enrollment_velocity"] = (df["enrollment_current"] / df["months_to_interim"].clip(1)).round(1)

    # Phase success based on realistic probabilities
    phase_probs = {"Phase I": 0.63, "Phase II": 0.31, "Phase III": 0.493}
    df["phase_success"] = df["phase"].apply(
        lambda p: np.random.binomial(1, phase_probs[p])
    )
    return df


# ──────────────────────────────────────────────────────
# MAIN GENERATOR
# ──────────────────────────────────────────────────────

def generate_all_datasets(output_dir: str = "data/raw", seed: int = 42) -> dict:
    """Generate and save all synthetic datasets."""
    import os
    os.makedirs(output_dir, exist_ok=True)

    print("Generating synthetic datasets...")
    patients    = generate_patients(n=10_000, seed=seed)
    admissions  = generate_admissions(patients, seed=seed)
    labs        = generate_lab_events(admissions.head(2000), seed=seed)  # subset for speed
    diagnoses   = generate_diagnoses(admissions, seed=seed)
    notes       = generate_clinical_notes(admissions.head(2000), seed=seed)
    hospitals   = generate_hospital_financials(n_hospitals=200, seed=seed)
    pharma      = generate_pharma_pipeline(n_trials=300, seed=seed)

    datasets = {
        "patients":   patients,
        "admissions": admissions,
        "labs":       labs,
        "diagnoses":  diagnoses,
        "notes":      notes,
        "hospitals":  hospitals,
        "pharma":     pharma,
    }

    for name, df in datasets.items():
        path = os.path.join(output_dir, f"{name}.csv")
        df.to_csv(path, index=False)
        print(f"  ✓ Saved {name}: {df.shape[0]:,} rows × {df.shape[1]} cols → {path}")

    return datasets


if __name__ == "__main__":
    generate_all_datasets()
