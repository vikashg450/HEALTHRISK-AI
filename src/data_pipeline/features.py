"""
HealthRisk AI — Data Cleaning & Preprocessing (Day 3)
Cleans raw MIMIC-IV and synthetic data, handles missing values,
standardizes formats, and validates data quality.
"""

import numpy as np
import pandas as pd
from typing import Tuple, List, Optional


# ──────────────────────────────────────────────────────
# PATIENT / ADMISSION CLEANER
# ──────────────────────────────────────────────────────

class ClinicalDataCleaner:
    """Cleans and validates clinical data from MIMIC-IV or synthetic sources."""

    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.cleaning_report = {}

    def clean_admissions(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean admissions table."""
        n_raw = len(df)
        df = df.copy()

        # Parse datetime columns
        for col in ["admittime", "dischtime", "deathtime", "edregtime"]:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors="coerce")

        # Compute LOS if not present
        if "los_days" not in df.columns and "admittime" in df and "dischtime" in df:
            df["los_days"] = (df["dischtime"] - df["admittime"]).dt.total_seconds() / 86400

        # Remove implausible LOS
        mask = df["los_days"].between(0.1, 365)
        df = df[mask]

        # Flag ICU admits
        if "care_unit" in df.columns:
            df["icu_flag"] = df["care_unit"].str.contains("ICU|CCU|CSRU", na=False).astype(int)

        self.cleaning_report["admissions"] = {
            "n_raw": n_raw, "n_clean": len(df),
            "n_removed": n_raw - len(df)
        }
        if self.verbose:
            print(f"Admissions: {n_raw:,} → {len(df):,} (removed {n_raw - len(df):,})")
        return df

    def clean_lab_events(
        self, df: pd.DataFrame,
        valid_ranges: Optional[dict] = None
    ) -> pd.DataFrame:
        """Clean lab events with physiologically plausible ranges."""
        default_ranges = {
            "Creatinine":  (0.1, 25.0),
            "Hemoglobin":  (2.0, 20.0),
            "WBC":         (0.1, 100.0),
            "Sodium":      (100, 180),
            "Potassium":   (1.5, 10.0),
            "BNP":         (1, 50000),
            "HbA1c":       (3.0, 18.0),
            "Lactate":     (0.1, 30.0),
        }
        if valid_ranges:
            default_ranges.update(valid_ranges)

        df = df.copy()
        df["value"] = pd.to_numeric(df["value"], errors="coerce")

        # Remove values outside physiological ranges
        def clip_lab(row):
            lab = row.get("lab_name", "")
            val = row["value"]
            if lab in default_ranges and pd.notna(val):
                lo, hi = default_ranges[lab]
                if val < lo or val > hi:
                    return np.nan
            return val

        df["value"] = df.apply(clip_lab, axis=1)
        df = df.dropna(subset=["value"])
        return df

    def clean_diagnoses(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and standardize ICD codes."""
        df = df.copy()
        # Remove dots from ICD-10 codes (e.g., I50.9 → I509)
        if "icd_code" in df.columns:
            df["icd_code_nodot"] = df["icd_code"].str.replace(".", "", regex=False)
            # Extract ICD chapter (first character for ICD-10)
            df["icd_chapter"] = df["icd_code"].str[0]
            # Extract 3-char category
            df["icd_category"] = df["icd_code"].str[:3]
        return df

    def impute_missing(
        self, df: pd.DataFrame,
        strategy: str = "median",
        numeric_cols: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """Impute missing values in numeric columns."""
        df = df.copy()
        if numeric_cols is None:
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        for col in numeric_cols:
            missing_pct = df[col].isna().mean()
            if missing_pct == 0:
                continue
            if strategy == "median":
                fill_val = df[col].median()
            elif strategy == "mean":
                fill_val = df[col].mean()
            elif strategy == "zero":
                fill_val = 0
            else:
                fill_val = df[col].median()
            df[col] = df[col].fillna(fill_val)
            if self.verbose and missing_pct > 0.05:
                print(f"  Imputed {col}: {missing_pct:.1%} missing → {strategy}={fill_val:.3f}")

        return df

    def generate_quality_report(self, df: pd.DataFrame, name: str = "dataset") -> dict:
        """Generate data quality report."""
        report = {
            "name": name,
            "n_rows": len(df),
            "n_cols": len(df.columns),
            "missing_pct": df.isna().mean().round(4).to_dict(),
            "dtypes": df.dtypes.astype(str).to_dict(),
            "duplicates": df.duplicated().sum(),
        }
        return report


# ──────────────────────────────────────────────────────
# FEATURE ENGINEERING
# ──────────────────────────────────────────────────────

class ClinicalFeatureEngineer:
    """
    Computes clinical features from cleaned MIMIC-IV or synthetic data.
    All features respect look-ahead bias prevention rules.
    """

    # Charlson Comorbidity Index ICD-10 weights
    CHARLSON_WEIGHTS = {
        "I21": 1, "I22": 1,   # Acute MI
        "I50": 1,              # CHF
        "I70": 1, "I71": 1,   # PVD
        "G45": 1,              # CVD/Stroke
        "F00": 1, "F01": 1, "F02": 1, "F03": 1,  # Dementia
        "J44": 1, "J43": 1,   # COPD
        "M05": 1, "M06": 1,   # Rheumatoid disease
        "K25": 1, "K26": 1,   # Peptic ulcer
        "B18": 1, "K70": 1,   # Liver disease (mild)
        "E10": 1, "E11": 1,   # Diabetes without complications
        "E12": 2, "E13": 2, "E14": 2,  # Diabetes with complications
        "G81": 2, "G82": 2,   # Hemi/paraplegia
        "N18": 2, "N19": 2,   # Renal disease
        "C00": 2,              # Cancer (solid tumor)
        "K72": 3, "K76": 3,   # Liver disease (severe)
        "C77": 6, "C78": 6, "C79": 6,  # Metastatic cancer
        "B20": 6, "B21": 6, "B22": 6, "B24": 6,  # AIDS/HIV
    }

    def compute_charlson_index(
        self, diagnoses_df: pd.DataFrame,
        id_col: str = "hadm_id"
    ) -> pd.Series:
        """Compute Charlson Comorbidity Index per admission."""
        def calc_cci(icd_codes):
            score = 0
            for code in icd_codes:
                prefix = str(code)[:3]
                score += self.CHARLSON_WEIGHTS.get(prefix, 0)
            return min(score, 15)  # Cap at 15

        cci = (
            diagnoses_df.groupby(id_col)["icd_code"]
            .apply(lambda codes: calc_cci(codes.tolist()))
            .rename("charlson_index")
        )
        return cci

    def compute_lab_trends(
        self,
        labs_df: pd.DataFrame,
        lab_name: str,
        window_hours: int = 72,
        id_col: str = "hadm_id"
    ) -> pd.DataFrame:
        """
        Compute lab trend features: last value, mean, std, slope, min, max.
        """
        subset = labs_df[labs_df["lab_name"] == lab_name].copy()
        if "draw_offset_hours" in subset.columns:
            subset = subset[subset["draw_offset_hours"] <= window_hours]

        def trend_stats(group):
            vals = group["value"].values
            if len(vals) == 0:
                return pd.Series({
                    f"{lab_name}_last": np.nan,
                    f"{lab_name}_mean": np.nan,
                    f"{lab_name}_std": np.nan,
                    f"{lab_name}_slope": np.nan,
                    f"{lab_name}_min": np.nan,
                    f"{lab_name}_max": np.nan,
                    f"{lab_name}_n_draws": 0,
                })
            x = np.arange(len(vals), dtype=float)
            slope = np.polyfit(x, vals, 1)[0] if len(vals) > 1 else 0.0
            return pd.Series({
                f"{lab_name}_last": vals[-1],
                f"{lab_name}_mean": vals.mean(),
                f"{lab_name}_std": vals.std() if len(vals) > 1 else 0.0,
                f"{lab_name}_slope": slope,
                f"{lab_name}_min": vals.min(),
                f"{lab_name}_max": vals.max(),
                f"{lab_name}_n_draws": len(vals),
            })

        return subset.groupby(id_col).apply(trend_stats).reset_index()

    def compute_polypharmacy_features(
        self,
        prescriptions_df: pd.DataFrame,
        id_col: str = "hadm_id"
    ) -> pd.DataFrame:
        """Compute medication-related features."""
        if "drug_class" not in prescriptions_df.columns:
            prescriptions_df = prescriptions_df.copy()
            prescriptions_df["drug_class"] = "Unknown"

        feats = prescriptions_df.groupby(id_col).agg(
            n_medications=("drug", "nunique"),
            n_drug_classes=("drug_class", "nunique"),
        ).reset_index()
        feats["polypharmacy_flag"] = (feats["n_medications"] >= 5).astype(int)
        feats["hyperpolypharmacy_flag"] = (feats["n_medications"] >= 10).astype(int)
        return feats

    def compute_utilization_features(
        self,
        admissions_df: pd.DataFrame,
        id_col: str = "subject_id",
        lookback_months: int = 12
    ) -> pd.DataFrame:
        """Compute prior utilization features (admissions, LOS, etc.)."""
        feats = admissions_df.groupby(id_col).agg(
            total_admissions=("hadm_id", "count"),
            total_los=("los_days", "sum"),
            mean_los=("los_days", "mean"),
            icu_admissions=("icu_flag", "sum") if "icu_flag" in admissions_df.columns else ("hadm_id", "count"),
        ).reset_index()
        return feats

    def build_master_feature_table(
        self,
        admissions: pd.DataFrame,
        patients: pd.DataFrame,
        labs: pd.DataFrame,
        diagnoses: pd.DataFrame,
        prescriptions: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """
        Join all features into a single master feature table.
        Returns one row per admission with all features.
        """
        print("Building master feature table...")

        # Start from admissions
        master = admissions.merge(
            patients[["subject_id", "age", "gender", "race", "insurance",
                      "num_prior_admissions", "charlson_index"]],
            on="subject_id", how="left"
        )

        # CCI from diagnoses
        cci = self.compute_charlson_index(diagnoses)
        master = master.merge(cci.rename("cci_computed"), on="hadm_id", how="left")

        # Lab trends
        for lab in ["Creatinine", "Hemoglobin", "WBC", "Sodium", "Lactate"]:
            if "lab_name" in labs.columns and lab in labs["lab_name"].values:
                lab_feats = self.compute_lab_trends(labs, lab_name=lab)
                master = master.merge(lab_feats, on="hadm_id", how="left")

        # Polypharmacy
        if prescriptions is not None:
            poly = self.compute_polypharmacy_features(prescriptions)
            master = master.merge(poly, on="hadm_id", how="left")

        # Encode categorical
        master["gender_binary"] = (master["gender"] == "M").astype(int)
        master["emergency_flag"] = (master.get("admission_type", "") == "EMERGENCY").astype(int)

        # Fill remaining NaN
        numeric_cols = master.select_dtypes(include=[np.number]).columns
        master[numeric_cols] = master[numeric_cols].fillna(master[numeric_cols].median())

        print(f"Master feature table: {master.shape[0]:,} rows × {master.shape[1]} features")
        return master


if __name__ == "__main__":
    # Quick test with synthetic data
    from synthetic_data import generate_all_datasets

    datasets = generate_all_datasets(output_dir="data/raw", seed=42)
    cleaner = ClinicalDataCleaner(verbose=True)
    engineer = ClinicalFeatureEngineer()

    admissions_clean = cleaner.clean_admissions(datasets["admissions"])
    labs_clean = cleaner.clean_lab_events(datasets["labs"])
    diagnoses_clean = cleaner.clean_diagnoses(datasets["diagnoses"])

    master = engineer.build_master_feature_table(
        admissions=admissions_clean,
        patients=datasets["patients"],
        labs=labs_clean,
        diagnoses=diagnoses_clean,
    )
    master.to_csv("data/features/master_features.csv", index=False)
    print("\nSample features:")
    print(master.head(3))
