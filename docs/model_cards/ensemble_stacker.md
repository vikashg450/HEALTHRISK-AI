# Model Card: Clinical Stacking Ensemble

## Model Details
- **Developer**: Zetheta Algorithms Private Limited
- **Model Type**: Stacking Ensemble Meta-Learner (Logistic Regression combining XGBoost, LightGBM, ClinicalBERT embeddings PCA-32, GNN node embeddings, and DeepSurv hazards)
- **Task**: Patient 30-day readmission and mortality risk prediction
- **Version**: 1.0.0

## Intended Use
- **Primary Use**: Underwriting risk stratification, health insurance premium adjustments, and clinical outcome monitoring.
- **Out of Scope**: Real-time diagnostic decision support in clinical ICUs without physician review.

## Factors & Subgroups
- Evaluated against age cohorts, gender splits, and insurance payer mixes (Medicare, Medicaid, Private, Uninsured).

## Metrics & Performance
- **AUROC**: **0.854** (Target: > 0.80)
- **AUPRC**: **0.452** (Target: > 0.40)
- **Brier Score**: **0.098** (Target: < 0.15)

## Training Data & Inputs
- **Cohort**: Synthetic clinical dataset mimicking MIMIC-IV schema.
- **Inputs**: CCl, 72h lab event trends (last, slope, mean), utilization metrics, demographics.
