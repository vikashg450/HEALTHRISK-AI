# 🏥 HealthRisk AI

> **Dual-Domain AI System: Healthcare Intelligence × Financial Risk Models**  
> *Category: Data Science | Healthcare Analytics | Financial Risk Modelling | AI/ML*  
> *Timeline: 15 Days | Zetheta Algorithms Private Limited*

---

## 🎯 Project Overview

HealthRisk AI builds a first-of-its-kind convergence system that embeds clinical intelligence — patient risk stratification, epidemiological modelling, pharmaceutical efficacy signals — directly into financial risk models for:

- **🏦 Health Insurance** — Actuarial premium pricing with clinical trajectory features
- **🏥 Hospital Credit Risk** — Hospital bond credit scoring using clinical quality as leading indicators
- **💊 Pharmaceutical Portfolio** — rNPV valuation, trial pipeline monitoring, portfolio optimisation
- **🎮 HealthRisk Lab** — Gamified simulation where players manage healthcare investments

---

## 🗂️ Project Structure

```
healthrisk_ai/
├── config/
│   └── config.yaml              # Master configuration
├── data/
│   ├── raw/                     # MIMIC-IV, WHO, FDA FAERS (downloaded)
│   ├── processed/               # Cleaned datasets
│   └── features/                # Feature-engineered tables
├── src/
│   ├── data_pipeline/
│   │   ├── ingestion.py         # MIMIC-IV, ClinicalTrials, FDA FAERS, WHO APIs
│   │   ├── features.py          # Cleaning + Charlson/lab/polypharmacy features
│   │   └── synthetic_data.py   # Realistic synthetic data generator (MIMIC fallback)
│   ├── clinical_nlp/
│   │   └── clinicalbert.py     # ClinicalBERT fine-tuning, NER pipeline, complexity scoring
│   ├── graph_nn/
│   │   └── gat_model.py        # Graph Attention Network (patient-disease-drug graph)
│   ├── survival/
│   │   └── survival_models.py  # Cox PH, DeepSurv, hospital covenant breach prediction
│   ├── ensemble/
│   │   └── stacker.py          # XGBoost + LightGBM + BERT + GNN + Survival stacking
│   ├── financial/
│   │   ├── insurance/
│   │   │   └── actuarial.py    # GLM pricing, IBNR reserves, risk stratification
│   │   ├── credit_risk/
│   │   │   └── hospital_credit.py  # Credit scorecard, PD model, early warning system
│   │   └── pharma/
│   │       └── rnpv_calculator.py  # Phase success model, Monte Carlo rNPV, portfolio opt
│   ├── simulation/
│   │   └── engine.py           # HealthRisk Lab — 10+ scenarios, AI opponent, 1000pt scoring
│   └── explainability/
│       └── shap_analysis.py    # SHAP, counterfactuals, PDP, model cards, regulatory mapping
├── notebooks/                   # Jupyter EDA and exploration notebooks
├── tests/
│   └── test_all.py             # 60+ tests covering all modules (target: ≥80% coverage)
├── models/                      # Saved model artifacts
├── reports/                     # Generated analysis reports & plots
├── docs/
│   └── model_cards/            # Model card markdown files
├── environment.yml              # Conda environment specification
└── README.md
```

---

## ⚡ Quick Start

### 1. Setup Environment
```bash
conda env create -f environment.yml
conda activate healthrisk
```

### 2. Generate Synthetic Data (if MIMIC-IV not yet available)
```bash
cd healthrisk_ai
python src/data_pipeline/synthetic_data.py
```

### 3. Run Data Pipeline
```python
from src.data_pipeline.features import ClinicalDataCleaner, ClinicalFeatureEngineer
from src.data_pipeline.synthetic_data import generate_all_datasets

datasets = generate_all_datasets(output_dir="data/raw")
cleaner = ClinicalDataCleaner()
engineer = ClinicalFeatureEngineer()

admissions = cleaner.clean_admissions(datasets["admissions"])
master_features = engineer.build_master_feature_table(
    admissions, datasets["patients"], datasets["labs"], datasets["diagnoses"]
)
```

### 4. Train Clinical AI Ensemble
```python
from src.ensemble.stacker import HealthRiskEnsemble
import numpy as np

# Load your feature matrix and labels
X = master_features[feature_cols].values
y = master_features["readmission_30d"].values

ensemble = HealthRiskEnsemble(n_splits=5, seed=42)
metrics = ensemble.fit(X, y)
print(f"AUROC: {metrics['auroc']:.4f}")  # Target > 0.80
```

### 5. Run Hospital Credit Risk Scoring
```python
from src.financial.credit_risk.hospital_credit import HospitalPDModel

pd_model = HospitalPDModel()
pd_model.fit(datasets["hospitals"])
predictions = pd_model.predict_pd(datasets["hospitals"])
print(predictions[["hospital_id", "pd_enhanced", "rating_enhanced", "spread_delta_bps"]].head())
```

### 6. Play HealthRisk Lab
```python
from src.simulation.engine import HealthRiskLabEngine

engine = HealthRiskLabEngine(start_year=2010, end_year=2020, seed=42)
while not engine.is_game_over():
    player_decision = {
        "rebalance": {
            "Hospital Bond": "NEUTRAL",
            "Pharma Equity": "OVERWEIGHT",
            "Insurance Book": "UNDERWEIGHT",
        }
    }
    result = engine.run_quarter(player_decision)
    print(f"Q{result['quarter']}: {result['scenario']['name']}")

final = engine.get_final_results()
print(f"Player: {final['player_final_score']} | AI: {final['ai_final_score']}")
```

### 7. Run Tests
```bash
cd healthrisk_ai
pytest tests/ -v --cov=src --cov-report=html --cov-fail-under=80
```

---

## 📊 Evaluation Targets

| Domain | Metric | Target |
|---|---|---|
| Clinical Prediction | AUROC | > 0.80 |
| Clinical Prediction | AUPRC | > 0.40 |
| Clinical Prediction | Brier Score | < 0.15 |
| Survival Models | C-index | > 0.70 |
| Insurance Actuarial | R-squared | > 25% |
| Insurance Actuarial | MAPE | < 15% |
| Insurance Actuarial | Predictive Ratio | 0.95 – 1.05 |
| Hospital Credit | Gini Coefficient | > 0.50 |
| Hospital Credit | KS Statistic | > 0.30 |
| Pharma Portfolio | Sharpe Ratio | > 1.0 |
| Pharma Portfolio | Information Ratio | > 0.50 |
| Pharma Portfolio | Max Drawdown | < 25% |

---

## 📚 Data Sources

| Source | Access | Usage |
|---|---|---|
| [MIMIC-IV](https://physionet.org/content/mimiciv/) | PhysioNet (credentialed) | Clinical data — patient outcomes, labs, diagnoses |
| [ClinicalTrials.gov](https://clinicaltrials.gov/data-api/api) | Free API | Pharma pipeline signals |
| [FDA FAERS](https://open.fda.gov/apis/drug/event/) | Free API | Drug adverse event signals |
| [WHO GHO](https://www.who.int/data/gho/info/gho-odata-api) | Free API | Population epidemiology |
| Synthetic data | `src/data_pipeline/synthetic_data.py` | Development fallback |

---

## 🔑 Key Models

| Module | Model | Purpose |
|---|---|---|
| Clinical NLP | ClinicalBERT | Risk classification, NER, cost prediction from notes |
| Graph Neural Network | 3-layer GAT | Patient-disease-drug interaction modelling |
| Survival Analysis | DeepSurv / Cox PH | Time-to-readmission, financial covenant breach |
| Ensemble | XGBoost + LightGBM + stacker | Final clinical risk prediction |
| Insurance | GLM + Gradient Boosting | Premium pricing, IBNR reserves |
| Credit Risk | Gradient Boosting PD | Hospital default probability |
| Pharma | Monte Carlo rNPV | Drug pipeline valuation |
| Simulation | Rule-based AI opponent | HealthRisk Lab game engine |

---

## ⚠️ Important Notes

1. **MIMIC-IV Access**: Apply at [PhysioNet](https://physionet.org) immediately (takes 2-4 weeks). Use `synthetic_data.py` as fallback.
2. **API Keys**: Store in `.env` file (never in code). Use `python-dotenv` to load.
3. **GPU**: ClinicalBERT and GNN training benefit greatly from GPU. Use Google Colab or Kaggle if local GPU unavailable.
4. **Data Leakage**: Always use time-aware cross-validation (see `stacker.py`).

---

## 📋 Mandatory Deliverables Checklist

### Code (12 modules)
- [x] Data pipeline (ingestion, cleaning, features)
- [x] Clinical NLP module (ClinicalBERT, NER, complexity scoring)
- [x] Graph Neural Network (GAT, graph construction, embeddings)
- [x] Survival analysis (Cox PH, DeepSurv, financial survival)
- [x] XGBoost/LightGBM baselines
- [x] Stacking ensemble with meta-learner
- [x] Insurance actuarial module (pricing, IBNR, stratification)
- [x] Hospital credit risk module (scorecard, PD model, early warning)
- [x] Pharmaceutical analytics (rNPV, portfolio optimisation)
- [x] HealthRisk Lab simulation engine (11 scenarios, AI opponent)
- [x] Model explainability (SHAP, counterfactuals, model cards)
- [x] Unit & integration tests (60+ tests, target ≥80% coverage)

### Documentation (8 docs)
- [ ] `docs/01_technical_architecture.md`
- [ ] `docs/02_model_performance_report.md`
- [ ] `docs/03_feature_engineering.md`
- [ ] `docs/04_healthcare_domain_knowledge.md`
- [ ] `docs/05_financial_applications.md`
- [ ] `docs/06_healthrisk_lab_game_design.md`
- [ ] `docs/model_cards/*.md` (one per model)
- [x] `README.md`

---

*Project by Zetheta Algorithms Private Limited — HealthRisk Capital Partners simulation.*
