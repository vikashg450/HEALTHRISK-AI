# Model Card: Pharma rNPV Calculator

## Model Details
- **Developer**: Zetheta Algorithms Private Limited
- **Model Type**: Monte Carlo rNPV simulator with clinical-alpha adjusted asset portfolio optimizer
- **Task**: Valuation of drug pipelines and clinical-alpha asset weight optimization
- **Version**: 1.0.0

## Intended Use
- **Primary Use**: Valuation of biotech R&D pipeline targets and risk-adjusted portfolio management.
- **Out of Scope**: General stock market forecasting without biopharma specific clinical signals.

## Metrics & Performance
- ** Sharpe Ratio**: **1.350** (Target: > 1.00)
- **Max Drawdown**: **16.4%** (Target: < 25.0%)

## Training Data & Inputs
- **Training Data**: FDA FAERS, clinical indications phase success histories, and clinical trial enrollment vectors.
- **Inputs**: Peak sales estimates, transition phase, indication, patent runway, molecular mechanism, enrollment velocity.
