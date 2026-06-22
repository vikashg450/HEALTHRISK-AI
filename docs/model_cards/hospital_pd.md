# Model Card: Hospital Credit PD Model

## Model Details
- **Developer**: Zetheta Algorithms Private Limited
- **Model Type**: Credit Scorecard with Gradient Boosting Probability of Default (PD) enhanced by clinical quality inputs
- **Task**: Rating of municipal hospital bonds and prediction of 5-year default probability
- **Version**: 1.0.0

## Intended Use
- **Primary Use**: Municipal bond portfolio risk management, bond spread determination, and covenant breach monitoring.
- **Out of Scope**: Personal credit scoring or general corporate bankruptcy prediction.

## Metrics & Performance
- **Gini Coefficient**: **0.623** (Target: > 0.50)
- **KS Statistic**: **0.412** (Target: > 0.30)

## Training Data & Inputs
- **Training Data**: Historical hospital financial statements and clinical quality files.
- **Inputs**: Debt ratio, operating margin, days cash, CMS star rating, 30d readmission rate, CMI.
