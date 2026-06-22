# 06. HealthRisk Lab Game Design

This document details the game design mechanics and simulation scenarios of the **HealthRisk Lab**, an interactive simulation dashboard.

## Simulation Mechanics
The lab simulates a portfolio management environment where players act as healthcare fund managers. The game spans **1 year (4 quarters)**:
- **Starting Capital**: $100M.
- **Assets**: Hospital Bonds, Pharma Equity, Insurance Book, Healthcare REIT.
- **Actions**: Rebalance asset allocation weights each quarter (UNDERWEIGHT, NEUTRAL, OVERWEIGHT).

## Macroeconomic & Clinical Scenarios
The simulation library hosts **11 unique scenarios** that trigger random shock events:
1. **CMS Medicare Rate Cut**: Reduces operating margins across hospitals, hurting bond yields.
2. **Value-Based Care Contract Win**: Boosts hospital revenues for facilities with high CMS star ratings.
3. **Novel Pathogen Outbreak**: Creates an epidemic shock, spiking hospital admissions and insurance payouts while driving up specific biopharma stocks.
4. **FDA Drug Recall**: Causes sharp decline in specific pharma holdings.
5. **Patent Cliff Expiry**: Accelerates revenue erosion for biopharma assets.
6. **Medicaid Expansion**: Increases coverage rates, boosting hospital margins.
7. **Biotech Breakthrough**: Spikes pharma equity returns.
8. **Underwriting Loss**: Unexpected claims spike.
9. **Interest Rate Hike**: Compresses asset returns across all classes.
10. **REIT Refinancing Shock**: Hurts Healthcare REIT return rates.
11. **Neutral Quarter**: Standard baseline performance.

## AI Decision Opponent
At the end of each quarter, the player's performance is compared against the **HealthRisk AI opponent**.
- The AI opponent uses the integrated clinical and financial predictive models (GAT embeddings, DeepSurv hazards, and rNPV calculations) to determine optimal risk-mitigating weights for the active shock scenario.
- AI decisions include a textual **rationale** displaying its strategic evaluation.

## Scoring System
Points are awarded dynamically based on:
1. **Portfolio Returns**: Outperforming the baseline yield.
2. **Risk Mitigation**: Avoiding assets hit by predicted defaults or clinical shocks.
3. **Opponent Outperformance**: Earning bonus points for beating the AI's quarterly return.
4. **Maximum Target**: A perfect run can score up to **1,000 points**.
