"""
HealthRisk AI — Hospital Credit Risk Module (Day 11)
Implements:
  1. Traditional financial scorecard (financial ratios only)
  2. Enhanced scorecard (financial + clinical quality metrics)
  3. Probability of Default (PD) model
  4. Early Warning System detecting clinical quality deterioration
  5. Bond spread prediction linking credit scores to market risk

Key insight: Clinical quality metrics lead financial metrics by 6-12 months.
Targets: Gini Coefficient > 0.50, KS Statistic > 0.30
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings("ignore")


# ──────────────────────────────────────────────────────
# CREDIT SCORECARD
# ──────────────────────────────────────────────────────

FINANCIAL_FEATURES = [
    "operating_margin",
    "dscr",
    "days_cash_on_hand",
    "debt_to_capitalization",
    "revenue_growth_yoy",
    "total_revenue_m",
]

CLINICAL_QUALITY_FEATURES = [
    "readmission_rate_30d",
    "hcahps_star",
    "case_mix_index",
    "cmi_trend_yoy",
    "ed_boarding_hours",
    "cms_star_rating",
    "medicare_pct",
    "medicaid_pct",
]


class HospitalCreditScorecard:
    """
    Generates credit scores for hospitals using financial and/or clinical metrics.
    Scores are on a 300–850 scale (similar to FICO credit scores).
    """

    SCORE_MIN = 300
    SCORE_MAX = 850

    CREDIT_RATING_MAP = [
        (800, "AAA"), (770, "AA+"), (740, "AA"), (720, "AA-"),
        (700, "A+"),  (680, "A"),   (660, "A-"),
        (640, "BBB+"), (620, "BBB"), (600, "BBB-"),
        (570, "BB+"), (540, "BB"),  (510, "BB-"),
        (480, "B+"),  (450, "B"),   (420, "B-"),
        (0,   "CCC"),
    ]

    def __init__(self, include_clinical: bool = True):
        self.include_clinical = include_clinical
        self.weights = self._get_weights()
        self.scaler = StandardScaler()

    def _get_weights(self) -> Dict[str, Dict]:
        """
        Scoring weights for each metric.
        Higher is better for each positive metric.
        """
        weights = {
            # Financial (positive = higher is better)
            "operating_margin": {"weight": 15, "direction": 1, "ideal": 0.05, "penalty": -0.02},
            "dscr": {"weight": 20, "direction": 1, "ideal": 3.0, "floor": 1.0},
            "days_cash_on_hand": {"weight": 15, "direction": 1, "ideal": 150, "floor": 30},
            "debt_to_capitalization": {"weight": 10, "direction": -1, "ideal": 0.30, "ceiling": 0.80},
            "revenue_growth_yoy": {"weight": 10, "direction": 1, "ideal": 0.05, "floor": -0.10},
            # Clinical Quality
            "readmission_rate_30d": {"weight": 10, "direction": -1, "ideal": 0.12, "ceiling": 0.25},
            "hcahps_star": {"weight": 8, "direction": 1, "ideal": 5, "floor": 1},
            "case_mix_index": {"weight": 5, "direction": 1, "ideal": 1.8, "floor": 0.8},
            "cmi_trend_yoy": {"weight": 4, "direction": 1, "ideal": 0.02, "floor": -0.10},
            "ed_boarding_hours": {"weight": 3, "direction": -1, "ideal": 2.0, "ceiling": 12.0},
        }
        return weights

    def score_hospital(self, hospital: pd.Series) -> Dict:
        """Compute credit score for a single hospital."""
        score_components = {}
        raw_score = 0

        for metric, params in self.weights.items():
            if metric not in hospital.index:
                continue
            if not self.include_clinical and metric in CLINICAL_QUALITY_FEATURES:
                continue

            value = hospital[metric]
            weight = params["weight"]
            direction = params["direction"]
            ideal = params["ideal"]

            # Normalize: 0 = worst, 1 = ideal
            floor = params.get("floor", None)
            ceiling = params.get("ceiling", None)

            if direction == 1:  # Higher = better
                actual_floor = floor if floor else ideal * 0.5
                norm = np.clip((value - actual_floor) / (ideal - actual_floor + 1e-9), 0, 1)
            else:  # Lower = better
                actual_ceiling = ceiling if ceiling else ideal * 2
                norm = np.clip((actual_ceiling - value) / (actual_ceiling - ideal + 1e-9), 0, 1)

            component_score = norm * weight
            score_components[metric] = round(component_score, 2)
            raw_score += component_score

        # Scale to 300-850
        max_possible = sum(p["weight"] for m, p in self.weights.items()
                          if m in hospital.index and
                          (self.include_clinical or m not in CLINICAL_QUALITY_FEATURES))
        scaled_score = self.SCORE_MIN + (raw_score / max_possible) * (self.SCORE_MAX - self.SCORE_MIN)
        scaled_score = int(np.clip(scaled_score, self.SCORE_MIN, self.SCORE_MAX))

        # Implied credit rating
        rating = "CCC"
        for threshold, rat in self.CREDIT_RATING_MAP:
            if scaled_score >= threshold:
                rating = rat
                break

        return {
            "credit_score": scaled_score,
            "implied_rating": rating,
            "components": score_components,
        }

    def score_portfolio(self, hospitals_df: pd.DataFrame) -> pd.DataFrame:
        """Score an entire hospital portfolio."""
        results = []
        for _, row in hospitals_df.iterrows():
            result = self.score_hospital(row)
            results.append({
                "hospital_id": row.get("hospital_id", _),
                "credit_score": result["credit_score"],
                "implied_rating": result["implied_rating"],
            })
        return pd.DataFrame(results)

    @staticmethod
    def rating_to_spread_bps(rating: str) -> float:
        """Convert implied credit rating to approximate bond spread (basis points)."""
        RATING_SPREADS = {
            "AAA": 20, "AA+": 30, "AA": 40, "AA-": 50,
            "A+": 65, "A": 80, "A-": 100,
            "BBB+": 130, "BBB": 160, "BBB-": 200,
            "BB+": 280, "BB": 350, "BB-": 430,
            "B+": 550, "B": 700, "B-": 900,
            "CCC": 1200,
        }
        return RATING_SPREADS.get(rating, 500)


# ──────────────────────────────────────────────────────
# PROBABILITY OF DEFAULT MODEL
# ──────────────────────────────────────────────────────

class HospitalPDModel:
    """
    Predicts 1-year and 5-year Probability of Default for hospitals.
    Compares traditional (financial only) vs enhanced (financial + clinical) models.
    Targets: Gini Coefficient > 0.50, KS Statistic > 0.30
    """

    def __init__(self):
        self.traditional_model = GradientBoostingClassifier(
            n_estimators=200, learning_rate=0.05, max_depth=4,
            subsample=0.8, random_state=42
        )
        self.enhanced_model = GradientBoostingClassifier(
            n_estimators=200, learning_rate=0.05, max_depth=4,
            subsample=0.8, random_state=42
        )
        self.scaler_trad = StandardScaler()
        self.scaler_enh = StandardScaler()

    def fit(self, hospitals_df: pd.DataFrame, target_col: str = "default_within_5yr"):
        """Fit both traditional and enhanced PD models."""
        y = hospitals_df[target_col].values

        # Traditional: financial only
        fin_cols = [c for c in FINANCIAL_FEATURES if c in hospitals_df.columns]
        X_trad = self.scaler_trad.fit_transform(hospitals_df[fin_cols].fillna(0))
        self.traditional_model.fit(X_trad, y)
        self.financial_features_ = fin_cols

        # Enhanced: financial + clinical
        all_cols = [c for c in FINANCIAL_FEATURES + CLINICAL_QUALITY_FEATURES
                    if c in hospitals_df.columns]
        X_enh = self.scaler_enh.fit_transform(hospitals_df[all_cols].fillna(0))
        self.enhanced_model.fit(X_enh, y)
        self.all_features_ = all_cols

        print("Fitted traditional and enhanced PD models.")
        return self

    def predict_pd(self, hospitals_df: pd.DataFrame) -> pd.DataFrame:
        """Predict PD for both models and compute implied credit metrics."""
        # Traditional
        fin_cols = [c for c in self.financial_features_ if c in hospitals_df.columns]
        X_trad = self.scaler_trad.transform(hospitals_df[fin_cols].fillna(0))
        pd_trad = self.traditional_model.predict_proba(X_trad)[:, 1]

        # Enhanced
        all_cols = [c for c in self.all_features_ if c in hospitals_df.columns]
        X_enh = self.scaler_enh.transform(hospitals_df[all_cols].fillna(0))
        pd_enh = self.enhanced_model.predict_proba(X_enh)[:, 1]

        scorecard = HospitalCreditScorecard(include_clinical=True)
        results = []
        for i, (_, row) in enumerate(hospitals_df.iterrows()):
            score_info = scorecard.score_hospital(row)
            pd_delta = pd_enh[i] - pd_trad[i]
            spread_trad = HospitalCreditScorecard.rating_to_spread_bps(
                self._pd_to_rating(pd_trad[i])
            )
            spread_enh = HospitalCreditScorecard.rating_to_spread_bps(
                self._pd_to_rating(pd_enh[i])
            )
            results.append({
                "hospital_id": row.get("hospital_id", i),
                "pd_traditional": round(pd_trad[i], 4),
                "pd_enhanced": round(pd_enh[i], 4),
                "pd_delta": round(pd_delta, 4),
                "rating_traditional": self._pd_to_rating(pd_trad[i]),
                "rating_enhanced": self._pd_to_rating(pd_enh[i]),
                "spread_trad_bps": spread_trad,
                "spread_enh_bps": spread_enh,
                "spread_delta_bps": spread_enh - spread_trad,
                "credit_score": score_info["credit_score"],
            })
        return pd.DataFrame(results)

    @staticmethod
    def _pd_to_rating(pd_value: float) -> str:
        """Convert PD to implied credit rating."""
        if pd_value < 0.001: return "AAA"
        elif pd_value < 0.002: return "AA"
        elif pd_value < 0.004: return "A"
        elif pd_value < 0.008: return "BBB+"
        elif pd_value < 0.012: return "BBB"
        elif pd_value < 0.020: return "BBB-"
        elif pd_value < 0.035: return "BB+"
        elif pd_value < 0.060: return "BB"
        elif pd_value < 0.100: return "B+"
        elif pd_value < 0.150: return "B"
        else: return "CCC"

    def evaluate_models(
        self, hospitals_df: pd.DataFrame, target_col: str = "default_within_5yr"
    ) -> Dict[str, Dict]:
        """Compare traditional vs enhanced model performance."""
        y_true = hospitals_df[target_col].values
        preds = self.predict_pd(hospitals_df)

        metrics = {}
        for model_name, pd_col in [("traditional", "pd_traditional"), ("enhanced", "pd_enhanced")]:
            y_prob = preds[pd_col].values
            auroc = roc_auc_score(y_true, y_prob)
            gini = 2 * auroc - 1

            # KS Statistic
            from scipy.stats import ks_2samp
            default_scores = y_prob[y_true == 1]
            non_default_scores = y_prob[y_true == 0]
            ks_stat, _ = ks_2samp(default_scores, non_default_scores) if (len(default_scores) > 0 and len(non_default_scores) > 0) else (0, 1)

            metrics[model_name] = {"auroc": auroc, "gini": gini, "ks_stat": ks_stat}
            print(f"\n[{model_name.title()}] AUROC: {auroc:.4f} | Gini: {gini:.4f} | KS: {ks_stat:.4f}")
            print(f"  Gini target > 0.50: {'✓ PASS' if gini > 0.50 else '✗ FAIL'}")
            print(f"  KS target > 0.30:   {'✓ PASS' if ks_stat > 0.30 else '✗ FAIL'}")

        improvement = {
            "auroc_lift": metrics["enhanced"]["auroc"] - metrics["traditional"]["auroc"],
            "gini_lift": metrics["enhanced"]["gini"] - metrics["traditional"]["gini"],
        }
        print(f"\n[Improvement] AUROC lift: +{improvement['auroc_lift']:.4f} | Gini lift: +{improvement['gini_lift']:.4f}")
        return {**metrics, "improvement": improvement}


# ──────────────────────────────────────────────────────
# EARLY WARNING SYSTEM
# ──────────────────────────────────────────────────────

class HospitalEarlyWarningSystem:
    """
    Detects clinical quality deterioration 6-12 months BEFORE
    financial metrics worsen. Uses leading clinical indicators.
    """

    ALERT_THRESHOLDS = {
        "readmission_rate_30d": {"delta": 0.02, "direction": "increase"},  # +2pp QoQ
        "cmi_trend_yoy": {"consecutive_quarters": 3, "direction": "decline"},
        "hcahps_star": {"delta": -1, "direction": "decrease"},
        "ed_boarding_hours": {"delta": 1.5, "direction": "increase"},
        "operating_margin": {"threshold": 0.0, "direction": "below"},
    }

    def detect_alerts(
        self,
        hospital_id: str,
        quarterly_data: pd.DataFrame,
    ) -> List[Dict]:
        """Detect early warning alerts from quarterly hospital data."""
        alerts = []

        if len(quarterly_data) < 2:
            return alerts

        latest = quarterly_data.iloc[-1]
        previous = quarterly_data.iloc[-2]

        # Readmission rate spike
        if "readmission_rate_30d" in latest:
            delta = latest["readmission_rate_30d"] - previous["readmission_rate_30d"]
            if delta > 0.02:
                alerts.append({
                    "hospital_id": hospital_id,
                    "alert_type": "READMISSION_RATE_SPIKE",
                    "severity": "HIGH" if delta > 0.04 else "MEDIUM",
                    "metric": "30-day readmission rate",
                    "value": f"{latest['readmission_rate_30d']:.1%}",
                    "change": f"+{delta:.1%} vs prior quarter",
                    "financial_impact": "Potential CMS penalty: $50K–$500K",
                    "lead_time_estimate": "6-9 months before financial impact",
                })

        # CMI decline trend
        if "case_mix_index" in latest and len(quarterly_data) >= 3:
            cmi_trend = [quarterly_data.iloc[i]["case_mix_index"] for i in [-3, -2, -1]]
            if all(cmi_trend[i] > cmi_trend[i+1] for i in range(len(cmi_trend)-1)):
                alerts.append({
                    "hospital_id": hospital_id,
                    "alert_type": "CMI_DECLINING_TREND",
                    "severity": "HIGH",
                    "metric": "Case Mix Index",
                    "value": f"{latest['case_mix_index']:.3f}",
                    "change": f"{cmi_trend[-1] - cmi_trend[0]:+.3f} over 3 quarters",
                    "financial_impact": "Revenue per case reduction, potential coding issue",
                    "lead_time_estimate": "9-12 months before financial impact",
                })

        # HCAHPS star drop
        if "hcahps_star" in latest:
            star_drop = latest["hcahps_star"] - previous["hcahps_star"]
            if star_drop <= -1:
                alerts.append({
                    "hospital_id": hospital_id,
                    "alert_type": "HCAHPS_STAR_DROP",
                    "severity": "MEDIUM",
                    "metric": "HCAHPS Patient Satisfaction",
                    "value": f"{int(latest['hcahps_star'])} stars",
                    "change": f"{int(star_drop)} star(s) vs prior quarter",
                    "financial_impact": "VBP penalty exposure: 1-2% of Medicare payments",
                    "lead_time_estimate": "6-12 months before financial impact",
                })

        return alerts


if __name__ == "__main__":
    from src.data_pipeline.synthetic_data import generate_hospital_financials

    print("=== Testing Hospital Credit Risk Module ===")
    hospitals = generate_hospital_financials(n_hospitals=200, seed=42)

    # Score portfolio
    scorecard = HospitalCreditScorecard(include_clinical=True)
    scores = scorecard.score_portfolio(hospitals.head(5))
    print("\nSample Credit Scores:")
    print(scores.to_string())

    # PD model
    pd_model = HospitalPDModel()
    pd_model.fit(hospitals)
    preds = pd_model.predict_pd(hospitals.head(10))
    print("\nSample PD Predictions:")
    print(preds[["hospital_id", "pd_traditional", "pd_enhanced", "pd_delta",
                  "rating_traditional", "rating_enhanced", "spread_delta_bps"]].to_string())

    metrics = pd_model.evaluate_models(hospitals)
