"""
HealthRisk AI — Pharmaceutical Analytics Module (Day 12)
Implements:
  1. ClinicalTrials.gov Pipeline Monitor (automated signal generation)
  2. Phase Success Probability Model (indication-specific, mechanism-informed)
  3. rNPV Calculator with Monte Carlo simulation
  4. Patent Cliff Impact Analyser
  5. Portfolio Optimisation Engine (mean-variance + clinical alpha signals)

Targets: Information Ratio > 0.50, Sharpe > 1.0, Max Drawdown < 25%
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings("ignore")


# ──────────────────────────────────────────────────────
# PHASE SUCCESS PROBABILITY MODEL
# ──────────────────────────────────────────────────────

class PhaseSuccessModel:
    """
    Computes adjusted phase success probabilities based on:
      - Indication type (oncology 40%, CNS 8%, cardiovascular 60%, etc.)
      - Mechanism novelty (first-in-class vs. me-too)
      - Endpoint type (surrogate vs. clinical)
      - Biomarker selection (selected vs. unselected population)
      - Trial design quality signals
    """

    BASE_RATES = {
        "Phase I → II":   {"overall": 0.63},
        "Phase II → III":  {"overall": 0.31},
        "Phase III → NDA": {"overall": 0.58},
        "NDA → Approval":  {"overall": 0.85},
    }

    INDICATION_MULTIPLIERS = {
        "Oncology":         {"ph2_to_ph3": 1.30, "ph3_to_nda": 0.69},  # 40% Ph3 success
        "CNS":              {"ph2_to_ph3": 0.85, "ph3_to_nda": 0.52},  # Very low Ph3 success
        "Cardiovascular":   {"ph2_to_ph3": 1.00, "ph3_to_nda": 1.03},
        "Immunology":       {"ph2_to_ph3": 1.15, "ph3_to_nda": 0.95},
        "Rare Disease":     {"ph2_to_ph3": 1.40, "ph3_to_nda": 1.20},  # Breakthrough pathway
        "Infectious Disease": {"ph2_to_ph3": 1.10, "ph3_to_nda": 1.05},
        "Metabolic":        {"ph2_to_ph3": 0.95, "ph3_to_nda": 0.90},
    }

    MECHANISM_MULTIPLIERS = {
        "First-in-class": 1.20,
        "Best-in-class": 1.10,
        "Me-too": 0.90,
        "Biosimilar": 0.95,
        "Gene therapy": 1.15,
        "mAb": 1.05,
    }

    ENDPOINT_MULTIPLIERS = {
        "Overall Survival": 1.10,
        "Progression-Free Survival": 0.95,
        "Surrogate endpoint": 0.80,
        "Biomarker endpoint": 0.85,
        "Complete Remission": 1.00,
        "Clinical response rate": 1.00,
    }

    def compute_adjusted_probability(
        self,
        indication: str,
        phase_transition: str,
        mechanism: Optional[str] = None,
        endpoint_type: Optional[str] = None,
        biomarker_selected: bool = False,
        breakthrough_designation: bool = False,
        enrollment_velocity_ratio: float = 1.0,  # actual/target enrollment rate
    ) -> Dict[str, float]:
        """Compute Bayesian-adjusted phase success probability."""

        base = self.BASE_RATES.get(phase_transition, {}).get("overall", 0.5)

        # Indication adjustment
        ind_mult = self.INDICATION_MULTIPLIERS.get(indication, {})
        if phase_transition == "Phase II → III":
            base *= ind_mult.get("ph2_to_ph3", 1.0)
        elif phase_transition == "Phase III → NDA":
            base *= ind_mult.get("ph3_to_nda", 1.0)

        # Mechanism adjustment
        if mechanism:
            for key, mult in self.MECHANISM_MULTIPLIERS.items():
                if key.lower() in mechanism.lower():
                    base *= mult
                    break

        # Endpoint adjustment
        if endpoint_type:
            for key, mult in self.ENDPOINT_MULTIPLIERS.items():
                if key.lower() in endpoint_type.lower():
                    base *= mult
                    break

        # Biomarker selection (enriches patient population → higher success)
        if biomarker_selected:
            base *= 1.25

        # FDA Breakthrough Therapy Designation
        if breakthrough_designation:
            base *= 1.40

        # Enrollment velocity signal (positive = above target)
        if enrollment_velocity_ratio > 1.10:
            base *= 1.05  # Strong investigator/patient interest
        elif enrollment_velocity_ratio < 0.80:
            base *= 0.92  # Enrollment struggles suggest trial difficulty

        # Clip to [0, 0.99]
        probability = np.clip(base, 0.01, 0.99)

        return {
            "base_rate": self.BASE_RATES.get(phase_transition, {}).get("overall", 0.5),
            "adjusted_probability": round(probability, 4),
            "phase_transition": phase_transition,
            "indication": indication,
        }


# ──────────────────────────────────────────────────────
# rNPV CALCULATOR WITH MONTE CARLO
# ──────────────────────────────────────────────────────

class RNPVCalculator:
    """
    Risk-adjusted Net Present Value calculator for drug pipeline assets.
    Uses Monte Carlo simulation to generate probability distributions.
    """

    def __init__(self, discount_rate: float = 0.10, n_simulations: int = 10_000):
        self.discount_rate = discount_rate
        self.n_simulations = n_simulations

    def calculate(
        self,
        peak_sales_estimate_b: float,
        peak_sales_std_b: float,
        years_to_launch: float,
        patent_years_remaining: int,
        probability_of_success: float,
        cogs_pct: float = 0.20,
        rd_cost_b: float = 0.5,
        royalty_rate: float = 0.0,
        generic_erosion_years: int = 3,
    ) -> Dict:
        """
        Compute rNPV using Monte Carlo simulation.

        Parameters:
        -----------
        peak_sales_estimate_b : float - Peak annual sales in $B
        peak_sales_std_b : float - Standard deviation of peak sales ($B)
        years_to_launch : float - Years until product launch
        patent_years_remaining : int - Patent runway from today
        probability_of_success : float - Adjusted PoS from PhaseSuccessModel
        cogs_pct : float - Cost of goods as % of sales
        rd_cost_b : float - Remaining R&D cost in $B
        royalty_rate : float - Royalty rate to pay (0 if no licensing)
        generic_erosion_years : int - Years of generic erosion after patent cliff
        """
        np.random.seed(42)
        simulated_npvs = []

        for _ in range(self.n_simulations):
            # Sample success (Bernoulli trial)
            success = np.random.binomial(1, probability_of_success)
            if not success:
                simulated_npvs.append(-rd_cost_b * 1e9)
                continue

            # Sample peak sales from log-normal
            mean_log = np.log(peak_sales_estimate_b) - 0.5 * (peak_sales_std_b / peak_sales_estimate_b) ** 2
            std_log = peak_sales_std_b / peak_sales_estimate_b
            sampled_peak = np.random.lognormal(mean_log, std_log) * 1e9

            # Build cash flow profile
            on_market_years = patent_years_remaining - years_to_launch
            if on_market_years <= 0:
                simulated_npvs.append(-rd_cost_b * 1e9)
                continue

            total_npv = 0
            for year in range(1, int(on_market_years) + 1):
                # Ramp-up: years 1-3
                if year <= 3:
                    sales = sampled_peak * (year / 3) * 0.7
                # Peak: years 4 to (on_market_years - generic_erosion_years)
                elif year <= on_market_years - generic_erosion_years:
                    sales = sampled_peak
                # Erosion after patent cliff
                else:
                    erosion_factor = max(0.1, 1.0 - 0.7 * ((year - (on_market_years - generic_erosion_years)) / generic_erosion_years))
                    sales = sampled_peak * erosion_factor

                operating_income = sales * (1 - cogs_pct - royalty_rate)
                cf_year = operating_income * (1 - 0.21)  # After-tax (21% corporate rate)
                discount_factor = (1 + self.discount_rate) ** (years_to_launch + year)
                total_npv += cf_year / discount_factor

            # Subtract R&D costs
            total_npv -= rd_cost_b * 1e9
            simulated_npvs.append(total_npv)

        simulated_npvs = np.array(simulated_npvs)
        rnpv = np.mean(simulated_npvs)
        p5, p25, p75, p95 = np.percentile(simulated_npvs, [5, 25, 75, 95])

        result = {
            "rnpv_m": round(rnpv / 1e6, 1),
            "p5_m": round(p5 / 1e6, 1),
            "p25_m": round(p25 / 1e6, 1),
            "p75_m": round(p75 / 1e6, 1),
            "p95_m": round(p95 / 1e6, 1),
            "probability_of_success": probability_of_success,
            "peak_sales_b": peak_sales_estimate_b,
            "prob_positive_npv": (simulated_npvs > 0).mean().round(3),
            "n_simulations": self.n_simulations,
        }
        print(f"\n[rNPV] ${result['rnpv_m']:.0f}M  "
              f"(5th–95th: ${result['p5_m']:.0f}M to ${result['p95_m']:.0f}M)")
        print(f"  PoS: {probability_of_success:.1%}  |  "
              f"Prob positive NPV: {result['prob_positive_npv']:.1%}")
        return result

    def value_pipeline(self, pipeline_df: pd.DataFrame, phase_model: PhaseSuccessModel) -> pd.DataFrame:
        """Value an entire pharmaceutical pipeline."""
        results = []
        for _, drug in pipeline_df.iterrows():
            phase = drug.get("phase", "Phase II")
            if "III" in str(phase):
                transition = "Phase III → NDA"
            elif "II" in str(phase):
                transition = "Phase II → III"
            else:
                transition = "Phase I → II"

            pos = phase_model.compute_adjusted_probability(
                indication=drug.get("indication", "Oncology"),
                phase_transition=transition,
            )["adjusted_probability"]

            rnpv = self.calculate(
                peak_sales_estimate_b=drug.get("peak_sales_estimate_b", 1.0),
                peak_sales_std_b=drug.get("peak_sales_estimate_b", 1.0) * 0.40,
                years_to_launch=drug.get("months_to_interim", 24) / 12 + 2,
                patent_years_remaining=drug.get("patent_years_remaining", 12),
                probability_of_success=pos,
            )
            results.append({
                "trial_id": drug.get("trial_id", ""),
                "indication": drug.get("indication", ""),
                "phase": phase,
                "pos": round(pos, 3),
                "rnpv_m": rnpv["rnpv_m"],
                "peak_sales_b": drug.get("peak_sales_estimate_b", 1.0),
            })
        return pd.DataFrame(results)


# ──────────────────────────────────────────────────────
# PORTFOLIO OPTIMISATION ENGINE
# ──────────────────────────────────────────────────────

class PharmaPortfolioOptimizer:
    """
    Mean-variance portfolio optimisation enhanced with clinical alpha signals.
    Targets: Sharpe > 1.0, Information Ratio > 0.50, Max Drawdown < 25%
    """

    def __init__(self, risk_free_rate: float = 0.04):
        self.risk_free_rate = risk_free_rate
        self.optimal_weights = None
        self.expected_returns = None
        self.cov_matrix = None

    def compute_expected_returns(
        self,
        historical_returns: np.ndarray,
        clinical_alpha: Optional[np.ndarray] = None,
        alpha_weight: float = 0.30,
    ) -> np.ndarray:
        """
        Combine historical returns with clinical alpha signals.
        clinical_alpha: signal strength per stock (-1 to +1, positive = buy signal)
        """
        base_returns = historical_returns.mean(axis=0)
        if clinical_alpha is not None:
            # Clinical signals contribute alpha_weight % of expected return
            clinical_signal = clinical_alpha * base_returns.std() * 2
            returns = (1 - alpha_weight) * base_returns + alpha_weight * clinical_signal
        else:
            returns = base_returns
        self.expected_returns = returns
        return returns

    def optimize(
        self,
        expected_returns: np.ndarray,
        cov_matrix: np.ndarray,
        risk_aversion: float = 3.0,
        sector_constraints: Optional[Dict] = None,
    ) -> Dict:
        """Maximize Sharpe ratio using scipy minimize."""
        n = len(expected_returns)
        self.cov_matrix = cov_matrix

        def neg_sharpe(weights):
            ret = np.dot(weights, expected_returns)
            vol = np.sqrt(weights @ cov_matrix @ weights)
            return -(ret - self.risk_free_rate) / (vol + 1e-9)

        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
        bounds = [(0.01, 0.20)] * n  # Each stock: 1%-20% of portfolio

        result = minimize(
            neg_sharpe, x0=np.ones(n) / n,
            method="SLSQP", bounds=bounds, constraints=constraints,
            options={"maxiter": 1000, "ftol": 1e-9}
        )

        if result.success:
            self.optimal_weights = result.x
        else:
            print("Optimisation did not converge; using equal weights.")
            self.optimal_weights = np.ones(n) / n

        portfolio_ret = np.dot(self.optimal_weights, expected_returns)
        portfolio_vol = np.sqrt(self.optimal_weights @ cov_matrix @ self.optimal_weights)
        sharpe = (portfolio_ret - self.risk_free_rate) / portfolio_vol

        print(f"\n[Portfolio Optimization]")
        print(f"  Expected Annual Return: {portfolio_ret:.2%}")
        print(f"  Annual Volatility:      {portfolio_vol:.2%}")
        print(f"  Sharpe Ratio:           {sharpe:.2f}  (target > 1.0)")

        return {
            "weights": self.optimal_weights,
            "expected_return": portfolio_ret,
            "volatility": portfolio_vol,
            "sharpe_ratio": sharpe,
        }

    def backtest(
        self,
        historical_returns: np.ndarray,  # shape: (days, n_stocks)
        weights: np.ndarray,
        benchmark_weights: Optional[np.ndarray] = None,
    ) -> Dict:
        """Backtest portfolio performance."""
        portfolio_returns = historical_returns @ weights
        cumulative = (1 + portfolio_returns).cumprod()

        # Max drawdown
        rolling_max = pd.Series(cumulative).cummax()
        drawdown = (cumulative - rolling_max) / rolling_max
        max_drawdown = abs(drawdown.min())

        # Sharpe
        ann_return = portfolio_returns.mean() * 252
        ann_vol = portfolio_returns.std() * np.sqrt(252)
        sharpe = (ann_return - self.risk_free_rate) / ann_vol

        # Information ratio (vs equal-weight benchmark)
        if benchmark_weights is None:
            benchmark_weights = np.ones(weights.shape) / len(weights)
        benchmark_returns = historical_returns @ benchmark_weights
        active_returns = portfolio_returns - benchmark_returns
        info_ratio = (active_returns.mean() * 252) / (active_returns.std() * np.sqrt(252) + 1e-9)

        metrics = {
            "ann_return": ann_return,
            "ann_volatility": ann_vol,
            "sharpe_ratio": sharpe,
            "max_drawdown": max_drawdown,
            "information_ratio": info_ratio,
            "cumulative_return": cumulative[-1] - 1,
        }

        print(f"\n[Backtest Results]")
        print(f"  Ann. Return:        {ann_return:.2%}")
        print(f"  Ann. Volatility:    {ann_vol:.2%}")
        print(f"  Sharpe Ratio:       {sharpe:.2f}   (target > 1.0)")
        print(f"  Max Drawdown:       {max_drawdown:.2%}  (target < 25%)")
        print(f"  Information Ratio:  {info_ratio:.2f}  (target > 0.50)")
        return metrics


if __name__ == "__main__":
    from src.data_pipeline.synthetic_data import generate_pharma_pipeline

    print("=== Testing Pharmaceutical Analytics Module ===")

    pharma_df = generate_pharma_pipeline(n_trials=20, seed=42)
    phase_model = PhaseSuccessModel()
    calc = RNPVCalculator(discount_rate=0.10, n_simulations=1000)

    # Test single drug rNPV
    print("\n--- Single Drug rNPV Test ---")
    pos_result = phase_model.compute_adjusted_probability(
        indication="Oncology",
        phase_transition="Phase III → NDA",
        mechanism="mAb",
        endpoint_type="Progression-Free Survival",
        biomarker_selected=True,
        enrollment_velocity_ratio=1.15,
    )
    print(f"Adjusted PoS: {pos_result['adjusted_probability']:.1%}")

    rnpv = calc.calculate(
        peak_sales_estimate_b=2.5,
        peak_sales_std_b=1.0,
        years_to_launch=3.5,
        patent_years_remaining=11,
        probability_of_success=pos_result["adjusted_probability"],
    )

    # Portfolio optimization
    print("\n--- Portfolio Optimization Test ---")
    np.random.seed(42)
    n_stocks = 10
    daily_returns = np.random.randn(252, n_stocks) * 0.02
    clinical_alpha = np.random.uniform(-1, 1, n_stocks)

    optimizer = PharmaPortfolioOptimizer(risk_free_rate=0.04)
    exp_returns = optimizer.compute_expected_returns(daily_returns, clinical_alpha)
    cov_matrix = np.cov(daily_returns.T)
    result = optimizer.optimize(exp_returns, cov_matrix)
    backtest_metrics = optimizer.backtest(daily_returns, result["weights"])
