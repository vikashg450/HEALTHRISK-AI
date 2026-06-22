"""
HealthRisk AI — Simulation Engine: HealthRisk Lab (Day 13)
Implements the gamified portfolio management simulation:
  - Quarterly simulation cycle
  - 10+ distinct financial shock scenarios
  - AI opponent using HealthRisk AI models
  - 1000-point scoring system
  - Historical scenario replay capability
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from enum import Enum
import json
import os


# ──────────────────────────────────────────────────────
# ENUMS & DATA CLASSES
# ──────────────────────────────────────────────────────

class AssetType(Enum):
    HOSPITAL_BOND = "Hospital Bond"
    PHARMA_EQUITY = "Pharma Equity"
    INSURANCE_BOOK = "Insurance Book"
    HEALTH_REIT = "Healthcare REIT"
    CASH = "Cash"


class ScenarioSeverity(Enum):
    LOW = "Low"
    MEDIUM = "Medium"
    HIGH = "High"
    EXTREME = "Extreme"


@dataclass
class PortfolioAsset:
    asset_id: str
    asset_type: AssetType
    name: str
    value: float          # Current market value ($M)
    weight: float         # Portfolio weight (0-1)
    yield_rate: float     # Annual yield/return
    risk_score: float     # Internal risk score (0-10)
    clinical_signals: Dict = field(default_factory=dict)


@dataclass
class GameState:
    year: int
    quarter: int
    portfolio: List[PortfolioAsset]
    cash: float
    player_score: int
    ai_score: int
    events_log: List[str] = field(default_factory=list)
    scenario_history: List[Dict] = field(default_factory=list)

    @property
    def portfolio_value(self) -> float:
        return sum(a.value for a in self.portfolio) + self.cash

    @property
    def quarter_label(self) -> str:
        return f"{self.year} Q{self.quarter}"


# ──────────────────────────────────────────────────────
# SCENARIO LIBRARY (10+ Types)
# ──────────────────────────────────────────────────────

class ScenarioLibrary:
    """
    Library of 10+ distinct financial shock scenarios for HealthRisk Lab.
    Each scenario has: description, asset impacts, clinical signals, optimal_action.
    """

    SCENARIOS = [
        {
            "id": "PANDEMIC_OUTBREAK",
            "name": "Novel Pathogen Outbreak (R₀ = 2.8)",
            "severity": "EXTREME",
            "description": (
                "A novel respiratory pathogen is detected in 12 countries. "
                "R₀ estimated at 2.8. WHO declares Public Health Emergency. "
                "Clinical AI models detect 340% spike in influenza-like illness in sentinel hospitals."
            ),
            "impacts": {
                AssetType.HOSPITAL_BOND: -0.08,     # Revenue collapse: elective cancellations
                AssetType.PHARMA_EQUITY: +0.25,     # Vaccine/antiviral pipeline boost
                AssetType.INSURANCE_BOOK: -0.15,    # Claims surge
                AssetType.HEALTH_REIT: -0.05,       # Operational disruption
            },
            "clinical_signals": {
                "ili_sentinel_spike_pct": 340,
                "hospital_occupancy_pct": 94,
                "icu_capacity_pct": 87,
                "r0_estimate": 2.8,
            },
            "optimal_action": "Underweight hospital bonds; overweight vaccine/antiviral pharma; increase insurance reserves by 20%",
            "lead_time_quarters": 2,  # AI can detect this 2 quarters early
        },
        {
            "id": "DRUG_TRIAL_FAILURE",
            "name": "Phase III Oncology Trial Failure",
            "severity": "HIGH",
            "description": (
                "MegaPharma Corp announces failure of MEGA-001, its lead Phase III "
                "NSCLC candidate, missing the primary PFS endpoint (HR: 0.89, p=0.32). "
                "Stock expected to drop 45-60% at open."
            ),
            "impacts": {
                AssetType.PHARMA_EQUITY: -0.50,
                AssetType.HOSPITAL_BOND: 0.00,
                AssetType.INSURANCE_BOOK: 0.00,
                AssetType.HEALTH_REIT: 0.00,
            },
            "clinical_signals": {
                "enrollment_velocity_below_target": True,
                "interim_futility_crossing": True,
                "competitive_trial_success": True,  # Competitor succeeded first
                "biomarker_readout_negative": True,
            },
            "optimal_action": "Exit MegaPharma position pre-readout when enrollment velocity slows; rotate to competitor",
            "lead_time_quarters": 3,
        },
        {
            "id": "FDA_BREAKTHROUGH_APPROVAL",
            "name": "Surprise FDA Breakthrough Approval",
            "severity": "HIGH",
            "description": (
                "FDA grants accelerated approval to NovaBio's gene therapy ABC-001 "
                "for rare muscular dystrophy, 6 months ahead of schedule. "
                "HealthRisk AI detected strong compassionate use demand 2 quarters ago."
            ),
            "impacts": {
                AssetType.PHARMA_EQUITY: +0.45,
                AssetType.INSURANCE_BOOK: -0.03,   # New expensive therapy
                AssetType.HOSPITAL_BOND: +0.01,
                AssetType.HEALTH_REIT: 0.00,
            },
            "clinical_signals": {
                "compassionate_use_requests": 847,
                "breakthrough_therapy_designation": True,
                "fda_advisory_committee_score": 9.2,
            },
            "optimal_action": "Overweight NovaBio pre-approval based on compassionate use demand signals",
            "lead_time_quarters": 2,
        },
        {
            "id": "HOSPITAL_CAPACITY_CRISIS",
            "name": "Regional Hospital Capacity Crisis",
            "severity": "HIGH",
            "description": (
                "Midwest Regional Health System (MRHS) declares capacity crisis: "
                "95% occupancy, nursing strike, 3 consecutive quarters of CMI decline. "
                "Clinical AI flagged deteriorating quality signals 9 months ago."
            ),
            "impacts": {
                AssetType.HOSPITAL_BOND: -0.12,
                AssetType.PHARMA_EQUITY: 0.00,
                AssetType.INSURANCE_BOOK: -0.02,
                AssetType.HEALTH_REIT: -0.04,
            },
            "clinical_signals": {
                "readmission_rate_pct": 19.2,  # Above 15% national avg
                "cmi_trend_3q": -0.08,
                "hcahps_stars": 2,
                "ed_boarding_hours": 8.5,
            },
            "optimal_action": "Exit MRHS bonds 2 quarters before crisis when clinical deterioration signals emerge",
            "lead_time_quarters": 3,
        },
        {
            "id": "MEDICARE_RATE_CUT",
            "name": "CMS Medicare Rate Cut (-3%)",
            "severity": "MEDIUM",
            "description": (
                "CMS finalizes IPPS rule with 3% reduction in inpatient Medicare rates, "
                "affecting 4,500 hospitals. Impact largest on high-Medicare-payer hospitals. "
                "Hospitals with >60% Medicare exposure see operating margin compressed by 50-80bps."
            ),
            "impacts": {
                AssetType.HOSPITAL_BOND: -0.05,
                AssetType.PHARMA_EQUITY: 0.00,
                AssetType.INSURANCE_BOOK: +0.02,   # Government pays less → private insurers gain
                AssetType.HEALTH_REIT: -0.03,
            },
            "clinical_signals": {
                "medicare_payer_concentration": "HIGH",
                "cms_proposal_period": True,
                "hospital_lobbying_intensity": "HIGH",
            },
            "optimal_action": "Rotate from high-Medicare-exposure hospital bonds to commercial-focused systems",
            "lead_time_quarters": 2,
        },
        {
            "id": "PATENT_CLIFF",
            "name": "Blockbuster Drug Patent Cliff",
            "severity": "HIGH",
            "description": (
                "GloboPharma's diabetes blockbuster GLOBO-DM ($4.2B peak sales) loses "
                "exclusivity this quarter. 12 generic filers approved. "
                "Price erosion of 80% expected within 18 months."
            ),
            "impacts": {
                AssetType.PHARMA_EQUITY: -0.25,
                AssetType.INSURANCE_BOOK: +0.08,   # Generics reduce drug costs
                AssetType.HOSPITAL_BOND: 0.00,
                AssetType.HEALTH_REIT: 0.00,
            },
            "clinical_signals": {
                "patent_expiry_date_known": True,
                "generic_filer_count": 12,
                "biosimilar_entrant_announced": False,
            },
            "optimal_action": "Exit GloboPharma 4 quarters before patent expiry; position in generic manufacturer",
            "lead_time_quarters": 4,
        },
        {
            "id": "OPIOID_LITIGATION",
            "name": "Opioid Litigation Settlement Wave",
            "severity": "EXTREME",
            "description": (
                "Federal court approves $18B settlement against OpioidCorp covering "
                "1,200 counties. Legal exposure significantly higher than consensus estimates. "
                "Clinical AI detected rising overdose trends in key geographies 6 quarters ago."
            ),
            "impacts": {
                AssetType.PHARMA_EQUITY: -0.35,
                AssetType.HOSPITAL_BOND: +0.02,   # Opioid treatment revenue increases
                AssetType.INSURANCE_BOOK: -0.04,
                AssetType.HEALTH_REIT: 0.00,
            },
            "clinical_signals": {
                "od_death_rate_per_100k": 28.4,
                "ed_visits_opioid_pct": 8.9,
                "state_ag_actions": 47,
            },
            "optimal_action": "Exit OpioidCorp as litigation signals compound; underweight broad pharma sector",
            "lead_time_quarters": 6,
        },
        {
            "id": "EMERGING_DISEASE",
            "name": "Novel Emerging Disease Alert",
            "severity": "MEDIUM",
            "description": (
                "WHO reports cluster of hemorrhagic fever cases in Central Africa. "
                "R₀ currently 1.1 — borderline epidemic threshold. "
                "Healthcare AI models flag 60% probability of international spread within 6 months."
            ),
            "impacts": {
                AssetType.HOSPITAL_BOND: -0.02,
                AssetType.PHARMA_EQUITY: +0.08,
                AssetType.INSURANCE_BOOK: -0.03,
                AssetType.HEALTH_REIT: 0.00,
            },
            "clinical_signals": {
                "who_alert_level": 2,
                "genomic_novelty_score": 0.72,
                "international_travel_volume": 1.2e6,
                "r0_estimate": 1.1,
            },
            "optimal_action": "Establish small long position in diagnostics/vaccine pure-plays as optionality",
            "lead_time_quarters": 4,
        },
        {
            "id": "HOSPITAL_MA",
            "name": "Major Hospital System M&A",
            "severity": "MEDIUM",
            "description": (
                "MegaHealth acquires RegionalCare Network for $12B (2.8x revenue). "
                "MegaHealth bond spread tightens post-announcement. "
                "Regulatory review expected: 60% probability of FTC approval."
            ),
            "impacts": {
                AssetType.HOSPITAL_BOND: +0.04,   # Credit improvement post-merger
                AssetType.PHARMA_EQUITY: 0.00,
                AssetType.INSURANCE_BOOK: -0.01,  # Reduced competition → higher prices
                AssetType.HEALTH_REIT: +0.02,
            },
            "clinical_signals": {
                "combined_market_share_pct": 38,
                "cmi_combined": 1.75,
                "hcahps_weighted_avg": 3.8,
            },
            "optimal_action": "Buy RegionalCare bonds pre-merger at discount; exit after spread tightening",
            "lead_time_quarters": 1,
        },
        {
            "id": "ESG_DOWNGRADE",
            "name": "ESG Governance Failure & Downgrade",
            "severity": "MEDIUM",
            "description": (
                "PharmaCo receives ESG downgrade from MSCI (A → BBB) following "
                "whistleblower report on clinical trial data manipulation. "
                "ESG fund outflows expected: $2.4B in mandatory selling pressure."
            ),
            "impacts": {
                AssetType.PHARMA_EQUITY: -0.18,
                AssetType.HOSPITAL_BOND: 0.00,
                AssetType.INSURANCE_BOOK: 0.00,
                AssetType.HEALTH_REIT: -0.01,
            },
            "clinical_signals": {
                "adverse_event_underreporting_score": 7.8,
                "fda_warning_letters": 2,
                "whistleblower_claims": 1,
            },
            "optimal_action": "Monitor FDA warning letter history as leading ESG risk signal",
            "lead_time_quarters": 2,
        },
        {
            "id": "VALUE_BASED_CARE_WIN",
            "name": "Value-Based Care Contract Win",
            "severity": "LOW",
            "description": (
                "HealthSystem A wins a major ACO contract covering 120,000 Medicare "
                "lives. Clinical AI predicts $45M annual savings from care management programs. "
                "Operating margin uplift of 60-80bps expected."
            ),
            "impacts": {
                AssetType.HOSPITAL_BOND: +0.04,
                AssetType.PHARMA_EQUITY: 0.00,
                AssetType.INSURANCE_BOOK: +0.02,
                AssetType.HEALTH_REIT: +0.01,
            },
            "clinical_signals": {
                "aco_savings_rate_pct": 4.2,
                "care_management_enrollment_pct": 67,
                "high_risk_member_pct": 8.3,
            },
            "optimal_action": "Overweight HealthSystem A bonds; sector tailwind for value-based care winners",
            "lead_time_quarters": 1,
        },
    ]

    @classmethod
    def get_scenario(cls, scenario_id: str) -> Optional[Dict]:
        for s in cls.SCENARIOS:
            if s["id"] == scenario_id:
                return s
        return None

    @classmethod
    def get_random_scenario(cls, severity_filter: Optional[str] = None) -> Dict:
        candidates = cls.SCENARIOS
        if severity_filter:
            candidates = [s for s in cls.SCENARIOS if s["severity"] == severity_filter]
        return np.random.choice(candidates)


# ──────────────────────────────────────────────────────
# AI OPPONENT
# ──────────────────────────────────────────────────────

class HealthRiskAIOpponent:
    """
    AI opponent that uses HealthRisk AI models to make near-optimal decisions.
    The AI has access to clinical signals and uses them to front-run scenarios.
    """

    def __init__(self, intelligence_level: float = 0.85):
        """
        intelligence_level: 0.0 = random, 1.0 = perfect foresight
        Default 0.85 = very strong but beatable with good play
        """
        self.intelligence = intelligence_level
        self.portfolio_weights = None

    def decide(self, state: GameState, scenario: Dict) -> Dict:
        """AI decision given current game state and upcoming scenario."""
        if np.random.random() > self.intelligence:
            # Occasionally suboptimal (make AI beatable)
            action = self._random_action(state)
            rationale = "AI: Uncertainty in signal quality → conservative position"
        else:
            action = self._optimal_action(state, scenario)
            rationale = f"AI: Detected {scenario['clinical_signals']} → acting {self.intelligence:.0%} confidence"

        return {"action": action, "rationale": rationale}

    def _optimal_action(self, state: GameState, scenario: Dict) -> Dict:
        """Optimal action based on scenario impacts."""
        rebalance = {}
        for asset_type, impact in scenario["impacts"].items():
            if impact < -0.05:
                rebalance[asset_type.value] = "UNDERWEIGHT"
            elif impact > 0.05:
                rebalance[asset_type.value] = "OVERWEIGHT"
            else:
                rebalance[asset_type.value] = "NEUTRAL"
        return {"rebalance": rebalance, "hedge": impact < -0.10}

    def _random_action(self, state: GameState) -> Dict:
        return {
            "rebalance": {at.value: np.random.choice(["UNDERWEIGHT", "NEUTRAL", "OVERWEIGHT"])
                         for at in AssetType if at != AssetType.CASH},
            "hedge": False,
        }


# ──────────────────────────────────────────────────────
# SIMULATION ENGINE
# ──────────────────────────────────────────────────────

class HealthRiskLabEngine:
    """
    Core HealthRisk Lab simulation engine.
    Manages quarterly portfolio management gameplay over a 10-year horizon.
    """

    SCORING = {
        "portfolio_return_vs_benchmark": 300,
        "risk_management_drawdown": 200,
        "scenario_response_speed": 150,
        "esg_score_maintenance": 150,
        "insurance_reserve_accuracy": 100,
        "beat_ai_opponent": 100,
    }

    def __init__(
        self,
        start_year: int = 2010,
        end_year: int = 2020,
        initial_portfolio_value: float = 100_000_000,  # $100M
        intelligence_level: float = 0.85,
        seed: int = 42,
    ):
        np.random.seed(seed)
        self.start_year = start_year
        self.end_year = end_year
        self.ai = HealthRiskAIOpponent(intelligence_level)
        self.state = self._init_game_state(initial_portfolio_value)
        self.scenario_library = ScenarioLibrary()
        self._scenario_queue = []
        self._benchmark_returns = []
        self._player_returns = []
        self._ai_returns = []

    def _init_game_state(self, initial_value: float) -> GameState:
        """Initialize portfolio with diversified health sector assets."""
        assets = [
            PortfolioAsset("HOSP_BOND_1", AssetType.HOSPITAL_BOND,
                          "MidWest Health System 5.2% 2035", 30_000_000, 0.30, 0.052, 5.0),
            PortfolioAsset("HOSP_BOND_2", AssetType.HOSPITAL_BOND,
                          "Academic Medical Center 4.8% 2030", 15_000_000, 0.15, 0.048, 4.0),
            PortfolioAsset("PHARMA_1", AssetType.PHARMA_EQUITY,
                          "MegaPharma Corp (Phase III pipeline)", 20_000_000, 0.20, 0.08, 6.5),
            PortfolioAsset("PHARMA_2", AssetType.PHARMA_EQUITY,
                          "NovaBio (Gene therapy)", 10_000_000, 0.10, 0.15, 8.0),
            PortfolioAsset("INS_BOOK", AssetType.INSURANCE_BOOK,
                          "Medicare Advantage Portfolio", 15_000_000, 0.15, 0.06, 4.5),
            PortfolioAsset("REIT_1", AssetType.HEALTH_REIT,
                          "HealthCare REIT Partners", 5_000_000, 0.05, 0.07, 3.0),
        ]
        return GameState(
            year=self.start_year, quarter=1,
            portfolio=assets,
            cash=5_000_000,
            player_score=0,
            ai_score=0,
        )

    def run_quarter(self, player_decision: Optional[Dict] = None) -> Dict:
        """Execute one simulation quarter."""
        label = self.state.quarter_label

        # Generate scenario for this quarter
        scenario = self._get_next_scenario()

        # Apply scenario impacts to portfolio
        player_outcome = self._apply_scenario(scenario, player_decision, is_player=True)
        ai_decision = self.ai.decide(self.state, scenario)
        ai_outcome = self._apply_scenario(scenario, ai_decision["action"], is_player=False)

        # Apply natural market drift (±5% annually)
        self._apply_market_drift()

        # Score the quarter
        player_points = self._score_quarter(player_outcome, ai_outcome, scenario)
        ai_points = self._score_quarter(ai_outcome, player_outcome, scenario)
        self.state.player_score += player_points
        self.state.ai_score += ai_points

        # Advance quarter
        self._advance_quarter()

        result = {
            "quarter": label,
            "scenario": {
                "id": scenario["id"],
                "name": scenario["name"],
                "severity": scenario["severity"],
                "description": scenario["description"],
                "clinical_signals": scenario["clinical_signals"],
                "optimal_action": scenario["optimal_action"],
            },
            "player": {
                "decision": player_decision,
                "portfolio_return": player_outcome["return"],
                "points_earned": player_points,
                "total_score": self.state.player_score,
                "portfolio_value": self.state.portfolio_value,
            },
            "ai": {
                "decision": ai_decision,
                "portfolio_return": ai_outcome["return"],
                "points_earned": ai_points,
                "total_score": self.state.ai_score,
            },
        }
        self.state.scenario_history.append(result)
        return result

    def _get_next_scenario(self) -> Dict:
        """Select next scenario (mix of random and pre-planned)."""
        if self._scenario_queue:
            return self._scenario_queue.pop(0)
        # Random scenario with seasonal weighting
        severity_weights = {"LOW": 0.25, "MEDIUM": 0.45, "HIGH": 0.25, "EXTREME": 0.05}
        severity = np.random.choice(list(severity_weights.keys()), p=list(severity_weights.values()))
        candidates = [s for s in ScenarioLibrary.SCENARIOS if s["severity"] == severity]
        return candidates[np.random.randint(len(candidates))]

    def _apply_scenario(self, scenario: Dict, decision: Optional[Dict], is_player: bool) -> Dict:
        """Apply scenario impact to portfolio based on decision."""
        total_return = 0
        for asset in self.state.portfolio:
            base_impact = scenario["impacts"].get(asset.asset_type, 0)

            # If player/AI underweights a negative impact asset, reduce loss
            if decision and "rebalance" in decision:
                action = decision["rebalance"].get(asset.asset_type.value, "NEUTRAL")
                if action == "UNDERWEIGHT" and base_impact < 0:
                    base_impact *= 0.50  # Mitigated by 50%
                elif action == "OVERWEIGHT" and base_impact > 0:
                    base_impact *= 1.30  # Amplified gains

            asset.value *= (1 + base_impact)
            total_return += base_impact * asset.weight

        return {"return": round(total_return, 4)}

    def _apply_market_drift(self):
        """Apply natural quarterly market movement."""
        for asset in self.state.portfolio:
            if asset.asset_type == AssetType.PHARMA_EQUITY:
                drift = np.random.normal(0.01, 0.08)  # Higher volatility
            elif asset.asset_type == AssetType.HOSPITAL_BOND:
                drift = np.random.normal(0.012, 0.02)  # Stable
            else:
                drift = np.random.normal(0.01, 0.04)
            asset.value *= (1 + drift)

    def _score_quarter(self, own_outcome: Dict, opponent_outcome: Dict, scenario: Dict) -> int:
        """Score a quarter based on relative performance."""
        points = 0
        own_ret = own_outcome["return"]
        opp_ret = opponent_outcome["return"]

        # Relative performance (max 75 pts/quarter)
        if own_ret > opp_ret:
            points += min(75, int((own_ret - opp_ret) * 5000))

        # Beat benchmark (equal-weight return)
        benchmark_return = np.mean([v for v in scenario["impacts"].values()])
        if own_ret > benchmark_return:
            points += 25

        return max(0, points)

    def _advance_quarter(self):
        """Advance game time by one quarter."""
        self.state.quarter += 1
        if self.state.quarter > 4:
            self.state.quarter = 1
            self.state.year += 1

    def is_game_over(self) -> bool:
        return self.state.year > self.end_year

    def get_final_results(self) -> Dict:
        """Generate end-of-game summary."""
        initial_value = 100_000_000
        final_value = self.state.portfolio_value
        total_return = (final_value - initial_value) / initial_value

        return {
            "player_final_score": self.state.player_score,
            "ai_final_score": self.state.ai_score,
            "player_wins": self.state.player_score > self.state.ai_score,
            "total_portfolio_return": round(total_return, 4),
            "final_portfolio_value_m": round(final_value / 1e6, 2),
            "quarters_played": len(self.state.scenario_history),
            "scenarios_encountered": [s["scenario"]["id"] for s in self.state.scenario_history],
        }

    def replay(self, scenario_id: str) -> Dict:
        """Replay a specific historical scenario for learning."""
        scenario = ScenarioLibrary.get_scenario(scenario_id)
        if not scenario:
            return {"error": f"Scenario {scenario_id} not found"}
        return {
            "scenario": scenario,
            "replay": True,
            "hint": scenario["optimal_action"],
            "clinical_signals_explained": scenario["clinical_signals"],
        }


if __name__ == "__main__":
    print("=== Running HealthRisk Lab Demo (3 quarters) ===\n")
    engine = HealthRiskLabEngine(start_year=2020, end_year=2025, seed=42)

    for q in range(3):
        # Simulate player decision (in production: from UI input)
        player_decision = {
            "rebalance": {
                "Hospital Bond": "NEUTRAL",
                "Pharma Equity": "OVERWEIGHT",
                "Insurance Book": "UNDERWEIGHT",
                "Healthcare REIT": "NEUTRAL",
            }
        }
        result = engine.run_quarter(player_decision)
        print(f"Quarter: {result['quarter']}")
        print(f"  Scenario: {result['scenario']['name']} [{result['scenario']['severity']}]")
        print(f"  Player return: {result['player']['portfolio_return']:.2%} | Points: {result['player']['points_earned']}")
        print(f"  AI return:     {result['ai']['portfolio_return']:.2%} | Points: {result['ai']['points_earned']}")
        print(f"  Scores → Player: {result['player']['total_score']} | AI: {result['ai']['total_score']}\n")

    print("=== Game Summary ===")
    summary = engine.get_final_results()
    print(f"Player Score: {summary['player_final_score']}")
    print(f"AI Score:     {summary['ai_final_score']}")
    print(f"Player Wins:  {summary['player_wins']}")
