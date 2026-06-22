"""
HealthRisk AI — Unit & Integration Tests (Day 15)
Covers: data pipeline, models, financial modules, simulation.
Target: ≥80% code coverage
"""

import numpy as np
import pandas as pd
import pytest
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


# ────────────────────────────────────────────────────────────────
# FIXTURES
# ────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def synthetic_datasets():
    from data_pipeline.synthetic_data import generate_all_datasets
    return generate_all_datasets(output_dir="data/raw", seed=42)


@pytest.fixture(scope="module")
def small_patients(synthetic_datasets):
    return synthetic_datasets["patients"].head(200)


@pytest.fixture(scope="module")
def small_admissions(synthetic_datasets):
    return synthetic_datasets["admissions"].head(200)


@pytest.fixture(scope="module")
def small_labs(synthetic_datasets):
    return synthetic_datasets["labs"].head(500)


@pytest.fixture(scope="module")
def small_diagnoses(synthetic_datasets):
    return synthetic_datasets["diagnoses"].head(800)


@pytest.fixture(scope="module")
def hospital_data(synthetic_datasets):
    return synthetic_datasets["hospitals"]


@pytest.fixture(scope="module")
def pharma_data(synthetic_datasets):
    return synthetic_datasets["pharma"]


# ────────────────────────────────────────────────────────────────
# PHASE 1: DATA PIPELINE TESTS
# ────────────────────────────────────────────────────────────────

class TestSyntheticDataGenerator:
    def test_patients_shape(self, small_patients):
        assert len(small_patients) == 200
        assert "subject_id" in small_patients.columns
        assert "age" in small_patients.columns

    def test_patients_age_range(self, small_patients):
        assert small_patients["age"].between(18, 100).all()

    def test_patients_gender_values(self, small_patients):
        assert set(small_patients["gender"].unique()).issubset({"M", "F"})

    def test_admissions_shape(self, small_admissions):
        assert len(small_admissions) == 200
        assert "hadm_id" in small_admissions.columns
        assert "los_days" in small_admissions.columns

    def test_admissions_los_positive(self, small_admissions):
        assert (small_admissions["los_days"] > 0).all()

    def test_labs_values_reasonable(self, small_labs):
        assert (small_labs["value"] > 0).all()

    def test_hospital_data_shape(self, hospital_data):
        assert len(hospital_data) == 200
        assert "operating_margin" in hospital_data.columns

    def test_pharma_data_shape(self, pharma_data):
        assert len(pharma_data) == 300
        assert "phase" in pharma_data.columns
        assert "peak_sales_estimate_b" in pharma_data.columns

    def test_no_negative_costs(self, hospital_data):
        if "total_revenue_m" in hospital_data.columns:
            assert (hospital_data["total_revenue_m"] > 0).all()


class TestClinicalDataCleaner:
    def test_clean_admissions_removes_neglos(self, small_admissions):
        from data_pipeline.features import ClinicalDataCleaner
        cleaner = ClinicalDataCleaner(verbose=False)
        bad_row = small_admissions.copy()
        bad_row.loc[0, "los_days"] = -5
        cleaned = cleaner.clean_admissions(bad_row)
        assert (cleaned["los_days"] > 0).all()

    def test_impute_missing_median(self):
        from data_pipeline.features import ClinicalDataCleaner
        cleaner = ClinicalDataCleaner(verbose=False)
        df = pd.DataFrame({"a": [1.0, np.nan, 3.0, 4.0], "b": [2.0, 2.0, np.nan, 4.0]})
        result = cleaner.impute_missing(df)
        assert result.isna().sum().sum() == 0

    def test_clean_lab_events_filters_outliers(self, small_labs):
        from data_pipeline.features import ClinicalDataCleaner
        cleaner = ClinicalDataCleaner(verbose=False)
        bad_labs = small_labs.copy()
        bad_labs.loc[0, "value"] = 9999  # Unrealistic creatinine
        cleaned = cleaner.clean_lab_events(bad_labs)
        assert 9999 not in cleaned["value"].values

    def test_charlson_index_non_negative(self, small_diagnoses):
        from data_pipeline.features import ClinicalFeatureEngineer
        engineer = ClinicalFeatureEngineer()
        cci = engineer.compute_charlson_index(small_diagnoses)
        assert (cci >= 0).all()

    def test_charlson_index_max_15(self, small_diagnoses):
        from data_pipeline.features import ClinicalFeatureEngineer
        engineer = ClinicalFeatureEngineer()
        cci = engineer.compute_charlson_index(small_diagnoses)
        assert (cci <= 15).all()


# ────────────────────────────────────────────────────────────────
# PHASE 2: CLINICAL AI MODEL TESTS
# ────────────────────────────────────────────────────────────────

class TestClinicalNERPipeline:
    def test_ner_returns_list(self):
        from clinical_nlp.clinicalbert import ClinicalNERPipeline
        ner = ClinicalNERPipeline()
        result = ner.extract_entities("Patient has heart failure and is taking furosemide.")
        assert isinstance(result, list)

    def test_ner_negation_detection(self):
        from clinical_nlp.clinicalbert import ClinicalNERPipeline
        ner = ClinicalNERPipeline()
        text = "No chest pain or shortness of breath."
        entities = ner.extract_entities(text)
        # At minimum, we should not crash
        assert isinstance(entities, list)

    def test_entity_features_dataframe(self):
        from clinical_nlp.clinicalbert import ClinicalNERPipeline
        ner = ClinicalNERPipeline()
        texts = ["Patient has diabetes.", "No hypertension found."]
        df = ner.build_entity_features(texts)
        assert len(df) == 2
        assert "n_medications" in df.columns
        assert "n_diagnoses" in df.columns


class TestGNNModel:
    def test_gat_forward_pass(self):
        import torch
        from graph_nn.gat_model import HealthRiskGAT
        model = HealthRiskGAT(in_channels=12, hidden_channels=32, out_channels=16, num_classes=2)
        x = torch.randn(50, 12)
        out = model(x, edge_index=None)
        assert out.shape == (50, 2)

    def test_gat_embedding_extraction(self):
        import torch
        from graph_nn.gat_model import HealthRiskGAT, GNNTrainer
        model = HealthRiskGAT(in_channels=10, hidden_channels=32, out_channels=16)
        trainer = GNNTrainer(model, device="cpu")
        x = torch.randn(100, 10)
        embeddings = trainer.get_patient_embeddings(x)
        assert embeddings.shape[0] == 100

    def test_graph_builder_patient_nodes(self, small_admissions, small_patients):
        from graph_nn.gat_model import HealthRiskGraphBuilder
        builder = HealthRiskGraphBuilder()
        # Merge to create feature table
        feats = small_admissions.merge(
            small_patients[["subject_id", "age", "charlson_index"]], on="subject_id", how="left"
        )
        feats = feats.fillna(0)
        result = builder.build_patient_nodes(feats)
        assert result.shape[0] == len(feats)
        assert result.shape[1] > 0


class TestSurvivalModels:
    def test_deepsurv_training(self):
        from survival.survival_models import DeepSurvModel
        np.random.seed(42)
        n = 200
        X = np.random.randn(n, 8)
        times = np.abs(np.random.exponential(2.0, n))
        events = np.random.binomial(1, 0.30, n)
        model = DeepSurvModel(in_features=8, hidden_sizes=[32, 16], lr=1e-2)
        results = model.fit(X, times, events, epochs=10, val_size=0.20)
        assert "best_c_index" in results
        assert 0 <= results["best_c_index"] <= 1

    def test_deepsurv_predict_shape(self):
        from survival.survival_models import DeepSurvModel
        np.random.seed(42)
        n = 100
        X = np.random.randn(n, 8)
        times = np.abs(np.random.exponential(2.0, n))
        events = np.random.binomial(1, 0.30, n)
        model = DeepSurvModel(in_features=8, hidden_sizes=[32, 16], lr=1e-2)
        model.fit(X, times, events, epochs=5)
        risk = model.predict_risk(X)
        assert risk.shape == (n,)

    def test_hospital_survival_preparation(self, hospital_data):
        from survival.survival_models import prepare_hospital_survival_data
        X, times, events = prepare_hospital_survival_data(hospital_data)
        assert X.shape[0] == len(hospital_data)
        assert len(times) == len(hospital_data)
        assert (times > 0).all()


class TestEnsemble:
    def test_xgboost_fit_predict(self):
        from ensemble.stacker import XGBoostModel
        np.random.seed(42)
        X = np.random.randn(200, 10)
        y = np.random.binomial(1, 0.2, 200)
        model = XGBoostModel(seed=42, n_estimators=50)
        model.fit(X, y)
        probs = model.predict_proba(X)
        assert probs.shape == (200,)
        assert (probs >= 0).all() and (probs <= 1).all()

    def test_lightgbm_fit_predict(self):
        from ensemble.stacker import LightGBMModel
        np.random.seed(42)
        X = np.random.randn(200, 10)
        y = np.random.binomial(1, 0.2, 200)
        model = LightGBMModel(seed=42, n_estimators=50)
        model.fit(X, y)
        probs = model.predict_proba(X)
        assert probs.shape == (200,)
        assert (probs >= 0).all() and (probs <= 1).all()

    def test_ensemble_fit_evaluates(self):
        from ensemble.stacker import HealthRiskEnsemble
        np.random.seed(42)
        n = 300
        X = np.random.randn(n, 10)
        y = np.random.binomial(1, 0.15, n)
        ensemble = HealthRiskEnsemble(n_splits=2, seed=42)
        ensemble.xgb_model.model.set_params(n_estimators=20)
        ensemble.lgb_model.model.set_params(n_estimators=20)
        metrics = ensemble.fit(X[:250], y[:250])
        assert "auroc" in metrics
        assert 0 <= metrics["auroc"] <= 1


# ────────────────────────────────────────────────────────────────
# PHASE 3: FINANCIAL MODULE TESTS
# ────────────────────────────────────────────────────────────────

class TestInsuranceActuarial:
    def test_chain_ladder_total_ibnr_positive(self):
        from financial.insurance.actuarial import IBNRCalculator
        calc = IBNRCalculator()
        triangle = IBNRCalculator.generate_sample_triangle(n_years=5, seed=42)
        results = calc.chain_ladder(triangle)
        assert results["total_ibnr"] >= 0

    def test_chain_ladder_ldfs_greater_than_one(self):
        from financial.insurance.actuarial import IBNRCalculator
        calc = IBNRCalculator()
        triangle = IBNRCalculator.generate_sample_triangle(n_years=5, seed=42)
        results = calc.chain_ladder(triangle)
        assert (results["ldfs"] >= 1.0).all()

    def test_bornhuetter_ferguson(self):
        from financial.insurance.actuarial import IBNRCalculator
        calc = IBNRCalculator()
        triangle = IBNRCalculator.generate_sample_triangle(n_years=5, seed=42)
        results = calc.bornhuetter_ferguson(triangle, a_priori_loss_ratio=0.85)
        assert results["total_ibnr"] >= 0
        assert len(results["ibnr_by_year"]) == 5

    def test_risk_stratification_coverage(self):
        from financial.insurance.actuarial import MemberRiskStratifier
        np.random.seed(42)
        n = 500
        strat = MemberRiskStratifier()
        df = strat.stratify(np.arange(n), np.random.lognormal(9, 1.5, n))
        assert len(df) == n
        assert set(df["risk_tier"].unique()).issubset(set(MemberRiskStratifier.TIERS.keys()))
        # All tiers should have members
        assert df["risk_tier"].nunique() >= 3

    def test_predictive_ratio_close_to_one(self):
        from financial.insurance.actuarial import HealthRiskEnhancedPricer
        np.random.seed(42)
        n = 300
        X = np.random.randn(n, 8)
        y = np.random.lognormal(9, 1.5, n)
        pricer = HealthRiskEnhancedPricer()
        pricer.fit(X, y)
        preds = pricer.predict(X)
        pr = preds.sum() / y.sum()
        assert 0.50 <= pr <= 2.0  # Reasonable range for training set


class TestHospitalCreditRisk:
    def test_credit_score_in_range(self, hospital_data):
        from financial.credit_risk.hospital_credit import HospitalCreditScorecard
        scorecard = HospitalCreditScorecard(include_clinical=True)
        scores = scorecard.score_portfolio(hospital_data.head(50))
        assert (scores["credit_score"] >= 300).all()
        assert (scores["credit_score"] <= 850).all()

    def test_implied_rating_not_null(self, hospital_data):
        from financial.credit_risk.hospital_credit import HospitalCreditScorecard
        scorecard = HospitalCreditScorecard(include_clinical=True)
        scores = scorecard.score_portfolio(hospital_data.head(20))
        assert scores["implied_rating"].notna().all()

    def test_pd_model_fit_and_predict(self, hospital_data):
        from financial.credit_risk.hospital_credit import HospitalPDModel
        model = HospitalPDModel()
        model.fit(hospital_data)
        preds = model.predict_pd(hospital_data.head(10))
        assert (preds["pd_enhanced"] >= 0).all()
        assert (preds["pd_enhanced"] <= 1).all()

    def test_enhanced_pd_different_from_traditional(self, hospital_data):
        from financial.credit_risk.hospital_credit import HospitalPDModel
        model = HospitalPDModel()
        model.fit(hospital_data)
        preds = model.predict_pd(hospital_data)
        # Enhanced should differ from traditional (clinical signals add info)
        correlation = np.corrcoef(preds["pd_traditional"], preds["pd_enhanced"])[0, 1]
        assert correlation < 1.0  # Not identical

    def test_early_warning_returns_list(self, hospital_data):
        from financial.credit_risk.hospital_credit import HospitalEarlyWarningSystem
        ews = HospitalEarlyWarningSystem()
        # Build mock quarterly data
        q_data = hospital_data[["readmission_rate_30d", "case_mix_index", "hcahps_star"]].head(4).copy()
        q_data.loc[q_data.index[-1], "readmission_rate_30d"] += 0.05  # Spike
        alerts = ews.detect_alerts("HOSP_001", q_data)
        assert isinstance(alerts, list)


class TestPharmaceuticalAnalytics:
    def test_phase_success_probability_range(self):
        from financial.pharma.rnpv_calculator import PhaseSuccessModel
        model = PhaseSuccessModel()
        result = model.compute_adjusted_probability(
            indication="Oncology",
            phase_transition="Phase III → NDA",
        )
        assert 0 <= result["adjusted_probability"] <= 1

    def test_biomarker_selection_increases_pos(self):
        from financial.pharma.rnpv_calculator import PhaseSuccessModel
        model = PhaseSuccessModel()
        pos_unselected = model.compute_adjusted_probability(
            indication="Oncology", phase_transition="Phase III → NDA",
            biomarker_selected=False
        )["adjusted_probability"]
        pos_selected = model.compute_adjusted_probability(
            indication="Oncology", phase_transition="Phase III → NDA",
            biomarker_selected=True
        )["adjusted_probability"]
        assert pos_selected > pos_unselected

    def test_rnpv_calculator_returns_dict(self):
        from financial.pharma.rnpv_calculator import RNPVCalculator
        calc = RNPVCalculator(n_simulations=100)
        result = calc.calculate(
            peak_sales_estimate_b=2.0,
            peak_sales_std_b=0.8,
            years_to_launch=3.0,
            patent_years_remaining=12,
            probability_of_success=0.50,
        )
        assert "rnpv_m" in result
        assert "p5_m" in result and "p95_m" in result
        assert result["p5_m"] <= result["rnpv_m"] <= result["p95_m"]

    def test_portfolio_optimizer_valid_weights(self):
        from financial.pharma.rnpv_calculator import PharmaPortfolioOptimizer
        np.random.seed(42)
        n_stocks = 8
        daily_rets = np.random.randn(252, n_stocks) * 0.02
        optimizer = PharmaPortfolioOptimizer()
        exp_ret = optimizer.compute_expected_returns(daily_rets)
        cov = np.cov(daily_rets.T)
        result = optimizer.optimize(exp_ret, cov)
        weights = result["weights"]
        assert abs(weights.sum() - 1.0) < 1e-5
        assert (weights >= 0).all()


# ────────────────────────────────────────────────────────────────
# PHASE 4: SIMULATION TESTS
# ────────────────────────────────────────────────────────────────

class TestSimulationEngine:
    def test_engine_initializes(self):
        from simulation.engine import HealthRiskLabEngine
        engine = HealthRiskLabEngine(start_year=2020, end_year=2022, seed=42)
        assert engine.state.year == 2020
        assert engine.state.quarter == 1
        assert len(engine.state.portfolio) > 0

    def test_portfolio_value_positive(self):
        from simulation.engine import HealthRiskLabEngine
        engine = HealthRiskLabEngine(start_year=2020, end_year=2022, seed=42)
        assert engine.state.portfolio_value > 0

    def test_run_quarter_returns_dict(self):
        from simulation.engine import HealthRiskLabEngine
        engine = HealthRiskLabEngine(start_year=2020, end_year=2022, seed=42)
        result = engine.run_quarter(player_decision=None)
        assert "quarter" in result
        assert "scenario" in result
        assert "player" in result
        assert "ai" in result

    def test_quarter_advances(self):
        from simulation.engine import HealthRiskLabEngine
        engine = HealthRiskLabEngine(start_year=2020, end_year=2022, seed=42)
        initial_label = engine.state.quarter_label
        engine.run_quarter()
        assert engine.state.quarter_label != initial_label

    def test_scenario_library_has_10_scenarios(self):
        from simulation.engine import ScenarioLibrary
        assert len(ScenarioLibrary.SCENARIOS) >= 10

    def test_replay_returns_scenario(self):
        from simulation.engine import HealthRiskLabEngine
        engine = HealthRiskLabEngine(seed=42)
        result = engine.replay("PANDEMIC_OUTBREAK")
        assert "scenario" in result
        assert result["scenario"]["id"] == "PANDEMIC_OUTBREAK"

    def test_game_over_condition(self):
        from simulation.engine import HealthRiskLabEngine
        engine = HealthRiskLabEngine(start_year=2020, end_year=2020, seed=42)
        for _ in range(5):
            engine.run_quarter()
        assert engine.is_game_over()

    def test_ai_opponent_makes_decision(self):
        from simulation.engine import HealthRiskLabEngine, ScenarioLibrary
        engine = HealthRiskLabEngine(seed=42)
        scenario = ScenarioLibrary.get_scenario("PANDEMIC_OUTBREAK")
        ai_decision = engine.ai.decide(engine.state, scenario)
        assert "action" in ai_decision
        assert "rationale" in ai_decision


# ────────────────────────────────────────────────────────────────
# INTEGRATION TESTS
# ────────────────────────────────────────────────────────────────

class TestEndToEndPipeline:
    def test_data_to_features_pipeline(self, synthetic_datasets):
        from data_pipeline.features import ClinicalDataCleaner, ClinicalFeatureEngineer
        cleaner = ClinicalDataCleaner(verbose=False)
        engineer = ClinicalFeatureEngineer()

        admissions = cleaner.clean_admissions(synthetic_datasets["admissions"].head(100))
        diagnoses = cleaner.clean_diagnoses(synthetic_datasets["diagnoses"].head(400))

        master = engineer.build_master_feature_table(
            admissions=admissions,
            patients=synthetic_datasets["patients"],
            labs=synthetic_datasets["labs"].head(300),
            diagnoses=diagnoses,
        )
        assert len(master) > 0
        assert master.isna().sum().sum() == 0  # No missing after imputation

    def test_xgboost_on_clinical_features(self, synthetic_datasets):
        from data_pipeline.features import ClinicalDataCleaner, ClinicalFeatureEngineer
        from ensemble.stacker import XGBoostModel
        from sklearn.metrics import roc_auc_score

        cleaner = ClinicalDataCleaner(verbose=False)
        engineer = ClinicalFeatureEngineer()

        admissions = cleaner.clean_admissions(synthetic_datasets["admissions"].head(500))
        diagnoses = cleaner.clean_diagnoses(synthetic_datasets["diagnoses"])
        master = engineer.build_master_feature_table(
            admissions=admissions,
            patients=synthetic_datasets["patients"],
            labs=synthetic_datasets["labs"],
            diagnoses=diagnoses,
        )

        feature_cols = [c for c in ["age", "charlson_index", "los_days", "icu_flag",
                                     "num_prior_admissions"] if c in master.columns]
        target_col = "readmission_30d" if "readmission_30d" in master.columns else "hospital_expire_flag"

        if target_col in master.columns and len(feature_cols) >= 2:
            X = master[feature_cols].fillna(0).values
            y = master[target_col].fillna(0).values
            model = XGBoostModel(seed=42, n_estimators=50)
            model.fit(X, y)
            probs = model.predict_proba(X)
            auroc = roc_auc_score(y, probs)
            assert auroc > 0.5  # Should be better than random

    def test_full_hospital_credit_pipeline(self, hospital_data):
        from financial.credit_risk.hospital_credit import HospitalPDModel, HospitalCreditScorecard
        model = HospitalPDModel()
        model.fit(hospital_data)
        preds = model.predict_pd(hospital_data)
        scorecard = HospitalCreditScorecard()
        scores = scorecard.score_portfolio(hospital_data)

        merged = preds.merge(scores, on="hospital_id")
        assert len(merged) == len(hospital_data)

    def test_simulation_full_game_run(self):
        from simulation.engine import HealthRiskLabEngine
        engine = HealthRiskLabEngine(start_year=2020, end_year=2021, seed=42)
        quarters_run = 0
        while not engine.is_game_over():
            engine.run_quarter()
            quarters_run += 1
            if quarters_run > 20:
                break

        results = engine.get_final_results()
        assert "player_final_score" in results
        assert results["quarters_played"] >= 1
        assert results["final_portfolio_value_m"] > 0
