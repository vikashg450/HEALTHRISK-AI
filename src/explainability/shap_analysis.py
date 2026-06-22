"""
HealthRisk AI — Model Explainability Module (Day 14)
Implements:
  1. SHAP analysis for all prediction models (tree + deep)
  2. Counterfactual explanation generator (DiCE)
  3. Partial Dependence Plots (PDP)
  4. Model Card documentation generator
  5. Regulatory compliance mapper
"""

import numpy as np
import pandas as pd
import os
from typing import Dict, List, Optional, Any
import warnings
warnings.filterwarnings("ignore")


# ──────────────────────────────────────────────────────
# SHAP ANALYSIS
# ──────────────────────────────────────────────────────

class SHAPAnalyzer:
    """
    Comprehensive SHAP analysis for all HealthRisk AI models.
    Supports tree, linear, kernel, and deep explainers.
    """

    def __init__(self, output_dir: str = "reports/explainability"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def explain_tree_model(
        self,
        model,           # XGBoost or LightGBM
        X: np.ndarray,
        feature_names: List[str],
        model_name: str = "tree_model",
        max_display: int = 20,
    ) -> Dict:
        """SHAP TreeExplainer — fast for gradient boosting models."""
        try:
            import shap
        except ImportError:
            print("Install shap: pip install shap")
            return {}

        print(f"Computing SHAP values for {model_name}...")
        explainer = shap.TreeExplainer(model)
        shap_values = explainer(X)

        # Global feature importance
        mean_abs_shap = np.abs(shap_values.values).mean(axis=0)
        importance_df = pd.DataFrame({
            "feature": feature_names[:len(mean_abs_shap)],
            "mean_abs_shap": mean_abs_shap
        }).sort_values("mean_abs_shap", ascending=False)

        print(f"\nTop 10 features by SHAP importance ({model_name}):")
        print(importance_df.head(10).to_string(index=False))

        # Save summary plot
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            shap.summary_plot(shap_values, X, feature_names=feature_names,
                             max_display=max_display, show=False)
            plt.tight_layout()
            save_path = os.path.join(self.output_dir, f"shap_summary_{model_name}.png")
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            plt.close()
            print(f"Saved SHAP summary plot: {save_path}")
        except Exception as e:
            print(f"Plot save failed (no display): {e}")

        return {
            "importance_df": importance_df,
            "shap_values": shap_values.values,
            "explainer": explainer,
        }

    def explain_single_prediction(
        self,
        explainer,
        X_single: np.ndarray,
        feature_names: List[str],
        model_name: str = "model",
    ) -> Dict:
        """Generate waterfall plot for a single patient/hospital prediction."""
        try:
            import shap
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            return {}

        shap_val = explainer(X_single.reshape(1, -1))

        # Save waterfall plot
        shap.waterfall_plot(shap_val[0], max_display=15, show=False)
        save_path = os.path.join(self.output_dir, f"waterfall_{model_name}.png")
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()

        # Build textual explanation
        contributions = sorted(
            zip(feature_names, shap_val.values[0]),
            key=lambda x: abs(x[1]), reverse=True
        )
        explanation_lines = [
            f"  {'↑' if v > 0 else '↓'} {name}: {v:+.4f}" for name, v in contributions[:10]
        ]
        print(f"\nTop drivers for this prediction ({model_name}):")
        print("\n".join(explanation_lines))

        return {
            "top_features": contributions[:10],
            "plot_path": save_path,
            "explanation_text": "\n".join(explanation_lines),
        }

    def interaction_effects(
        self,
        model,
        X: np.ndarray,
        feature_names: List[str],
        feature_pair: Optional[tuple] = None,
        model_name: str = "model",
    ) -> None:
        """SHAP dependence plot for feature interactions."""
        try:
            import shap
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            return

        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)
        if isinstance(shap_values, list):
            shap_values = shap_values[1]

        feat_idx = 0
        if feature_pair:
            feat_names_list = list(feature_names)
            if feature_pair[0] in feat_names_list:
                feat_idx = feat_names_list.index(feature_pair[0])

        shap.dependence_plot(feat_idx, shap_values, X,
                            feature_names=feature_names,
                            interaction_index="auto", show=False)
        save_path = os.path.join(self.output_dir, f"dependence_{model_name}_{feature_names[feat_idx]}.png")
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved dependence plot: {save_path}")


# ──────────────────────────────────────────────────────
# COUNTERFACTUAL EXPLANATIONS
# ──────────────────────────────────────────────────────

class CounterfactualGenerator:
    """
    Generates counterfactual explanations:
    "What would need to change for this patient/hospital to be low-risk?"
    Uses DiCE (Diverse Counterfactual Explanations) library.
    """

    def __init__(self, feature_names: List[str], feature_ranges: Optional[Dict] = None):
        self.feature_names = feature_names
        self.feature_ranges = feature_ranges or {}

    def generate(
        self,
        model,
        X_query: np.ndarray,
        X_train: np.ndarray,
        y_train: np.ndarray,
        desired_outcome: int = 0,  # 0 = low risk
        n_counterfactuals: int = 5,
    ) -> List[Dict]:
        """Generate counterfactual examples using DiCE."""
        try:
            import dice_ml
        except ImportError:
            print("Install dice-ml: pip install dice-ml")
            return self._simple_counterfactuals(X_query, X_train, n_counterfactuals)

        try:
            # Build DiCE data interface
            train_df = pd.DataFrame(X_train, columns=self.feature_names)
            train_df["outcome"] = y_train

            data = dice_ml.Data(
                dataframe=train_df,
                continuous_features=self.feature_names,
                outcome_name="outcome"
            )
            m = dice_ml.Model(model=model, backend="sklearn")
            dice = dice_ml.Dice(data, m, method="random")

            query_df = pd.DataFrame(X_query.reshape(1, -1), columns=self.feature_names)
            cf = dice.generate_counterfactuals(
                query_df, total_CFs=n_counterfactuals,
                desired_class=desired_outcome
            )
            return cf.cf_examples_list[0].final_cfs_df.to_dict(orient="records")
        except Exception as e:
            print(f"DiCE failed ({e}), using simple fallback...")
            return self._simple_counterfactuals(X_query, X_train, n_counterfactuals)

    def _simple_counterfactuals(
        self,
        X_query: np.ndarray,
        X_train: np.ndarray,
        n_counterfactuals: int = 5,
    ) -> List[Dict]:
        """Simple perturbation-based counterfactuals (fallback)."""
        cfs = []
        for _ in range(n_counterfactuals):
            cf = X_query.copy().flatten()
            # Perturb 2-3 features toward their population medians
            perturb_idx = np.random.choice(len(cf), size=min(3, len(cf)), replace=False)
            for idx in perturb_idx:
                median_val = np.median(X_train[:, idx])
                cf[idx] = cf[idx] + 0.5 * (median_val - cf[idx])
            cfs.append({name: round(float(val), 4)
                       for name, val in zip(self.feature_names, cf)})
        return cfs

    def format_explanation(self, original: np.ndarray, counterfactual: Dict) -> str:
        """Generate human-readable counterfactual explanation."""
        lines = ["📋 What needs to change to reduce risk:"]
        for name, cf_val in counterfactual.items():
            if name in self.feature_names:
                idx = self.feature_names.index(name)
                if idx < len(original.flatten()):
                    orig_val = original.flatten()[idx]
                    if abs(cf_val - orig_val) > 0.01:
                        direction = "↓ decrease" if cf_val < orig_val else "↑ increase"
                        lines.append(f"  • {name}: {orig_val:.3f} → {cf_val:.3f} ({direction})")
        return "\n".join(lines)


# ──────────────────────────────────────────────────────
# PARTIAL DEPENDENCE PLOTS
# ──────────────────────────────────────────────────────

class PDPAnalyzer:
    """
    Partial Dependence Plots showing marginal effect of each feature on predictions.
    """

    def __init__(self, output_dir: str = "reports/explainability"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def plot_pdp(
        self,
        model,
        X: np.ndarray,
        feature_names: List[str],
        features_to_plot: Optional[List[str]] = None,
        model_name: str = "model",
    ):
        """Generate PDP plots for key features."""
        try:
            from sklearn.inspection import PartialDependenceDisplay
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            print("sklearn or matplotlib not available")
            return

        if features_to_plot is None:
            features_to_plot = feature_names[:6]

        feature_indices = [feature_names.index(f) for f in features_to_plot if f in feature_names]
        if not feature_indices:
            return

        fig, axes = plt.subplots(2, 3, figsize=(15, 8))
        axes = axes.flatten()

        for i, feat_idx in enumerate(feature_indices[:6]):
            try:
                display = PartialDependenceDisplay.from_estimator(
                    model, X, [feat_idx],
                    feature_names=feature_names,
                    ax=axes[i],
                    grid_resolution=50,
                )
                axes[i].set_title(feature_names[feat_idx], fontsize=10)
            except Exception:
                axes[i].text(0.5, 0.5, f"PDP failed for\n{feature_names[feat_idx]}",
                            ha="center", va="center", transform=axes[i].transAxes)

        plt.suptitle(f"Partial Dependence Plots — {model_name}", fontsize=14)
        plt.tight_layout()
        save_path = os.path.join(self.output_dir, f"pdp_{model_name}.png")
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved PDP plot: {save_path}")


# ──────────────────────────────────────────────────────
# MODEL CARD GENERATOR
# ──────────────────────────────────────────────────────

class ModelCardGenerator:
    """
    Generates model cards following Google's Model Card framework.
    """

    def generate(
        self,
        model_name: str,
        model_type: str,
        intended_use: str,
        training_data: str,
        evaluation_data: str,
        performance_metrics: Dict[str, float],
        limitations: List[str],
        ethical_considerations: List[str],
        regulatory_notes: str,
        output_dir: str = "docs/model_cards",
    ) -> str:
        """Generate a model card in Markdown format."""
        os.makedirs(output_dir, exist_ok=True)

        card = f"""# Model Card: {model_name}

## Model Details
- **Model type**: {model_type}
- **Version**: 1.0.0
- **Date**: {pd.Timestamp.now().strftime("%Y-%m-%d")}
- **Framework**: PyTorch / scikit-learn / XGBoost

## Intended Use
{intended_use}

**Primary use cases**:
- Clinical risk stratification for high-risk patient identification
- Health insurance actuarial pricing as an enhanced rating factor
- Hospital credit risk assessment as a leading quality indicator

**Out-of-scope uses**:
- Individual clinical diagnosis or treatment decisions
- Real-time clinical decision support without human oversight
- Use in populations significantly different from training data

## Training Data
{training_data}

## Evaluation Data
{evaluation_data}

## Performance Metrics

| Metric | Value | Target | Status |
|--------|-------|--------|--------|
"""
        targets = {
            "auroc": 0.80, "auprc": 0.40, "brier_score": 0.15,
            "c_index": 0.70, "r2": 0.25, "gini": 0.50
        }
        for metric, value in performance_metrics.items():
            target = targets.get(metric.lower(), "N/A")
            if isinstance(target, float):
                passed = (value >= target if metric != "brier_score" else value <= target)
                status = "✅ PASS" if passed else "❌ FAIL"
            else:
                status = "—"
            card += f"| {metric} | {value:.4f} | {target} | {status} |\n"

        card += f"""
## Limitations
"""
        for lim in limitations:
            card += f"- {lim}\n"

        card += f"""
## Ethical Considerations
"""
        for eth in ethical_considerations:
            card += f"- {eth}\n"

        card += f"""
## Regulatory Notes
{regulatory_notes}

## Fairness Analysis
This model should be evaluated for performance disparities across:
- Age groups (pediatric, adult, geriatric)
- Gender
- Race/ethnicity
- Insurance type (Medicare, Medicaid, Commercial)
- Geographic region

Subgroup AUROC should not deviate more than ±0.05 from overall AUROC.

## Contact & Governance
- Model owners must review performance quarterly
- PSI > 0.25 triggers mandatory model recalibration
- All predictions should be reviewed by clinical/financial experts
"""
        filename = model_name.lower().replace(" ", "_") + "_model_card.md"
        filepath = os.path.join(output_dir, filename)
        with open(filepath, "w") as f:
            f.write(card)

        print(f"Model card saved: {filepath}")
        return filepath


# ──────────────────────────────────────────────────────
# REGULATORY COMPLIANCE MAPPER
# ──────────────────────────────────────────────────────

REGULATORY_COMPLIANCE_MATRIX = {
    "SHAP Global Importance": {
        "HIPAA": "Data minimization principle: identifies which PHI features are necessary",
        "FDA_SaMD": "Algorithm transparency requirement for Software as a Medical Device",
        "CMS_MLR": "Demonstrates factors driving premium rate changes",
        "Fair_Credit": "Supports adverse action notices explaining credit decisions",
    },
    "SHAP Individual Explanation": {
        "HIPAA": "Patient right to understand automated decisions affecting their care",
        "FDA_SaMD": "Required for clinical decision support tool documentation",
        "Fair_Credit": "Required explanation for adverse credit actions",
        "ACA_Rating": "Documents rating factors for non-discriminatory pricing",
    },
    "Counterfactual Explanations": {
        "HIPAA": "Actionable care recommendations based on modifiable risk factors",
        "Fair_Credit": "Shows borrower specific remediation steps",
        "CMS_Star": "Identifies quality improvement levers for hospital star ratings",
    },
    "Model Cards": {
        "FDA_SaMD": "Predicate device documentation, intended use, performance claims",
        "ISO_14155": "Clinical investigation documentation requirements",
        "ICH_E6": "Good Clinical Practice documentation for clinical applications",
    },
}


if __name__ == "__main__":
    print("=== Testing Explainability Module ===\n")

    # Quick test with dummy model
    from sklearn.ensemble import GradientBoostingClassifier
    np.random.seed(42)
    n = 500
    feature_names = ["age", "charlson_index", "readmission_rate", "los_days",
                     "Creatinine_last", "n_medications", "hcahps_star", "dscr"]
    X = np.random.randn(n, len(feature_names))
    y = np.random.binomial(1, 0.15, n)

    model = GradientBoostingClassifier(n_estimators=50, random_state=42)
    model.fit(X, y)

    # SHAP
    analyzer = SHAPAnalyzer(output_dir="reports/explainability")
    results = analyzer.explain_tree_model(model, X[:100], feature_names, "test_model")

    # Model Card
    generator = ModelCardGenerator()
    generator.generate(
        model_name="ICU Mortality Predictor",
        model_type="Stacking Ensemble (XGBoost + LightGBM + ClinicalBERT + GNN)",
        intended_use="Predict ICU mortality risk for insurance risk stratification",
        training_data="MIMIC-IV (2019–2022), 50,000 ICU admissions",
        evaluation_data="MIMIC-IV held-out (2023), 10,000 admissions",
        performance_metrics={"AUROC": 0.854, "AUPRC": 0.612, "Brier_Score": 0.098, "C_index": 0.812},
        limitations=[
            "Trained on US academic medical center data; may not generalise to community hospitals",
            "Does not account for genetic polymorphisms affecting drug metabolism",
            "Performance may degrade for rare diseases not well-represented in MIMIC-IV",
        ],
        ethical_considerations=[
            "Model shows slight underperformance for Black patients (AUROC 0.82 vs 0.87 overall) — under investigation",
            "Should not be used for triage decisions without physician override capability",
            "Insurance use must comply with ACA non-discrimination requirements",
        ],
        regulatory_notes=(
            "This model is intended as decision support only. "
            "Classification as FDA Software as a Medical Device (SaMD) if used clinically. "
            "HIPAA compliant: uses de-identified data for training. "
            "All predictions must be reviewed by a licensed clinician or actuary."
        ),
    )
    print("\n✓ Model card generated")
    print("\nRegulatory Compliance Matrix:")
    for tool, regs in REGULATORY_COMPLIANCE_MATRIX.items():
        print(f"\n  [{tool}]")
        for reg, note in regs.items():
            print(f"    {reg}: {note}")
