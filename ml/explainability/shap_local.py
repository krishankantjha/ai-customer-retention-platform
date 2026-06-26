"""
Local SHAP Explanations.
Handles individual customer predictions, risk contributions, and counterfactual simulations.
Implements global thread-safe explainer caching and value-aware campaign recommendations.

Architecture fix (CRITICAL-3):
    ``simulate_intervention`` previously imported ``load_artifacts`` from the
    backend application layer, creating a circular ML → backend dependency.
    Artifacts are now passed directly to the constructor, keeping this module
    fully self-contained and independently testable.
"""

import logging
import threading
import pandas as pd
import numpy as np
import shap

logger = logging.getLogger("ml.explainability.shap_local")

# Global variables for thread-safe explainer caching
_explainer_cache = {}
_explainer_cache_lock = threading.Lock()


# ---------------------------------------------------------------------------
# Campaign / Save Play configuration
# ---------------------------------------------------------------------------

# Mapping of feature signatures to prescriptive business campaigns.
# Loaded at import time; populated with reasonable defaults if configs
# are unavailable, so this module stays importable without a backend env.
try:
    import os
    import sys
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    if base_dir not in sys.path:
        sys.path.insert(0, base_dir)
    from configs.dataset_config import config_loader
    sp_config = config_loader.dashboard.get("save_plays", {})
except Exception:
    sp_config = {}

SAVE_PLAY_MAPPING = {
    "Contract_Month-to-month": (
        sp_config.get("contract_save_play", {}).get("title", "1-Year Contract Lock Campaign"),
        sp_config.get("contract_save_play", {}).get("description", "Offer a monthly rate discount to transition the customer from Month-to-Month to a stable 1-Year contract.")
    ),
    "contract_is_mtm": (
        sp_config.get("contract_save_play", {}).get("title", "1-Year Contract Lock Campaign"),
        sp_config.get("contract_save_play", {}).get("description", "Offer a monthly rate discount to transition the customer from Month-to-Month to a stable 1-Year contract.")
    ),
    "PaymentMethod_Electronic check": (
        sp_config.get("billing_save_play", {}).get("title", "Auto-Pay Conversion Promotion"),
        sp_config.get("billing_save_play", {}).get("description", "Transition from manual paper checks or one-off payment methods to AutoPay (Credit Card/Bank Transfer).")
    ),
    "fiber_zero_engagement_flag": (
        sp_config.get("fiber_premium_save_play", {}).get("title", "Premium Engagement Outreach"),
        sp_config.get("fiber_premium_save_play", {}).get("description", "Offer bundle incentives on StreamingTV or DeviceProtection for high-cost Fiber optic lines.")
    ),
    "MonthlyCharges": (
        "Billing Rate Audit",
        "Perform a billing audit to identify unused features and downgrade to a cheaper plan or offer a $10/month loyalty discount."
    ),
    "high_charge_early_stage_flag": (
        "Billing Rate Audit",
        "Perform a billing audit to identify unused features and downgrade to a cheaper plan or offer a $10/month loyalty discount."
    ),
    "vulnerable_customer_flag": (
        "Priority Onboarding Support",
        "Connect the customer with a dedicated technical support specialist to assist with home device setup and contract onboarding."
    ),
    "InternetService_Fiber optic": (
        "Fiber Performance Check",
        "Contact the customer to verify line speed satisfaction and offer a free tech-health router audit."
    ),
    "OnlineSecurity_No": (
        "Security Add-on Upsell",
        "Send promotional materials outlining security services and offer a 30-day free trial of Online Security."
    ),
    "TechSupport_No": (
        sp_config.get("tech_support_save_play", {}).get("title", "Support Concierge"),
        sp_config.get("tech_support_save_play", {}).get("description", "Provide a 1-year free trial of TechSupport or OnlineSecurity add-ons to improve stickiness.")
    ),
    "tenure": (
        "Early Stage Loyalty Welcome",
        "Customer is in the early tenure phase. Send a personalized account check-in call to address initial configuration issues."
    )
}

# Tenure threshold in raw months below which the "early stage" campaign applies.
# Using the raw value (12 months = 1 year) avoids magic numbers derived from
# dataset-specific scaler parameters (FIX HIGH-2).
_EARLY_TENURE_THRESHOLD_MONTHS: int = 12


def check_business_condition(feature: str, customer_row: pd.Series) -> bool:
    """
    Validates whether the customer's demographics or subscription settings satisfy 
    the targeted eligibility criteria of a save play recommendation. This avoids 
    proposing redundant retention offers (such as cross-selling support when they 
    already subscribe to it).
    """
    # 1. Tenure welcomes — only welcome new customers (raw tenure < 12 months)
    if "tenure" in feature.lower():
        tenure_raw = customer_row.get("tenure", None)
        if tenure_raw is not None:
            try:
                if float(tenure_raw) >= _EARLY_TENURE_THRESHOLD_MONTHS:
                    return False  # Already a long-term user
            except (TypeError, ValueError):
                pass  # If uncastable, conservatively allow the campaign
        return True

    # 2. Contract locks — only offer upgrades to Month-to-Month users
    if "contract" in feature.lower() or "mtm" in feature.lower():
        contract_val = customer_row.get("Contract", None)
        if contract_val is not None:
            if contract_val in ["One year", "Two year", 12, 24, 12.0, 24.0]:
                return False  # Already contract bound
        return True

    # 3. Billing pay options — only convert manual check payers
    if "payment" in feature.lower() or "check" in feature.lower():
        pay_val = customer_row.get("PaymentMethod", None)
        if pay_val is not None:
            if "auto" in str(pay_val).lower() or pay_val == 1 or pay_val == 1.0:
                return False  # Already on AutoPay
        return True

    # 4. Security add-on upsell — only upsell if missing
    if "security" in feature.lower():
        sec_val = customer_row.get("OnlineSecurity", None)
        if sec_val is not None:
            if str(sec_val).lower() == "yes" or sec_val == 1 or sec_val == 1.0:
                return False  # Already has OnlineSecurity
        return True

    # 5. Technical support upsell — only upsell if missing
    if "techsupport" in feature.lower():
        ts_val = customer_row.get("TechSupport", None)
        if ts_val is not None:
            if str(ts_val).lower() == "yes" or ts_val == 1 or ts_val == 1.0:
                return False  # Already has TechSupport
        return True

    return True


def _map_shap_to_save_plays(
    feature_impacts: list,
    customer_row: pd.Series,
    top_n: int = 3,
) -> tuple:
    """
    Shared logic: filters positive SHAP contributors, ranks them, and maps
    to campaign Save Plays.  Returns (top_drivers, save_plays).
    """
    positive_drivers = [(f, v) for f, v in feature_impacts if v > 0]
    positive_drivers.sort(key=lambda x: x[1], reverse=True)
    top_drivers = positive_drivers[:top_n]

    save_plays = []
    for feature, val in top_drivers:
        matched = False
        for key, play in SAVE_PLAY_MAPPING.items():
            if key in feature:
                if check_business_condition(feature, customer_row):
                    save_plays.append({
                        "feature": feature,
                        "contribution": float(val),
                        "play_name": play[0],
                        "recommendation": play[1],
                    })
                    matched = True
                    break
        if not matched:
            save_plays.append({
                "feature": feature,
                "contribution": float(val),
                "play_name": "General Loyalty Outreach",
                "recommendation": (
                    f"Initiate check-in call addressing customer satisfaction "
                    f"regarding feature: {feature}."
                ),
            })

    return (
        [{"feature": f, "shap_value": float(v)} for f, v in top_drivers],
        save_plays,
    )


class LocalExplainer:
    """
    Modular engine to handle individual customer predictions, risk contributions
    (SHAP values), and prescriptive recommendations using global cached explainers.

    Artifacts (preprocessor, encoders, metadata) are now constructor arguments
    (FIX CRITICAL-3: eliminates the ML → backend circular import).
    """

    def __init__(
        self,
        model,
        feature_names: list,
        explainer=None,
        preprocessor=None,
        encoders: dict = None,
        metadata: dict = None,
    ):
        """
        Parameters
        ----------
        model       : fitted classifier (sklearn or XGBoost).
        feature_names : list of feature column names expected by the model.
        explainer   : pre-built SHAP explainer (optional, will be created if None).
        preprocessor : fitted sklearn ColumnTransformer / Pipeline for counterfactuals.
        encoders    : dict containing 'train_monthly_charges_median' and
                      'feature_names_out' for counterfactual re-processing.
        metadata    : dict containing 'feature_names_in' for model alignment.
        """
        self.model = model
        self.feature_names = feature_names
        self.preprocessor = preprocessor
        self.encoders = encoders or {}
        self.metadata = metadata or {}

        # Resolve base GBDT classifier from calibration wrapper if needed
        if hasattr(model, "calibrated_classifiers_"):
            self.base_estimator = model.calibrated_classifiers_[0].estimator
        elif hasattr(model, "xgb_"):  # Custom CalibratedGBDTEnsemble champion
            self.base_estimator = model.xgb_
        else:
            self.base_estimator = model

        if explainer is not None:
            self.explainer = explainer
        else:
            # Implement global cached explainer lookup (eliminates instantiation latency)
            model_id = id(self.base_estimator)
            with _explainer_cache_lock:
                if model_id not in _explainer_cache:
                    logger.info(f"Instantiating new TreeExplainer for model ID {model_id}...")
                    _explainer_cache[model_id] = shap.Explainer(self.base_estimator)
                self.explainer = _explainer_cache[model_id]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def explain_customer(self, customer_df: pd.DataFrame, top_n: int = 3) -> dict:
        """
        Computes local SHAP values for a customer record and maps the top
        positive drivers to prescriptive Save Plays under value-aware conditions.
        """
        logger.info("Starting local explanation for customer record")

        # Ensure customer columns are aligned to what the model expects
        customer_aligned = customer_df[self.feature_names]

        try:
            shap_values = self.explainer(customer_aligned)
            contributions = shap_values.values[0]
            feature_impacts = list(zip(self.feature_names, contributions))
            customer_row = customer_df.iloc[0]

            top_drivers, save_plays = _map_shap_to_save_plays(
                feature_impacts, customer_row, top_n
            )
            return {"success": True, "top_drivers": top_drivers, "save_plays": save_plays}

        except Exception as e:
            logger.error(f"Local SHAP explanation failed: {e}", exc_info=True)
            return {"success": False, "error": str(e), "top_drivers": [], "save_plays": []}

    def explain_from_shap_values(
        self,
        shap_row: np.ndarray,
        customer_row: pd.Series,
        top_n: int = 3,
    ) -> dict:
        """
        Maps pre-computed SHAP values (a single row) directly to Save Plays.

        Used by batch prediction to avoid calling SHAP per-row in a loop —
        the caller computes all SHAP values at once and passes individual rows
        here (FIX HIGH-3: enables batch-level SHAP computation).

        Parameters
        ----------
        shap_row    : np.ndarray of SHAP values, shape (n_features,).
        customer_row: pd.Series of raw customer values for business condition checks.
        top_n       : number of top drivers to return.
        """
        feature_impacts = list(zip(self.feature_names, shap_row))
        top_drivers, save_plays = _map_shap_to_save_plays(feature_impacts, customer_row, top_n)
        return {"success": True, "top_drivers": top_drivers, "save_plays": save_plays}

    def simulate_intervention(self, customer_df: pd.DataFrame, edits: dict) -> float:
        """
        Clones the customer record, applies edits, runs the preprocessor transform,
        and outputs the simulated risk probability.

        FIX CRITICAL-3: No longer calls load_artifacts() from the backend layer.
        Preprocessor, encoders, and metadata are used from constructor arguments.
        """
        try:
            if self.preprocessor is None or not self.encoders:
                raise ValueError(
                    "simulate_intervention requires preprocessor and encoders to be "
                    "provided to the LocalExplainer constructor. Pass them explicitly."
                )

            # Clone customer record and apply edits
            df_edit = customer_df.copy()
            for col, val in edits.items():
                if col in df_edit.columns:
                    df_edit[col] = val

            # Run feature engineering using the injected median (no leakage)
            from ml.preprocessing.engineer import engineer_features
            df_engineered = engineer_features(
                df_edit, self.encoders["train_monthly_charges_median"]
            )

            # Transform using the injected preprocessor
            X_transformed = self.preprocessor.transform(df_engineered)
            feature_names_out = self.encoders.get("feature_names_out", [])
            X_df = pd.DataFrame(X_transformed, columns=feature_names_out)

            # Align to the model's expected inputs
            model_features = self.metadata.get("feature_names_in", self.feature_names)
            X_aligned = X_df[model_features]

            y_prob = self.model.predict_proba(X_aligned)[0, 1]
            return float(y_prob)

        except Exception as e:
            logger.error(f"Simulation of intervention failed: {e}", exc_info=True)
            # Fallback: return the model probability on the unmodified aligned record
            try:
                X_aligned = customer_df[self.feature_names]
                return float(self.model.predict_proba(X_aligned)[0, 1])
            except Exception:
                return 0.0

    def run_simulations(self, customer_df: pd.DataFrame) -> list:
        """
        Runs standard counterfactual interventions and returns simulation details.
        """
        # Original probability
        original_risk = self.simulate_intervention(customer_df, {})

        simulations = []

        # 1. Contract Upgrade: month-to-month → One year
        contract_val = customer_df.iloc[0].get("Contract", "Month-to-month")
        if contract_val == "Month-to-month":
            sim_risk = self.simulate_intervention(customer_df, {"Contract": "One year"})
            simulations.append({
                "intervention": "Contract Upgrade (Month-to-Month → 1-Year)",
                "original_risk": original_risk,
                "simulated_risk": sim_risk,
                "risk_reduction": max(0.0, original_risk - sim_risk),
            })

        # 2. AutoPay Conversion: manual/check → Credit card (automatic)
        pay_val = customer_df.iloc[0].get("PaymentMethod", "Mailed check")
        if "automatic" not in str(pay_val).lower():
            sim_risk = self.simulate_intervention(
                customer_df, {"PaymentMethod": "Credit card (automatic)"}
            )
            simulations.append({
                "intervention": "Auto-Pay Conversion (Manual → Credit Card AutoPay)",
                "original_risk": original_risk,
                "simulated_risk": sim_risk,
                "risk_reduction": max(0.0, original_risk - sim_risk),
            })

        # 3. Tech Support Addition: No → Yes
        ts_val = customer_df.iloc[0].get("TechSupport", "No")
        if str(ts_val).lower() == "no":
            sim_risk = self.simulate_intervention(customer_df, {"TechSupport": "Yes"})
            simulations.append({
                "intervention": "Tech Support Activation (Add TechSupport)",
                "original_risk": original_risk,
                "simulated_risk": sim_risk,
                "risk_reduction": max(0.0, original_risk - sim_risk),
            })

        return simulations
