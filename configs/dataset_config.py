"""
Configuration Loader Module.
Provides centralized access to YAML settings (model, training, features, dashboard).
"""

import os
import yaml

MODEL_DEFAULTS = {
    "random_seed": 42,
    "smote": {
        "k_neighbors": 5
    },
    "decision_threshold": 0.528,
    "pruned_columns": ["binary__has_support"],
    "linear_model_exclusions": ["binary__is_early_stage", "AvgMonthlyCharge"],
    "cost_coefficients": {
        "false_negative_cost": 5.0,
        "false_positive_cost": 1.0
    },
    "tuning": {
        "logistic_regression": {
            "C": [0.01, 0.1, 1.0, 10.0],
            "penalty": ["l1", "l2"],
            "solver": ["liblinear"]
        },
        "random_forest": {
            "n_estimators": [50, 100, 200],
            "max_depth": [5, 10, 15, None],
            "min_samples_split": [2, 5, 10]
        },
        "adaboost": {
            "n_estimators": [50, 100, 200],
            "learning_rate": [0.01, 0.1, 1.0]
        },
        "gradient_boosting": {
            "n_estimators": [50, 100, 150],
            "learning_rate": [0.01, 0.05, 0.1],
            "max_depth": [3, 4, 5]
        },
        "xgboost": {
            "max_depth": [3, 4, 5],
            "learning_rate": [0.01, 0.05, 0.1],
            "n_estimators": [50, 100, 150],
            "min_child_weight": [1, 3, 5]
        },
        "lightgbm": {
            "max_depth": [3, 4, 5],
            "learning_rate": [0.01, 0.05, 0.1],
            "n_estimators": [50, 100, 150],
            "num_leaves": [7, 15, 31]
        },
        "mlp": {
            "hidden_layer_sizes": [[50], [100], [50, 25]],
            "alpha": [0.0001, 0.001, 0.01],
            "learning_rate_init": [0.001, 0.01]
        }
    },
    "champion_model": {
        "algorithm": "xgboost",
        "learning_rate": 0.05,
        "max_depth": 4,
        "min_child_weight": 3,
        "n_estimators": 50,
        "eval_metric": "logloss"
    }
}

TRAINING_DEFAULTS = {
    "data_paths": {
        "raw_data": "data/raw/Telco_Customer_Churn.csv",
        "processed_dir": "data/processed",
        "clean_data": "data/processed/telco_churn_clean.csv",
        "train_features": "data/processed/train_features.csv",
        "test_features": "data/processed/test_features.csv",
        "artifacts_dir": "ml/artifacts"
    }
}

FEATURE_DEFAULTS = {
    "target_column": "Churn",
    "key_column": "customerID",
    "categorical_columns": [
        "gender",
        "MultipleLines",
        "InternetService",
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
        "Contract",
        "PaymentMethod"
    ],
    "binary_columns": [
        "SeniorCitizen",
        "Partner",
        "Dependents",
        "PhoneService",
        "PaperlessBilling"
    ],
    "numeric_columns": [
        "tenure",
        "MonthlyCharges",
        "TotalCharges"
    ]
}

DASHBOARD_DEFAULTS = {
    "theme": {
        "mode": "dark",
        "primary_color": "#6366F1",
        "secondary_color": "#F97316",
        "danger_color": "#EF4444"
    },
    "risk_thresholds": {
        "low_max": 0.30,
        "medium_max": 0.70
    },
    "save_plays": {
        "tech_support_save_play": {
            "title": "Support Concierge",
            "description": "Provide a 1-year free trial of TechSupport or OnlineSecurity add-ons to improve stickiness."
        },
        "billing_save_play": {
            "title": "Billing Alignment Plan",
            "description": "Transition from manual paper checks or one-off payment methods to AutoPay (Credit Card/Bank Transfer)."
        },
        "contract_save_play": {
            "title": "Long-Term Retention Play",
            "description": "Proactively offer a discounted 1-year or 2-year contract rate to secure lock-in."
        },
        "fiber_premium_save_play": {
            "title": "Premium Engagement Outreach",
            "description": "Offer bundle incentives on StreamingTV or DeviceProtection for high-cost Fiber optic lines."
        }
    }
}

def deep_merge(default_dict, loaded_dict):
    """Recursively merges a loaded dictionary into a defaults template."""
    if not isinstance(loaded_dict, dict):
        return default_dict
    merged = default_dict.copy()
    for k, v in loaded_dict.items():
        if k in merged and isinstance(merged[k], dict) and isinstance(v, dict):
            merged[k] = deep_merge(merged[k], v)
        else:
            merged[k] = v
    return merged

class ConfigLoader:
    """
    Centralized configuration manager that reads YAML files from the configs directory
    and exposes them as parsed dictionaries.
    """
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ConfigLoader, cls).__new__(cls)
            cls._instance._load_all_configs()
        return cls._instance
        
    def _load_all_configs(self):
        # Locate the configs directory relative to this file
        current_dir = os.path.dirname(os.path.abspath(__file__))
        
        # 1. Load model configuration
        self.model = MODEL_DEFAULTS
        try:
            model_path = os.path.join(current_dir, "model_config.yaml")
            if os.path.exists(model_path):
                with open(model_path, "r", encoding="utf-8") as f:
                    loaded = yaml.safe_load(f)
                    if isinstance(loaded, dict):
                        self.model = deep_merge(MODEL_DEFAULTS, loaded)
        except Exception as e:
            print(f"Warning: Failed to load model_config.yaml, utilizing defaults. Details: {e}")
            
        # 2. Load training configuration
        self.training = TRAINING_DEFAULTS
        try:
            training_path = os.path.join(current_dir, "training_config.yaml")
            if os.path.exists(training_path):
                with open(training_path, "r", encoding="utf-8") as f:
                    loaded = yaml.safe_load(f)
                    if isinstance(loaded, dict):
                        self.training = deep_merge(TRAINING_DEFAULTS, loaded)
        except Exception as e:
            print(f"Warning: Failed to load training_config.yaml, utilizing defaults. Details: {e}")
            
        # 3. Load feature schema configuration
        self.feature = FEATURE_DEFAULTS
        try:
            feature_path = os.path.join(current_dir, "feature_config.yaml")
            if os.path.exists(feature_path):
                with open(feature_path, "r", encoding="utf-8") as f:
                    loaded = yaml.safe_load(f)
                    if isinstance(loaded, dict):
                        self.feature = deep_merge(FEATURE_DEFAULTS, loaded)
        except Exception as e:
            print(f"Warning: Failed to load feature_config.yaml, utilizing defaults. Details: {e}")
            
        # 4. Load dashboard metadata configuration
        self.dashboard = DASHBOARD_DEFAULTS
        try:
            dashboard_path = os.path.join(current_dir, "dashboard_config.yaml")
            if os.path.exists(dashboard_path):
                with open(dashboard_path, "r", encoding="utf-8") as f:
                    loaded = yaml.safe_load(f)
                    if isinstance(loaded, dict):
                        self.dashboard = deep_merge(DASHBOARD_DEFAULTS, loaded)
        except Exception as e:
            print(f"Warning: Failed to load dashboard_config.yaml, utilizing defaults. Details: {e}")


# Singleton instance for global importing
config_loader = ConfigLoader()
