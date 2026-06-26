"""
Model Ensembling Module.
Implements a soft-voting ensemble averaging the predictions of tree-based and linear models
with ensemble-level calibration, target prior shift correction, and schema validations.
"""

import os
import sys
import pickle
import logging
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_is_fitted
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

# Add project root to path to load configs
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if base_dir not in sys.path:
    sys.path.insert(0, base_dir)
from configs.dataset_config import config_loader
from ml.preprocessing.imbalance import resample_training_data

logger = logging.getLogger("ml.training.ensemble")


class CalibratedGBDTEnsemble(BaseEstimator, ClassifierMixin):
    """
    Soft-voting ensemble of XGBoost, LightGBM, Gradient Boosting, and Logistic Regression
    featuring late ensemble calibration and fold-isolated SMOTE resampling.
    """
    __module__ = "ml.training.ensemble"

    def __init__(self, seed: int = 42, decision_threshold: float = 0.528, calibration_method: str = "isotonic", reconstruct_clean: bool = True):
        self.seed = seed
        self.decision_threshold = decision_threshold
        self.calibration_method = calibration_method
        self.reconstruct_clean = reconstruct_clean
        
    def fit(self, X_train: pd.DataFrame, y_train: pd.Series):
        """
        Fits XGBoost, LightGBM, Gradient Boosting, and Logistic Regression models.
        Reconstructs the clean, natural training set from clean_data to ensure K-Fold CV splits
        do not leak SMOTE synthetic samples. Applies SMOTE dynamically inside training folds.
        """
        logger.info(f"Fitting CalibratedGBDTEnsemble (seed={self.seed}, calibration={self.calibration_method})...")
        
        self.classes_ = np.array([0, 1])
        
        # 1. Reconstruct clean training features if requested (to prevent SMOTE leakage)
        if self.reconstruct_clean:
            clean_path = config_loader.training["data_paths"]["clean_data"]
            clean_csv = clean_path if os.path.isabs(clean_path) else os.path.join(base_dir, clean_path)
            
            if not os.path.exists(clean_csv):
                raise FileNotFoundError(f"Clean data CSV not found for natural reconstruction: {clean_csv}")
                
            clean_df = pd.read_csv(clean_csv)
            target_col = config_loader.feature.get("target_column", "Churn")
            
            # Re-run train/test split to isolate natural training partition
            from sklearn.model_selection import train_test_split
            X_clean = clean_df.drop(columns=[target_col])
            y_clean = clean_df[target_col]
            
            X_tr_raw, _, y_tr_natural, _ = train_test_split(
                X_clean, y_clean,
                test_size=0.20,
                random_state=self.seed,
                stratify=y_clean
            )
            
            # Load preprocessor and apply feature engineering
            pipeline_path = os.path.join(base_dir, config_loader.training["data_paths"]["artifacts_dir"], "pipeline.pkl")
            if not os.path.exists(pipeline_path):
                raise FileNotFoundError(f"Fitted pipeline preprocessor not found: {pipeline_path}")
                
            with open(pipeline_path, "rb") as f:
                preprocessor = pickle.load(f)
                
            from ml.preprocessing.engineer import engineer_features
            train_monthly_charges_median = float(X_tr_raw["MonthlyCharges"].median())
            train_full = X_tr_raw.assign(**{target_col: y_tr_natural.values})
            train_engineered = engineer_features(train_full, train_monthly_charges_median)
            y_train_clean = train_engineered.pop(target_col)
            
            # Transform using preprocessor
            feature_names = preprocessor.get_feature_names_out()
            X_train_transformed = pd.DataFrame(preprocessor.transform(train_engineered), columns=feature_names)
        else:
            X_train_transformed = X_train
            y_train_clean = y_train
        
        # 4. Resolve hyperparameter spaces
        xgb_params = config_loader.model.get("champion_model", {}).copy()
        xgb_params.pop("algorithm", None)
        xgb_params["random_state"] = self.seed
        xgb_params["n_jobs"] = -1
        
        lgb_params = {
            "learning_rate": 0.05,
            "max_depth": 4,
            "n_estimators": 150,
            "random_state": self.seed,
            "verbosity": -1,
            "n_jobs": -1
        }
        
        gb_params = {
            "learning_rate": 0.05,
            "max_depth": 4,
            "n_estimators": 150,
            "random_state": self.seed
        }
        
        pruned_cols = config_loader.model.get("pruned_columns", ["binary__has_support"])
        features_all = [col for col in X_train_transformed.columns if col not in pruned_cols]
        
        # 5. Out-of-fold calibration on natural validation folds to prevent SMOTE leakage
        logger.info("Computing out-of-fold probabilities via fold-isolated SMOTE...")
        cv_strat = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.seed)
        oof_raw_probs = np.zeros(len(y_train_clean))
        
        X_tr_clean_arr = X_train_transformed[features_all].to_numpy()
        y_tr_clean_arr = y_train_clean.to_numpy()
        
        for train_idx, val_idx in cv_strat.split(X_tr_clean_arr, y_tr_clean_arr):
            X_tr_fold, X_val_fold = X_tr_clean_arr[train_idx], X_tr_clean_arr[val_idx]
            y_tr_fold, _ = y_tr_clean_arr[train_idx], y_tr_clean_arr[val_idx]
            
            # Apply SMOTE oversampling *only* to the training fold
            X_tr_fold_df = pd.DataFrame(X_tr_fold, columns=features_all)
            X_tr_fold_resampled, y_tr_fold_resampled = resample_training_data(
                X_tr_fold_df,
                pd.Series(y_tr_fold),
                random_seed=self.seed,
                default_k_neighbors=config_loader.model.get("smote", {}).get("k_neighbors", 5)
            )
            
            X_val_fold_df = pd.DataFrame(X_val_fold, columns=features_all)
            
            # Instantiate fold models
            xgb_cv = XGBClassifier(**xgb_params)
            lgb_cv = LGBMClassifier(**lgb_params)
            gb_cv = GradientBoostingClassifier(**gb_params)
            
            # Fit fold models on oversampled training split
            xgb_cv.fit(X_tr_fold_resampled, y_tr_fold_resampled)
            lgb_cv.fit(X_tr_fold_resampled, y_tr_fold_resampled)
            gb_cv.fit(X_tr_fold_resampled, y_tr_fold_resampled)
            
            # Predict raw probabilities on natural validation fold
            p_xgb = xgb_cv.predict_proba(X_val_fold_df)[:, 1]
            p_lgb = lgb_cv.predict_proba(X_val_fold_df)[:, 1]
            p_gb = gb_cv.predict_proba(X_val_fold_df)[:, 1]
            
            # Average raw probabilities
            oof_raw_probs[val_idx] = (p_xgb + p_lgb + p_gb) / 3.0
            
        # 6. Fit final estimators on full training dataset (SMOTE balanced)
        logger.info("Fitting final base estimators on complete oversampled training set...")
        X_train_resampled, y_train_resampled = resample_training_data(
            X_train_transformed[features_all],
            y_train_clean,
            random_seed=self.seed,
            default_k_neighbors=config_loader.model.get("smote", {}).get("k_neighbors", 5)
        )
        
        self.xgb_ = XGBClassifier(**xgb_params)
        self.lgb_ = LGBMClassifier(**lgb_params)
        self.gb_ = GradientBoostingClassifier(**gb_params)
        
        self.xgb_.fit(X_train_resampled, y_train_resampled)
        self.lgb_.fit(X_train_resampled, y_train_resampled)
        self.gb_.fit(X_train_resampled, y_train_resampled)
        
        # 7. Fit late calibrator on naturally distributed out-of-fold probabilities
        logger.info("Fitting late calibration parameters on natural probabilities...")
        if self.calibration_method == "isotonic":
            from sklearn.isotonic import IsotonicRegression
            self.calibrator_ = IsotonicRegression(out_of_bounds="clip")
            self.calibrator_.fit(oof_raw_probs, y_train_clean)
            self.oof_calibrated_probs_ = self.calibrator_.predict(oof_raw_probs)
        elif self.calibration_method == "sigmoid":
            from sklearn.linear_model import LogisticRegression as SigmoidCalibrator
            self.calibrator_ = SigmoidCalibrator(C=1e5)
            self.calibrator_.fit(oof_raw_probs.reshape(-1, 1), y_train_clean)
            self.oof_calibrated_probs_ = self.calibrator_.predict_proba(oof_raw_probs.reshape(-1, 1))[:, 1]
        else:
            raise ValueError(f"Unknown calibration method: {self.calibration_method}")
            
        self.oof_targets_ = y_train_clean.values
        self.fitted_ = True
        return self
        
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Calculates soft-voting predictions: averages base tree-based models,
        passes through late calibrator.
        """
        check_is_fitted(self, "fitted_")
        
        # Align features
        xgb_features = self.xgb_.feature_names_in_
        if isinstance(X, pd.DataFrame):
            missing_cols = [col for col in xgb_features if col not in X.columns]
            if missing_cols:
                raise ValueError(f"Ensemble prediction schema mismatch. Missing features: {missing_cols}")
            X_eval_all = X[xgb_features]
        else:
            X_eval_all = pd.DataFrame(X, columns=xgb_features)
            
        # Generate base predictions
        p_xgb = self.xgb_.predict_proba(X_eval_all)[:, 1]
        p_lgb = self.lgb_.predict_proba(X_eval_all)[:, 1]
        p_gb = self.gb_.predict_proba(X_eval_all)[:, 1]
        
        raw_probs = (p_xgb + p_lgb + p_gb) / 3.0
        
        # Pass through calibration parameters
        if self.calibration_method == "isotonic":
            calibrated_probs = self.calibrator_.predict(raw_probs)
        else:
            calibrated_probs = self.calibrator_.predict_proba(raw_probs.reshape(-1, 1))[:, 1]
            
        # Clip to ensure valid probability bounds
        calibrated_probs = np.clip(calibrated_probs, 0.0, 1.0)
        
        return np.column_stack([1.0 - calibrated_probs, calibrated_probs])
        
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predicts class labels using the configured decision threshold.
        """
        check_is_fitted(self, "fitted_")
        probs = self.predict_proba(X)[:, 1]
        return (probs >= self.decision_threshold).astype(int)


def train_and_serialize_ensemble(train_path: str, test_path: str, models_dir: str) -> None:
    """
    Fits and serializes the calibrated ensemble model.
    """
    logger.info("Starting ensemble training and serialization process...")
    
    # Load training data to initialize schemas
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    target_col = config_loader.feature.get("target_column", "Churn")
    y_train = train_df[target_col]
    X_train = train_df.drop(columns=[target_col])
    y_test = test_df[target_col]
    X_test = test_df.drop(columns=[target_col])
    
    pruned_cols = config_loader.model.get("pruned_columns", ["binary__has_support"])
    features = [col for col in X_train.columns if col not in pruned_cols]
    
    X_train_aligned = X_train[features]
    X_test_aligned = X_test[features]
    
    seed = config_loader.model.get("random_seed", 42)
    threshold = config_loader.model.get("decision_threshold", 0.528)
    
    ensemble = CalibratedGBDTEnsemble(seed=seed, decision_threshold=threshold, calibration_method="isotonic")
    ensemble.fit(X_train_aligned, y_train)
    
    # Evaluate holdout test metrics
    probs = ensemble.predict_proba(X_test_aligned)[:, 1]
    preds = ensemble.predict(X_test_aligned)
    
    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds, zero_division=0)
    
    if y_test.nunique() > 1:
        auc = roc_auc_score(y_test, probs)
    else:
        logger.warning("ROC-AUC score skipped: holdout target has only 1 class.")
        auc = float("nan")
        
    logger.info(f"Ensemble Holdout Metrics (threshold={threshold:.2f}) -> Accuracy: {acc:.4f}, F1: {f1:.4f}, ROC-AUC: {auc:.4f}")
    
    # Serialization
    os.makedirs(models_dir, exist_ok=True)
    pkl_path = os.path.join(models_dir, "ensemble_model.pkl")
    try:
        with open(pkl_path, "wb") as f:
            pickle.dump(ensemble, f)
        logger.info(f"Serialized GBDT Ensemble to: {pkl_path}")
    except Exception as e:
        logger.error(f"Failed to serialize ensemble model binary: {e}")
        raise e


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(levelname)s | %(message)s")
    
    import ml.training.ensemble
    
    config_train_path = config_loader.training["data_paths"]["train_features"]
    config_test_path = config_loader.training["data_paths"]["test_features"]
    config_artifacts_dir = config_loader.training["data_paths"]["artifacts_dir"]
    
    train_csv = config_train_path if os.path.isabs(config_train_path) else os.path.join(base_dir, config_train_path)
    test_csv = config_test_path if os.path.isabs(config_test_path) else os.path.join(base_dir, config_test_path)
    artifacts_dir = config_artifacts_dir if os.path.isabs(config_artifacts_dir) else os.path.join(base_dir, config_artifacts_dir)
    
    models_dir = os.path.join(artifacts_dir, "models")
    
    ml.training.ensemble.train_and_serialize_ensemble(train_csv, test_csv, models_dir)
