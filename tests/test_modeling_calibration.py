import os
import sys
import pytest
import numpy as np
import pandas as pd
from scipy import stats

# Add project root to path
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if base_dir not in sys.path:
    sys.path.insert(0, base_dir)

from ml.training.calibration import compute_ece
from ml.training.ensemble import CalibratedGBDTEnsemble
from ml.training.benchmark import run_bootstrap_validation
from ml.training.statistical_validation import run_statistical_validation


def test_compute_ece():
    # 1. Perfect calibration
    y_true = np.array([0, 0, 1, 1])
    y_prob = np.array([0.0, 0.0, 1.0, 1.0])
    ece = compute_ece(y_true, y_prob, n_bins=2)
    assert ece == 0.0

    # 2. Imperfect calibration
    y_true = np.array([0, 0, 1, 1])
    y_prob = np.array([0.5, 0.5, 0.5, 0.5])
    ece = compute_ece(y_true, y_prob, n_bins=2)
    # The accuracy in the bin [0.5, 1.0] is 0.5 (2 out of 4)
    # The average confidence is 0.5. Since they match, ECE is 0.0.
    # Let's try another set where accuracy and confidence differ
    y_true = np.array([0, 1, 0, 1])
    y_prob = np.array([0.1, 0.2, 0.8, 0.9])
    ece = compute_ece(y_true, y_prob, n_bins=2)
    assert ece > 0.0

    # 3. Invalid inputs
    assert np.isnan(compute_ece(None, None))
    assert np.isnan(compute_ece([], []))
    assert np.isnan(compute_ece([1], [0.5, 0.6]))


def test_ensemble_aggregation():
    # Verify that CalibratedGBDTEnsemble class can be instantiated and exposes correct APIs
    ensemble = CalibratedGBDTEnsemble(seed=42, decision_threshold=0.528, calibration_method="isotonic")
    assert ensemble.seed == 42
    assert ensemble.decision_threshold == 0.528
    assert ensemble.calibration_method == "isotonic"


def test_paired_t_test_mock():
    # Test paired t-test logic on deterministic mock values
    # Model A is significantly better than Model B
    model_a_scores = [0.85, 0.86, 0.84, 0.87, 0.88]
    model_b_scores = [0.70, 0.72, 0.69, 0.71, 0.73]
    t_stat, p_val = stats.ttest_rel(model_a_scores, model_b_scores)
    assert p_val < 0.05
    assert t_stat > 0

    # Model A and Model B are identical
    model_c_scores = [0.80, 0.80, 0.80, 0.80, 0.80]
    model_d_scores = [0.80, 0.80, 0.80, 0.80, 0.80]
    t_stat, p_val = stats.ttest_rel(model_c_scores, model_d_scores)
    assert p_val > 0.5 or np.isnan(p_val) or t_stat == 0.0


class DummyEstimator:
    def __init__(self, prob_val=0.6):
        self.prob_val = prob_val

    def predict_proba(self, X):
        # returns probabilities
        return np.column_stack([1.0 - np.full(len(X), self.prob_val), np.full(len(X), self.prob_val)])


def test_bootstrap_validation():
    # Test bootstrap logic using a dummy estimator
    y_test = pd.Series([1, 0, 1, 0, 1, 0, 1, 0, 1, 0])
    X_test = pd.DataFrame(np.zeros((10, 2)))
    model = DummyEstimator(prob_val=0.6)

    # Threshold below prob_val => predictions are all 1s
    stats_low = run_bootstrap_validation(model, X_test, y_test, threshold=0.5, n_bootstraps=50, seed=42)
    assert "F1_Mean" in stats_low
    assert "F1_Std" in stats_low
    assert "F1_CI_Lower" in stats_low
    assert "F1_CI_Upper" in stats_low
    assert "AUC_Mean" in stats_low

    # Check bounds
    assert 0.0 <= stats_low["F1_Mean"] <= 1.0


def test_benchmarking_output_schema():
    # Verify the benchmark results CSV has correct format if it exists
    metrics_dir = os.path.join(base_dir, "ml", "artifacts", "metrics")
    results_path = os.path.join(metrics_dir, "benchmark_results.csv")
    if os.path.exists(results_path):
        df = pd.read_csv(results_path)
        assert "Model" in df.columns
        assert "F1-Score (Holdout)" in df.columns or "F1 (Bootstrap Mean)" in df.columns
