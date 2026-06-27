"""
Neural Feature Representation Compression (Autoencoder).

Implements a proper deep (encoder-bottleneck-decoder) symmetric autoencoder
using an MLPRegressor with three hidden layers: 64 → 16 → 64.

The encoder projects all transformed training features into a continuous
16-dimensional latent space that feeds into K-Means clustering.  Using
three hidden layers (rather than a single bottleneck) allows the network
to learn genuinely non-linear feature interactions, not just an affine
projection.

Latent extraction:
    ReLU(ReLU(X · W1 + b1) · W2 + b2)

which is a proper two-stage encoder.
"""

import numpy as np
from sklearn.neural_network import MLPRegressor
import logging

logger = logging.getLogger("ml.segmentation.autoencoder")

# Default MSE acceptance threshold.  If validation MSE exceeds this value
# after training, the autoencoder is considered degenerate and a RuntimeError
# is raised so the caller can fall back gracefully instead of silently
# clustering noise.
_DEFAULT_MSE_THRESHOLD: float = 0.5


class AutoencoderWrapper:
    """
    Wrapper around scikit-learn MLPRegressor functioning as a deep symmetric
    autoencoder.

    Architecture (3 hidden layers):
        Input (n) → 64 (ReLU) → Latent 16 (ReLU) → 64 (ReLU) → Output (n)

    The ``transform`` method extracts the 16-dimensional latent representation
    by applying only the encoder half of the network (layers 0 and 1).
    """

    def __init__(
        self,
        latent_dim: int = 16,
        random_seed: int = 42,
        mse_threshold: float = _DEFAULT_MSE_THRESHOLD,
    ):
        self.latent_dim = latent_dim
        self.random_seed = random_seed
        self.mse_threshold = mse_threshold
        self.mlp = None

    def fit(self, X: np.ndarray) -> float:
        """
        Fits the deep MLPRegressor autoencoder to reconstruct the input.

        Returns
        -------
        float
            Validation-set reconstruction MSE (not training-set MSE — no data
            leakage).  Raises ``RuntimeError`` if MSE exceeds ``mse_threshold``.
        """
        input_dim = X.shape[1]
        logger.info(
            f"Training deep Autoencoder: "
            f"{input_dim} → 64 → {self.latent_dim} → 64 → {input_dim}"
        )

        # Three hidden layers: encoder (64 → 16) + decoder (16 → 64).
        # This architecture learns genuinely non-linear latent representations.
        self.mlp = MLPRegressor(
            hidden_layer_sizes=(64, self.latent_dim, 64),
            activation="relu",
            solver="adam",
            max_iter=500,
            random_state=self.random_seed,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=15,
            tol=1e-4,
        )
        # Train to reconstruct X (inputs are targets — autoencoder objective)
        # Hold out 10% for validation MSE computation (mirrors early_stopping split)
        from sklearn.model_selection import train_test_split as _train_test_split
        X_train_ae, X_val_ae = _train_test_split(
            X, test_size=0.10, random_state=self.random_seed
        )
        self.mlp.fit(X_train_ae, X_train_ae)

        # ----------------------------------------------------------------
        # FIX CRITICAL-2: Compute MSE on the HELD-OUT validation split,
        # not on training data.  MLPRegressor.best_validation_score_ is an
        # R² score, not a negative MSE — we compute MSE directly here.
        # ----------------------------------------------------------------
        reconstructed_val = self.mlp.predict(X_val_ae)
        validation_mse = float(np.mean((X_val_ae - reconstructed_val) ** 2))
        logger.info(
            f"Autoencoder training complete. "
            f"Best validation reconstruction MSE: {validation_mse:.6f} "
            f"(threshold: {self.mse_threshold})"
        )

        # ----------------------------------------------------------------
        # FIX HIGH-1: Quality gate — raise on degenerate latent space.
        # ----------------------------------------------------------------
        if validation_mse > self.mse_threshold:
            raise RuntimeError(
                f"Autoencoder validation MSE ({validation_mse:.4f}) exceeds the "
                f"acceptable threshold ({self.mse_threshold}). The latent space is "
                f"likely degenerate. Check input data scaling or increase max_iter."
            )

        return validation_mse

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Projects inputs into the bottleneck latent space (encoder half only).

        Parameters
        ----------
        X : np.ndarray
            Pre-processed feature matrix of shape (n_samples, n_features).

        Returns
        ----------
        np.ndarray
            Latent representation of shape (n_samples, latent_dim).
        """
        if self.mlp is None:
            raise ValueError("Autoencoder has not been fitted yet. Call fit() first.")

        # Handle backward compatibility based on loaded network architecture depth
        n_layers = len(self.mlp.coefs_)
        if n_layers == 2:
            # Single hidden bottleneck layer architecture (input -> 16 -> output)
            w1 = self.mlp.coefs_[0]
            b1 = self.mlp.intercepts_[0]
            latent = np.maximum(np.dot(X, w1) + b1, 0)
            return latent
        else:
            # Multi-layer deep symmetric autoencoder (input -> 64 -> 16 -> 64 -> output)
            w1 = self.mlp.coefs_[0]
            b1 = self.mlp.intercepts_[0]
            h1 = np.maximum(np.dot(X, w1) + b1, 0)

            w2 = self.mlp.coefs_[1]
            b2 = self.mlp.intercepts_[1]
            latent = np.maximum(np.dot(h1, w2) + b2, 0)
            return latent
