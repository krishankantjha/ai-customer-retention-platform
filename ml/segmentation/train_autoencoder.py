"""
Training script for Neural Feature Representation Compression (Autoencoder).
Fits the Autoencoder and serializes the model to a pickle file.
"""

import os
import sys
import pickle
import logging
import numpy as np

# Add project root to path
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if base_dir not in sys.path:
    sys.path.insert(0, base_dir)

from configs.dataset_config import config_loader
from ml.segmentation.autoencoder import AutoencoderWrapper
from ml.segmentation.kmeans import reconstruct_natural_features

logger = logging.getLogger("ml.segmentation.train_autoencoder")


def run_autoencoder_pipeline(train_csv: str, output_dir: str, latent_dim: int = 16, random_seed: int = 42) -> dict:
    """
    Fits and serializes the Autoencoder model over reconstructed training features.
    """
    logger.info("Initializing Autoencoder representation training...")
    
    # Reconstruct natural preprocessed training features (all variables)
    X_natural_all, _ = reconstruct_natural_features(random_seed)
    
    X_matrix = X_natural_all.values.astype(np.float32)
    
    autoencoder = AutoencoderWrapper(latent_dim=latent_dim, random_seed=random_seed)
    mse = autoencoder.fit(X_matrix)
    
    # Save the model pickle
    os.makedirs(os.path.join(output_dir, "models"), exist_ok=True)
    model_path = os.path.join(output_dir, "models", "autoencoder_model.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(autoencoder, f)
        
    logger.info(f"Autoencoder serialized successfully to {model_path}")
    
    return {
        "model_path": model_path,
        "reconstruction_mse": mse,
        "input_dim": X_matrix.shape[1],
        "latent_dim": latent_dim
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    
    config_train_path = config_loader.training["data_paths"]["train_features"]
    config_artifacts_dir = config_loader.training["data_paths"]["artifacts_dir"]
    
    train_csv = config_train_path if os.path.isabs(config_train_path) else os.path.join(base_dir, config_train_path)
    artifacts_dir = config_artifacts_dir if os.path.isabs(config_artifacts_dir) else os.path.join(base_dir, config_artifacts_dir)
    
    seed = config_loader.model.get("random_seed", 42)
    
    try:
        run_autoencoder_pipeline(train_csv, artifacts_dir, latent_dim=16, random_seed=seed)
        print("Autoencoder training succeeded.")
    except Exception as e:
        logger.exception(f"Autoencoder pipeline execution failed: {e}")
        sys.exit(1)
