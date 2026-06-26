"""
Behavioral Customer Segmentation.
Fits K-Means++ clustering on natural, continuous training features to identify distinct risk cohorts
and persists the model, Elbow/Silhouette analysis, and persona summaries.
"""

import os
import sys
import json
import pickle
import logging
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

# Add project root to path to load configs
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if base_dir not in sys.path:
    sys.path.insert(0, base_dir)
from configs.dataset_config import config_loader

logger = logging.getLogger("ml.segmentation.kmeans")


def reconstruct_natural_features(random_seed: int) -> tuple:
    """
    Reconstructs the natural, non-SMOTE training features by running raw train_test_split
    and transforming features using the pre-fitted pipeline.pkl preprocessor.
    """
    logger.info("Reconstructing natural, non-SMOTE training feature coordinates...")
    clean_path = config_loader.training["data_paths"]["clean_data"]
    clean_csv = clean_path if os.path.isabs(clean_path) else os.path.join(base_dir, clean_path)
    if not os.path.exists(clean_csv):
        raise FileNotFoundError(f"Clean CSV file not found at: {clean_csv}")
        
    clean_df = pd.read_csv(clean_csv)
    target_col = config_loader.feature.get("target_column", "Churn")
    
    X_clean = clean_df.drop(columns=[target_col])
    y_clean = clean_df[target_col]
    
    X_tr_raw, _, y_tr_natural, _ = train_test_split(
        X_clean, y_clean,
        test_size=0.20,
        random_state=random_seed,
        stratify=y_clean
    )
    
    pipeline_path = os.path.join(base_dir, config_loader.training["data_paths"]["artifacts_dir"], "pipeline.pkl")
    if not os.path.exists(pipeline_path):
        raise FileNotFoundError(f"Pipeline preprocessor path not found: {pipeline_path}")
        
    with open(pipeline_path, "rb") as f:
        preprocessor = pickle.load(f)
        
    from ml.preprocessing.engineer import engineer_features
    train_monthly_charges_median = float(X_tr_raw["MonthlyCharges"].median())
    train_full = X_tr_raw.assign(**{target_col: y_tr_natural.values})
    train_engineered = engineer_features(train_full, train_monthly_charges_median)
    y_train_clean = train_engineered.pop(target_col)
    
    feature_names = preprocessor.get_feature_names_out()
    X_train_transformed = pd.DataFrame(preprocessor.transform(train_engineered), columns=feature_names)
    
    return X_train_transformed, y_train_clean


def validate_k_selection(X: pd.DataFrame, max_k: int = 6, output_dir: str = None, random_seed: int = 42):
    """
    Saves an Elbow Curve (inertia) and Silhouette Analysis plot across multiple K values
    to justify our chosen cluster count.
    """
    logger.info("Executing Elbow Curve and Silhouette sweeps to justify K count...")
    k_range = list(range(2, max_k + 1))
    inertias = []
    silhouettes = []
    
    for k in k_range:
        kmeans = KMeans(n_clusters=k, init="k-means++", random_state=random_seed, n_init=10)
        labels = kmeans.fit_predict(X)
        inertias.append(kmeans.inertia_)
        # FIX MEDIUM-2: Pass sample_size so random_state actually controls sampling.
        # Without sample_size, random_state is silently ignored by sklearn.
        n_samples = len(X) if hasattr(X, "__len__") else X.shape[0]
        sample_size = min(2000, n_samples) if n_samples > 2000 else None
        silhouettes.append(
            silhouette_score(X, labels, sample_size=sample_size, random_state=random_seed)
        )
        
    _, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Elbow Plot
    ax1.plot(k_range, inertias, "bo-", markersize=8)
    ax1.set_xlabel("Number of Clusters (K)")
    ax1.set_ylabel("Inertia (Within-Cluster Sum of Squares)")
    ax1.set_title("Elbow Method For Optimal K")
    ax1.grid(True)
    
    # Silhouette Plot
    ax2.plot(k_range, silhouettes, "ro-", markersize=8)
    ax2.set_xlabel("Number of Clusters (K)")
    ax2.set_ylabel("Silhouette Score")
    ax2.set_title("Silhouette Analysis For Optimal K")
    ax2.grid(True)
    
    plt.tight_layout()
    analysis_path = os.path.join(output_dir, "metrics", "k_selection_analysis.png")
    plt.savefig(analysis_path, dpi=150)
    plt.close()
    logger.info(f"Saved Elbow/Silhouette K validation plots to: {analysis_path}")


def generate_cluster_profiles(kmeans, X: pd.DataFrame, y_true: pd.Series, output_dir: str):
    """
    Computes centroid stats and churn rate profiles per cluster and exports a stakeholder-friendly
    summary markdown file detailing cluster personas.
    """
    logger.info("Generating customer persona summaries and profiles...")
    labels = kmeans.labels_
    
    # Combine features and natural targets
    df_profile = X.copy()
    df_profile["Cluster"] = labels
    df_profile["Churn"] = y_true.values
    
    # Calculate means and churn rates per cluster
    profile_means = df_profile.groupby("Cluster").mean()
    cluster_counts = df_profile["Cluster"].value_counts().to_dict()
    
    markdown_content = """# Customer Segmentation Personas & Profiles

This document outlines the customer behavioral segments identified via K-Means++ clustering on natural, continuous feature coordinates. 

---

## Behavioral Personas Overview

"""
    # Business interpretation mapping for 3 clusters
    interpretations = {
        0: {
            "name": "Cluster 0: Moderate-Value, Budget-Conscious Users",
            "desc": "Medium-tenure customers paying low-to-moderate monthly charges with moderate ecosystem services. This represents your budget-conscious core user base.",
            "action": "Trigger Auto-Pay conversion and cross-sell technical security add-ons to improve retention friction."
        },
        1: {
            "name": "Cluster 1: High-Value Premium Cohort",
            "desc": "Long-tenure customers with high ecosystem service counts and high monthly billing rates. This is your most valuable premium group.",
            "action": "Ensure high-priority VIP customer support. Check fiber router performance and offer loyalty credits proactively."
        },
        2: {
            "name": "Cluster 2: New Churn-Risk Users",
            "desc": "Short-tenure customers with high initial monthly charges, short contract types, and low ecosystem subscription counts. This represents your highest churn-risk group.",
            "action": "Prioritize direct welcome onboarding check-ins, rate audits, and transition them to long-term contract lock-in campaigns."
        }
    }
    
    for cluster in sorted(profile_means.index):
        count = cluster_counts.get(cluster, 0)
        c_pct = (count / len(X)) * 100
        row = profile_means.loc[cluster]
        
        churn_rate = row.get("Churn", 0.0) * 100
        tenure_scaled = row.get("numeric__tenure", 0.0)
        monthly_scaled = row.get("numeric__MonthlyCharges", 0.0)
        services_scaled = row.get("numeric__num_services", 0.0)
        
        # We also extract unscaled properties for readability if possible
        # (Since we standardized them, we just describe their scaled relations: higher/lower than mean 0)
        tenure_desc = "High" if tenure_scaled > 0 else "Low"
        charge_desc = "High" if monthly_scaled > 0 else "Low"
        service_desc = "High" if services_scaled > 0 else "Low"
        
        interp = interpretations.get(cluster, {
            "name": f"Cluster {cluster}",
            "desc": "Behavioral cluster.",
            "action": "Provide customer satisfaction reviews."
        })
        
        markdown_content += f"""### {interp['name']}
* **Size**: {count} customers ({c_pct:.2f}% of training set)
* **Average Churn Rate**: **{churn_rate:.2f}%**
* **Scaled Behavioral Scores**:
  * Tenure: {tenure_scaled:+.3f} ({tenure_desc} tenure)
  * Monthly Charges: {monthly_scaled:+.3f} ({charge_desc} monthly billing)
  * Ecosystem Services: {services_scaled:+.3f} ({service_desc} ecosystem lock-in)
* **Description**: {interp['desc']}
* **Retention Save Play Strategy**: {interp['action']}

---

"""
    personas_path = os.path.join(output_dir, "metrics", "kmeans_personas.md")
    with open(personas_path, "w", encoding="utf-8") as f:
        f.write(markdown_content)
    logger.info(f"Saved cluster persona profile summaries to: {personas_path}")


def run_segmentation(train_path: str, output_dir: str, n_clusters: int = 3, random_seed: int = 42) -> dict:
    """
    Fits K-Means++ on continuous features reconstructed from natural training data (before SMOTE)
    projected into the Autoencoder's 16-dimensional latent space.
    """
    logger.info("Starting behavioral customer segmentation process...")
    
    os.makedirs(os.path.join(output_dir, "models"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "metrics"), exist_ok=True)
    
    # 1. Reconstruct natural data splits (no SMOTE)
    X_natural_all, y_natural = reconstruct_natural_features(random_seed)
    
    # 2. Extract continuous features from config (Option A)
    seg_cfg = config_loader.model.get("segmentation", {})
    cont_cols = seg_cfg.get("continuous_features", [
        "numeric__tenure",
        "numeric__MonthlyCharges",
        "numeric__addon_count",
        "numeric__commitment_score",
        "numeric__Contract",
        "numeric__AvgMonthlyCharge",
        "numeric__num_services"
    ])
    
    # Keep only continuous variables to satisfy Euclidean distance assumptions
    X_continuous = X_natural_all[cont_cols]
    logger.info(f"Filtered coordinates to {len(cont_cols)} continuous features.")
    
    # Load the autoencoder model if it exists, otherwise fall back to raw continuous features
    ae_path = os.path.join(output_dir, "models", "autoencoder_model.pkl")
    if os.path.exists(ae_path):
        from ml.segmentation.autoencoder import AutoencoderWrapper
        with open(ae_path, "rb") as f:
            autoencoder = pickle.load(f)
            
        # Project all features to 16-dimensional latent space
        X_fit = autoencoder.transform(X_natural_all.values.astype(np.float32))
        logger.info(f"Projected features to latent space of shape: {X_fit.shape}")
        space_name = "Latent Representation"
    else:
        logger.warning(f"Autoencoder model not found at {ae_path}. Falling back to raw continuous features.")
        X_fit = X_continuous.values
        space_name = "Raw Continuous Features"
    
    # 3. Justify K Selection (Elbow and Silhouette analysis plots) on fitted space
    validate_k_selection(X_fit, max_k=6, output_dir=output_dir, random_seed=random_seed)
    
    # 4. Fit final K-Means++ clustering model on fitted space
    kmeans = KMeans(
        n_clusters=n_clusters,
        init="k-means++",
        random_state=random_seed,
        n_init=10
    )
    labels = kmeans.fit_predict(X_fit)
    
    # 4b. Stable Centroid Sorting: Sort clusters based on average monthly charges of assigned labels ascending
    sort_col = "numeric__MonthlyCharges"
    if sort_col in X_continuous.columns:
        logger.info(f"Applying Stable Centroid Sorting using continuous feature: {sort_col}")
        # Compute mean of sort_col per current label
        cluster_means = []
        for i in range(n_clusters):
            mean_val = X_continuous.loc[labels == i, sort_col].mean()
            # Handle potential empty cluster edge case
            if np.isnan(mean_val):
                mean_val = 0.0
            cluster_means.append((i, mean_val))
        
        # Sort by the mean value ascending
        cluster_means.sort(key=lambda x: x[1])
        sort_order = [x[0] for x in cluster_means]

        # In-place reassignment of cluster centers (reorder centroids by sort_order)
        kmeans.cluster_centers_ = kmeans.cluster_centers_[sort_order]

        # FIX MEDIUM-1: Recompute labels using predict() against the re-sorted centers.
        # Directly patching kmeans.labels_ without re-running predict() creates an
        # inconsistency between cluster_centers_ and labels_ — inertia_ would be
        # mis-attributed.  predict() recomputes nearest-centroid assignment using
        # the updated cluster_centers_, producing a fully consistent state.
        labels = kmeans.predict(X_fit)
        kmeans.labels_ = labels
    else:
        logger.warning(f"Sorting feature {sort_col} not found in continuous columns. Skipping stable centroid sorting.")
    
    # 5. Evaluate clustering quality on fitted space
    sil = float(silhouette_score(X_fit, labels, random_state=random_seed))
    db_index = float(davies_bouldin_score(X_fit, labels))
    ch_index = float(calinski_harabasz_score(X_fit, labels))
    
    logger.info(f"{space_name} Clustering -> Silhouette: {sil:.4f}, Davies-Bouldin: {db_index:.4f}")
    
    metrics = {
        "n_clusters": n_clusters,
        "silhouette_score": sil,
        "davies_bouldin_index": db_index,
        "calinski_harabasz_index": ch_index,
        "cluster_sizes": {f"cluster_{i}": int((labels == i).sum()) for i in range(n_clusters)}
    }
    
    # Save validation metrics
    metrics_path = os.path.join(output_dir, "metrics", "segmentation_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4)
        
    # Serialize KMeans model
    model_path = os.path.join(output_dir, "models", "kmeans_model.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(kmeans, f)
        
    # 6. Generate Automatic Cluster Personas Profile Summary using physical features
    generate_cluster_profiles(kmeans, X_continuous, y_natural, output_dir)
    
    return {
        "metrics": metrics,
        "model_path": model_path,
        "metrics_path": metrics_path
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    
    config_train_path = config_loader.training["data_paths"]["train_features"]
    config_artifacts_dir = config_loader.training["data_paths"]["artifacts_dir"]
    
    train_csv = config_train_path if os.path.isabs(config_train_path) else os.path.join(base_dir, config_train_path)
    artifacts_dir = config_artifacts_dir if os.path.isabs(config_artifacts_dir) else os.path.join(base_dir, config_artifacts_dir)
    
    seed = config_loader.model.get("random_seed", 42)
    
    try:
        run_segmentation(train_csv, artifacts_dir, n_clusters=3, random_seed=seed)
        print("K-Means segmentation refactor succeeded.")
    except Exception as e:
        logger.exception(f"K-Means segmentation execution failed: {e}")
        sys.exit(1)
