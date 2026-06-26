import streamlit as st
import pandas as pd
import os
import json

def render_diagnostics_view(
    diagnostics_metadata: dict,
    model_health: dict,
    cohort_data: list,
    project_root: str,
    primary_color_hex: str,
    secondary_color_hex: str,
    danger_color_hex: str
):
    """
    Renders the Model Diagnostics view.
    Includes validation curves (ROC, PR, Calibration, cost Curves), confusion matrix, benchmarks, and t-tests.
    """
    st.subheader("Model Diagnostics & Performance")
    st.write("Review calibrated validation metrics, model calibration diagrams, cost thresholds, confusion matrix, and artifact checksum diagnostics.")

    artifacts_dir = os.path.join(project_root, "ml", "artifacts")
    
    # Version Drift Alert
    if diagnostics_metadata:
        if diagnostics_metadata.get("drift_detected", False):
            st.warning(
                "⚠️ **Diagnostics Version Drift Detected!**  \n"
                "The static diagnostic plots displayed below represent an older model version and are "
                "out of sync with the currently active classifier binary. Please regenerate evaluation assets "
                "via the training pipeline script (`evaluate.py`) to reconcile this difference.  \n"
                f"- **Active Model Binary SHA-256**: `{diagnostics_metadata.get('actual_model_sha256', 'unknown')[:16]}...`  \n"
                f"- **Diagnostics Target SHA-256**: `{diagnostics_metadata.get('model_sha256', 'unknown')[:16]}...`"
            )
    else:
        st.warning("⚠️ Could not load diagnostics metadata. Drift check skipped.")

    st.markdown("<br>", unsafe_allow_html=True)

    diag_tab1, diag_tab2, diag_tab3, diag_tab4 = st.tabs([
        "📈 Predictive Performance (ROC & PR)", 
        "⚖️ Probability Calibration", 
        "🧩 Confusion Matrix",
        "🩺 Artifact Verification"
    ])

    with diag_tab1:
        st.markdown("### Champion Model Performance Curves")
        st.write("Holdout test dataset evaluation for the soft-voting GBDT ensemble.")
        
        c1, c2 = st.columns(2)
        roc_path = os.path.join(artifacts_dir, "roc_curve.png")
        pr_path = os.path.join(artifacts_dir, "precision_recall_curve.png")
        
        with c1:
            if os.path.exists(roc_path):
                st.image(roc_path, caption="ROC Curve (Holdout set)", use_container_width=True)
            else:
                st.info("ROC curve plot not found in artifacts.")
        with c2:
            if os.path.exists(pr_path):
                st.image(pr_path, caption="Precision-Recall Curve (Holdout set)", use_container_width=True)
            else:
                st.info("Precision-Recall curve plot not found in artifacts.")

        st.markdown("---")
        st.markdown("### Cross-Validation Benchmarking")
        bench_path = os.path.join(artifacts_dir, "metrics", "benchmark_results.csv")
        if os.path.exists(bench_path):
            try:
                bench_df = pd.read_csv(bench_path)
                st.dataframe(bench_df, use_container_width=True, hide_index=True)
            except Exception as e:
                st.warning(f"Failed to read benchmark results: {e}")

        st.markdown("---")
        st.markdown("### Paired t-Test Significance comparison")
        stat_results_path = os.path.join(artifacts_dir, "metrics", "statistical_results.json")
        if os.path.exists(stat_results_path):
            try:
                with open(stat_results_path, "r") as f:
                    stats = json.load(f)
                
                sc1, sc2, sc3 = st.columns(3)
                with sc1:
                    st.metric("5x2cv t-statistic", f"{stats.get('t_statistic', 0.0):.4f}")
                with sc2:
                    st.metric("p-value", f"{stats.get('p_value', 0.0):.4f}")
                with sc3:
                    is_sig = stats.get("p_value", 1.0) < 0.05
                    st.metric("Significant Advantage?", "Yes" if is_sig else "No")
                    
                st.write(f"**Interpretation**: {stats.get('interpretation', 'No interpretation log available.')}")
            except Exception as e:
                st.warning(f"Failed to read statistical results: {e}")

    with diag_tab2:
        st.markdown("### Probability Calibration & Threshold Curves")
        st.write("Reliability curves representing observed vs predicted frequencies, and cost-benefit optimization bounds.")
        
        cc1, cc2 = st.columns(2)
        cal_path = os.path.join(artifacts_dir, "calibration_comparison.png")
        thresh_path = os.path.join(artifacts_dir, "threshold_optimization.png")
        
        with cc1:
            if os.path.exists(cal_path):
                st.image(cal_path, caption="Probability Reliability Curves", use_container_width=True)
            else:
                st.info("Calibration reliability curve plot not found in artifacts.")
        with cc2:
            if os.path.exists(thresh_path):
                st.image(thresh_path, caption="Asymmetric Business Cost Minimization Curves", use_container_width=True)
            else:
                st.info("Threshold optimization plot not found in artifacts.")

        st.markdown("---")
        st.markdown("### Calibration Score metrics")
        cal_metrics_path = os.path.join(artifacts_dir, "calibration", "calibration_metrics.json")
        if os.path.exists(cal_metrics_path):
            try:
                with open(cal_metrics_path, "r") as f:
                    cal_data = json.load(f)
                
                cal_rows = []
                for model_name, m_vals in cal_data.items():
                    cal_rows.append({
                        "Calibration Scheme": model_name.replace("_", " ").title(),
                        "Brier Score (lower is better)": f"{m_vals.get('Brier_Score', 0.0):.4f}",
                        "Expected Calibration Error (ECE)": f"{m_vals.get('ECE', 0.0):.4f}"
                    })
                st.table(pd.DataFrame(cal_rows).set_index("Calibration Scheme"))
            except Exception as e:
                st.warning(f"Failed to load calibration metrics: {e}")

    with diag_tab3:
        st.markdown("### Confusion Matrix")
        st.write("Visualize the ratio of True Positives, True Negatives, False Positives, and False Negatives on holdout validation data.")
        
        cm_path = os.path.join(artifacts_dir, "confusion_matrix.png")
        if os.path.exists(cm_path):
            st.image(cm_path, caption="Confusion Matrix heatmap", width=480)
        else:
            st.info("Confusion matrix image file not found in model artifacts.")

    with diag_tab4:
        st.markdown("### Diagnostic Metadata & Checksums")
        st.write("Ensure production models are safe and verified against validation manifest registers.")
        
        if diagnostics_metadata:
            meta_rows = [
                {"Parameter": "Artifact Version", "Value": diagnostics_metadata.get("model_version", "1.1.0")},
                {"Parameter": "Evaluation Date", "Value": diagnostics_metadata.get("evaluation_timestamp", "N/A")},
                {"Parameter": "Expected File SHA-256", "Value": diagnostics_metadata.get("model_sha256", "N/A")},
                {"Parameter": "Active File SHA-256", "Value": diagnostics_metadata.get("actual_model_sha256", "N/A")},
                {"Parameter": "Verification Status", "Value": "🟢 Verified (Matches)" if not diagnostics_metadata.get("drift_detected", False) else "🔴 Drift/Mismatch"}
            ]
            st.table(pd.DataFrame(meta_rows).set_index("Parameter"))
        else:
            st.info("No diagnostics metadata register loaded.")
