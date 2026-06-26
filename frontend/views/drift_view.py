import streamlit as st
import pandas as pd

def render_drift_view(
    model_health: dict,
    primary_color_hex: str,
    secondary_color_hex: str,
    danger_color_hex: str
):
    """
    Renders the Drift Detection page.
    Displays health banners, KS numeric drift stats, and Chi-Square/PSI categorical stats.
    """
    st.subheader("Data Drift & Health Monitoring")
    st.write("Monitor operational feature distribution stability to detect drift between incoming customer cohorts and training baseline populations.")

    if not model_health or "status" not in model_health:
        st.warning("⚠️ Could not load model health and drift statistics.")
        st.stop()

    status = model_health.get("status", "Healthy")
    msg = model_health.get("message", "")
    drift_ratio = model_health.get("drift_ratio", 0.0)
    drift_details = model_health.get("drift_details", {})

    st.markdown("<br>", unsafe_allow_html=True)

    # 1. Health Banner with vibrant status glow
    if status == "Healthy":
        st.markdown(f"""
        <div class="health-banner health-banner-success">
            <div style="font-weight: 800; font-size: 1.3rem; color: #10b981; font-family: 'Outfit', sans-serif;">🩺 SYSTEM STATUS: HEALTHY</div>
            <div style="color: #e2e8f0; font-size: 0.92rem; margin-top: 4px; line-height: 1.4;">{msg}</div>
        </div>
        """, unsafe_allow_html=True)
    elif status == "Warning":
        st.markdown(f"""
        <div class="health-banner health-banner-warning">
            <div style="font-weight: 800; font-size: 1.3rem; color: #f59e0b; font-family: 'Outfit', sans-serif;">⚠️ SYSTEM STATUS: WARNING (DRIFT DETECTED)</div>
            <div style="color: #e2e8f0; font-size: 0.92rem; margin-top: 4px; line-height: 1.4;">{msg}</div>
        </div>
        """, unsafe_allow_html=True)
    else:  # Degraded
        st.markdown(f"""
        <div class="health-banner health-banner-danger">
            <div style="font-weight: 800; font-size: 1.3rem; color: {danger_color_hex}; font-family: 'Outfit', sans-serif;">🚨 SYSTEM STATUS: DEGRADED</div>
            <div style="color: #e2e8f0; font-size: 0.92rem; margin-top: 4px; line-height: 1.4;">{msg}</div>
        </div>
        """, unsafe_allow_html=True)

    # Stat Row Cards
    s_col1, s_col2, s_col3 = st.columns(3)
    with s_col1:
        st.markdown(f"""
        <div class="glass-card" style="padding: 1.2rem !important; margin-bottom: 0px !important;">
            <span style="color: #94a3b8; font-size: 0.85rem; font-weight: 500; text-transform: uppercase;">Classifier Name</span><br>
            <span style="font-weight: 700; color: #f8fafc; font-size: 1.25rem; font-family: 'Outfit', sans-serif;">{model_health.get('model_name', 'GBDT Ensemble')}</span>
        </div>
        """, unsafe_allow_html=True)
    with s_col2:
        st.markdown(f"""
        <div class="glass-card" style="padding: 1.2rem !important; margin-bottom: 0px !important;">
            <span style="color: #94a3b8; font-size: 0.85rem; font-weight: 500; text-transform: uppercase;">Active Version</span><br>
            <span style="font-weight: 700; color: #f8fafc; font-size: 1.25rem; font-family: 'Outfit', sans-serif;">{model_health.get('model_version', '1.1.0')}</span>
        </div>
        """, unsafe_allow_html=True)
    with s_col3:
        st.markdown(f"""
        <div class="glass-card" style="padding: 1.2rem !important; margin-bottom: 0px !important;">
            <span style="color: #94a3b8; font-size: 0.85rem; font-weight: 500; text-transform: uppercase;">Drift Ratio</span><br>
            <span style="font-weight: 700; color: {danger_color_hex if status == 'Degraded' else '#f59e0b' if status == 'Warning' else '#10b981'}; font-size: 1.25rem; font-family: 'Outfit', sans-serif;">{drift_ratio*100:.1f}%</span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br><br>", unsafe_allow_html=True)

    # Process metrics into numeric and categorical splits
    numeric_rows = []
    categorical_rows = []

    for f_name, details in drift_details.items():
        clean_feat = f_name.replace("numeric__", "").replace("binary__", "").replace("categorical__", "").replace("ordinal__", "").replace("_", " ").title()
        method = details.get("method", "ks_test")
        is_drifted = details.get("drifted", False)
        status_str = "🔴 Drifted" if is_drifted else "🟢 Stable"
        
        if method == "ks_test":
            numeric_rows.append({
                "Feature Name": clean_feat,
                "KS Statistic": f"{details.get('ks_statistic', 0.0):.4f}",
                "p-Value": f"{details.get('p_value', 1.0):.6f}",
                "Status": status_str
            })
        else:
            categorical_rows.append({
                "Feature Name": clean_feat,
                "Chi-Square Stat": f"{details.get('chi2_statistic', 0.0):.3f}",
                "p-Value": f"{details.get('p_value', 1.0):.6f}",
                "PSI (Population Stability Index)": f"{details.get('psi', 0.0):.4f}",
                "Status": status_str
            })

    d_tab1, d_tab2 = st.tabs(["📈 Numerical Feature Drift (KS Test)", "📊 Categorical Feature Drift (PSI & Chi2)"])

    with d_tab1:
        st.write("**Methodology:** Runs the two-sample Kolmogorov-Smirnov (KS) test comparing continuous variables in current database records vs baseline training set features ($p < 0.05$ flags significant distribution divergence).")
        if numeric_rows:
            st.table(pd.DataFrame(numeric_rows).set_index("Feature Name"))
        else:
            st.info("No continuous numeric features checked.")

    with d_tab2:
        st.write("**Methodology:** Calculates Chi-Square Goodness-of-Fit and Population Stability Index (PSI). PSI measures magnitude changes between categorical bins ($PSI \\ge 0.25$ indicates significant structural shift).")
        if categorical_rows:
            st.table(pd.DataFrame(categorical_rows).set_index("Feature Name"))
        else:
            st.info("No categorical features checked.")
