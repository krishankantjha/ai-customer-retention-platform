import streamlit as st
import os

def render_explainability_view(
    project_root: str,
    primary_color_hex: str,
    secondary_color_hex: str,
    danger_color_hex: str
):
    """
    Renders the Explainability & SHAP Analysis tab.
    Loads static SHAP summary and beeswarm plots and provides business definitions.
    """
    st.subheader("Model Explainability (SHAP)")
    st.write("Understand the global drivers behind customer churn predictions using SHAP (SHapley Additive exPlanations) values.")

    artifacts_dir = os.path.join(project_root, "ml", "artifacts")
    shap_summary_path = os.path.join(artifacts_dir, "shap_summary_plot.png")
    shap_beeswarm_path = os.path.join(artifacts_dir, "shap_beeswarm_plot.png")

    st.markdown("<br>", unsafe_allow_html=True)

    # Tabs for Global Importance vs Beeswarm Plots
    exp_tab1, exp_tab2, exp_tab3 = st.tabs([
        "📊 Global Feature Importance",
        "🐝 Beeswarm Value Distribution",
        "💡 Business Rules & Interpretation"
    ])

    with exp_tab1:
        st.markdown("### Mean Absolute SHAP Importance")
        st.write("This chart represents the average impact of each feature on overall churn predictions across the entire customer base.")
        
        if os.path.exists(shap_summary_path):
            st.image(shap_summary_path, caption="SHAP Global Summary Plot", use_container_width=True)
        else:
            st.info("SHAP Summary Plot not found in model artifacts. Run the model training pipeline to generate this asset.")

    with exp_tab2:
        st.markdown("### Beeswarm Impact Chart")
        st.write("This plot shows how high and low values of each feature push the churn prediction probability up or down.")
        
        if os.path.exists(shap_beeswarm_path):
            st.image(shap_beeswarm_path, caption="SHAP Beeswarm Value Distribution Plot", use_container_width=True)
        else:
            st.info("SHAP Beeswarm Plot not found in model artifacts. Run the model training pipeline to generate this asset.")

    with exp_tab3:
        st.markdown("### Business Interpretation & Insights")
        st.write("A simplified guide to how the key drivers are interpreted to direct marketing and retention plays:")

        st.markdown(f"""
        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; margin-top: 1rem;">
            <div style="background: rgba(15, 23, 42, 0.45); border: 1px solid rgba(255, 255, 255, 0.08); border-radius: 12px; padding: 1.5rem; backdrop-filter: blur(20px);">
                <div style="font-weight: 700; color: {primary_color_hex}; font-size: 1.1rem; margin-bottom: 0.5rem;">📅 Contract Type</div>
                <p style="color: #cbd5e1; font-size: 0.9rem; margin: 0; line-height: 1.5;">
                    <b>Month-to-month contracts</b> are the strongest predictor of churn. Customers on month-to-month contracts have high SHAP contributions towards churn. Transitioning customers to 1-year or 2-year contracts drastically shifts their SHAP score to stable/retained territory.
                </p>
            </div>
            <div style="background: rgba(15, 23, 42, 0.45); border: 1px solid rgba(255, 255, 255, 0.08); border-radius: 12px; padding: 1.5rem; backdrop-filter: blur(20px);">
                <div style="font-weight: 700; color: {primary_color_hex}; font-size: 1.1rem; margin-bottom: 0.5rem;">⏱️ Customer Tenure</div>
                <p style="color: #cbd5e1; font-size: 0.9rem; margin: 0; line-height: 1.5;">
                    <b>Low tenure (first 3-6 months)</b> carries high churn risk. As tenure increases, SHAP contribution decreases, representing natural customer stabilization. First-month welcoming programs and quick tech support onboarding are critical.
                </p>
            </div>
            <div style="background: rgba(15, 23, 42, 0.45); border: 1px solid rgba(255, 255, 255, 0.08); border-radius: 12px; padding: 1.5rem; backdrop-filter: blur(20px);">
                <div style="font-weight: 700; color: {primary_color_hex}; font-size: 1.1rem; margin-bottom: 0.5rem;">💸 Monthly Billing Charges</div>
                <p style="color: #cbd5e1; font-size: 0.9rem; margin: 0; line-height: 1.5;">
                    Customers with <b>extremely high monthly charges</b> without bundled ecosystem subscriptions are highly sensitive. They perceive lower value for their spend, leading the model to flag high price risk. Price auditing or loyalty discounts are suggested.
                </p>
            </div>
            <div style="background: rgba(15, 23, 42, 0.45); border: 1px solid rgba(255, 255, 255, 0.08); border-radius: 12px; padding: 1.5rem; backdrop-filter: blur(20px);">
                <div style="font-weight: 700; color: {primary_color_hex}; font-size: 1.1rem; margin-bottom: 0.5rem;">🛡️ Technical Ecosystem Lock-in</div>
                <p style="color: #cbd5e1; font-size: 0.9rem; margin: 0; line-height: 1.5;">
                    Services like <b>Online Security, Tech Support, and Online Backup</b> act as retention anchors. Customers subscribing to these options have stable baseline SHAP scores. Bundling these add-ons for at-risk users is a highly effective campaign.
                </p>
            </div>
        </div>
        """, unsafe_allow_html=True)
