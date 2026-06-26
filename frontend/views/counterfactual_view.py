import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from backend.app.database.session import SessionLocal
from backend.app.database.models.customer import Customer
from backend.app.services.prediction_service import load_artifacts
from ml.explainability.shap_local import LocalExplainer
from frontend.api_client import RetainIQAPIClient

def render_counterfactual_view(
    api_client: RetainIQAPIClient,
    check_401_callback,
    get_explanation_callback,
    primary_color_hex: str,
    secondary_color_hex: str,
    danger_color_hex: str
):
    """
    Renders the Counterfactual Simulator interface, enabling what-if analysis by 
    simulating adjustments to contracts, charges, and services to observe risk impact.
    """
    st.subheader("Counterfactual Simulator")
    st.write("Simulate 'what-if' scenarios to identify exact paths for risk reduction and revenue recovery.")

    st.markdown("<br>", unsafe_allow_html=True)

    # Initialize session states for autocomplete queries to ensure state sync
    if "counterfactual_prev_search" not in st.session_state:
        st.session_state.counterfactual_prev_search = ""
    if "counterfactual_suggestions" not in st.session_state:
        st.session_state.counterfactual_suggestions = []

    # Search for customer
    search_q = st.text_input("Search Customer ID (starts with, e.g. '7590', '9237', '5380'):", key="counterfactual_search").strip()
    
    # Clear suggestions and refresh suggestions list when search text triggers a difference
    if search_q != st.session_state.counterfactual_prev_search:
        st.session_state.counterfactual_prev_search = search_q
        if search_q:
            status_code, suggestions = api_client.search_customers(search_q)
            st.session_state.counterfactual_suggestions = suggestions if status_code == 200 else []
        else:
            st.session_state.counterfactual_suggestions = []
            
    selected_id = None
    if st.session_state.counterfactual_suggestions:
        selected_id = st.selectbox("Select Customer to Simulate:", st.session_state.counterfactual_suggestions, key="counterfactual_select")
    elif search_q:
        selected_id = search_q
            
    if not selected_id:
        st.info("Enter a Customer ID in the field above to start the simulator.")
        st.stop()

    # Load customer data from database and explanation from API
    with st.spinner("Initializing simulator environment..."):
        db = SessionLocal()
        try:
            customer = db.query(Customer).filter(Customer.customer_id == selected_id).first()
            if not customer:
                st.warning(f"No records found matching Customer ID: `{selected_id}`")
                st.stop()
            
            # Query customer explanation using the cached callback (15s TTL)
            status_code, explain_data = get_explanation_callback(st.session_state.jwt_token, selected_id)
            check_401_callback(status_code)
            
            if status_code != 200:
                st.error("Failed to load customer analytical explanation from the backend.")
                st.stop()

            # Load model artifacts for local simulation
            try:
                model_obj, preprocessor_obj, encoders_meta, metadata, explainer_obj, _ = load_artifacts()
                local_explainer = LocalExplainer(
                    model_obj,
                    metadata["feature_names_in"],
                    explainer=explainer_obj,
                    preprocessor=preprocessor_obj,
                    encoders=encoders_meta,
                    metadata=metadata,
                )
            except Exception as e:
                st.error(f"Error loading prediction models: {e}")
                st.stop()

            st.markdown("<br>", unsafe_allow_html=True)
            
            # Setup columns for simulator layout
            col_profile, col_sim, col_result = st.columns([1, 1.2, 1])
            
            with col_profile:
                st.markdown("### Current Profile")
                
                # Render current profile traits using centralized glass-card class
                st.markdown(f"""
                <div class="glass-card">
                    <div style="margin-bottom: 0.8rem; border-bottom: 1px solid rgba(255, 255, 255, 0.05); padding-bottom: 0.5rem;">
                        <span style="color: #94a3b8; font-size: 0.85rem;">Customer ID</span><br>
                        <span style="font-weight: 700; color: #f8fafc; font-size: 1.1rem;">{customer.customer_id}</span>
                    </div>
                    <div style="margin-bottom: 0.8rem;">
                        <span style="color: #94a3b8; font-size: 0.85rem;">Contract Type</span><br>
                        <span style="font-weight: 600; color: #e2e8f0;">{customer.contract}</span>
                    </div>
                    <div style="margin-bottom: 0.8rem;">
                        <span style="color: #94a3b8; font-size: 0.85rem;">Tenure</span><br>
                        <span style="font-weight: 600; color: #e2e8f0;">{customer.tenure} months</span>
                    </div>
                    <div style="margin-bottom: 0.8rem;">
                        <span style="color: #94a3b8; font-size: 0.85rem;">Monthly Charges</span><br>
                        <span style="font-weight: 600; color: #e2e8f0;">${customer.monthly_charges:.2f}</span>
                    </div>
                    <div style="margin-bottom: 0.8rem;">
                        <span style="color: #94a3b8; font-size: 0.85rem;">Tech Support</span><br>
                        <span style="font-weight: 600; color: #e2e8f0;">{customer.tech_support}</span>
                    </div>
                    <div style="margin-bottom: 0.8rem;">
                        <span style="color: #94a3b8; font-size: 0.85rem;">Online Security</span><br>
                        <span style="font-weight: 600; color: #e2e8f0;">{customer.online_security}</span>
                    </div>
                    <div style="margin-bottom: 0.8rem;">
                        <span style="color: #94a3b8; font-size: 0.85rem;">Payment Method</span><br>
                        <span style="font-weight: 600; color: #e2e8f0;">{customer.payment_method}</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # Render original risk gauge
                orig_prob = explain_data["churn_probability"]
                fig_orig = go.Figure(go.Indicator(
                    mode = "gauge+number",
                    value = orig_prob * 100,
                    number = {'suffix': "%", 'font': {'size': 24, 'color': '#ffffff'}},
                    gauge = {
                        'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "rgba(255,255,255,0.2)"},
                        'bar': {'color': danger_color_hex if orig_prob >= 0.5 else secondary_color_hex if orig_prob >= 0.25 else '#10b981'},
                        'bgcolor': "rgba(255,255,255,0.05)",
                        'borderwidth': 1,
                        'bordercolor': "rgba(255,255,255,0.1)",
                        'steps': [
                            {"range": [0, 25], "color": "rgba(16, 185, 129, 0.1)"},
                            {"range": [25, 50], "color": "rgba(245, 158, 11, 0.1)"},
                            {"range": [50, 100], "color": "rgba(239, 68, 68, 0.1)"}
                        ]
                    }
                ))
                fig_orig.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    margin=dict(t=30, b=10, l=10, r=10),
                    height=130
                )
                st.plotly_chart(fig_orig, use_container_width=True)
                
            with col_sim:
                st.markdown("### Simulate Changes")
                
                # Render controls
                sim_contract = st.selectbox(
                    "Contract Type",
                    options=["Month-to-month", "One year", "Two year"],
                    index=["Month-to-month", "One year", "Two year"].index(customer.contract) if customer.contract in ["Month-to-month", "One year", "Two year"] else 0
                )
                
                sim_tenure = st.slider(
                    "Tenure (Months)",
                    min_value=0,
                    max_value=72,
                    value=int(customer.tenure)
                )
                
                sim_charges = st.slider(
                    "Monthly Charges ($)",
                    min_value=15.0,
                    max_value=150.0,
                    value=float(customer.monthly_charges),
                    step=0.5
                )
                
                sim_support = st.selectbox(
                    "Tech Support",
                    options=["No", "Yes", "No internet service"],
                    index=["No", "Yes", "No internet service"].index(customer.tech_support) if customer.tech_support in ["No", "Yes", "No internet service"] else 0
                )
                
                sim_security = st.selectbox(
                    "Online Security",
                    options=["No", "Yes", "No internet service"],
                    index=["No", "Yes", "No internet service"].index(customer.online_security) if customer.online_security in ["No", "Yes", "No internet service"] else 0
                )
                
                sim_billing = st.selectbox(
                    "Paperless Billing",
                    options=["No", "Yes"],
                    index=["No", "Yes"].index(customer.paperless_billing) if customer.paperless_billing in ["No", "Yes"] else 0
                )
                
                sim_payment = st.selectbox(
                    "Payment Method",
                    options=["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
                    index=["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"].index(customer.payment_method) if customer.payment_method in ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"] else 0
                )

            with col_result:
                st.markdown("### Simulation Result")
                
                # Execute simulation
                customer_dict = {
                    "customerID": customer.customer_id,
                    "gender": customer.gender,
                    "SeniorCitizen": customer.senior_citizen,
                    "Partner": customer.partner,
                    "Dependents": customer.dependents,
                    "tenure": sim_tenure,
                    "PhoneService": customer.phone_service,
                    "MultipleLines": customer.multiple_lines,
                    "InternetService": customer.internet_service,
                    "OnlineSecurity": sim_security,
                    "OnlineBackup": customer.online_backup,
                    "DeviceProtection": customer.device_protection,
                    "TechSupport": sim_support,
                    "StreamingTV": customer.streaming_tv,
                    "StreamingMovies": customer.streaming_movies,
                    "Contract": sim_contract,
                    "PaperlessBilling": sim_billing,
                    "PaymentMethod": sim_payment,
                    "MonthlyCharges": sim_charges,
                    "TotalCharges": sim_charges * sim_tenure,
                    "Churn": customer.churn
                }
                customer_df = pd.DataFrame([customer_dict])
                
                sim_prob = local_explainer.simulate_intervention(customer_df, {})
                risk_reduction = orig_prob - sim_prob
                annual_saved_revenue = max(0.0, risk_reduction) * sim_charges * 12
                
                # Render simulated gauge
                fig_sim = go.Figure(go.Indicator(
                    mode = "gauge+number",
                    value = sim_prob * 100,
                    number = {'suffix': "%", 'font': {'size': 32, 'color': '#ffffff'}},
                    gauge = {
                        'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "rgba(255,255,255,0.2)"},
                        'bar': {'color': '#10b981' if sim_prob < 0.25 else secondary_color_hex if sim_prob < 0.5 else danger_color_hex},
                        'bgcolor': "rgba(255,255,255,0.05)",
                        'borderwidth': 1,
                        'bordercolor': "rgba(255,255,255,0.1)",
                        'steps': [
                            {"range": [0, 25], "color": "rgba(16, 185, 129, 0.1)"},
                            {"range": [25, 50], "color": "rgba(245, 158, 11, 0.1)"},
                            {"range": [50, 100], "color": "rgba(239, 68, 68, 0.1)"}
                        ]
                    }
                ))
                fig_sim.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    margin=dict(t=30, b=10, l=10, r=10),
                    height=180
                )
                st.plotly_chart(fig_sim, use_container_width=True)
                
                # Render metrics comparison card using centralized glass-card class
                status_class = "LOW RISK" if sim_prob < 0.25 else "MEDIUM RISK" if sim_prob < 0.5 else "HIGH RISK"
                status_color = "#10b981" if sim_prob < 0.25 else secondary_color_hex if sim_prob < 0.5 else danger_color_hex
                
                st.markdown(f"""
                <div class="glass-card" style="text-align: center !important;">
                    <div style="font-size: 1.2rem; font-weight: 800; color: {status_color}; text-transform: uppercase; margin-bottom: 0.8rem;">{status_class}</div>
                    
                    <div style="display: flex; justify-content: space-around; margin-bottom: 1.2rem; border-top: 1px solid rgba(255, 255, 255, 0.05); border-bottom: 1px solid rgba(255, 255, 255, 0.05); padding: 0.8rem 0;">
                        <div>
                            <span style="color: #94a3b8; font-size: 0.78rem;">Risk Reduced</span><br>
                            <span style="font-weight: 700; color: #10b981; font-size: 1.1rem;">{max(0.0, risk_reduction)*100:.1f}%</span>
                        </div>
                        <div>
                            <span style="color: #94a3b8; font-size: 0.78rem;">Original Risk</span><br>
                            <span style="font-weight: 600; color: #cbd5e1; font-size: 1.0rem;">{orig_prob*100:.1f}%</span>
                        </div>
                    </div>
                    
                    <div>
                        <span style="color: #94a3b8; font-size: 0.8rem;">Estimated Revenue Saved</span><br>
                        <span style="font-weight: 800; color: #10b981; font-size: 1.6rem;">${annual_saved_revenue:,.2f}</span>
                        <span style="color: #94a3b8; font-size: 0.75rem;">/ year</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Internal error executing simulator: {e}")
        finally:
            db.close()
