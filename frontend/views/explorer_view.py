import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from frontend.api_client import RetainIQAPIClient

def render_explorer_view(
    api_client: RetainIQAPIClient,
    check_401_callback,
    get_explanation_callback,
    primary_color_hex: str,
    secondary_color_hex: str,
    danger_color_hex: str
):
    """
    Renders the Single Customer Risk Profiler, enabling customer lookup via prefix 
    autocomplete, and visualizes local SHAP risk attributions and tailored save play plays.
    """
    st.subheader("Customer Profile Explorer")
    st.write("Search and analyze individual customer churn risk metrics, behavioral demographics, and localized SHAP drivers.")

    st.markdown("<br>", unsafe_allow_html=True)
    
    # Initialize session states for autocomplete queries to ensure state sync
    if "explorer_prev_search" not in st.session_state:
        st.session_state.explorer_prev_search = ""
    if "explorer_suggestions" not in st.session_state:
        st.session_state.explorer_suggestions = []

    # Autocomplete Search Field
    search_q = st.text_input("Search Customer ID (starts with, e.g. '7590', '9237', '5380'):", key="explorer_search_input").strip()
    
    # Clear suggestions and refresh suggestions list when search text triggers a difference
    if search_q != st.session_state.explorer_prev_search:
        st.session_state.explorer_prev_search = search_q
        if search_q:
            status_code, suggestions = api_client.search_customers(search_q)
            st.session_state.explorer_suggestions = suggestions if status_code == 200 else []
        else:
            st.session_state.explorer_suggestions = []
            
    selected_id = None
    if st.session_state.explorer_suggestions:
        selected_id = st.selectbox("Matching Customer IDs:", st.session_state.explorer_suggestions, key="explorer_search_selectbox")
    elif search_q:
        selected_id = search_q
            
    if not selected_id:
        st.info("Enter a Customer ID prefix in the field above to query account profiles.")
        st.stop()
        
    with st.spinner("Retrieving customer details & SHAP attributions..."):
        try:
            # Query customer details via the cached explanation callback (15s TTL)
            response_code, data = get_explanation_callback(st.session_state.jwt_token, selected_id)
            check_401_callback(response_code)
            
            if response_code == 404:
                st.warning(f"No records found matching Customer ID: `{selected_id}`")
            elif response_code == 200:
                st.markdown("<br>", unsafe_allow_html=True)
                
                # Layout Setup: Demographics & Risk Score
                col_info, col_gauge = st.columns([1.4, 1])
                
                with col_info:
                    st.markdown("### Profile Demographics")
                    
                    # Using centralized glass-card and profile-grid classes
                    st.markdown(f"""
                    <div class="glass-card profile-grid">
                        <div>
                            <span style="color: #94a3b8; font-size: 0.8rem;">Gender</span><br>
                            <span style="font-weight: 600; color: #f8fafc; font-size: 0.95rem;">{data['gender']}</span>
                        </div>
                        <div>
                            <span style="color: #94a3b8; font-size: 0.8rem;">Tenure</span><br>
                            <span style="font-weight: 600; color: #f8fafc; font-size: 0.95rem;">{data['tenure']} months</span>
                        </div>
                        <div>
                            <span style="color: #94a3b8; font-size: 0.8rem;">Monthly Billing</span><br>
                            <span style="font-weight: 600; color: #f8fafc; font-size: 0.95rem;">${data['monthly_charges']:.2f} / mo</span>
                        </div>
                        <div>
                            <span style="color: #94a3b8; font-size: 0.8rem;">Total Billing Charges</span><br>
                            <span style="font-weight: 600; color: #f8fafc; font-size: 0.95rem;">${data['total_charges']:.2f}</span>
                        </div>
                        <div style="grid-column: span 2; border-top: 1px solid rgba(255, 255, 255, 0.05); padding-top: 0.8rem; margin-top: 0.4rem;">
                            <span style="color: #94a3b8; font-size: 0.8rem;">Customer Segment Persona</span><br>
                            <span style="font-weight: 700; color: {primary_color_hex}; font-size: 1.0rem;">{data.get('cohort_persona') or 'Unassigned Core Client'}</span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                with col_gauge:
                    st.markdown("<h3 style='text-align:center;'>Churn Risk Score</h3>", unsafe_allow_html=True)
                    prob = data["churn_probability"]
                    risk_class = "HIGH RISK" if data["is_high_risk"] else "NORMAL RISK"
                    risk_color = danger_color_hex if data["is_high_risk"] else "#10b981"
                    
                    fig = go.Figure(go.Indicator(
                        mode = "gauge+number",
                        value = prob * 100,
                        number = {'suffix': "%", 'font': {'size': 36, 'color': '#ffffff'}},
                        gauge = {
                            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "rgba(255,255,255,0.2)"},
                            'bar': {'color': risk_color},
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
                    fig.update_layout(
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)",
                        margin=dict(t=20, b=10, l=30, r=30),
                        height=160
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    st.markdown(f"<h4 style='text-align:center;color:{risk_color};letter-spacing:1px;font-family:Outfit;'>{risk_class}</h4>", unsafe_allow_html=True)

                st.markdown("<br><br>", unsafe_allow_html=True)
                
                # SHAP drivers + Save Plays layout
                col_drivers, col_plays = st.columns([1.2, 1])
                
                with col_drivers:
                    st.markdown("### Top Churn Risk Drivers (SHAP)")
                    drivers_list = data["top_drivers"]
                    if drivers_list:
                        drivers_df = pd.DataFrame(drivers_list)
                        drivers_df["display_feature"] = drivers_df["feature"]\
                            .str.replace("numeric__", "")\
                            .str.replace("binary__", "")\
                            .str.replace("categorical__", "")\
                            .str.replace("ordinal__", "")\
                            .str.replace("_", " ")\
                            .str.title()
                            
                        fig_drivers = px.bar(
                            drivers_df,
                            x="shap_value",
                            y="display_feature",
                            orientation="h",
                            labels={"shap_value": "Risk Contribution Score (SHAP)", "display_feature": "Feature Factor"},
                            color="shap_value",
                            color_continuous_scale="Reds",
                        )
                        fig_drivers.update_layout(
                            template="plotly_dark",
                            paper_bgcolor="rgba(0,0,0,0)",
                            plot_bgcolor="rgba(0,0,0,0)",
                            font=dict(family="Inter, sans-serif", color="#94a3b8"),
                            margin=dict(t=10, b=10, l=10, r=10),
                            height=260,
                            coloraxis_showscale=False
                        )
                        fig_drivers.update_traces(
                            marker_line_color="rgba(255,255,255,0.05)",
                            marker_line_width=1
                        )
                        st.plotly_chart(fig_drivers, use_container_width=True)
                    else:
                        st.info("This customer demonstrates minimal risk factors (no positive SHAP contributions).")

                with col_plays:
                    st.markdown("### Recommended Save Plays")
                    plays_list = data["save_plays"]
                    if plays_list:
                        for play in plays_list:
                            clean_feat = play.get("feature", "General Churn Risk")\
                                .replace("numeric__", "")\
                                .replace("binary__", "")\
                                .replace("categorical__", "")\
                                .replace("ordinal__", "")\
                                .replace("_", " ")\
                                .title()
                                
                            impact_val = play.get("estimated_impact", 0.0) * 100
                            
                            # Using centralized play-card and play-card-header classes
                            st.markdown(f"""
                            <div class="play-card" style="border-left-color: {primary_color_hex} !important;">
                                <div class="play-card-header">
                                    <span class="play-card-title">🎯 {play['campaign']}</span>
                                    <span class="status-badge status-badge-success">-{impact_val:.1f}% risk</span>
                                </div>
                                <p style="margin: 0; font-size: 0.82rem; color: #94a3b8;"><b>Trigger Factor:</b> {clean_feat}</p>
                                <p style="margin: 0.4rem 0 0 0; font-size: 0.88rem; color: #e2e8f0; line-height: 1.4;"><b>Action:</b> {play['action'] if 'action' in play else play.get('recommendation', '')}</p>
                            </div>
                            """, unsafe_allow_html=True)
                    else:
                        st.info("No retention plays mapped for low risk profiles.")

            else:
                st.error(f"Error retrieving explanation: {data.get('detail', 'Unknown error')}")
        except Exception as e:
            st.error(f"Network connection failed: {e}")
