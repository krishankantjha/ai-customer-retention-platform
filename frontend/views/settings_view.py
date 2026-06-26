import streamlit as st

def render_settings_view(
    primary_color_hex: str,
    secondary_color_hex: str,
    danger_color_hex: str
):
    """
    Renders the Settings & Preferences view.
    Includes Profile forms, Theme/Language preferences, Alert thresholds, and API access placeholders.
    """
    st.subheader("Console Settings")
    st.write("Manage your console user profile, alert configurations, API credentials, and application preferences.")

    # Initialize settings defaults in session state if not set, or if user changed
    current_user = st.session_state.get("current_user") or "Admin User"
    default_email = f"{current_user.lower()}@retainiq.com" if current_user != "Admin User" else "admin@retainiq.com"
    
    if "settings_profile_user" not in st.session_state or st.session_state.settings_profile_user != current_user:
        st.session_state.settings_profile_user = current_user
        st.session_state.settings_profile_name = current_user
        st.session_state.settings_profile_email = default_email

    if "settings_alert_threshold" not in st.session_state:
        st.session_state.settings_alert_threshold = 50

    st.markdown("<br>", unsafe_allow_html=True)

    set_tab1, set_tab2, set_tab3, set_tab4 = st.tabs([
        "👤 Profile Information",
        "⚙️ Application Preferences",
        "🔔 Notifications & Alerts",
        "🔑 Security & API Keys"
    ])

    with set_tab1:
        st.markdown("### Profile Information")
        with st.form("profile_settings_form"):
            prof_name = st.text_input("Full Name", value=st.session_state.settings_profile_name)
            prof_email = st.text_input("Email Address", value=st.session_state.settings_profile_email)
            prof_role = st.text_input("Role", value="Administrator", disabled=True)
            prof_dept = st.text_input("Department", value="Customer Success / Operations", disabled=True)
            
            save_prof = st.form_submit_button("Update Profile Profile", type="primary")
            if save_prof:
                st.session_state.settings_profile_name = prof_name
                st.session_state.settings_profile_email = prof_email
                st.toast("Profile settings updated successfully!", icon="👤")

    with set_tab2:
        st.markdown("### Application Preferences")
        pref_theme = st.selectbox("Interface Theme", ["Dark Theme (Default)", "Light Theme"])
        pref_lang = st.selectbox("Language", ["English (US)", "Spanish", "German", "French"])
        pref_timezone = st.selectbox("Timezone", ["Asia/Kolkata", "UTC", "America/New_York", "Europe/London"])
        pref_date = st.selectbox("Date Format", ["DD/MM/YYYY", "YYYY-MM-DD", "MM/DD/YYYY"])
        
        if st.button("Save Preferences", key="save_pref_btn"):
            st.toast("App preferences updated! (Theme stays Dark as mandated by design system)", icon="⚙️")

    with set_tab3:
        st.markdown("### Notifications & Churn Alert Thresholds")
        st.write("Configure threshold levels to trigger real-time high-risk notifications.")
        
        alert_email = st.toggle("Email Churn Alerts", value=True)
        alert_thresh = st.slider(
            "High Churn Risk Trigger Threshold (%)",
            min_value=10,
            max_value=90,
            value=st.session_state.settings_alert_threshold,
            help="Customers exceeding this churn probability score will be auto-flagged in the Explorer and Dashboard as High Risk."
        )
        alert_webhook = st.text_input("Webhook Destination URL", placeholder="https://api.company.com/v1/churn-webhooks")
        
        if st.button("Save Alert Settings", key="save_alert_btn"):
            st.session_state.settings_alert_threshold = alert_thresh
            st.toast("Alert notification settings saved!", icon="🔔")

    with set_tab4:
        st.markdown("### Security Credentials")
        st.write("Verify secure API connections or configure programmatic script keys.")
        
        st.markdown("""
        <div style="background: rgba(15, 23, 42, 0.45); border: 1px solid rgba(255, 255, 255, 0.08); border-radius: 12px; padding: 1.5rem; backdrop-filter: blur(20px); margin-bottom: 1.5rem;">
            <div style="font-weight: 700; color: #f8fafc; font-size: 0.95rem; margin-bottom: 0.5rem;">Programmatic JWT Token</div>
            <div style="font-family: monospace; background: rgba(0,0,0,0.3); padding: 0.8rem; border-radius: 8px; border: 1px solid rgba(255,255,255,0.05); color: #8b5cf6; font-size: 0.82rem; overflow-x: auto; white-space: nowrap; margin-bottom: 1rem;">
                eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJuYW1lIjoiQWRtaW4iLCJyb2xlIjoiYWRtaW4iLCJleHAiOjE4MDAwfQ...
            </div>
            <p style="font-size: 0.78rem; color: #94a3b8; margin: 0;">Copy this JWT token to authenticate programmatic command-line clients or curl pipelines.</p>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("Regenerate Token Credentials", key="regen_token_btn"):
            st.toast("Credentials regenerated successfully!", icon="🔑")
