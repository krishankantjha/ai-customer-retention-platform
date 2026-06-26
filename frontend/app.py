import streamlit as st
import time
import os
import sys

# Add project root to path to load configs
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
from configs.dataset_config import config_loader
from frontend.api_client import RetainIQAPIClient

# Import modular sub-views
from frontend.views.auth_view import render_auth_view
from frontend.views.dashboard_view import render_dashboard_view
from frontend.views.executive_view import render_executive_view
from frontend.views.diagnostics_view import render_diagnostics_view
from frontend.views.ingestion_view import render_ingestion_view
from frontend.views.explorer_view import render_explorer_view
from frontend.views.counterfactual_view import render_counterfactual_view
from frontend.views.explainability_view import render_explainability_view
from frontend.views.segments_view import render_segments_view
from frontend.views.drift_view import render_drift_view
from frontend.views.settings_view import render_settings_view

# Fetch dynamic styling tokens from config
theme_cfg = config_loader.dashboard.get("theme", {})
primary_color_hex = theme_cfg.get("primary_color", "#6366F1").lower()
secondary_color_hex = theme_cfg.get("secondary_color", "#F97316").lower()
danger_color_hex = theme_cfg.get("danger_color", "#EF4444").lower()

# Instantiate class-based API client
api_client = RetainIQAPIClient()

st.set_page_config(
    page_title="RetainIQ — Customer Churn Dashboard",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom premium styling rules
st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Outfit:wght@600;700;800&display=swap');
    
    /* Global layout settings and ambient grid background */
    .stApp {{
        background-color: #030014 !important;
        background-image: 
            radial-gradient(circle at 12% 18%, rgba(99, 102, 241, 0.12), transparent 40%),
            radial-gradient(circle at 88% 82%, rgba(139, 92, 246, 0.12), transparent 40%),
            radial-gradient(circle at 50% 50%, rgba(2, 6, 23, 0.95), transparent 90%),
            linear-gradient(rgba(255, 255, 255, 0.008) 1px, transparent 1px),
            linear-gradient(90deg, rgba(255, 255, 255, 0.008) 1px, transparent 1px) !important;
        background-size: 100% 100%, 100% 100%, 100% 100%, 36px 36px, 36px 36px !important;
        background-attachment: fixed !important;
        color: #f8fafc !important;
        font-family: 'Inter', sans-serif !important;
    }}

    /* Title blocks */
    .main-title {{
        font-family: 'Outfit', sans-serif;
        font-weight: 800;
        font-size: 2.8rem;
        background: linear-gradient(135deg, {primary_color_hex} 0%, #a78bfa 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
        letter-spacing: -0.8px;
    }}
    .subtitle {{
        font-family: 'Inter', sans-serif;
        color: #94a3b8;
        font-size: 1.05rem;
        margin-bottom: 2rem;
        line-height: 1.5;
    }}

    /* Glassmorphism containers and cards */
    .glass-card {{
        background: rgba(15, 23, 42, 0.45) !important;
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
        border-radius: 12px !important;
        padding: 1.5rem !important;
        backdrop-filter: blur(20px) !important;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3) !important;
        margin-bottom: 1rem !important;
    }}

    /* Native metrics styling overrides */
    div[data-testid="metric-container"] {{
        background: rgba(15, 23, 42, 0.45) !important;
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
        border-radius: 12px !important;
        padding: 1.2rem 1.5rem !important;
        backdrop-filter: blur(20px) !important;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.2) !important;
    }}
    div[data-testid="metric-container"] label {{
        color: #94a3b8 !important;
        font-weight: 600 !important;
        text-transform: uppercase !important;
        letter-spacing: 0.05em !important;
        font-family: 'Inter', sans-serif !important;
        font-size: 0.78rem !important;
    }}
    div[data-testid="metric-container"] div[data-testid="stMetricValue"] {{
        font-weight: 800 !important;
        font-size: 1.9rem !important;
        color: #ffffff !important;
        font-family: 'Outfit', sans-serif !important;
    }}

    /* Sidebar glass effect and overrides */
    section[data-testid="stSidebar"] {{
        background-color: rgba(10, 11, 16, 0.8) !important;
        border-right: 1px solid rgba(255, 255, 255, 0.08) !important;
        backdrop-filter: blur(20px) !important;
    }}
    
    /* Navigation radio button makeover */
    div[data-testid="stSidebar"] div[data-testid="stRadio"] > div {{
        display: flex !important;
        flex-direction: column !important;
    }}
    div[data-testid="stSidebar"] div[data-testid="stRadio"] label {{
        background: rgba(255, 255, 255, 0.02) !important;
        border: 1px solid rgba(255, 255, 255, 0.04) !important;
        border-radius: 8px !important;
        padding: 0.6rem 0.9rem !important;
        color: #cbd5e1 !important;
        margin-bottom: 0.4rem !important;
        transition: all 0.2s ease-in-out !important;
        cursor: pointer !important;
        display: flex !important;
        align-items: center !important;
    }}
    div[data-testid="stSidebar"] div[data-testid="stRadio"] label:hover {{
        background: rgba(99, 102, 241, 0.12) !important;
        border-color: rgba(99, 102, 241, 0.25) !important;
        color: #ffffff !important;
    }}
    div[data-testid="stSidebar"] div[data-testid="stRadio"] label[data-checked="true"] {{
        background: linear-gradient(135deg, rgba(99, 102, 241, 0.25) 0%, rgba(99, 102, 241, 0.1) 100%) !important;
        border-color: {primary_color_hex} !important;
        color: #ffffff !important;
        font-weight: 600 !important;
        box-shadow: 0 4px 12px rgba(99, 102, 241, 0.15) !important;
    }}
    /* Hide the default radio circle inputs */
    div[data-testid="stSidebar"] div[data-testid="stRadio"] label div:first-child {{
        display: none !important;
    }}
    div[data-testid="stSidebar"] div[data-testid="stRadio"] label div[data-testid="stMarkdownContainer"] {{
        margin-left: 0 !important;
    }}

    /* Global inputs and selectboxes */
    div[data-baseweb="select"] > div, input {{
        background-color: rgba(15, 23, 42, 0.6) !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 8px !important;
        color: #f8fafc !important;
    }}
    div[data-baseweb="select"] > div:hover, input:hover {{
        border-color: rgba(255, 255, 255, 0.2) !important;
    }}

    /* Tabs Styling */
    button[data-baseweb="tab"] {{
        color: #94a3b8 !important;
        font-weight: 600 !important;
        font-family: 'Inter', sans-serif !important;
    }}
    button[data-baseweb="tab"][aria-selected="true"] {{
        color: #f8fafc !important;
        border-bottom-color: {primary_color_hex} !important;
    }}

    /* Custom scrollbars */
    ::-webkit-scrollbar {{
        width: 8px;
        height: 8px;
    }}
    ::-webkit-scrollbar-track {{
        background: rgba(15, 23, 42, 0.1);
    }}
    ::-webkit-scrollbar-thumb {{
        background: rgba(255, 255, 255, 0.1);
        border-radius: 4px;
    }}
    ::-webkit-scrollbar-thumb:hover {{
        background: rgba(255, 255, 255, 0.2);
    }}

    /* Centralized reusable visual styling guide */
    .profile-grid {{
        display: grid !important;
        grid-template-columns: 1fr 1fr !important;
        gap: 1rem 1.5rem !important;
    }}
    
    .status-badge {{
        font-size: 0.78rem !important;
        font-weight: 600 !important;
        padding: 2px 8px !important;
        border-radius: 12px !important;
        display: inline-block !important;
    }}
    .status-badge-success {{
        color: #10b981 !important;
        background: rgba(16, 185, 129, 0.1) !important;
    }}
    .status-badge-warning {{
        color: #f59e0b !important;
        background: rgba(245, 158, 11, 0.1) !important;
    }}
    .status-badge-danger {{
        color: #ef4444 !important;
        background: rgba(239, 68, 68, 0.1) !important;
    }}

    .play-card {{
        background: rgba(15, 23, 42, 0.45) !important;
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
        border-left: 4px solid {primary_color_hex} !important;
        border-radius: 8px !important;
        padding: 1.1rem !important;
        margin-bottom: 0.8rem !important;
        backdrop-filter: blur(20px) !important;
    }}
    .play-card-header {{
        display: flex !important;
        justify-content: space-between !important;
        align-items: center !important;
        margin-bottom: 0.4rem !important;
    }}
    .play-card-title {{
        font-weight: 700 !important;
        font-size: 0.95rem !important;
        color: #38bdf8 !important;
    }}

    .health-banner {{
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
        border-radius: 12px !important;
        padding: 1.5rem !important;
        backdrop-filter: blur(20px) !important;
        margin-bottom: 2.0rem !important;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3) !important;
    }}
    .health-banner-success {{
        background: rgba(16, 185, 129, 0.08) !important;
        border-color: rgba(16, 185, 129, 0.2) !important;
        border-left: 5px solid #10b981 !important;
    }}
    .health-banner-warning {{
        background: rgba(245, 158, 11, 0.08) !important;
        border-color: rgba(245, 158, 11, 0.2) !important;
        border-left: 5px solid #f59e0b !important;
    }}
    .health-banner-danger {{
        background: rgba(239, 68, 68, 0.08) !important;
        border-color: rgba(239, 68, 68, 0.2) !important;
        border-left: 5px solid #ef4444 !important;
    }}
</style>
""", unsafe_allow_html=True)

# Initialize session state variables
if "jwt_token" not in st.session_state:
    st.session_state.jwt_token = None
if "current_user" not in st.session_state:
    st.session_state.current_user = None

def logout():
    st.session_state.jwt_token = None
    st.session_state.current_user = None
    st.cache_data.clear()  # Clear cache on logout
    st.rerun()

def check_401(status_code: int):
    """Detects expired or invalid tokens and automatically redirects to the login screen."""
    if status_code == 401:
        st.session_state.jwt_token = None
        st.session_state.current_user = None
        st.cache_data.clear()
        st.toast("Session expired. Please log in again.", icon="🔒")
        time.sleep(1.0)  # brief pause for toast display
        st.rerun()

# --- Caching Layers for In-Memory Performance ---
@st.cache_data(ttl=60)
def cached_get_overview(token: str):
    """Fetches cohort aggregation summary from the API (cached for 60s)."""
    status, data = api_client.get_overview()
    return status, data

@st.cache_data(ttl=60)
def cached_get_save_plays(token: str):
    """Fetches retention plays recommendation data from the API (cached for 60s)."""
    status, data = api_client.get_save_plays()
    return status, data

@st.cache_data(ttl=60)
def cached_get_cohort_data(token: str):
    """Fetches all customer details for cohort analysis (cached for 60s)."""
    status, data = api_client.get_cohort_data()
    return status, data

@st.cache_data(ttl=60)
def cached_get_diagnostics_metadata(token: str):
    """Fetches diagnostics metadata from the API (cached for 60s)."""
    status, data = api_client.get_diagnostics_metadata()
    return status, data

@st.cache_data(ttl=60)
def cached_get_model_health(token: str):
    """Fetches model health and feature drift statistics (cached for 60s)."""
    status, data = api_client.get_model_health()
    return status, data

@st.cache_data(ttl=15, show_spinner=False)
def cached_get_customer_explanation(token: str, customer_id: str):
    """Fetches SHAP force plots, demographics, recommendations and counterfactuals for an ID (cached 15s)."""
    status, data = api_client.get_customer_explanation(customer_id)
    return status, data

# --- Authentication Overlay ---
if st.session_state.jwt_token is None:
    render_auth_view(api_client, primary_color_hex, secondary_color_hex)
    st.stop()

# --- Authenticated App Context ---
# Sidebar Navigation
with st.sidebar:
    st.markdown("<div style='margin-bottom:1rem;'><span style='font-size: 1.5rem; font-weight: 800; font-family: Outfit; color: #ffffff;'>Retain<span style='color:#6366f1;'>IQ</span></span></div>", unsafe_allow_html=True)
    
    page = st.radio(
        "Navigation", 
        [
            "📊 Dashboard", 
            "🔍 Customer Explorer", 
            "🔮 Counterfactual Simulator", 
            "📈 Analytics", 
            "🌍 Explainability", 
            "🧩 Customer Segments", 
            "📤 Upload Dataset", 
            "🩺 Model Diagnostics", 
            "🚨 Drift Detection", 
            "⚙️ Settings"
        ],
        label_visibility="collapsed"
    )
    
    # Styled Admin Profile Card
    st.markdown("""
    <div style="background: rgba(255, 255, 255, 0.03); border: 1px solid rgba(255, 255, 255, 0.06); border-radius: 12px; padding: 0.8rem; margin-top: 3rem; display: flex; align-items: center; gap: 10px;">
        <div style="width: 34px; height: 34px; border-radius: 50%; background: linear-gradient(135deg, #6366f1 0%, #a78bfa 100%); display: flex; align-items: center; justify-content: center; font-weight: 700; color: #ffffff; font-size: 0.82rem; font-family: 'Outfit';">
            AD
        </div>
        <div style="flex: 1; min-width: 0;">
            <div style="font-weight: 700; font-size: 0.82rem; color: #f8fafc; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;">Admin User</div>
            <div style="font-size: 0.7rem; color: #94a3b8; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;">admin@retainiq.com</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<div style='margin-top:0.8rem;'></div>", unsafe_allow_html=True)
    if st.button("Log Out", use_container_width=True, type="secondary"):
        logout()

# Header Section
st.markdown("<div class='main-title'>RetainIQ Churn Portal</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Analyze customer lifetime risk, local explainability drivers, and map targeted Save Plays.</div>", unsafe_allow_html=True)

# Dispatch router
if page == "📊 Dashboard":
    status_code, overview = cached_get_overview(st.session_state.jwt_token)
    check_401(status_code)
    
    if status_code != 200:
        st.error(f"Failed to load overview data from API: {overview.get('detail', 'Unknown error')}")
        st.stop()
        
    _, plays_data_raw = cached_get_save_plays(st.session_state.jwt_token)
    _, cohort_data = cached_get_cohort_data(st.session_state.jwt_token)
    
    render_dashboard_view(
        overview=overview,
        plays_data_raw=plays_data_raw,
        cohort_data=cohort_data or [],
        primary_color_hex=primary_color_hex,
        secondary_color_hex=secondary_color_hex,
        danger_color_hex=danger_color_hex
    )

elif page == "🔍 Customer Explorer":
    render_explorer_view(
        api_client=api_client,
        check_401_callback=check_401,
        get_explanation_callback=cached_get_customer_explanation,
        primary_color_hex=primary_color_hex,
        secondary_color_hex=secondary_color_hex,
        danger_color_hex=danger_color_hex
    )

elif page == "🔮 Counterfactual Simulator":
    render_counterfactual_view(
        api_client=api_client,
        check_401_callback=check_401,
        get_explanation_callback=cached_get_customer_explanation,
        primary_color_hex=primary_color_hex,
        secondary_color_hex=secondary_color_hex,
        danger_color_hex=danger_color_hex
    )

elif page == "📈 Analytics":
    status_code_cohort, cohort_data = cached_get_cohort_data(st.session_state.jwt_token)
    check_401(status_code_cohort)
    
    if status_code_cohort != 200:
        st.error("Failed to load cohort data for Executive Analytics.")
        st.stop()
        
    _, plays_data_raw = cached_get_save_plays(st.session_state.jwt_token)
    
    render_executive_view(
        cohort_data=cohort_data,
        plays_data_raw=plays_data_raw,
        primary_color_hex=primary_color_hex,
        secondary_color_hex=secondary_color_hex,
        danger_color_hex=danger_color_hex
    )

elif page == "🌍 Explainability":
    render_explainability_view(
        project_root=project_root,
        primary_color_hex=primary_color_hex,
        secondary_color_hex=secondary_color_hex,
        danger_color_hex=danger_color_hex
    )

elif page == "🧩 Customer Segments":
    status_code_cohort, cohort_data = cached_get_cohort_data(st.session_state.jwt_token)
    check_401(status_code_cohort)
    
    if status_code_cohort != 200:
        st.error("Failed to load cohort data for Customer Segments.")
        st.stop()
        
    render_segments_view(
        cohort_data=cohort_data,
        primary_color_hex=primary_color_hex,
        secondary_color_hex=secondary_color_hex,
        danger_color_hex=danger_color_hex
    )

elif page == "📤 Upload Dataset":
    render_ingestion_view(
        api_client=api_client,
        check_401_callback=check_401
    )

elif page == "🩺 Model Diagnostics":
    m_status, m_data = cached_get_diagnostics_metadata(st.session_state.jwt_token)
    check_401(m_status)
    
    h_status, health = cached_get_model_health(st.session_state.jwt_token)
    check_401(h_status)
    
    _, cohort_data = cached_get_cohort_data(st.session_state.jwt_token)
    
    render_diagnostics_view(
        diagnostics_metadata=m_data if m_status == 200 else {},
        model_health=health if h_status == 200 else {},
        cohort_data=cohort_data,
        project_root=project_root,
        primary_color_hex=primary_color_hex,
        secondary_color_hex=secondary_color_hex,
        danger_color_hex=danger_color_hex
    )

elif page == "🚨 Drift Detection":
    h_status, health = cached_get_model_health(st.session_state.jwt_token)
    check_401(h_status)
    
    render_drift_view(
        model_health=health if h_status == 200 else {},
        primary_color_hex=primary_color_hex,
        secondary_color_hex=secondary_color_hex,
        danger_color_hex=danger_color_hex
    )

elif page == "⚙️ Settings":
    render_settings_view(
        primary_color_hex=primary_color_hex,
        secondary_color_hex=secondary_color_hex,
        danger_color_hex=danger_color_hex
    )
