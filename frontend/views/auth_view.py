import streamlit as st
from frontend.api_client import RetainIQAPIClient

def render_auth_view(api_client: RetainIQAPIClient, primary_color_hex: str, secondary_color_hex: str):
    """
    Renders the login view with custom styles, handling session authentication 
    and JWT token handshakes.
    """
    def hex_to_rgb(hex_str):
        hex_str = hex_str.lstrip('#')
        return tuple(int(hex_str[i:i+2], 16) for i in (0, 2, 4))
        
    rgb_primary = hex_to_rgb(primary_color_hex)
    
    # Inject Custom CSS styles to customize the app theme, forms, inputs, and layout
    st.markdown(f"""<style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Outfit:wght@700;800&display=swap');
        
        /* Hide default Streamlit overlays */
        footer {{
            display: none !important;
        }}
        header[data-testid="stHeader"] {{
            background-color: transparent !important;
            backdrop-filter: none !important;
        }}
        section[data-testid="stSidebar"] {{
            display: none !important;
        }}
        div[data-testid="collapsedControl"] {{
            display: none !important;
        }}
        
        /* Set base premium background with dark grid and color glows */
        .stApp {{
            background-color: #030014 !important;
            background-image: 
                radial-gradient(circle at 10% 20%, rgba({rgb_primary[0]}, {rgb_primary[1]}, {rgb_primary[2]}, 0.15), transparent 45%),
                radial-gradient(circle at 90% 80%, rgba(139, 92, 246, 0.15), transparent 45%),
                radial-gradient(circle at 50% 50%, rgba(2, 6, 23, 0.9), transparent 90%),
                linear-gradient(rgba(255, 255, 255, 0.01) 1px, transparent 1px),
                linear-gradient(90deg, rgba(255, 255, 255, 0.01) 1px, transparent 1px) !important;
            background-size: 100% 100%, 100% 100%, 100% 100%, 40px 40px, 40px 40px !important;
            background-attachment: fixed !important;
        }}
        
        /* Layout row configuration */
        div[data-testid="stHorizontalBlock"] {{
            align-items: center !important;
            margin-top: 8vh !important;
            gap: 3rem !important;
        }}
        
        /* Custom Glassmorphism card for login form with stronger glass and depth */
        div[data-testid="stForm"] {{
            background-color: rgba(15, 23, 42, 0.65) !important;
            backdrop-filter: blur(32px) !important;
            border: 1px solid rgba(255, 255, 255, 0.12) !important;
            border-top: 1.5px solid rgba({rgb_primary[0]}, {rgb_primary[1]}, {rgb_primary[2]}, 0.45) !important;
            border-radius: 16px !important;
            padding: 3rem 2.5rem !important;
            box-shadow: 0 30px 60px -15px rgba(0, 0, 0, 0.85), inset 0 1px 0 rgba(255, 255, 255, 0.05) !important;
            transition: transform 0.3s ease, box-shadow 0.3s ease !important;
        }}
        
        /* Force remove Streamlit's default border and hover border inside form container */
        div[data-testid="stForm"] > div {{
            border: none !important;
            padding: 0 !important;
        }}
        
        /* Custom styles for text inputs including refined placeholder contrast and svg icons */
        div[data-testid="stForm"] input[type="text"] {{
            background-color: rgba(2, 6, 23, 0.75) !important;
            border: 1px solid rgba(255, 255, 255, 0.12) !important;
            color: #f8fafc !important;
            border-radius: 8px !important;
            padding: 0.8rem 0.8rem 0.8rem 2.8rem !important;
            font-family: 'Inter', sans-serif !important;
            font-size: 0.9rem !important;
            background-image: url('data:image/svg+xml;utf8,<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="%2364748b" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"></path><circle cx="12" cy="7" r="4"></circle></svg>') !important;
            background-repeat: no-repeat !important;
            background-position: 14px center !important;
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
        }}
        
        div[data-testid="stForm"] input[type="password"] {{
            background-color: rgba(2, 6, 23, 0.75) !important;
            border: 1px solid rgba(255, 255, 255, 0.12) !important;
            color: #f8fafc !important;
            border-radius: 8px !important;
            padding: 0.8rem 0.8rem 0.8rem 2.8rem !important;
            font-family: 'Inter', sans-serif !important;
            font-size: 0.9rem !important;
            background-image: url('data:image/svg+xml;utf8,<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="%2364748b" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="3" y="11" width="18" height="11" rx="2" ry="2"></rect><path d="M7 11V7a5 5 0 0 1 10 0v4"></path></svg>') !important;
            background-repeat: no-repeat !important;
            background-position: 14px center !important;
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
        }}
        
        /* Refine eye icon visibility button to match glass theme */
        div[data-testid="stForm"] button[data-testid="stTextInputAdornment"] {{
            background-color: transparent !important;
            border: none !important;
            color: #64748b !important;
            border-radius: 50% !important;
            margin-right: 6px !important;
            transition: color 0.2s, background-color 0.2s !important;
        }}
        div[data-testid="stForm"] button[data-testid="stTextInputAdornment"]:hover {{
            color: #f8fafc !important;
            background-color: rgba(255, 255, 255, 0.05) !important;
        }}
        
        div[data-testid="stForm"] input:focus {{
            border-color: {primary_color_hex} !important;
            box-shadow: 0 0 0 3px rgba({rgb_primary[0]}, {rgb_primary[1]}, {rgb_primary[2]}, 0.3), 0 0 12px rgba({rgb_primary[0]}, {rgb_primary[1]}, {rgb_primary[2]}, 0.15) !important;
            background-color: rgba(2, 6, 23, 0.85) !important;
            outline: none !important;
        }}
        
        /* Custom styled checkboxes with better spacing and alignment */
        div[data-testid="stCheckbox"] label {{
            color: #cbd5e1 !important;
            font-size: 0.88rem !important;
            font-family: 'Inter', sans-serif !important;
            font-weight: 500 !important;
            letter-spacing: 0.2px !important;
            display: flex !important;
            align-items: center !important;
            gap: 6px !important;
        }}
        div[data-testid="stCheckbox"] div[role="checkbox"] {{
            background-color: rgba(2, 6, 23, 0.8) !important;
            border-color: rgba(255, 255, 255, 0.2) !important;
            border-radius: 4px !important;
            width: 16px !important;
            height: 16px !important;
            transition: all 0.2s ease !important;
        }}
        div[data-testid="stCheckbox"] div[role="checkbox"][aria-checked="true"] {{
            background-color: {primary_color_hex} !important;
            border-color: {primary_color_hex} !important;
        }}
        
        /* Submit Button premium gradient and hover styling */
        div[data-testid="stFormSubmitButton"] button {{
            background: linear-gradient(135deg, {primary_color_hex} 0%, #8b5cf6 100%) !important;
            color: #ffffff !important;
            border: none !important;
            border-radius: 8px !important;
            font-weight: 700 !important;
            font-size: 0.95rem !important;
            letter-spacing: 0.5px !important;
            padding: 0.85rem 0 !important;
            width: 100% !important;
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
            box-shadow: 0 4px 18px rgba({rgb_primary[0]}, {rgb_primary[1]}, {rgb_primary[2]}, 0.35) !important;
            cursor: pointer !important;
            margin-top: 0.4rem !important;
        }}
        div[data-testid="stFormSubmitButton"] button p {{
            color: #ffffff !important;
            font-weight: 700 !important;
        }}
        div[data-testid="stFormSubmitButton"] button:hover {{
            filter: brightness(1.1) !important;
            transform: translateY(-2px) !important;
            box-shadow: 0 8px 24px rgba({rgb_primary[0]}, {rgb_primary[1]}, {rgb_primary[2]}, 0.5) !important;
        }}
        div[data-testid="stFormSubmitButton"] button:active {{
            transform: translateY(0px) !important;
            box-shadow: 0 4px 12px rgba({rgb_primary[0]}, {rgb_primary[1]}, {rgb_primary[2]}, 0.25) !important;
        }}
        
        /* Micro-interactions for features cards */
        .feature-card {{
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
        }}
        .feature-card:hover {{
            transform: translateY(-3px) !important;
            border-color: rgba({rgb_primary[0]}, {rgb_primary[1]}, {rgb_primary[2]}, 0.25) !important;
            background-color: rgba(15, 23, 42, 0.45) !important;
            box-shadow: 0 10px 20px rgba(0, 0, 0, 0.3) !important;
        }}
        
        /* SSO login button hover styling */
        .sso-btn {{
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
        }}
        .sso-btn:hover {{
            border-color: rgba(167, 139, 250, 0.3) !important;
            background-color: rgba(255, 255, 255, 0.04) !important;
            transform: translateY(-1px) !important;
        }}
        
        /* Segmented pill tab buttons (no underlines, Apple/Stripe style) */
        .stTabs [data-baseweb="tab-list"] {{
            gap: 0.5rem !important;
            justify-content: center !important;
            background: rgba(15, 23, 42, 0.45) !important;
            border: 1px solid rgba(255, 255, 255, 0.05) !important;
            border-radius: 20px !important;
            padding: 4px 6px !important;
            margin-bottom: 1.5rem !important;
        }}
        .stTabs [data-baseweb="tab"] {{
            font-family: 'Outfit', sans-serif !important;
            font-weight: 600 !important;
            font-size: 0.92rem !important;
            color: #94a3b8 !important;
            background-color: transparent !important;
            border: none !important;
            border-radius: 16px !important;
            padding: 0.45rem 1.4rem !important;
            transition: all 0.2s !important;
        }}
        .stTabs [data-baseweb="tab"]:hover {{
            color: #f8fafc !important;
            background-color: rgba(255, 255, 255, 0.03) !important;
        }}
        .stTabs [aria-selected="true"] {{
            color: #ffffff !important;
            background-color: rgba(255, 255, 255, 0.08) !important;
            border: none !important;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15) !important;
        }}
        .stTabs [data-baseweb="tab-highlight-id"] {{
            display: none !important;
        }}
        
        /* Subtle radial ambient glow behind left brand panel to balance empty space */
        .brand-container {{
            background: radial-gradient(circle at 10% 20%, rgba(99, 102, 241, 0.05) 0%, transparent 60%),
                        radial-gradient(circle at 20% 80%, rgba(139, 92, 246, 0.05) 0%, transparent 60%) !important;
            border: 1px solid rgba(255, 255, 255, 0.02) !important;
            border-radius: 24px !important;
            padding: 3rem 2rem !important;
            margin-top: 1rem !important;
            box-shadow: inset 0 0 20px rgba(255, 255, 255, 0.01) !important;
        }}
        
        /* Hide default Streamlit Input instructions (Press Enter to submit form) */
        div[data-testid="InputInstructions"] {{
            display: none !important;
        }}
        
        /* Align Remember Me checkbox and Forgot Password link row robustly */
        div[data-testid="stForm"] div[data-testid="stHorizontalBlock"] {{
            display: flex !important;
            flex-direction: row !important;
            flex-wrap: nowrap !important;
            align-items: center !important;
            justify-content: space-between !important;
            margin-top: 0.8rem !important;
            margin-bottom: 1rem !important;
            gap: 1rem !important;
        }}
        
        /* Ensure individual columns don't wrap, stack, or have negative/excessive margins */
        div[data-testid="stForm"] div[data-testid="stHorizontalBlock"] > div[data-testid="column"] {{
            width: auto !important;
            flex: 1 1 auto !important;
            margin: 0 !important;
            padding: 0 !important;
        }}
        
        /* Center "Forgot Password" link container text alignment */
        div[data-testid="stForm"] div[data-testid="stHorizontalBlock"] div[data-testid="column"]:nth-of-type(2) {{
            display: flex !important;
            justify-content: flex-end !important;
            align-items: center !important;
        }}
        
        /* Submit button spacing */
        div[data-testid="stFormSubmitButton"] {{
            margin-top: 0.5rem !important;
        }}
        
        /* Hide the awkward vertical line caret inside the selectbox input */
        div[data-testid="stSelectbox"] input,
        div[data-testid="stSelectbox"] [role="combobox"],
        div[data-testid="stSelectbox"] [role="combobox"] *,
        [data-baseweb="select"] input,
        [data-baseweb="select"] [role="combobox"],
        [data-baseweb="select"] [role="combobox"] * {{
            caret-color: transparent !important;
        }}

        /* Force the hand/pointer icon when hovering over the selectbox */
        div[data-testid="stSelectbox"], 
        div[data-testid="stSelectbox"] * {{
            cursor: pointer !important;
        }}

        /* Remove the vertical separator line between the text and the dropdown arrow */
        div[data-testid="stSelectbox"] [data-baseweb="select"] [role="button"] + div,
        div[data-testid="stSelectbox"] [data-baseweb="select"] [role="combobox"] + div,
        div[data-testid="stSelectbox"] [data-baseweb="select"] [class*="indicator"] {{
            border-left: none !important;
            border-right: none !important;
            border-left-width: 0px !important;
        }}
    </style>""".replace("\n", " "), unsafe_allow_html=True)
    
    # 2. Setup structural grid columns for Left informational / Right form panels
    _, col_left, col_right, _ = st.columns([0.1, 1.1, 0.9, 0.1])
    
    with col_left:
        # Left branding block & Product values
        branding_html = f"""
        <div class="brand-container" style="padding: 3.5rem 2.5rem; font-family: 'Inter', sans-serif; margin-top: 1.5rem; margin-bottom: 1.5rem;">
            <!-- Premium Logo -->
            <div style="display: flex; align-items: center; gap: 16px; margin-bottom: 3.5rem;">
                <svg role="img" aria-label="RetainIQ Crystal Logo" width="48" height="48" viewBox="0 0 100 100" fill="none" xmlns="http://www.w3.org/2000/svg">
                    <polygon points="50,10 85,35 85,65 50,90 15,65 15,35" fill="url(#gemGrad)" />
                    <polygon points="50,10 50,90 85,65 85,35" fill="url(#gemGlow)" opacity="0.6" />
                    <polygon points="50,10 50,90 15,65 15,35" fill="url(#gemShadow)" opacity="0.4" />
                    <polygon points="50,10 85,35 50,45" fill="#a78bfa" opacity="0.3" />
                    <polygon points="50,10 15,35 50,45" fill="#6366f1" opacity="0.3" />
                    <polygon points="50,90 85,65 50,75" fill="#a78bfa" opacity="0.3" />
                    <polygon points="50,90 15,65 50,75" fill="#6366f1" opacity="0.3" />
                    <defs>
                        <linearGradient id="gemGrad" x1="0%" y1="0%" x2="100%" y2="100%">
                            <stop offset="0%" stop-color="#8b5cf6" />
                            <stop offset="50%" stop-color="#6366f1" />
                            <stop offset="100%" stop-color="#4f46e5" />
                        </linearGradient>
                        <linearGradient id="gemGlow" x1="50%" y1="0%" x2="50%" y2="100%">
                            <stop offset="0%" stop-color="#c084fc" stop-opacity="0.8"/>
                            <stop offset="100%" stop-color="#6366f1" stop-opacity="0.1"/>
                        </linearGradient>
                        <linearGradient id="gemShadow" x1="0%" y1="50%" x2="100%" y2="50%">
                            <stop offset="0%" stop-color="#000000" stop-opacity="0.6"/>
                            <stop offset="100%" stop-color="#000000" stop-opacity="0"/>
                        </linearGradient>
                    </defs>
                </svg>
                <div>
                    <span style="font-size: 2.2rem; font-weight: 800; font-family: 'Outfit', sans-serif; color: #ffffff; letter-spacing: -0.5px;">Retain<span style="color: #6366f1;">IQ</span></span>
                    <div style="font-size: 0.7rem; color: #8b5cf6; font-weight: 600; text-transform: uppercase; letter-spacing: 1.2px; font-family: 'Inter', sans-serif; margin-top: 1px;">AI-Powered Customer Retention Intelligence Platform</div>
                </div>
            </div>

            <!-- Header and Tagline -->
            <div style="margin-top: 2rem;">
                <h1 style="font-family: 'Outfit', sans-serif; font-weight: 800; font-size: 2.6rem; line-height: 1.25; color: #f8fafc; margin-bottom: 1.2rem; letter-spacing: -0.8px;">
                    Predict churn.<br>
                    <span style="background: linear-gradient(135deg, #a78bfa, #6366f1); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">Understand customers.</span><br>
                    Retain revenue.
                </h1>
                <p style="color: #94a3b8; font-size: 0.95rem; line-height: 1.6; font-family: 'Inter', sans-serif; margin-bottom: 4.5rem; max-width: 480px;">
                    RetainIQ uses advanced AI and explainable models to identify at-risk customers, uncover key drivers of churn, and recommend actions that drive retention and revenue growth.
                </p>
            </div>

            <!-- Features Cards Grid - Refined Padding, Icon Sizing, and Typography -->
            <div style="display: flex; gap: 14px; flex-wrap: wrap; margin-top: 2rem; margin-bottom: 2rem;">
                <div class="feature-card" style="flex: 1; min-width: 105px; padding: 18px 14px; background: rgba(15, 23, 42, 0.3); border: 1px solid rgba(255, 255, 255, 0.08); border-radius: 12px; text-align: center; box-shadow: 0 6px 16px rgba(0,0,0,0.18);">
                    <div style="background: rgba(99, 102, 241, 0.1); border: 1px solid rgba(99, 102, 241, 0.2); border-radius: 8px; width: 38px; height: 38px; display: inline-flex; align-items: center; justify-content: center; margin-bottom: 12px; color: #6366f1;">
                        <svg role="img" aria-label="Predictions Icon" xmlns="http://www.w3.org/2000/svg" width="17" height="17" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="18" y1="20" x2="18" y2="10"></line><line x1="12" y1="20" x2="12" y2="4"></line><line x1="6" y1="20" x2="6" y2="14"></line></svg>
                    </div>
                    <div style="font-weight: 600; font-size: 0.82rem; color: #f8fafc; font-family: 'Inter', sans-serif; line-height: 1.4;">AI-Powered Predictions</div>
                </div>
                <div class="feature-card" style="flex: 1; min-width: 105px; padding: 18px 14px; background: rgba(15, 23, 42, 0.3); border: 1px solid rgba(255, 255, 255, 0.08); border-radius: 12px; text-align: center; box-shadow: 0 6px 16px rgba(0,0,0,0.18);">
                    <div style="background: rgba(99, 102, 241, 0.1); border: 1px solid rgba(99, 102, 241, 0.2); border-radius: 8px; width: 38px; height: 38px; display: inline-flex; align-items: center; justify-content: center; margin-bottom: 12px; color: #6366f1;">
                        <svg role="img" aria-label="Insights Icon" xmlns="http://www.w3.org/2000/svg" width="17" height="17" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M17 21v-2a4 4 0 0 0-4-4H5a4 4 0 0 0-4 4v2"></path><circle cx="9" cy="7" r="4"></circle><path d="M23 21v-2a4 4 0 0 0-3-3.87"></path><path d="M16 3.13a4 4 0 0 1 0 7.75"></path></svg>
                    </div>
                    <div style="font-weight: 600; font-size: 0.82rem; color: #f8fafc; font-family: 'Inter', sans-serif; line-height: 1.4;">Customer Insights</div>
                </div>
                <div class="feature-card" style="flex: 1; min-width: 105px; padding: 18px 14px; background: rgba(15, 23, 42, 0.3); border: 1px solid rgba(255, 255, 255, 0.08); border-radius: 12px; text-align: center; box-shadow: 0 6px 16px rgba(0,0,0,0.18);">
                    <div style="background: rgba(99, 102, 241, 0.1); border: 1px solid rgba(99, 102, 241, 0.2); border-radius: 8px; width: 38px; height: 38px; display: inline-flex; align-items: center; justify-content: center; margin-bottom: 12px; color: #6366f1;">
                        <svg role="img" aria-label="Explainable AI Icon" xmlns="http://www.w3.org/2000/svg" width="17" height="17" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"></path></svg>
                    </div>
                    <div style="font-weight: 600; font-size: 0.82rem; color: #f8fafc; font-family: 'Inter', sans-serif; line-height: 1.4;">Explainable AI (XAI)</div>
                </div>
                <div class="feature-card" style="flex: 1; min-width: 105px; padding: 18px 14px; background: rgba(15, 23, 42, 0.3); border: 1px solid rgba(255, 255, 255, 0.08); border-radius: 12px; text-align: center; box-shadow: 0 6px 16px rgba(0,0,0,0.18);">
                    <div style="background: rgba(99, 102, 241, 0.1); border: 1px solid rgba(99, 102, 241, 0.2); border-radius: 8px; width: 38px; height: 38px; display: inline-flex; align-items: center; justify-content: center; margin-bottom: 12px; color: #6366f1;">
                        <svg role="img" aria-label="Retention Chart Icon" xmlns="http://www.w3.org/2000/svg" width="17" height="17" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M3 3v18h18"></path><path d="M18.7 8l-5.1 5.2-2.8-2.7L7 14.3"></path></svg>
                    </div>
                    <div style="font-weight: 600; font-size: 0.82rem; color: #f8fafc; font-family: 'Inter', sans-serif; line-height: 1.4;">Data-Driven Retention</div>
                </div>
            </div>
        </div>
        """
        st.markdown(branding_html.replace("\n", " "), unsafe_allow_html=True)
        
    with col_right:
        # Glassmorphic Login / Signup Tabs
        query_params = st.query_params
        show_forgot = query_params.get("action") == "forgot"
        
        if show_forgot:
            with st.form("forgot_password_form", clear_on_submit=False):
                # Centered header inside card
                forgot_header_html = """
                <div style="display: flex; flex-direction: column; align-items: center; margin-bottom: 1.5rem;">
                    <div style="background: rgba(99, 102, 241, 0.1); border: 1px solid rgba(99, 102, 241, 0.25); border-radius: 12px; padding: 4px 10px; display: inline-flex; align-items: center; gap: 6px; color: #6366f1; font-family: 'Inter', sans-serif; font-size: 0.72rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 1rem;">
                        <svg role="img" aria-label="Key Reset" xmlns="http://www.w3.org/2000/svg" width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="#6366f1" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><path d="M21 2l-2 2m-7.61 7.61a5.5 5.5 0 1 1-7.778 7.778 5.5 5.5 0 0 1 7.777-7.777zm0 0L15.5 7.5m0 0l3 3L22 7l-3-3m-3.5 3.5L19 4"></path></svg>
                        Password Recovery
                    </div>
                    <h2 style="text-align: center; margin: 0; font-family: 'Outfit', sans-serif; font-weight: 700; font-size: 1.7rem; color: #f8fafc;">Reset Password</h2>
                    <p style="text-align: center; margin: 0.3rem 0 0 0; color: #94a3b8; font-size: 0.88rem; font-family: 'Inter', sans-serif;">Verify your identity to choose a new password</p>
                </div>
                """
                st.markdown(forgot_header_html, unsafe_allow_html=True)
                
                # Username lookup step
                st.markdown("<div style='font-family: Inter, sans-serif; font-size: 0.85rem; font-weight: 500; color: #cbd5e1; margin-bottom: 6px;'>Username</div>", unsafe_allow_html=True)
                forgot_username = st.text_input("Username", value="", placeholder="Enter your username", label_visibility="collapsed", key="forgot_username_field")
                
                # Fetch security question if username is provided
                security_q = ""
                if forgot_username:
                    status_code, res_data = api_client.get_security_question(forgot_username)
                    if status_code == 200:
                        security_q = res_data.get("security_question", "")
                    else:
                        detail = res_data.get("detail", "Username not found or recovery unavailable")
                        st.markdown(f"<div style='color: #f87171; font-size: 0.82rem; font-family: Inter, sans-serif; margin-bottom: 10px; font-weight: 500;'>⚠️ {detail}</div>", unsafe_allow_html=True)
                
                # Render question and answer inputs if question was fetched successfully
                if security_q:
                    st.markdown(f"<div style='font-family: Inter, sans-serif; font-size: 0.85rem; font-weight: 500; color: #cbd5e1; margin-bottom: 6px; margin-top: 1.2rem;'>Security Question: <span style='color: #a78bfa; font-weight: 600;'>{security_q}</span></div>", unsafe_allow_html=True)
                    security_ans = st.text_input("Security Answer", value="", placeholder="Enter your security answer", label_visibility="collapsed", key="forgot_security_answer_field")
                    
                    st.markdown("<div style='font-family: Inter, sans-serif; font-size: 0.85rem; font-weight: 500; color: #cbd5e1; margin-bottom: 6px; margin-top: 1.2rem;'>New Password</div>", unsafe_allow_html=True)
                    new_pass = st.text_input("New Password", type="password", value="", placeholder="Choose a new password", label_visibility="collapsed", key="forgot_new_password_field")
                else:
                    security_ans = ""
                    new_pass = ""
                    
                st.markdown("<div style='margin-top: 1.5rem;'></div>", unsafe_allow_html=True)
                
                reset_submitted = st.form_submit_button("Reset Password   →", use_container_width=True)
                
                # Link back to Sign In using query param target=_self
                st.markdown("<div style='text-align: center; font-family: Inter, sans-serif; font-size: 0.85rem; margin-top: 1rem;'><a href='/?action=signin' target='_self' style='color: #6366f1; text-decoration: none; font-weight: 500;'>← Back to Sign In</a></div>", unsafe_allow_html=True)
                
                if reset_submitted:
                    if not forgot_username:
                        st.error("Please enter your username first.")
                    elif not security_q:
                        st.error("Please enter a valid username that has a security question.")
                    elif not security_ans or not new_pass:
                        st.error("All recovery fields are required.")
                    elif len(new_pass) < 6:
                        st.error("New password must be at least 6 characters long.")
                    else:
                        with st.spinner("Verifying answer and updating password..."):
                            status_code, res_data = api_client.reset_password(forgot_username, security_ans, new_pass)
                            if status_code == 200:
                                st.success("Password reset successfully! Redirecting you to sign in...")
                                st.toast("Password reset successful!", icon="🔐")
                                # Clear query params to return to sign in
                                st.query_params.clear()
                                import time
                                time.sleep(1.5)
                                st.rerun()
                            else:
                                detail = res_data.get("detail", "Failed to reset password")
                                st.error(detail)
        else:
            tab_signin, tab_signup = st.tabs(["Sign In", "Sign Up"])
            
            with tab_signin:
                with st.form("login_form", clear_on_submit=False):
                    # Centered Lock graphic header inside card
                    lock_header_html = """
                    <div style="display: flex; flex-direction: column; align-items: center; margin-bottom: 1.5rem;">
                        <div style="background: rgba(16, 185, 129, 0.1); border: 1px solid rgba(16, 185, 129, 0.25); border-radius: 12px; padding: 4px 10px; display: inline-flex; align-items: center; gap: 6px; color: #10b981; font-family: 'Inter', sans-serif; font-size: 0.72rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 1rem;">
                            <svg role="img" aria-label="Secure Connection Shield" xmlns="http://www.w3.org/2000/svg" width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="#10b981" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"></path></svg>
                            Secure Connection
                        </div>
                        <h2 style="text-align: center; margin: 0; font-family: 'Outfit', sans-serif; font-weight: 700; font-size: 1.7rem; color: #f8fafc;">Welcome Back</h2>
                        <p style="text-align: center; margin: 0.3rem 0 0 0; color: #94a3b8; font-size: 0.88rem; font-family: 'Inter', sans-serif;">Sign in to access your RetainIQ dashboard</p>
                    </div>
                    """
                    st.markdown(lock_header_html, unsafe_allow_html=True)
                    
                    # Form Fields
                    st.markdown("<div style='font-family: Inter, sans-serif; font-size: 0.85rem; font-weight: 500; color: #cbd5e1; margin-bottom: 6px;'>Username</div>", unsafe_allow_html=True)
                    username = st.text_input("Username", value="", placeholder="Enter your username", label_visibility="collapsed", key="login_username")
                    
                    st.markdown("<div style='font-family: Inter, sans-serif; font-size: 0.85rem; font-weight: 500; color: #cbd5e1; margin-bottom: 6px; margin-top: 1.2rem;'>Password</div>", unsafe_allow_html=True)
                    password = st.text_input("Password", type="password", value="", placeholder="Enter your password", label_visibility="collapsed", key="login_password")
                    
                    # Forgot Password link aligned to the right
                    st.markdown("<div style='text-align: right; font-family: Inter, sans-serif; font-size: 0.85rem; margin-top: 10px; margin-bottom: 10px;'><a href='/?action=forgot' target='_self' style='color: #6366f1; text-decoration: none; font-weight: 500;'>Forgot Password?</a></div>", unsafe_allow_html=True)
                    
                    submitted = st.form_submit_button("Login to Dashboard   →", use_container_width=True)
                    
                    # Footer Terms and Encryption status (no SSO button)
                    footer_html = """
                    <div style="text-align: center; font-size: 0.78rem; color: #64748b; font-family: 'Inter', sans-serif; margin-top: 2rem; line-height: 1.4;">
                        By continuing, you agree to our <a href="#" style="color: #94a3b8; text-decoration: none; border-bottom: 1px dotted #94a3b8;">Terms of Service</a> and <a href="#" style="color: #94a3b8; text-decoration: none; border-bottom: 1px dotted #94a3b8;">Privacy Policy</a>
                    </div>
                    
                    <div style="display: flex; justify-content: center; margin-top: 1.5rem; margin-bottom: 0.5rem;">
                        <div style="display: flex; align-items: center; gap: 6px; padding: 4px 12px; background: rgba(16, 185, 129, 0.05); border: 1px solid rgba(16, 185, 129, 0.15); border-radius: 20px; color: #10b981; font-family: 'Inter', sans-serif; font-size: 0.72rem; font-weight: 500;">
                            <svg role="img" aria-label="Encryption Padlock" xmlns="http://www.w3.org/2000/svg" width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="#10b981" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><rect x="3" y="11" width="18" height="11" rx="2" ry="2"></rect><path d="M7 11V7a5 5 0 0 1 10 0v4"></path></svg>
                            Encrypted Connection
                        </div>
                    </div>
                    """
                    st.markdown(footer_html.replace("\n", " "), unsafe_allow_html=True)
                    
                    if submitted:
                        if not username or not password:
                            st.error("Username and password are required to sign in.")
                        else:
                            with st.spinner("Signing in..."):
                                status_code, data = api_client.login(username, password)
                                if status_code == 200:
                                    st.session_state.jwt_token = data["access_token"]
                                    st.session_state.current_user = username
                                    st.toast("Login successful. Welcome back!", icon="👋")
                                    st.rerun()
                                else:
                                    detail = data.get("detail", "Invalid credentials")
                                    st.error(f"Authentication failed: {detail}")
                                    
            with tab_signup:
                with st.form("signup_form", clear_on_submit=False):
                    # Centered Sign Up graphic header inside card
                    signup_header_html = """
                    <div style="display: flex; flex-direction: column; align-items: center; margin-bottom: 1.5rem;">
                        <div style="background: rgba(16, 185, 129, 0.1); border: 1px solid rgba(16, 185, 129, 0.25); border-radius: 12px; padding: 4px 10px; display: inline-flex; align-items: center; gap: 6px; color: #10b981; font-family: 'Inter', sans-serif; font-size: 0.72rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 1rem;">
                            <svg role="img" aria-label="Secure Connection Shield" xmlns="http://www.w3.org/2000/svg" width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="#10b981" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"></path></svg>
                            Secure Connection
                        </div>
                        <h2 style="text-align: center; margin: 0; font-family: 'Outfit', sans-serif; font-weight: 700; font-size: 1.7rem; color: #f8fafc;">Create Account</h2>
                        <p style="text-align: center; margin: 0.3rem 0 0 0; color: #94a3b8; font-size: 0.88rem; font-family: 'Inter', sans-serif;">Register to set up your profile and dashboard</p>
                    </div>
                    """
                    st.markdown(signup_header_html, unsafe_allow_html=True)
                    
                    st.markdown("<div style='font-family: Inter, sans-serif; font-size: 0.85rem; font-weight: 500; color: #cbd5e1; margin-bottom: 6px;'>Username</div>", unsafe_allow_html=True)
                    new_username = st.text_input("Username", value="", placeholder="Choose a username", label_visibility="collapsed", key="signup_username")
                    
                    st.markdown("<div style='font-family: Inter, sans-serif; font-size: 0.85rem; font-weight: 500; color: #cbd5e1; margin-bottom: 6px; margin-top: 1.2rem;'>Password</div>", unsafe_allow_html=True)
                    new_password = st.text_input("Password", type="password", value="", placeholder="Choose a password", label_visibility="collapsed", key="signup_password")
                    if new_password and len(new_password) < 6:
                        st.markdown("<div style='color: #f87171; font-size: 0.78rem; margin-top: 6px; margin-bottom: -6px; font-family: Inter, sans-serif; font-weight: 500;'>Password must be at least 6 characters long</div>", unsafe_allow_html=True)
                    
                    st.markdown("<div style='font-family: Inter, sans-serif; font-size: 0.85rem; font-weight: 500; color: #cbd5e1; margin-bottom: 6px; margin-top: 1.2rem;'>Confirm Password</div>", unsafe_allow_html=True)
                    confirm_password = st.text_input("Confirm Password", type="password", value="", placeholder="Confirm your password", label_visibility="collapsed", key="signup_confirm_password")
                    
                    # Security Question selection dropdown and Answer input
                    st.markdown("<div style='font-family: Inter, sans-serif; font-size: 0.85rem; font-weight: 500; color: #cbd5e1; margin-bottom: 6px; margin-top: 1.2rem;'>Security Question</div>", unsafe_allow_html=True)
                    security_questions = [
                        "What is your favorite sport?",
                        "What was the name of your first school?",
                        "In what city or town were you born?",
                        "Who is your favorite celebrity?",
                        "What is your favorite subject?"
                    ]
                    new_security_question = st.selectbox("Security Question", options=security_questions, label_visibility="collapsed", key="signup_security_question")
                    
                    st.markdown("<div style='font-family: Inter, sans-serif; font-size: 0.85rem; font-weight: 500; color: #cbd5e1; margin-bottom: 6px; margin-top: 1.2rem;'>Security Answer</div>", unsafe_allow_html=True)
                    new_security_answer = st.text_input("Security Answer", value="", placeholder="Enter your answer", label_visibility="collapsed", key="signup_security_answer")
                    
                    # Symmetrical vertical padding spacer instead of checkbox row
                    st.markdown("<div style='margin-top: 1.5rem;'></div>", unsafe_allow_html=True)
                    
                    register_submitted = st.form_submit_button("Register Account   →", use_container_width=True)
                    
                    # Bottom Terms and connection badge inside signup tab for perfect height symmetry
                    signup_footer_html = """
                    <div style="text-align: center; font-size: 0.78rem; color: #64748b; font-family: 'Inter', sans-serif; margin-top: 1.8rem; line-height: 1.4;">
                        By continuing, you agree to our <a href="#" style="color: #94a3b8; text-decoration: none; border-bottom: 1px dotted #94a3b8;">Terms of Service</a> and <a href="#" style="color: #94a3b8; text-decoration: none; border-bottom: 1px dotted #94a3b8;">Privacy Policy</a>
                    </div>
                    
                    <div style="display: flex; justify-content: center; margin-top: 1.5rem; margin-bottom: 0.5rem;">
                        <div style="display: flex; align-items: center; gap: 6px; padding: 4px 12px; background: rgba(16, 185, 129, 0.05); border: 1px solid rgba(16, 185, 129, 0.15); border-radius: 20px; color: #10b981; font-family: 'Inter', sans-serif; font-size: 0.72rem; font-weight: 500;">
                            <svg role="img" aria-label="Encryption Padlock" xmlns="http://www.w3.org/2000/svg" width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="#10b981" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><rect x="3" y="11" width="18" height="11" rx="2" ry="2"></rect><path d="M7 11V7a5 5 0 0 1 10 0v4"></path></svg>
                            Encrypted Connection
                        </div>
                    </div>
                    """
                    st.markdown(signup_footer_html.replace("\n", " "), unsafe_allow_html=True)
                    
                    if register_submitted:
                        if not new_username or not new_password or not confirm_password or not new_security_answer:
                            st.error("All registration fields are required (including security answer).")
                        elif len(new_username) < 3:
                            st.error("Username must be at least 3 characters long.")
                        elif len(new_password) < 6:
                            st.error("Password must be at least 6 characters long.")
                        elif new_password != confirm_password:
                            st.error("Passwords do not match.")
                        else:
                            with st.spinner("Creating your account..."):
                                status_code, res_data = api_client.register(
                                    new_username, 
                                    new_password, 
                                    new_security_question, 
                                    new_security_answer
                                )
                                if status_code == 201:
                                    st.success("Account registered successfully! Please sign in using the 'Sign In' tab.")
                                else:
                                    detail = res_data.get("detail", "Registration failed")
                                    st.error(f"Registration failed: {detail}")
                                    
    st.stop()
