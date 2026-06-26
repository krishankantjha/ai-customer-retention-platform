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
        
        /* Custom Glassmorphism card for login form */
        div[data-testid="stForm"] {{
            background-color: rgba(15, 23, 42, 0.4) !important;
            backdrop-filter: blur(24px) !important;
            border: 1px solid rgba(255, 255, 255, 0.08) !important;
            border-top: 1.5px solid rgba({rgb_primary[0]}, {rgb_primary[1]}, {rgb_primary[2]}, 0.3) !important;
            border-radius: 16px !important;
            padding: 3rem 2.5rem !important;
            box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.6) !important;
        }}
        
        /* Force remove Streamlit's default border and hover border inside form container */
        div[data-testid="stForm"] > div {{
            border: none !important;
            padding: 0 !important;
        }}
        
        /* Custom styles for text inputs including background SVG icons */
        div[data-testid="stForm"] input[type="text"] {{
            background-color: rgba(2, 6, 23, 0.7) !important;
            border: 1px solid rgba(255, 255, 255, 0.1) !important;
            color: #f8fafc !important;
            border-radius: 8px !important;
            padding: 0.75rem 0.75rem 0.75rem 2.5rem !important;
            font-family: 'Inter', sans-serif !important;
            font-size: 0.9rem !important;
            background-image: url('data:image/svg+xml;utf8,<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="%2394a3b8" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"></path><circle cx="12" cy="7" r="4"></circle></svg>') !important;
            background-repeat: no-repeat !important;
            background-position: 12px center !important;
            transition: all 0.2s !important;
        }}
        
        div[data-testid="stForm"] input[type="password"] {{
            background-color: rgba(2, 6, 23, 0.7) !important;
            border: 1px solid rgba(255, 255, 255, 0.1) !important;
            color: #f8fafc !important;
            border-radius: 8px !important;
            padding: 0.75rem 0.75rem 0.75rem 2.5rem !important;
            font-family: 'Inter', sans-serif !important;
            font-size: 0.9rem !important;
            background-image: url('data:image/svg+xml;utf8,<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="%2394a3b8" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="3" y="11" width="18" height="11" rx="2" ry="2"></rect><path d="M7 11V7a5 5 0 0 1 10 0v4"></path></svg>') !important;
            background-repeat: no-repeat !important;
            background-position: 12px center !important;
            transition: all 0.2s !important;
        }}
        
        div[data-testid="stForm"] input:focus {{
            border-color: {primary_color_hex} !important;
            box-shadow: 0 0 0 2px rgba({rgb_primary[0]}, {rgb_primary[1]}, {rgb_primary[2]}, 0.25) !important;
            outline: none !important;
        }}
        
        /* Custom styled checkboxes */
        div[data-testid="stCheckbox"] label {{
            color: #94a3b8 !important;
            font-size: 0.85rem !important;
            font-family: 'Inter', sans-serif !important;
        }}
        div[data-testid="stCheckbox"] div[role="checkbox"] {{
            background-color: rgba(2, 6, 23, 0.7) !important;
            border-color: rgba(255, 255, 255, 0.15) !important;
            border-radius: 4px !important;
        }}
        div[data-testid="stCheckbox"] div[role="checkbox"][aria-checked="true"] {{
            background-color: {primary_color_hex} !important;
            border-color: {primary_color_hex} !important;
        }}
        
        /* Submit Button Customizer */
        div[data-testid="stForm"] button[type="submit"] {{
            background: linear-gradient(135deg, {primary_color_hex} 0%, {primary_color_hex}dd 100%) !important;
            color: #ffffff !important;
            border: none !important;
            border-radius: 8px !important;
            font-weight: 600 !important;
            font-size: 0.95rem !important;
            padding: 0.75rem 0 !important;
            width: 100% !important;
            transition: all 0.2s ease-in-out !important;
            box-shadow: 0 4px 14px rgba({rgb_primary[0]}, {rgb_primary[1]}, {rgb_primary[2]}, 0.3) !important;
            cursor: pointer !important;
            margin-top: 1.2rem !important;
        }}
        div[data-testid="stForm"] button[type="submit"]:hover {{
            filter: brightness(1.1) !important;
            transform: translateY(-1px) !important;
            box-shadow: 0 6px 20px rgba({rgb_primary[0]}, {rgb_primary[1]}, {rgb_primary[2]}, 0.4) !important;
        }}
        div[data-testid="stForm"] button[type="submit"]:active {{
            transform: translateY(1px) !important;
            box-shadow: 0 2px 10px rgba({rgb_primary[0]}, {rgb_primary[1]}, {rgb_primary[2]}, 0.2) !important;
        }}
    </style>""".replace("\n", " "), unsafe_allow_html=True)
    
    # 1. Float Top Right Pill: Secure Connection Banner
    secure_pill_html = """
    <div style="position: absolute; top: 30px; right: 40px; z-index: 999; display: flex; align-items: center; gap: 8px; padding: 6px 14px; background: rgba(15, 23, 42, 0.4); border: 1px solid rgba(255, 255, 255, 0.08); border-radius: 20px; color: #f8fafc; font-family: 'Inter', sans-serif; font-size: 0.8rem; font-weight: 500; box-shadow: 0 4px 12px rgba(0,0,0,0.25);">
        <svg role="img" aria-label="Secure Connection Shield" xmlns="http://www.w3.org/2000/svg" width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="#10b981" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"></path></svg>
        Secure Login
    </div>
    """
    st.markdown(secure_pill_html, unsafe_allow_html=True)

    # 2. Setup structural grid columns for Left informational / Right form panels
    _, col_left, col_right, _ = st.columns([0.1, 1.1, 0.9, 0.1])
    
    with col_left:
        # Left branding block & Product values
        branding_html = f"""
        <div class="brand-container" style="padding-right: 1.5rem; font-family: 'Inter', sans-serif;">
            <!-- Premium Logo -->
            <div style="display: flex; align-items: center; gap: 16px; margin-bottom: 2.5rem;">
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
                    <div style="font-size: 0.8rem; color: #8b5cf6; font-weight: 600; text-transform: uppercase; letter-spacing: 1.5px; font-family: 'Inter', sans-serif; margin-top: 1px;">AI Customer Retention Platform</div>
                </div>
            </div>

            <!-- Header and Tagline -->
            <div style="margin-top: 2rem;">
                <h1 style="font-family: 'Outfit', sans-serif; font-weight: 800; font-size: 2.6rem; line-height: 1.25; color: #f8fafc; margin-bottom: 1.2rem; letter-spacing: -0.8px;">
                    Predict churn.<br>
                    <span style="background: linear-gradient(135deg, #a78bfa, #6366f1); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">Understand customers.</span><br>
                    Retain revenue.
                </h1>
                <p style="color: #94a3b8; font-size: 0.95rem; line-height: 1.6; font-family: 'Inter', sans-serif; margin-bottom: 3rem; max-width: 480px;">
                    RetainIQ uses advanced AI and explainable models to identify at-risk customers, uncover key drivers of churn, and recommend actions that drive retention and revenue growth.
                </p>
            </div>

            <!-- Features Cards Grid -->
            <div style="display: flex; gap: 14px; flex-wrap: wrap; margin-top: 2rem; margin-bottom: 2rem;">
                <div style="flex: 1; min-width: 105px; padding: 16px 12px; background: rgba(15, 23, 42, 0.3); border: 1px solid rgba(255, 255, 255, 0.04); border-radius: 12px; text-align: center; box-shadow: 0 4px 12px rgba(0,0,0,0.15);">
                    <div style="background: rgba(99, 102, 241, 0.1); border: 1px solid rgba(99, 102, 241, 0.2); border-radius: 8px; width: 34px; height: 34px; display: inline-flex; align-items: center; justify-content: center; margin-bottom: 10px; color: #6366f1;">
                        <svg role="img" aria-label="Predictions Icon" xmlns="http://www.w3.org/2000/svg" width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="18" y1="20" x2="18" y2="10"></line><line x1="12" y1="20" x2="12" y2="4"></line><line x1="6" y1="20" x2="6" y2="14"></line></svg>
                    </div>
                    <div style="font-weight: 600; font-size: 0.75rem; color: #f8fafc; font-family: 'Inter', sans-serif; line-height: 1.3;">AI-Powered Predictions</div>
                </div>
                <div style="flex: 1; min-width: 105px; padding: 16px 12px; background: rgba(15, 23, 42, 0.3); border: 1px solid rgba(255, 255, 255, 0.04); border-radius: 12px; text-align: center; box-shadow: 0 4px 12px rgba(0,0,0,0.15);">
                    <div style="background: rgba(99, 102, 241, 0.1); border: 1px solid rgba(99, 102, 241, 0.2); border-radius: 8px; width: 34px; height: 34px; display: inline-flex; align-items: center; justify-content: center; margin-bottom: 10px; color: #6366f1;">
                        <svg role="img" aria-label="Insights Icon" xmlns="http://www.w3.org/2000/svg" width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M17 21v-2a4 4 0 0 0-4-4H5a4 4 0 0 0-4 4v2"></path><circle cx="9" cy="7" r="4"></circle><path d="M23 21v-2a4 4 0 0 0-3-3.87"></path><path d="M16 3.13a4 4 0 0 1 0 7.75"></path></svg>
                    </div>
                    <div style="font-weight: 600; font-size: 0.75rem; color: #f8fafc; font-family: 'Inter', sans-serif; line-height: 1.3;">Customer Insights</div>
                </div>
                <div style="flex: 1; min-width: 105px; padding: 16px 12px; background: rgba(15, 23, 42, 0.3); border: 1px solid rgba(255, 255, 255, 0.04); border-radius: 12px; text-align: center; box-shadow: 0 4px 12px rgba(0,0,0,0.15);">
                    <div style="background: rgba(99, 102, 241, 0.1); border: 1px solid rgba(99, 102, 241, 0.2); border-radius: 8px; width: 34px; height: 34px; display: inline-flex; align-items: center; justify-content: center; margin-bottom: 10px; color: #6366f1;">
                        <svg role="img" aria-label="Explainable AI Icon" xmlns="http://www.w3.org/2000/svg" width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"></path></svg>
                    </div>
                    <div style="font-weight: 600; font-size: 0.75rem; color: #f8fafc; font-family: 'Inter', sans-serif; line-height: 1.3;">Explainable AI (XAI)</div>
                </div>
                <div style="flex: 1; min-width: 105px; padding: 16px 12px; background: rgba(15, 23, 42, 0.3); border: 1px solid rgba(255, 255, 255, 0.04); border-radius: 12px; text-align: center; box-shadow: 0 4px 12px rgba(0,0,0,0.15);">
                    <div style="background: rgba(99, 102, 241, 0.1); border: 1px solid rgba(99, 102, 241, 0.2); border-radius: 8px; width: 34px; height: 34px; display: inline-flex; align-items: center; justify-content: center; margin-bottom: 10px; color: #6366f1;">
                        <svg role="img" aria-label="Retention Chart Icon" xmlns="http://www.w3.org/2000/svg" width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M3 3v18h18"></path><path d="M18.7 8l-5.1 5.2-2.8-2.7L7 14.3"></path></svg>
                    </div>
                    <div style="font-weight: 600; font-size: 0.75rem; color: #f8fafc; font-family: 'Inter', sans-serif; line-height: 1.3;">Data-Driven Retention</div>
                </div>
            </div>
        </div>
        """
        st.markdown(branding_html.replace("\n", " "), unsafe_allow_html=True)
        
    with col_right:
        # Glassmorphic Login Form
        with st.form("login_form", clear_on_submit=False):
            # Centered Lock graphic header inside card
            lock_header_html = """
            <div style="display: flex; justify-content: center; margin-bottom: 1rem;">
                <div style="background: rgba(99, 102, 241, 0.15); border: 1px solid rgba(99, 102, 241, 0.3); border-radius: 12px; width: 52px; height: 52px; display: flex; align-items: center; justify-content: center; color: #a78bfa; box-shadow: 0 8px 16px rgba(0,0,0,0.3);">
                    <svg role="img" aria-label="Lock Icon" xmlns="http://www.w3.org/2000/svg" width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="3" y="11" width="18" height="11" rx="2" ry="2"></rect><path d="M7 11V7a5 5 0 0 1 10 0v4"></path></svg>
                </div>
            </div>
            <h2 style="text-align: center; margin: 0; font-family: 'Outfit', sans-serif; font-weight: 700; font-size: 1.7rem; color: #f8fafc;">Welcome Back</h2>
            <p style="text-align: center; margin: 0.3rem 0 2rem 0; color: #94a3b8; font-size: 0.88rem; font-family: 'Inter', sans-serif;">Sign in to access your RetainIQ dashboard</p>
            """
            st.markdown(lock_header_html, unsafe_allow_html=True)
            
            # Form Fields: Inputs are empty by default as requested
            st.markdown("<div style='font-family: Inter, sans-serif; font-size: 0.85rem; font-weight: 500; color: #e2e8f0; margin-bottom: 6px;'>Username</div>", unsafe_allow_html=True)
            username = st.text_input("Username", value="", placeholder="Enter your username", label_visibility="collapsed")
            
            st.markdown("<div style='font-family: Inter, sans-serif; font-size: 0.85rem; font-weight: 500; color: #e2e8f0; margin-bottom: 6px; margin-top: 1.2rem;'>Password</div>", unsafe_allow_html=True)
            password = st.text_input("Password", type="password", value="", placeholder="Enter your password", label_visibility="collapsed")
            
            # Columns layout for Remember Me and Forgot Password
            c_chk, c_lnk = st.columns([1.1, 0.9])
            with c_chk:
                remember_me = st.checkbox("Remember Me", value=True)
            with c_lnk:
                st.markdown("<div style='text-align: right; font-family: Inter, sans-serif; font-size: 0.85rem; margin-top: 4px;'><a href='#' style='color: #6366f1; text-decoration: none; font-weight: 500;'>Forgot Password?</a></div>", unsafe_allow_html=True)
            
            submitted = st.form_submit_button("Login to Dashboard   →")
            
            # Divider and visual Token secondary button
            divider_and_sso_html = """
            <div style="display: flex; align-items: center; margin: 1.5rem 0; font-family: 'Inter', sans-serif; font-size: 0.85rem; color: #475569;">
                <div style="flex: 1; height: 1px; background: rgba(255, 255, 255, 0.08);"></div>
                <span style="padding: 0 10px; text-transform: uppercase; font-size: 0.72rem; font-weight: 600; letter-spacing: 1px; color: #64748b;">or</span>
                <div style="flex: 1; height: 1px; background: rgba(255, 255, 255, 0.08);"></div>
            </div>
            
            <div style="border: 1px solid rgba(255, 255, 255, 0.08); border-radius: 8px; padding: 0.75rem 0; text-align: center; font-weight: 600; font-size: 0.9rem; color: #f8fafc; font-family: 'Inter', sans-serif; display: flex; align-items: center; justify-content: center; gap: 8px; background-color: rgba(255,255,255,0.01); transition: all 0.2s; box-shadow: 0 4px 6px rgba(0,0,0,0.1); cursor: pointer;">
                <svg role="img" aria-label="Token Key Shield" xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="#a78bfa" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"></path></svg>
                Login with Token
            </div>
            
            <div style="text-align: center; font-size: 0.78rem; color: #64748b; font-family: 'Inter', sans-serif; margin-top: 1.8rem; line-height: 1.4;">
                By continuing, you agree to our <a href="#" style="color: #94a3b8; text-decoration: none; border-bottom: 1px dotted #94a3b8;">Terms of Service</a> and <a href="#" style="color: #94a3b8; text-decoration: none; border-bottom: 1px dotted #94a3b8;">Privacy Policy</a>
            </div>
            """
            st.markdown(divider_and_sso_html.replace("\n", " "), unsafe_allow_html=True)
            
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
                            
    # 3. Bottom Centered Encrypted Data Banner
    bottom_encrypt_html = """
    <div style="display: flex; justify-content: center; margin-top: 4rem;">
        <div style="display: flex; align-items: center; gap: 8px; padding: 6px 16px; background: rgba(15, 23, 42, 0.4); border: 1px solid rgba(255, 255, 255, 0.05); border-radius: 20px; color: #94a3b8; font-family: 'Inter', sans-serif; font-size: 0.78rem; font-weight: 500;">
            <svg role="img" aria-label="Encryption Padlock" xmlns="http://www.w3.org/2000/svg" width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="#10b981" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><rect x="3" y="11" width="18" height="11" rx="2" ry="2"></rect><path d="M7 11V7a5 5 0 0 1 10 0v4"></path></svg>
            Your data is encrypted and secure
        </div>
    </div>
    """
    st.markdown(bottom_encrypt_html, unsafe_allow_html=True)
    st.stop()
