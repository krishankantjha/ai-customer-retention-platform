import streamlit as st
import time
from frontend.api_client import RetainIQAPIClient

def render_ingestion_view(api_client: RetainIQAPIClient, check_401_callback):
    """
    Renders the Batch Dataset Ingestion view, enabling drag-and-drop CSV uploads 
    and displaying step-by-step pipeline execution tracking.
    """
    st.subheader("Batch Dataset Ingestion")
    st.write("Upload raw telecom churn client lists to trigger dynamic schema validation, feature engineering, and ensemble prediction pipelines.")

    st.markdown("<br>", unsafe_allow_html=True)

    # Styles for premium drag-drop box and stepper
    st.markdown("""
    <style>
        /* Drag-drop wrapper custom theme */
        div[data-testid="stFileUploader"] {
            background: rgba(15, 23, 42, 0.3) !important;
            border: 2px dashed rgba(99, 102, 241, 0.4) !important;
            border-radius: 12px !important;
            padding: 2rem !important;
            transition: all 0.2s ease-in-out !important;
        }
        div[data-testid="stFileUploader"]:hover {
            border-color: #6366f1 !important;
            background: rgba(99, 102, 241, 0.05) !important;
        }
    </style>
    """, unsafe_allow_html=True)

    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"], label_visibility="collapsed")
    
    if uploaded_file is not None:
        st.markdown(f"""
        <div style="background: rgba(99, 102, 241, 0.08); border: 1px solid rgba(99, 102, 241, 0.2); border-radius: 8px; padding: 0.8rem; margin-top: 1rem; color: #cbd5e1; display: flex; align-items: center; justify-content: space-between;">
            <div>
                <span style="font-weight: 700; color: #ffffff;">📄 Selected File:</span> {uploaded_file.name}
            </div>
            <span style="font-size: 0.8rem; color: #94a3b8; font-weight: 500;">{uploaded_file.size / 1024:.1f} KB</span>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        if st.button("Start Ingestion Pipeline", type="primary", use_container_width=True):
            # Render empty visual steps
            progress_container = st.container()
            
            with st.spinner("Registering cohort file..."):
                try:
                    files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "text/csv")}
                    status_code, res_data = api_client.upload_cohort(files)
                    check_401_callback(status_code)
                    
                    if status_code == 202:
                        upload_id = res_data['upload_id']
                        
                        # Interactive Status Tracker loop
                        for _ in range(40):
                            time.sleep(1.5)
                            st_code, st_data = api_client.get_upload_status(upload_id)
                            check_401_callback(st_code)
                            
                            if st_code == 200:
                                status_val = st_data.get("status", "pending")
                                row_count = st_data.get("row_count", 0)
                                err_msg = st_data.get("error_message")
                                
                                # Clear existing elements inside the container and redraw progress steps
                                with progress_container:
                                    st.markdown("### Pipeline Tracking Status")
                                    
                                    # Calculate stepper states
                                    step1_icon, step1_color = "✅", "#10b981"
                                    
                                    if status_val == "pending":
                                        step2_icon, step2_color = "🔄", "#f59e0b"
                                        step3_icon, step3_color = "⏳", "#475569"
                                        step4_icon, step4_color = "⏳", "#475569"
                                        step5_icon, step5_color = "⏳", "#475569"
                                        p_val = 20
                                    elif status_val == "processing":
                                        step2_icon, step2_color = "✅", "#10b981"
                                        step3_icon, step3_color = "🔄", "#f59e0b"
                                        step4_icon, step4_color = "⏳", "#475569"
                                        step5_icon, step5_color = "⏳", "#475569"
                                        p_val = 50
                                    elif status_val == "completed":
                                        step2_icon, step2_color = "✅", "#10b981"
                                        step3_icon, step3_color = "✅", "#10b981"
                                        step4_icon, step4_color = "✅", "#10b981"
                                        step5_icon, step5_color = "✅", "#10b981"
                                        p_val = 100
                                    else: # failed
                                        step2_icon, step2_color = "❌", "#ef4444"
                                        step3_icon, step3_color = "❌", "#ef4444"
                                        step4_icon, step4_color = "❌", "#ef4444"
                                        step5_icon, step5_color = "❌", "#ef4444"
                                        p_val = 0
                                        
                                    st.progress(p_val)
                                    
                                    st.markdown(f"""
                                    <div style="background: rgba(15, 23, 42, 0.45); border: 1px solid rgba(255, 255, 255, 0.08); border-radius: 12px; padding: 1.5rem; backdrop-filter: blur(20px); margin-top: 1rem;">
                                        <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 0.8rem;">
                                            <span style="font-size: 1.1rem; color: {step1_color};">{step1_icon}</span>
                                            <span style="font-weight: 600; color: #f8fafc;">File Uploaded & Registered</span>
                                        </div>
                                        <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 0.8rem;">
                                            <span style="font-size: 1.1rem; color: {step2_color};">{step2_icon}</span>
                                            <span style="font-weight: 600; color: #f8fafc;">Schema Validation Checks</span>
                                        </div>
                                        <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 0.8rem;">
                                            <span style="font-size: 1.1rem; color: {step3_color};">{step3_icon}</span>
                                            <span style="font-weight: 600; color: #f8fafc;">Feature Engineering Transformations</span>
                                        </div>
                                        <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 0.8rem;">
                                            <span style="font-size: 1.1rem; color: {step4_color};">{step4_icon}</span>
                                            <span style="font-weight: 600; color: #f8fafc;">Model Ensemble Predictions</span>
                                        </div>
                                        <div style="display: flex; align-items: center; gap: 10px;">
                                            <span style="font-size: 1.1rem; color: {step5_color};">{step5_icon}</span>
                                            <span style="font-weight: 600; color: #f8fafc;">Segmentation & Explainability Analysis</span>
                                        </div>
                                    </div>
                                    """, unsafe_allow_html=True)
                                    
                                if status_val == "completed":
                                    st.cache_data.clear()
                                    st.success(f"Ingestion pipeline completed successfully! Mapped {row_count} customer accounts.")
                                    st.balloons()
                                    break
                                elif status_val == "failed":
                                    st.error(f"Ingestion pipeline failed: {err_msg or 'Unknown core processing failure'}")
                                    break
                            else:
                                pass
                        else:
                            st.warning("Ingestion process timed out. Pipeline execution remains running in the background.")
                    else:
                        st.error(f"Ingestion request rejected: {res_data.get('detail', 'Unknown error')}")
                except Exception as e:
                    st.error(f"Inference pipeline execution error: {e}")
