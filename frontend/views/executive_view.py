import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from frontend.utils import get_risk_cat
from backend.app.database.session import SessionLocal
from backend.app.database.models.customer import Customer
from backend.app.database.models.prediction import Prediction

def render_executive_view(
    cohort_data: list,
    plays_data_raw: list,
    primary_color_hex: str,
    secondary_color_hex: str,
    danger_color_hex: str
):
    """
    Renders the tabbed Analytics Dashboard page, displaying Contract Analysis,
    Payment Analysis, Revenue, Tenure, and Cohorts.
    """
    if not cohort_data:
        st.info("No cohort records available to analyze. Please ingest a dataset first.")
        st.stop()
        
    df = pd.DataFrame(cohort_data)
    
    # Classify risk categories
    df["risk_category"] = df["churn_probability"].apply(get_risk_cat)

    # 1. Row of top-level MRR KPIs
    total_rev = df["monthly_charges"].sum()
    var_val = df[df["churn_probability"] >= 0.25]["monthly_charges"].sum()
    high_count = (df["churn_probability"] >= 0.50).sum()

    ek1, ek2, ek3, ek4 = st.columns(4)
    with ek1:
        st.markdown(f"""
        <div style="background: rgba(15, 23, 42, 0.45); border: 1px solid rgba(255, 255, 255, 0.08); border-top: 3.5px solid {primary_color_hex}; border-radius: 12px; padding: 1rem; backdrop-filter: blur(20px);">
            <div style="color: #94a3b8; font-size: 0.75rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em;">Cohort Count</div>
            <div style="font-size: 1.6rem; font-weight: 800; color: #ffffff; font-family: 'Outfit'; margin-top: 0.2rem;">{len(df):,}</div>
        </div>
        """, unsafe_allow_html=True)
    with ek2:
        st.markdown(f"""
        <div style="background: rgba(15, 23, 42, 0.45); border: 1px solid rgba(255, 255, 255, 0.08); border-top: 3.5px solid {secondary_color_hex}; border-radius: 12px; padding: 1rem; backdrop-filter: blur(20px);">
            <div style="color: #94a3b8; font-size: 0.75rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em;">High Risk Count</div>
            <div style="font-size: 1.6rem; font-weight: 800; color: #ffffff; font-family: 'Outfit'; margin-top: 0.2rem;">{high_count:,}</div>
        </div>
        """, unsafe_allow_html=True)
    with ek3:
        st.markdown(f"""
        <div style="background: rgba(15, 23, 42, 0.45); border: 1px solid rgba(255, 255, 255, 0.08); border-top: 3.5px solid {danger_color_hex}; border-radius: 12px; padding: 1rem; backdrop-filter: blur(20px);">
            <div style="color: #94a3b8; font-size: 0.75rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em;">Value-at-Risk</div>
            <div style="font-size: 1.6rem; font-weight: 800; color: #ffffff; font-family: 'Outfit'; margin-top: 0.2rem;">${var_val:,.2f}</div>
        </div>
        """, unsafe_allow_html=True)
    with ek4:
        st.markdown(f"""
        <div style="background: rgba(15, 23, 42, 0.45); border: 1px solid rgba(255, 255, 255, 0.08); border-top: 3.5px solid #10b981; border-radius: 12px; padding: 1rem; backdrop-filter: blur(20px);">
            <div style="color: #94a3b8; font-size: 0.75rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em;">Total Cohort MRR</div>
            <div style="font-size: 1.6rem; font-weight: 800; color: #ffffff; font-family: 'Outfit'; margin-top: 0.2rem;">${total_rev:,.2f}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # 2. Main tabs setup
    t_rev, t_contract, t_payment, t_tenure, t_cohorts = st.tabs([
        "💰 Revenue",
        "📅 Contract Analysis",
        "💳 Payment Analysis",
        "⏱️ Tenure Relationship",
        "👥 Cohort Profiles"
    ])

    with t_rev:
        st.markdown("### Revenue at Risk Analytics")
        st.write("Box plot analysis of monthly billing charges across risk tiers and recommended save play campaigns.")
        
        c1, c2 = st.columns([1, 1.2])
        with c1:
            fig_box = px.box(
                df,
                x="risk_category",
                y="monthly_charges",
                color="risk_category",
                category_orders={"risk_category": ["Low Risk", "Medium Risk", "High Risk"]},
                color_discrete_map={"High Risk": danger_color_hex, "Medium Risk": secondary_color_hex, "Low Risk": "#10b981"},
                labels={"risk_category": "Risk Category", "monthly_charges": "Monthly Charges ($)"}
            )
            fig_box.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(family="Inter, sans-serif", color="#94a3b8"),
                margin=dict(t=10, b=20, l=20, r=20),
                showlegend=False,
                height=300
            )
            st.plotly_chart(fig_box, use_container_width=True)
            
        with c2:
            if plays_data_raw:
                table_df = pd.DataFrame(plays_data_raw).rename(columns={
                    "campaign": "Campaign Name",
                    "recommendation_count": "Targets",
                    "average_estimated_impact": "Expected Churn Reduction"
                })
                table_df["Expected Churn Reduction"] = table_df["Expected Churn Reduction"].apply(lambda v: f"{v*100:.1f}%")
                st.write("**Recommended Campaigns Reach**")
                st.dataframe(table_df.head(4), use_container_width=True, hide_index=True)

    with t_contract:
        st.markdown("### Contract Exposure Analysis")
        st.write("Examine customer count and risk concentration split by contract duration.")
        
        fig_contract = px.histogram(
            df,
            x="contract",
            color="risk_category",
            barmode="stack",
            category_orders={"risk_category": ["Low Risk", "Medium Risk", "High Risk"]},
            color_discrete_map={"High Risk": danger_color_hex, "Medium Risk": secondary_color_hex, "Low Risk": "#10b981"},
            labels={"contract": "Contract Type", "count": "Customer Count"},
        )
        fig_contract.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(family="Inter, sans-serif", color="#94a3b8"),
            margin=dict(t=10, b=20, l=20, r=20),
            height=300,
            legend_title_text="Risk Level"
        )
        st.plotly_chart(fig_contract, use_container_width=True)

    with t_payment:
        st.markdown("### Payment Method Analysis")
        st.write("Understand how monthly billing methods map to customer churn risk concentration.")
        
        # Load payment details dynamically from database
        db = SessionLocal()
        try:
            db_res = db.query(Customer.payment_method, Prediction.churn_probability)\
                .join(Prediction, Customer.id == Prediction.customer_id).all()
            
            if db_res:
                pay_df = pd.DataFrame(db_res, columns=["payment_method", "churn_probability"])
                pay_df["risk_category"] = pay_df["churn_probability"].apply(get_risk_cat)
                
                fig_pay = px.histogram(
                    pay_df,
                    x="payment_method",
                    color="risk_category",
                    barmode="group",
                    category_orders={"risk_category": ["Low Risk", "Medium Risk", "High Risk"]},
                    color_discrete_map={"High Risk": danger_color_hex, "Medium Risk": secondary_color_hex, "Low Risk": "#10b981"},
                    labels={"payment_method": "Billing Method", "count": "Customer Count"},
                )
                fig_pay.update_layout(
                    template="plotly_dark",
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(family="Inter, sans-serif", color="#94a3b8"),
                    margin=dict(t=10, b=20, l=20, r=20),
                    height=300,
                    legend_title_text="Risk Level"
                )
                st.plotly_chart(fig_pay, use_container_width=True)
            else:
                st.info("No payment details found.")
        except Exception as e:
            st.error(f"Error querying payment details: {e}")
        finally:
            db.close()

    with t_tenure:
        st.markdown("### Tenure vs. Churn Probability")
        st.write("Visual relationship between customer tenure (months) and predicted churn probability with OLS trendline overlay.")
        
        x_val = df["tenure"]
        y_val = df["churn_probability"]
        
        # Fit trendline manually
        slope, intercept = np.polyfit(x_val, y_val, 1)
        trend_x = np.linspace(x_val.min(), x_val.max(), 100)
        trend_y = slope * trend_x + intercept
        
        fig_scat = px.scatter(
            df,
            x="tenure",
            y="churn_probability",
            color="churn_probability",
            color_continuous_scale="Reds",
            labels={"tenure": "Tenure (Months)", "churn_probability": "Churn Probability (%)"},
        )
        fig_scat.add_traces(
            go.Scatter(
                x=trend_x,
                y=trend_y,
                mode="lines",
                name="OLS Trendline",
                line=dict(color=primary_color_hex, width=2.5, dash="dash")
            )
        )
        fig_scat.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(family="Inter, sans-serif", color="#94a3b8"),
            margin=dict(t=10, b=20, l=20, r=20),
            height=300,
            coloraxis_showscale=False
        )
        st.plotly_chart(fig_scat, use_container_width=True)

    with t_cohorts:
        st.markdown("### Tenure Cohorts Performance")
        st.write("Understand churn concentration grouped by tenure cohorts.")
        
        # Segment into cohorts
        def get_tenure_cohort(t):
            if t < 12: return "0-12 mos (New)"
            elif t < 24: return "12-24 mos"
            elif t < 48: return "24-48 mos"
            return "48+ mos (Loyal)"
            
        df["tenure_cohort"] = df["tenure"].apply(get_tenure_cohort)
        
        cohort_summary = df.groupby("tenure_cohort").agg(
            total_count=("customer_id", "count"),
            avg_risk=("churn_probability", "mean"),
            var_charges=("monthly_charges", lambda x: df.loc[x.index, "monthly_charges"][df.loc[x.index, "churn_probability"] >= 0.25].sum())
        ).reset_index()
        
        cohort_summary["Avg Risk (%)"] = cohort_summary["avg_risk"].apply(lambda r: f"{r*100:.1f}%")
        cohort_summary["Value at Risk"] = cohort_summary["var_charges"].apply(lambda v: f"${v:,.2f}")
        
        st.table(cohort_summary[["tenure_cohort", "total_count", "Avg Risk (%)", "Value at Risk"]].set_index("tenure_cohort").rename(columns={
            "total_count": "Customer Count"
        }))
