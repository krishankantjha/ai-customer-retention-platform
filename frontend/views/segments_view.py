import streamlit as st
import pandas as pd
import plotly.express as px
from frontend.utils import get_risk_cat

def render_segments_view(
    cohort_data: list,
    primary_color_hex: str,
    secondary_color_hex: str,
    danger_color_hex: str
):
    """
    Renders the Customer Segments page.
    Groups predictions dynamically using the K-Means clusters and lists behavioral personas.
    """
    st.subheader("Customer Segments & Personas")
    st.write("Understand customer personas and behavior categories mapped using K-Means++ clustering on behavioral coordinates.")

    if not cohort_data:
        st.info("No cohort records available to segment. Go to Upload Dataset to add new data.")
        st.stop()

    df = pd.DataFrame(cohort_data)
    
    # Check if cluster column exists
    if "cluster" not in df.columns or df["cluster"].isnull().all():
        st.warning("No K-Means cluster assignments found in customer predictions. Please re-run batch predictions.")
        st.stop()

    # Pre-aggregate stats
    total_customers = len(df)

    # Classify churn risk categories for breakdown
    df["risk_level"] = df["churn_probability"].apply(get_risk_cat)

    st.markdown("<br>", unsafe_allow_html=True)

    # Grid of Persona Cards
    st.markdown("### Behavioral Personas Overview")
    
    # Dynamic computation of cluster stats
    stats_by_cluster = df.groupby("cluster").agg(
        avg_tenure=("tenure", "mean"),
        avg_charges=("monthly_charges", "mean"),
        avg_risk=("churn_probability", "mean"),
        count=("customer_id", "count")
    ).to_dict("index")

    # Let's map cluster IDs to definitions from kmeans_personas.md
    persona_mapping = {
        0: {
            "title": "Moderate-Value Core (Cluster 0)",
            "icon": "👤",
            "desc": "Medium-tenure customers paying low-to-moderate monthly charges with moderate ecosystem services. This represents your budget-conscious core user base.",
            "save_play": "Trigger Auto-Pay conversion and cross-sell technical security add-ons to improve retention friction.",
            "color": primary_color_hex
        },
        1: {
            "title": "Premium High-Value (Cluster 1)",
            "icon": "👑",
            "desc": "Long-tenure customers with high ecosystem service counts and high monthly billing rates. This is your most valuable premium group.",
            "save_play": "Ensure high-priority VIP customer support. Check fiber router performance and offer loyalty credits proactively.",
            "color": "#c084fc"
        },
        2: {
            "title": "New Churn-Risk Users (Cluster 2)",
            "icon": "⚠️",
            "desc": "Short-tenure customers with high initial monthly charges, short contract types, and low ecosystem subscription counts. This represents your highest churn-risk group.",
            "save_play": "Prioritize direct welcome onboarding check-ins, rate audits, and transition them to long-term contract lock-in campaigns.",
            "color": danger_color_hex
        }
    }

    # Render Persona Grid
    cols = st.columns(3)
    for c_id in [0, 1, 2]:
        col = cols[c_id]
        meta = persona_mapping.get(c_id, {
            "title": f"Cluster {c_id}",
            "icon": "👥",
            "desc": "Behavioral cluster segment",
            "save_play": "Check account details.",
            "color": "#cbd5e1"
        })
        
        c_stats = stats_by_cluster.get(c_id, {"count": 0, "avg_risk": 0.0, "avg_tenure": 0.0, "avg_charges": 0.0})
        count_val = c_stats["count"]
        pct = (count_val / total_customers) * 100 if total_customers > 0 else 0.0
        
        col.markdown(f"""
        <div class="glass-card" style="border-top: 3px solid {meta['color']}; height: 380px; display: flex; flex-direction: column; justify-content: space-between; margin-bottom: 0px !important;">
            <div>
                <div style="font-size: 1.8rem; margin-bottom: 0.5rem;">{meta['icon']}</div>
                <div style="font-weight: 700; color: #f8fafc; font-size: 1.15rem; margin-bottom: 0.2rem;">{meta['title']}</div>
                <div style="color: {meta['color']}; font-weight: 600; font-size: 0.85rem; margin-bottom: 0.8rem;">
                    {count_val:,} customers ({pct:.1f}%)
                </div>
                <p style="color: #cbd5e1; font-size: 0.85rem; line-height: 1.4; margin: 0 0 1rem 0;">{meta['desc']}</p>
            </div>
            <div style="border-top: 1px solid rgba(255, 255, 255, 0.05); padding-top: 0.8rem;">
                <div style="font-weight: 600; font-size: 0.78rem; color: #94a3b8; text-transform: uppercase; margin-bottom: 4px;">Save Play Campaign</div>
                <div style="font-size: 0.82rem; color: #f1f5f9; line-height: 1.3;">{meta['save_play']}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br><br>", unsafe_allow_html=True)

    # Secondary layout row: Cluster Distribution Pie and Characteristics Table
    col_chart, col_table = st.columns([1, 1.2])

    with col_chart:
        st.markdown("### Cluster Distribution")
        # Prepare plot data
        plot_df = df.groupby("cluster").size().reset_index(name="counts")
        plot_df["Segment"] = plot_df["cluster"].map(lambda x: persona_mapping.get(x, {}).get("title", f"Cluster {x}"))
        
        fig = px.pie(
            plot_df,
            names="Segment",
            values="counts",
            color="Segment",
            color_discrete_map={
                persona_mapping[0]["title"]: primary_color_hex,
                persona_mapping[1]["title"]: "#c084fc",
                persona_mapping[2]["title"]: danger_color_hex
            },
            hole=0.45
        )
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(family="Inter, sans-serif", color="#94a3b8"),
            margin=dict(t=10, b=10, l=10, r=10),
            height=280,
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5)
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_table:
        st.markdown("### Characteristics Comparison")
        # Build comparison grid
        grid_data = []
        for c_id in sorted(stats_by_cluster.keys()):
            c_stats = stats_by_cluster[c_id]
            meta = persona_mapping.get(c_id, {"title": f"Cluster {c_id}"})
            grid_data.append({
                "Persona Segment": meta["title"],
                "Customer Count": f"{c_stats['count']:,}",
                "Avg Tenure": f"{c_stats['avg_tenure']:.1f} mos",
                "Avg Monthly Bill": f"${c_stats['avg_charges']:.2f}",
                "Avg Churn Risk": f"{c_stats['avg_risk']*100:.1f}%"
            })
        
        st.table(pd.DataFrame(grid_data).set_index("Persona Segment"))
