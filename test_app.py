import os
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px


@st.cache_data(show_spinner=False)
def load_csv(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def format_currency(x: float) -> str:
    try:
        return f"£{x:,.0f}"
    except Exception:
        return "-"


def executive_summary(df_segments: pd.DataFrame):
    st.subheader("Executive Summary")
    if df_segments.empty:
        st.info("Segmentation file not found. Ensure data/processed/kmeans_customer_segments.csv exists.")
        return
    seg_counts = df_segments['Segment_Name'].value_counts()
    seg_revenue = df_segments.groupby('Segment_Name')['monetary'].sum().sort_values(ascending=False)
    top_size = seg_counts.idxmax() if not seg_counts.empty else None
    top_rev = seg_revenue.idxmax() if not seg_revenue.empty else None
    c1, c2, c3 = st.columns(3)
    c1.metric("Total Customers", f"{len(df_segments):,}")
    c2.metric("Total Revenue", format_currency(df_segments['monetary'].sum()))
    c3.metric("Segments", f"{df_segments['Segment_Name'].nunique()}")
    st.markdown(f"- **Largest segment**: {top_size or 'N/A'}")
    st.markdown(f"- **Top revenue segment**: {top_rev or 'N/A'}")
    st.caption("Goal: Prioritize retention for at-risk and monetize loyal/high-value segments with targeted actions.")


def segmentation_overview(df_segments: pd.DataFrame):
    st.subheader("Segmentation Overview")
    if df_segments.empty:
        st.info("No segmentation data available.")
        return
    # Filters
    segs = sorted(df_segments['Segment_Name'].dropna().unique().tolist())
    selected = st.multiselect("Segments", segs, default=segs)
    dff = df_segments[df_segments['Segment_Name'].isin(selected)]
    # KPIs
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Customers", f"{len(dff):,}")
    c2.metric("Revenue", format_currency(dff['monetary'].sum()))
    c3.metric("Avg Recency", f"{dff['recency_days'].mean():.1f}")
    c4.metric("Avg Frequency", f"{dff['frequency'].mean():.2f}")
    # Charts
    agg = dff.groupby('Segment_Name').agg(Customers=('CustomerID', 'count'), Revenue=('monetary', 'sum')).reset_index()
    c5, c6 = st.columns(2)
    with c5:
        st.plotly_chart(px.bar(agg.sort_values('Customers', ascending=False), x='Segment_Name', y='Customers', color='Segment_Name', title='Customers by Segment'), use_container_width=True)
    with c6:
        st.plotly_chart(px.pie(agg, names='Segment_Name', values='Revenue', title='Revenue by Segment', hole=0.4), use_container_width=True)
    st.dataframe(agg.sort_values('Revenue', ascending=False), use_container_width=True)


def segment_drilldown(df_segments: pd.DataFrame):
    st.subheader("Segment Drilldown")
    if df_segments.empty:
        st.info("No segmentation data available.")
        return
    segs = sorted(df_segments['Segment_Name'].dropna().unique().tolist())
    seg = st.selectbox("Select segment", segs)
    sdf = df_segments[df_segments['Segment_Name'] == seg]
    c1, c2, c3 = st.columns(3)
    c1.metric("Customers", f"{len(sdf):,}")
    c2.metric("Revenue", format_currency(sdf['monetary'].sum()))
    c3.metric("Avg Monetary", format_currency(sdf['monetary'].mean()))
    st.plotly_chart(px.histogram(sdf, x='monetary', nbins=50, title=f'Monetary Distribution - {seg}'), use_container_width=True)
    st.dataframe(sdf.nlargest(100, 'monetary')[['CustomerID', 'recency_days', 'frequency', 'monetary']], use_container_width=True)


def raw_insights(df_raw: pd.DataFrame):
    st.subheader("Raw Data Insights (Gender & Location)")
    if df_raw.empty:
        st.info("Raw data not found: data/bank_data_C.csv")
        return
    # Normalize needed columns
    if 'TransactionAmount (INR)' in df_raw.columns and 'TransactionAmount' not in df_raw.columns:
        df_raw['TransactionAmount'] = df_raw['TransactionAmount (INR)']
    # Gender
    if 'CustGender' in df_raw.columns:
        g = df_raw.groupby('CustGender').agg(Customers=('CustomerID', 'nunique'), Transactions=('TransactionID', 'count'), Revenue=('TransactionAmount', 'sum')).reset_index()
        c1, c2 = st.columns(2)
        with c1:
            st.dataframe(g.sort_values('Revenue', ascending=False), use_container_width=True)
        with c2:
            st.plotly_chart(px.pie(g, names='CustGender', values='Revenue', title='Revenue by Gender', hole=0.4), use_container_width=True)
    else:
        st.info("CustGender column missing.")
    # Location
    if 'CustLocation' in df_raw.columns:
        l = df_raw.groupby('CustLocation').agg(Customers=('CustomerID', 'nunique'), Transactions=('TransactionID', 'count'), Revenue=('TransactionAmount', 'sum')).reset_index()
        top_l = l.nlargest(20, 'Revenue')
        st.plotly_chart(px.bar(top_l, x='CustLocation', y='Revenue', title='Top 20 Locations by Revenue'), use_container_width=True)
        st.dataframe(top_l, use_container_width=True)
    else:
        st.info("CustLocation column missing.")


def recommended_actions():
    st.subheader("Recommended Actions")
    st.markdown("- Champions: VIP perks, early access, premium support")
    st.markdown("- High Value: personalized wealth offers, cross-sell credit/investments")
    st.markdown("- Loyal: loyalty rewards, referrals, tiered benefits")
    st.markdown("- At Risk: win-back incentives, service feedback, reactivation journeys")
    st.markdown("- Standard: education, product fit nudges, A/B tested offers")


def main():
    st.set_page_config(page_title="BankTrust Segmentation – Test", page_icon="📈", layout="wide")
    st.title("BankTrust: RFM-Based Customer Segmentation (Test App)")
    st.caption("Business goal: Reduce churn, grow CLV, and maximize ROI via targeted strategies.")

    base = os.path.join('data', 'processed')
    df_segments = load_csv(os.path.join(base, 'kmeans_customer_segments.csv'))
    df_raw = load_csv(os.path.join('data', 'bank_data_C.csv'))

    section = st.sidebar.radio(
        "Navigate",
        ["Summary", "Segments", "Drilldown", "Raw Insights", "Actions"],
        index=0
    )

    if section == "Summary":
        executive_summary(df_segments)
    elif section == "Segments":
        segmentation_overview(df_segments)
    elif section == "Drilldown":
        segment_drilldown(df_segments)
    elif section == "Raw Insights":
        raw_insights(df_raw)
    elif section == "Actions":
        recommended_actions()

    st.info("Run: streamlit run test_app.py")


if __name__ == "__main__":
    main()


