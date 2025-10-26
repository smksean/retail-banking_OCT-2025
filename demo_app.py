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


def fmt_money(x: float) -> str:
    try:
        return f"£{x:,.0f}"
    except Exception:
        return "-"


def section_header(title: str, caption: str | None = None):
    st.subheader(title)
    if caption:
        st.caption(caption)


def page_overview(df: pd.DataFrame):
    section_header("Executive Overview", "Segmentation outcomes and key KPIs at a glance")
    if df.empty:
        st.info("No segmentation data available. Upload or generate kmeans_customer_segments.csv.")
        return
    # KPIs
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Customers", f"{len(df):,}")
    c2.metric("Revenue", fmt_money(df['monetary'].sum()))
    c3.metric("Avg Recency", f"{df['recency_days'].mean():.1f}")
    c4.metric("Avg Frequency", f"{df['frequency'].mean():.2f}")
    # Customers and revenue by segment
    agg = df.groupby('Segment_Name').agg(Customers=('CustomerID','count'), Revenue=('monetary','sum')).reset_index()
    agg['Revenue_%'] = (agg['Revenue']/agg['Revenue'].sum()*100).round(1)
    a1, a2 = st.columns(2)
    with a1:
        st.plotly_chart(px.bar(agg.sort_values('Customers', ascending=False), x='Segment_Name', y='Customers', color='Segment_Name', title='Customers by Segment'), use_container_width=True)
    with a2:
        st.plotly_chart(px.pie(agg, names='Segment_Name', values='Revenue', hole=0.4, title='Revenue Share by Segment'), use_container_width=True)
    st.dataframe(agg.sort_values('Revenue', ascending=False), use_container_width=True)


def page_segments(df: pd.DataFrame):
    section_header("Segments", "Profiles, KPIs, and distributions by segment")
    if df.empty:
        st.info("No segmentation data available.")
        return
    segs = sorted(df['Segment_Name'].dropna().unique().tolist())
    selected = st.multiselect("Filter segments", segs, default=segs)
    dff = df[df['Segment_Name'].isin(selected)]
    prof = dff.groupby('Segment_Name').agg(
        Customers=('CustomerID','count'),
        Recency_Mean=('recency_days','mean'),
        Frequency_Mean=('frequency','mean'),
        Monetary_Mean=('monetary','mean'),
        Revenue=('monetary','sum')
    ).round(2)
    prof['Revenue_%'] = (prof['Revenue']/prof['Revenue'].sum()*100).round(1)
    st.dataframe(prof.sort_values('Revenue', ascending=False), use_container_width=True)

    fig = px.scatter(prof.reset_index(), x='Frequency_Mean', y='Recency_Mean', size='Monetary_Mean', color='Segment_Name', title='Segment Profile Map (Bubble = Monetary Mean)')
    st.plotly_chart(fig, use_container_width=True)

    s1, s2, s3 = st.columns(3)
    s1.plotly_chart(px.histogram(dff, x='recency_days', nbins=50, color='Segment_Name', title='Recency'), use_container_width=True)
    s2.plotly_chart(px.histogram(dff, x='frequency', nbins=50, color='Segment_Name', title='Frequency'), use_container_width=True)
    s3.plotly_chart(px.histogram(dff, x='monetary', nbins=50, color='Segment_Name', title='Monetary'), use_container_width=True)


def page_drilldown(df: pd.DataFrame):
    section_header("Drilldown", "Deep dive into a single segment")
    if df.empty:
        st.info("No segmentation data available.")
        return
    segs = sorted(df['Segment_Name'].dropna().unique().tolist())
    seg = st.selectbox("Choose segment", segs)
    sdf = df[df['Segment_Name'] == seg]
    c1, c2, c3 = st.columns(3)
    c1.metric("Customers", f"{len(sdf):,}")
    c2.metric("Revenue", fmt_money(sdf['monetary'].sum()))
    c3.metric("Avg Monetary", fmt_money(sdf['monetary'].mean()))
    st.dataframe(sdf.nlargest(100, 'monetary')[['CustomerID','recency_days','frequency','monetary']], use_container_width=True)
    st.download_button("Download segment (CSV)", data=sdf.to_csv(index=False).encode('utf-8'), file_name=f"segment_{seg.replace(' ','_').lower()}.csv", mime='text/csv')


def page_actions(df: pd.DataFrame):
    section_header("Actionable Recommendations", "Plan campaigns and export target lists")
    if df.empty:
        st.info("No data to plan actions.")
        return
    segs = sorted(df['Segment_Name'].dropna().unique().tolist())
    seg = st.selectbox("Target segment", segs)
    sdf = df[df['Segment_Name'] == seg]
    st.caption(f"Segment size: {len(sdf):,}")
    c1, c2, c3 = st.columns(3)
    with c1:
        target_size = st.slider("Target customers", 0, int(len(sdf)), min(5000, int(len(sdf))))
    with c2:
        uplift_pct = st.number_input("Expected uplift %", 0.0, 100.0, 5.0, 0.5)
    with c3:
        budget = st.number_input("Campaign budget (£)", 0.0, 1e9, 10000.0, 500.0)
    avg_spend = sdf['monetary'].mean()
    baseline = avg_spend * target_size
    incremental = baseline * (uplift_pct/100.0)
    roi = (incremental - budget) / budget * 100 if budget > 0 else float('inf')
    k1, k2, k3 = st.columns(3)
    k1.metric("Baseline revenue", fmt_money(baseline))
    k2.metric("Incremental revenue", fmt_money(incremental))
    k3.metric("ROI % (est)", f"{roi:.1f}%")
    top_targets = sdf.nlargest(target_size, 'monetary')[['CustomerID','recency_days','frequency','monetary']]
    st.dataframe(top_targets.head(50), use_container_width=True)
    st.download_button("Download target list (CSV)", data=top_targets.to_csv(index=False).encode('utf-8'), file_name=f"targets_{seg.replace(' ','_').lower()}.csv", mime='text/csv')


def page_raw(base_dir: str):
    section_header("Raw Data Insights", "Gender, location, and simple summaries")
    raw = load_csv(os.path.join('data', 'bank_data_C.csv'))
    if raw.empty:
        st.info("Raw data not found or could not be loaded: data/bank_data_C.csv")
        return
    if 'TransactionAmount (INR)' in raw.columns and 'TransactionAmount' not in raw.columns:
        raw['TransactionAmount'] = raw['TransactionAmount (INR)']
    # Gender
    if 'CustGender' in raw.columns:
        g = raw.groupby('CustGender').agg(Customers=('CustomerID','nunique'), Transactions=('TransactionID','count'), Revenue=('TransactionAmount','sum')).reset_index()
        c1, c2 = st.columns(2)
        c1.dataframe(g.sort_values('Revenue', ascending=False), use_container_width=True)
        c2.plotly_chart(px.pie(g, names='CustGender', values='Revenue', title='Revenue by Gender', hole=0.4), use_container_width=True)
    else:
        st.info("CustGender column missing.")
    # Location
    if 'CustLocation' in raw.columns:
        l = raw.groupby('CustLocation').agg(Customers=('CustomerID','nunique'), Transactions=('TransactionID','count'), Revenue=('TransactionAmount','sum')).reset_index()
        top_l = l.nlargest(20, 'Revenue')
        st.plotly_chart(px.bar(top_l, x='CustLocation', y='Revenue', title='Top 20 Locations by Revenue'), use_container_width=True)
        st.dataframe(top_l, use_container_width=True)
    else:
        st.info("CustLocation column missing.")


def page_model(base_dir: str):
    section_header("Model Summary", "Metrics and cluster profiles")
    ms = load_csv(os.path.join('data','processed','kmeans_model_summary.csv'))
    cp = load_csv(os.path.join('data','processed','cluster_profiles.csv'))
    if not ms.empty:
        st.dataframe(ms, use_container_width=True)
    else:
        st.info("Model summary not found.")
    if not cp.empty:
        st.dataframe(cp, use_container_width=True)
    else:
        st.info("Cluster profiles not found.")


def main():
    st.set_page_config(page_title="BankTrust Segmentation Demo", page_icon="📈", layout="wide")
    st.title("BankTrust: RFM Segmentation Demo App")
    st.caption("Explore segments, raw data, and model outcomes; plan actionable campaigns.")

    segments = load_csv(os.path.join('data','processed','kmeans_customer_segments.csv'))

    nav = st.sidebar.radio("Navigate", ["Overview","Segments","Drilldown","Actions","Raw Data","Model"], index=0)
    if nav == "Overview":
        page_overview(segments)
    elif nav == "Segments":
        page_segments(segments)
    elif nav == "Drilldown":
        page_drilldown(segments)
    elif nav == "Actions":
        page_actions(segments)
    elif nav == "Raw Data":
        page_raw('data')
    elif nav == "Model":
        page_model('data')

    st.info("Run: streamlit run demo_app.py")


if __name__ == "__main__":
    main()


