import os
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px


@st.cache_data(show_spinner=False)
def load_data(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        st.error(f"Data not found at '{path}'. Make sure to run the notebook to generate it.")
        return pd.DataFrame()
    df = pd.read_csv(path)
    # Normalize expected columns
    expected_cols = {
        'CustomerID': 'CustomerID',
        'recency_days': 'recency_days',
        'frequency': 'frequency',
        'monetary': 'monetary',
        'Cluster': 'Cluster',
        'Segment_Name': 'Segment_Name'
    }
    missing = [c for c in expected_cols if c not in df.columns]
    if missing:
        st.warning(f"Missing columns in data: {missing}")
    return df


def format_currency(x: float) -> str:
    try:
        return f"£{x:,.0f}"
    except Exception:
        return "-"


def download_button(label: str, df: pd.DataFrame, filename: str):
    if df.empty:
        st.warning("Nothing to download for the current selection.")
        return
    st.download_button(
        label=label,
        data=df.to_csv(index=False).encode('utf-8'),
        file_name=filename,
        mime='text/csv'
    )


@st.cache_data(show_spinner=False)
def load_cluster_profiles(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame()
    return pd.read_csv(path)


@st.cache_data(show_spinner=False)
def load_model_summary(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame()
    return pd.read_csv(path)


@st.cache_data(show_spinner=False)
def load_transactions(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame()
    df = pd.read_csv(path)
    # Try to parse date if exists
    for col in ['TransactionDate', 'transaction_date', 'date']:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')
            break
    return df


def sidebar_filters(df: pd.DataFrame) -> pd.DataFrame:
    st.sidebar.header("Filters")
    segments = sorted(df['Segment_Name'].dropna().unique().tolist()) if not df.empty else []
    selected_segments = st.sidebar.multiselect("Segments", segments, default=segments)
    min_tx, max_tx = (0.0, float(df['monetary'].max())) if not df.empty else (0.0, 0.0)
    amt_range = st.sidebar.slider("Total Spent (Monetary) range", min_value=float(0.0), max_value=float(max_tx), value=(float(0.0), float(max_tx)))
    freq_max = float(df['frequency'].max()) if not df.empty else 0.0
    freq_range = st.sidebar.slider("Frequency range", min_value=float(0.0), max_value=float(freq_max), value=(float(0.0), float(freq_max)))
    rec_max = float(df['recency_days'].max()) if not df.empty else 0.0
    rec_range = st.sidebar.slider("Recency (days) range", min_value=float(0.0), max_value=float(rec_max), value=(float(0.0), float(rec_max)))

    if df.empty:
        return df

    mask = df['Segment_Name'].isin(selected_segments)
    mask &= df['monetary'].between(amt_range[0], amt_range[1])
    mask &= df['frequency'].between(freq_range[0], freq_range[1])
    mask &= df['recency_days'].between(rec_range[0], rec_range[1])
    return df[mask]


def kpi_cards(df: pd.DataFrame):
    total_customers = len(df)
    total_revenue = df['monetary'].sum()
    avg_recency = df['recency_days'].mean()
    avg_frequency = df['frequency'].mean()
    avg_monetary = df['monetary'].mean()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Customers", f"{total_customers:,}")
    c2.metric("Total Revenue", format_currency(total_revenue))
    c3.metric("Avg Recency (days)", f"{avg_recency:.1f}")
    c4.metric("Avg Frequency", f"{avg_frequency:.2f}")
    st.caption("KPIs update with filters; Monetary assumed in GBP (£). Adjust in app if needed.")


def segment_distribution(df: pd.DataFrame):
    seg = df.groupby('Segment_Name').agg(Customers=('CustomerID', 'count'), Revenue=('monetary', 'sum')).reset_index()
    seg['Revenue_%'] = seg['Revenue'] / seg['Revenue'].sum() * 100.0

    c1, c2 = st.columns(2)
    with c1:
        fig1 = px.bar(seg.sort_values('Customers', ascending=False), x='Segment_Name', y='Customers', color='Segment_Name', title='Customers by Segment')
        st.plotly_chart(fig1, use_container_width=True)
    with c2:
        fig2 = px.pie(seg, names='Segment_Name', values='Revenue', title='Revenue Share by Segment', hole=0.4)
        st.plotly_chart(fig2, use_container_width=True)

    st.dataframe(seg.sort_values('Revenue', ascending=False), use_container_width=True)


def rfm_distributions(df: pd.DataFrame):
    c1, c2, c3 = st.columns(3)
    with c1:
        st.write("Recency (days)")
        st.plotly_chart(px.histogram(df, x='recency_days', nbins=50, color='Segment_Name'), use_container_width=True)
    with c2:
        st.write("Frequency")
        st.plotly_chart(px.histogram(df, x='frequency', nbins=50, color='Segment_Name'), use_container_width=True)
    with c3:
        st.write("Monetary")
        st.plotly_chart(px.histogram(df, x='monetary', nbins=50, color='Segment_Name'), use_container_width=True)


def segment_profiles(df: pd.DataFrame):
    prof = df.groupby('Segment_Name').agg(
        Customers=('CustomerID', 'count'),
        Recency_Mean=('recency_days', 'mean'),
        Frequency_Mean=('frequency', 'mean'),
        Monetary_Mean=('monetary', 'mean'),
        Monetary_Sum=('monetary', 'sum'),
    ).reset_index()
    prof['Revenue_%'] = prof['Monetary_Sum'] / prof['Monetary_Sum'].sum() * 100.0

    st.subheader("Segment Profiles")
    st.dataframe(prof.sort_values('Monetary_Sum', ascending=False), use_container_width=True)

    fig = px.scatter(prof, x='Frequency_Mean', y='Recency_Mean', size='Monetary_Mean', color='Segment_Name', hover_name='Segment_Name', title='Segment Profile Map (Bubble size = Monetary Mean)')
    st.plotly_chart(fig, use_container_width=True)


def compare_segments(df: pd.DataFrame):
    st.subheader("Compare Segments")
    if df.empty:
        st.info("No data available.")
        return
    segments = sorted(df['Segment_Name'].dropna().unique().tolist())
    selected = st.multiselect("Choose segments to compare", segments, default=segments[:3])
    if not selected:
        st.info("Select at least one segment.")
        return

    cmp = df[df['Segment_Name'].isin(selected)].groupby('Segment_Name').agg(
        Customers=('CustomerID', 'count'),
        Recency_Mean=('recency_days', 'mean'),
        Frequency_Mean=('frequency', 'mean'),
        Monetary_Mean=('monetary', 'mean'),
        Revenue=('monetary', 'sum')
    ).round(2)
    cmp['Revenue_%'] = (cmp['Revenue'] / cmp['Revenue'].sum() * 100).round(1)
    st.dataframe(cmp.sort_values('Revenue', ascending=False), use_container_width=True)

    if len(selected) >= 2:
        norm = cmp[['Recency_Mean','Frequency_Mean','Monetary_Mean']].copy()
        norm['Recency_Mean'] = norm['Recency_Mean'].max() - norm['Recency_Mean']
        for col in norm.columns:
            rng = norm[col].max() - norm[col].min()
            norm[col] = 0.5 if rng == 0 else (norm[col] - norm[col].min()) / rng
        plot_df = norm.reset_index().melt(id_vars='Segment_Name', var_name='Metric', value_name='Score')
        fig = px.line_polar(plot_df, r='Score', theta='Metric', color='Segment_Name', line_close=True, range_r=[0,1])
        fig.update_traces(fill='toself', opacity=0.6)
        st.plotly_chart(fig, use_container_width=True)


def action_planner(df: pd.DataFrame):
    st.subheader("Action Planner")
    if df.empty:
        st.info("No data to plan actions.")
        return
    segments = sorted(df['Segment_Name'].dropna().unique().tolist())
    seg = st.selectbox("Target segment", segments)
    sdf = df[df['Segment_Name'] == seg]
    st.caption(f"Segment size: {len(sdf):,}")

    c1, c2, c3 = st.columns(3)
    with c1:
        target_size = st.slider("Target customers", min_value=0, max_value=int(len(sdf)), value=min(5000, int(len(sdf))))
    with c2:
        uplift_pct = st.number_input("Expected uplift %", min_value=0.0, max_value=100.0, value=5.0, step=0.5)
    with c3:
        budget = st.number_input("Campaign budget (£)", min_value=0.0, value=10000.0, step=500.0)

    avg_spend = sdf['monetary'].mean()
    baseline_rev = avg_spend * target_size
    incremental = baseline_rev * (uplift_pct / 100.0)
    roi = (incremental - budget) / budget * 100 if budget > 0 else float('inf')

    k1, k2, k3 = st.columns(3)
    k1.metric("Baseline revenue (est)", format_currency(baseline_rev))
    k2.metric("Incremental revenue (est)", format_currency(incremental))
    k3.metric("ROI % (est)", f"{roi:.1f}%")

    top_targets = sdf.nlargest(target_size, 'monetary')[['CustomerID','recency_days','frequency','monetary']]
    st.dataframe(top_targets.head(50), use_container_width=True)
    download_button("Download target list (CSV)", top_targets, f"targets_{seg.replace(' ','_').lower()}.csv")

 


def revenue_trend(transactions: pd.DataFrame):
    # Try to detect a date column and amount
    if transactions.empty:
        st.info("No transactions file found for trend. Skipping monthly revenue chart.")
        return
    date_col = None
    for col in ['TransactionDate', 'transaction_date', 'date']:
        if col in transactions.columns:
            date_col = col
            break
    amt_col = None
    for col in ['TransactionAmount', 'amount', 'txn_amount']:
        if col in transactions.columns:
            amt_col = col
            break
    if not date_col or not amt_col:
        st.info("Transactions file missing date/amount columns. Skipping trend.")
        return
    tx = transactions.dropna(subset=[date_col, amt_col]).copy()
    tx['month'] = tx[date_col].dt.to_period('M').dt.to_timestamp()
    monthly = tx.groupby('month')[amt_col].sum().reset_index()
    fig = px.line(monthly, x='month', y=amt_col, title='Monthly Revenue Trend')
    st.plotly_chart(fig, use_container_width=True)


def segment_drilldown(df: pd.DataFrame):
    if df.empty:
        return
    segs = sorted(df['Segment_Name'].dropna().unique().tolist())
    seg = st.selectbox("Select segment", segs)
    sdf = df[df['Segment_Name'] == seg]
    st.subheader(f"Segment: {seg}")
    kpi_cards(sdf)
    c1, c2 = st.columns([2, 1])
    with c1:
        st.write("Top 100 Customers by Monetary")
        top = sdf.nlargest(100, 'monetary')
        st.dataframe(top[['CustomerID', 'recency_days', 'frequency', 'monetary']])
    with c2:
        st.download_button(
            label="Download segment customers (CSV)",
            data=sdf.to_csv(index=False).encode('utf-8'),
            file_name=f"segment_{seg.replace(' ', '_').lower()}.csv",
            mime='text/csv'
        )


def customer_lookup(df: pd.DataFrame):
    st.subheader("Customer Lookup")
    cid = st.text_input("Enter CustomerID")
    if not cid:
        return
    try:
        # Try numeric first, else string match
        cid_num = pd.to_numeric(cid, errors='coerce')
        if not np.isnan(cid_num):
            row = df[df['CustomerID'] == cid_num]
        else:
            row = df[df['CustomerID'].astype(str) == cid]
    except Exception:
        row = df[df['CustomerID'].astype(str) == cid]
    if row.empty:
        st.warning("Customer not found in current dataset/filters.")
        return
    r = row.iloc[0]
    st.write({
        'CustomerID': r['CustomerID'],
        'Segment_Name': r.get('Segment_Name', None),
        'Recency (days)': r.get('recency_days', None),
        'Frequency': r.get('frequency', None),
        'Monetary': r.get('monetary', None),
    })
    st.caption("Tip: Use sidebar filters to widen the search if not found.")


def main():
    st.set_page_config(page_title="BankTrust RFM Segmentation", page_icon="📊", layout="wide")
    st.title("Optimizing Retail Banking Strategies Through RFM-Based Customer Segmentation")

    st.markdown(
        """
        BankTrust seeks to improve retention, ROI, and CLV using behavioral segmentation.
        This app visualizes the RFM-based clustering results and links them to actionable strategies.
        """
    )

    base = os.path.join('data', 'processed')
    df = load_data(os.path.join(base, 'kmeans_customer_segments.csv'))
    if df.empty:
        st.stop()
    cluster_prof = load_cluster_profiles(os.path.join(base, 'cluster_profiles.csv'))
    model_sum = load_model_summary(os.path.join(base, 'kmeans_model_summary.csv'))
    transactions = load_transactions(os.path.join(base, 'transactions_clean.csv'))

    # Sidebar filters
    filtered = sidebar_filters(df)

    tab_overview, tab_segments, tab_drill, tab_compare, tab_actions, tab_raw, tab_model = st.tabs([
        "Overview", "Segments", "Drilldown", "Compare", "Actions", "Raw Data", "Model"
    ])

    with tab_overview:
        st.subheader("Key Performance Indicators")
        kpi_cards(filtered)
        st.subheader("Segment Overview")
        segment_distribution(filtered)
        st.subheader("RFM Distributions")
        rfm_distributions(filtered)
        

    with tab_segments:
        st.subheader("Segment Profiles")
        segment_profiles(filtered)
        st.subheader("Customer Lookup")
        customer_lookup(filtered)

    with tab_drill:
        segment_drilldown(filtered)

    with tab_compare:
        compare_segments(filtered)

    with tab_actions:
        action_planner(filtered)

    with tab_raw:
        st.subheader("Raw Data Insights (Gender & Location)")
        raw_path = os.path.join('data', 'bank_data_C.csv')
        # Lightweight load (use pandas read_csv with dtype fallback)
        try:
            raw = pd.read_csv(raw_path)
        except Exception:
            raw = pd.DataFrame()
        if raw.empty:
            st.info("Raw file not found or could not be loaded: data/bank_data_C.csv")
        else:
            # Normalize column names we rely on
            rename_map = {
                'CustGender': 'CustGender',
                'CustLocation': 'CustLocation',
                'TransactionAmount (INR)': 'TransactionAmount',
            }
            for k, v in rename_map.items():
                if k in raw.columns and v not in raw.columns:
                    raw[v] = raw[k]
            # Basic summaries
            st.markdown("### Gender Breakdown")
            if 'CustGender' in raw.columns:
                g = raw.groupby('CustGender').agg(Customers=('CustomerID', 'nunique'), Transactions=('TransactionID', 'count'), Revenue=('TransactionAmount', 'sum')).reset_index()
                g['Revenue_%'] = g['Revenue'] / g['Revenue'].sum() * 100.0
                c1, c2 = st.columns(2)
                with c1:
                    st.dataframe(g.sort_values('Revenue', ascending=False), use_container_width=True)
                with c2:
                    st.plotly_chart(px.pie(g, names='CustGender', values='Revenue', title='Revenue by Gender', hole=0.4), use_container_width=True)
            else:
                st.info("Column 'CustGender' not found in raw file.")

            st.markdown("### Location Breakdown")
            if 'CustLocation' in raw.columns:
                l = raw.groupby('CustLocation').agg(Customers=('CustomerID', 'nunique'), Transactions=('TransactionID', 'count'), Revenue=('TransactionAmount', 'sum')).reset_index()
                top_l = l.nlargest(20, 'Revenue')
                st.dataframe(top_l, use_container_width=True)
                st.plotly_chart(px.bar(top_l, x='CustLocation', y='Revenue', title='Top 20 Locations by Revenue'), use_container_width=True)
            else:
                st.info("Column 'CustLocation' not found in raw file.")

        st.subheader("Monthly Revenue Trend (if available)")
        revenue_trend(transactions)

    with tab_model:
        st.subheader("Model Summary")
        if not model_sum.empty:
            st.dataframe(model_sum, use_container_width=True)
        else:
            st.info("Model summary file not found.")
        st.subheader("Cluster Profiles (raw)")
        if not cluster_prof.empty:
            st.dataframe(cluster_prof, use_container_width=True)
        else:
            st.info("Cluster profiles file not found.")

    st.info("Run locally: 'streamlit run app.py'. Data source: data/processed/kmeans_customer_segments.csv.")


if __name__ == "__main__":
    main()


