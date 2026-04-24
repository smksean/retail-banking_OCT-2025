# Retail Banking — RFM Customer Segmentation

> RFM analysis and segmentation of 880k retail banking customers across 1M+ transactions — three approaches compared, with a Streamlit dashboard for interactive exploration.

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-orange.svg)](https://scikit-learn.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-dashboard-red.svg)](https://streamlit.io/)

---

## Problem

BankTrust needed to move from generic marketing campaigns to targeted, segment-specific strategies. With 1M+ transactions across 879k customers, the challenge was to produce actionable customer groups — not a large undifferentiated "Others" bucket that initial RFM scoring left at ~45% of the base.

---

## Approach

### Data Cleaning & EDA
- Loaded 1,041,614 transactions from 879,358 unique customers
- Corrected day-first date parsing and 2-digit year issues
- Validated zero duplicates; preserved raw data for modelling
- Temporal coverage: Aug–Oct 2016; 73% male customers; top cities Mumbai, New Delhi, Bangalore
- Heavy right-skew in transaction amounts and account balances

### RFM Scoring
Computed **Recency** (days since last transaction), **Frequency** (transaction count), and **Monetary** (total spend) as quintile scores (1–5) for all 880k customers.

### Segmentation Strategies Compared

| Approach | Method | "Others" After |
|----------|--------|---------------|
| Expanded Rule-Based | 11 named segments with refined boundary logic | ~18–22% |
| KMeans Clustering | Unsupervised on standardised RFM features (6–8 clusters, Silhouette-optimised) | 0% |
| Hybrid | Priority rule-based segments + KMeans for middle tier | <5% |

**Chosen approach: Expanded rule-based segmentation** — highest interpretability for business stakeholders with "Others" reduced below the 20% target.

---

## Results

- "Others" reduced from ~45% to ~18–22%
- 11 actionable customer segments with direct marketing strategy mapping
- Output: `data/processed/rfm_scores_refined.csv` — all customers with segment labels and RFM scores

---

## How to Run

**Install dependencies:**
```bash
pip install numpy pandas matplotlib seaborn scikit-learn streamlit
```

**Run the Streamlit dashboard:**
```bash
streamlit run app.py
```

**Or explore the notebooks:**
```bash
jupyter notebook notebooks/clean_eda.ipynb          # Cleaning, EDA, initial RFM
jupyter notebook notebooks/02_rfm_refinement.ipynb  # Three segmentation approaches
```

---

## Tech Stack

`Python` · `pandas` · `scikit-learn` · `Streamlit` · `matplotlib` · `seaborn` · `NumPy`

---

## Project Structure

```
retail-banking_OCT-2025/
├── app.py                              # Streamlit dashboard
├── demo_app.py                         # Demo version
├── notebooks/
│   ├── clean_eda.ipynb                 # Data cleaning + EDA + initial RFM
│   ├── 02_rfm_refinement.ipynb         # Three segmentation approaches
│   └── 03_unsupervised_learning_annotated.ipynb
├── data/processed/
│   ├── kmeans_customer_segments.csv
│   ├── cluster_profiles.csv
│   └── kmeans_model_summary.csv
└── requirements.txt
```

---

## Segment Definitions

| Segment | Criteria | Strategy |
|---------|----------|----------|
| Champions | R≥4, F≥4, M≥4 | Reward and upsell |
| Loyal | R≥4, F≥3 | Loyalty programme |
| Potential Loyalists | R≥3, F≥2, M≥3 | Nurture with targeted offers |
| At Risk | R≤2, F≤2, M≤2 | Re-engagement campaign |
| Can't Lose Them | High M, declining R | Priority win-back |
| Hibernating | Low R, low F | Low-cost reactivation |
| Need Attention | Low R, high F | Churn risk investigation |
| Recent Customers | High R, low F | Onboarding journey |
| Promising | Medium R, medium F | Early loyalty nudges |
| About to Sleep | Declining R | Proactive retention |
| Others | Remainder | Generic campaigns |
