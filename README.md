# Retail Banking — RFM-Based Customer Segmentation

**Project Status:** ✅ Data Cleaning & EDA Complete | 🔄 RFM Refinement In Progress

## 📋 Project Overview

This project applies **RFM (Recency, Frequency, Monetary)** analysis to retail banking transaction data to segment customers and develop targeted retention and engagement strategies for BankTrust.

**Business Goal:** Reduce churn, improve personalization, and optimize marketing efficiency through data-driven customer segmentation.

---

## 📁 Project Structure

```
retail-banking/
│
├── data/
│   ├── bank_data_C.csv              # Raw transaction data
│   └── processed/
│       ├── transactions_clean.csv   # Cleaned transactions (1.04M rows)
│       └── rfm_scores.csv           # RFM scores & segments (880k customers)
│
├── notebooks/
│   └── clean_eda.ipynb              # Data cleaning, EDA, RFM segmentation
│
├── instructions.md                  # Original project brief
└── README.md                        # This file
```

---

## 🎯 Current Project State

### ✅ Completed

1. **Data Ingestion & Cleaning**
   - Loaded 1,041,614 transactions from 879,358 unique customers
   - Parsed day-first dates with explicit format handling
   - Fixed 2-digit year parsing issues (future DOBs corrected)
   - Combined date + time into `TransactionDateTime`
   - Validated: Zero duplicates

2. **Exploratory Data Analysis (EDA)**
   - **Temporal coverage:** Aug 1 – Oct 21, 2016 (~2.5 months)
   - **Demographics:** 73% male customers; top cities: Mumbai, New Delhi, Bangalore
   - **Distributions:** Heavy right skew in transaction amounts and account balances
   - **Outlier handling:**
     - IQR method flags 10–13% as outliers
     - Applied 1%–99% percentile capping for visualizations only
     - Raw data preserved for modeling
   - **Monthly trends:** Declining transaction volume from Aug → Oct (data coverage issue)

3. **RFM Segmentation (Initial)**
   - Computed **Recency** (days since last transaction), **Frequency** (transaction count), **Monetary** (total amount)
   - Assigned quintile scores (1–5) for each metric
   - Mapped to business segments using rule-based logic:
     - **Champions:** R≥4, F≥4, M≥4
     - **Loyal:** R≥4, F≥3
     - **Potential Loyalists:** R≥3, F≥2, M≥3
     - **At Risk:** R≤2, F≤2, M≤2
     - **Need Attention:** R≤2, F≥4
     - **Others:** Everything else

### 🚨 Key Issue Identified

**Problem:** A large proportion of customers fall into the **"Others"** segment, limiting the actionability of our segmentation.

**Current Segment Distribution** (approximate):
- Others: ~40–50% of customers
- Champions: ~10%
- Loyal: ~15%
- At Risk: ~8%
- Potential Loyalists: ~12%
- Need Attention: ~5%

**Impact:**
- Poor targeting precision for the "Others" group
- Wasted marketing spend on generic campaigns
- Missed opportunities for personalized engagement

---

## 🎯 **ASSIGNMENT: Refine RFM Segmentation to Reduce "Others"**

### Objective

**Develop and justify alternative segmentation approaches** to capture more customers into meaningful, actionable segments and reduce the "Others" category to <20% of the customer base.

### Requirements

Your solution must:

1. **Propose at least TWO alternative approaches** to refine RFM segmentation:
   - Option A: Modify the rule-based logic (e.g., expand segment definitions, add new segments)
   - Option B: Use unsupervised learning (e.g., KMeans clustering on RFM features)
   - Option C: Hybrid approach (combine rule-based + clustering)
   - Option D: Create sub-segments within "Others" based on additional features (e.g., age from DOB, location tier)

2. **Implement your chosen approach(es)** in a Jupyter notebook (`notebooks/02_rfm_refinement.ipynb`)

3. **Provide clear justification** for each approach:
   - **Why** this method is appropriate for the data
   - **Trade-offs** (complexity vs. interpretability, precision vs. recall)
   - **Expected business impact** (how will this improve targeting?)

4. **Evaluate results:**
   - Compare segment distributions before vs. after
   - Profile each segment (avg R/F/M, customer count, total revenue contribution)
   - Assess interpretability and actionability

5. **Document findings:**
   - Summary table of all approaches tested
   - Recommendation: Which approach to deploy and why
   - Markdown cells explaining reasoning at each step

### Evaluation Criteria

- **Creativity:** Novel approaches beyond basic RFM rules
- **Rigor:** Statistical justification, quantitative evaluation
- **Business Acumen:** Segmentation must be interpretable and actionable
- **Code Quality:** Clean, well-commented, reproducible
- **Communication:** Clear markdown explanations in notebook

### Deliverable

- New notebook: `notebooks/02_rfm_refinement.ipynb`
- Updated `README.md` with your approach summary
- (Optional) Updated `rfm_scores.csv` with refined segments

---

## 🛠 Installation & Setup

### Prerequisites

- Python 3.8+
- Jupyter Notebook / JupyterLab
- Git

### Dependencies

```bash
pip install numpy pandas matplotlib seaborn scikit-learn streamlit
```

### Run the Notebook

```bash
cd retail-banking
jupyter notebook notebooks/clean_eda.ipynb
```

---

## 📊 Data Dictionary

| Field                  | Description                          |
|------------------------|--------------------------------------|
| `TransactionID`        | Unique transaction identifier        |
| `CustomerID`           | Unique customer identifier           |
| `CustomerDOB`          | Customer date of birth               |
| `CustGender`           | Customer gender (M/F/T)              |
| `CustLocation`         | Customer city/location               |
| `CustAccountBalance`   | Current account balance (INR)        |
| `TransactionDate`      | Date of transaction                  |
| `TransactionTime`      | Time of transaction (HHMMSS)         |
| `TransactionAmount`    | Transaction value (INR)              |

---

## 📈 Next Steps (After Assignment)

1. **Deploy Streamlit Dashboard**
   - Interactive RFM explorer
   - Segment filters and drill-downs
   - "What-if" scenario simulation

2. **Predictive Modeling**
   - Churn prediction using RFM + demographics
   - Customer lifetime value (CLV) estimation

3. **A/B Testing Framework**
   - Test targeted campaigns per segment
   - Measure uplift in engagement/revenue

---

## 🤝 Contributing

For questions or issues, contact the project lead or open an issue in the repository.

---



**Last Updated:** October 9, 2025  
**Author:** [Your Name]  
**Course:** Optimizing Retail Banking Strategies Through RFM-Based Customer Segmentation

