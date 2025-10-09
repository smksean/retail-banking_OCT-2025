# Notebooks

## Overview

This directory contains Jupyter notebooks for the RFM segmentation project.

## Notebooks

1. **`clean_eda.ipynb`** — Data Cleaning & Exploratory Data Analysis
   - Data ingestion and schema validation
   - Date/time parsing and cleaning
   - Outlier detection (IQR vs. percentile methods)
   - Distribution analysis and visualizations
   - Initial RFM segmentation

2. **`02_rfm_refinement.ipynb`** — (TO BE CREATED)
   - Assignment notebook for refining RFM segments
   - Alternative segmentation approaches
   - Evaluation and comparison
   - Final segment recommendations

3. **`03_clustering.ipynb`** — (OPTIONAL/FUTURE)
   - KMeans clustering for unsupervised segmentation
   - Elbow method and silhouette analysis
   - Cluster profiling

4. **`04_streamlit_prep.ipynb`** — (OPTIONAL/FUTURE)
   - Prepare data and logic for Streamlit dashboard
   - Test visualizations and filters

## Running Notebooks

```bash
# Start Jupyter
jupyter notebook

# Or use JupyterLab
jupyter lab
```

## Best Practices

- Run cells top-to-bottom sequentially
- Restart kernel before final run to ensure reproducibility
- Add markdown cells to explain your reasoning
- Save outputs inline for easy review

