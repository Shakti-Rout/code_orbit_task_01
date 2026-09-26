# -*- coding: utf-8 -*-
"""
End-to-End Execution Pipeline for Data Preprocessing and EDA on Titanic Dataset.
"""

import os
import sys
import pandas as pd

from src.data_cleaning import download_raw_data, clean_titanic_data, save_cleaned_data
from src.eda import run_full_eda, setup_plotting_style

def main():
    print("=================================================================")
    print("      DATA PREPROCESSING & EXPLORATORY DATA ANALYSIS PIPELINE     ")
    print("=================================================================")
    
    print("\n>>> STEP 1: Ingesting Raw Dataset...")
    df_raw = download_raw_data()
    print(f"Raw dataset shape: {df_raw.shape[0]} rows, {df_raw.shape[1]} columns")

    # 2. Clean & Preprocess
    print("\n>>> STEP 2: Cleaning Data (Missing, Duplicates, Outliers)...")
    df_clean, audit = clean_titanic_data(df_raw)
    cleaned_path = save_cleaned_data(df_clean)

    # 3. Exploratory Data Analysis & Visualizations
    print("\n>>> STEP 3: Performing Exploratory Data Analysis...")
    num_summary, cat_summary = run_full_eda(df_raw, df_clean)

    # Export statistical summaries to CSV
    stats_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "processed")
    num_summary.to_csv(os.path.join(stats_dir, "numerical_summary.csv"), index=False)
    cat_summary.to_csv(os.path.join(stats_dir, "categorical_summary.csv"), index=False)
    print(f"[SUCCESS] Exported summary statistics to {stats_dir}")

    # 4. Pipeline Verification
    print("\n>>> STEP 4: Verifying Pipeline Integrity...")
    figures_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
    expected_figures = [
        "01_missing_values_profile.png",
        "02_outlier_detection_treatment.png",
        "03_univariate_distributions.png",
        "04_bivariate_survival_factors.png",
        "05_correlation_matrix.png",
        "06_multivariate_interactions.png"
    ]
    for fig_name in expected_figures:
        fig_path = os.path.join(figures_dir, fig_name)
        assert os.path.exists(fig_path), f"Missing figure: {fig_name}"
        assert os.path.getsize(fig_path) > 1000, f"Figure too small: {fig_name}"
        print(f"  [OK] Figure verified: {fig_name} ({os.path.getsize(fig_path)/1024:.1f} KB)")

    critical_cols = ["Survived", "Pclass", "Sex", "Age", "Fare_Capped", "Embarked", "Title", "FamilySize", "IsAlone"]
    null_counts = df_clean[critical_cols].isnull().sum().sum()
    assert null_counts == 0, f"Critical columns contain {null_counts} nulls!"
    print(f"  [OK] Verified 0 missing values across all {len(critical_cols)} critical modeling columns.")

    print("\n=================================================================")
    print("      PIPELINE EXECUTION COMPLETED SUCCESSFULLY!                ")
    print("=================================================================")

if __name__ == "__main__":
    main()
