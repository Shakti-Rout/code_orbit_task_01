"""
Data Preprocessing and Cleaning Module for the Titanic Dataset.

This module handles:
1. Downloading and loading the raw Titanic dataset.
2. Missing value diagnosis and intelligent imputation (e.g. title-based age imputation).
3. Duplicate record detection and handling.
4. Outlier detection using the Interquartile Range (IQR) method and Winsorization/capping.
5. Domain-specific feature engineering (FamilySize, IsAlone, Deck, Title).
6. Exporting both raw and cleaned data.
"""

import os
import urllib.request
import numpy as np
import pandas as pd

RAW_DATA_URL = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_DATA_PATH = os.path.join(DATA_DIR, "raw", "titanic_raw.csv")
PROCESSED_DATA_PATH = os.path.join(DATA_DIR, "processed", "titanic_cleaned.csv")


def ensure_directories():
    os.makedirs(os.path.join(DATA_DIR, "raw"), exist_ok=True)
    os.makedirs(os.path.join(DATA_DIR, "processed"), exist_ok=True)


def download_raw_data(force=False) -> pd.DataFrame:
    ensure_directories()
    if not os.path.exists(RAW_DATA_PATH) or force:
        print(f"[INFO] Downloading raw dataset from {RAW_DATA_URL}...")
        urllib.request.urlretrieve(RAW_DATA_URL, RAW_DATA_PATH)
        print(f"[SUCCESS] Raw dataset saved to {RAW_DATA_PATH}")
    else:
        print(f"[INFO] Raw dataset already exists at {RAW_DATA_PATH}")
    return pd.read_csv(RAW_DATA_PATH)


def extract_title(name: str) -> str:
    if not isinstance(name, str):
        return "Unknown"
    try:
        title = name.split(",")[1].split(".")[0].strip()
    except Exception:
        return "Unknown"
    title_mapping = {
        "Mr": "Mr",
        "Miss": "Miss",
        "Mrs": "Mrs",
        "Master": "Master",
        "Dr": "Rare",
        "Rev": "Rare",
        "Col": "Rare",
        "Major": "Rare",
        "Mlle": "Miss",
        "Mme": "Mrs",
        "Ms": "Miss",
        "Lady": "Rare",
        "Sir": "Rare",
        "Capt": "Rare",
        "Countess": "Rare",
        "Jonkheer": "Rare",
        "Don": "Rare",
        "Dona": "Rare"
    }
    return title_mapping.get(title, "Rare")


def detect_outliers_iqr(series: pd.Series, factor: float = 1.5):
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - factor * iqr
    upper_bound = q3 + factor * iqr
    outlier_mask = (series < lower_bound) | (series > upper_bound)
    return float(lower_bound), float(upper_bound), outlier_mask, int(outlier_mask.sum())


def clean_titanic_data(df_raw: pd.DataFrame) -> tuple:
    df = df_raw.copy()
    audit = {}

    # 1. Dimensions and initial missing count audit
    audit["initial_shape"] = df.shape
    audit["initial_missing"] = df.isnull().sum().to_dict()

    # 2. Duplicate Detection
    exact_duplicates = int(df.duplicated().sum())
    id_duplicates = int(df.duplicated(subset=["PassengerId"]).sum())
    audit["exact_duplicates"] = exact_duplicates
    audit["id_duplicates"] = id_duplicates
    if exact_duplicates > 0:
        df = df.drop_duplicates().reset_index(drop=True)

    # 3. Missing Value Handling
    # (a) Embarked: Impute missing values with statistical mode ('S')
    embarked_mode = df["Embarked"].mode()[0]
    df["Embarked"] = df["Embarked"].fillna(embarked_mode)
    audit["embarked_imputed_count"] = int(audit["initial_missing"].get("Embarked", 0))

    # (b) Feature Engineering: Title extraction for granular Age imputation
    df["Title"] = df["Name"].apply(extract_title)

    # (c) Age: Grouped median imputation by Title and Pclass
    df["Age_Original"] = df["Age"].copy()
    grouped_age_medians = df.groupby(["Title", "Pclass"])["Age"].transform("median")
    overall_age_median = df["Age"].median()
    df["Age"] = df["Age"].fillna(grouped_age_medians).fillna(overall_age_median)
    audit["age_imputed_count"] = int(audit["initial_missing"].get("Age", 0))

    # (d) Cabin: High missingness (~77%)
    # Retain domain signal: engineer Has_Cabin binary indicator and extract Deck initial
    df["Has_Cabin"] = df["Cabin"].notnull().astype(int)
    df["Deck"] = df["Cabin"].apply(lambda c: str(c)[0] if pd.notnull(c) else "Unknown")
    audit["cabin_missing_count"] = int(audit["initial_missing"].get("Cabin", 0))

    # 4. Outlier Detection and Treatment
    # (a) Fare Outliers: identify with IQR, Winsorize at 99th percentile, log-transform
    fare_lb, fare_ub, fare_mask, fare_outliers = detect_outliers_iqr(df["Fare"])
    df["Fare_Original"] = df["Fare"].copy()
    fare_p99 = float(df["Fare"].quantile(0.99))
    df["Fare_Capped"] = np.clip(df["Fare"], a_min=0.0, a_max=fare_p99)
    df["Fare_Log"] = np.log1p(df["Fare_Capped"])
    
    audit["fare_iqr_bounds"] = (fare_lb, fare_ub)
    audit["fare_outlier_count"] = fare_outliers
    audit["fare_p99_cap"] = fare_p99

    # (b) Age Outliers: detect with IQR
    age_lb, age_ub, age_mask, age_outliers = detect_outliers_iqr(df["Age"])
    audit["age_iqr_bounds"] = (age_lb, age_ub)
    audit["age_outlier_count"] = age_outliers

    # 5. Additional Domain Feature Engineering
    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
    df["IsAlone"] = (df["FamilySize"] == 1).astype(int)

    # Age Group Binning
    age_bins = [0, 12, 18, 35, 60, 120]
    age_labels = ["Child", "Adolescent", "Young Adult", "Middle-Aged", "Senior"]
    df["AgeGroup"] = pd.cut(df["Age"], bins=age_bins, labels=age_labels, right=True)

    # Final audit state
    audit["cleaned_shape"] = df.shape
    audit["final_missing"] = df.isnull().sum().to_dict()

    return df, audit


def save_cleaned_data(df_clean: pd.DataFrame) -> str:
    ensure_directories()
    df_clean.to_csv(PROCESSED_DATA_PATH, index=False)
    print(f"[SUCCESS] Cleaned dataset saved to {PROCESSED_DATA_PATH}")
    return PROCESSED_DATA_PATH


if __name__ == "__main__":
    df_raw = download_raw_data()
    df_clean, audit = clean_titanic_data(df_raw)
    save_cleaned_data(df_clean)
    print("\n--- Cleaning Audit Summary ---")
    for k, v in audit.items():
        print(f"{k}: {v}")
