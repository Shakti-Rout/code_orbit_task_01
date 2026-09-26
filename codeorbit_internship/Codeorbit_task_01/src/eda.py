# -*- coding: utf-8 -*-
"""
Exploratory Data Analysis (EDA) Module for the Titanic Dataset.

Generates summary statistics, distribution metrics, cross-tabulations,
and publication-grade visualizations saved to the figures/ directory.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIGURES_DIR = os.path.join(BASE_DIR, "figures")


def setup_plotting_style():
    sns.set_theme(style="whitegrid", palette="deep")
    plt.rcParams.update({
        "font.sans-serif": "Arial",
        "font.family": "sans-serif",
        "figure.titlesize": 16,
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.autolayout": False,
        "savefig.dpi": 300,
        "savefig.bbox": "tight"
    })
    os.makedirs(FIGURES_DIR, exist_ok=True)


def compute_summary_statistics(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    num_cols = ["Age", "Age_Original", "Fare", "Fare_Capped", "Fare_Log", "SibSp", "Parch", "FamilySize"]
    num_cols = [c for c in num_cols if c in df.columns]
    
    stats_list = []
    for col in num_cols:
        s = df[col].dropna()
        q1 = s.quantile(0.25)
        q3 = s.quantile(0.75)
        iqr = q3 - q1
        stats_list.append({
            "Feature": col,
            "Count": int(s.count()),
            "Mean": round(s.mean(), 2),
            "Std": round(s.std(), 2),
            "Min": round(s.min(), 2),
            "Q1 (25%)": round(q1, 2),
            "Median (50%)": round(s.median(), 2),
            "Q3 (75%)": round(q3, 2),
            "Max": round(s.max(), 2),
            "IQR": round(iqr, 2),
            "Skewness": round(s.skew(), 2),
            "Kurtosis": round(s.kurt(), 2)
        })
    num_summary = pd.DataFrame(stats_list)

    cat_cols = ["Survived", "Pclass", "Sex", "Embarked", "Title", "AgeGroup", "IsAlone", "Has_Cabin"]
    cat_cols = [c for c in cat_cols if c in df.columns]
    cat_list = []
    for col in cat_cols:
        vc = df[col].value_counts(dropna=False)
        for cat_val, count in vc.items():
            pct = round((count / len(df)) * 100, 2)
            surv_rate = round(df[df[col] == cat_val]["Survived"].mean() * 100, 2) if "Survived" in df.columns else np.nan
            cat_list.append({
                "Feature": col,
                "Category": str(cat_val),
                "Count": count,
                "Percentage (%)": pct,
                "Survival Rate (%)": surv_rate
            })
    cat_summary = pd.DataFrame(cat_list)

    return num_summary, cat_summary


def plot_missing_values_profile(df_raw: pd.DataFrame, df_clean: pd.DataFrame) -> str:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Raw missing
    raw_missing = df_raw.isnull().sum()
    raw_missing = raw_missing[raw_missing > 0].sort_values(ascending=False)
    raw_pct = (raw_missing / len(df_raw)) * 100
    
    bars1 = axes[0].bar(raw_missing.index, raw_pct, color="#d9534f", edgecolor="black", alpha=0.85)
    axes[0].set_title("Missing Data Percentage - Raw Dataset", fontweight="bold", pad=12)
    axes[0].set_ylabel("Missing Percentage (%)")
    axes[0].set_ylim(0, 100)
    for bar, count, pct in zip(bars1, raw_missing, raw_pct):
        yval = bar.get_height()
        axes[0].text(bar.get_x() + bar.get_width() / 2, yval + 2, f"{count} ({pct:.1f}%)", ha="center", va="bottom", fontsize=10, fontweight="bold")
    
    # Cleaned missing
    critical_features = ["Survived", "Pclass", "Sex", "Age", "SibSp", "Parch", "Fare_Capped", "Embarked", "Title", "Has_Cabin", "FamilySize"]
    clean_missing = df_clean[critical_features].isnull().sum()
    bars2 = axes[1].bar(clean_missing.index, clean_missing.values, color="#5cb85c", edgecolor="black", alpha=0.85)
    axes[1].set_title("Missing Data in Processed Predictors & Targets", fontweight="bold", pad=12)
    axes[1].set_ylabel("Missing Count")
    axes[1].set_ylim(0, 10)
    axes[1].tick_params(axis="x", rotation=45)
    for bar, count in zip(bars2, clean_missing.values):
        axes[1].text(bar.get_x() + bar.get_width() / 2, count + 0.2, f"{count}", ha="center", va="bottom", fontsize=10, fontweight="bold")
    
    plt.tight_layout()
    out_path = os.path.join(FIGURES_DIR, "01_missing_values_profile.png")
    fig.savefig(out_path)
    plt.close(fig)
    print(f"[SAVED] {out_path}")
    return out_path


def plot_outlier_detection_treatment(df_raw: pd.DataFrame, df_clean: pd.DataFrame) -> str:
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Fare Boxplot Comparison
    fare_df = pd.DataFrame({
        "Fare (Raw)": df_clean["Fare_Original"],
        "Fare (Capped at 99th %)": df_clean["Fare_Capped"]
    })
    sns.boxplot(data=fare_df, ax=axes[0, 0], palette=["#f0ad4e", "#5bc0de"])
    axes[0, 0].set_title("Fare Distribution: Raw vs. Capped (Winsorized)", fontweight="bold")
    axes[0, 0].set_ylabel("Fare ($)")

    # 2. Fare Distribution (Raw vs Capped vs Log)
    sns.kdeplot(df_clean["Fare_Original"], ax=axes[0, 1], label="Raw Fare", color="#d9534f", lw=2)
    sns.kdeplot(df_clean["Fare_Capped"], ax=axes[0, 1], label="Capped Fare (99th)", color="#f0ad4e", lw=2)
    ax_twin = axes[0, 1].twinx()
    sns.kdeplot(df_clean["Fare_Log"], ax=ax_twin, label="Log1p(Fare) [Right Axis]", color="#0275d8", lw=2, linestyle="--")
    axes[0, 1].set_title("Fare Density Distributions (Skewness Reduction)", fontweight="bold")
    axes[0, 1].set_xlabel("Fare ($)")
    axes[0, 1].legend(loc="upper left")
    ax_twin.legend(loc="upper right")
    ax_twin.grid(False)

    # 3. Age Boxplots by Class and Survival
    sns.boxplot(x="Pclass", y="Age", hue="Survived", data=df_clean, ax=axes[1, 0], palette={0: "#d9534f", 1: "#5cb85c"})
    axes[1, 0].set_title("Age Distribution by Passenger Class & Survival", fontweight="bold")
    axes[1, 0].set_xlabel("Passenger Class")
    axes[1, 0].set_ylabel("Age (Years)")
    axes[1, 0].legend(title="Survived", labels=["No (0)", "Yes (1)"])

    # 4. Age KDE Comparison (Raw vs Imputed)
    sns.kdeplot(df_clean["Age_Original"].dropna(), ax=axes[1, 1], label=f"Original (n={df_clean['Age_Original'].count()})", color="#6c757d", lw=2, linestyle=":")
    sns.kdeplot(df_clean["Age"], ax=axes[1, 1], label=f"Post-Imputation (n={len(df_clean)})", color="#28a745", lw=2)
    axes[1, 1].axvline(df_clean["Age"].median(), color="black", linestyle="--", alpha=0.7, label=f"Median ({df_clean['Age'].median():.1f} yrs)")
    axes[1, 1].set_title("Age Density Before vs. After Grouped Imputation", fontweight="bold")
    axes[1, 1].set_xlabel("Age (Years)")
    axes[1, 1].legend()

    plt.tight_layout()
    out_path = os.path.join(FIGURES_DIR, "02_outlier_detection_treatment.png")
    fig.savefig(out_path)
    plt.close(fig)
    print(f"[SAVED] {out_path}")
    return out_path


def plot_univariate_distributions(df: pd.DataFrame) -> str:
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    
    # 1. Survival Distribution
    surv_counts = df["Survived"].value_counts().sort_index()
    labels = ["Died (0)", "Survived (1)"]
    colors = ["#d9534f", "#5cb85c"]
    bars1 = axes[0, 0].bar(labels, surv_counts.values, color=colors, edgecolor="black", alpha=0.85)
    axes[0, 0].set_title("Target Variable: Survival Breakdown", fontweight="bold")
    axes[0, 0].set_ylabel("Passenger Count")
    for bar in bars1:
        h = bar.get_height()
        pct = (h / len(df)) * 100
        axes[0, 0].text(bar.get_x() + bar.get_width() / 2, h / 2, f"{h} ({pct:.1f}%)", ha="center", va="center", color="white", fontweight="bold", fontsize=11)

    # 2. Passenger Class Distribution
    pclass_counts = df["Pclass"].value_counts().sort_index()
    pclass_labels = ["1st Class", "2nd Class", "3rd Class"]
    bars2 = axes[0, 1].bar(pclass_labels, pclass_counts.values, color="#5bc0de", edgecolor="black", alpha=0.85)
    axes[0, 1].set_title("Passenger Class Distribution", fontweight="bold")
    axes[0, 1].set_ylabel("Count")
    for bar in bars2:
        h = bar.get_height()
        pct = (h / len(df)) * 100
        axes[0, 1].text(bar.get_x() + bar.get_width() / 2, h + 8, f"{h} ({pct:.1f}%)", ha="center", va="bottom", fontweight="bold")

    # 3. Gender Distribution
    sex_counts = df["Sex"].value_counts()
    bars3 = axes[0, 2].bar(sex_counts.index.str.capitalize(), sex_counts.values, color=["#0275d8", "#f0ad4e"], edgecolor="black", alpha=0.85)
    axes[0, 2].set_title("Gender Breakdown", fontweight="bold")
    axes[0, 2].set_ylabel("Count")
    for bar in bars3:
        h = bar.get_height()
        pct = (h / len(df)) * 100
        axes[0, 2].text(bar.get_x() + bar.get_width() / 2, h + 8, f"{h} ({pct:.1f}%)", ha="center", va="bottom", fontweight="bold")

    # 4. Age Distribution
    sns.histplot(df["Age"], kde=True, ax=axes[1, 0], color="#20c997", bins=25, edgecolor="black")
    axes[1, 0].axvline(df["Age"].mean(), color="red", linestyle="--", label=f"Mean: {df['Age'].mean():.1f}")
    axes[1, 0].axvline(df["Age"].median(), color="blue", linestyle="-.", label=f"Median: {df['Age'].median():.1f}")
    axes[1, 0].set_title("Age Distribution (Imputed)", fontweight="bold")
    axes[1, 0].set_xlabel("Age (Years)")
    axes[1, 0].legend()

    # 5. Fare Distribution
    sns.histplot(df["Fare_Capped"], kde=True, ax=axes[1, 1], color="#fd7e14", bins=25, edgecolor="black")
    axes[1, 1].set_title("Fare Distribution (Capped)", fontweight="bold")
    axes[1, 1].set_xlabel("Fare ($)")

    # 6. Family Size Distribution
    fam_counts = df["FamilySize"].value_counts().sort_index()
    axes[1, 2].bar(fam_counts.index, fam_counts.values, color="#6f42c1", edgecolor="black", alpha=0.85)
    axes[1, 2].set_title("Family Size Distribution (SibSp + Parch + 1)", fontweight="bold")
    axes[1, 2].set_xlabel("Family Members Count")
    axes[1, 2].set_ylabel("Count")

    plt.tight_layout()
    out_path = os.path.join(FIGURES_DIR, "03_univariate_distributions.png")
    fig.savefig(out_path)
    plt.close(fig)
    print(f"[SAVED] {out_path}")
    return out_path


def plot_bivariate_survival_factors(df: pd.DataFrame) -> str:
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Survival by Sex
    sns.barplot(x="Sex", y="Survived", hue="Sex", legend=False, data=df, ax=axes[0, 0], palette=["#0275d8", "#e83e8c"], capsize=0.1, edgecolor="black")
    axes[0, 0].set_title("Survival Rate by Gender", fontweight="bold")
    axes[0, 0].set_ylabel("Survival Rate")
    axes[0, 0].set_ylim(0, 1.0)
    for p in axes[0, 0].patches:
        h = p.get_height()
        axes[0, 0].text(p.get_x() + p.get_width() / 2, h + 0.03, f"{h*100:.1f}%", ha="center", fontweight="bold")

    # 2. Survival by Pclass
    sns.barplot(x="Pclass", y="Survived", hue="Pclass", legend=False, data=df, ax=axes[0, 1], palette="Blues_r", capsize=0.1, edgecolor="black")
    axes[0, 1].set_title("Survival Rate by Passenger Class", fontweight="bold")
    axes[0, 1].set_ylabel("Survival Rate")
    axes[0, 1].set_ylim(0, 1.0)
    for p in axes[0, 1].patches:
        h = p.get_height()
        axes[0, 1].text(p.get_x() + p.get_width() / 2, h + 0.03, f"{h*100:.1f}%", ha="center", fontweight="bold")

    # 3. Survival by Embarkation Port
    port_map = {"S": "Southampton", "C": "Cherbourg", "Q": "Queenstown"}
    df_port = df.copy()
    df_port["Port_Name"] = df_port["Embarked"].map(port_map)
    sns.barplot(x="Port_Name", y="Survived", hue="Port_Name", legend=False, data=df_port, ax=axes[1, 0], palette="Set2", capsize=0.1, edgecolor="black")
    axes[1, 0].set_title("Survival Rate by Embarkation Port", fontweight="bold")
    axes[1, 0].set_ylabel("Survival Rate")
    axes[1, 0].set_ylim(0, 1.0)
    for p in axes[1, 0].patches:
        h = p.get_height()
        axes[1, 0].text(p.get_x() + p.get_width() / 2, h + 0.03, f"{h*100:.1f}%", ha="center", fontweight="bold")

    # 4. Survival by Age Group
    sns.barplot(x="AgeGroup", y="Survived", hue="AgeGroup", legend=False, data=df, ax=axes[1, 1], palette="Purples_r", capsize=0.1, edgecolor="black")
    axes[1, 1].set_title("Survival Rate by Age Cohort", fontweight="bold")
    axes[1, 1].set_ylabel("Survival Rate")
    axes[1, 1].set_ylim(0, 1.0)
    axes[1, 1].tick_params(axis="x", rotation=20)
    for p in axes[1, 1].patches:
        h = p.get_height()
        axes[1, 1].text(p.get_x() + p.get_width() / 2, h + 0.03, f"{h*100:.1f}%", ha="center", fontweight="bold")

    plt.tight_layout()
    out_path = os.path.join(FIGURES_DIR, "04_bivariate_survival_factors.png")
    fig.savefig(out_path)
    plt.close(fig)
    print(f"[SAVED] {out_path}")
    return out_path


def plot_correlation_matrix(df: pd.DataFrame) -> str:
    fig, ax = plt.subplots(figsize=(10, 8))
    
    corr_df = df.copy()
    corr_df["Sex_Female"] = (corr_df["Sex"] == "female").astype(int)
    
    cols = ["Survived", "Pclass", "Sex_Female", "Age", "SibSp", "Parch", "FamilySize", "IsAlone", "Fare_Capped", "Fare_Log", "Has_Cabin"]
    labels = ["Survived", "Pclass", "Sex (Female=1)", "Age", "SibSp", "Parch", "Family Size", "Is Alone", "Fare (Capped)", "Fare (Log)", "Has Cabin"]
    
    corr_matrix = corr_df[cols].corr()
    corr_matrix.columns = labels
    corr_matrix.index = labels
    
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    
    sns.heatmap(corr_matrix, mask=mask, annot=True, fmt=".2f", cmap="vlag", vmin=-0.6, vmax=0.6,
                linewidths=1.0, linecolor="white", cbar_kws={"shrink": 0.8, "label": "Pearson Correlation Coefficient"}, ax=ax)
    
    ax.set_title("Correlation Heatmap: Demographic, Economic & Survival Variables", fontweight="bold", pad=15)
    plt.tight_layout()
    out_path = os.path.join(FIGURES_DIR, "05_correlation_matrix.png")
    fig.savefig(out_path)
    plt.close(fig)
    print(f"[SAVED] {out_path}")
    return out_path


def plot_multivariate_interactions(df: pd.DataFrame) -> str:
    fig, axes = plt.subplots(2, 2, figsize=(15, 11))

    # 1. Pclass x Sex Interaction
    sns.pointplot(x="Pclass", y="Survived", hue="Sex", data=df, ax=axes[0, 0], palette={"male": "#0275d8", "female": "#e83e8c"}, markers=["o", "s"], linestyles=["-", "--"], capsize=0.1)
    axes[0, 0].set_title("Class ? Gender Interaction on Survival Rate", fontweight="bold")
    axes[0, 0].set_ylabel("Survival Probability")
    axes[0, 0].set_ylim(0, 1.05)

    # 2. Fare vs Age by Survival
    sns.scatterplot(x="Age", y="Fare_Capped", hue="Survived", style="Sex", data=df, ax=axes[0, 1], palette={0: "#d9534f", 1: "#28a745"}, alpha=0.75, s=60)
    axes[0, 1].set_title("Age vs. Fare (Capped) Colored by Survival", fontweight="bold")
    axes[0, 1].set_xlabel("Age (Years)")
    axes[0, 1].set_ylabel("Fare ($)")
    axes[0, 1].legend(title="Outcome", loc="upper right")

    # 3. Survival by Family Size and Sex
    sns.barplot(x="FamilySize", y="Survived", hue="Sex", data=df, ax=axes[1, 0], palette={"male": "#0275d8", "female": "#e83e8c"}, errorbar=None, edgecolor="black")
    axes[1, 0].set_title("Survival Rate by Family Size & Gender", fontweight="bold")
    axes[1, 0].set_xlabel("Family Size (SibSp + Parch + 1)")
    axes[1, 0].set_ylabel("Survival Rate")
    axes[1, 0].set_ylim(0, 1.05)

    # 4. Survival by Title
    top_titles = ["Mr", "Miss", "Mrs", "Master", "Rare"]
    sns.barplot(x="Title", y="Survived", hue="Title", legend=False, data=df[df["Title"].isin(top_titles)], ax=axes[1, 1], palette="tab10", order=top_titles, capsize=0.1, edgecolor="black")
    axes[1, 1].set_title("Survival Rate by Extracted Passenger Title", fontweight="bold")
    axes[1, 1].set_ylabel("Survival Rate")
    axes[1, 1].set_ylim(0, 1.05)
    for p in axes[1, 1].patches:
        h = p.get_height()
        axes[1, 1].text(p.get_x() + p.get_width() / 2, h + 0.03, f"{h*100:.1f}%", ha="center", fontweight="bold")

    plt.tight_layout()
    out_path = os.path.join(FIGURES_DIR, "06_multivariate_interactions.png")
    fig.savefig(out_path)
    plt.close(fig)
    print(f"[SAVED] {out_path}")
    return out_path


def run_full_eda(df_raw: pd.DataFrame, df_clean: pd.DataFrame):
    setup_plotting_style()
    print("[INFO] Computing summary statistics...")
    num_summary, cat_summary = compute_summary_statistics(df_clean)
    
    print("\n--- Numerical Feature Summary Statistics ---")
    print(num_summary.to_string(index=False))

    print("\n--- Categorical Feature Distributions & Survival Rates ---")
    print(cat_summary.to_string(index=False))

    print("\n[INFO] Generating publication-grade visualizations...")
    plot_missing_values_profile(df_raw, df_clean)
    plot_outlier_detection_treatment(df_raw, df_clean)
    plot_univariate_distributions(df_clean)
    plot_bivariate_survival_factors(df_clean)
    plot_correlation_matrix(df_clean)
    plot_multivariate_interactions(df_clean)
    print("[SUCCESS] All 6 figures generated and saved successfully!")
    return num_summary, cat_summary
