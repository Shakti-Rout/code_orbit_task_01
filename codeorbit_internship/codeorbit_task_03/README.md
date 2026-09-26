# 🏡 California Housing Price Prediction: Linear vs. Polynomial Regression

This repository contains a complete, reproducible regression modeling project designed to predict median house values using the **California Housing Dataset** (1990 U.S. Census data).

---

## 📌 Project Overview

- **Problem Type**: Continuous Regression
- **Dataset**: California Housing (`20,640` samples, `8` continuous predictor features, `1` continuous target)
- **Target Variable**: `MedHouseVal` (Median house value in \$100,000s)
- **Models Developed & Compared**:
  1. **Baseline Simple Linear Regression**: Univariate model based on Median Income (`MedInc`)
  2. **Multiple Linear Regression (OLS)**: Multivariate model utilizing all 8 standardized features
  3. **Polynomial Regression (Degree 2 with Ridge Regularization)**: Captures quadratic and interaction effects with $L_2$ regularization ($\alpha = 50.0$)

---

## 📁 Repository Structure

```
codeorbit_task_03/
├── data/
│   └── california_housing.csv          # Local offline dataset (20,640 rows x 9 cols)
├── images/
│   ├── 01_feature_distributions.png    # Histograms & KDE distributions
│   ├── 02_correlation_heatmap.png      # Pearson correlation matrix
│   ├── 03_actual_vs_predicted.png      # Actual vs. Predicted scatter comparison
│   ├── 04_residual_analysis.png        # Residuals vs fitted and error normality
│   ├── 05_sample_predictions_comparison.png # 75-sample prediction trajectory
│   └── 06_linear_coefficients.png      # Standardized feature importance weights
├── regression_analysis.ipynb           # Fully executed Jupyter Notebook with all outputs & insights
├── run_analysis.py                     # Standalone Python script for training, plotting & logging
├── build_notebook.py                   # Automated script to build & execute the notebook
├── INSIGHTS_REPORT.md                  # Comprehensive write-up of insights & error analysis
├── requirements.txt                    # Python package dependencies
└── README.md                           # Documentation & execution guide
```

---

## 📊 Quantitative Results Summary

All models were evaluated on an 80/20 train/test split ($N_{\text{test}} = 4,128$) with `random_state=42`:

| Model | Train $R^2$ | Test $R^2$ | Train MAE | Test MAE | Train RMSE | Test RMSE | Overfitting Gap ($\Delta R^2$) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Simple Linear (`MedInc`)** | 0.4735 | 0.4740 | \$62,790 | \$62,640 | \$83,850 | \$83,060 | -0.0005 |
| **Multiple Linear Regression** | 0.6126 | 0.5758 | \$52,860 | \$53,320 | \$71,970 | \$74,560 | +0.0368 |
| **Polynomial Regression (Deg 2 Ridge)** | **0.6816** | **0.6673** | **\$46,750** | **\$47,080** | **\$65,240** | **\$66,030** | **+0.0143** |

### Key Takeaways:
- **Polynomial Regression** lifts explained variance by **+9.15 percentage points** ($R^2 = 0.6673$ vs $0.5758$) and reduces root mean squared error by **\$8,530** compared to linear regression.
- **Ridge Regularization** ($\alpha = 50.0$) controls multicollinearity among the 44 polynomial interaction terms, keeping the generalization gap to a low **0.0143**.

---

## 🚀 How to Run the Project

### 1. Prerequisites & Installation
Ensure Python 3.10+ is installed. Install required packages:
```bash
pip install -r requirements.txt
```

### 2. Run the End-to-End Python Script
To train the models, output performance metrics to terminal, and regenerate high-resolution visual plots:
```bash
python run_analysis.py
```

### 3. Open the Jupyter Notebook
Open the notebook in Jupyter Notebook, JupyterLab, VS Code, or Google Colab:
```bash
jupyter notebook regression_analysis.ipynb
```
*Note: The notebook is already pre-executed with all output tables and inline visualization plots.*

---

## 🔍 Key Insights Summary

1. **Purchasing Power Dictates Prices**: Median Income (`MedInc`) is the primary driver ($\beta = +0.854$, $r = 0.690$).
2. **Geographic Non-Linearity**: Housing price geography in California cannot be represented as independent linear latitude/longitude slopes; the interaction term $\text{Latitude} \times \text{Longitude}$ captured by the polynomial model effectively isolates coastal metros from inland valleys.
3. **The \$500k Ceiling Artifact**: Residual analysis uncovers a downward-sloping linear error trace at $\text{Actual} = 5.0$, caused by census truncation of luxury properties exceeding \$500,000.
4. **Error Normality**: Residual errors follow a zero-centered bell curve ($\mu = -0.002$), satisfying standard regression assumptions while displaying slight heteroscedasticity for upper-tier homes.
