# CodeOrbit Tech — Machine Learning Internship: Task 02
## Simple Classification Model & Performance Evaluation

**Author:** Shakti Ranjan Rout  
**Internship Role:** Machine Learning Intern  
**Organization:** CodeOrbit Tech  

---

## 📌 Project Overview
This repository contains the complete implementation for **Task 02: Simple Classification Model**. 
The objective of this project is to:
1. Select a small benchmark dataset for classification (Breast Cancer Wisconsin Diagnostic dataset).
2. Split the data into stratified training ($80\%$) and testing ($20\%$) sets.
3. Train two standard classification models:
   - **Logistic Regression** (Linear probabilistic classifier with L2 regularization)
   - **Decision Tree Classifier** (Non-linear rule-based tree model with depth pruning)
4. Evaluate and compare both models using **Accuracy**, **Precision**, **Recall**, **F1-Score**, and **ROC-AUC**.
5. Provide actionable visual insights including Confusion Matrices, ROC Curves, and Feature Importances.

---

## 📂 Project Structure
```text
codeorbit_task_02/
│
├── data/
│   └── breast_cancer_dataset.csv     # Extracted and structured benchmark dataset
│
├── figures/                          # High-resolution evaluation charts (300 DPI)
│   ├── confusion_matrices.png        # Confusion heatmaps for both models
│   ├── feature_importance.png        # Top 10 feature importances (Decision Tree)
│   ├── metrics_comparison.png        # Bar chart comparing Accuracy, Precision, Recall, F1
│   └── roc_curves.png                # ROC Curves with AUC scores
│
├── src/
│   └── classification_pipeline.py    # Modular production pipeline class
│
├── CLASSIFICATION_REPORT.md          # Comprehensive executive performance report
├── generate_notebook.py              # Notebook builder & executor utility
├── README.md                         # Project documentation and quickstart
├── run_pipeline.py                   # Main pipeline entrypoint script
└── simple_classification_model.ipynb # Interactive, fully-executed Jupyter Notebook
```

---

## 🚀 Quickstart & Execution

### 1. Requirements
Ensure you have Python 3.8+ installed along with the required libraries:
```bash
pip install numpy pandas scikit-learn matplotlib seaborn
```

### 2. Run the End-to-End Pipeline
To execute the pipeline, compute all metrics, and regenerate figures:
```bash
python run_pipeline.py
```

### 3. Open the Jupyter Notebook
```bash
jupyter notebook simple_classification_model.ipynb
```

---

## 📊 Summary of Evaluation Results

| Metric | Logistic Regression | Decision Tree ($\text{depth}=4$) |
| :--- | :---: | :---: |
| **Accuracy** | **98.25%** | **93.86%** |
| **Precision** | **98.61%** | **95.77%** |
| **Recall** | **98.61%** | **94.44%** |
| **F1-Score** | **0.9861** | **0.9510** |
| **ROC AUC** | **0.9954** | **0.9342** |

*Detailed insights and analysis are documented in [CLASSIFICATION_REPORT.md](CLASSIFICATION_REPORT.md).*
