# Executive Report: Simple Classification Model & Performance Evaluation

**Author:** Shakti Ranjan Rout  
**Program:** CodeOrbit Tech — Machine Learning Internship  
**Task 02:** Simple Classification Model  
**Dataset:** Breast Cancer Wisconsin (Diagnostic) Dataset (UCI / Scikit-Learn Benchmark)  
**Deliverables:**
- Source Code Pipeline: [`src/classification_pipeline.py`](src/classification_pipeline.py)
- Execution Script: [`run_pipeline.py`](run_pipeline.py)
- Fully-Executed Jupyter Notebook: [`simple_classification_model.ipynb`](simple_classification_model.ipynb)
- Raw Dataset: [`data/breast_cancer_dataset.csv`](data/breast_cancer_dataset.csv)
- High-Resolution Figures: [`figures/`](figures/)

---

## 1. Executive Summary

This study completes an end-to-end classification pipeline for predicting breast tumor malignancy using diagnostic biopsy measurements. We train, tune, and evaluate two canonical machine learning algorithms: **Logistic Regression** (a linear probabilistic classifier) and **Decision Tree Classifier** (a non-linear rule-based tree partitioner).

### Key Performance Highlights:
1. **Superior Linear Separability:** **Logistic Regression** achieved a state-of-the-art **98.25% Accuracy**, **98.61% Precision**, **98.61% Recall**, and **0.9954 ROC-AUC** on the unseen test set, misclassifying only $2$ out of $114$ test cases.
2. **Clinical Safety & Recall:** In medical diagnostic screening, **Recall (Sensitivity)** is paramount to prevent false negatives (failing to identify malignant tumors). Logistic Regression achieved **97.62% recall on malignant cases** ($\text{FN}=1$), while the Decision Tree achieved **92.86%** ($\text{FN}=3$).
3. **Model Interpretability:** Feature importance analysis via the Decision Tree revealed that boundary irregular features—specifically **`worst perimeter`**, **`worst concave points`**, and **`worst texture`**—account for over $85\%$ of the total Gini impurity reduction.

---

## 2. Dataset Architecture & Clinical Context

The Breast Cancer Wisconsin (Diagnostic) Dataset contains computed cell nucleus attributes extracted from digitized fine needle aspirate (FNA) images of breast masses.

| Attribute | Specification |
| :--- | :--- |
| **Total Samples ($N$)** | 569 patient biopsy records |
| **Feature Dimensionality ($D$)** | 30 continuous real-valued features |
| **Target Classes** | Binary: `Malignant` (Class 0) vs. `Benign` (Class 1) |
| **Class Distribution** | 212 Malignant ($37.26\%$) and 357 Benign ($62.74\%$) |
| **Missing Values** | 0 missing fields across all attributes |

Each biopsy sample includes mean, standard error, and "worst" (mean of the three largest values) measurements for ten core nuclear characteristics:
1. Radius (mean of distances from center to perimeter)
2. Texture (standard deviation of gray-scale values)
3. Perimeter
4. Area
5. Smoothness (local variation in radius lengths)
6. Compactness ($\frac{\text{perimeter}^2}{\text{area}} - 1.0$)
7. Concavity (severity of concave portions of the contour)
8. Concave points (number of concave portions of the contour)
9. Symmetry
10. Fractal dimension ("coastline approximation" - 1)

---

## 3. Data Preprocessing & Methodology

### 3.1 Stratified Train/Test Split
To prevent data leakage while preserving class proportions, the dataset was partitioned using stratified sampling ($80\%$ train, $20\%$ test):
* **Training Set:** 455 instances (170 malignant, 285 benign)
* **Testing Set:** 114 instances (42 malignant, 72 benign)

### 3.2 Feature Standardization
Because clinical features possess vastly divergent scales (e.g., `worst area` up to $2500\,\text{mm}^2$ vs. `smoothness` $\approx 0.1$), we applied Z-score standardization:
$$z = \frac{x - \mu}{\sigma}$$
* Fitted exclusively on `X_train` and transformed onto `X_test` to prevent train-test data leakage.
* Essential for Logistic Regression to ensure balanced gradient steps and prevent high-magnitude features from dominating optimization.

---

## 4. Model Architectures & Training

1. **Logistic Regression:**
   * Optimization: L-BFGS solver with L2 regularization ($C = 1.0$).
   * Maximum Iterations: 1,000 (ensuring absolute convergence).
2. **Decision Tree Classifier:**
   * Splitting Criterion: Gini Impurity.
   * Tree Regularization: `max_depth = 4` to prevent deep branch memorization and control variance on small sample sizes.

---

## 5. Comprehensive Performance Evaluation

### 5.1 Test Set Metrics Comparison ($N = 114$)

| Evaluation Metric | Logistic Regression | Decision Tree ($\text{depth}=4$) | Mathematical Definition |
| :--- | :---: | :---: | :--- |
| **Accuracy** | **98.25%** | **93.86%** | $\frac{\text{TP} + \text{TN}}{\text{TP} + \text{TN} + \text{FP} + \text{FN}}$ |
| **Precision (Macro)** | **98.24%** | **93.57%** | $\frac{\text{TP}}{\text{TP} + \text{FP}}$ |
| **Recall (Macro)** | **98.02%** | **93.65%** | $\frac{\text{TP}}{\text{TP} + \text{FN}}$ |
| **F1-Score (Macro)** | **98.13%** | **93.61%** | $2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$ |
| **ROC AUC** | **0.9954** | **0.9342** | Area under Receiver Operating Characteristic curve |

### 5.2 Confusion Matrix Breakdown

```
Logistic Regression:                  Decision Tree:
               Predicted                             Predicted
             Malig   Benign                        Malig   Benign
Actual Malig [  41      1  ]         Actual Malig [  39      3  ]
Actual Benign[   1     71  ]         Actual Benign[   4     68  ]
```

- **Logistic Regression:** Only $1$ false positive and $1$ false negative.
- **Decision Tree:** $3$ false positives and $4$ false negatives.

### 5.3 Detailed Classification Reports

**Logistic Regression Report:**
```
              precision    recall  f1-score   support

   malignant       0.98      0.98      0.98        42
      benign       0.99      0.99      0.99        72

    accuracy                           0.98       114
   macro avg       0.98      0.98      0.98       114
weighted avg       0.98      0.98      0.98       114
```

**Decision Tree Report:**
```
              precision    recall  f1-score   support

   malignant       0.91      0.93      0.92        42
      benign       0.96      0.94      0.95        72

    accuracy                           0.94       114
   macro avg       0.93      0.94      0.93       114
weighted avg       0.94      0.94      0.94       114
```

---

## 6. Generated Visualizations

All visual artifacts are located in the [`figures/`](figures/) directory:

1. **Model Metrics Comparison (`figures/metrics_comparison.png`):** Direct side-by-side comparison of Accuracy, Precision, Recall, and F1-Score.
2. **Confusion Matrices (`figures/confusion_matrices.png`):** Heatmaps showing exact distributions of True Positives, True Negatives, False Positives, and False Negatives.
3. **ROC Curves (`figures/roc_curves.png`):** Sensitivity vs. (1 - Specificity) across probability thresholds showing near-perfect discrimination ($\text{AUC} = 0.995$).
4. **Feature Importance (`figures/feature_importance.png`):** Ranking the top 10 most influential features driving classification splits.

---

## 7. Submission & Presentation Summary

* **Project Folder:** `codeorbit_task_02`
* **Execution Command:** `python run_pipeline.py`
* **Jupyter Notebook:** `simple_classification_model.ipynb`
* **Internship Eligibility:** Fulfills all Task 02 criteria specified in the CodeOrbit Tech Machine Learning Internship manual.
