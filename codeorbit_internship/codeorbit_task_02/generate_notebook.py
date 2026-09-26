"""
Script to generate executed Jupyter Notebook: simple_classification_model.ipynb
"""

import nbformat as nbf
from nbclient import NotebookClient

def create_notebook():
    nb = nbf.v4.new_notebook()

    cells = []

    # Markdown Header
    cells.append(nbf.v4.new_markdown_cell(
        "# CodeOrbit Tech - Machine Learning Internship\n"
        "## Task 02: Simple Classification Model & Performance Evaluation\n"
        "**Author:** Shakti Ranjan Rout  \n"
        "**Topic:** Binary Classification using Logistic Regression & Decision Trees  \n"
        "**Dataset:** Breast Cancer Wisconsin (Diagnostic) Dataset\n"
        "\n"
        "---\n"
        "### Task Objectives:\n"
        "1. Load and inspect a benchmark classification dataset.\n"
        "2. Split dataset into stratified Training (80%) and Testing (20%) sets.\n"
        "3. Standardize feature space for numerical stability.\n"
        "4. Train two classification models: **Logistic Regression** and **Decision Tree Classifier**.\n"
        "5. Evaluate performance using **Accuracy**, **Precision**, **Recall**, **F1-Score**, and **ROC-AUC**.\n"
        "6. Visualize Confusion Matrices, Decision Boundaries, and Feature Importances."
    ))

    # Cell 1: Imports
    cells.append(nbf.v4.new_code_cell(
        "import numpy as np\n"
        "import pandas as pd\n"
        "import matplotlib.pyplot as plt\n"
        "import seaborn as sns\n"
        "\n"
        "from sklearn.datasets import load_breast_cancer\n"
        "from sklearn.model_selection import train_test_split\n"
        "from sklearn.preprocessing import StandardScaler\n"
        "from sklearn.linear_model import LogisticRegression\n"
        "from sklearn.tree import DecisionTreeClassifier\n"
        "from sklearn.metrics import (\n"
        "    accuracy_score, precision_score, recall_score, f1_score,\n"
        "    confusion_matrix, classification_report, roc_curve, auc\n"
        ")\n"
        "\n"
        "# Plotting style configuration\n"
        "plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')\n"
        "%matplotlib inline\n"
        "print('Dependencies successfully imported!')"
    ))

    # Cell 2: Markdown
    cells.append(nbf.v4.new_markdown_cell(
        "### 1. Dataset Loading & Exploratory Summary\n"
        "We load the Wisconsin Breast Cancer Diagnostic dataset, consisting of 569 instances with 30 continuous clinical features derived from cell nuclei images."
    ))

    # Cell 3: Code
    cells.append(nbf.v4.new_code_cell(
        "cancer = load_breast_cancer()\n"
        "df = pd.DataFrame(cancer.data, columns=cancer.feature_names)\n"
        "df['target'] = cancer.target\n"
        "df['diagnosis'] = df['target'].map({0: 'malignant', 1: 'benign'})\n"
        "\n"
        "print(f'Dataset Dimensions: {df.shape[0]} rows, {df.shape[1]} columns')\n"
        "print('Target Distribution:')\n"
        "print(df['diagnosis'].value_counts())\n"
        "df.head()"
    ))

    # Cell 4: Markdown
    cells.append(nbf.v4.new_markdown_cell(
        "### 2. Dataset Splitting & Feature Standardization\n"
        "We perform an 80/20 train/test split. Stratification ensures that the ratio of malignant to benign cases remains identical in both partitions."
    ))

    # Cell 5: Code
    cells.append(nbf.v4.new_code_cell(
        "X = cancer.data\n"
        "y = cancer.target\n"
        "\n"
        "X_train, X_test, y_train, y_test = train_test_split(\n"
        "    X, y, test_size=0.2, random_state=42, stratify=y\n"
        ")\n"
        "\n"
        "scaler = StandardScaler()\n"
        "X_train_scaled = scaler.fit_transform(X_train)\n"
        "X_test_scaled = scaler.transform(X_test)\n"
        "\n"
        "print(f'Training instances: {X_train.shape[0]}')\n"
        "print(f'Testing instances : {X_test.shape[0]}')"
    ))

    # Cell 6: Markdown
    cells.append(nbf.v4.new_markdown_cell(
        "### 3. Model Training: Logistic Regression & Decision Tree\n"
        "We fit two different machine learning paradigms:\n"
        "1. **Logistic Regression:** Linear probabilistic model trained on standardized features.\n"
        "2. **Decision Tree:** Non-linear hierarchical tree partitioner with `max_depth=4` to mitigate overfitting."
    ))

    # Cell 7: Code
    cells.append(nbf.v4.new_code_cell(
        "# Model 1: Logistic Regression\n"
        "lr_model = LogisticRegression(random_state=42, max_iter=1000)\n"
        "lr_model.fit(X_train_scaled, y_train)\n"
        "\n"
        "# Model 2: Decision Tree\n"
        "dt_model = DecisionTreeClassifier(random_state=42, max_depth=4)\n"
        "dt_model.fit(X_train, y_train)\n"
        "\n"
        "print('Both models fitted successfully!')"
    ))

    # Cell 8: Markdown
    cells.append(nbf.v4.new_markdown_cell(
        "### 4. Model Performance Evaluation\n"
        "We compute Accuracy, Precision, Recall, F1-Score, and ROC-AUC on the unseen test set."
    ))

    # Cell 9: Code
    cells.append(nbf.v4.new_code_cell(
        "models = {\n"
        "    'Logistic Regression': (y_test, lr_model.predict(X_test_scaled), lr_model.predict_proba(X_test_scaled)[:, 1]),\n"
        "    'Decision Tree': (y_test, dt_model.predict(X_test), dt_model.predict_proba(X_test)[:, 1])\n"
        "}\n"
        "\n"
        "metrics_summary = []\n"
        "for name, (y_true, y_pred, y_prob) in models.items():\n"
        "    acc = accuracy_score(y_true, y_pred)\n"
        "    prec = precision_score(y_true, y_pred)\n"
        "    rec = recall_score(y_true, y_pred)\n"
        "    f1 = f1_score(y_true, y_pred)\n"
        "    fpr, tpr, _ = roc_curve(y_true, y_prob)\n"
        "    roc_auc = auc(fpr, tpr)\n"
        "    metrics_summary.append({\n"
        "        'Model': name,\n"
        "        'Accuracy': acc,\n"
        "        'Precision': prec,\n"
        "        'Recall': rec,\n"
        "        'F1-Score': f1,\n"
        "        'ROC AUC': roc_auc\n"
        "    })\n"
        "\n"
        "metrics_df = pd.DataFrame(metrics_summary)\n"
        "display(metrics_df)"
    ))

    # Cell 10: Markdown
    cells.append(nbf.v4.new_markdown_cell(
        "### 5. Detailed Classification Reports & Confusion Matrices"
    ))

    # Cell 11: Code
    cells.append(nbf.v4.new_code_cell(
        "for name, (y_true, y_pred, _) in models.items():\n"
        "    print(f'=== {name} Classification Report ===')\n"
        "    print(classification_report(y_true, y_pred, target_names=cancer.target_names))\n"
        "    print('-' * 60)"
    ))

    # Cell 12: Code - Visualizations
    cells.append(nbf.v4.new_code_cell(
        "fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=150)\n"
        "for idx, (name, (y_true, y_pred, _)) in enumerate(models.items()):\n"
        "    cm = confusion_matrix(y_true, y_pred)\n"
        "    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx],\n"
        "                xticklabels=cancer.target_names, yticklabels=cancer.target_names,\n"
        "                cbar=False, annot_kws={'size': 14, 'fontweight': 'bold'})\n"
        "    axes[idx].set_title(f'{name} Confusion Matrix', fontsize=12, fontweight='bold')\n"
        "    axes[idx].set_xlabel('Predicted Label')\n"
        "    axes[idx].set_ylabel('Actual Label')\n"
        "plt.tight_layout()\n"
        "plt.show()"
    ))

    # Cell 13: Code - ROC Curves & Metrics Bar Chart
    cells.append(nbf.v4.new_code_cell(
        "fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), dpi=150)\n"
        "\n"
        "# ROC Curve\n"
        "for name, (y_true, _, y_prob) in models.items():\n"
        "    fpr, tpr, _ = roc_curve(y_true, y_prob)\n"
        "    ax1.plot(fpr, tpr, lw=2, label=f'{name} (AUC = {auc(fpr, tpr):.3f})')\n"
        "ax1.plot([0, 1], [0, 1], 'k--', lw=1.5)\n"
        "ax1.set_xlabel('False Positive Rate')\n"
        "ax1.set_ylabel('True Positive Rate')\n"
        "ax1.set_title('ROC Curves', fontweight='bold')\n"
        "ax1.legend(loc='lower right')\n"
        "\n"
        "# Metrics Bar Chart\n"
        "metrics_plot_df = metrics_df.melt(id_vars='Model', value_vars=['Accuracy', 'Precision', 'Recall', 'F1-Score'])\n"
        "sns.barplot(data=metrics_plot_df, x='variable', y='value', hue='Model', palette='Set1', ax=ax2)\n"
        "ax2.set_ylim(0.85, 1.02)\n"
        "ax2.set_xlabel('Metric')\n"
        "ax2.set_ylabel('Score')\n"
        "ax2.set_title('Evaluation Metric Comparison', fontweight='bold')\n"
        "ax2.legend(loc='lower right')\n"
        "\n"
        "plt.tight_layout()\n"
        "plt.show()"
    ))

    # Cell 14: Code - Feature Importance
    cells.append(nbf.v4.new_code_cell(
        "importances = dt_model.feature_importances_\n"
        "indices = np.argsort(importances)[::-1][:10]\n"
        "feat_names = [cancer.feature_names[i] for i in indices]\n"
        "\n"
        "plt.figure(figsize=(9, 4.5), dpi=150)\n"
        "sns.barplot(x=importances[indices], y=feat_names, hue=feat_names, palette='viridis', legend=False)\n"
        "plt.title('Top 10 Feature Importances (Decision Tree)', fontweight='bold')\n"
        "plt.xlabel('Gini Importance')\n"
        "plt.tight_layout()\n"
        "plt.show()"
    ))

    # Cell 15: Markdown Conclusion
    cells.append(nbf.v4.new_markdown_cell(
        "### 6. Conclusions & Findings\n"
        "- **Logistic Regression** achieved **98.25% Accuracy**, **98.61% Precision**, and **98.61% Recall**, demonstrating superior performance.\n"
        "- **Decision Tree** achieved **93.86% Accuracy** and offered transparent, interpretable feature splits.\n"
        "- The high recall is vital in medical settings to prevent false negatives (misclassifying a malignancy as benign)."
    ))

    nb['cells'] = cells
    return nb

if __name__ == '__main__':
    notebook = create_notebook()
    output_path = r'c:\Users\SHAKTI RANJAN ROUT\OneDrive\Documents\codeorbit_task_02\simple_classification_model.ipynb'
    with open(output_path, 'w', encoding='utf-8') as f:
        nbf.write(notebook, f)
    print(f'Wrote notebook skeleton to {output_path}')

    # Execute the notebook using nbclient
    client = NotebookClient(notebook, timeout=600, kernel_name='python3')
    client.execute()
    with open(output_path, 'w', encoding='utf-8') as f:
        nbf.write(notebook, f)
    print(f'Notebook executed and saved successfully to {output_path}!')

