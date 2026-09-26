"""
CodeOrbit Tech - Machine Learning Internship
Task 02: Simple Classification Model Pipeline
Author: Shakti Ranjan Rout
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    auc
)


class ClassificationPipeline:
    def __init__(self, data_dir="data", figures_dir="figures", random_state=42):
        self.data_dir = data_dir
        self.figures_dir = figures_dir
        self.random_state = random_state
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.figures_dir, exist_ok=True)

        self.X = None
        self.y = None
        self.feature_names = None
        self.target_names = None
        self.df = None

        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scaler = None
        self.X_train_scaled = None
        self.X_test_scaled = None

        self.lr_model = None
        self.dt_model = None
        self.results = {}

    def load_data(self):
        """Loads Breast Cancer dataset and saves a local CSV for reproducibility."""
        cancer = load_breast_cancer()
        self.X = cancer.data
        self.y = cancer.target
        self.feature_names = cancer.feature_names
        self.target_names = cancer.target_names  # 0: malignant, 1: benign

        # Create DataFrame
        self.df = pd.DataFrame(self.X, columns=self.feature_names)
        self.df['target'] = self.y
        self.df['diagnosis'] = self.df['target'].map({0: 'malignant', 1: 'benign'})

        # Save to local CSV
        csv_path = os.path.join(self.data_dir, "breast_cancer_dataset.csv")
        self.df.to_csv(csv_path, index=False)
        print(f"[+] Dataset loaded successfully: {self.X.shape[0]} samples, {self.X.shape[1]} features.")
        print(f"[+] Saved dataset copy to {csv_path}")
        return self.df

    def split_and_preprocess(self, test_size=0.2):
        """Splits into train/test sets (80/20) and standardizes features."""
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=test_size, random_state=self.random_state, stratify=self.y
        )

        self.scaler = StandardScaler()
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)

        print(f"[+] Train/Test Split completed:")
        print(f"    - Training set: {self.X_train.shape[0]} samples")
        print(f"    - Testing set : {self.X_test.shape[0]} samples (stratified)")

    def train_models(self):
        """Trains Logistic Regression and Decision Tree models."""
        print("[+] Training Logistic Regression...")
        self.lr_model = LogisticRegression(random_state=self.random_state, max_iter=1000)
        self.lr_model.fit(self.X_train_scaled, self.y_train)

        print("[+] Training Decision Tree Classifier...")
        self.dt_model = DecisionTreeClassifier(random_state=self.random_state, max_depth=4)
        self.dt_model.fit(self.X_train, self.y_train)

    def evaluate_models(self):
        """Evaluates models using Accuracy, Precision, Recall, and F1-Score."""
        # Predictions & Probabilities
        lr_pred = self.lr_model.predict(self.X_test_scaled)
        lr_prob = self.lr_model.predict_proba(self.X_test_scaled)[:, 1]

        dt_pred = self.dt_model.predict(self.X_test)
        dt_prob = self.dt_model.predict_proba(self.X_test)[:, 1]

        eval_data = {
            "Logistic Regression": (self.y_test, lr_pred, lr_prob),
            "Decision Tree": (self.y_test, dt_pred, dt_prob)
        }

        print("\n" + "=" * 65)
        print("          MODEL PERFORMANCE EVALUATION ON TEST SET")
        print("=" * 65)

        for model_name, (y_true, y_pred, y_prob) in eval_data.items():
            acc = accuracy_score(y_true, y_pred)
            prec = precision_score(y_true, y_pred)
            rec = recall_score(y_true, y_pred)
            f1 = f1_score(y_true, y_pred)
            cm = confusion_matrix(y_true, y_pred)
            fpr, tpr, _ = roc_curve(y_true, y_prob)
            roc_auc = auc(fpr, tpr)

            self.results[model_name] = {
                "accuracy": acc,
                "precision": prec,
                "recall": rec,
                "f1": f1,
                "cm": cm,
                "fpr": fpr,
                "tpr": tpr,
                "auc": roc_auc,
                "y_pred": y_pred
            }

            print(f"\n--- {model_name} ---")
            print(f"Accuracy  : {acc:.4f} ({acc * 100:.2f}%)")
            print(f"Precision : {prec:.4f} ({prec * 100:.2f}%)")
            print(f"Recall    : {rec:.4f} ({rec * 100:.2f}%)")
            print(f"F1-Score  : {f1:.4f}")
            print(f"ROC AUC   : {roc_auc:.4f}")
            print(f"Confusion Matrix: [TN={cm[0,0]}, FP={cm[0,1]} | FN={cm[1,0]}, TP={cm[1,1]}]")
            print("\nClassification Report:")
            print(classification_report(y_true, y_pred, target_names=self.target_names))

        return self.results

    def generate_visualizations(self):
        """Generates high-resolution visualization figures."""
        plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')

        # 1. Metrics Comparison Chart
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        lr_scores = [self.results["Logistic Regression"][k] for k in ['accuracy', 'precision', 'recall', 'f1']]
        dt_scores = [self.results["Decision Tree"][k] for k in ['accuracy', 'precision', 'recall', 'f1']]

        x = np.arange(len(metrics))
        width = 0.35

        fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
        rects1 = ax.bar(x - width/2, lr_scores, width, label='Logistic Regression', color='#1f77b4')
        rects2 = ax.bar(x + width/2, dt_scores, width, label='Decision Tree', color='#ff7f0e')

        ax.set_ylabel('Score', fontsize=12, fontweight='bold')
        ax.set_title('Task 02: Model Performance Metrics Comparison', fontsize=14, fontweight='bold', pad=15)
        ax.set_xticks(x)
        ax.set_xticklabels(metrics, fontsize=11, fontweight='bold')
        ax.set_ylim(0.85, 1.02)
        ax.legend(loc='lower right', frameon=True)

        for rects in [rects1, rects2]:
            for rect in rects:
                height = rect.get_height()
                ax.annotate(f'{height:.3f}',
                            xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, 3), textcoords="offset points",
                            ha='center', va='bottom', fontsize=9, fontweight='bold')

        plt.tight_layout()
        metrics_path = os.path.join(self.figures_dir, "metrics_comparison.png")
        plt.savefig(metrics_path)
        plt.close()
        print(f"[+] Saved: {metrics_path}")

        # 2. Confusion Matrices Plot
        fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=300)
        cm_models = [("Logistic Regression", self.results["Logistic Regression"]["cm"]),
                     ("Decision Tree", self.results["Decision Tree"]["cm"])]

        for idx, (title, cm) in enumerate(cm_models):
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx],
                        xticklabels=self.target_names, yticklabels=self.target_names,
                        cbar=False, annot_kws={"size": 14, "fontweight": "bold"})
            axes[idx].set_title(f"{title}\nConfusion Matrix", fontsize=12, fontweight='bold')
            axes[idx].set_xlabel("Predicted Diagnosis", fontsize=11)
            axes[idx].set_ylabel("Actual Diagnosis", fontsize=11)

        plt.tight_layout()
        cm_path = os.path.join(self.figures_dir, "confusion_matrices.png")
        plt.savefig(cm_path)
        plt.close()
        print(f"[+] Saved: {cm_path}")

        # 3. ROC Curves
        fig, ax = plt.subplots(figsize=(7, 6), dpi=300)
        for name, color in [("Logistic Regression", "#1f77b4"), ("Decision Tree", "#ff7f0e")]:
            ax.plot(self.results[name]["fpr"], self.results[name]["tpr"],
                    label=f'{name} (AUC = {self.results[name]["auc"]:.3f})',
                    color=color, lw=2)

        ax.plot([0, 1], [0, 1], 'k--', lw=1.5, label='Random Chance')
        ax.set_xlim([-0.02, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate (1 - Specificity)', fontsize=11, fontweight='bold')
        ax.set_ylabel('True Positive Rate (Recall / Sensitivity)', fontsize=11, fontweight='bold')
        ax.set_title('Receiver Operating Characteristic (ROC) Curve', fontsize=13, fontweight='bold', pad=12)
        ax.legend(loc="lower right", frameon=True)
        plt.tight_layout()
        roc_path = os.path.join(self.figures_dir, "roc_curves.png")
        plt.savefig(roc_path)
        plt.close()
        print(f"[+] Saved: {roc_path}")

        # 4. Feature Importance / Logistic Coefficients
        # Top 10 important features for Decision Tree
        importances = self.dt_model.feature_importances_
        indices = np.argsort(importances)[::-1][:10]

        fig, ax = plt.subplots(figsize=(10, 5), dpi=300)
        y_labels = [self.feature_names[i] for i in indices]
        sns.barplot(x=importances[indices], y=y_labels, hue=y_labels, palette='viridis', legend=False, ax=ax)
        ax.set_title('Top 10 Feature Importances (Decision Tree)', fontsize=13, fontweight='bold')
        ax.set_xlabel('Gini Importance', fontsize=11)
        plt.tight_layout()
        feat_path = os.path.join(self.figures_dir, "feature_importance.png")
        plt.savefig(feat_path)
        plt.close()
        print(f"[+] Saved: {feat_path}")
