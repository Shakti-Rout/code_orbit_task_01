"""
Classification Model Implementation
Dataset: Breast Cancer Wisconsin (Diagnostic) Dataset
Models: Logistic Regression and Decision Tree Classifier
Evaluation Metrics: Accuracy, Precision, Recall, F1-Score, Confusion Matrix
"""

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
    classification_report
)
import pandas as pd


def main():
    print("=" * 60)
    print("   SIMPLE CLASSIFICATION MODEL DEMO: BREAST CANCER DATASET")
    print("=" * 60)

    # 1. Load Dataset
    data = load_breast_cancer()
    X = data.data
    y = data.target
    target_names = data.target_names  # 0: 'malignant', 1: 'benign'

    print(f"\n[1] Dataset Overview:")
    print(f"  - Total samples: {X.shape[0]}")
    print(f"  - Total features: {X.shape[1]}")
    print(f"  - Target classes: {list(enumerate(target_names))}")
    print(f"  - Class distribution: Malignant (0) = {sum(y == 0)}, Benign (1) = {sum(y == 1)}")

    # 2. Split Data into Training and Testing Sets (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"\n[2] Data Split:")
    print(f"  - Training samples: {X_train.shape[0]}")
    print(f"  - Testing samples:  {X_test.shape[0]}")

    # Standardize features for Logistic Regression (beneficial for gradient-based convergence)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 3. Model 1: Logistic Regression
    lr_model = LogisticRegression(random_state=42, max_iter=1000)
    lr_model.fit(X_train_scaled, y_train)
    y_pred_lr = lr_model.predict(X_test_scaled)

    # 4. Model 2: Decision Tree Classifier
    dt_model = DecisionTreeClassifier(random_state=42, max_depth=4)
    dt_model.fit(X_train, y_train)
    y_pred_dt = dt_model.predict(X_test)

    # 5. Model Evaluation
    models = {
        "Logistic Regression (Standardized)": (y_test, y_pred_lr),
        "Decision Tree (max_depth=4)": (y_test, y_pred_dt)
    }

    print("\n" + "=" * 60)
    print("   MODEL EVALUATION RESULTS (TEST SET)")
    print("=" * 60)

    for name, (actual, preds) in models.items():
        acc = accuracy_score(actual, preds)
        prec = precision_score(actual, preds)
        rec = recall_score(actual, preds)
        f1 = f1_score(actual, preds)
        cm = confusion_matrix(actual, preds)

        print(f"\n--- {name} ---")
        print(f"Accuracy  : {acc:.4f} ({acc * 100:.2f}%)")
        print(f"Precision : {prec:.4f} ({prec * 100:.2f}%)")
        print(f"Recall    : {rec:.4f} ({rec * 100:.2f}%)")
        print(f"F1-Score  : {f1:.4f}")
        print("\nConfusion Matrix:")
        print(f"  TN={cm[0,0]} | FP={cm[0,1]}")
        print(f"  FN={cm[1,0]} | TP={cm[1,1]}")
        print("\nDetailed Classification Report:")
        print(classification_report(actual, preds, target_names=target_names))


if __name__ == "__main__":
    main()
