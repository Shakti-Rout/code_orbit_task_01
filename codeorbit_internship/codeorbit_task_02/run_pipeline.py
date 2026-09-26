"""
CodeOrbit Tech - Machine Learning Internship
Task 02 Execution Entrypoint
Run this script to reproduce the entire classification workflow:
- Load Breast Cancer Diagnostic Dataset
- Preprocess and split into Train/Test sets (stratified)
- Train Logistic Regression & Decision Tree models
- Evaluate on Accuracy, Precision, Recall, and F1
- Generate figures and store artifacts
"""

import sys
import os

# Add local directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from src.classification_pipeline import ClassificationPipeline


def main():
    print("=" * 65)
    print(" CODEORBIT TECH - MACHINE LEARNING INTERNSHIP (TASK 02)")
    print(" Task: Simple Classification Model & Performance Evaluation")
    print("=" * 65)

    base_dir = os.path.dirname(os.path.abspath(__file__))
    pipeline = ClassificationPipeline(
        data_dir=os.path.join(base_dir, "data"),
        figures_dir=os.path.join(base_dir, "figures"),
        random_state=42
    )

    # 1. Load Data
    pipeline.load_data()

    # 2. Split and Preprocess
    pipeline.split_and_preprocess(test_size=0.2)

    # 3. Train Models
    pipeline.train_models()

    # 4. Evaluate Models
    pipeline.evaluate_models()

    # 5. Generate Figures
    print("\n[+] Generating high-resolution evaluation figures...")
    pipeline.generate_visualizations()

    print("\n" + "=" * 65)
    print(" Pipeline execution finished successfully!")
    print(f" Datasets saved in : {os.path.join(base_dir, 'data')}")
    print(f" Figures saved in  : {os.path.join(base_dir, 'figures')}")
    print("=" * 65)


if __name__ == "__main__":
    main()
