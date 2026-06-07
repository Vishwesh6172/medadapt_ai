# MedAdapt-AI 🩺

> **Explainable Diabetes Risk Prediction** using GAN-Augmented Deep Learning and SHAP-Based Feature Selection

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![SHAP](https://img.shields.io/badge/XAI-SHAP-brightgreen)](https://shap.readthedocs.io/)

---

## Overview

MedAdapt-AI is a clinical decision-support framework for early diabetes risk prediction. It tackles two of the biggest challenges in medical AI — **class imbalance** and **lack of interpretability** — by combining GAN-based data augmentation with SHAP-driven explainability.

The system is designed for healthcare professionals who need not just accurate predictions, but transparent, trustworthy reasoning behind every result.

---

## Key Features

-**GAN + SMOTE-ENN** hybrid augmentation for realistic minority-class synthesis
-**SHAP-based feature selection** for explainable, clinically meaningful inputs
-**Dual deep learning models** — DNN and RBFNN — with comparative evaluation
-**Automated visual analytics** — EDA plots, SHAP diagrams, confusion matrices, and more
- **Multi-dataset support** — works with Pima Indians and Early Stage Diabetes datasets
- 📄 **Auto-generated reports** — evaluation metrics, feature rankings, explainability outputs

---

## Architecture

```
Raw Dataset (PIMA)
        │
        ▼
    Cleaning  →  Encoding
        │
        ▼
  ◆ Controller Decision Engine (SMOTE + GAN) ◆
        │
        ▼
  Train-Test Split  →  Scaling  →  SMOTE (Balancing)
        │
        ▼
  SHAP Feature Selection
        │
        ▼
  Cross Validation (XGBoost)
        │
        ▼
  GAN (Data Augmentation)  →  Final Training (XGBoost)
        │
        ▼
      Evaluation
   ┌──────┼──────┐
   ▼      ▼      ▼
Confusion  ROC+AUC  SHAP
Matrix             Importance
```

---

## Results

Evaluated on the **PIMA Indians Diabetes Dataset**:

| Metric | Score |
|---|---|
| **Accuracy** | **91.99%** |
| **AUC Score** | **94.34%** |

### Key Finding — Top Predictive Feature
SHAP analysis identified **HbA1c** as the dominant predictor by a significant margin, followed by glucose-related biomarkers, diabetes risk score, and insulin levels. The remaining features contribute much smaller but collectively meaningful signals.

---

## Datasets

### 1. Pima Indians Diabetes Dataset
Medical records from female patients of Pima Indian heritage.

| Feature | Description |
|---|---|
| Pregnancies | Number of pregnancies |
| Glucose | Plasma glucose concentration |
| Blood Pressure | Diastolic blood pressure (mm Hg) |
| Skin Thickness | Triceps skin fold thickness (mm) |
| Insulin | 2-hour serum insulin (mu U/ml) |
| BMI | Body mass index |
| Diabetes Pedigree Function | Genetic influence score |
| Age | Age in years |
| **Outcome** | **0 = Non-Diabetic, 1 = Diabetic** |

### 2. Early Stage Diabetes Risk Prediction Dataset
Symptom-based dataset covering early clinical indicators such as polyuria, polydipsia, sudden weight loss, visual blurring, and more.

**Target:** `Diabetes Class` (Positive / Negative)

---

## Methodology

### Step 1 — Exploratory Data Analysis
Statistical inspection, correlation analysis, class distribution checks, and missing value handling. Outputs include distribution plots, heatmaps, and class balance charts.

### Step 2 — Data Preprocessing & Augmentation
- Clinically invalid zero-values are identified and imputed
- Features are standardized using z-score normalization
- A **GAN** (Generator + Discriminator) is trained to synthesize realistic diabetic samples
- **SMOTE-ENN** further cleans noise and balances the final dataset

### Step 3 — SHAP-Based Feature Selection
A Random Forest is trained on the balanced dataset, then **SHAP TreeExplainer** computes Shapley values for every feature. Features are ranked by mean absolute SHAP value and the top-K are selected for deep learning.

Outputs: SHAP summary plots, importance plots, decision plots, and a feature ranking report.

### Step 4 — Deep Neural Network (DNN)
A fully connected neural network with dropout regularization, trained on SHAP-selected features.

### Step 5 — Radial Basis Function Neural Network (RBFNN)
An RBF network with Gaussian activation, used as a benchmark comparison against the DNN.

### Step 6 — Evaluation
Both models are evaluated using Accuracy, Precision, Recall, F1-Score, ROC-AUC, and Confusion Matrix. The best-performing model is selected.

---

## Generated Outputs

| Category | File |
|---|---|
| EDA | `eda_distributions.png`, `eda_correlation.png`, `eda_class_balance.png` |
| SHAP | `shap_summary.png`, `shap_importance.png`, `shap_decision_plot.png`, `shap_architecture_diagram.png` |
| DNN | `dnn_training_history.png`, `dnn_confusion_matrix.png` |
| RBFNN | `rbfnn_confusion_matrix.png` |
| Evaluation | `model_comparison.png`, `evaluation_metrics.csv` |

---

## Tech Stack

| Category | Tools |
|---|---|
| Language | Python 3.8+ |
| Deep Learning | TensorFlow, Keras |
| Machine Learning | Scikit-Learn, XGBoost |
| Explainable AI | SHAP |
| Augmentation | GAN, SMOTE-ENN (imbalanced-learn) |
| Data Processing | NumPy, Pandas |
| Visualization | Matplotlib, Seaborn |

---

## Why Explainability Matters

Most clinical ML systems act as black boxes — they generate a prediction but offer no reasoning. MedAdapt-AI uses **SHAP (SHapley Additive exPlanations)** to show exactly which features drove each prediction and by how much. This makes the system:

- **Trustworthy** — clinicians can verify reasoning
- **Auditable** — predictions can be traced and challenged
- **Actionable** — patients and doctors see which risk factors matter most

---

## Future Roadmap

- [ ] Federated learning for privacy-preserving multi-hospital training
- [ ] Transformer-based prediction models
- [ ] Multi-disease diagnosis support
- [ ] Real-time hospital deployment pipeline
- [ ] Web-based clinical dashboard
- [ ] EHR (Electronic Health Record) integration
- [ ] Mobile application

---

<p align="center">Built for transparency, accuracy, and clinical trust.</p>
