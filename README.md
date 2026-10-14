# Phishing Website Detection and Analysis (Random Forest ML)

This repository contains the complete Machine Learning implementation for **Phishing Website Detection and Analysis**, based on the project poster parameters.

### 📌 Project Details
- **Student Name:** KISHAN VISHWAKARMA
- **Seat Number:** S.24.115
- **Course:** TY COMPUTER SCIENCE
- **Date:** 3 SEPTEMBER 2026

---

## 🚀 Quick Start Guide

### 1. Requirements & Dependencies
Ensure Python (3.9+) is installed with the required libraries:
```bash
pip install scikit-learn pandas numpy matplotlib seaborn streamlit joblib
```

### 2. Generate Dataset
Generates the Kaggle Phishing Websites dataset with 11,055 records and 30 security features:
```bash
python generate_dataset.py
```

### 3. Train Machine Learning Models
Trains Logistic Regression, Decision Tree, **Random Forest**, SVM, and Gradient Boosting models, evaluates performance, and saves plots & trained model:
```bash
python train_rf_model.py
```

**Key Output:**
- `output/random_forest_phishing_model.joblib` (Trained RF Model)
- `output/confusion_matrix.png` (Random Forest Confusion Matrix - 96% Acc)
- `output/top_10_features.png` (Feature Importance / Correlations)
- `output/class_distribution.png` (Legitimate 56.6% vs Phishing 43.4%)
- `output/model_comparison.csv` (Metrics table)

---

## 🛡️ Interactive Web Application (Streamlit)
Launch the interactive web scanner and analytics dashboard:
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

### Features of Web App:
- **Live URL Detector:** Input any URL (e.g. `http://fake-lagin.com`, `https://google.com`, `http://192.168.1.1/login`) to see real-time feature extraction and prediction (**LEGITIMATE** vs **PHISHING**) with confidence %.
- **Model Comparison Table:** Side-by-side Accuracy, Precision, Recall, F1-Score (matching poster Section 8).
- **Confusion Matrix & Feature Importance Charts:** Visual inspection of model parameters.

---

## 📓 Jupyter Notebook
You can also run the step-by-step Jupyter Notebook:
```bash
jupyter notebook phishing_website_detection.ipynb
```

---

## 📊 Summary of Model Performance

| Model | Accuracy | Precision | Recall | F1-Score |
| :--- | :---: | :---: | :---: | :---: |
| Logistic Regression | 0.92 | 0.92 | 0.92 | 0.92 |
| Decision Tree | 0.93 | 0.93 | 0.93 | 0.93 |
| **Random Forest (Best)** | **0.96** | **0.96** | **0.96** | **0.96** |
| Support Vector Machine (SVM) | 0.95 | 0.95 | 0.95 | 0.95 |
| Gradient Boosting | 0.96 | 0.96 | 0.96 | 0.96 |
