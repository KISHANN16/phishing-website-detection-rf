# 🛡️ PhishingGuard AI — Real-Time Threat Intelligence Platform
### Autonomous Phishing Website Detection using Machine Learning (Random Forest Classifier)

[![GitHub License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![Model Accuracy](https://img.shields.io/badge/Model%20Accuracy-97.11%25-brightgreen.svg)]()
[![Dataset](https://img.shields.io/badge/Dataset-UCI%20%7C%20Kaggle%20(11k%20Records)-orange.svg)](https://www.kaggle.com/datasets/eswarchandt/phishing-website-detector)

---

## 📌 Executive Summary

Phishing is one of the most widespread cybersecurity vectors targeting users and corporate networks. Attackers engineer deceptive websites that mimic legitimate banking, e-commerce, and enterprise portals to capture credentials and authentication tokens.

**PhishingGuard AI** is a production-grade machine learning platform engineered to detect and classify phishing websites in real-time. By extracting **30 structural, protocol, and domain-heuristic security attributes** from raw URLs, the underlying **100-Tree Random Forest Classifier** delivers **97.11% test accuracy** on 11,055 real-world benchmark websites from the **UCI Machine Learning Repository** and **Kaggle**.

---

## 🚀 Key Platform Features

- **⚡ Real-Time Threat Scanner:** Immediate feature extraction and threat inference for arbitrary URLs with confidence percentages and diagnostic breakdowns.
- **⚡ Viva Demonstration Presets:** 1-click test buttons for instantaneous live testing of phishing patterns (IP addresses, shortened URLs, brand-spoofing domains) and legitimate platforms.
- **🗄️ Relational Database Management:** SQLite-powered database (`phishing_database.db`) indexing all 11,055 benchmark records with an interactive SQL terminal runner.
- **📊 Diagnostic Analytics:** High-resolution confusion matrix telemetry, Gini impurity feature importance rankings, and multi-algorithm benchmark comparisons.

---

## 📊 Benchmark Model Performance

Evaluated on **3,317 unseen test samples** with stratified sampling:

| Classifier Algorithm | Test Accuracy | Precision | Recall | F1-Score |
| :--- | :---: | :---: | :---: | :---: |
| **Random Forest (Selected)** | **97.11%** | **0.9751** | **0.9592** | **0.9671** |
| Gradient Boosting | 96.40% | 0.9680 | 0.9590 | 0.9634 |
| Support Vector Machine (RBF) | 95.20% | 0.9510 | 0.9480 | 0.9495 |
| Decision Tree (Single Tree) | 93.10% | 0.9320 | 0.9290 | 0.9305 |
| Logistic Regression (Baseline) | 92.30% | 0.9210 | 0.9200 | 0.9205 |

---

## 🗂️ Dataset & Attributes

- **Source:** [UCI Machine Learning Repository (Dataset #327)](https://archive.ics.uci.edu/dataset/327/phishing+websites)
- **Kaggle Mirror:** [Phishing Website Detector — 11,055 Records](https://www.kaggle.com/datasets/eswarchandt/phishing-website-detector)
- **Total Records:** 11,055 website instances
- **Class Balance:** 6,157 Legitimate (55.7%) vs 4,898 Phishing (44.3%)
- **Indexed Database:** `phishing_database.db` (SQLite relational table `phishing_websites`)

### Top Diagnostic Features:
1. **SSL / TLS Certificate State** (`SSLfinal_State`)
2. **Anchor Link Target Structure** (`URL_of_Anchor`)
3. **Domain Registration Length** (`Domain_registeration_length`)
4. **Presence of Raw IP Address** (`having_IP_Address`)
5. **URL Character Length** (`URL_Length`)
6. **Subdomain Hierarchy Depth** (`having_Sub_Domain`)

---

## 💻 Quickstart & Execution

### 1. Environment Setup
```bash
git clone https://github.com/KISHANN16/phishing-website-detection-rf.git
cd phishing-website-detection-rf
pip install -r requirements.txt # or: pip install scikit-learn pandas numpy matplotlib seaborn streamlit joblib
```

### 2. Launch the Streamlit Threat Platform
```bash
streamlit run app.py
```
Open **`http://localhost:8501`** in your browser.

### 3. Model Training Pipeline (Optional)
To retrain the Random Forest model from the SQLite database:
```bash
python setup_real_data.py
```

---

## 📁 Repository Structure

```
phishing-website-detection-rf/
├── app.py                            # Enterprise Streamlit Threat Platform
├── phishing_database.db               # SQLite database (11,055 indexed records)
├── phishing_dataset_real.csv         # Clean tabular CSV with verified labels
├── setup_real_data.py                # Pipeline: ARFF -> SQLite DB -> Model Training
├── train_rf_model.py                 # Multi-algorithm benchmark & evaluation script
├── url_feature_extractor.py          # Real-time URL parsing & feature vector engine
├── generate_dataset.py               # Synthetic benchmark generator
├── phishing_website_detection.ipynb  # Comprehensive Jupyter Notebook walkthrough
└── output/
    ├── confusion_matrix.png          # High-resolution confusion matrix
    ├── top_10_features.png           # Gini feature importance chart
    └── class_distribution.png        # Class balance visualization
```

---

*Academic Project Submission — TY Computer Science*
