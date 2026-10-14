"""
Real Phishing Website Detection System
Data: UCI ML Repository - Phishing Websites Dataset (11,055 records)
Source: https://archive.ics.uci.edu/ml/datasets/phishing+websites
Algorithm: Random Forest Classifier
"""

import numpy as np
import pandas as pd
import sqlite3
import joblib
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, classification_report)

# ─── CONFIG ────────────────────────────────────────────────────────────────────
ARFF_FILE  = "real_phishing_data.arff"
DB_FILE    = "phishing_database.db"
MODEL_FILE = "output/random_forest_phishing_model.joblib"
OUTPUT_DIR = "output"

# ─── STEP 1 : Parse ARFF → DataFrame ───────────────────────────────────────────
def parse_arff(filepath):
    columns, data = [], []
    in_data = False
    with open(filepath, "r", errors="ignore") as f:
        for line in f:
            line = line.strip().replace("\r", "")
            if not line or line.startswith("%"):
                continue
            if line.lower().startswith("@attribute"):
                col = line.split()[1]
                columns.append(col)
            elif line.lower().startswith("@data"):
                in_data = True
            elif in_data and line:
                row = [int(x) for x in line.split(",")]
                data.append(row)
    df = pd.DataFrame(data, columns=columns)
    # UCI dataset: Result column has -1 (phishing) and 1 (legitimate)
    # Convert to 0 (legitimate) and 1 (phishing) — same as Kaggle convention
    df["Result"] = df["Result"].map({1: 0, -1: 1})
    df.rename(columns={"Result": "Target"}, inplace=True)
    print(f"✅ Parsed ARFF: {df.shape[0]} rows, {df.shape[1]} columns")
    return df

# ─── STEP 2 : Store in SQLite Database ─────────────────────────────────────────
def store_in_sqlite(df, db_path):
    conn = sqlite3.connect(db_path)
    df.to_sql("phishing_websites", conn, if_exists="replace", index=True,
              index_label="id")
    # Create summary stats table
    summary = df.groupby("Target").size().reset_index(name="count")
    summary["label"] = summary["Target"].map({0: "Legitimate", 1: "Phishing"})
    summary.to_sql("class_summary", conn, if_exists="replace", index=False)
    conn.commit()
    conn.close()
    print(f"✅ Database saved: {db_path}")
    print(f"   → Table 'phishing_websites': {df.shape[0]} records, {df.shape[1]} columns")

# ─── STEP 3 : Train Random Forest ──────────────────────────────────────────────
def train_model(df):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    X = df.drop(columns=["Target"])
    y = df["Target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y
    )
    print(f"\n📊 Train: {len(X_train)} records | Test: {len(X_test)} records")

    rf = RandomForestClassifier(n_estimators=100, max_depth=15,
                                random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)

    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec  = recall_score(y_test, y_pred)
    f1   = f1_score(y_test, y_pred)
    cm   = confusion_matrix(y_test, y_pred)

    print(f"\n🎯 RANDOM FOREST RESULTS:")
    print(f"   Accuracy  : {acc*100:.2f}%")
    print(f"   Precision : {prec:.4f}")
    print(f"   Recall    : {rec:.4f}")
    print(f"   F1-Score  : {f1:.4f}")
    print(f"\nConfusion Matrix:\n{cm}")
    print(f"\n{classification_report(y_test, y_pred, target_names=['Legitimate','Phishing'])}")

    joblib.dump(rf, MODEL_FILE)
    print(f"\n💾 Model saved → {MODEL_FILE}")

    # ── Plots ──────────────────────────────────────────────────────────────────

    # 1. Confusion Matrix
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=["Legitimate (0)", "Phishing (1)"],
                yticklabels=["Legitimate (0)", "Phishing (1)"],
                annot_kws={"size": 14, "weight": "bold"})
    plt.title(f"Random Forest — Confusion Matrix\nAccuracy: {acc*100:.1f}%",
              fontsize=14, fontweight="bold", pad=12)
    plt.xlabel("Predicted Label", fontsize=12)
    plt.ylabel("Actual Label", fontsize=12)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/confusion_matrix.png", dpi=300)
    plt.close()

    # 2. Top 10 Feature Importance (from model itself, not correlation)
    importances = pd.Series(rf.feature_importances_, index=X.columns)
    top10 = importances.nlargest(10).sort_values()
    plt.figure(figsize=(8, 5))
    colors = sns.color_palette("Blues_r", 10)
    bars = plt.barh(top10.index, top10.values, color=colors)
    for bar in bars:
        w = bar.get_width()
        plt.text(w + 0.002, bar.get_y() + bar.get_height()/2,
                 f"{w:.3f}", va="center", fontsize=9, fontweight="bold")
    plt.title("Top 10 Feature Importances (Random Forest)", fontsize=13, fontweight="bold")
    plt.xlabel("Importance Score", fontsize=11)
    plt.xlim(0, top10.max() + 0.04)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/top_10_features.png", dpi=300)
    plt.close()

    # 3. Class Distribution
    counts = df["Target"].value_counts()
    plt.figure(figsize=(5, 5))
    plt.pie(counts, labels=["Phishing (1)", "Legitimate (0)"],
            autopct="%1.1f%%", colors=["#d62728", "#1f77b4"],
            startangle=90, explode=(0.03, 0.03),
            textprops={"fontsize": 12, "weight": "bold"})
    plt.title("Class Distribution", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/class_distribution.png", dpi=300)
    plt.close()

    print(f"\n📈 Plots saved to '{OUTPUT_DIR}/'")
    return rf, acc, prec, rec, f1, cm

# ─── MAIN ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("  PHISHING WEBSITE DETECTION — REAL DATA PIPELINE")
    print("=" * 60)

    df = parse_arff(ARFF_FILE)
    store_in_sqlite(df, DB_FILE)
    train_model(df)

    print("\n✅ PIPELINE COMPLETE!")
    print(f"   Database  : {DB_FILE}")
    print(f"   ML Model  : {MODEL_FILE}")
    print(f"   Charts    : {OUTPUT_DIR}/")
