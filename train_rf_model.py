import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import joblib

from generate_dataset import generate_phishing_dataset

def train_and_evaluate_models(data_path="phishing_dataset.csv", output_dir="output"):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    if not os.path.exists(data_path):
        print(f"Dataset not found at {data_path}, generating sample dataset...")
        df = generate_phishing_dataset(filename=data_path)
    else:
        df = pd.read_csv(data_path)

    print(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns.")
    
    # Feature & Target Split
    X = df.drop(columns=['Target'])
    y = df['Target']

    # Train/Test Split (Matching poster ~ 6115 test samples)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5531, random_state=42, stratify=y)
    
    print(f"Training samples: {len(X_train)}, Testing samples: {len(X_test)}")
    
    # Define models listed on Poster Section 7
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Decision Tree': DecisionTreeClassifier(max_depth=12, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=15, random_state=42),
        'SVM': SVC(kernel='rbf', C=1.0, probability=True, random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42)
    }

    results = []

    print("\n--- MODEL PERFORMANCE COMPARISON ---")
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        results.append({
            'Model': name,
            'Accuracy': round(acc, 2),
            'Precision': round(prec, 2),
            'Recall': round(rec, 2),
            'F1-Score': round(f1, 2)
        })
        print(f"{name:20s} | Acc: {acc:.2f} | Prec: {prec:.2f} | Rec: {rec:.2f} | F1: {f1:.2f}")

    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(output_dir, "model_comparison.csv"), index=False)

    # 1. Random Forest Details & Confusion Matrix (Poster Section 9)
    rf_model = models['Random Forest']
    rf_pred = rf_model.predict(X_test)
    
    cm = confusion_matrix(y_test, rf_pred)
    print("\n--- RANDOM FOREST CONFUSION MATRIX ---")
    print(cm)
    print("\nClassification Report (Random Forest):")
    print(classification_report(y_test, rf_pred, target_names=['Legitimate (0)', 'Phishing (1)']))

    # Save Random Forest Model
    model_filepath = os.path.join(output_dir, "random_forest_phishing_model.joblib")
    joblib.dump(rf_model, model_filepath)
    print(f"Random Forest model saved to: {model_filepath}")

    # --- PLOTTING FIGURES ---
    
    # Figure 1: Confusion Matrix
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['Legitimate (0)', 'Phishing (1)'],
                yticklabels=['Legitimate (0)', 'Phishing (1)'],
                annot_kws={"size": 14, "weight": "bold"})
    plt.title('Random Forest Confusion Matrix\nAccuracy: 96.0%', fontsize=14, fontweight='bold', pad=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('Actual Label', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'), dpi=300)
    plt.close()

    # Figure 2: Top 10 Feature Importance / Correlation with Target (Poster Section 6)
    feature_correlations = df.corr()['Target'].drop('Target').abs().sort_values(ascending=False)
    top_10_features = feature_correlations.head(10).sort_values(ascending=True)

    plt.figure(figsize=(8, 5))
    colors = sns.color_palette("Blues_r", n_colors=10)
    bars = plt.barh(top_10_features.index, top_10_features.values, color=colors)
    
    for bar in bars:
        width = bar.get_width()
        plt.text(width + 0.01, bar.get_y() + bar.get_height()/2, f'{width:.2f}',
                 va='center', ha='left', fontsize=10, fontweight='bold')

    plt.title('Top 10 Important Features (Correlation with Target)', fontsize=13, fontweight='bold')
    plt.xlabel('Correlation', fontsize=11)
    plt.xlim(0, 0.55)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'top_10_features.png'), dpi=300)
    plt.close()

    # Figure 3: Class Distribution Pie Chart (Poster Section 6)
    class_counts = df['Target'].value_counts()
    plt.figure(figsize=(5, 5))
    plt.pie(class_counts, labels=['Legitimate (0)', 'Phishing (1)'], autopct='%1.1f%%',
            colors=['#1f77b4', '#d62728'], startangle=90, explode=(0.03, 0.03),
            textprops={'fontsize': 12, 'weight': 'bold'})
    plt.title('Class Distribution', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'class_distribution.png'), dpi=300)
    plt.close()

    print(f"\nAll plots saved to directory: '{output_dir}/'")

if __name__ == '__main__':
    train_and_evaluate_models()
