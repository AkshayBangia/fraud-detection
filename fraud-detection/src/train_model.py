"""
Fraud Detection Model — Training Script
Run this once to train and save your model.
Usage: python src/train_model.py
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, precision_score, recall_score, f1_score
)
import xgboost as xgb
import pickle
import os

DATA_PATH = "C:/Users/aksha/Downloads/fraud_detection data/creditcard.csv"
MODEL_PATH = "models/fraud_model.pkl"
SCALER_PATH = "models/scaler.pkl"


def load_data(path):
    print(f"Loading data from {path}...")
    df = pd.read_csv(path)
    print(f"Dataset shape: {df.shape}")
    print(f"Fraud cases: {df['Class'].sum()} ({df['Class'].mean()*100:.2f}%)")
    return df


def preprocess(df):
    # Scale Amount and Time (V1-V28 are already PCA-transformed)
    scaler = StandardScaler()
    df["Amount_scaled"] = scaler.fit_transform(df[["Amount"]])
    df["Time_scaled"] = scaler.fit_transform(df[["Time"]])

    feature_cols = [c for c in df.columns if c.startswith("V")] + ["Amount_scaled", "Time_scaled"]
    X = df[feature_cols]
    y = df["Class"]
    return X, y, scaler


def evaluate(model, X_test, y_test, name):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    print(f"\n{'='*40}")
    print(f"Model: {name}")
    print(f"{'='*40}")
    print(f"Precision : {precision_score(y_test, y_pred):.4f}")
    print(f"Recall    : {recall_score(y_test, y_pred):.4f}")
    print(f"F1 Score  : {f1_score(y_test, y_pred):.4f}")
    print(f"ROC-AUC   : {roc_auc_score(y_test, y_prob):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["Legit", "Fraud"]))
    return roc_auc_score(y_test, y_prob)


def train():
    if not os.path.exists(DATA_PATH):
        print(f"\n ERROR: '{DATA_PATH}' not found.")
        print("Please download the dataset from:")
        print("  https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud")
        print("And place 'creditcard.csv' inside the 'data/' folder.\n")
        return

    df = load_data(DATA_PATH)
    X, y, scaler = preprocess(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"\nTrain size: {len(X_train)} | Test size: {len(X_test)}")

    # --- Logistic Regression (baseline) ---
    print("\nTraining Logistic Regression (baseline)...")
    lr = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)
    lr.fit(X_train, y_train)
    evaluate(lr, X_test, y_test, "Logistic Regression")

    # --- Random Forest ---
    print("\nTraining Random Forest...")
    rf = RandomForestClassifier(
        n_estimators=100, class_weight="balanced",
        random_state=42, n_jobs=-1
    )
    rf.fit(X_train, y_train)
    evaluate(rf, X_test, y_test, "Random Forest")

    # --- XGBoost (best model) ---
    print("\nTraining XGBoost...")
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    xgb_model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        scale_pos_weight=scale_pos_weight,
        eval_metric="aucpr",
        random_state=42,
        use_label_encoder=False
    )
    xgb_model.fit(X_train, y_train)
    auc = evaluate(xgb_model, X_test, y_test, "XGBoost")

    # Save the best model (XGBoost)
    os.makedirs("models", exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(xgb_model, f)
    with open(SCALER_PATH, "wb") as f:
        pickle.dump(scaler, f)

    print(f"\n Model saved to {MODEL_PATH}")
    print(f" Scaler saved to {SCALER_PATH}")
    print(f"\n Final XGBoost ROC-AUC: {auc:.4f}")
    print("\nRun the app now with: streamlit run app.py")


if __name__ == "__main__":
    train()
