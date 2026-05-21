"""
Fraud Detection — Exploratory Data Analysis
Run this to understand the dataset before training.
Usage: python src/eda.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

DATA_PATH = "C:/Users/aksha/Downloads/fraud_detection data/creditcard.csv"
os.makedirs("data/plots", exist_ok=True)

sns.set_theme(style="whitegrid", palette="muted")


def run_eda():
    if not os.path.exists(DATA_PATH):
        print(f"ERROR: '{DATA_PATH}' not found. Download from Kaggle first.")
        return

    df = pd.read_csv(DATA_PATH)

    print("=" * 50)
    print("DATASET OVERVIEW")
    print("=" * 50)
    print(f"Shape         : {df.shape}")
    print(f"Columns       : {list(df.columns)}")
    print(f"Missing values: {df.isnull().sum().sum()}")
    print(f"Duplicates    : {df.duplicated().sum()}")
    print(f"\nClass distribution:")
    print(df["Class"].value_counts())
    print(f"\nFraud rate    : {df['Class'].mean()*100:.4f}%")

    # --- Plot 1: Class Imbalance ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    counts = df["Class"].value_counts()
    axes[0].bar(["Legitimate", "Fraud"], counts.values, color=["#4CAF50", "#F44336"])
    axes[0].set_title("Class Distribution (Raw Count)")
    axes[0].set_ylabel("Count")
    for i, v in enumerate(counts.values):
        axes[0].text(i, v + 500, f"{v:,}", ha="center", fontweight="bold")

    axes[1].pie(counts.values, labels=["Legitimate", "Fraud"],
                autopct="%1.2f%%", colors=["#4CAF50", "#F44336"], startangle=90)
    axes[1].set_title("Class Distribution (%)")
    plt.tight_layout()
    plt.savefig("data/plots/class_distribution.png", dpi=150)
    print("\nSaved: data/plots/class_distribution.png")
    plt.close()

    # --- Plot 2: Transaction Amount by Class ---
    fig, ax = plt.subplots(figsize=(10, 5))
    df[df["Class"] == 0]["Amount"].clip(upper=2000).hist(
        bins=60, alpha=0.6, color="#4CAF50", label="Legitimate", ax=ax)
    df[df["Class"] == 1]["Amount"].clip(upper=2000).hist(
        bins=60, alpha=0.8, color="#F44336", label="Fraud", ax=ax)
    ax.set_title("Transaction Amount Distribution (clipped at $2000)")
    ax.set_xlabel("Amount ($)")
    ax.set_ylabel("Count")
    ax.legend()
    plt.tight_layout()
    plt.savefig("data/plots/amount_distribution.png", dpi=150)
    print("Saved: data/plots/amount_distribution.png")
    plt.close()

    # --- Plot 3: Transaction Time by Class ---
    fig, ax = plt.subplots(figsize=(10, 5))
    df[df["Class"] == 0]["Time"].hist(
        bins=60, alpha=0.6, color="#4CAF50", label="Legitimate", ax=ax)
    df[df["Class"] == 1]["Time"].hist(
        bins=60, alpha=0.8, color="#F44336", label="Fraud", ax=ax)
    ax.set_title("Transaction Time Distribution (seconds from first transaction)")
    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Count")
    ax.legend()
    plt.tight_layout()
    plt.savefig("data/plots/time_distribution.png", dpi=150)
    print("Saved: data/plots/time_distribution.png")
    plt.close()

    # --- Plot 4: Correlation heatmap (fraud transactions) ---
    fraud_df = df[df["Class"] == 1]
    feature_cols = [c for c in df.columns if c.startswith("V")][:10]
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(fraud_df[feature_cols].corr(), annot=True, fmt=".2f",
                cmap="coolwarm", center=0, ax=ax)
    ax.set_title("Correlation Heatmap — Top 10 Features (Fraud Transactions)")
    plt.tight_layout()
    plt.savefig("data/plots/correlation_heatmap.png", dpi=150)
    print("Saved: data/plots/correlation_heatmap.png")
    plt.close()

    print("\nEDA complete! All plots saved to data/plots/")
    print("Next step: python src/train_model.py")


if __name__ == "__main__":
    run_eda()
