# =============================================================
# Classification Model - Bank Marketing Dataset
# Logistic Regression + SageMaker-ready inference script
# =============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, classification_report)
import joblib
import os

# ── 1. Load dataset ───────────────────────────────────────────
# Downloads automatically from UCI via URL
print("Loading Bank Marketing dataset...")
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00222/bank-additional-full.csv"

try:
    df = pd.read_csv(url, sep=";")
    print(f"Downloaded from UCI. Shape: {df.shape}")
except Exception:
    # Fallback: load from local file if already downloaded
    df = pd.read_csv("bank-additional-full.csv", sep=";")
    print(f"Loaded locally. Shape: {df.shape}")

print(df.head())

# ── 2. EDA ────────────────────────────────────────────────────
print("\n── EDA ──")
print(df.describe())
print("\nMissing values:", df.isnull().sum().sum())
print("\nTarget distribution:\n", df["y"].value_counts())

# Target balance
df["y"].value_counts().plot(kind="bar", color=["steelblue", "coral"])
plt.title("Target Class Distribution (y: subscribed?)")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig("target_class_distribution.png")
plt.close()
print("Saved target_class_distribution.png")

# ── 3. Preprocessing ──────────────────────────────────────────
# Encode target
df["y"] = (df["y"] == "yes").astype(int)

# Encode all categorical columns
cat_cols = df.select_dtypes(include="object").columns.tolist()
print(f"\nEncoding categorical columns: {cat_cols}")

encoders = {}
for col in cat_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le

# Split features and target
X = df.drop("y", axis=1)
y = df["y"]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\nTrain size: {X_train.shape[0]} | Test size: {X_test.shape[0]}")

# Normalize numeric features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# ── 4. Train model ────────────────────────────────────────────
print("\nTraining Logistic Regression classifier...")
model = LogisticRegression(
    max_iter=1000,
    random_state=42,
    class_weight="balanced"   # handles class imbalance
)
model.fit(X_train_scaled, y_train)
print("Training complete.")

# ── 5. Evaluate ───────────────────────────────────────────────
y_pred      = model.predict(X_test_scaled)
y_pred_prob = model.predict_proba(X_test_scaled)[:, 1]

acc  = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec  = recall_score(y_test, y_pred)
f1   = f1_score(y_test, y_pred)
cm   = confusion_matrix(y_test, y_pred)

print("\n── Evaluation Metrics ──")
print(f"  Accuracy  : {acc:.4f}")
print(f"  Precision : {prec:.4f}")
print(f"  Recall    : {rec:.4f}")
print(f"  F1 Score  : {f1:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=["No", "Yes"]))

# Confusion matrix plot
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["No","Yes"], yticklabels=["No","Yes"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.savefig("confusion_matrix.png")
plt.close()
print("Saved confusion_matrix.png")

# ── 6. Save artifacts ─────────────────────────────────────────
os.makedirs("model_artifacts", exist_ok=True)
joblib.dump(model,    "model_artifacts/logistic_classifier.joblib")
joblib.dump(scaler,   "model_artifacts/scaler_classification.joblib")
joblib.dump(encoders, "model_artifacts/label_encoders.joblib")
joblib.dump(list(X.columns), "model_artifacts/feature_columns.joblib")
print("\nSaved model_artifacts/logistic_classifier.joblib")
print("Saved model_artifacts/scaler_classification.joblib")
print("Saved model_artifacts/label_encoders.joblib")
print("Saved model_artifacts/feature_columns.joblib")

# ── 7. Inference function (used by SageMaker endpoint) ────────
def predict_subscription(features: dict) -> dict:
    """
    features: dict matching the bank marketing dataset columns
    returns: { prediction: 0 or 1, probability: float, label: 'yes'/'no' }
    """
    model    = joblib.load("model_artifacts/logistic_classifier.joblib")
    scaler   = joblib.load("model_artifacts/scaler_classification.joblib")
    encoders = joblib.load("model_artifacts/label_encoders.joblib")
    columns  = joblib.load("model_artifacts/feature_columns.joblib")

    input_df = pd.DataFrame([features])
    for col, le in encoders.items():
        if col in input_df.columns:
            input_df[col] = le.transform(input_df[col])

    input_df = input_df[columns]
    scaled   = scaler.transform(input_df)
    pred     = int(model.predict(scaled)[0])
    prob     = round(float(model.predict_proba(scaled)[0][1]), 4)

    return {"prediction": pred, "probability": prob,
            "label": "yes" if pred == 1 else "no"}

# Quick test
sample = {
    "age": 35, "job": "management", "marital": "married",
    "education": "university.degree", "default": "no",
    "housing": "yes", "loan": "no", "contact": "cellular",
    "month": "may", "day_of_week": "mon", "duration": 300,
    "campaign": 2, "pdays": 999, "previous": 0,
    "poutcome": "nonexistent", "emp.var.rate": -1.8,
    "cons.price.idx": 92.893, "cons.conf.idx": -46.2,
    "euribor3m": 1.299, "nr.employed": 5099.1
}
print(f"\nSample prediction: {predict_subscription(sample)}")
