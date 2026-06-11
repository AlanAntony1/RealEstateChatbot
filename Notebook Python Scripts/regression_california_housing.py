# =============================================================
# Regression Model - California Housing Dataset
# Random Forest Regressor + SageMaker-ready inference script
# =============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import os

# ── 1. Load dataset ───────────────────────────────────────────
print("Loading California Housing dataset...")
housing = fetch_california_housing(as_frame=True)
df = housing.frame
print(f"Shape: {df.shape}")
print(df.head())

# ── 2. Exploratory Data Analysis ──────────────────────────────
print("\n── EDA ──")
print(df.describe())
print("\nMissing values:", df.isnull().sum().sum())

# Correlation heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Feature Correlation Heatmap")
plt.tight_layout()
plt.savefig("correlation_heatmap.png")
plt.close()
print("Saved correlation_heatmap.png")

# Target distribution
plt.figure(figsize=(8, 4))
sns.histplot(df["MedHouseVal"], bins=50, kde=True)
plt.title("Distribution of Median House Value")
plt.xlabel("Median House Value ($100,000s)")
plt.tight_layout()
plt.savefig("target_distribution.png")
plt.close()
print("Saved target_distribution.png")

# ── 3. Prepare features ───────────────────────────────────────
X = df.drop("MedHouseVal", axis=1)
y = df["MedHouseVal"]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"\nTrain size: {X_train.shape[0]} | Test size: {X_test.shape[0]}")

# Normalize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# ── 4. Train model ────────────────────────────────────────────
print("\nTraining Random Forest Regressor...")
model = RandomForestRegressor(
    n_estimators=100,
    max_depth=None,
    min_samples_split=2,
    random_state=42,
    n_jobs=-1
)
model.fit(X_train_scaled, y_train)
print("Training complete.")

# ── 5. Evaluate ───────────────────────────────────────────────
y_pred = model.predict(X_test_scaled)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae  = mean_absolute_error(y_test, y_pred)
r2   = r2_score(y_test, y_pred)

print("\n── Evaluation Metrics ──")
print(f"  RMSE : {rmse:.4f}")
print(f"  MAE  : {mae:.4f}")
print(f"  R²   : {r2:.4f}")

# Actual vs predicted plot
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.3, s=10)
plt.plot([y_test.min(), y_test.max()],
         [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel("Actual Value")
plt.ylabel("Predicted Value")
plt.title("Actual vs Predicted House Values")
plt.tight_layout()
plt.savefig("actual_vs_predicted.png")
plt.close()
print("Saved actual_vs_predicted.png")

# Feature importance
importances = pd.Series(model.feature_importances_, index=X.columns)
importances.sort_values().plot(kind="barh", figsize=(8, 5))
plt.title("Feature Importances")
plt.tight_layout()
plt.savefig("feature_importances.png")
plt.close()
print("Saved feature_importances.png")

# ── 6. Save artifacts ─────────────────────────────────────────
os.makedirs("model_artifacts", exist_ok=True)
joblib.dump(model,  "model_artifacts/rf_regressor.joblib")
joblib.dump(scaler, "model_artifacts/scaler_regression.joblib")
print("\nSaved model_artifacts/rf_regressor.joblib")
print("Saved model_artifacts/scaler_regression.joblib")

# ── 7. Inference function (used by SageMaker endpoint) ────────
def predict_house_value(features: dict) -> float:
    """
    features: dict with keys:
      MedInc, HouseAge, AveRooms, AveBedrms,
      Population, AveOccup, Latitude, Longitude
    returns: predicted median house value (in $100,000s)
    """
    model  = joblib.load("model_artifacts/rf_regressor.joblib")
    scaler = joblib.load("model_artifacts/scaler_regression.joblib")
    input_df = pd.DataFrame([features])
    scaled   = scaler.transform(input_df)
    return round(float(model.predict(scaled)[0]), 4)

# Quick test
sample = {
    "MedInc": 8.3252, "HouseAge": 41.0, "AveRooms": 6.984,
    "AveBedrms": 1.023, "Population": 322.0, "AveOccup": 2.555,
    "Latitude": 37.88, "Longitude": -122.23
}
print(f"\nSample prediction: ${predict_house_value(sample) * 100_000:,.0f}")
