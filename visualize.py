# basic_visualizations.py
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from sklearn.metrics import mean_absolute_error

# ---------------------------------------------
# USER PATHS (change if needed)
# ---------------------------------------------
BASE_DIR = r"C:\Users\gupta\Downloads\javelin_yolo_project"

FEATURE_FILE = os.path.join(BASE_DIR, "features", "extracted_features.csv")
LABEL_FILE = os.path.join(BASE_DIR, "real_distances.csv")
MODEL_PATH = os.path.join(BASE_DIR, "models", "regressor_with_synth.pkl")

# ---------------------------------------------
# 1. LOAD DATA + MODEL
# ---------------------------------------------
features_df = pd.read_csv(FEATURE_FILE)
labels_df = pd.read_csv(LABEL_FILE)

# Merge real features + real distances
df = pd.merge(features_df, labels_df, on="video_name", how="inner")

bundle = joblib.load(MODEL_PATH)
model = bundle["model"]
scaler = bundle["scaler"]

feature_cols = ["max_wrist_speed", "elbow_angle", "release_height",
                "wrist_horizontal_range", "wrist_vertical_range", "fps"]

# Prepare inputs
X = df[feature_cols].values
y = df["real_distance"].values
X_scaled = scaler.transform(X)

# Predict
pred = model.predict(X_scaled)
mae = mean_absolute_error(y, pred)

print("MAE:", mae)

# ---------------------------------------------
# 2. PLOT 1: Predicted vs Real Scatter Plot
# ---------------------------------------------
plt.figure(figsize=(7,5))
plt.scatter(y, pred, color='blue')
plt.plot([min(y), max(y)], [min(y), max(y)], 'r--', label="Perfect Fit")
plt.xlabel("Real Distance (m)")
plt.ylabel("Predicted Distance (m)")
plt.title("Predicted vs Real Javelin Distance")
plt.legend()
plt.grid(True)
plt.show()

# ---------------------------------------------
# 3. PLOT 2: Absolute Error Plot
# ---------------------------------------------
errors = abs(pred - y)

plt.figure(figsize=(7,5))
plt.plot(errors, marker='o')
plt.xlabel("Sample Index")
plt.ylabel("Absolute Error (m)")
plt.title("Absolute Error per Sample")
plt.grid(True)
plt.show()

# ---------------------------------------------
# 4. PLOT 3: Error Distribution Histogram
# ---------------------------------------------
plt.figure(figsize=(7,5))
plt.hist(pred - y, bins=10, color='green')
plt.xlabel("Error (Predicted - Real) [m]")
plt.ylabel("Count")
plt.title("Error/Residual Distribution")
plt.grid(True)
plt.show()

# ---------------------------------------------
# 5. PLOT 4: Feature Distributions (Histograms)
# ---------------------------------------------
plt.figure(figsize=(12,8))
for i, col in enumerate(feature_cols):
    plt.subplot(2, 3, i+1)
    plt.hist(df[col], bins=10, color='orange')
    plt.title(col)
    plt.grid(True)

plt.tight_layout()
plt.show()

# ---------------------------------------------
# 6. PLOT 5: Feature Importance (RandomForest only)
# ---------------------------------------------
if hasattr(model, "feature_importances_"):
    plt.figure(figsize=(7,5))
    plt.barh(feature_cols, model.feature_importances_, color='purple')
    plt.xlabel("Importance Score")
    plt.title("Feature Importance")
    plt.grid(True, axis='x')
    plt.show()
else:
    print("Model does not provide feature importance.")
