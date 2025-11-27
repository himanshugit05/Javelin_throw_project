# report_plots.py
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

BASE_DIR = r"C:\Users\gupta\Downloads\javelin_yolo_project"
FEATURE_FILE = os.path.join(BASE_DIR, "features", "extracted_features.csv")
LABEL_FILE = os.path.join(BASE_DIR, "real_distances.csv")
MODEL_PATH = os.path.join(BASE_DIR, "models", "regressor_with_synth.pkl")

SAVE_DIR = os.path.join(BASE_DIR, "report_plots")
os.makedirs(SAVE_DIR, exist_ok=True)

# ----------------------------
# Load Data + Model
# ----------------------------
features_df = pd.read_csv(FEATURE_FILE)
labels_df = pd.read_csv(LABEL_FILE)
df = pd.merge(features_df, labels_df, on="video_name", how="inner")

bundle = joblib.load(MODEL_PATH)
model = bundle["model"]
scaler = bundle["scaler"]

feature_cols = ["max_wrist_speed", "elbow_angle", "release_height",
                "wrist_horizontal_range", "wrist_vertical_range", "fps"]

X = df[feature_cols].values
y = df["real_distance"].values
X_scaled = scaler.transform(X)
pred = model.predict(X_scaled)
errors = abs(pred - y)
residuals = pred - y

# ----------------------------
# 1. Feature Distribution Plots
# ----------------------------
plt.figure(figsize=(12, 8))
for i, col in enumerate(feature_cols):
    plt.subplot(2, 3, i + 1)
    plt.hist(df[col], bins=10, color='skyblue')
    plt.title(f"{col} Distribution")
    plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "feature_distributions.png"), dpi=300)
plt.close()

# ----------------------------
# 2. Predicted vs Real Scatter Plot
# ----------------------------
plt.figure(figsize=(7, 5))
plt.scatter(y, pred, s=70, color='blue')
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', label="Ideal Fit")
plt.xlabel("Real Distance (m)")
plt.ylabel("Predicted Distance (m)")
plt.title("Predicted vs Real Throw Distance")
plt.grid(True)
plt.legend()
plt.savefig(os.path.join(SAVE_DIR, "pred_vs_real.png"), dpi=300)
plt.close()

# ----------------------------
# 3. Absolute Error Plot
# ----------------------------
plt.figure(figsize=(7, 5))
plt.plot(errors, marker='o', color='green')
plt.xlabel("Sample Index")
plt.ylabel("Absolute Error (m)")
plt.title("Absolute Error per Sample")
plt.grid(True)
plt.savefig(os.path.join(SAVE_DIR, "absolute_error.png"), dpi=300)
plt.close()

# ----------------------------
# 4. Residual Histogram
# ----------------------------
plt.figure(figsize=(7, 5))
plt.hist(residuals, bins=10, color='orange')
plt.xlabel("Error (Pred - Real) (m)")
plt.ylabel("Frequency")
plt.title("Prediction Error Distribution")
plt.grid(True)
plt.savefig(os.path.join(SAVE_DIR, "residual_histogram.png"), dpi=300)
plt.close()

# ----------------------------
# 5. Feature Importance (Only for RandomForest)
# ----------------------------
if hasattr(model, "feature_importances_"):
    plt.figure(figsize=(7, 5))
    plt.barh(feature_cols, model.feature_importances_, color='purple')
    plt.xlabel("Importance Score")
    plt.title("Feature Importance (RandomForest)")
    plt.grid(True, axis='x')
    plt.savefig(os.path.join(SAVE_DIR, "feature_importance.png"), dpi=300)
    plt.close()
