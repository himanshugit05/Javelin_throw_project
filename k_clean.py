from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
import os

print(">>> K-FOLD VALIDATION STARTED")

# --- USER CONFIGURATION ---
BASE_DIR = r"C:\Users\gupta\Downloads\javelin_yolo_project"
FEATURE_FILE = os.path.join(BASE_DIR, "features", "extracted_features.csv")
LABEL_FILE = os.path.join(BASE_DIR, "real_distances.csv")
SYNTH_FILE = os.path.join(BASE_DIR, "synthetic", "synthetic_features.csv")

# --- LOAD DATA ---
features_df = pd.read_csv(FEATURE_FILE)
labels_df = pd.read_csv(LABEL_FILE)
real_df = pd.merge(features_df, labels_df, on="video_name", how="inner")

# Synthetic optional check
use_synth = True
if use_synth and os.path.exists(SYNTH_FILE):
    synth_df = pd.read_csv(SYNTH_FILE)
    print(f"Loaded synthetic data: {len(synth_df)} rows")
else:
    synth_df = pd.DataFrame()
    print("Synthetic data not found or disabled.")

feature_cols = ["max_wrist_speed", "elbow_angle", "release_height",
                "wrist_horizontal_range", "wrist_vertical_range", "fps"]

X_real = real_df[feature_cols].values
y_real = real_df["real_distance"].values

kf = KFold(n_splits=5, shuffle=True, random_state=42)

mae_scores = []
rmse_scores = []

print(">>> ENTERING K-FOLD LOOP")

for fold, (train_idx, test_idx) in enumerate(kf.split(X_real)):
    print(f"Running fold {fold + 1}...")

    X_train_real = X_real[train_idx]
    y_train_real = y_real[train_idx]

    X_test = X_real[test_idx]
    y_test = y_real[test_idx]

    # Add synthetic data ONLY to training set (avoids data leakage)
    if use_synth and len(synth_df) > 0:
        X_synth = synth_df[feature_cols].values
        y_synth = synth_df["real_distance"].values

        X_train = np.concatenate([X_train_real, X_synth], axis=0)
        y_train = np.concatenate([y_train_real, y_synth], axis=0)
    else:
        X_train = X_train_real
        y_train = y_train_real

    # Scale Data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train Model
    model = RandomForestRegressor(n_estimators=300, random_state=42)
    model.fit(X_train_scaled, y_train)

    # Predict
    preds = model.predict(X_test_scaled)

    # Calculate Metrics
    mae = mean_absolute_error(y_test, preds)
    
    # --- FIX: Calculate MSE first, then square root it manually ---
    mse = mean_squared_error(y_test, preds)
    rmse = np.sqrt(mse)

    mae_scores.append(mae)
    rmse_scores.append(rmse)

print("\n===== K-FOLD RESULTS =====")
print(f"Mean MAE  : {np.mean(mae_scores):.4f}")
print(f"Mean RMSE : {np.mean(rmse_scores):.4f}")
print(f"Std MAE   : {np.std(mae_scores):.4f}")
print(f"Std RMSE  : {np.std(rmse_scores):.4f}")