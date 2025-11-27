from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
import os

print(">>> K-FOLD VALIDATION STARTED")

BASE_DIR = r"C:\Users\gupta\Downloads\javelin_yolo_project"
FEATURE_FILE = os.path.join(BASE_DIR, "features", "extracted_features.csv")
LABEL_FILE = os.path.join(BASE_DIR, "real_distances.csv")
SYNTH_FILE = os.path.join(BASE_DIR, "synthetic", "synthetic_features.csv")

# Load data
features_df = pd.read_csv(FEATURE_FILE)
labels_df = pd.read_csv(LABEL_FILE)
real_df = pd.merge(features_df, labels_df, on="video_name", how="inner")

# Synthetic optional
use_synth = True
if use_synth and os.path.exists(SYNTH_FILE):
    synth_df = pd.read_csv(SYNTH_FILE)
else:
    synth_df = pd.DataFrame()

feature_cols = ["max_wrist_speed", "elbow_angle", "release_height",
                "wrist_horizontal_range", "wrist_vertical_range", "fps"]

X_real = real_df[feature_cols].values
y_real = real_df["real_distance"].values

kf = KFold(n_splits=5, shuffle=True, random_state=42)

mae_scores = []
rmse_scores = []

print(">>> ENTERING K-FOLD LOOP")

for train_idx, test_idx in kf.split(X_real):

    print("Running fold...")

    X_train_real = X_real[train_idx]
    y_train_real = y_real[train_idx]

    X_test = X_real[test_idx]
    y_test = y_real[test_idx]

    # Add synthetic only to training
    if use_synth and len(synth_df) > 0:
        X_synth = synth_df[feature_cols].values
        y_synth = synth_df["real_distance"].values

        X_train = np.concatenate([X_train_real, X_synth], axis=0)
        y_train = np.concatenate([y_train_real, y_synth], axis=0)
    else:
        X_train = X_train_real
        y_train = y_train_real

    # Scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Model
    model = RandomForestRegressor(n_estimators=300)
    model.fit(X_train_scaled, y_train)

    preds = model.predict(X_test_scaled)

    mae = mean_absolute_error(y_test, preds)
    rmse = mean_squared_error(y_test, preds, squared=False)

    mae_scores.append(mae)
    rmse_scores.append(rmse)

print("\n===== K-FOLD RESULTS =====")
print("Mean MAE  :", np.mean(mae_scores))
print("Mean RMSE :", np.mean(rmse_scores))
print("Std MAE   :", np.std(mae_scores))
print("Std RMSE  :", np.std(rmse_scores))
