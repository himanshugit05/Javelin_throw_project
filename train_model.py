# train_model.py
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import joblib

BASE_DIR = r"C:\Users\gupta\Downloads\javelin_yolo_project"
FEATURE_FILE = os.path.join(BASE_DIR, "features", "extracted_features.csv")
LABEL_FILE = os.path.join(BASE_DIR, "real_distances.csv")
SYNTH_FILE = os.path.join(BASE_DIR, "synthetic", "synthetic_features.csv")
MODEL_SAVE_PATH = os.path.join(BASE_DIR, "models", "regressor_with_synth.pkl")

os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)

# Toggle synthetic usage
USE_SYNTHETIC = True
TEST_SIZE = 0.25
RANDOM_STATE = 42

# ----------------------
# Load real labeled features
# ----------------------
feat_df = pd.read_csv(FEATURE_FILE)
lab_df = pd.read_csv(LABEL_FILE)
real_df = pd.merge(feat_df, lab_df, on="video_name", how="inner")
print("Real labeled samples:", len(real_df))

# ----------------------
# Optionally load synthetic
# ----------------------
if USE_SYNTHETIC and os.path.exists(SYNTH_FILE):
    synth_df = pd.read_csv(SYNTH_FILE)
    print("Synthetic samples available:", len(synth_df))
    # keep same columns ordering
    train_df = pd.concat([real_df.assign(source="real"), synth_df.assign(source="synthetic")], ignore_index=True)
else:
    train_df = real_df.copy()

print("Total training pool:", len(train_df), "(real + synthetic where used)")

# ----------------------
# Prepare X, y
# ----------------------
feature_columns = ["max_wrist_speed", "elbow_angle", "release_height",
                   "wrist_horizontal_range", "wrist_vertical_range", "fps"]

# If synthetic exists it already has 'real_distance' column
X_all = train_df[feature_columns].values
y_all = train_df["real_distance"].values

# Standardize features using scaler fit on training pool
scaler = StandardScaler()
X_all_scaled = scaler.fit_transform(X_all)

# ----------------------
# IMPORTANT: Split so that real samples are reserved for final test.
# We will keep some real samples for test. Strategy:
# - First, split the real_df into train_real / test_real
# - Combine train_real with synthetic (if any) to form final training set
# - Test on test_real only.
# ----------------------
# Split real into train_real/test_real
if len(real_df) >= 4:
    X_real = real_df[feature_columns].values
    y_real = real_df["real_distance"].values
    Xr_train, Xr_test, yr_train, yr_test = train_test_split(
        X_real, y_real, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    print("Real train:", len(Xr_train), "Real test:", len(Xr_test))
else:
    # fallback: if very few real samples, use all as train (but evaluation will be weak)
    Xr_train = real_df[feature_columns].values
    yr_train = real_df["real_distance"].values
    Xr_test = np.empty((0, len(feature_columns)))
    yr_test = np.empty((0,))

# Build final training set
# Take train_real rows from real_df, and optionally combine with synth_df
train_real_df = real_df.iloc[:len(Xr_train)].copy()  # simplest selection (we used split above)
train_frames = [train_real_df]
if USE_SYNTHETIC and os.path.exists(SYNTH_FILE):
    synth_df = pd.read_csv(SYNTH_FILE)
    # ensure column types align
    train_frames.append(synth_df[feature_columns + ["real_distance"]])

train_pool_df = pd.concat(train_frames, ignore_index=True)
X_train = train_pool_df[feature_columns].values
y_train = train_pool_df["real_distance"].values

# Fit scaler again on X_train (it may be slightly different than earlier)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# For testing, scale Xr_test using this scaler
if len(Xr_test) > 0:
    X_test_scaled = scaler.transform(Xr_test)
else:
    X_test_scaled = np.empty((0, X_train_scaled.shape[1]))

# ----------------------
# Train model
# ----------------------
model = RandomForestRegressor(n_estimators=300, max_depth=12, random_state=RANDOM_STATE)
model.fit(X_train_scaled, y_train)
print("Model trained on:", X_train_scaled.shape[0], "samples (real+synthetic).")

# ----------------------
# Evaluate on real test only
# ----------------------
# ----------------------
# Evaluate on real test only
# ----------------------
if len(X_test_scaled) > 0:
    preds = model.predict(X_test_scaled)
    mae = mean_absolute_error(yr_test, preds)

    # Average actual test value
    avg_real_distance = np.mean(yr_test)

    # Percentage error
    percentage_error = (mae / avg_real_distance) * 100

    print("----------------------------------------------------")
    print("EVALUATION ON REAL TEST SET ({} samples)".format(len(yr_test)))
    print("MAE           : {:.3f} meters".format(mae))
    print("Avg Distance  : {:.3f} meters".format(avg_real_distance))
    print("Percent Error : {:.2f}%".format(percentage_error))
    print("Accuracy      : {:.2f}%".format(100 - percentage_error))
    print("----------------------------------------------------")
else:
    print("No real test samples available to evaluate (too few labeled samples).")

joblib.dump({"model": model, "scaler": scaler, "use_synthetic": USE_SYNTHETIC}, MODEL_SAVE_PATH)
print("Saved model to:", MODEL_SAVE_PATH)
