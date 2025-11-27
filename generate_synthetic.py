# generate_synthetic.py
import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

BASE_DIR = r"C:\Users\gupta\Downloads\javelin_yolo_project"
FEATURE_FILE = os.path.join(BASE_DIR, "features", "extracted_features.csv")
LABEL_FILE = os.path.join(BASE_DIR, "real_distances.csv")
OUT_DIR = os.path.join(BASE_DIR, "synthetic")
OUT_FILE = os.path.join(OUT_DIR, "synthetic_features.csv")

os.makedirs(OUT_DIR, exist_ok=True)

# Parameters
N_SYNTH = 1000        # number of synthetic samples to create
RANDOM_SEED = 42

# Load real labeled features (only the labeled ones)
feat_df = pd.read_csv(FEATURE_FILE)
lab_df = pd.read_csv(LABEL_FILE)
df = pd.merge(feat_df, lab_df, on="video_name", how="inner")

if len(df) < 5:
    raise SystemExit("Need at least a few real labelled samples to derive stats. Found: {}".format(len(df)))

# Use stats from real data
feature_columns = ["max_wrist_speed", "elbow_angle", "release_height",
                   "wrist_horizontal_range", "wrist_vertical_range", "fps"]

X_real = df[feature_columns].values
y_real = df["real_distance"].values

# Fit simple linear regressor to estimate mapping (for label generation)
reg = LinearRegression()
reg.fit(X_real, y_real)

coef = reg.coef_
intercept = reg.intercept_

rng = np.random.RandomState(RANDOM_SEED)

# compute per-feature distributions
means = X_real.mean(axis=0)
stds = X_real.std(axis=0) + 1e-6
mins = X_real.min(axis=0)
maxs = X_real.max(axis=0)

synth_rows = []
for i in range(N_SYNTH):
    # sample each feature from a normal distribution around real mean,
    # with extra variability
    sample = rng.normal(loc=means, scale=stds * rng.uniform(0.8, 1.5, size=stds.shape))
    # Add some clipping to keep within reasonable bounds
    sample = np.maximum(sample, mins - 0.3 * np.abs(mins))
    sample = np.minimum(sample, maxs + 0.3 * np.abs(maxs))

    # small structured transformations to mimic real movement variation
    # time-warp style: randomly scale horizontal/vertical ranges slightly
    sample[3] *= rng.uniform(0.8, 1.2)  # wrist_horizontal_range
    sample[4] *= rng.uniform(0.8, 1.2)  # wrist_vertical_range

    # fps realistic integer
    sample[5] = int(round(rng.choice([24, 25, 30, 50, 60, int(means[5])])))

    # Generate synthetic distance using linear model + noise
    base_dist = float(np.dot(coef, sample) + intercept)
    noise = rng.normal(0, max(0.03 * base_dist, 0.5))  # relative noise + min noise
    synth_dist = base_dist + noise

    # clip distance to physically reasonable range (you used 65-85 earlier)
    synth_dist = float(np.clip(synth_dist, 50.0, 110.0))

    row = dict(zip(feature_columns, sample))
    row["real_distance"] = synth_dist
    # optional meta
    row["source"] = "synthetic"
    row["video_name"] = f"synth_{i:05d}.npy"
    synth_rows.append(row)

synth_df = pd.DataFrame(synth_rows)
synth_df.to_csv(OUT_FILE, index=False)
print("Saved synthetic features to:", OUT_FILE)
print("Synthetic samples:", len(synth_df))
