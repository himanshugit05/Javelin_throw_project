import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO
import os
from tqdm import tqdm

BASE_DIR = r"C:\Users\gupta\Downloads\javelin_yolo_project"
VIDEO_FOLDER = os.path.join(BASE_DIR, "videos")
FEATURE_SAVE_PATH = os.path.join(BASE_DIR, "features", "extracted_features.csv")

os.makedirs(os.path.dirname(FEATURE_SAVE_PATH), exist_ok=True)

# Load YOLO pose model
model = YOLO("yolov8n-pose.pt")

# Keypoint indices
SHOULDER = 5
ELBOW = 7
WRIST = 9

def compute_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba = a - b
    bc = c - b
    cosine = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    return np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))

def extract_features(video_path):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30

    shoulder_track = []
    elbow_track = []
    wrist_track = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        result = model(frame, verbose=False)
        if len(result[0].keypoints.xy) == 0:
            continue

        kp = result[0].keypoints.xy[0].cpu().numpy()

        shoulder_track.append(kp[SHOULDER])
        elbow_track.append(kp[ELBOW])
        wrist_track.append(kp[WRIST])

    cap.release()

    if len(wrist_track) < 5:
        return None

    shoulder_track = np.array(shoulder_track)
    elbow_track = np.array(elbow_track)
    wrist_track = np.array(wrist_track)

    # SPEED
    wrist_speed = np.linalg.norm(np.diff(wrist_track, axis=0), axis=1)
    max_wrist_speed = np.max(wrist_speed) * fps

    # ELBOW ANGLE (final frame)
    elbow_angle = compute_angle(
        shoulder_track[-1],
        elbow_track[-1],
        wrist_track[-1]
    )

    # RELEASE HEIGHT
    height_diff = abs(shoulder_track[-1][1] - wrist_track[-1][1])

    # MOVEMENT RANGE
    horizontal_range = wrist_track[:, 0].max() - wrist_track[:, 0].min()
    vertical_range = wrist_track[:, 1].max() - wrist_track[:, 1].min()

    return {
        "max_wrist_speed": max_wrist_speed,
        "elbow_angle": elbow_angle,
        "release_height": height_diff,
        "wrist_horizontal_range": horizontal_range,
        "wrist_vertical_range": vertical_range,
        "fps": fps
    }


def process_all_videos():
    rows = []
    files = [f for f in os.listdir(VIDEO_FOLDER) if f.lower().endswith((".mp4", ".mov", ".avi"))]

    for f in tqdm(files, desc="Extracting features"):
        video_path = os.path.join(VIDEO_FOLDER, f)
        feats = extract_features(video_path)

        if feats:
            feats["video_name"] = f
            rows.append(feats)

    df = pd.DataFrame(rows)
    df.to_csv(FEATURE_SAVE_PATH, index=False)
    print("Saved to:", FEATURE_SAVE_PATH)



if __name__ == "__main__":
    process_all_videos()
