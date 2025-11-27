import pandas as pd

df = pd.read_csv(r"C:\Users\gupta\Downloads\javelin_yolo_project\data\features.csv")
print("FEATURE FILE VIDEO NAMES:")
print(df["video_name"].tolist())
