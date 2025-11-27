import pandas as pd

df = pd.read_csv(r"C:\Users\gupta\Downloads\javelin_yolo_project\data\video_distances.csv")
print("LABEL FILE VIDEO NAMES:")
print(df["video_name"].tolist())
