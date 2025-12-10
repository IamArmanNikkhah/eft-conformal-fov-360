import os
import json
import numpy as np
import pandas as pd


CLEANED_DIR = os.path.join(
    os.path.dirname(__file__),
    "..",
    "data",
    "2018-AVTrack360",
    "cleaned"
)


def getData(user_id, dataset_name="avtrack360"):
    """
    Loads the CLEANED JSON dataset produced by utils/data_cleaner.py.
    Returns a DataFrame with:
        user_id, video_id, timestamp, yaw, pitch, roll, yaw_rad, pitch_rad
    """

    if dataset_name != "avtrack360":
        print("[ERROR] Unknown dataset:", dataset_name)
        return pd.DataFrame()

    # Path to cleaned JSON
    cleaned_path = os.path.join(CLEANED_DIR, f"{user_id}.json")

    if not os.path.exists(cleaned_path):
        print(f"[ERROR] Cleaned JSON not found: {cleaned_path}")
        print("Run `python utils/data_cleaner.py` to generate cleaned files.")
        return pd.DataFrame()

    # load the cleaned file
    with open(cleaned_path, "r") as f:
        data = json.load(f)

    rows = []


    for video in data["data"]:
        video_id = video["filename"]
        frames = video["pitch_yaw_roll_data_hmd"]

        for frame in frames:
            rows.append({
                "user_id": int(user_id),
                "video_id": video_id,
                "timestamp": frame["sec"],
                "yaw": frame["yaw"],
                "pitch": frame["pitch"],
                "roll": frame["roll"]
            })

    df = pd.DataFrame(rows)

    # Convert yaw/pitch to radians 
    df["yaw_rad"] = np.deg2rad(df["yaw"])
    df["pitch_rad"] = np.deg2rad(df["pitch"])

    return df
