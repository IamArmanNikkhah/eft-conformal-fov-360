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
    #Datasets are locally found in the ../data repository
    data_path = os.path.join(os.path.dirname(__file__), "..", "data")
    if dataset_name == "avtrack360":
        # FIXED PATH: matches data/2018-AVTrack360/Head_rotation/<user_id>.json
        user_json_path = os.path.join(
            data_path,
            "2018-AVTrack360",
            "Head_rotation",
            f"{user_id}.json"
        )
        #does not exist, return empty dataframe
        if not os.path.exists(user_json_path):
            print("Path not found: ", user_json_path)
            return pd.DataFrame()
        with open(user_json_path, "r") as f:
            data = json.load(f)
        rows = []
        for entry in data["data"]:
            video_id = entry.get("filename", "")  # <--- minimal addition
            for timeframe in entry["pitch_yaw_roll_data_hmd"]:
                # Append user_id, video, timestamp, yaw, and pitch to rows
                rows.append([user_id, video_id, timeframe["sec"], timeframe["yaw"], timeframe["pitch"]])

        df = pd.DataFrame(rows, columns=["user_id", "video_id", "timestamp", "yaw", "pitch"])

        ## radian equivalent
        df["yaw_rad"] = np.deg2rad(df["yaw"])
        df["pitch_rad"] = np.deg2rad(df["pitch"])

        return df
    else:
        return pd.DataFrame()  # Placeholder
    
