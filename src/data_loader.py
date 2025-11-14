import pandas as pd
import os
import json

#Function takes in user_id and dataset_name
def getData(user_id, dataset_name: str) ->  pd.DataFrame:
    """
    Fetches data for a given user from the specified dataset.

    Parameters:
    user_id : The ID of the user whose data is to be fetched.
    dataset_name : The name of the dataset from which to fetch data.
    """
    #Datasets are locally found in the ../data repository
    data_path = os.path.join(os.path.dirname(__file__), "..", "data")
    if dataset_name == "avtrack360":
        user_json_path = os.path.join(data_path, "AVTrack360_dataset", "Head_rotation", f"{user_id}.json")
        #does not exist, return empty dataframe
        if not os.path.exists(user_json_path):
            print("Path not found: ", user_json_path)
            return pd.DataFrame()
        with open(user_json_path, "r") as f:
            data = json.load(f)
        rows = []
        for entry in data["data"]:
            for timeframe in entry["pitch_yaw_roll_data_hmd"]:
                # Append user_id, video, timestamp, yaw, and pitch to rows
                rows.append([user_id, timeframe["sec"], timeframe["yaw"], timeframe["pitch"]])
        print("Loaded data for user:", user_id)
        return pd.DataFrame(rows, columns=["user_id", "timestamp", "yaw", "pitch"])
    else:
        return pd.DataFrame()  # Placeholder for Deep360Pilot loading logic
    
if __name__ == "__main__":
    df = getData("2", "avtrack360")
    print(df.head())