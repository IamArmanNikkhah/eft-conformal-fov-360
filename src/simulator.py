import time
import pandas as pd
from src.data_loader import getData

class Simulator:
    def __init__(self, user_id, dataset_name: str):
        """
        Initialize the simulator with user trajectory data.
        
        Parameters:
        - user_id: The ID of the user whose data is to be fetched.
        - dataset_name: The name of the dataset from which to fetch data.
        """
        self.user_id = user_id
        self.dataset_name = dataset_name
        self.data = getData(user_id, dataset_name)
        self.previoustimestamp = 0.0
        if self.data.empty:
            print(f"No trajectory data for user {user_id} in dataset {dataset_name}.")
        else:
            print(f"Simulator initialized for user {user_id} with {len(self.data)} timesteps.")
            #This is a Pandas DataFrame with columns: user_id, timestamp, yaw, pitch
            #The first rows correlate to the first video, then the next rows to the next video, etc.

    def run(self):
        """
        Simulate looping through the trajectory timestamps and print the current FoV.
        """
        if self.data.empty:
            print("No trajectory data to simulate.")
            return
        
        print("Starting simulation...")
        for _ , row in self.data.iterrows():
            yaw = row["yaw"]
            pitch = row["pitch"]
            timestamp = row["timestamp"]
            print(f"Timestamp: {timestamp}s | Yaw: {yaw} | Pitch: {pitch}")
            # Simulate waiting until the next timestamp
            # If the video ends and the next video starts, we simply wait for the next video's first timestamp
            if timestamp - self.previoustimestamp > 0.0:
                time.sleep(timestamp - self.previoustimestamp)
            else: 
                time.sleep(timestamp)
            self.previoustimestamp = timestamp

if __name__ == "__main__":
    sim = Simulator("2", "avtrack360")
    sim.run()
