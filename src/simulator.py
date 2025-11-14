import pandas as pd
from data_loader import getData


class Simulator:
	def __init__(self, user_id: str, dataset_name: str):
		self.user_id = user_id
		self.dataset_name = dataset_name

		# Expects data: ["user_id", "timestamp", "yaw", "pitch"]
		self.data = getData(user_id, dataset_name)

		if not self.data.empty:
			self.data = self.data.sort_values("timestamp").reset_index(drop=True)

		self.log = []

	def run(self) -> pd.DataFrame:
	 
		if self.data.empty:
			print(f"No trajectory data for user {self.user_id} in {self.dataset_name}.")
			return pd.DataFrame(columns=["timestamp", "yaw", "pitch"])

		self.log = []

		for _, row in self.data.iterrows():
			timestamp = float(row["timestamp"])
			yaw = float(row["yaw"])
			pitch = float(row["pitch"])

			viewport = {
				"timestamp": timestamp,
				"yaw": yaw,
				"pitch": pitch,
			}

			# "Log" the current ground-truth viewport
			self.log.append(viewport)

			print(f"t={timestamp:.3f}s | yaw={yaw:.2f} | pitch={pitch:.2f}")
   
		return pd.DataFrame(self.log)


if __name__ == "__main__":
	user_id = input("Enter user ID (e.g., 2): ").strip()
	sim = Simulator(user_id, "avtrack360")
	log_df = sim.run()
	print("\nLog:")
	print(log_df)