# to run from root, "python -m src.simulator". "--context_len <int>" to change window sizes. Baseline currently is 1.5s since data is at 60 fps

import os
import time
import numpy as np
import pandas as pd
import torch

try:
    from src.data_loader import getData
    from src.geometry_utils import fetch_tiles_in_margin
    from src.model import PooledFoVTransformer
except ImportError:
    from data_loader import getData
    from geometry_utils import fetch_tiles_in_margin
    from model import PooledFoVTransformer


class Simulator:
    def __init__(self, user_id: str, dataset_name: str = "avtrack360", model_path: str = "model/pooled_model.pth",
                 context_len: int = 90, margin_degrees: float = 20.0, device: str = "cuda",):
        self.user_id = user_id
        self.dataset_name = dataset_name
        self.context_len = int(context_len)
        self.margin_degrees = float(margin_degrees)
        
        # Expects data: ["user_id", "video_id", "timestamp", "yaw", "pitch"]
        self.data = getData(user_id, dataset_name)
        self.log = []

        self.device = torch.device(
            device if (device == "cuda" and torch.cuda.is_available()) else "cpu"
        )

        if os.path.exists(model_path):
            self.model = PooledFoVTransformer(
                input_dim=2,
                d_model=256,
                n_heads=4,
                dim_feedforward=512,
                dropout=0.1,
                max_seq_len=context_len,
            ).to(self.device)

            state_dict = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
            self.model.eval()
        else:
            raise FileNotFoundError(f"Model file not found: {model_path}")

    def run(self):
        if self.data.empty:
            print(f"No data found for user {self.user_id}.")
            return pd.DataFrame()

        context = []
        self.log = []

        # Iterate through each timestep
        for _, row in self.data.iterrows():

            timestamp = float(row["timestamp"])
            yaw_deg = float(row["yaw"])
            pitch_deg = float(row["pitch"])
            video_id = row["video_id"]

            yaw_rad = float(row["yaw_rad"])
            pitch_rad = float(row["pitch_rad"])

            # Fill and rotate buffer
            context.append([yaw_rad, pitch_rad])
            if len(context) > self.context_len:
                context.pop(0)

            # Default values before prediction
            pf_yaw_deg = None
            pf_pitch_deg = None
            dl_yaw_deg = None
            dl_pitch_deg = None
            tiles_pf = None
            tiles_dl = None

            # Only run if buffer is full
            if len(context) == self.context_len:

                x = torch.tensor(context, dtype=torch.float32, device=self.device)
                x = x.unsqueeze(0)  # (1, T, 2)

                with torch.no_grad():
                    y_pf, y_dl = self.model(x)

                # Convert to degrees for tile fetch
                pf_yaw_deg   = float(np.degrees(y_pf[0, 0].cpu().item()))
                pf_pitch_deg = float(np.degrees(y_pf[0, 1].cpu().item()))
                dl_yaw_deg   = float(np.degrees(y_dl[0, 0].cpu().item()))
                dl_pitch_deg = float(np.degrees(y_dl[0, 1].cpu().item()))

                # Fetch tiles for both horizons
                tiles_pf = fetch_tiles_in_margin(pf_yaw_deg, pf_pitch_deg, self.margin_degrees)
                tiles_dl = fetch_tiles_in_margin(dl_yaw_deg, dl_pitch_deg, self.margin_degrees)

				# Old print log, now prints to CSV to make the data more readable
                # print(
                #     f"video={video_id} t={timestamp:.3f} | GT=({yaw_deg:.1f}, {pitch_deg:.1f}) | prefetch=({pf_yaw_deg:.1f}, {pf_pitch_deg:.1f}) | "
                #     f"deadline=({dl_yaw_deg:.1f}, {dl_pitch_deg:.1f}) | tiles_prefetch={tiles_pf} | tiles_deadline={tiles_dl}"
                # )

            #else:
                # Left here in case someone wants them back in the console
                # print(f"video={video_id} t={timestamp:.3f} | GT=({yaw_deg:.1f}, {pitch_deg:.1f}) | warmup {len(context)}/{self.context_len}")

            # Log ground truth, prefetch, and deadline
            self.log.append({
                "timestamp": timestamp,
                "yaw_deg": yaw_deg,
                "pitch_deg": pitch_deg,
                "video_id": video_id,
                "prefetch_yaw_deg": pf_yaw_deg,
                "prefetch_pitch_deg": pf_pitch_deg,
                "deadline_yaw_deg": dl_yaw_deg,
                "deadline_pitch_deg": dl_pitch_deg,
                "tiles_prefetch": tiles_pf,
                "tiles_deadline": tiles_dl,
            })

        return pd.DataFrame(self.log)

if __name__ == "__main__":
	user_id = input("Enter user ID (e.g., 2): ").strip()
	sim = Simulator(user_id)
	log_df = sim.run()
	# print("\nLog:")
	# print(log_df)

	stamp = time.strftime("%Y%m%d_%H%M%S")
	out_path = f"data/sim_user_{user_id}_{stamp}.csv"
	log_df.to_csv(out_path, index=False)
	print(f"\nSaved simulation output to:\n{out_path}\n")