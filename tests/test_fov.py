# tests/test_fov_dataset.py

import os, sys
import torch
import pandas as pd

# allow imports from project root
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.data_loader import getData
from src.datasets import FoVSequenceDataset


def main():
   
    user_input = input("Enter user IDs to load: ").strip()

    # Splitting IDs by spaces
    user_ids = [u for u in user_input.split() if u != ""]

     # Load and concatenate users
    dfs = []
    for uid in user_ids:
        df = getData(uid, "avtrack360")
        print(f"User {uid}: Loaded {len(df)} rows")
        dfs.append(df)

    df_all = pd.concat(dfs, ignore_index=True)
    print(f"Total users loaded: {len(dfs)}")
    print(f"User IDs: {user_ids}")
    print(f"Total combined rows: {len(df_all)}\n")

    # set lengths in steps
    context_len = 15
    prefetch_h = 15
    deadline_h = 3

    feature_cols = ["yaw_rad", "pitch_rad"]

    dataset = FoVSequenceDataset(
        df=df_all,
        context_len=context_len,
        prefetch_horizon=prefetch_h,
        deadline_horizon=deadline_h,
        feature_cols=feature_cols,
    )

    print("Dataset length:", len(dataset))

    # single sample
    X, y_pf, y_nd = dataset[0]
    print("X shape:", X.shape)
    print("y_prefetch shape:", y_pf.shape)
    print("y_deadline shape:", y_nd.shape)

    # dataloader batch
    loader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=True)
    Xb, ypfb, yndb = next(iter(loader))

    print("Batch X:", Xb.shape)
    print("Batch y_prefetch:", ypfb.shape)
    print("Batch y_deadline:", yndb.shape)


if __name__ == "__main__":
    main()
