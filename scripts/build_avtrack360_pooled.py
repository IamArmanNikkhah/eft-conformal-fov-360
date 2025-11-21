# scripts/build_avtrack360_pooled.py

import os
import json
import numpy as np
import pandas as pd


def build_avtrack360_df(head_root: str) -> pd.DataFrame:
    """
    head_root: path to AVTrack360 Head_rotation folder, e.g.
        data/2018-AVTrack360/AVTrack360_dataset/Head_rotation

    Returns a DataFrame with columns:
        user_id, video_id, timestamp, yaw_rad, pitch_rad
    """
    rows = []

    for fname in os.listdir(head_root):
        if not fname.endswith(".json"):
            continue

        user_id = os.path.splitext(fname)[0]  # e.g., "2" from "2.json"
        path = os.path.join(head_root, fname)

        with open(path, "r") as f:
            obj = json.load(f)

        for video_entry in obj.get("data", []):
            video_id = video_entry.get("filename", "")

            for frame in video_entry.get("pitch_yaw_roll_data_hmd", []):
                sec = frame["sec"]
                yaw_deg = frame["yaw"]
                pitch_deg = frame["pitch"]

                rows.append(
                    {
                        "user_id": user_id,
                        "video_id": video_id,
                        "timestamp": sec,
                        "yaw_rad": np.deg2rad(yaw_deg),
                        "pitch_rad": np.deg2rad(pitch_deg),
                    }
                )

    df = pd.DataFrame(rows)
    return df


def main():
    # 👈 adjust this path if your repo structure is different
    head_root = "data/2018-AVTrack360/Head_rotation"

    out_all = "data/avtrack360_pooled.parquet"
    out_train = "data/avtrack360_train.parquet"
    out_val = "data/avtrack360_val.parquet"

    df = build_avtrack360_df(head_root)
    df = df.sort_values(["user_id", "video_id", "timestamp"]).reset_index(drop=True)

    # Simple train/val split by user_id (so validation users are unseen)
    rng = np.random.RandomState(42)
    users = df["user_id"].unique()
    rng.shuffle(users)
    n_train = int(0.8 * len(users))
    train_users = users[:n_train]
    val_users = users[n_train:]

    df_train = df[df["user_id"].isin(train_users)].reset_index(drop=True)
    df_val = df[df["user_id"].isin(val_users)].reset_index(drop=True)

    os.makedirs("data", exist_ok=True)
    df.to_parquet(out_all)
    df_train.to_parquet(out_train)
    df_val.to_parquet(out_val)

    print("Saved:")
    print("  all  :", out_all, f"(rows={len(df)})")
    print("  train:", out_train, f"(rows={len(df_train)})")
    print("  val  :", out_val, f"(rows={len(df_val)})")


if __name__ == "__main__":
    main()
