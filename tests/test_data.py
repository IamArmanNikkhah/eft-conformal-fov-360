# run_specific_user.py

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.data_loader import getData


DATASET_NAME = "avtrack360"


def list_all_users():
    """
    Returns a list of user IDs that exist in the dataset folder.
    Also prints them in condensed preview format:
        [1, 2, 3, 4, 5, ..., 49]
    """

    base_path = os.path.join(os.path.dirname(__file__), "..", "data")
    folder = os.path.join(base_path, "2018-AVTrack360", "Head_rotation")

    if not os.path.exists(folder):
        print("[WARN] Folder not found:", folder)
        return []

    # Collect JSON files
    user_files = [f for f in os.listdir(folder) if f.endswith(".json")]

    # Extract numeric ids only
    user_ids = []
    for f in user_files:
        name = os.path.splitext(f)[0]
        if name.isdigit():
            user_ids.append(name)

    # Sort numerically
    user_ids = sorted(user_ids, key=lambda x: int(x))

    # ---- PREVIEW PRINT ----
    if len(user_ids) <= 5:
        preview = "[" + ", ".join(user_ids) + "]"
    else:
        first_five = ", ".join(user_ids[:5])
        preview = f"[{first_five}, ..., {user_ids[-1]}]"

    print(f"Found {len(user_ids)} users: {preview}")

    return user_ids



def run_specific_user():
    print("Do you want to see ALL users’ datasets (first 10 + last 10 rows)?")
    ans = input("Type 'Y' or 'N': ").strip().lower()

    # ---------------------------------------------------------
    # OPTION 1 — YES → show all users preview
    # ---------------------------------------------------------
    if ans == "y":

        user_ids = list_all_users()

        if not user_ids:
            print("No users found in dataset.")
            return

        print("\nShowing preview (first 10 + last 10 rows) for each user:\n")

        for uid in user_ids:
            print("\n" + "=" * 80)
            print(f"USER: {uid} | DATASET: {DATASET_NAME}")
            print("=" * 80)

            df = getData(uid, DATASET_NAME)

            if df.empty:
                print("No data for this user.")
                continue

            print(f"Total rows: {len(df)}")
            print(f"Total columns: {df.shape[1]}")
            print(f"Columns: {list(df.columns)}")

            log_cols = [
                c for c in ["timestamp", "yaw", "pitch", "yaw_rad", "pitch_rad", "video_id"]
                if c in df.columns
            ]

            print("\nFIRST 10 rows:")
            print(df[log_cols].head(10))

            print("\nLAST 10 rows:")
            print(df[log_cols].tail(10))

            print("-" * 80)

        print("\nDone showing previews for all users.")
        return

    # ---------------------------------------------------------
    # OPTION 2 — NO → ask for specific user, show full dataset
    # ---------------------------------------------------------
    elif ans == "n":
        user_id = input("\nEnter the user ID you want to inspect fully: ").strip()

        print("\nLoading data...\n")
        df = getData(user_id, DATASET_NAME)

        if df.empty:
            print("No data found for this user.")
            return

        print(f"User: {user_id}")
        print(f"Dataset: {DATASET_NAME}")
        print(f"Total rows: {len(df)}")
        print(f"Total columns: {df.shape[1]}")
        print(f"Columns: {list(df.columns)}\n")

        print("----- FULL DATASET -----")
        print(df.to_string())
        print("------------------------")

    else:
        print("Invalid input. Please type 'Y' or 'N'.")




if __name__ == "__main__":
    run_specific_user()
