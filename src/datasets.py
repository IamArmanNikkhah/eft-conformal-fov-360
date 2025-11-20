# src/datasets.py
"""
Arguments:
    df (pd.DataFrame):
        A long-format DataFrame containing rows of yaw/pitch for each
        (user_id, video_id, timestamp).
    context_len (int):
        # of past timesteps used as Transformer input.
    prefetch_horizon (int):
        # of timesteps into the future for long-term prediction.
    deadline_horizon (int):
        # of timesteps into the future for short-term prediction.
    feature_cols (Sequence[str]):
        Column names used as input features (e.g., ["yaw_rad","pitch_rad"]).
    target_cols (Sequence[str]):
        Column names used as prediction targets (default: yaw_rad, pitch_rad).

Outputs (per sample):
    X          : Tensor of shape (context_len, input_dim)
    y_prefetch : Tensor of shape (2,)     # future yaw, pitch at prefetch horizon
    y_deadline : Tensor of shape (2,)     # future yaw, pitch at deadline horizon
"""
import torch
from torch.utils.data import Dataset
import pandas as pd
from typing import Sequence, List, Tuple


class FoVSequenceDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        context_len: int,
        prefetch_horizon: int,
        deadline_horizon: int,
        feature_cols: Sequence[str],
        target_cols: Sequence[str] = ("yaw_rad", "pitch_rad"),
    ):
        super().__init__()
        self.context_len = int(context_len)
        self.prefetch_horizon = int(prefetch_horizon)
        self.deadline_horizon = int(deadline_horizon)
        self.feature_cols = list(feature_cols)
        self.target_cols = list(target_cols)

        self.trajectories: List[torch.Tensor] = []
        self.targets: List[torch.Tensor] = []
        self.index: List[Tuple[int, int]] = []

        df = df.sort_values(["user_id", "video_id", "timestamp"])
        max_h = max(self.prefetch_horizon, self.deadline_horizon)

        for (_, _), group_df in df.groupby(["user_id", "video_id"]):
            feats = torch.tensor(group_df[self.feature_cols].to_numpy(), dtype=torch.float32)
            targs = torch.tensor(group_df[self.target_cols].to_numpy(), dtype=torch.float32)

            T = feats.shape[0]
            if T < self.context_len + max_h:
                continue

            traj_idx = len(self.trajectories)
            self.trajectories.append(feats)
            self.targets.append(targs)

            first_center = self.context_len - 1
            last_center = T - 1 - max_h

            for center_t in range(first_center, last_center + 1):
                self.index.append((traj_idx, center_t))

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int):
        traj_idx, center_t = self.index[idx]
        feats = self.trajectories[traj_idx]
        targs = self.targets[traj_idx]

        start_t = center_t - self.context_len + 1
        end_t = center_t + 1
        X = feats[start_t:end_t, :]

        pf_t = center_t + self.prefetch_horizon
        nd_t = center_t + self.deadline_horizon

        y_prefetch = targs[pf_t]
        y_deadline = targs[nd_t]

        return X, y_prefetch, y_deadline
