# scripts/train_pooled.py

import argparse
import os
import random

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.datasets import FoVSequenceDataset
from src.model import PooledFoVTransformer


def parse_args():
    parser = argparse.ArgumentParser(description="Train pooled FoV Transformer (Week 2 baseline)")
    parser.add_argument(
        "--data_path",
        type=str,
        default="data/avtrack360_train.parquet",
        help="Path to pooled training DataFrame (parquet/csv).",
    )
    parser.add_argument(
        "--val_data_path",
        type=str,
        default="data/avtrack360_val.parquet",
        help="Path to validation DataFrame.",
    )
    parser.add_argument(
        "--context_len",
        type=int,
        default=90,
        help="Number of past steps used as context.",
    )
    parser.add_argument(
        "--prefetch_horizon",
        type=int,
        default=45,
        help="Future offset (in steps) for prefetch horizon.",
    )
    parser.add_argument(
        "--deadline_horizon",
        type=int,
        default=9,
        help="Future offset (in steps) for near-deadline horizon.",
    )
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_path", type=str, default="models/pooled_model.pth")
    return parser.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def geodesic_loss_rad(pred_rad: torch.Tensor, target_rad: torch.Tensor) -> torch.Tensor:
    """
    Batch geodesic distance between predicted and target FoV directions.

    pred_rad, target_rad: [B, 2] yaw/pitch in RADIANS.
    Returns: scalar tensor = mean angular distance in radians.
    """
    yaw1, pitch1 = pred_rad[:, 0], pred_rad[:, 1]
    yaw2, pitch2 = target_rad[:, 0], target_rad[:, 1]

    x1 = torch.cos(yaw1) * torch.cos(pitch1)
    y1 = torch.sin(yaw1) * torch.cos(pitch1)
    z1 = torch.sin(pitch1)

    x2 = torch.cos(yaw2) * torch.cos(pitch2)
    y2 = torch.sin(yaw2) * torch.cos(pitch2)
    z2 = torch.sin(pitch2)

    cos_angle = x1 * x2 + y1 * y2 + z1 * z2
    cos_angle = torch.clamp(cos_angle, -1.0, 1.0)
    angle = torch.acos(cos_angle)  # radians
    return angle.mean()


def load_pooled_df(path: str):
    """
    Load the pooled DataFrame for training/validation.

    The DataFrame MUST contain at least:
        user_id, video_id, timestamp, yaw_rad, pitch_rad
    """
    import pandas as pd

    print(f"[INFO] Loading DataFrame from: {path}")
    if path.endswith(".parquet"):
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path)

    print(f"[INFO] Loaded df with shape: {df.shape}")
    return df


def build_dataloaders(args):
    df_train = load_pooled_df(args.data_path)
    df_val = load_pooled_df(args.val_data_path)

    feature_cols = ["yaw_rad", "pitch_rad"]
    target_cols = ("yaw_rad", "pitch_rad")

    print("[INFO] Building FoVSequenceDataset for TRAIN...")
    train_ds = FoVSequenceDataset(
        df=df_train,
        context_len=args.context_len,
        prefetch_horizon=args.prefetch_horizon,
        deadline_horizon=args.deadline_horizon,
        feature_cols=feature_cols,
        target_cols=target_cols,
    )
    print(f"[INFO] Train dataset size: {len(train_ds)} samples")

    print("[INFO] Building FoVSequenceDataset for VAL...")
    val_ds = FoVSequenceDataset(
        df=df_val,
        context_len=args.context_len,
        prefetch_horizon=args.prefetch_horizon,
        deadline_horizon=args.deadline_horizon,
        feature_cols=feature_cols,
        target_cols=target_cols,
    )
    print(f"[INFO] Val dataset size: {len(val_ds)} samples")

    # Use num_workers=0 for safety on Windows
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )
    return train_loader, val_loader


def build_model_and_optim(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    model = PooledFoVTransformer(
        input_dim=2,      # yaw_rad, pitch_rad
        d_model=256,
        n_heads=4,
        dim_feedforward=512,
        dropout=0.1,
        max_seq_len=args.context_len,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    print("[INFO] Model and optimizer initialized.")
    return model, optimizer, device


def train_one_epoch(model, optimizer, train_loader, device, epoch, total_epochs):
    model.train()
    total_loss = 0.0
    n_samples = 0

    print(f"[INFO] Starting epoch {epoch}/{total_epochs} (train)...")
    for batch_idx, (X, y_pref, y_dead) in enumerate(train_loader):
        X = X.to(device, dtype=torch.float32)
        y_pref = y_pref.to(device, dtype=torch.float32)
        y_dead = y_dead.to(device, dtype=torch.float32)

        optimizer.zero_grad()
        pred_pref, pred_dead = model(X)

        loss_pref = geodesic_loss_rad(pred_pref, y_pref)
        loss_dead = geodesic_loss_rad(pred_dead, y_dead)
        loss = loss_pref + loss_dead

        loss.backward()
        optimizer.step()

        batch_size = X.size(0)
        total_loss += loss.item() * batch_size
        n_samples += batch_size

        # Progress print every 100 batches
        if (batch_idx + 1) % 100 == 0:
            avg_loss = total_loss / max(n_samples, 1)
            print(f"[INFO] Epoch {epoch} - Batch {batch_idx + 1}, running avg loss={avg_loss:.4f}")

    return total_loss / max(n_samples, 1)


@torch.no_grad()
def evaluate(model, val_loader, device, epoch, total_epochs):
    model.eval()
    total_loss = 0.0
    n_samples = 0

    print(f"[INFO] Starting epoch {epoch}/{total_epochs} (val)...")
    for X, y_pref, y_dead in val_loader:
        X = X.to(device, dtype=torch.float32)
        y_pref = y_pref.to(device, dtype=torch.float32)
        y_dead = y_dead.to(device, dtype=torch.float32)

        pred_pref, pred_dead = model(X)

        loss_pref = geodesic_loss_rad(pred_pref, y_pref)
        loss_dead = geodesic_loss_rad(pred_dead, y_dead)
        loss = loss_pref + loss_dead

        batch_size = X.size(0)
        total_loss += loss.item() * batch_size
        n_samples += batch_size

    return total_loss / max(n_samples, 1)


def save_weights(model, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)
    print(f"[INFO] Saved pooled model weights to {path}")


def main():
    args = parse_args()
    print("[INFO] Parsed args:", args)
    set_seed(args.seed)

    train_loader, val_loader = build_dataloaders(args)
    model, optimizer, device = build_model_and_optim(args)

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, optimizer, train_loader, device, epoch, args.epochs)
        val_loss = evaluate(model, val_loader, device, epoch, args.epochs)

        print(
            f"[RESULT] Epoch {epoch}/{args.epochs} "
            f"train_loss={train_loss:.4f} val_loss={val_loss:.4f}"
        )

    save_weights(model, args.save_path)


if __name__ == "__main__":
    main()
