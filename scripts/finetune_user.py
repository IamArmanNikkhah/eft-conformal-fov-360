# scripts/finetune_user.py
# 
# to run-> 
#   python -m scripts.finetune_user --user_id 10

"""
Task 2: User Fine-Tuner

Arguments:
    user_id (str):
        Note: user_id is stored as a string in the pooled parquet file.

    data_path (str):
        (path to the pooled training parquet )
        has to contain columns: user_id, video_id, timestamp, yaw_rad, pitch_rad.

    base_weights (str):
        (pooled_model.pth)
        produced in Task 1 / Issue #12.

    epochs (int):
        
    batch_size (int):

    context_len (int):

    prefetch_horizon (int):

    deadline_horizon (int):

Outputs:

    Personalized LoRA weights for user saved
    models/user_<ID>_adapter.pt
    
"""

import argparse
import os
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_

from src.datasets import FoVSequenceDataset
from src.model import PooledFoVTransformer
from src.peft_wrapper import LoRA_Adapter


def geodesic_loss_rad(pred_rad, target_rad):
    pred = torch.nan_to_num(pred_rad, nan=0.0, posinf=0.0, neginf=0.0)
    targ = torch.nan_to_num(target_rad, nan=0.0, posinf=0.0, neginf=0.0)

    pi = torch.pi

    yaw_pred = torch.clamp(pred[:, 0], -pi, pi)
    pitch_pred = torch.clamp(pred[:, 1], -pi / 2, pi / 2)

    yaw_true = torch.clamp(targ[:, 0], -pi, pi)
    pitch_true = torch.clamp(targ[:, 1], -pi / 2, pi / 2)

    x1 = torch.cos(yaw_pred) * torch.cos(pitch_pred)
    y1 = torch.sin(yaw_pred) * torch.cos(pitch_pred)
    z1 = torch.sin(pitch_pred)

    x2 = torch.cos(yaw_true) * torch.cos(pitch_true)
    y2 = torch.sin(yaw_true) * torch.cos(pitch_true)
    z2 = torch.sin(pitch_true)

    cos_angle = x1 * x2 + y1 * y2 + z1 * z2
    cos_angle = torch.nan_to_num(cos_angle, nan=1.0)
    cos_angle = torch.clamp(cos_angle, -1.0, 1.0)

    eps = 1e-6
    cos_angle = torch.clamp(cos_angle, -1.0 + eps, 1.0 - eps)
    angle = torch.acos(cos_angle)
    angle = torch.nan_to_num(angle, nan=0.0, posinf=0.0, neginf=0.0)

    return angle.mean()


def parse_args():
    parser = argparse.ArgumentParser(description="User Fine-Tuning Script (Task 2)")
    parser.add_argument("--user_id", type=str, required=True)
    parser.add_argument("--data_path", type=str, default="data/avtrack360_train.parquet")
    parser.add_argument("--base_weights", type=str, default="models/pooled_model.pth")
    parser.add_argument("--save_dir", type=str, default="models")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lora_rank", type=int, default=8)
    parser.add_argument("--grad_clip", type=float, default=0.5)
    parser.add_argument("--context_len", type=int, default=15)
    parser.add_argument("--prefetch_horizon", type=int, default=15)
    parser.add_argument("--deadline_horizon", type=int, default=3)
    parser.add_argument("--train_frac", type=float, default=0.8)

    return parser.parse_args()

def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\n[INFO] Loading base model weights from: {args.base_weights}")

    base_model = PooledFoVTransformer(
        input_dim=2,
        d_model=256,
        n_heads=4,
        dim_feedforward=512,
        dropout=0.1,
        max_seq_len=args.context_len,
    ).to(device)

    sd = torch.load(args.base_weights, map_location=device)
    if isinstance(sd, dict) and "state_dict" in sd and isinstance(sd["state_dict"], dict):
        sd = sd["state_dict"]
    base_model.load_state_dict(sd, strict=True)

    # wrap base model with LoRA adapter
    model = LoRA_Adapter(base_model, rank=args.lora_rank).to(device)

    # Optimizer trains ONLY LoRA parameters
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    print(f"[INFO] Trainable LoRA parameters: {len(trainable_params)}")

    optimizer = torch.optim.Adam(trainable_params, lr=args.lr)

    print(f"[INFO] Loading user {args.user_id} from {args.data_path}")

    df = pd.read_parquet(args.data_path) if args.data_path.endswith(".parquet") else pd.read_csv(args.data_path)

    # normalize user_id type
    df["user_id"] = df["user_id"].astype(str)
    df_user = df[df["user_id"] == str(args.user_id)].copy()

    if df_user.empty:
        print("[ERROR] No rows matched that user_id.")
        raise ValueError(f"No data found for user {args.user_id}")

    # keep chronological order for proper sequence windows
    if "video_id" in df_user.columns and "timestamp" in df_user.columns:
        df_user = df_user.sort_values(["video_id", "timestamp"]).reset_index(drop=True)
    elif "timestamp" in df_user.columns:
        df_user = df_user.sort_values(["timestamp"]).reset_index(drop=True)
    else:
        df_user = df_user.reset_index(drop=True)

    n_total = len(df_user)

    # train split
    n_train = int(args.train_frac * n_total)
    n_train = max(1, n_train)
    if n_total > 1:
        n_train = min(n_train, n_total - 1)

    df_train = df_user.iloc[:n_train].reset_index(drop=True)
    df_test = df_user.iloc[n_train:].reset_index(drop=True)

    print(f"[INFO] User samples: total={n_total}, train={len(df_train)}, test={len(df_test)}")

    #  creating datasets 
    train_ds = FoVSequenceDataset(
        df=df_train,
        context_len=args.context_len,
        prefetch_horizon=args.prefetch_horizon,
        deadline_horizon=args.deadline_horizon,
        feature_cols=["yaw_rad", "pitch_rad"],
    )

    test_ds = FoVSequenceDataset(
        df=df_test,
        context_len=args.context_len,
        prefetch_horizon=args.prefetch_horizon,
        deadline_horizon=args.deadline_horizon,
        feature_cols=["yaw_rad", "pitch_rad"],
    )

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    print(f"[INFO] Train windows={len(train_ds)}, Test windows={len(test_ds)}")

    # training loop
    print(f"\n[INFO] Starting fine-tuning for user {args.user_id}")

    # save best adapter so worse iterations don't overwrite it
    best_loss = float("inf")
    best_adapter_state = None

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        n_batches = 0

        for X, y_pref, y_dead in train_loader:
            X, y_pref, y_dead = X.to(device), y_pref.to(device), y_dead.to(device)
            optimizer.zero_grad(set_to_none=True)

            pred_pref, pred_dead = model(X)
            loss = geodesic_loss_rad(pred_pref, y_pref) + geodesic_loss_rad(pred_dead, y_dead)

            loss.backward()

            # prevent LoRA blowing up
            if args.grad_clip and args.grad_clip > 0:
                clip_grad_norm_(trainable_params, max_norm=args.grad_clip)

            optimizer.step()
            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / max(n_batches, 1)
        print(f"[EPOCH {epoch}] train_loss={avg_loss:.4f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            best_adapter_state = {k: v.detach().cpu().clone()
                                  for k, v in model.state_dict().items()
                                  if "lora" in k.lower()}

    os.makedirs(args.save_dir, exist_ok=True)

    out_path = os.path.join(args.save_dir, f"user_{args.user_id}_adapter.pt")
    torch.save(best_adapter_state, out_path)
    print(f"\n[SAVED] Best adapter (train_loss={best_loss:.4f}) → {out_path}\n")


if __name__ == "__main__":
    main()
