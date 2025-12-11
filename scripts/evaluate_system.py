# scripts/evaluate_system.py
# Run:
#   python -m scripts.evaluate_system --user_id 2

import os, sys, math, argparse
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data_loader import getData
from src.model import PooledFoVTransformer
from src.peft_wrapper import LoRA_Adapter
from src.datasets import FoVSequenceDataset
from src.geometry_utils import geodesic_distance_radians
from src.conformal import get_prediction_intervals
from src.tiling import get_tiles_in_radius_rad
from torch.utils.data import DataLoader

CONTEXT_LEN = 15
PREFETCH_H = 15
DEADLINE_H = 3

TRAIN_FRAC = 0.20      # "reserved for training"
CALIB_FRAC = 0.30      # next chunk used for conformal calibration
N_STEPS = 50           # like the mini-project

# buffer sim
INIT_BUFFER = 10.0
MAX_BUFFER = 60.0
DANGER_ZONE = 5.0

# controller
MIN_ALPHA = 0.01
MAX_ALPHA = 0.15
MIDPOINT = 13.0
STEEPNESS = 0.15

# conformal baseline alpha
ALPHA_BASE = 0.05

# buffer update scaling (keeps buffer from instantly exploding or dying)
COST_SCALE = 1.5  # download_cost = dynamic_radius * COST_SCALE

SEED = 0


def postprocess_yaw_pitch(pred: torch.Tensor) -> torch.Tensor:
    """Wrap yaw to [-pi,pi], clamp pitch to [-pi/2, pi/2]. pred: (B,2) radians."""
    yaw = torch.atan2(torch.sin(pred[:, 0]), torch.cos(pred[:, 0]))
    pitch = torch.clamp(pred[:, 1], -math.pi / 2, math.pi / 2)
    return torch.stack([yaw, pitch], dim=1)


class PostProcessWrapper(nn.Module):
    """Ensures LoRA outputs are wrapped/clamped"""
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x):
        y_pf, y_dl = self.model(x)
        return postprocess_yaw_pitch(y_pf), postprocess_yaw_pitch(y_dl)


def calibrate_uncertainty_prefetch(model, ds, device, alpha=0.05):
    """
    Mini-project style conformal calibration:
      - compute residuals = distance(pred_pref, gt_pref)
      - choose radius using (n+1) style index
    """
    model.eval()
    residuals = []

    with torch.no_grad():
        for i in range(len(ds)):
            X, y_pref, _ = ds[i]
            X = X.unsqueeze(0).to(device, dtype=torch.float32)
            y_pref = y_pref.unsqueeze(0).to(device, dtype=torch.float32)
            y_pref = postprocess_yaw_pitch(y_pref)

            pred_pref, _ = model(X)
            # geodesic_distance_radians expects shape [B,2] tensors
            d = geodesic_distance_radians(pred_pref, y_pref)[0].item()
            residuals.append(float(d))

    residuals.sort()
    n = len(residuals)
    if n == 0:
        return None

    q_val = math.ceil((n + 1) * (1.0 - alpha)) / n
    q_val = min(1.0, max(0.0, q_val))
    cutoff_index = int(q_val * n) - 1
    cutoff_index = min(max(cutoff_index, 0), n - 1)

    return float(residuals[cutoff_index])


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--user_id", type=str, required=True)
    return p.parse_args()


def main():
    args = parse_args()
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1) Load user trace (ALL videos)
    df = getData(args.user_id, "avtrack360")
    if df is None or df.empty:
        raise RuntimeError(f"No data found for user {args.user_id}.")

    df["user_id"] = df["user_id"].astype(str)
    df = df.sort_values(["user_id", "video_id", "timestamp"]).reset_index(drop=True)

    # 2) Build dataset windows (does not mix videos)
    ds_all = FoVSequenceDataset(
        df=df,
        context_len=CONTEXT_LEN,
        prefetch_horizon=PREFETCH_H,
        deadline_horizon=DEADLINE_H,
        feature_cols=["yaw_rad", "pitch_rad"],
    )
    if len(ds_all) < 50:
        raise RuntimeError(f"Not enough windows: {len(ds_all)}. Need more data or smaller horizons.")

    # 3) Split windows: train-reserved / calib / eval
    n = len(ds_all)
    i_train_end = int(TRAIN_FRAC * n)
    i_calib_end = int((TRAIN_FRAC + CALIB_FRAC) * n)

    i_train_end = min(max(i_train_end, 0), n - 1)
    i_calib_end = min(max(i_calib_end, i_train_end + 10), n - 1)

    calib_indices = list(range(i_train_end, i_calib_end))
    eval_indices = list(range(i_calib_end, n))

    if len(calib_indices) < 20:
        raise RuntimeError(f"Calibration set too small: {len(calib_indices)} windows.")
    if len(eval_indices) < 20:
        raise RuntimeError(f"Eval set too small: {len(eval_indices)} windows.")

    # Wrap ds_all with index lists (super simple)
    class IndexView:
        def __init__(self, ds, idxs):
            self.ds = ds
            self.idxs = idxs
        def __len__(self):
            return len(self.idxs)
        def __getitem__(self, i):
            return self.ds[self.idxs[i]]

    calib_ds = IndexView(ds_all, calib_indices)
    eval_ds = IndexView(ds_all, eval_indices)

    print(f"[INFO] windows total={n} calib={len(calib_ds)} eval={len(eval_ds)}")

    # 4) Load pooled model (+ adapter if exists)
    base = PooledFoVTransformer(
        input_dim=2, d_model=256, n_heads=4,
        dim_feedforward=512, dropout=0.1,
        max_seq_len=CONTEXT_LEN,
    ).to(device)

    base_sd = torch.load("models/pooled_model.pth", map_location=device)
    base.load_state_dict(base_sd, strict=False)
    base.eval()

    adapter_path = f"models/user_{args.user_id}_adapter.pt"
    if os.path.exists(adapter_path):
        ad = torch.load(adapter_path, map_location=device)
        rank = int(ad["lora_A"].shape[1])
        model = LoRA_Adapter(base, rank=rank).to(device)
        model.load_state_dict(ad, strict=False)
        model.eval()
        print(f"[INFO] Using adapter: {adapter_path}")
    else:
        model = base
        print("[INFO] No adapter found -> pooled model only")

    model = PostProcessWrapper(model).to(device).eval()

    # 5) Calibrate baseline radius (prefetch)
    # baseline_radius = calibrate_uncertainty_prefetch(model, calib_ds, device, alpha=ALPHA_BASE)
    calib_loader = DataLoader(calib_ds, batch_size=32)
    radii = get_prediction_intervals(model, calib_loader, ALPHA_BASE)
    if len(radii) == 0:
        raise RuntimeError("Calibration failed (no residuals).")
    # average of prefetch and deadline quantile radii from the conformal script
    baseline_radius = sum(radii.values()) / len(radii)

    print(f"[INFO] baseline_radius (alpha={ALPHA_BASE}) = {baseline_radius:.4f} rad "
          f"({math.degrees(baseline_radius):.1f} deg)")

    # 6) Simulate streaming session (similar to mini project)
    bandwidth = np.sin(np.linspace(0, 10, N_STEPS)) + 1.5  # volatile bandwidth
    buffer_level = float(INIT_BUFFER)

    results = {"alpha": [], "radius": [], "hit": [], "buffer": []}

    for t in range(N_STEPS):
        current_bandwidth = float(bandwidth[t])

        # alpha controller
        target_alpha = MIN_ALPHA + (MAX_ALPHA - MIN_ALPHA) / (
            1.0 + np.exp(STEEPNESS * (buffer_level - MIDPOINT))
        )

        # dynamic radius approximation
        dynamic_radius = baseline_radius * (ALPHA_BASE / max(target_alpha, 1e-6))

        # pick random eval sample
        sample_idx = np.random.randint(len(eval_ds))
        X, y_pref, _ = eval_ds[sample_idx]

        X = X.unsqueeze(0).to(device, dtype=torch.float32)
        y_pref = y_pref.unsqueeze(0).to(device, dtype=torch.float32)
        y_pref = postprocess_yaw_pitch(y_pref)

        with torch.no_grad():
            pred_pref, _ = model(X)

        print(get_tiles_in_radius_rad(y_pref[:, 0], y_pref[:, 1], dynamic_radius))

        dist = geodesic_distance_radians(pred_pref, y_pref)[0].item()
        is_hit = 1 if dist <= dynamic_radius else 0

        # buffer update
        download_cost = max(1e-3, dynamic_radius * COST_SCALE)
        buffer_change = (current_bandwidth / download_cost) - 1.0  # -1.0 playback
        buffer_level = max(0.0, min(MAX_BUFFER, buffer_level + buffer_change))

        results["alpha"].append(float(target_alpha))
        results["radius"].append(float(dynamic_radius))
        results["hit"].append(int(is_hit))
        results["buffer"].append(float(buffer_level))

    # 7) Plot (same 3 graphs as mini-project)
    os.makedirs("results", exist_ok=True)
    hits = np.array(results["hit"], dtype=np.int32)
    hit_rate = float(hits.mean() * 100.0)

    fig, ax = plt.subplots(3, 1, figsize=(10, 12), sharex=True)

    # Plot 1: Bandwidth & Buffer
    ax[0].plot(bandwidth, label="Bandwidth", linestyle="--")
    ax[0].plot(results["buffer"], label="Buffer Level", linewidth=2)
    ax[0].axhline(y=DANGER_ZONE, linestyle=":", label="Danger Zone")
    ax[0].set_ylabel("Level")
    ax[0].legend()
    ax[0].set_title("System Status")

    # Plot 2: Radius
    ax[1].plot(results["radius"], label="Prediction Radius")
    ax[1].set_ylabel("Radius Size (rad)")
    ax[1].legend()
    ax[1].set_title("Adaptive Risk Control")

    # Plot 3: Hits
    ax[2].bar(range(N_STEPS), hits, color=["green" if h else "red" for h in hits])
    ax[2].set_ylabel("Hit (1) / Miss (0)")
    ax[2].set_title(f"User Experience (Hit Rate: {hit_rate:.1f}%)")

    out_path = os.path.join("results", f"eval_user_{args.user_id}_simple.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)

    print(f"\n[SAVED] {out_path}")
    print(f"[SUMMARY] steps={N_STEPS} hit_rate={hit_rate:.2f}% "
          f"avg_alpha={np.mean(results['alpha']):.3f} "
          f"avg_radius={np.mean(results['radius']):.3f} rad\n")


if __name__ == "__main__":
    main()
