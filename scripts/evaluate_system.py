# scripts/evaluate_system.py
# Run:
#   python -m scripts.evaluate_system --user_id 2
#
# Outputs:
#   results/eval_user_<id>_prefetch.png
#   results/eval_user_<id>_deadline.png

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

# tiling
N_YAW = 12
N_PITCH = 6

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


def gt_tile_id_from_yawpitch_rad(yaw_rad: float, pitch_rad: float,
                                 n_yaw: int, n_pitch: int) -> int:
    """
    Convert (yaw, pitch) in radians to a tile ID, using the same convention
    as the tiling utilities:
      - yaw in [-180, 180)
      - pitch in [-90, 90]
      - row = 0 at pitch = -90 (bottom), increasing upward.
    """
    yaw_deg = math.degrees(float(yaw_rad))
    pitch_deg = math.degrees(float(pitch_rad))

    # wrap/clamp to valid ranges
    yaw_deg = ((yaw_deg + 180.0) % 360.0) - 180.0
    pitch_deg = max(-90.0, min(90.0, pitch_deg))

    col = int(((yaw_deg + 180.0) / 360.0) * n_yaw)
    col = max(0, min(n_yaw - 1, col))

    row = int(((pitch_deg + 90.0) / 180.0) * n_pitch)
    row = max(0, min(n_pitch - 1, row))

    return row * n_yaw + col


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--user_id", type=str, required=True)
    return p.parse_args()


def run_simulation_for_mode(
    mode: str,
    baseline_radius: float,
    eval_ds,
    device,
    user_id: str,
    bandwidth,
    model: nn.Module,
):

    # Run the  simulation for a single mode ("prefetch" or "deadline"),
 
    assert mode in ("prefetch", "deadline")

    buffer_level = float(INIT_BUFFER)
    results = {"alpha": [], "radius": [], "hit": [], "buffer": [], "tiles": []}

    for t in range(N_STEPS):
        current_bandwidth = float(bandwidth[t])

        # 1) Alpha controller
        target_alpha = MIN_ALPHA + (MAX_ALPHA - MIN_ALPHA) / (
            1.0 + np.exp(STEEPNESS * (buffer_level - MIDPOINT))
        )

        # 2) Dynamic radius: scale baseline_radius by alpha
        dynamic_radius = baseline_radius * (ALPHA_BASE / max(target_alpha, 1e-6))

        # 3) Pick random eval sample
        sample_idx = np.random.randint(len(eval_ds))
        X, y_pref, y_dead = eval_ds[sample_idx]

        X = X.unsqueeze(0).to(device, dtype=torch.float32)
        y_pref = y_pref.unsqueeze(0).to(device, dtype=torch.float32)
        y_dead = y_dead.unsqueeze(0).to(device, dtype=torch.float32)

        # wrap/clamp GT angles
        y_pref = postprocess_yaw_pitch(y_pref)
        y_dead = postprocess_yaw_pitch(y_dead)

        with torch.no_grad():
            pred_pref, pred_dead = model(X)

        if mode == "prefetch":
            pred = pred_pref
            gt = y_pref
        else:
            pred = pred_dead
            gt = y_dead

        # 4) Tiling-based hit:
        #    - tiles fetched = all tiles whose area intersects the prediction circle
        #    - GT tile = tile containing the GT yaw/pitch
        tiles = get_tiles_in_radius_rad(
            pred[0, 0].item(),
            pred[0, 1].item(),
            dynamic_radius,
            n_yaw=N_YAW,
            n_pitch=N_PITCH,
        )
        gt_tid = gt_tile_id_from_yawpitch_rad(
            gt[0, 0].item(),
            gt[0, 1].item(),
            N_YAW,
            N_PITCH,
        )
        is_hit = 1 if gt_tid in tiles else 0

        # (Optional) geodesic distance
        _dist = geodesic_distance_radians(pred, gt)[0].item()

        # 5) Buffer update
        download_cost = max(1e-3, dynamic_radius * COST_SCALE)
        buffer_change = (current_bandwidth / download_cost) - 1.0  # -1 playback
        buffer_level = max(0.0, min(MAX_BUFFER, buffer_level + buffer_change))

        # 6) Record stats
        results["alpha"].append(float(target_alpha))
        results["radius"].append(float(dynamic_radius))
        results["hit"].append(int(is_hit))
        results["buffer"].append(float(buffer_level))
        results["tiles"].append(len(tiles))

    # ----- summarize + plot -----
    os.makedirs("results", exist_ok=True)
    hits = np.array(results["hit"], dtype=np.int32)
    hit_rate = float(hits.mean() * 100.0)
    avg_tiles = float(np.mean(results["tiles"]))
    avg_radius = float(np.mean(results["radius"]))
    avg_alpha = float(np.mean(results["alpha"]))

    fig, ax = plt.subplots(3, 1, figsize=(10, 12), sharex=True)

    # Plot 1: Bandwidth & Buffer
    ax[0].plot(bandwidth, label="Bandwidth", linestyle="--")
    ax[0].plot(results["buffer"], label="Buffer Level", linewidth=2)
    ax[0].axhline(y=DANGER_ZONE, linestyle=":", label="Danger Zone")
    ax[0].set_ylabel("Level")
    ax[0].legend()
    ax[0].set_title(f"System Status ({mode})")

    # Plot 2: Radius
    ax[1].plot(results["radius"], label="Prediction Radius")
    ax[1].set_ylabel("Radius Size (rad)")
    ax[1].legend()
    ax[1].set_title("Adaptive Risk Control")

    # Plot 3: Hits
    ax[2].bar(range(N_STEPS), hits, color=["green" if h else "red" for h in hits])
    ax[2].set_ylabel("Hit (1) / Miss (0)")
    ax[2].set_title(f"User Experience (Hit Rate: {hit_rate:.1f}%)")

    out_path = os.path.join("results", f"eval_user_{user_id}_{mode}.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)

    print(f"\n[SAVED] {out_path}")
    print(
        f"[SUMMARY {mode}] steps={N_STEPS} hit_rate={hit_rate:.2f}% "
        f"avg_alpha={avg_alpha:.3f} avg_radius={avg_radius:.3f} rad "
        f"avg_tiles={avg_tiles:.2f}"
    )


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

    # Wrap ds_all with index lists
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

    # 5) Calibrate baseline radii (prefetch + deadline)
    calib_loader = DataLoader(calib_ds, batch_size=32, shuffle=False)
    radii = get_prediction_intervals(model, calib_loader, ALPHA_BASE)
    if not radii:
        raise RuntimeError("Calibration failed (no residuals).")

    baseline_radius_prefetch = float(radii["prefetch_radius"])
    baseline_radius_deadline = float(radii["deadline_radius"])

    print(
        f"[INFO] prefetch_radius (alpha={ALPHA_BASE}) = "
        f"{baseline_radius_prefetch:.4f} rad ({math.degrees(baseline_radius_prefetch):.1f} deg)"
    )
    print(
        f"[INFO] deadline_radius (alpha={ALPHA_BASE}) = "
        f"{baseline_radius_deadline:.4f} rad ({math.degrees(baseline_radius_deadline):.1f} deg)"
    )

    # 6) Shared synthetic bandwidth signal (same for both modes)
    bandwidth = np.sin(np.linspace(0, 10, N_STEPS)) + 1.5  # volatile bandwidth

    # 7) Run simulations for both modes
    run_simulation_for_mode(
        mode="prefetch",
        baseline_radius=baseline_radius_prefetch,
        eval_ds=eval_ds,
        device=device,
        user_id=args.user_id,
        bandwidth=bandwidth,
        model=model,
    )

    run_simulation_for_mode(
        mode="deadline",
        baseline_radius=baseline_radius_deadline,
        eval_ds=eval_ds,
        device=device,
        user_id=args.user_id,
        bandwidth=bandwidth,
        model=model,
    )


if __name__ == "__main__":
    main()
