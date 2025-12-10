# Usage:
#   python -u scripts/overlay_from_sim_csv.py --video_path data/test.mp4 --sim_csv data/sim_user_2_YYYYMMDD_HHMMSS.csv --out_path outputs/overlay_from_sim.mp4 --video_id 5.mp4 --mode both --verbose

import os
import argparse
import numpy as np
import pandas as pd

try:
    import cv2
except ImportError as e:
    raise SystemExit("Missing cv2. Install with: pip install opencv-python") from e


def wrap_yaw_deg(yaw: float) -> float:
    return ((float(yaw) + 180.0) % 360.0) - 180.0


def clamp_pitch_deg(pitch: float) -> float:
    return max(-90.0, min(90.0, float(pitch)))


def yawpitch_deg_to_pixel(yaw_deg: float, pitch_deg: float, W: int, H: int):
    yaw_deg = wrap_yaw_deg(yaw_deg)
    pitch_deg = clamp_pitch_deg(pitch_deg)

    if not np.isfinite(yaw_deg):
        yaw_deg = 0.0
    if not np.isfinite(pitch_deg):
        pitch_deg = 0.0

    x = int((yaw_deg + 180.0) / 360.0 * (W - 1))
    y = int((90.0 - pitch_deg) / 180.0 * (H - 1))
    x = max(0, min(W - 1, x))
    y = max(0, min(H - 1, y))
    return x, y


def estimate_dt_from_ts(ts: np.ndarray) -> float:
    if len(ts) < 3:
        return 1.0 / 60.0
    diffs = np.diff(ts)
    diffs = diffs[(diffs > 1e-6) & (diffs < 1.0)]
    if len(diffs) == 0:
        return 1.0 / 60.0
    return float(np.median(diffs))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--video_path", required=True)
    p.add_argument("--sim_csv", required=True)
    p.add_argument("--out_path", required=True)
    p.add_argument("--video_id", default=None, help="Filter sim rows by video_id (recommended)")
    p.add_argument("--mode", choices=["prefetch", "deadline", "both"], default="both")
    p.add_argument("--every_n_frames", type=int, default=1)
    p.add_argument("--time_offset", type=float, default=0.0, help="Add to video time before matching sim timestamp")
    p.add_argument("--verbose", action="store_true")

    # NEW: how many steps ahead GT should be shown for each horizon
    p.add_argument("--prefetch_steps", type=int, default=15)
    p.add_argument("--deadline_steps", type=int, default=3)

    return p.parse_args()


def main():
    args = parse_args()

    if args.verbose:
        print("[INFO] video_path:", args.video_path)
        print("[INFO] sim_csv:", args.sim_csv)
        print("[INFO] out_path:", args.out_path)
        print("[INFO] mode:", args.mode)

    df = pd.read_csv(args.sim_csv)

    # normalize video_id filter if requested
    if args.video_id is not None and "video_id" in df.columns:
        df["video_id"] = df["video_id"].astype(str).str.strip()
        vid = str(args.video_id).strip()

        # small convenience: allow passing "5" when csv stores "5.mp4"
        if vid not in set(df["video_id"]) and not vid.endswith(".mp4") and (vid + ".mp4") in set(df["video_id"]):
            vid = vid + ".mp4"

        df = df[df["video_id"] == vid].copy()

    if df.empty:
        raise ValueError("No rows left after filtering (check --video_id and sim_csv contents).")

    df = df.sort_values("timestamp").reset_index(drop=True)
    ts = df["timestamp"].astype(float).to_numpy()

    dt = estimate_dt_from_ts(ts)
    if args.verbose:
        print(f"[INFO] estimated dt from sim timestamps: {dt:.6f} sec (~{1.0/dt:.1f} Hz)")

    # Only overlay where predictions exist (buffer warmed)
    pred_cols = []
    if args.mode in ("prefetch", "both"):
        pred_cols += ["prefetch_yaw_deg", "prefetch_pitch_deg"]
    if args.mode in ("deadline", "both"):
        pred_cols += ["deadline_yaw_deg", "deadline_pitch_deg"]

    valid_mask = np.ones(len(df), dtype=bool)
    for c in pred_cols:
        valid_mask &= df[c].notna().to_numpy()

    dfv = df[valid_mask].copy().reset_index(drop=True)
    if dfv.empty:
        raise ValueError("No valid prediction rows found in CSV (all NaNs).")

    ts_v = dfv["timestamp"].astype(float).to_numpy()

    cap = cv2.VideoCapture(args.video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {args.video_path}")

    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    # frame window based on sim timestamps
    start_time = float(ts_v[0])
    end_time = float(ts_v[-1])

    start_frame = max(0, int((start_time - args.time_offset) * fps))
    end_frame = min(total_frames - 1, int((end_time - args.time_offset) * fps))

    if args.verbose:
        print(f"[INFO] video opened: {W}x{H} fps={fps:.3f} frames={total_frames}")
        print(f"[INFO] overlay time window: [{start_time:.3f}, {end_time:.3f}] sec")
        print(f"[INFO] overlay frame window: [{start_frame}, {end_frame}] (~{(end_frame-start_frame)/fps:.2f}s)")

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    os.makedirs(os.path.dirname(args.out_path) or ".", exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(args.out_path, fourcc, fps, (W, H))

    wrote = 0
    frame_idx = start_frame

    # helper: nearest row by timestamp in df (not dfv)
    ts_all = df["timestamp"].astype(float).to_numpy()

    def nearest_row_index_in_all(t: float) -> int:
        k = int(np.searchsorted(ts_all, t, side="left"))
        if k <= 0:
            return 0
        if k >= len(ts_all):
            return len(ts_all) - 1
        if abs(ts_all[k] - t) < abs(ts_all[k - 1] - t):
            return k
        return k - 1

    # helper: nearest row by timestamp in dfv (prediction rows)
    def nearest_row_index_in_valid(t: float) -> int:
        k = int(np.searchsorted(ts_v, t, side="left"))
        if k <= 0:
            return 0
        if k >= len(ts_v):
            return len(ts_v) - 1
        if abs(ts_v[k] - t) < abs(ts_v[k - 1] - t):
            return k
        return k - 1

    while frame_idx <= end_frame:
        ok, frame = cap.read()
        if not ok:
            break

        if (frame_idx - start_frame) % max(1, args.every_n_frames) != 0:
            out.write(frame)
            frame_idx += 1
            continue

        t_video = frame_idx / fps
        t_sim = t_video + args.time_offset

        # row with predictions (needs warmed buffer)
        ridx = nearest_row_index_in_valid(t_sim)
        row = dfv.iloc[ridx]

        # --- GT current (white) ---
        gt_yaw = float(row["yaw_deg"])
        gt_pitch = float(row["pitch_deg"])
        xw, yw = yawpitch_deg_to_pixel(gt_yaw, gt_pitch, W, H)
        cv2.circle(frame, (xw, yw), 10, (255, 255, 255), -1)

        legend = ["white=GT(now)"]

        # --- predicted prefetch (red) ---
        if args.mode in ("prefetch", "both"):
            py = float(row["prefetch_yaw_deg"])
            pp = float(row["prefetch_pitch_deg"])
            xr, yr = yawpitch_deg_to_pixel(py, pp, W, H)
            cv2.circle(frame, (xr, yr), 10, (0, 0, 255), -1)
            legend.append("red=pred_pref")

            # true prefetch GT (magenta)
            t_pf = t_sim + args.prefetch_steps * dt
            gidx_pf = nearest_row_index_in_all(t_pf)
            row_pf = df.iloc[gidx_pf]
            gt_pf_yaw = float(row_pf["yaw_deg"])
            gt_pf_pitch = float(row_pf["pitch_deg"])
            xm, ym = yawpitch_deg_to_pixel(gt_pf_yaw, gt_pf_pitch, W, H)
            cv2.circle(frame, (xm, ym), 10, (255, 0, 255), -1)
            legend.append("magenta=GT_pref")

        # --- predicted deadline (green) ---
        if args.mode in ("deadline", "both"):
            dy = float(row["deadline_yaw_deg"])
            dp = float(row["deadline_pitch_deg"])
            xg, yg = yawpitch_deg_to_pixel(dy, dp, W, H)
            cv2.circle(frame, (xg, yg), 10, (0, 255, 0), -1)
            legend.append("green=pred_dead")

            # true deadline GT (cyan)
            t_dl = t_sim + args.deadline_steps * dt
            gidx_dl = nearest_row_index_in_all(t_dl)
            row_dl = df.iloc[gidx_dl]
            gt_dl_yaw = float(row_dl["yaw_deg"])
            gt_dl_pitch = float(row_dl["pitch_deg"])
            xc, yc = yawpitch_deg_to_pixel(gt_dl_yaw, gt_dl_pitch, W, H)
            cv2.circle(frame, (xc, yc), 10, (255, 255, 0), -1)
            legend.append("cyan=GT_dead")

        txt = f"t_video={t_video:.2f}s  t_sim={t_sim:.2f}s  mode={args.mode}"
        cv2.putText(frame, txt, (30, 55), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, "  ".join(legend), (30, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

        out.write(frame)
        wrote += 1

        if args.verbose and wrote % 100 == 0:
            print(f"[INFO] wrote {wrote} frames... (frame_idx={frame_idx})")

        frame_idx += 1

    cap.release()
    out.release()

    if args.verbose:
        print(f"[DONE] wrote {wrote} frames -> {args.out_path}")
    else:
        print(args.out_path)


if __name__ == "__main__":
    main()
