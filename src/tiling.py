# src/tiling.py  (or utils/tiling.py — whichever you import)
import os
import sys
import math

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.geometry_utils import yaw_pitch_to_unit_vector, geodesic_distance


def wrap_yaw(yaw):
    return ((yaw + 180.0) % 360.0) - 180.0


def clamp_pitch(pitch):
    return max(-90.0, min(90.0, pitch))


def tile_center_deg(tile_id, n_yaw=8, n_pitch=6):
    tile_w = 360.0 / n_yaw
    tile_h = 180.0 / n_pitch
    row = tile_id // n_yaw
    col = tile_id % n_yaw
    yaw = -180.0 + (col + 0.5) * tile_w
    pitch = -90.0 + (row + 0.5) * tile_h
    return yaw, pitch


def tile_bounds_deg(tile_id, n_yaw=8, n_pitch=6):
    """Returns (yaw_min, yaw_max, pitch_min, pitch_max) in degrees."""
    tile_w = 360.0 / n_yaw
    tile_h = 180.0 / n_pitch
    row = tile_id // n_yaw
    col = tile_id % n_yaw

    yaw_min = -180.0 + col * tile_w
    yaw_max = yaw_min + tile_w
    pitch_min = -90.0 + row * tile_h
    pitch_max = pitch_min + tile_h
    return yaw_min, yaw_max, pitch_min, pitch_max


def tile_id_for_point(yaw, pitch, n_yaw=8, n_pitch=6):
    yaw = wrap_yaw(float(yaw))
    pitch = clamp_pitch(float(pitch))

    tile_w = 360.0 / n_yaw
    tile_h = 180.0 / n_pitch

    col = int((yaw + 180.0) / tile_w)
    row = int((pitch + 90.0) / tile_h)

    col = min(n_yaw - 1, max(0, col))
    row = min(n_pitch - 1, max(0, row))

    return row * n_yaw + col


def get_tiles_in_radius(center_yaw, center_pitch, radius, n_yaw=8, n_pitch=6):
    center_yaw = wrap_yaw(float(center_yaw))
    center_pitch = clamp_pitch(float(center_pitch))
    radius = float(radius)

    center_vec = yaw_pitch_to_unit_vector(center_yaw, center_pitch)

    tiles = []
    # Always include the tile that contains the center point
    tiles.append(tile_id_for_point(center_yaw, center_pitch, n_yaw, n_pitch))

    S = 3          # 3x3 sample points per tile (fast + usually enough)
    EPS = 1e-4     # keep samples just inside bounds (avoid seam/edge weirdness)

    for tid in range(n_yaw * n_pitch):
        if tid in tiles:
            continue

        yaw_min, yaw_max, p_min, p_max = tile_bounds_deg(tid, n_yaw, n_pitch)

        # keep samples inside [min, max)
        yaw_hi = yaw_max - EPS
        p_hi = p_max - EPS
        yaw_lo = yaw_min + EPS
        p_lo = p_min + EPS

        hit = False
        for r in range(S):
            v = (r / (S - 1)) if S > 1 else 0.5
            pitch = p_lo + v * (p_hi - p_lo)

            for c in range(S):
                u = (c / (S - 1)) if S > 1 else 0.5
                yaw = yaw_lo + u * (yaw_hi - yaw_lo)
                yaw = wrap_yaw(yaw)
                pitch = clamp_pitch(pitch)

                tile_vec = yaw_pitch_to_unit_vector(yaw, pitch)
                d = geodesic_distance(center_vec, tile_vec)
                if d <= radius:
                    hit = True
                    break

            if hit:
                break

        if hit:
            tiles.append(tid)

    return tiles


def get_tiles_in_radius_rad(center_yaw_rad, center_pitch_rad, radius_rad, n_yaw=8, n_pitch=6):
    return get_tiles_in_radius(
        math.degrees(center_yaw_rad),
        math.degrees(center_pitch_rad),
        math.degrees(radius_rad),
        n_yaw=n_yaw,
        n_pitch=n_pitch,
    )
