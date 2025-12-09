# utils/tiling.py

import os
import sys

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
    tiles.append(tile_id_for_point(center_yaw, center_pitch, n_yaw, n_pitch))

    for tid in range(n_yaw * n_pitch):
        tyaw, tpitch = tile_center_deg(tid, n_yaw, n_pitch)
        tile_vec = yaw_pitch_to_unit_vector(tyaw, tpitch)
        d = geodesic_distance(center_vec, tile_vec)
        if d <= radius and tid not in tiles:
            tiles.append(tid)

    return tiles