import math
import numpy as np

def yaw_pitch_to_unit_vector(yaw, pitch):
    yaw_r = math.radians(yaw)
    pitch_r = math.radians(pitch)
    x = math.cos(yaw_r)*math.cos(pitch_r)
    y = math.sin(yaw_r)*math.cos(pitch_r)
    z = math.sin(pitch_r)
    return np.array([x,y,z])

def unit_vector_to_yaw_pitch(vec):
    x,y,z = vec
    yaw_r = math.atan2(y, x)
    pitch_r = math.asin(z)
    yaw = math.degrees(yaw_r)
    pitch = math.degrees(pitch_r)
    return yaw, pitch

def geodesic_distance(vec1, vec2):
    cos_angle = np.dot(vec1, vec2)
    cos_angle = max(-1.0, min(1.0, float(cos_angle)))
    angle_r = math.acos(cos_angle)
    return math.degrees(angle_r)

#currently just returns a list of tuples of row and column index until we get the tiling encoding sorted. Expects degrees
def fetch_tiles_in_margin(yaw, pitch, margin_degrees=20, rows=6, cols=12):
    tile_width = 360.0 / cols
    tile_height = 180.0 / rows
    vec = yaw_pitch_to_unit_vector(yaw, pitch)
    tiles = []

    for row in range(rows):
        for col in range(cols):
            tile_yaw = (col * tile_width + tile_width / 2) - 180
            tile_pitch = (row * tile_height + tile_height  / 2) - 90
            tile_vec = yaw_pitch_to_unit_vector(tile_yaw, tile_pitch)
            distance = geodesic_distance(vec, tile_vec)
            if distance <= margin_degrees:
                tiles.append((row, col))

    return tiles