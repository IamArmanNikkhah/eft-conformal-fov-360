import math
import numpy as np
import torch

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
    '''
    Week 1 geodesic distance 
    meant to take in 2 vectors as np arrays
    returns distance in degrees
    '''
    cos_angle = np.dot(vec1, vec2)
    cos_angle = max(-1.0, min(1.0, float(cos_angle)))
    angle_r = math.acos(cos_angle)
    return math.degrees(angle_r)

''' Arman's geodesic distance for reference, it has shape differences compared to the current model, so I just made a different one
def geodesic_distance_radians(y_pred, y_true):
    """
    Code given by Arman
    Calculates the angular distance between two batches of (yaw, pitch) sequences.
    Inputs are Tensors in RADIANS.
    Shape: [Batch, Seq_Len, 2]
    """
    # 1. Unpack yaw (lon) and pitch (lat)
    # Assuming the last dimension is [yaw, pitch]
    yaw_pred, pitch_pred = y_pred[..., 0], y_pred[..., 1]
    yaw_true, pitch_true = y_true[..., 0], y_true[..., 1]

    # 2. Spherical Law of Cosines (Efficient for Tensors)
    # Formula: acos( sin(lat1)*sin(lat2) + cos(lat1)*cos(lat2)*cos(lon2-lon1) )
    
    cos_d = (torch.sin(pitch_pred) * torch.sin(pitch_true)) + \
            (torch.cos(pitch_pred) * torch.cos(pitch_true) * torch.cos(yaw_pred - yaw_true))

    # ⚠️ CRITICAL: Floating point math can sometimes result in 1.0000001
    # arccos(1.0000001) = NaN (Crash). We must clamp the value.
    cos_d = torch.clamp(cos_d, -1.0, 1.0)

    # 3. Calculate distance
    distance_radians = torch.acos(cos_d)
    
    return distance_radians
'''

def geodesic_distance_radians(y_pred, y_true):
    """
    Based on the code given by Arman with minor change to match current shape.
    His explanation of it is above.
    Calculates the angular distance between two batches of (yaw, pitch) sequences.
    Inputs are Tensors in RADIANS.
    Shape: [Batch, 2]
    """
    yaw_pred, pitch_pred = y_pred[:, 0], y_pred[:, 1]
    yaw_true, pitch_true = y_true[:, 0], y_true[:, 1]
    cos_d = (torch.sin(pitch_pred) * torch.sin(pitch_true)) + (torch.cos(pitch_pred) * torch.cos(pitch_true) * torch.cos(yaw_pred - yaw_true))
    cos_d = torch.clamp(cos_d, -1.0, 1.0)
    distance_radians = torch.acos(cos_d)
    return distance_radians

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