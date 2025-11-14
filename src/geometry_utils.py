import math
import numpy as np

def yaw_pitch_to_unit_vector(yaw, pitch):
    """
    Convert yaw/pitch angles (in degrees) to a 3D unit vector on the unit sphere.

    Convention:
    - yaw: rotation in the x-y plane, in degrees.
        yaw = 0   -> +x axis
        yaw = 90  -> +y axis
    - pitch: rotation up/down towards the z-axis, in degrees.
        pitch = 0   -> x-y plane
        pitch = 90  -> +z
        pitch = -90 -> -z

    Returns
    -------
    vec : np.ndarray of shape (3,)
        3D unit vector [x, y, z].
    """
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
    cos_angle = max(-1.0, min(1.0, float(cos_angle)))  # clamp to [-1, 1]
    angle_r = math.acos(cos_angle)
    return math.degrees(angle_r)

