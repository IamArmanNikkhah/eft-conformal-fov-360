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
    angle_r = math.acos(cos_angle)
    return math.degrees(angle_r)
