import pytest
import numpy as np
import math
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))
from geometry_utils import yaw_pitch_to_unit_vector, unit_vector_to_yaw_pitch, geodesic_distance

def test_yaw_pitch_to_unit_vector():
    vec = yaw_pitch_to_unit_vector(0, 0)
    expected = np.array([1, 0, 0])
    np.testing.assert_array_almost_equal(vec, expected)

def test_unit_vector_to_yaw_pitch():
    vec = np.array([1, 0, 0])
    yaw, pitch = unit_vector_to_yaw_pitch(vec)
    assert abs(yaw - 0) < 1e-10
    assert abs(pitch - 0) < 1e-10

def test_unit_and_yaw_pitch():
    original_yaw = 45
    original_pitch = 30
    vec = yaw_pitch_to_unit_vector(original_yaw, original_pitch)
    yaw, pitch = unit_vector_to_yaw_pitch(vec)
    assert abs(yaw - original_yaw) < 1e-10
    assert abs(pitch - original_pitch) < 1e-10

def test_geodesic_distance_same_point():
    vec = np.array([1, 0, 0])
    distance = geodesic_distance(vec, vec)
    assert abs(distance - 0) < 1e-10

def test_geodesic_distance_perpendicular():
    vec1 = np.array([1, 0, 0])
    vec2 = np.array([0, 1, 0])
    distance = geodesic_distance(vec1, vec2)
    expected = 90
    assert abs(distance - expected) < 1e-10