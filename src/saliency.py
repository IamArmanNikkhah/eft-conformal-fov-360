# src/saliency.py

import numpy as np

def get_saliency_feature(frame_data=None, current_yaw=None, current_pitch=None):
    """
    Returns a dummy saliency feature vector.
    This placeholder ensures the rest of the pipeline has a consistent saliency input.

    Args:
        frame_data: placeholder for frame or content features (unused for now)
        current_yaw: optional current yaw angle (degrees or radians)
        current_pitch: optional current pitch angle (degrees or radians)

    Returns:
        np.ndarray: a small, consistent feature vector (e.g., shape (3,))
    """

    # Simple center-biased dummy: higher near center, otherwise uniform
    if current_yaw is not None and current_pitch is not None:
        bias = np.exp(-0.01 * (current_yaw**2 + current_pitch**2))
    else:
        bias = 0.5  # neutral center bias if not provided

    # Fixed shape dummy vector
    feature = np.array([bias, 1.0, 0.0], dtype=np.float32)
    return feature


if __name__ == "__main__":
    # Example quick tests
    f = get_saliency_feature(current_yaw=7, current_pitch=3)
    print("Dummy saliency feature:", f)
    f = get_saliency_feature()
    print("Dummy saliency feature:", f) 
