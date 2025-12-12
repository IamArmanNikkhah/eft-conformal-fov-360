import torch
import numpy as np

from src.geometry_utils import geodesic_distance_radians

def calculate_residuals(model, validation_data):
    """
    Calculate residuals for conformal prediction
    
    Returns separate residuals for prefetch and deadline predictions since they
    likely have different uncertainty
    
    Args:
        model: PooledFoVTransformer
        validation_data: DataLoader
    
    Returns:
        tuple: (residuals_prefetch, residuals_deadline) as numpy arrays
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    residuals_prefetch = []
    residuals_deadline = []

    with torch.no_grad():
        for X, y_prefetch, y_deadline in validation_data:
            X = X.to(device, dtype=torch.float32)
            y_prefetch = y_prefetch.to(device, dtype=torch.float32)
            y_deadline = y_deadline.to(device, dtype=torch.float32)

            pred_prefetch, pred_deadline = model(X)

            error_prefetch = geodesic_distance_radians(pred_prefetch, y_prefetch)
            error_deadline = geodesic_distance_radians(pred_deadline, y_deadline)

            residuals_prefetch.append(error_prefetch.cpu().numpy())
            residuals_deadline.append(error_deadline.cpu().numpy())

    return np.concatenate(residuals_prefetch), np.concatenate(residuals_deadline)

def get_quantile_radius(residuals, alpha):
    """
    Calculate the conformal prediction radius at confidence level (1 - alpha)
    
    Args:
        residuals: Array of validation residuals
        alpha: cutoff point, e.g. (1 - cutoff) * 100 = Percentile error
    
    Returns:
        float: Radius in radians that has (1-alpha)% of predictions fall within it
    """
    
    quantile = 1.0 - alpha
    radius = np.quantile(residuals, quantile)
    return float(radius)

def get_prediction_intervals(model, validation_data, alpha):
    """
    helper to get both radii at once. Combines above 2 functions
    
    Args:
        model: Trained model
        validation_data: Validation DataLoader
        alpha: Significance level (default 0.1 for 90% confidence)
    
    Returns:
        dict: {'prefetch_radius': float, 'deadline_radius': float}
    """
    residuals_prefetch, residuals_deadline = calculate_residuals(model, validation_data)
    
    radius_prefetch = get_quantile_radius(residuals_prefetch, alpha)
    radius_deadline = get_quantile_radius(residuals_deadline, alpha)
    
    return {'prefetch_radius': radius_prefetch, 'deadline_radius': radius_deadline}