import pytest
import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import sys
from pathlib import Path

# Add src directory to path
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

from conformal import calculate_residuals, get_quantile_radius, get_prediction_intervals
from model import PooledFoVTransformer


def test_conformal_coverage_95_percent():
    """
    Unit Test: If you ask for α=0.05 (95% confidence),
    your radius should cover exactly 95% of the points.
    pytest -v -s in root shows print statements as well
    """
    # small test dataset
    torch.manual_seed(42)
    num_samples = 100
    seq_len = 15
    
    X = torch.randn(num_samples, seq_len, 2) * 0.5
    y_prefetch = torch.randn(num_samples, 2) * 0.3
    y_deadline = torch.randn(num_samples, 2) * 0.2
    
    dataset = TensorDataset(X, y_prefetch, y_deadline)
    validation_data = DataLoader(dataset, batch_size=32, shuffle=False)
    
    # model
    model = PooledFoVTransformer(input_dim=2, d_model=64, n_heads=2, dim_feedforward=128, dropout=0.1, max_seq_len=50)
    model.eval()
    
    # get residuals
    residuals_pf, residuals_dl = calculate_residuals(model, validation_data)
    
    # get radius of alpha of .05
    alpha = 0.05
    radius_pf = get_quantile_radius(residuals_pf, alpha)
    radius_dl = get_quantile_radius(residuals_dl, alpha)
    
    # check for coverage of 95%
    coverage_pf = np.mean(residuals_pf <= radius_pf)
    coverage_dl = np.mean(residuals_dl <= radius_dl)

    intervals = get_prediction_intervals(model, validation_data, alpha)
    
    assert abs(intervals['prefetch_radius'] - radius_pf) < 1e-6, f"get_prediction_intervals prefetch radius {intervals['prefetch_radius']:.6f} != {radius_pf:.6f}"
    assert abs(intervals['deadline_radius'] - radius_dl) < 1e-6, f"get_prediction_intervals deadline radius {intervals['deadline_radius']:.6f} != {radius_dl:.6f}"
    
    print(f"Prefetch coverage: {coverage_pf*100:.1f}%")
    print(f"Deadline coverage: {coverage_dl*100:.1f}%")
    print(f"get_prediction_intervals output: {intervals}")
    print(f"Radii match: prefetch={radius_pf:.6f}, deadline={radius_dl:.6f}")