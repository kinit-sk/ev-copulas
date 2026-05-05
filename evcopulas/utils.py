"""
Utility functions for EV copula modeling and data processing.

This module provides essential utility functions for setting seed,
inverse transform sampling, data transformation between respective data spaces, and daily load curve calculation.

All functions support both NumPy arrays and PyTorch tensors with automatic
conversion handling for seamless integration across different modeling frameworks.
"""

import numpy as np
import scipy.interpolate as interpolate
import torch


def set_seed(seed=42):
    """Set random seed for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)


def inverse_transform_sampling(hist, bin_edges):
    """Perform inverse transform sampling using histogram."""
    cum_values = np.zeros(bin_edges.shape)
    cum_values[1:] = np.cumsum(hist)
    inv_cdf = interpolate.interp1d(cum_values, bin_edges, bounds_error=False, fill_value=(bin_edges[0], bin_edges[-1]))
    return inv_cdf

def data_sampling(uv_samples, train_data, grid_points=10):
        """
        Transform uniform copula samples back to original data space.
        
        Args:
            uv_samples: torch.Tensor or np.ndarray of uniform samples from copula
            train_data: torch.Tensor or np.ndarray of training data
            grid_points: number of bins for histogram

        Returns:
            np.ndarray of samples in original data space
        """
        # Convert to numpy if needed
        if isinstance(uv_samples, torch.Tensor):
            uv_samples = uv_samples.cpu().numpy()
        if isinstance(train_data, torch.Tensor):
            train_data = train_data.cpu().numpy()
            
        xy_samples = np.zeros_like(uv_samples)
        for i in range(xy_samples.shape[1]):
            hist, bin_edges = np.histogram(train_data[:, i], bins=grid_points, density=True)
            hist = hist / np.sum(hist)
            icdf = inverse_transform_sampling(hist, bin_edges)

            for t in range(uv_samples.shape[0]):
                xy_samples[t, i] = icdf(uv_samples[t, i])

        return xy_samples

def precompute_pseudo_obs(full_data):
    """Compute pseudo-observations once on full dataset."""
    n = len(full_data)
    data_u = np.zeros_like(full_data)
    for i in range(full_data.shape[1]):
        sorted_vals = np.sort(full_data[:, i])
        ranks = np.searchsorted(sorted_vals, full_data[:, i])
        data_u[:, i] = (ranks + 1) / (n + 1)
    return data_u


def compute_daily_load_curve(data, bins=1440):
    """
    Compute average daily load curve from EV charging data.
    
    Args:
        data: np.ndarray with columns [start_hour, duration_hours, energy_kwh]
        bins: number of minute bins (default 1440 = 24*60)
        
    Returns:
        minutes: array of minute values
        load_curve: average load (kW) for each minute
    """
    minutes = np.linspace(0, 1440, bins + 1)
    load_curve = np.zeros(bins)
    counts = np.zeros(bins)
    
    for start_hour, duration, energy in data:
        start_hour = start_hour - 24 if start_hour > 24 else start_hour  # Shift to 0-24 range
        if duration > 0:
            power = energy / duration  # kW
            start_min = start_hour * 60
            end_min = start_min + duration * 60
            
            # Distribute load across minutes
            for i in range(bins):
                min_start = minutes[i]
                min_end = minutes[i + 1]
                
                # Calculate overlap between charging period and this minute bin
                overlap_start = max(start_min, min_start)
                overlap_end = min(end_min, min_end)
                overlap = max(0, overlap_end - overlap_start)
                
                if overlap > 0:
                    load_curve[i] += power * overlap
                    counts[i] += overlap
    
    # Average load per minute
    load_curve = np.where(counts > 0, load_curve / counts, 0)
    
    return minutes[:-1] / 60, load_curve  # Return in hours for plotting