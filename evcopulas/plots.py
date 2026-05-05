"""
Visualization functions for copula modeling results and EV charging data analysis.

All plotting functions use matplotlib and support customizable figure sizes,
feature names, and color schemes for publication-ready visualizations.
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde

from evcopulas.utils import compute_daily_load_curve


def plot_marginal_densities(real_data, simulated_data, feature_names=None, figsize=(15, 5)):
    """
    Plot marginal density comparisons between real and simulated data using KDE.
    
    Args:
        real_data: np.ndarray of real data (n_samples, n_features)
        simulated_data: np.ndarray of simulated data (n_samples, n_features)
        feature_names: list of feature names (optional)
        figsize: figure size tuple (optional)

    Returns:
        None
    """
    n_features = real_data.shape[1]
    
    if feature_names is None:
        feature_names = [f'Feature {i+1}' for i in range(n_features)]
    
    fig, axes = plt.subplots(1, n_features, figsize=figsize)
    if n_features == 1:
        axes = [axes]
    
    for i, ax in enumerate(axes):
        # KDE for real data
        kde_real = gaussian_kde(real_data[:, i])
        x_real = np.linspace(real_data[:, i].min(), real_data[:, i].max(), 200)
        ax.plot(x_real, kde_real(x_real), label='Real', color='blue', linewidth=2)
        
        # KDE for simulated data
        kde_sim = gaussian_kde(simulated_data[:, i])
        x_sim = np.linspace(simulated_data[:, i].min(), simulated_data[:, i].max(), 200)
        ax.plot(x_sim, kde_sim(x_sim), label='Simulated', color='red', linewidth=2)
        
        ax.set_xlabel(feature_names[i])
        ax.set_ylabel('Density')
        ax.legend()
        ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def plot_load_curve_comparison(real_data, simulated_data, model_names=None, bins=1440):
    """
    Plot comparison of daily load curves for multiple models.
    
    Args:
        real_data: np.ndarray of real data
        simulated_data: dict or list of simulated data arrays
        model_names: list of model names (optional)
        bins: number of bins for load curve (optional)
    
    Returns:
        fig: matplotlib figure
        metrics: dict of MAE and RMSE for each model
    """
    hours_real, load_real = compute_daily_load_curve(real_data, bins)
    
    # Handle single array input
    if isinstance(simulated_data, np.ndarray):
        simulated_data = [simulated_data]
        model_names = model_names or ['Simulated']
    elif isinstance(simulated_data, dict):
        model_names = list(simulated_data.keys())
        simulated_data = list(simulated_data.values())
    else:
        model_names = model_names or [f'Model {i+1}' for i in range(len(simulated_data))]
    
    fig, ax = plt.subplots()
    ax.plot(hours_real, load_real, label='Real', linewidth=2, alpha=0.9, color='black')
    
    metrics = {}
    colors = plt.cm.tab10(np.linspace(0, 1, len(simulated_data)))
    
    for i, (sim_data, name) in enumerate(zip(simulated_data, model_names)):
        hours_sim, load_sim = compute_daily_load_curve(sim_data, bins)
        ax.plot(hours_sim, load_sim, label=name, linewidth=1.5, alpha=0.8)
        
        # Compute metrics
        mae = np.mean(np.abs(load_real - load_sim))
        rmse = np.sqrt(np.mean((load_real - load_sim) ** 2))
        metrics[name] = {'MAE': mae, 'RMSE': rmse}
        print(f"{name} - MAE: {mae:.2f} kW, RMSE: {rmse:.2f} kW")
    
    ax.set_xlabel('Hour of Day')
    ax.set_ylabel('Average Load (kW)')
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_ylim(5,35)
    ax.set_xlim(0, 24)
    ax.set_xticks(range(0, 25, 2))
    plt.tight_layout()
    plt.show()
    
    return fig, metrics
