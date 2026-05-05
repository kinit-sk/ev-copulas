"""Performance metrics module for copula model evaluation.

This module provides comprehensive metrics for evaluating copula models and generative
models, including:

- Information criteria (AIC, BIC) for model selection
- Distribution distance metrics (Wasserstein distance)
- Correlation structure evaluation (Kendall's tau)
- Tail dependence analysis for extreme value modeling
- Generalization metrics for assessing model overfitting
- Custom loss functions for neural network training

All functions support both NumPy arrays and PyTorch tensors for flexibility
across different modeling frameworks.
"""

import numpy as np
import torch
from scipy.spatial.distance import cdist
from scipy.stats import kendalltau, wasserstein_distance


def wasserstein_loss(y_true, y_pred):
    """Compute Wasserstein loss for neural network training.
    
    Args:
        y_true: True values.
        y_pred: Predicted values.
        
    Returns:
        Mean Wasserstein loss.
    """
    return torch.mean(y_true * y_pred)

def reciprocal_loss(y_true, y_pred):
    """
    Compute reciprocal loss for neural network training.
    
    Args:
        y_true (torch.Tensor): True values.
        y_pred (torch.Tensor): Predicted values.
        
    Returns:
        torch.Tensor: Mean reciprocal loss.
    """
    return torch.mean(torch.pow(y_true * y_pred, -1))

def compute_aic(loglik, k):
    """
    Compute Akaike Information Criterion (AIC) for model selection.
    
    Args:
        loglik (float): Log-likelihood of the model.
        k (int): Number of parameters in the model.
        
    Returns:
        float: AIC value (lower is better).
    """
    return 2 * k - 2 * loglik

def compute_bic(loglik, k, n):
    """
    Compute Bayesian Information Criterion (BIC) for model selection.
    
    Args:
        loglik (float): Log-likelihood of the model.
        k (int): Number of parameters in the model.
        n (int): Number of observations.
        
    Returns:
        float: BIC value (lower is better).
    """
    return np.log(n) * k - 2 * loglik

def compute_wasserstein(real_data, generated_data):
    """
    Compute Wasserstein distance between real and generated data.
    
    Args:
        real_data: np.ndarray or torch.Tensor of real data
        generated_data: np.ndarray or torch.Tensor of generated data
        
    Returns:
        wasserstein_dist: Mean Wasserstein distance across all dimensions
    """
    if isinstance(real_data, torch.Tensor):
        real_data = real_data.cpu().numpy()
    if isinstance(generated_data, torch.Tensor):
        generated_data = generated_data.cpu().numpy()
    
    distances = []
    for i in range(real_data.shape[1]):
        dist = wasserstein_distance(real_data[:, i], generated_data[:, i])
        distances.append(dist)
    
    return np.mean(distances)


def compute_kendall_tau(real_data, generated_data):
    """
    Compute Kendall's tau correlation matrices and their difference.
    
    Args:
        real_data: np.ndarray or torch.Tensor of real data
        generated_data: np.ndarray or torch.Tensor of generated data
        
    Returns:
        tau_real: Kendall's tau correlation matrix for real data
        tau_gen: Kendall's tau correlation matrix for generated data
        tau_diff: Frobenius norm of difference between correlation matrices
    """
    if isinstance(real_data, torch.Tensor):
        real_data = real_data.cpu().numpy()
    if isinstance(generated_data, torch.Tensor):
        generated_data = generated_data.cpu().numpy()
    
    tau_real = np.zeros((real_data.shape[1], real_data.shape[1]))
    tau_gen = np.zeros((real_data.shape[1], real_data.shape[1]))
    
    for i in range(real_data.shape[1]):
        for j in range(real_data.shape[1]):
            if i == j:
                tau_real[i, j] = 1.0
                tau_gen[i, j] = 1.0
            else:
                tau_real[i, j], _ = kendalltau(real_data[:, i], real_data[:, j])
                tau_gen[i, j], _ = kendalltau(generated_data[:, i], generated_data[:, j])
    
    tau_diff = np.linalg.norm(tau_real - tau_gen, 'fro')
    
    return tau_real, tau_gen, tau_diff


def compute_tail_dependence(data, threshold=0.95):
    """
    Compute empirical tail dependence coefficients.
    
    Args:
        data: np.ndarray or torch.Tensor of data
        threshold: quantile threshold for tail dependence (default 0.95)
        
    Returns:
        upper_tail: Upper tail dependence coefficients (dict of pairs)
        lower_tail: Lower tail dependence coefficients (dict of pairs)
    """
    if isinstance(data, torch.Tensor):
        data = data.cpu().numpy()
    
    n = len(data)
    
    # Convert to pseudo-observations (empirical CDF)
    data_u = np.zeros_like(data)
    for i in range(data.shape[1]):
        sorted_vals = np.sort(data[:, i])
        ranks = np.searchsorted(sorted_vals, data[:, i])
        data_u[:, i] = (ranks + 1) / (n + 1)
    
    upper_tail = {}
    lower_tail = {}
    
    for i in range(data.shape[1]):
        for j in range(i + 1, data.shape[1]):
            # Upper tail dependence
            upper_exceedances = (data_u[:, i] > threshold) & (data_u[:, j] > threshold)
            lambda_u = np.sum(upper_exceedances) / np.sum(data_u[:, i] > threshold) if np.sum(data_u[:, i] > threshold) > 0 else 0
            upper_tail[f'{i}-{j}'] = lambda_u
            
            # Lower tail dependence
            lower_exceedances = (data_u[:, i] < (1 - threshold)) & (data_u[:, j] < (1 - threshold))
            lambda_l = np.sum(lower_exceedances) / np.sum(data_u[:, i] < (1 - threshold)) if np.sum(data_u[:, i] < (1 - threshold)) > 0 else 0
            lower_tail[f'{i}-{j}'] = lambda_l
    
    return upper_tail, lower_tail


def compute_generalization_metric(train_data, test_data, generated_data):
    """
    Calculates the generalization metric rho_2 based on the Li et al. paper's definition (https://doi.org/10.1016/j.commtr.2024.100128).
    
    Args:
        train_data (np.ndarray): Training set samples (n_samples_train, n_features)
        test_data (np.ndarray): Test set samples (n_samples_test, n_features)
        generated_data (np.ndarray): Generated samples (n_samples_test, n_features)
    
    Returns:
        float: The average ratio rho_2
    """
    # Compute distance from each training point t to all validation points v
    dist_train_to_test = cdist(train_data, test_data, metric='euclidean')
    
    # Compute distance from each training point t to all generated points s
    dist_train_to_gen = cdist(train_data, generated_data, metric='euclidean')
    
    # Find the minimum distance for each t in T to sets V and S
    min_dist_train_test = np.min(dist_train_to_test, axis=1)
    min_dist_train_gen = np.min(dist_train_to_gen, axis=1)
    
    # Calculate the ratio for each training sample: min_dist(t, V) / min_dist(t, S)
    ratios = min_dist_train_test / (min_dist_train_gen + 1e-10)
    
    metric = np.mean(ratios)
    
    return metric
