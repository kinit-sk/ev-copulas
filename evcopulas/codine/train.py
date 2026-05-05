"""CODINE (COpula DensIty Neural Estimation) training module.

This module implements the CODINE algorithm for learning complex dependency structures
in multivariate data using neural copula estimation. CODINE uses adversarial training
with discriminator networks to estimate copula densities through various divergences.

Implemented with improvements from the original paper: Letizia, N. A., Novello, N., & Tonello, A. M. (2025). 
Copula Density Neural Estimation. IEEE Transactions on Neural Networks and Learning Systems, 36(10), 19452–19459. 
https://doi.org/10.1109/TNNLS.2025.3585755

Key Features:

- Multiple divergence types: Kullback-Leibler (KL), Generative Adversarial Network (GAN), 
  and Hellinger Distance (HD)
- Early stopping with validation-based model selection
- Gibbs sampling for generating new samples from learned copulas
- Comprehensive evaluation metrics including Wasserstein distance, Kendall's tau,
  tail dependence, and generalization metrics
- Support for multiple EV charging datasets (Trondheim, Dundee, proprietary)
- Model comparison across different random seeds for robust evaluation

The training process:

1. Preprocesses EV charging data into pseudo-observations
2. Trains discriminator network to distinguish between data and uniform samples
3. Uses learned discriminator to estimate copula density
4. Generates synthetic samples via Gibbs sampling
5. Evaluates model performance using multiple statistical metrics

Usage:
    python train.py --dataset trondheim --divergence GAN --epochs 5000
"""

from __future__ import print_function, division
import sys
from math import *
from typing import Annotated

import numpy as np
import polars as pl
import torch
import torch.nn as nn
import torch.optim as optim
import typer
from loguru import logger
from skimpy import skim
from torch.utils.data import TensorDataset, DataLoader

from evcopulas.config import PROCESSED_DATA_DIR
from evcopulas.dataset import preprocess_trondheim_dataset, preprocess_dundee_dataset, preprocess_proprietary_dataset, split_dataframe
from evcopulas.metrics import compute_wasserstein, compute_kendall_tau, reciprocal_loss, wasserstein_loss, compute_tail_dependence, compute_generalization_metric
from evcopulas.utils import inverse_transform_sampling, precompute_pseudo_obs, set_seed, compute_daily_load_curve

# Remove default handler
logger.remove()

# Add console handler
logger.add(sys.stderr, format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {message}", level="INFO")

# Add file handler
logger.add("logs/training_codine_{time:YYYY-MM-DD HH:mm}.log", 
           format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} | {message}",
           level="DEBUG",
           rotation="500 MB",
           retention="10 days",
           )

app = typer.Typer()

class Discriminator(nn.Module):
    """Neural network discriminator for CODINE copula density estimation.
    
    A feedforward neural network that learns to distinguish between empirical
    copula data and uniform random samples, effectively learning the copula density.
    
    Args:
        latent_dim: Dimensionality of input data (number of variables).
        divergence: Type of divergence ('KL', 'GAN', or 'HD') which determines
                   the output activation function.
    
    Attributes:
        model: The neural network architecture with LeakyReLU activations.
    """
    def __init__(self, latent_dim, divergence):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 100),
            nn.LeakyReLU(0.2),
            nn.Linear(100, 100),
            nn.LeakyReLU(0.2),
            nn.Linear(100, 100),
            nn.LeakyReLU(0.2),
            nn.Linear(100, 1),
            nn.Sigmoid() if divergence == 'GAN' else nn.Softplus()
        )

    def forward(self, x):
        return self.model(x)


class CODINE():
    """CODINE (COpula DensIty Neural Estimation) copula learning algorithm.
    
    CODINE learns complex dependency structures in multivariate data by training
    a discriminator network to estimate copula densities through adversarial learning.
    
    The algorithm supports three divergence measures:
    
    - KL: Kullback-Leibler divergence
    - GAN: Generative Adversarial Network loss
    - HD: Hellinger Distance
    
    Args:
        latent_dim: Dimensionality of the data (number of variables).
        divergence: Divergence type ('KL', 'GAN', 'HD'). Defaults to 'KL'.
        seed: Random seed for reproducibility. Defaults to None.
    
    Attributes:
        latent_dim: Input dimensionality.
        divergence: Divergence type used for training.
        device: Computing device (CPU or CUDA).
        discriminator: Neural network discriminator.
        optimizer: Optimizer for training.
    """
    def __init__(self, latent_dim, divergence='KL', seed=None):
        if seed is not None:
            set_seed(seed)

        self.latent_dim = latent_dim
        self.divergence = divergence
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"device: {self.device}")
        
        self.discriminator = Discriminator(latent_dim, divergence).to(self.device)
        self.optimizer = optim.Adam(self.discriminator.parameters(), lr=0.0001, betas=(0.5, 0.999))

    def train(self, data_loader, epochs):
        for epoch in range(epochs):
            for batch_data in data_loader:
                batch_data = batch_data[0].numpy()
                batch_size = batch_data.shape[0]
                # Compute empirical CDF for batch
                data_u = batch_data

                data_pi = np.random.uniform(0, 1, (batch_size, self.latent_dim))

                # Convert to PyTorch tensors
                data_u_tensor = torch.FloatTensor(data_u).to(self.device)
                data_pi_tensor = torch.FloatTensor(data_pi).to(self.device)

                # Train discriminator
                self.optimizer.zero_grad()
                
                d_u = self.discriminator(data_u_tensor)
                d_pi = self.discriminator(data_pi_tensor)

                if self.divergence == 'KL':
                    loss = -torch.mean(torch.log(d_u)) + torch.mean(d_pi)
                elif self.divergence == 'GAN':
                    valid = torch.ones(batch_size, 1).to(self.device)
                    loss = nn.BCELoss()(d_u, torch.zeros_like(d_u)) + nn.BCELoss()(d_pi, valid)
                elif self.divergence == 'HD':
                    valid = torch.ones(batch_size, 1).to(self.device)
                    loss = wasserstein_loss(valid, d_u) + reciprocal_loss(valid, d_pi)

                loss.backward()
                self.optimizer.step()

            # Calculate metrics
            if epoch % 100 == 0:
                with torch.no_grad():
                    if self.divergence == 'KL':
                        R = d_u
                        SC = d_pi
                    elif self.divergence == 'GAN':
                        R = (1 - d_u) / d_u
                        SC = (1 - d_pi) / d_pi
                    elif self.divergence == 'HD':
                        R = 1 / (d_u**2)
                        SC = 1 / (d_pi**2)

                logger.info(f"{epoch} [D loss: {loss.item():.4f}, Copula estimates: {torch.mean(R).item():.4f}, "
                      f"Self-consistency mean test: {torch.mean(SC).item():.4f}]")
                
    def train_with_early_stopping(self, data_loader, epochs, train_data, validation_data=None, 
                               patience=50, min_delta=1e-4, metric='kendall'):
        """Train CODINE with early stopping based on validation metrics.
        
        Args:
            data_loader: Training data loader
            epochs: Maximum number of epochs
            train_data: Training data for sampling
            validation_data: Validation data (if None, use training data)
            patience: Number of epochs to wait for improvement
            min_delta: Minimum change to qualify as improvement
            metric: Validation metric ('wasserstein', 'kendall', or 'loss')

        Returns:
            Best validation metric achieved
        """
        best_metric = float('inf')
        patience_counter = 0
        best_state = self.discriminator.state_dict().copy()
        val_data = validation_data if validation_data is not None else train_data
        
        for epoch in range(epochs):
            # Training step (same as before)
            for batch_data in data_loader:
                batch_data = batch_data[0].numpy()
                batch_size = batch_data.shape[0]
                
                data_u = batch_data

                data_pi = np.random.uniform(0, 1, (batch_size, self.latent_dim))
                data_u_tensor = torch.FloatTensor(data_u).to(self.device)
                data_pi_tensor = torch.FloatTensor(data_pi).to(self.device)

                self.optimizer.zero_grad()
                d_u = self.discriminator(data_u_tensor)
                d_pi = self.discriminator(data_pi_tensor)

                if self.divergence == 'KL':
                    loss = -torch.mean(torch.log(d_u)) + torch.mean(d_pi)
                elif self.divergence == 'GAN':
                    valid = torch.ones(batch_size, 1).to(self.device)
                    loss = nn.BCELoss()(d_u, torch.zeros_like(d_u)) + nn.BCELoss()(d_pi, valid)
                elif self.divergence == 'HD':
                    valid = torch.ones(batch_size, 1).to(self.device)
                    loss = wasserstein_loss(valid, d_u) + reciprocal_loss(valid, d_pi)

                loss.backward()
                self.optimizer.step()

            # Validation check every epoch (or every N epochs for efficiency)
            if epoch % 10 == 0:
                uv_samples = self.copula_gibbs_sampling(test_size=len(val_data))
                gen_samples = self.data_sampling(uv_samples, train_data)

                # Compute metric
                if metric == 'wasserstein':
                    current_metric = compute_wasserstein(val_data, gen_samples)
                elif metric == 'kendall':
                    _, _, current_metric = compute_kendall_tau(val_data, gen_samples)
                elif metric == 'loss':
                    current_metric = loss.item()
                
                # Check for improvement
                if current_metric < best_metric - min_delta:
                    best_metric = current_metric
                    patience_counter = 0
                    best_state = self.discriminator.state_dict().copy()
                    logger.info(f"Epoch {epoch}: {metric}={current_metric:.4f} (improved)")
                else:
                    patience_counter += 10
                    logger.info(f"Epoch {epoch}: {metric}={current_metric:.4f} (no improvement, patience={patience_counter}/{patience})")
                
                # Early stopping
                if patience_counter >= patience:
                    logger.info(f"Early stopping at epoch {epoch}")
                    self.discriminator.load_state_dict(best_state)
                    break
        
        return best_metric

    def copula_gibbs_sampling(self, grid_points=30, test_size=1000):
        """Sample from learned copula using Gibbs sampling."""
        a = np.linspace(0, 1, grid_points)
        uv_samples = np.zeros((test_size, self.latent_dim))
        uv_samples[0, :] = np.random.uniform(0, 1, self.latent_dim)

        with torch.no_grad():
            for t in range(1, test_size):
                for i in range(self.latent_dim):
                    # Build conditional vector for dimension i
                    if i == 0:
                        # First dimension: condition on all future dimensions from previous sample
                        uv_i_vector = np.column_stack([
                            a,
                            np.tile(uv_samples[t-1, i+1:], (grid_points, 1))
                        ])
                    elif i < self.latent_dim - 1:
                        # Middle dimensions: condition on past (current) and future (previous)
                        uv_i_vector = np.column_stack([
                            np.tile(uv_samples[t, :i], (grid_points, 1)),
                            a,
                            np.tile(uv_samples[t-1, i+1:], (grid_points, 1))
                        ])
                    else:
                        # Last dimension: condition only on past dimensions from current sample
                        uv_i_vector = np.column_stack([
                            np.tile(uv_samples[t, :i], (grid_points, 1)),
                            a
                        ])

                    uv_i_tensor = torch.FloatTensor(uv_i_vector).to(self.device)
                    disc_output = self.discriminator(uv_i_tensor).cpu().numpy()

                    if self.divergence == 'KL':
                        copula_density_vector = disc_output
                    elif self.divergence == 'GAN':
                        copula_density_vector = (1 - disc_output) / disc_output
                    elif self.divergence == 'HD':
                        copula_density_vector = 1 / (disc_output ** 2)

                    copula_density_vector = copula_density_vector / np.sum(copula_density_vector)
                    icdf = inverse_transform_sampling(np.squeeze(copula_density_vector), np.linspace(0, 1, grid_points+1))
                    uv_samples[t, i] = icdf(np.random.uniform(0, 1))

        return uv_samples
    
    def data_sampling(self, uv_samples, train_data, grid_points=30):
        """Transform uniform copula samples back to original data space.
        
        Args:
            uv_samples: Uniform samples from copula
            train_data: Training data for transformation
            grid_points: Number of bins for histogram

        Returns:
            Samples transformed back to original data space.
        """
        # Convert to numpy if needed
        if isinstance(uv_samples, torch.Tensor):
            uv_samples = uv_samples.cpu().numpy()
        if isinstance(train_data, torch.Tensor):
            train_data = train_data.cpu().numpy()
            
        xy_samples = np.zeros_like(uv_samples)
        for i in range(self.latent_dim):
            hist, bin_edges = np.histogram(train_data[:, i], bins=grid_points, density=True)
            hist = hist / np.sum(hist)
            icdf = inverse_transform_sampling(hist, bin_edges)

            for t in range(uv_samples.shape[0]):
                xy_samples[t, i] = icdf(uv_samples[t, i])

        return xy_samples
    
    def log_likelihood(self, data):
        """Compute log-likelihood of data under learned copula."""
        if isinstance(data, torch.Tensor):
            data = data.cpu().numpy()
            
        n = len(data)
        
        # Compute empirical CDF (pseudo-observations)
        data_u = precompute_pseudo_obs(data)
        
        # Get copula density from discriminator
        with torch.no_grad():
            data_u_tensor = torch.FloatTensor(data_u).to(self.device)
            disc_output = self.discriminator(data_u_tensor).cpu().numpy()
            
            if self.divergence == 'KL':
                copula_density = disc_output
            elif self.divergence == 'GAN':
                copula_density = (1 - disc_output) / disc_output
            elif self.divergence == 'HD':
                copula_density = 1 / (disc_output ** 2)
        
        # Compute log-likelihood (add small epsilon to avoid log(0))
        log_likelihood = np.sum(np.log(copula_density + 1e-10))
        
        return log_likelihood

    def aic_bic(self, data):
        """Compute AIC and BIC for the fitted copula model."""
        log_likelihood = self.log_likelihood(data)
        
        # Count parameters in discriminator
        n_params = sum(p.numel() for p in self.discriminator.parameters())
        n_samples = len(data) if isinstance(data, np.ndarray) else data.shape[0]
        
        aic = 2 * n_params - 2 * log_likelihood
        bic = n_params * np.log(n_samples) - 2 * log_likelihood
        
        return aic, bic, log_likelihood

@app.command()
def main(
    dataset: Annotated[
        str, typer.Option(help="Which dataset to use: trondheim, dundee, proprietary")
    ] = "trondheim",
    val_size: Annotated[
        float, typer.Option(help="Validation size as fraction of data")
    ] = 0.1,
    test_size: Annotated[
        float, typer.Option(help="Test size as fraction of data")
    ] = 0.1,
    random_seeds: Annotated[
        list[int], typer.Option(help="Random seeds (e.g., --random-seeds 1 --random-seeds 2 --random-seeds 3)")
    ] = [2,3,5,6,7,8],
    divergence: Annotated[
        str, typer.Option(help="Divergence to use: KL, GAN, or HD")
    ] = "GAN",
    epochs: Annotated[
        int, typer.Option(help="Number of training epochs")
    ] = 5000,
    use_early_stopping: Annotated[
        bool, typer.Option("--use-early-stopping/--no-use-early-stopping", help="Use early stopping")
    ] = True,
    patience: Annotated[
        int, typer.Option(help="Early stopping patience")
    ] = 500,
    max_eval_samples: Annotated[
        int, typer.Option(help="Maximum samples for evaluation (0 = use all)")
    ] = 0,
) -> None:
    """
    Train CODINE copula models on EV charging datasets with comprehensive evaluation.
    
    This function orchestrates the complete CODINE training pipeline.    
    The training supports three divergence types (KL, GAN, HD) and includes
    optional early stopping based on validation performance. Multiple random
    seeds ensure robust model comparison and statistical significance.
    
    Evaluation metrics include:

    - Wasserstein distance for distribution similarity
    - Kendall's tau for correlation structure preservation
    - Tail dependence for extreme value modeling
    - Generalization metrics for overfitting assessment
    - Daily load curve analysis for practical EV applications
    - Information criteria (AIC, BIC) for model selection
    
    Args:
        dataset: Dataset name ('trondheim', 'dundee', or 'proprietary').
        val_size: Fraction of data reserved for validation (0.0-1.0).
        test_size: Fraction of data reserved for testing (0.0-1.0).
        random_seeds: List of random seeds for multiple training runs.
        divergence: Divergence type for training ('KL', 'GAN', or 'HD').
        epochs: Maximum number of training epochs.
        use_early_stopping: Whether to use early stopping based on validation metrics.
        patience: Number of epochs to wait for improvement before early stopping.
        max_eval_samples: Maximum samples for evaluation (0 uses all available data).
        
    Returns:
        None: Results are saved to CSV files in PROCESSED_DATA_DIR.
        
    Output Files:
        - test_data_{dataset}.csv: Original test data
        - generated_data_{dataset}_codine_seed_{best_seed}.csv: Best synthetic data
        - metrics_{dataset}_codine_all_seeds.csv: Comprehensive evaluation metrics
        
    Example:
        Train CODINE on Trondheim dataset with GAN divergence:
        
        ```python
         python train.py --dataset trondheim --divergence GAN --epochs 3000
        ```
        Train with custom seeds and early stopping:
        
        ```python
        python train.p --random-seeds 1 --random-seeds 5 --random-seeds 10 --use-early-stopping --patience 200
        ```
    """
    
    logger.info(f"Training with seeds: {random_seeds}")
    
    if dataset == "trondheim":
        # Load Trondheim data
        df = preprocess_trondheim_dataset()
    elif dataset == "dundee":
        # Load Dundee data
        df = preprocess_dundee_dataset()
    elif dataset == "proprietary":
        # Load proprietary data
        df = preprocess_proprietary_dataset()

    logger.info(skim(df))

    # split data into train, validation and test subsets
    df_train, df_val, df_test = split_dataframe(df, 1-(val_size+test_size), val_size, test_size)

    logger.info(f"Train set shape: {df_train.shape}")
    logger.info(f"Validation set shape: {df_val.shape}")
    logger.info(f"Test set shape: {df_test.shape}")

    # Convert to numpy arrays
    train_data = df_train.to_numpy()
    val_data = df_val.to_numpy()
    test_data = df_test.to_numpy()

    # Subsample if needed
    if max_eval_samples > 0 and len(test_data) > max_eval_samples:
        logger.info(f"Subsampling test data from {len(test_data)} to {max_eval_samples}")
        np.random.seed(1)  # Use first seed for consistency
        indices = np.random.choice(len(test_data), max_eval_samples, replace=False)
        test_data = test_data[indices]
    
    if max_eval_samples > 0 and len(train_data) > max_eval_samples:
        logger.info(f"Subsampling train data from {len(train_data)} to {max_eval_samples}")
        np.random.seed(1)
        indices = np.random.choice(len(train_data), max_eval_samples, replace=False)
        train_data_eval = train_data[indices]
    else:
        train_data_eval = train_data

    # Create DataLoader with precomputed pseudo-observations
    train_data_u = precompute_pseudo_obs(train_data)
    dataset_ev = TensorDataset(torch.FloatTensor(train_data_u))
    data_loader = DataLoader(dataset_ev, batch_size=64, shuffle=True)

    logger.info(f"Train data shape: {train_data.shape}")

    # Train CODINE for each seed
    results = {}

    # save data to csv
    df_test_out = pl.DataFrame({
        'start_hour_shifted': test_data[:, 0],
        'duration': test_data[:, 1],
        'energy': test_data[:, 2]
    })
    df_test_out.write_csv(PROCESSED_DATA_DIR / f"test_data_{dataset}.csv")
    logger.info(f"\nSaved test data to: test_data_{dataset}.csv")
    
    for seed in random_seeds:
        logger.info(f"\n{'='*100}")
        logger.info(f"Training with seed: {seed}")
        logger.info(f"\n{'='*100}")
        
        codine = CODINE(3, divergence, seed=seed)
        
        if use_early_stopping:
            logger.info("Training with early stopping")
            best_metric = codine.train_with_early_stopping(
                data_loader, epochs, train_data, val_data, patience,
            )
            logger.info(f"Best validation metric: {best_metric:.4f}")
        else:
            logger.info("Training without early stopping")
            codine.train(data_loader, epochs)
        
        # Generate test samples
        logger.info("Generating test samples...")
        uv_samples = codine.copula_gibbs_sampling(test_size=len(test_data))
        gen_samples = codine.data_sampling(uv_samples, train_data)

        # Impute NaN values conservatively with training data median
        if np.isnan(gen_samples).any():
            nan_count = np.isnan(gen_samples).sum()
            logger.warning(f"Found {nan_count} NaN values in generated samples, imputing with training median")
            
            for i in range(gen_samples.shape[1]):
                nan_mask = np.isnan(gen_samples[:, i])
                if nan_mask.any():
                    uv_samples[nan_mask, i] = np.median(precompute_pseudo_obs(train_data)[:,i])
                    gen_samples[nan_mask, i] = np.median(train_data[:, i])         
        
        # Compute AIC, BIC, log-likelihood
        aic, bic, loglik = codine.aic_bic(test_data)
        logger.info(f"AIC: {aic:.2f}, BIC: {bic:.2f}, LogLik: {loglik:.2f}")
        
        # Compute Wasserstein distance
        wass_dist = compute_wasserstein(test_data, gen_samples)
        logger.info(f"Wasserstein distance: {wass_dist:.4f}")

        # Compute generalization metric
        gen_metric = compute_generalization_metric(train_data_eval, test_data, gen_samples)
        logger.info(f"Generalization metric: {gen_metric:.4f}")
        
        # Compute Kendall's tau
        tau_real, tau_gen, tau_diff = compute_kendall_tau(test_data, gen_samples)
        logger.info(f"Kendall tau difference: {tau_diff:.4f}")
        logger.info("Kendall's tau matrix (real data):")
        logger.info(tau_real)
        logger.info("Kendall's tau matrix (generated data):")
        logger.info(tau_gen)
        
        # Compute tail dependence
        upper_tail_real, lower_tail_real = compute_tail_dependence(test_data)
        upper_tail_gen, lower_tail_gen = compute_tail_dependence(gen_samples)
        
        upper_diff = {k: abs(upper_tail_real[k] - upper_tail_gen[k]) for k in upper_tail_real.keys()}
        lower_diff = {k: abs(lower_tail_real[k] - lower_tail_gen[k]) for k in lower_tail_real.keys()}
        
        upper_mae = np.mean(list(upper_diff.values()))
        lower_mae = np.mean(list(lower_diff.values()))
        upper_max = np.max(list(upper_diff.values()))
        lower_max = np.max(list(lower_diff.values()))
        
        logger.info(f"Tail dependence - Upper MAE: {upper_mae:.4f}, Lower MAE: {lower_mae:.4f}")
        logger.info(f"Tail dependence - Upper Max: {upper_max:.4f}, Lower Max: {lower_max:.4f}")
        
        logger.info("\nPair-wise upper tail differences:")
        for pair, diff in upper_diff.items():
            logger.info(f"  {pair}: {diff:.4f}")
        
        logger.info("\nPair-wise lower tail differences:")
        for pair, diff in lower_diff.items():
            logger.info(f"  {pair}: {diff:.4f}")

        # compute daily load curves per minute and then its errors
        _, load_real = compute_daily_load_curve(test_data)
        _, load_gen = compute_daily_load_curve(gen_samples)
        load_mae = np.mean(np.abs(load_real - load_gen))
        load_rmse = np.sqrt(np.mean((load_real - load_gen) ** 2))
        
        # Store results
        results[seed] = {
            'kendall_tau_diff': tau_diff,
            'wasserstein': wass_dist,
            'generalization': gen_metric,
            'aic': aic,
            'bic': bic,
            'loglik': loglik,
            'upper_tail_mae': upper_mae,
            'lower_tail_mae': lower_mae,
            'upper_tail_max': upper_max,
            'lower_tail_max': lower_max,
            'load_mae': load_mae,
            'load_rmse': load_rmse,
            'gen_samples': gen_samples,
            'uv_samples': uv_samples,
        }
    
    # Summary
    logger.info(f"\n{'='*100}")
    logger.info("Training Summary")
    logger.info(f"\n{'='*100}")
    for seed, metrics in results.items():
        logger.info(f"Seed {seed}: Kendall={metrics['kendall_tau_diff']:.4f}, "
                   f"Wasserstein={metrics['wasserstein']:.4f}, "
                   f"Log-likelihood={metrics['loglik']:.2f},"
                   f"AIC={metrics['aic']:.2f}")
    
    # Find best seed by Kendall tau
    best_seed = min(results, key=lambda s: results[s]['kendall_tau_diff'])
    logger.info(f"\nBest seed (by Kendall tau): {best_seed} with difference: {results[best_seed]['kendall_tau_diff']:.4f}")
    
    # Save best results to CSV
    best_gen_samples = results[best_seed]['gen_samples']
    best_uv_samples = results[best_seed]["uv_samples"]
    
    df_gen_out = pl.DataFrame({
        'start_hour_shifted': best_gen_samples[:, 0],
        'duration': best_gen_samples[:, 1],
        'energy': best_gen_samples[:, 2],
        'uniform_sample_1': best_uv_samples[:, 0],
        'uniform_sample_2': best_uv_samples[:, 1],
        'uniform_sample_3': best_uv_samples[:, 2]
    })
    df_gen_out.write_csv(PROCESSED_DATA_DIR / f"generated_data_{dataset}_codine_seed_{best_seed}.csv")

    # save all metrics dict
    results_df = pl.DataFrame([
        {
            'seed': seed,
            'kendall_tau_diff': metrics['kendall_tau_diff'],
            'wasserstein': metrics['wasserstein'],
            'generalization': metrics['generalization'],
            'aic': metrics['aic'],
            'bic': metrics['bic'],
            'loglik': metrics['loglik'],
            'upper_tail_mae': metrics['upper_tail_mae'],
            'lower_tail_mae': metrics['lower_tail_mae'],
            'upper_tail_max': metrics['upper_tail_max'],
            'lower_tail_max': metrics['lower_tail_max'],
            'load_mae': metrics['load_mae'],
            'load_rmse': metrics['load_rmse'],
        }
        for seed, metrics in results.items()
    ])
    results_df.write_csv(PROCESSED_DATA_DIR / f"metrics_{dataset}_codine_all_seeds.csv")
    
    logger.info(f"\nSaved test data to: test_data_codine_seed_{best_seed}.csv")
    logger.info(f"Saved generated data to: generated_data_{dataset}_codine_seed_{best_seed}.csv")
    logger.info(f"Saved metrics to: metrics_{dataset}_codine_all_seeds.csv")
    logger.success("Training completed successfully!")


if __name__ == "__main__":
    app()
