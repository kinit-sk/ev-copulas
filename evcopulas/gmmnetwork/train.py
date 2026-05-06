"""GMM Network training module for neural copula modeling.

This module implements training and evaluation of Gaussian Mixture Model (GMM) neural
networks for copula-based dependency modeling in EV charging data. The approach uses
neural networks to predict GMM parameters for conditional distributions, enabling
flexible multivariate dependency learning.

Replication from paper Li, Z., Bian, Z., Chen, Z., Ozbay, K., & Zhong, M. (2024). 
Synthesis of electric vehicle charging data: A real-world data-driven approach. 
Communications in Transportation Research, 4, 100128. 
https://doi.org/10.1016/j.commtr.2024.100128.

Key Features:

- Neural GMM networks with configurable architecture (hidden layers, components)
- Early stopping with multiple validation metrics (Wasserstein, Kendall's tau, loss)
- Gibbs sampling for synthetic data generation from learned conditional distributions
- Comprehensive evaluation including tail dependence and daily load curve analysis
- Robust NaN handling with conservative median imputation
- Multi-seed evaluation for statistical significance
- GPU acceleration with automatic device detection

Training Process:

1. Data preprocessing and pseudo-observation conversion
2. Neural network training with batch normalization and dropout
3. Early stopping based on validation performance
4. Gibbs sampling for synthetic sample generation
5. Statistical evaluation using multiple metrics
6. Best model selection and result saving

Usage:
    python train.py --dataset trondheim --epochs 5000 --use-early-stopping
    
    python train.py --random-seeds 1 --random-seeds 5 --patience 200
"""

import sys

import numpy as np
import polars as pl
import torch
import torch.distributions as D
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import typer
from loguru import logger
from skimpy import skim
from torch.utils.data import TensorDataset, DataLoader
from typing import Annotated

from evcopulas.config import PROCESSED_DATA_DIR
from evcopulas.dataset import preprocess_trondheim_dataset, preprocess_dundee_dataset, preprocess_proprietary_dataset, split_dataframe
from evcopulas.metrics import compute_wasserstein, compute_generalization_metric, compute_kendall_tau, compute_tail_dependence
from evcopulas.utils import precompute_pseudo_obs, set_seed, compute_daily_load_curve


# Remove default handler
logger.remove()

# Add console handler
logger.add(sys.stderr, format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {message}", level="INFO")

# Add file handler
logger.add("logs/training_gmmnetwork_{time:YYYY-MM-DD HH:mm}.log", 
           format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} | {message}",
           level="DEBUG",
           rotation="500 MB",
           retention="10 days",
           )

app = typer.Typer()


class GMM(torch.nn.Module):
    """Gaussian Mixture Model neural network for conditional density estimation.
    
    A feedforward neural network that predicts GMM parameters for modeling conditional
    distributions. This implementation includes numerical stability features and robust 
    handling of edge cases.
    
    Args:
        input_dim: Input dimension (number of conditioning variables).
        hidden_dim: Hidden layer dimension.
        num_layers: Number of hidden layers.
        n_components: Number of Gaussian components in the mixture.
        dim: Target dimension index to predict.
        seed: Random seed for reproducible initialization. Defaults to None.
        
    Attributes:
        linear1: First linear layer.
        layers: List of hidden linear layers.
        linear3: Output layer.
        batchnorm1: First batch normalization layer.
        batchnorms: Batch normalization layers for hidden layers.
        softplus: Softplus activation for positive parameters.
        dim: Target dimension index.
        n_comp: Number of mixture components.
        device: Computing device (CPU or CUDA).
    """
    def __init__(self, input_dim, hidden_dim, num_layers, n_components, dim, seed=None):
        if seed is not None:
            set_seed(seed)

        super(GMM, self).__init__()
        self.linear1 = torch.nn.Linear(input_dim, hidden_dim)      
        self.layers = torch.nn.ModuleList([torch.nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers-1)])
        self.linear3 = torch.nn.Linear(hidden_dim, 3*n_components)
        self.batchnorm1 = torch.nn.BatchNorm1d(hidden_dim)
        self.batchnorms = torch.nn.ModuleList([torch.nn.BatchNorm1d(hidden_dim) for _ in range(num_layers-1)])
        self.softplus = torch.nn.Softplus(beta=1, threshold=20)
        self.dim = dim
        self.n_comp = n_components

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)
        logger.info(f"device: {self.device}")

    def forward(self, x):
        """Forward pass through the GMM network.
        
        Args:
            x: Input tensor of shape (batch_size, n_dimensions).
            
        Returns:
            Tuple of (gmm_distribution, loss) where:
            
            - gmm_distribution: torch.distributions.MixtureSameFamily object
            - loss: Negative log-likelihood of the target variable
        """
        y = x[:,self.dim]
        x = x[:,torch.arange(x.size(1)) != self.dim]
        x = torch.relu(self.batchnorm1(self.linear1(x)))
        for i in range(len(self.layers)):
            x = torch.relu(self.batchnorms[i](self.layers[i](x)))
        output = self.linear3(x)
        # Clamp output to prevent extreme values
        output = torch.clamp(output, 0, 1000)
        mus = output[:, :self.n_comp]
        sigs = self.softplus(output[:, self.n_comp:2*self.n_comp]) + 1e-6
        
        # Add numerical stability to weights
        weight_logits = output[:, 2*self.n_comp:3*self.n_comp]
        weights = torch.softmax(weight_logits, dim=1)
        
        # Handle any remaining NaN/inf values
        if torch.isnan(weights).any() or torch.isinf(weights).any():
            weights = torch.ones_like(weights) / weights.size(1)  # Uniform weights as fallback
        if torch.isnan(mus).any() or torch.isinf(mus).any():
            mus = torch.zeros_like(mus)  # Zero means as fallback
        if torch.isnan(sigs).any() or torch.isinf(sigs).any():
            sigs = torch.ones_like(sigs)  # Unit variance as fallback
        mix = D.Categorical(weights)
        comp = D.Normal(mus, sigs)

        gmm = D.MixtureSameFamily(mix, comp)
        loss = -gmm.log_prob(y).mean()
        return gmm, loss
    
    
def compute_gmm_log_likelihood(nets, test_data):
    """
    Compute log-likelihood of test data under trained GMM networks.
    
    Evaluates the joint log-likelihood by summing individual dimension likelihoods
    from each trained GMM network.
    
    Args:
        nets (list): List of trained GMM networks, one for each dimension.
        test_data (np.ndarray): Test data of shape (n_samples, n_dimensions).
        
    Returns:
        float: Total log-likelihood across all samples and dimensions.
    """

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    with torch.no_grad():
        test_tensor = torch.FloatTensor(test_data).to(device)
        total_log_likelihood = 0.0
        
        # For each sample in test data
        for sample in test_tensor:
            sample_log_likelihood = 0.0
            
            # For each dimension (network)
            for i, net in enumerate(nets):
                # Get distribution for this dimension
                distribution, _ = net.eval()(sample.unsqueeze(0))
                
                # Compute log probability of the actual value
                actual_value = sample[i]
                log_prob = distribution.log_prob(actual_value)
                sample_log_likelihood += log_prob.item()
            
            total_log_likelihood += sample_log_likelihood
    
    return total_log_likelihood


def train_gmm_network_with_early_stopping(
        dataloader,
        fNum,
        num_epochs=200,
        train_data=None,
        validation_data=None,
        patience=50,
        min_delta=1e-4,
        metric='kendall',
        hidden_size=32,
        num_layers=2,
        gmm_components=5,
        learning_rate=0.005,
        seed=42,
):
    """
    Train GMM networks with early stopping based on validation metrics.
    
    Trains separate GMM networks for each dimension with early stopping to prevent
    overfitting. Supports multiple validation metrics for robust model selection.
    
    Args:
        dataloader: Training data loader with batched samples.
        fNum (int): Number of features/dimensions in the data.
        num_epochs (int, optional): Maximum number of training epochs. Defaults to 200.
        train_data (np.ndarray, optional): Training data for sampling. Defaults to None.
        validation_data (np.ndarray, optional): Validation data. If None, uses training data.
        patience (int, optional): Early stopping patience in epochs. Defaults to 50.
        min_delta (float, optional): Minimum improvement threshold. Defaults to 1e-4.
        metric (str, optional): Validation metric ('wasserstein', 'kendall', 'loss'). Defaults to 'kendall'.
        hidden_size (int, optional): Hidden layer dimension. Defaults to 32.
        num_layers (int, optional): Number of hidden layers. Defaults to 2.
        gmm_components (int, optional): Number of GMM components. Defaults to 5.
        learning_rate (float, optional): Learning rate for optimization. Defaults to 0.005.
        seed (int, optional): Random seed for reproducibility. Defaults to 42.
        
    Returns:
        tuple: (nets, best_metric) where:
        
            - nets: List of trained GMM networks
            - best_metric: Best validation metric achieved
    """

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"device: {device}")

    input_dim = fNum - 1
    val_data = validation_data if validation_data is not None else train_data

    # Define probability model
    p_charge_amount = GMM(input_dim, hidden_size, num_layers, gmm_components, 0, seed)
    p_shour = GMM(input_dim, hidden_size, num_layers, gmm_components, 1, seed)
    p_duration = GMM(input_dim, hidden_size, num_layers, gmm_components, 2, seed)

    nets = [p_charge_amount, p_shour, p_duration]

    # define optimizer and scheduler
    optimizers = [optim.Adam(i.parameters(), lr=learning_rate) for i in nets]
    schedulers = [lr_scheduler.LinearLR(op, start_factor=1.0, end_factor=0.5, total_iters=50) for op in optimizers]

    # Early stopping variables
    best_metric = float('inf')
    patience_counter = 0
    best_states = [net.state_dict().copy() for net in nets]

    # Train the model
    for epoch in range(num_epochs):
        avg = []

        # Training step
        for item in dataloader:
            item = item[0].to(torch.float32).to(device)
            [op.zero_grad() for op in optimizers]
            loss = [net(item)[1] for net in nets]
            [l.backward() for l in loss]
            [op.step() for op in optimizers]
            avg.append([l.item() for l in loss])

        [scheduler.step() for scheduler in schedulers]

        avg = np.array(avg)
        current_loss = np.mean(avg)

        # Validation check every 10 epochs (for efficiency)
        if epoch % 10 == 0:
            gen_samples = gmm_gibbs_sampling(nets, fNum, len(val_data))
            
            # Handle NaN values
            if np.isnan(gen_samples).any():
                for i in range(gen_samples.shape[1]):
                    nan_mask = np.isnan(gen_samples[:, i])
                    if nan_mask.any():
                        gen_samples[nan_mask, i] = np.median(train_data[:, i])

            # Compute metric
            if metric == 'wasserstein':
                current_metric = compute_wasserstein(val_data, gen_samples)
            elif metric == 'kendall':
                _, _, current_metric = compute_kendall_tau(val_data, gen_samples)
            elif metric == 'loss':
                current_metric = current_loss
            else:
                raise ValueError(f"Unknown metric: {metric}")

            # Check for improvement
            if current_metric < best_metric - min_delta:
                best_metric = current_metric
                patience_counter = 0
                best_states = [net.state_dict().copy() for net in nets]
                logger.info(f"Epoch {epoch}: {metric}={current_metric:.4f} (improved)")
            else:
                patience_counter += 10
                logger.info(f"Epoch {epoch}: {metric}={current_metric:.4f} (no improvement, patience={patience_counter}/{patience})")

            # Early stopping
            if patience_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch}")
                # Restore best states
                for net, best_state in zip(nets, best_states):
                    net.load_state_dict(best_state)
                break
        elif epoch % 100 == 0:
            logger.info(f"Epoch {epoch}: Loss={current_loss:.4f}")

    return nets, best_metric


def train_gmm_network(
        dataloader,
        fNum,
        num_epochs=200, 
        hidden_size=32, 
        num_layers=2,
        gmm_components=5,
        learning_rate=0.005,
        seed=42,
):
    """
    Train GMM networks without early stopping.
    
    Trains separate GMM networks for each dimension for a fixed number of epochs.
    This is the basic training function without validation-based early stopping.
    
    Args:
        dataloader: Training data loader with batched samples.
        fNum (int): Number of features/dimensions in the data.
        num_epochs (int, optional): Number of training epochs. Defaults to 200.
        hidden_size (int, optional): Hidden layer dimension. Defaults to 32.
        num_layers (int, optional): Number of hidden layers. Defaults to 2.
        gmm_components (int, optional): Number of GMM components. Defaults to 5.
        learning_rate (float, optional): Learning rate for optimization. Defaults to 0.005.
        seed (int, optional): Random seed for reproducibility. Defaults to 42.
        
    Returns:
        list: Trained GMM networks, one for each dimension.
    """

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"device: {device}")

    input_dim = fNum - 1

    # Define probability model
    p_charge_amount = GMM(input_dim, hidden_size, num_layers, gmm_components, 0, seed)
    p_shour = GMM(input_dim, hidden_size, num_layers, gmm_components, 1, seed)
    p_duration = GMM(input_dim, hidden_size, num_layers, gmm_components, 2, seed)

    nets = [p_charge_amount, p_shour, p_duration]

    # define optimizer and scheduler
    optimizers = [optim.Adam(i.parameters(), lr=learning_rate) for i in nets]
    schedulers = [lr_scheduler.LinearLR(op, start_factor=1.0, end_factor=0.5, total_iters=50)  for op in optimizers]

    # Train the model
    train_losses=[]
    for epoch in range(num_epochs):
        avg=[]

        for item in dataloader:
            item=item[0].to(torch.float32).to(device)
            [op.zero_grad()  for op in optimizers]
            loss=[net(item)[1]  for net in nets]
            [l.backward()   for l in loss]
            [op.step() for op in optimizers]
            avg.append([l.item() for l in loss])

        [scheduler.step() for scheduler in schedulers]

        avg=np.array(avg)
        l=np.mean(avg,0)
        train_losses.append(l)


        if epoch % 10 == 0:
            logger.info(f"Epoch={epoch}, Loss={l}")

    return nets


def gmm_gibbs_sampling(
    nets,
    fNum,
    test_size,
):
    """
    Generate synthetic samples using Gibbs sampling from trained GMM networks.
    
    Uses Gibbs sampling to generate new samples by iteratively sampling each
    dimension from its conditional distribution given the other dimensions.
    
    Args:
        nets (list): List of trained GMM networks, one for each dimension.
        fNum (int): Number of features/dimensions.
        test_size (int): Number of samples to generate.
        
    Returns:
        np.ndarray: Generated samples of shape (test_size, fNum).
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    step = 200
    samples = np.zeros((0,fNum))

    with torch.no_grad():
        batch_samples = torch.rand(test_size, fNum)
        batch_samples = batch_samples.to(device).to(torch.float32)
        for _ in range(step):
            for i in range(fNum):
                distribution,_ = nets[i].eval()(batch_samples)
                batch_samples[:,i] = torch.clamp(distribution.sample().squeeze(), 0)

        samples = np.concatenate([samples, batch_samples.detach().cpu().numpy()], axis=0)

    return samples


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
    ] = [1,2,3,4,5],
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
    Train GMM neural networks for copula-based EV charging data modeling.
    
    This function orchestrates the complete GMM network training pipeline. The approach trains 
    separate neural networks for each dimension to predict GMM parameters for conditional
    distributions, enabling flexible multivariate dependency modeling.
    
    Training Pipeline:

    1. Loads and preprocesses the specified EV charging dataset
    2. Splits data into train/validation/test sets with temporal order preservation
    3. Trains GMM networks with configurable architecture and early stopping
    4. Generates synthetic samples using Gibbs sampling from learned conditionals
    5. Evaluates models using comprehensive statistical metrics
    6. Identifies best performing model and saves results
    
    The GMM networks learn to predict mixture parameters (means, standard deviations,
    and mixture weights) for each dimension conditioned on all other dimensions.
    This enables capturing complex, non-linear dependencies in the data.
    
    Evaluation includes:

    - Distribution similarity: Wasserstein distance
    - Correlation structure: Kendall's tau preservation
    - Extreme values: Tail dependence coefficients
    - Model selection: Information criteria (AIC, BIC)
    - Overfitting assessment: Generalization metrics
    - Practical applications: Daily load curve analysis
    
    Args:
        dataset: Dataset name ('trondheim', 'dundee', or 'proprietary').
        val_size: Fraction of data reserved for validation (0.0-1.0).
        test_size: Fraction of data reserved for testing (0.0-1.0).
        random_seeds: List of random seeds for robust multi-run evaluation.
        epochs: Maximum number of training epochs per network.
        use_early_stopping: Whether to use validation-based early stopping.
        patience: Number of epochs to wait for improvement before early stopping.
        max_eval_samples: Maximum samples for evaluation (0 uses all available data).
        
    Returns:
        None: Results are saved to CSV files in PROCESSED_DATA_DIR.
        
    Output Files:
        - test_data_{dataset}.csv: Original test data for evaluation
        - generated_data_{dataset}_gmmnetwork_seed_{best_seed}.csv: Best synthetic data
        - metrics_{dataset}_gmmnetwork_all_seeds.csv: Comprehensive evaluation metrics
        
    Example:
        Train GMM networks on Trondheim dataset with early stopping:
        ```python
        python train.py --dataset trondheim --epochs 3000 --use-early-stopping
        ```

        Train with custom seeds and patience:
        ```python
        python train.py --random-seeds 1 --random-seeds 5 --random-seeds 10 --patience 200 --max-eval-samples 5000
        ```

        Train without early stopping for fixed epochs:
        ```python
        python train.py --no-use-early-stopping --epochs 1000
        ```
        
    Note:
        - Each dimension requires a separate GMM network (3 networks for EV data)
        - Gibbs sampling iteratively samples each dimension from its conditional
        - NaN values in generated samples are imputed with training data medians
        - Best model selection is based on Kendall's tau preservation
        - GPU acceleration is automatically used when available
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

    # Create DataLoader
    dataset_ev = TensorDataset(torch.FloatTensor(train_data))
    data_loader = DataLoader(dataset_ev, batch_size=64, shuffle=True, drop_last=True)

    logger.info(f"Train data shape: {train_data.shape}")

    # Train GMM Network for each seed
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
        
        if use_early_stopping:
            logger.info("Training with early stopping")
            nets, best_metric = train_gmm_network_with_early_stopping(
                data_loader, train_data.shape[1], epochs, train_data, val_data, patience, seed=seed
            )
            logger.info(f"Best validation metric: {best_metric:.4f}")
        else:
            logger.info("Training without early stopping")
            nets = train_gmm_network(data_loader, train_data.shape[1], epochs, seed=seed)
        
        # Generate test samples
        logger.info("Generating test samples...")
        gen_samples = gmm_gibbs_sampling(nets, train_data.shape[1], len(test_data))
        uv_samples = precompute_pseudo_obs(gen_samples)
        logger.info(f"Generated samples shape: {gen_samples.shape}")

        # Impute NaN values conservatively with training data median
        if np.isnan(gen_samples).any():
            nan_count = np.isnan(gen_samples).sum()
            logger.warning(f"Found {nan_count} NaN values in generated samples, imputing with training median")
            
            for i in range(gen_samples.shape[1]):
                nan_mask = np.isnan(gen_samples[:, i])
                if nan_mask.any():
                    uv_samples[nan_mask, i] = np.median(precompute_pseudo_obs(train_data)[:,i])
                    gen_samples[nan_mask, i] = np.median(train_data[:, i])         
        
        # Calculate log-likelihood for GMM networks
        loglik = compute_gmm_log_likelihood(nets, test_data)
        
        # Count parameters across all networks
        n_params = sum(sum(p.numel() for p in net.parameters()) for net in nets)
        
        # Compute AIC and BIC
        n_samples = len(test_data)
        aic = 2 * n_params - 2 * loglik
        bic = n_params * np.log(n_samples) - 2 * loglik
        
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
    df_gen_out.write_csv(PROCESSED_DATA_DIR / f"generated_data_{dataset}_gmmnetwork_seed_{best_seed}.csv")

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
    results_df.write_csv(PROCESSED_DATA_DIR / f"metrics_{dataset}_gmmnetwork_all_seeds.csv")
    
    logger.info(f"\nSaved test data to: test_data_gmmnetwork_seed_{best_seed}.csv")
    logger.info(f"Saved generated data to: generated_data_{dataset}_gmmnetwork_seed_{best_seed}.csv")
    logger.info(f"Saved metrics to: metrics_{dataset}_gmmnetwork_all_seeds.csv")
    logger.success("Training completed successfully!")


if __name__ == "__main__":
    app()