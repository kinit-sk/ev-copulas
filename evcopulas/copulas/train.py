"""Classical copula training module for EV charging data modeling.

This module provides a comprehensive framework for training and evaluating classical
copula models on EV charging datasets. It supports multiple copula families including
Archimedean copulas (Clayton, Frank, Gumbel), elliptical copulas (Gaussian, Student-t),
and flexible vine copulas for complex dependency modeling.

Key Features:

- Six copula families: Gaussian, Clayton, Frank, Gumbel, Student-t, and Vine copulas
- Robust parameter estimation using Kendall's tau inversion method (itau)
- Multi-seed evaluation for statistical significance and model robustness
- Comprehensive evaluation metrics: Wasserstein distance, Kendall's tau preservation,
  tail dependence analysis, information criteria (AIC/BIC), and generalization metrics
- Inverse transform sampling for synthetic data generation
- Daily load curve analysis for practical EV charging applications
- Temporal data splitting preserving chronological order
- NaN handling with conservative median imputation

Evaluation Pipeline:

1. Data preprocessing and pseudo-observation conversion
2. Temporal train/validation/test splitting
3. Multi-seed copula parameter estimation
4. Synthetic sample generation via inverse transform sampling
5. Statistical evaluation using multiple metrics
6. Best model selection based on Kendall's tau preservation

Usage:
    python train.py --dataset trondheim --copulas gaussian --copulas vine

    python train.py --random-seeds 1 --random-seeds 5 --max-train-samples 10000
"""

from __future__ import print_function, division
import sys
from math import *
from typing import Annotated

import numpy as np
import polars as pl
import pyvinecopulib as pv
import typer
from copulae import GaussianCopula, ClaytonCopula, FrankCopula, GumbelCopula, StudentCopula
from loguru import logger
from pyvinecopulib import Vinecop
from skimpy import skim

from evcopulas.config import PROCESSED_DATA_DIR
from evcopulas.dataset import preprocess_trondheim_dataset, preprocess_dundee_dataset, preprocess_proprietary_dataset, split_dataframe
from evcopulas.metrics import compute_wasserstein, compute_kendall_tau, compute_tail_dependence, compute_aic, compute_bic, compute_generalization_metric
from evcopulas.utils import data_sampling, compute_daily_load_curve

# Remove default handler
logger.remove()

# Add console handler
logger.add(sys.stderr, format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {message}", level="INFO")

# Add file handler
logger.add("logs/training_copula_{time:YYYY-MM-DD HH:mm}.log", 
           format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} | {message}",
           level="DEBUG",
           rotation="500 MB",
           retention="10 days",
           )

app = typer.Typer()


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
    copulas: Annotated[
        list[str], typer.Option(help="What copulas to train (supported: gaussian, clayton, frank, gumbel, vine, student)")
    ] = ["gaussian", "clayton", "frank", "gumbel", "vine", "student"],
    max_train_samples: Annotated[
        int, typer.Option(help="Maximum samples for training (0 = use all)")
    ] = 0,
    max_eval_samples: Annotated[
        int, typer.Option(help="Maximum samples for evaluation (0 = use all)")
    ] = 0, 
) -> None:
    """Train and evaluate multiple copula models on EV charging datasets.
    
    This function provides a comprehensive copula modeling pipeline for EV charging data:
    
    1. Loads and preprocesses the specified EV charging dataset
    2. Splits data into train/validation/test sets with temporal order preservation
    3. Trains multiple copula types with robust multi-seed evaluation
    4. Generates synthetic samples using inverse transform sampling
    5. Evaluates models using comprehensive statistical metrics
    6. Identifies best performing models and saves results
    
    Supported copula families:
    
    - Gaussian: Multivariate normal copula with correlation matrix
    - Clayton: Archimedean copula with lower tail dependence
    - Frank: Archimedean copula with symmetric dependence
    - Gumbel: Archimedean copula with upper tail dependence
    - Student-t: Multivariate t-copula with tail dependence
    - Vine: Flexible vine copula for complex dependency structures
    
    All copulas are fitted using Kendall's tau inversion method (itau) for
    robust parameter estimation. Vine copulas use the pyvinecopulib library
    for advanced dependency modeling.
    
    Evaluation metrics include:
    
    - Wasserstein distance for overall distribution similarity
    - Kendall's tau preservation for correlation structure
    - Tail dependence coefficients for extreme value modeling
    - Information criteria (AIC, BIC) for model selection
    - Generalization metrics for overfitting assessment
    - Daily load curve analysis for practical EV applications
    
    Args:
        dataset: Dataset name ('trondheim', 'dundee', or 'proprietary').
        val_size: Fraction of data reserved for validation (0.0-1.0).
        test_size: Fraction of data reserved for testing (0.0-1.0).
        random_seeds: List of random seeds for multiple training runs.
        copulas: List of copula types to train and evaluate.
        max_train_samples: Maximum samples for training (0 uses all data).
        max_eval_samples: Maximum samples for evaluation (0 uses all data).
        
    Returns:
        Results are saved to CSV files in PROCESSED_DATA_DIR.
        
    Output Files:
        For each copula type:
    
        - test_data_{dataset}.csv: Original test data (shared across copulas)
        - generated_data_{dataset}_{copula}_seed_{best_seed}.csv: Best synthetic data
        - metrics_{dataset}_{copula}_all_seeds.csv: Comprehensive evaluation metrics
        
    Example:
        Train all copulas on Trondheim dataset:
        
        ```python
        python train.py --dataset trondheim
        ```
        
        Train specific copulas with custom seeds:
        
        ```python
        python train.py --copulas gaussian --copulas vine --random-seeds 1 --random-seeds 5 --random-seeds 10
        ```
        
        Train with data subsampling for faster evaluation:
        
        ```python
        python train.py --max-train-samples 10000 --max-eval-samples 5000
        ```
        
    Note:
        - Vine copulas may require longer training time due to complexity
        - NaN values in generated samples are imputed with training data medians
        - Best model selection is based on Kendall's tau preservation
        - All copulas are converted to pseudo-observations before fitting
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
    df_train, _, df_test = split_dataframe(df, 1-(val_size+test_size), val_size, test_size)

    logger.info(f"Train set shape: {df_train.shape}")
    logger.info(f"Test set shape: {df_test.shape}")

    # Convert to numpy arrays
    train_data = df_train.to_numpy()
    test_data = df_test.to_numpy()

    # Subsample if needed
    if max_eval_samples > 0 and len(test_data) > max_eval_samples:
        logger.info(f"Subsampling test data from {len(test_data)} to {max_eval_samples}")
        np.random.seed(random_seeds[0])  # Use first seed for consistency
        indices = np.random.choice(len(test_data), max_eval_samples, replace=False)
        test_data = test_data[indices]
    
    if max_eval_samples > 0 and len(train_data) > max_eval_samples:
        logger.info(f"Subsampling train data from {len(train_data)} to {max_eval_samples} for evaluation only")
        np.random.seed(random_seeds[0])
        indices = np.random.choice(len(train_data), max_eval_samples, replace=False)
        train_data_eval = train_data[indices]
    else:
        train_data_eval = train_data

    if max_train_samples > 0 and len(train_data) > max_train_samples:
        logger.info(f"Subsampling train data from {len(train_data)} to {max_train_samples}")
        np.random.seed(random_seeds[0])
        indices = np.random.choice(len(train_data), max_train_samples, replace=False)
        train_data = train_data[indices]

    # Train copula models for each seed
    results = {}

    # Save test data
    df_test_out = pl.DataFrame({
            'start_hour_shifted': test_data[:, 0],
            'duration': test_data[:, 1],
            'energy': test_data[:, 2]
        })
    df_test_out.write_csv(PROCESSED_DATA_DIR / f"test_data_{dataset}.csv")
    logger.info(f"\nSaved test data to: test_data_{dataset}.csv")

    for copula_type in copulas:

        if copula_type not in ["gaussian", "clayton", "frank", "gumbel", "vine", "student"]:
            raise ValueError("Unsupported copula type.")
        
        results = {}  # Reset results for each copula type

        logger.info(f"Fitting {copula_type} copula...")

        for seed in random_seeds:
            # Set numpy random seed before fitting
            np.random.seed(seed)

            if copula_type == "vine":
                cop = Vinecop.from_data(pv.to_pseudo_obs(train_data))
            else:
                if copula_type == "gaussian":
                    cop = GaussianCopula(dim=train_data.shape[1])
                elif copula_type == "clayton":
                    cop = ClaytonCopula(dim=train_data.shape[1])
                elif copula_type == "frank":
                    cop = FrankCopula(dim=train_data.shape[1])
                elif copula_type == "gumbel":
                    cop = GumbelCopula(dim=train_data.shape[1])
                elif copula_type == "student":
                    cop = StudentCopula(dim=train_data.shape[1])
        
                cop.fit(train_data, to_pobs=True, method="itau")

            logger.info(f"\n{'='*100}")
            logger.info(f"Generating for copula {copula_type} with seed: {seed}")
            logger.info(f"\n{'='*100}")
            
            logger.info("Generating test samples...")
            if copula_type != "vine":
                # Generate test samples
                uv_samples = cop.random(n=test_data.shape[0], seed=seed)
                gen_samples = data_sampling(uv_samples, train_data)
                
                # Impute NaN values conservatively with training data median
                if np.isnan(gen_samples).any():
                    nan_count = np.isnan(gen_samples).sum()
                    logger.warning(f"Found {nan_count} NaN values in generated samples, imputing with training median")
                    
                    for i in range(gen_samples.shape[1]):
                        nan_mask = np.isnan(gen_samples[:, i])
                        if nan_mask.any():
                            uv_samples[nan_mask, i] = np.median(pv.to_pseudo_obs(train_data)[:,i])
                            gen_samples[nan_mask, i] = np.median(train_data[:, i])
                
                # Compute AIC, BIC, log-likelihood
                loglik = cop.log_lik(test_data, to_pobs=True)
                if isinstance(cop.params, float):
                    n_params = 1
                elif hasattr(cop.params, 'size'):
                    n_params = cop.params.size
                elif hasattr(cop.params, '__len__'):
                    n_params = len(cop.params)
                else:
                    n_params = np.array(cop.params).size

                aic = compute_aic(loglik, n_params)
                bic = compute_bic(loglik, n_params, test_data.shape[0])
            else:
                # Generate test samples
                uv_samples = cop.simulate(test_data.shape[0], seeds=[seed, seed, seed])
                gen_samples = data_sampling(uv_samples, train_data)
                
                # Impute NaN values conservatively with training data median
                if np.isnan(gen_samples).any():
                    nan_count = np.isnan(gen_samples).sum()
                    logger.warning(f"Found {nan_count} NaN values in generated samples, imputing with training median")
                    
                    for i in range(gen_samples.shape[1]):
                        nan_mask = np.isnan(gen_samples[:, i])
                        if nan_mask.any():
                            uv_samples[nan_mask, i] = np.median(pv.to_pseudo_obs(train_data)[:,i])
                            gen_samples[nan_mask, i] = np.median(train_data[:, i])
                
                # Compute AIC, BIC, log-likelihood
                loglik = cop.loglik(pv.to_pseudo_obs(test_data))
                aic = cop.aic(pv.to_pseudo_obs(test_data))
                bic = cop.bic(pv.to_pseudo_obs(test_data))

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
        df_gen_out.write_csv(PROCESSED_DATA_DIR / f"generated_data_{dataset}_{copula_type}_seed_{best_seed}.csv")

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
        results_df.write_csv(PROCESSED_DATA_DIR / f"metrics_{dataset}_{copula_type}_all_seeds.csv")
        
        logger.info(f"Saved generated data to: generated_data_{dataset}_{copula_type}_seed_{best_seed}.csv")
        logger.info(f"Saved metrics to: metrics_{dataset}_{copula_type}_all_seeds.csv")

    
    logger.success("Training completed successfully!")


if __name__ == "__main__":
    app()
