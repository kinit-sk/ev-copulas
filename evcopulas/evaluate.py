"""Evaluation module for copula models using KDE-based negative log-likelihood.

This module provides functionality to evaluate different copula models by calculating
the negative log-likelihood using Kernel Density Estimation (KDE) on generated data
compared to test data.
"""

import glob

import numpy as np
import polars as pl
import typer
from loguru import logger
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KernelDensity
from typing_extensions import Annotated

from evcopulas.config import PROCESSED_DATA_DIR
from evcopulas.utils import set_seed

app = typer.Typer()


def calculate_kde_nll(gen_data, test_data):
    """
    Calculate KDE-based negative log-likelihood for model evaluation.
    
    Fits a Kernel Density Estimator to generated data and evaluates the
    negative log-likelihood of test data under this distribution.
    
    Args:
        gen_data (array-like): Generated data samples to fit KDE on.
        test_data (array-like): Test data to evaluate likelihood on.
        
    Returns:
        float: Negative log-likelihood score (lower is better).
    """
    # 1. Optimize Bandwidth (Optional but recommended for fairness)
    # This ensures the KDE isn't 'too blurry' or 'too jagged'
    params = {'bandwidth': np.logspace(-2, 1, 20)}
    grid = GridSearchCV(KernelDensity(kernel='gaussian'), params, cv=5)
    grid.fit(gen_data)
    best_bw = grid.best_params_['bandwidth']
    
    # 2. Fit the final KDE
    kde = KernelDensity(kernel='gaussian', bandwidth=best_bw)
    kde.fit(gen_data)
    
    # 3. Score the test data (returns log-likelihood)
    log_pdf = kde.score_samples(test_data)
    
    # 4. Return Negative Log-Likelihood
    return -np.mean(log_pdf)

@app.command()
def evaluate_all_models(
    datasets: Annotated[
        list[str], typer.Option(help="Datasets to evaluate")
    ] = ["proprietary", "trondheim", "dundee"], 
    models: Annotated[
        list[str], typer.Option(help="Models to evaluate")
    ] = ["clayton", "codine", "frank", "gaussian", "gumbel", "student", "vine", "gmmnetwork"],
    random_seed: Annotated[
        int, typer.Option(help="Random seed")
    ] = 42,
    ):
    """
    Evaluate all specified models using KDE-based negative log-likelihood.
    
    This function evaluates multiple copula models across different datasets by:

    1. Loading test data for each dataset
    2. Loading generated data for each model
    3. Calculating KDE-based negative log-likelihood scores
    4. Saving results to a CSV file
    
    Args:
        datasets: List of dataset names to evaluate on.
        models: List of model names to evaluate.
        random_seed: Random seed for reproducibility.
        
    Returns:
        pl.DataFrame: DataFrame containing evaluation results with columns:
        
            - Dataset: Name of the dataset
            - model: Name of the model
            - KDE_NLL: Negative log-likelihood score
    """

    set_seed(random_seed)

    results = []
    for dataset in datasets:
        logger.info(f"Calculating NLL for {dataset}...")
        real_data = pl.read_csv(PROCESSED_DATA_DIR / f"test_data_{dataset}.csv")
        
        for model in models:
            logger.info(f"Calculating NLL for {model}...")
            
            # Find the generated data file with glob pattern
            pattern = str(PROCESSED_DATA_DIR / f"generated_data_{dataset}_{model}_seed_*.csv")
            files = glob.glob(pattern)
            
            if not files:
                logger.warning(f"No files found for pattern: {pattern}")
                continue
                
            # Use the first matching file
            gen_data = pl.read_csv(files[0])[:,0:3]
            
            nll = calculate_kde_nll(gen_data.to_numpy(), real_data.to_numpy())
            results.append({
                'Dataset': dataset,
                'model': model,
                'KDE_NLL': nll
            })

    results = pl.DataFrame(results)
    results.write_csv(PROCESSED_DATA_DIR / "metrics_kdenll.csv")
    logger.success(f"Calculation of NLL from KDE done.")

    return results


if __name__ == "__main__":
    app()