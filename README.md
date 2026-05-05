# EV Copula Modeling Framework

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

A comprehensive framework for simulating and modeling Electric Vehicle (EV) charging sessions using advanced copula methods and neural networks. This repository implements multiple state-of-the-art approaches for learning complex dependency structures in EV charging data. This is a official implementation of the paper [**Capturing Multivariate Dependencies of EV Charging Events: From Parametric Copulas to Neural Density Estimation**](https://arxiv.org/abs/2603.29554).

## Overview

This repository provides implementations of several copula-based methods for EV charging data synthesis:

- **Classical Copulas**: Gaussian, Clayton, Frank, Gumbel, Student-t, and Vine copulas
- **CODINE**: Neural copula density estimation using adversarial training
- **GMM Networks**: Gaussian Mixture Model neural networks for conditional distributions
- **Comprehensive Evaluation**: Statistical metrics, tail dependence analysis, and practical EV applications

## Key Features

- 🔬 **Multiple Modeling Approaches**: Classical copulas, neural copulas, and GMM networks
- 📊 **Comprehensive Evaluation**: Wasserstein distance, Kendall's tau, tail dependence, information criteria
- 🚗 **EV-Specific Analysis**: Daily load curve modeling and practical charging applications
- 🎯 **Robust Training**: Multi-seed evaluation, early stopping, and validation-based model selection
- 📈 **Visualization Tools**: Marginal density plots, load curve comparisons, and statistical visualizations

## Supported Datasets

- **Trondheim**: Norwegian residential charging EV charging data with private user sessions ([download](https://drive.google.com/file/d/1TUQsq5ttHmmf2gmgsv-VUEXfpacJKTJz/view?usp=sharing))
- **Dundee**: Scottish public charger EV charging data focusing on rapid charging ([download](https://drive.google.com/file/d/1sPwlcYr3Hr6IIXBtZC-1wCcdNY38zjR3/view?usp=sharing))
- **Proprietary**: Custom household EV charging dataset from Slovakia (not available for download)

## Quick Start

### Installation

```bash
# Clone the repository or unzip the repository if downloaded
git clone https://github.com/kinit-sk/ev-copulas
cd ev-copulas

# Install dependencies using uv (reads from pyproject.toml)
uv sync
```

### Training Models

```bash
# Train classical copulas
python -m evcopulas.copulas.train --dataset trondheim --copulas gaussian --copulas vine

# Train CODINE neural copula
python -m evcopulas.codine.train --dataset trondheim --divergence GAN --epochs 5000

# Train GMM networks
python -m evcopulas.gmmnetwork.train --dataset trondheim --use-early-stopping
```

### Evaluation

```bash
# Evaluate all models using KDE-based negative log-likelihood
python -m evcopulas.evaluate --datasets trondheim --models gaussian codine gmmnetwork
```

## Project Structure

```
├── LICENSE                 <- Open-source license
├── Makefile               <- Convenience commands for data processing and training
├── README.md              <- Project documentation
├── pyproject.toml         <- Package configuration, metadata, and dependencies
├── setup.cfg              <- Tool configurations (flake8, etc.)
│
├── data/                  <- Data storage with organized subdirectories
│   ├── raw/              <- Original, immutable datasets
│   ├── interim/          <- Intermediate processed data
│   ├── processed/        <- Final datasets and model outputs
│   └── external/         <- Third-party data sources
│
├── docs/                  <- Documentation (MkDocs project)
├── notebooks/             <- Jupyter notebooks for analysis and exploration
├── references/            <- Data dictionaries and documentation
├── reports/               <- Generated analysis and figures
│   └── figures/          <- Visualization outputs
│
├── models/                <- Trained models and predictions
├── logs/                  <- Training and execution logs
│
└── evcopulas/             <- Main source code package
    ├── __init__.py        <- Package initialization
    ├── config.py          <- Configuration and paths
    ├── dataset.py         <- Data preprocessing utilities
    ├── evaluate.py        <- Model evaluation framework
    ├── metrics.py         <- Statistical evaluation metrics
    ├── plots.py           <- Visualization functions
    ├── utils.py           <- General utility functions
    │
    ├── copulas/           <- Classical copula implementations
    │   └── train.py       <- Training script for classical copulas
    │
    ├── codine/            <- CODINE neural copula implementation
    │   └── train.py       <- CODINE training and evaluation
    │
    └── gmmnetwork/        <- GMM neural network implementation
        └── train.py       <- GMM network training and evaluation
```

## Methodology

### Classical Copulas
Implements six copula families with robust parameter estimation:
- **Gaussian**: Linear correlation structure
- **Archimedean**: Clayton (lower tail), Frank (symmetric), Gumbel (upper tail)
- **Student-t**: Symmetric tail dependence
- **Vine**: Flexible pair-copula constructions

### CODINE (Neural Copulas)
Neural copula density estimation using discriminator networks:
- Multiple divergence measures (KL, GAN, Hellinger Distance)
- Adversarial training for density estimation
- Gibbs sampling for synthetic data generation

### GMM Networks
Neural networks predicting Gaussian Mixture Model parameters:
- Conditional distribution modeling
- Batch normalization and numerical stability
- Early stopping with multiple validation metrics

## Evaluation Framework

Comprehensive statistical evaluation including:

- **Distribution Similarity**: Wasserstein distance
- **Correlation Structure**: Kendall's tau preservation
- **Extreme Values**: Upper and lower tail dependence coefficients
- **Model Selection**: AIC, BIC information criteria
- **Generalization**: Overfitting assessment metrics
- **Practical Applications**: Daily load curve analysis for EV infrastructure

## Configuration

The framework uses a centralized configuration system:

- **Paths**: Automatic project structure detection
- **Logging**: Structured logging with tqdm integration
- **Reproducibility**: Seed management across NumPy and PyTorch
- **Device Management**: Automatic GPU/CPU detection

## Documentation

Documentation is available with:

- **API Documentation**: Auto-generated from docstrings using MkDocs
- **Methodology**: Explanations of each modeling approach

### Local Documentation Setup

After installing dependencies with `uv sync`, serve the docs locally:

```bash
# change the directory to docs
cd docs

# Serve with live reload (recommended during development)
mkdocs serve
```

Then open [http://127.0.0.1:8000](http://127.0.0.1:8000) in your browser.

To build a static site:

```bash
mkdocs build
```

The output will be in the `site/` directory.

## Research Applications

This framework supports research in:

- **EV Infrastructure Planning**: Load forecasting and capacity planning
- **Synthetic Data Generation**: Privacy-preserving EV data synthesis
- **Dependency Modeling**: Complex multivariate relationships in time series
- **Copula Theory**: Comparative analysis of classical and neural approaches

## Citation

If you use this framework, or any findings from the paper in your research, please cite:

```bibtex
@misc{Vyboh2026EVCopulas,
      title={Capturing Multivariate Dependencies of EV Charging Events: From Parametric Copulas to Neural Density Estimation}, 
      author={Martin Výboh and Gabriela Grmanová},
      year={2026},
      eprint={2603.29554},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2603.29554}, 
}
```

## References

- **CODINE**: Letizia, N. A., Novello, N., & Tonello, A. M. (2025). Copula Density Neural Estimation. IEEE Transactions on Neural Networks and Learning Systems.
- **GMM Network**: Li, Z., Bian, Z., Chen, Z., Ozbay, K., & Zhong, M. (2024). Synthesis of electric vehicle charging data: A real-world data-driven approach. Communications in Transportation Research.

## Contact
[martin.vyboh@kinit.sk](mailto:martin.vyboh@kinit.sk)

--------

