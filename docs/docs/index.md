# EV Copula Modeling Framework

Welcome to the documentation of the framework for simulating and modeling Electric Vehicle (EV) charging sessions using advanced copula methods and neural networks. This repository implements multiple state-of-the-art approaches for learning complex dependency structures in EV charging data. This is a official implementation of the paper [**Capturing Multivariate Dependencies of EV Charging Events: From Parametric Copulas to Neural Density Estimation**](https://arxiv.org/abs/2603.29554).

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

## Quick Navigation

- [Methodology Overview](methodology/classical-copulas) - Theoretical background
- [Getting Started](getting-started) - Installation and setup
- [API Reference](api/config) - Complete code documentation

## Supported Datasets

- **Trondheim**: Norwegian residential charging EV charging data with private user sessions
- **Dundee**: Scottish public charger EV charging data focusing on rapid charging
- **Proprietary**: Custom household EV charging dataset from Slovakia (not available for download)

## Installation

```bash
# Clone the repository or unzip the repository if downloaded
git clone https://github.com/kinit-sk/ev-copulas
cd ev-copulas

# Install dependencies using uv
uv sync
```

## Quick Start

```bash
# Train classical copulas
python -m evcopulas.copulas.train --dataset trondheim --copulas gaussian

# Train CODINE neural copula  
python -m evcopulas.codine.train --dataset trondheim --divergence GAN

# Train GMM networks
python -m evcopulas.gmmnetwork.train --dataset trondheim --use-early-stopping

# Evaluate all models
python -m evcopulas.evaluate --datasets trondheim --models gaussian codine gmmnetwork
```

## Research Impact

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

## Contact
[martin.vyboh@kinit.sk](mailto:martin.vyboh@kinit.sk)