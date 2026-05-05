# Getting Started

This guide will help you set up the EV Copula Modeling Framework and run your first experiments.

## Prerequisites

- Python 3.12 or higher
- Git
- [uv](https://github.com/astral-sh/uv) package manager (recommended)

## Installation

### 1. Clone the Repository or unzip the repository if downloaded

```bash
git clone https://github.com/kinit-sk/ev-copulas
cd ev-copulas
```

### 2. Install Dependencies

Using uv (recommended):
```bash
# Install all dependencies and create virtual environment
uv sync

# Activate the virtual environment
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

### 3. Verify Installation

```bash
# Test the installation
python -c "import evcopulas; print('Installation successful!')"
```

## Data Setup

### Supported Datasets

The framework supports three EV charging datasets:

1. **Trondheim**: Norwegian EV charging data ([download](https://drive.google.com/file/d/1TUQsq5ttHmmf2gmgsv-VUEXfpacJKTJz/view?usp=sharing))
2. **Dundee**: Scottish EV charging data ([download](https://drive.google.com/file/d/1sPwlcYr3Hr6IIXBtZC-1wCcdNY38zjR3/view?usp=sharing))
3. **Proprietary**: Custom EV charging dataset

### Data Structure

Place your raw data files in the `data/raw/` directory:

```
data/
├── raw/
│   ├── trondheim.csv
│   ├── dundee.csv
│   └── proprietary.csv # your own dataset
├── processed/     # Generated automatically
└── interim/       # Generated automatically
```

### Data Format

Each dataset should contain EV charging session data with columns for:

- Start time/datetime
- Duration or end time
- Energy consumption

If using your own proprietary dataset, it's necessary to change the underlying preprocessing function in the `dataset.py`.

## Data Preprocessing

Before training any model, download the dataset:

1. Download the dataset from the links in [Supported Datasets](#supported-datasets) above
2. Place the CSV file in `data/raw/` (e.g. `data/raw/trondheim.csv`)

```bash
python -m evcopulas.dataset --dataset trondheim
# or
python -m evcopulas.dataset --dataset dundee
```

This produces a processed file in `data/processed/` with three standardized columns: `start_hour_shifted`, `duration`, and `energy`.

## First Steps

### 1. Train a Classical Copula

Start with a simple Gaussian copula:

```bash
python -m evcopulas.copulas.train \
    --dataset trondheim \
    --copulas gaussian \
    --random-seeds 1 2 3
```

### 2. Train CODINE Neural Copula

```bash
python -m evcopulas.codine.train \
    --dataset trondheim \
    --divergence GAN \
    --epochs 1000 \
    --use-early-stopping
```

### 3. Train GMM Networks

```bash
python -m evcopulas.gmmnetwork.train \
    --dataset trondheim \
    --epochs 2000 \
    --use-early-stopping \
    --patience 200
```

### 4. Evaluate Models

```bash
python -m evcopulas.evaluate \
    --datasets trondheim \
    --models gaussian codine gmmnetwork
```

## Understanding the Output

### Generated Files

After training, you'll find these files in `data/processed/`:

- `test_data_{dataset}.csv`: Original test data
- `generated_data_{dataset}_{model}_seed_{seed}.csv`: Synthetic data
- `metrics_{dataset}_{model}_all_seeds.csv`: Evaluation metrics

After evaluating models, you'll find a file `metrics_kdenll.csv` in `data/processed/`.

### Key Metrics

- **Wasserstein Distance**: Lower is better (distribution similarity)
- **Kendall's Tau Difference**: Lower is better (correlation preservation)
- **Tail Dependence**: Measures extreme value dependencies
- **AIC/BIC**: Information criteria for model selection

## Configuration

### Logging

Training logs are automatically saved to the `logs/` directory with timestamps.

### Reproducibility

Set random seeds for reproducible results:

```bash
python -m evcopulas.copulas.train \
    --dataset trondheim \
    --random-seeds 42 123 456
```

### GPU Usage

The framework automatically detects and uses GPU when available for neural models.

## Next Steps

- Explore the [API Reference](../api/config) for detailed documentation of the code
- Read about [Methodology](../methodology/classical-copulas) for theoretical background
