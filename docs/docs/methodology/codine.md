# CODINE: Neural Copula Density Estimation

CODINE (COpula DensIty Neural Estimation) is a neural approach to copula density estimation using adversarial training with discriminator networks.

## Overview

CODINE learns copula densities by training a discriminator network to distinguish between empirical copula data and uniform random samples. This approach bypasses the need for explicit parametric assumptions while maintaining the theoretical foundation of copula theory.

## Divergence Measures

### Kullback-Leibler (KL) Divergence

- **Objective**: Minimize KL divergence between estimated and true density
- **Discriminator Output**: Direct density estimate
- **Loss Function**: -E[log D(u)] + E[D(v)] where u ~ data, v ~ uniform

### Generative Adversarial Network (GAN)

- **Objective**: Adversarial game between discriminator and uniform generator
- **Discriminator Output**: Probability of being real data
- **Loss Function**: Binary cross-entropy with sigmoid activation

### Hellinger Distance (HD)

- **Objective**: Minimize Hellinger distance between distributions
- **Discriminator Output**: Square root of density ratio
- **Loss Function**: Wasserstein and reciprocal losses combined

## Architecture

### Discriminator Network

```python
class Discriminator(nn.Module):
    def __init__(self, latent_dim, divergence):
        super().__init__()
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
```

### Key Design Choices

- **LeakyReLU Activations**: Prevent gradient vanishing
- **Multiple Hidden Layers**: Capture complex dependencies
- **Divergence-Specific Output**: Sigmoid for GAN, Softplus for KL/HD
- **Batch Processing**: Efficient GPU utilization

## Advantages

- **Non-parametric**: No assumptions about copula family
- **High-dimensional**: Scales to multiple variables
- **Complex Dependencies**: Captures non-linear relationships
- **Multiple Divergences**: Different optimization objectives
- **Early Stopping**: Prevents overfitting

## Limitations

- **Training Time**: Longer than classical copulas
- **GPU Dependency**: Benefits significantly from acceleration

## References

- Letizia, N. A., Novello, N., & Tonello, A. M. (2025). Copula Density Neural Estimation. IEEE Transactions on Neural Networks and Learning Systems, 36(10), 19452–19459. https://doi.org/10.1109/TNNLS.2025.3585755