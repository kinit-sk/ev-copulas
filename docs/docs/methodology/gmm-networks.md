# GMM Networks: Neural Conditional Distribution Modeling

GMM Networks use neural networks to predict Gaussian Mixture Model parameters for conditional distribution modeling.

## Overview

GMM Networks learn conditional distributions by training separate neural networks for each dimension. Each network predicts the parameters of a Gaussian Mixture Model that represents the conditional distribution of one variable given all others.

## Theoretical Foundation

### Gaussian Mixture Models

Each conditional distribution is represented as:

- **Mixture Weights**: π_k(x_{-i}) with Σπ_k = 1
- **Component Means**: μ_k(x_{-i}) ∈ ℝ  
- **Component Variances**: σ_k²(x_{-i}) > 0

## Architecture

### Network Structure

```python
class GMM(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, n_components, dim):
        super().__init__()
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.layers = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim) 
            for _ in range(num_layers-1)
        ])
        self.linear3 = nn.Linear(hidden_dim, 3 * n_components)
        self.batchnorms = nn.ModuleList([
            nn.BatchNorm1d(hidden_dim) 
            for _ in range(num_layers)
        ])
```

### Output Processing

The network outputs 3K parameters for K mixture components:

### Activation Functions

- **ReLU**: Hidden layer activations with batch normalization
- **Softplus**: Ensures positive standard deviations
- **Softmax**: Normalizes mixture weights to sum to 1

## Advantages

- **Non-parametric**: No assumptions about conditional distributions
- **Adaptive Complexity**: Mixture components adapt to data complexity

## Limitations

- **Multiple Networks**: d networks for d-dimensional data
- **Training Time**: Longer than single-network approaches
- **Memory Requirements**: Scales with number of dimensions

## References

- Li, Z., Bian, Z., Chen, Z., Ozbay, K., & Zhong, M. (2024). Synthesis of electric vehicle charging data: A real-world data-driven approach. Communications in Transportation Research, 4, 100128. https://doi.org/10.1016/j.commtr.2024.100128
