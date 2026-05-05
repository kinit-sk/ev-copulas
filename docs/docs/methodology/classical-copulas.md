# Classical Copulas

Classical copulas provide a mathematical framework for modeling dependencies between variables while separating the marginal distributions from the dependence structure.

## Overview

Copulas are functions that link multivariate distribution functions to their one-dimensional marginal distribution functions. For EV charging data, copulas help capture the complex relationships between charging session attributes like start time, duration, and energy consumption.

## Supported Copula Families

### Elliptical Copulas

#### Gaussian Copula

- **Structure**: Multivariate normal dependence
- **Parameters**: Correlation matrix
- **Characteristics**: Symmetric, no tail dependence
- **Use Case**: Linear relationships between variables

#### Student-t Copula  

- **Structure**: Multivariate t-distribution dependence
- **Parameters**: Correlation matrix and degrees of freedom
- **Characteristics**: Symmetric tail dependence
- **Use Case**: Heavy-tailed dependencies with extreme co-movements

### Archimedean Copulas

#### Clayton Copula

- **Structure**: Lower tail dependence
- **Parameters**: Single parameter θ > 0
- **Characteristics**: Strong dependence in lower tail, weak in upper tail
- **Use Case**: Modeling simultaneous low values (e.g., short charging sessions)

#### Frank Copula

- **Structure**: Symmetric dependence
- **Parameters**: Single parameter θ ∈ ℝ
- **Characteristics**: No tail dependence, symmetric
- **Use Case**: Moderate, symmetric dependencies

#### Gumbel Copula

- **Structure**: Upper tail dependence
- **Parameters**: Single parameter θ ≥ 1
- **Characteristics**: Strong dependence in upper tail, weak in lower tail
- **Use Case**: Modeling simultaneous high values (e.g., long charging sessions)

### Vine Copulas

#### Pair-Copula Constructions

- **Structure**: Flexible decomposition using bivariate copulas
- **Parameters**: Multiple bivariate copula parameters
- **Characteristics**: Can model complex, asymmetric dependencies
- **Use Case**: High-dimensional data with varying pairwise dependencies

## Limitations

- **Parametric Assumptions**: Limited to specific functional forms
- **High Dimensions**: Some families limited to moderate dimensions
- **Static Dependencies**: Cannot capture time-varying relationships

## References

- Aas, K., Czado, C., Frigessi, A., & Bakken, H. (2009). Pair-copula constructions of multiple dependence. Insurance: Mathematics and Economics, 44(2), 182–198. https://doi.org/10.1016/j.insmatheco.2007.02.001
- Einolander, J., & Lahdelma, R. (2022). Multivariate copula procedure for electric vehicle charging event simulation. Energy, 238, 121718. https://doi.org/10.1016/j.energy.2021.121718
