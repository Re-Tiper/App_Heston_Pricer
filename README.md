# Heston Option Pricer App
---
Check it out at: https://heston-pricer.streamlit.app

An interactive Streamlit-based application for pricing European and American options under the **Heston stochastic volatility model**, visualizing simulation paths, comparing pricing methods, and analyzing option pricing behavior across different parameter values.

## Features

### Option Pricing Methods
- **Fast Fourier Transform method** – Fast and efficient pricing using the Carr-Madan approach.
- **Closed-Form solution** – Semi-analytical solution via numerical integration, provided by Heston.
- **Least Squares Monte Carlo** – A simulation heavy approach for pricing American options.

### Heatmap Visualization
- Generate 2D heatmaps to explore how option prices respond to changes in two selected parameters (e.g. strike vs maturity, or volatility of volatility vs correlation).

### Simulation
- Simulate asset price paths and volatility paths using Euler-Maruyama discretization.
- Simulate the distribution of daily and terminal returns of the underlying asset.

### Heston vs Black-Scholes Comparison
- Plot the difference between Heston and BSM prices (Heston - BSM) as a function of initial asset price.
- Select different values for Heston parameters and compare results.

