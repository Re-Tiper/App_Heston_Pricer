import numpy as np
import matplotlib.pyplot as plt
# For the distribution of the returns:
import seaborn as sns
from scipy.stats import norm, skew, kurtosis, gaussian_kde

########################################################################################################################
# Heston Model Simulation and Plot
########################################################################################################################

def HestonModelSim(s0, T, r, kappa, theta, sigma, rho, v0, n, M):
    """
    Inputs:
     - s0, v0: initial parameters for asset and variance
     - rho   : correlation between asset returns and variance
     - kappa : rate of mean reversion in variance process
     - theta : long-term mean of variance process
     - sigma : vol of vol / volatility of variance process
     - r     : risk-free rate
     - T     : expiry time of simulation
     - n     : number of time steps
     - M     : number of simulations

    Outputs:
    - asset prices over time (numpy array)
    - variance over time (numpy array)
    """
    # Define time interval
    dt = T / n
    # Arrays for storing prices and variances
    S = np.full(shape=(n + 1, M), fill_value=s0)
    v = np.full(shape=(n + 1, M), fill_value=v0)
    # Generate correlated brownian motions
    Z_v = np.random.normal(0, 1, size=(n, M))
    Z_s = rho * Z_v + np.sqrt(1 - rho ** 2) * np.random.normal(0, 1, size=(n, M))

    for i in range(1, n + 1):
        S[i] = S[i - 1] * np.exp((r - 0.5 * v[i - 1]) * dt + np.sqrt(v[i - 1] * dt) * Z_s[i - 1,:])
        v[i] = np.maximum(v[i - 1] + kappa * (theta - v[i - 1]) * dt + sigma * np.sqrt(v[i - 1] * dt) * Z_v[i - 1,:], 0)

    return S, v


def PlotHestonModel(S, var, T, show_subset, show_mean, show_range, plot_prices, plot_vol, plot_returns, plot_dist, plot_dist_comp, plot_kde):
    """
    Plots simulated asset price paths and variance from the Heston model.

    Parameters:
        S (ndarray): Simulated asset paths, shape (n_steps+1, n_sims)
        var (ndarray): Simulated variance paths, shape (n_steps+1, n_sims)
        T (float): Time to maturity
        show_subset (str): One of ["Show all paths", "First 10 Paths", "First 50 Paths", "First 100 Paths"]
        show_mean (bool): Whether to plot only the mean path
        show_range (bool): Whether to plot min-max range as a shaded area
        plot_prices (bool): Whether to plot asset prices
        plot_vol (bool): Whether to plot volatility
        plot_returns (bool): Whether to plot log-returns
        plot_dist (bool): Whether to plot the distribution of log-returns

    Returns:
        dict: Dictionary of matplotlib figures
    """
    # Get the volatility and number of simulations
    v = np.sqrt(var)
    n_sims = S.shape[1]
    time = np.linspace(0, T, S.shape[0])

    # Determine how many paths to show
    if show_subset == "Show all paths":
        paths_to_show = n_sims
    else:
        paths_to_show = int(show_subset.split()[1])
        paths_to_show = min(paths_to_show, n_sims)

    figs = {}

    # === 1. Asset Prices ===
    if plot_prices:
        fig_price = plt.figure(figsize=(16, 8))
        ax = fig_price.add_subplot(1, 1, 1)

        if show_mean:
            mean_price = np.mean(S, axis=1)
            ax.plot(time, mean_price, color='black', lw=2, label='Mean Asset Price')
            if show_range:
                ax.fill_between(time, np.min(S, axis=1), np.max(S, axis=1),
                                color='grey', alpha=0.3, label='Min-Max Range')
        else:
            ax.plot(time, S[:, :paths_to_show], lw=1, alpha=0.7)
            if show_range:
                ax.fill_between(time, np.min(S, axis=1), np.max(S, axis=1),
                                color='grey', alpha=0.3, label='Min-Max Range')

        ax.set_title("Heston Model Asset Prices", fontsize=16)
        ax.set_xlabel("Time", fontsize=16)
        ax.set_ylabel("Asset Price", fontsize=16)
        ax.legend()
        figs['prices'] = fig_price

    # === 2. Volatility Paths ===
    if plot_vol:
        fig_vol = plt.figure(figsize=(16, 8))
        ax = fig_vol.add_subplot(1, 1, 1)

        if show_mean:
            mean_vol = np.mean(v, axis=1)
            ax.plot(time, mean_vol, color='blue', lw=2, label='Mean Volatility')
            if show_range:
                ax.fill_between(time, np.min(v, axis=1), np.max(v, axis=1),
                                color='grey', alpha=0.3, label='Min-Max Range')
        else:
            ax.plot(time, v[:, :paths_to_show], lw=1, alpha=0.7)
            if show_range:
                ax.fill_between(time, np.min(v, axis=1), np.max(v, axis=1),
                                color='grey', alpha=0.3, label='Min-Max Range')

        ax.set_title("Heston Model Volatility Paths", fontsize=16)
        ax.set_xlabel("Time", fontsize=16)
        ax.set_ylabel("Volatility", fontsize=16)
        ax.legend()
        figs['volatility'] = fig_vol

    # === 3. Log-Returns Plot ===
    if plot_returns:
        # Compute the log-returns
        returns = np.diff(np.log(S), axis=0)
        zero_row = np.zeros((1, returns.shape[1]))
        returns = np.vstack([zero_row, returns])  # Shape becomes (n_steps + 1, n_sims)

        fig_ret = plt.figure(figsize=(16, 8))
        ax = fig_ret.add_subplot(1, 1, 1)

        if show_mean:
            mean_ret = np.mean(returns, axis=1)
            ax.plot(time, mean_ret, color='green', lw=2, label='Mean Log Return')
            if show_range:
                ax.fill_between(time, np.min(returns, axis=1), np.max(returns, axis=1),
                                color='grey', alpha=0.3, label='Min-Max Range')
        else:
            ax.plot(time, returns[:, :paths_to_show], lw=1, alpha=0.7)
            if show_range:
                ax.fill_between(time, np.min(returns, axis=1), np.max(returns, axis=1),
                                color='grey', alpha=0.3, label='Min-Max Range')

        ax.set_title("Heston Model Log Returns", fontsize=16)
        ax.set_xlabel("Time", fontsize=16)
        ax.set_ylabel("Log Return", fontsize=16)
        ax.legend()
        figs['returns'] = fig_ret

    # === 4. Distribution of Log-Returns Plot ===
    if plot_dist:
        # Compute log-returns
        returns = np.diff(np.log(S), axis=0)
        zero_row = np.zeros((1, returns.shape[1]))
        returns = np.vstack([zero_row, returns])  # Shape (n_steps + 1, n_sims)
        flat_returns = returns.flatten()

        # Plot: histogram with probabilities
        fig_hist = plt.figure(figsize=(16, 8))
        ax = fig_hist.add_subplot(1, 1, 1)
        counts, bins, _ = ax.hist(flat_returns, bins=200, density=False,
                                  color='skyblue', edgecolor='black', label='Empirical', alpha=0.7)

        # Normalize to probability
        total = flat_returns.shape[0]
        probs = counts / total
        bin_centers = 0.5 * (bins[:-1] + bins[1:])
        ax.clear()  # Clear the original histogram

        # Re-plot as probability bars
        ax.bar(bin_centers, probs, width=(bins[1] - bins[0]),
               color='skyblue', edgecolor='black', alpha=0.7, label='Empirical')

        # Overlay scaled normal PDF
        mu, sigma = flat_returns.mean(), flat_returns.std()
        x_vals = np.linspace(min(flat_returns), max(flat_returns), 1000)
        dx = bins[1] - bins[0]
        ax.plot(x_vals, norm.pdf(x_vals, mu, sigma) * dx, 'r--', lw=2, label='Normal PDF')

        if plot_kde:
            # Overlay KDE
            kde = gaussian_kde(flat_returns)
            kde_probs = kde(x_vals) * dx  # scale to match bar height (probability units)
            ax.plot(x_vals, kde_probs, color='purple', linestyle='-', linewidth=2, label='KDE')

        # Statistics
        sk = skew(flat_returns)
        kurt = kurtosis(flat_returns, fisher=False)  # Use Pearson definition (3 = normal)

        stats_text = (
            f"Mean      = {mu:.5f}\n"
            f"Std Dev   = {sigma:.5f}\n"
            f"Skewness  = {sk:.5f}\n"
            f"Kurtosis  = {kurt:.5f}"
        )
        ax.text(0.95, 0.95, stats_text,
                transform=ax.transAxes,
                fontsize=12,
                verticalalignment='top',
                horizontalalignment='right',
                bbox=dict(facecolor='white', edgecolor='black', boxstyle='round'))

        ax.set_title("Distribution of Daily Log-Returns", fontsize=16)
        ax.set_xlabel("Log Return", fontsize=16)
        ax.set_ylabel("Probability", fontsize=16)
        ax.legend()

        figs['return_dist'] = fig_hist

    # === 5. Compounded Log-Return Distribution ===
    if plot_dist_comp:
        # Compute compounded log returns: log(S_T / S_0) for each path
        log_returns_T = np.log(S[-1, :] / S[0, :])  # shape: (n_sims,)

        # Plot histogram of compounded log-returns
        fig_hist = plt.figure(figsize=(16, 8))
        ax = fig_hist.add_subplot(1, 1, 1)

        # Histogram normalized to form a probability density
        counts, bins, _ = ax.hist(log_returns_T, bins=150, density=False,
                                  color='skyblue', edgecolor='black', alpha=0.7, label='Empirical Histogram')

        # Convert counts to probabilities
        probs = counts / np.sum(counts)
        bin_centers = 0.5 * (bins[1:] + bins[:-1])

        # Clear and re-plot as bar chart using probabilities
        ax.clear()
        ax.bar(bin_centers, probs, width=(bins[1] - bins[0]),
               color='skyblue', edgecolor='black', alpha=0.7, label='Empirical Probabilities')

        # Overlay scaled normal PDF
        mu, sigma = log_returns_T.mean(), log_returns_T.std()
        x_vals = np.linspace(min(log_returns_T), max(log_returns_T), 1000)
        dx = bins[1] - bins[0]
        ax.plot(x_vals, norm.pdf(x_vals, mu, sigma) * dx, 'r--', lw=2, label='Normal PDF')

        # KDE overlay
        if plot_kde:
            kde = gaussian_kde(log_returns_T)
            kde_probs = kde(x_vals) * dx  # approximate probability per bin
            ax.plot(x_vals, kde_probs, color='purple', lw=2, label='KDE (approx. probabilities)')

        # Statistics
        sk = skew(log_returns_T)
        kurt = kurtosis(log_returns_T, fisher=False)
        stats_text = (
            f"Mean      = {mu:.5f}\n"
            f"Std Dev   = {sigma:.5f}\n"
            f"Skewness  = {sk:.5f}\n"
            f"Kurtosis  = {kurt:.5f}"
        )
        ax.text(0.95, 0.95, stats_text,
                transform=ax.transAxes,
                fontsize=12,
                verticalalignment='top',
                horizontalalignment='right',
                bbox=dict(facecolor='white', edgecolor='black', boxstyle='round'))

        ax.set_title("Distribution of Compounded Log-Return", fontsize=16)
        ax.set_xlabel("Log-Return over T", fontsize=16)
        ax.set_ylabel("Probability Density", fontsize=16)
        ax.legend()
        figs['comp_return_dist'] = fig_hist

    return figs

"""
# EXAMPLE USAGE
# Define parameters
s0 = 100.0             # asset price
T = 1.0                # time in years
r = 0.03               # risk-free rate
n = 252                # number of time steps in simulation
M = 100                # number of simulations
# Heston dependent parameters
kappa = 0.5             # rate of mean reversion of variance under risk-neutral dynamics
theta = 0.04            # long-term mean of variance under risk-neutral dynamics
sigma = 0.3             # volatility of volatility
rho = 0.5               # correlation between returns and variances under risk-neutral dynamics
v0 = 0.02               # initial variance under risk-neutral dynamics
S_sim, v_sim = HestonModelSim(
            s0=s0,
            T=T,
            r=r,
            kappa=kappa,
            theta=theta,
            sigma=sigma,
            rho=rho,
            v0=v0,
            n=n,
            M=M
        )
fig = PlotHestonModel(S_sim, v_sim, T, show_subset="First 50 Paths", show_mean=False, show_range=False)
plt.show()
"""
