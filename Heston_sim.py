import numpy as np
import matplotlib.pyplot as plt

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


def PlotHestonModel(S, var, T, show_subset, show_mean, show_range, plot_returns):
    """
    Plots simulated asset price paths and variance from the Heston model.

    Parameters:
        S (ndarray): Simulated asset paths, shape (n_steps+1, n_sims)
        var (ndarray): Simulated variance paths, shape (n_steps+1, n_sims)
        T (float): Time to maturity
        show_subset (str): One of ["Show all paths", "First 10 Paths", "First 50 Paths", "First 100 Paths"]
        show_mean (bool): Whether to plot only the mean path
        show_range (bool): Whether to plot min-max range as a shaded area
        plot_returns (bool): Whether to plot returns instead of asset prices

    Returns:
        matplotlib.figure
    """
    # Get the volatility and number of simulations
    v = np.sqrt(var)
    n_sims = S.shape[1]
    time = np.linspace(0, T, S.shape[0])

    if plot_returns:
        returns = np.diff(np.log(S), axis=0)
        zero_row = np.zeros((1, returns.shape[1]))
        S = np.vstack([zero_row, returns])  # Shape becomes (n_steps + 1, n_sims)

    # Determine how many paths to show
    if show_subset == "Show all paths":
        paths_to_show = n_sims
    else:
        paths_to_show = int(show_subset.split()[1])
        paths_to_show = min(paths_to_show, n_sims)

    fig = plt.figure(figsize=(16, 8))
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2)

    if show_mean:
        # Only plot the mean asset price and volatility paths
        mean_price = np.mean(S, axis=1)
        mean_vol = np.mean(v, axis=1)

        ax1.plot(time, mean_price, color='black', lw=2, label='Mean Asset Price')
        ax2.plot(time, mean_vol, color='blue', lw=2, label='Mean Volatility')

        ax1.legend()
        ax2.legend()

        if show_range:
            min_path = np.min(S, axis=1)
            max_path = np.max(S, axis=1)
            ax1.fill_between(time, min_path, max_path, color='grey', alpha=0.3, label='Min-Max Range')

            min_var = np.min(v, axis=1)
            max_var = np.max(v, axis=1)
            ax2.fill_between(time, min_var, max_var, color='grey', alpha=0.3, label='Min-Max Range')


    else:
        # Plot subset of asset price paths
        ax1.plot(time, S[:, :paths_to_show], lw=1, alpha=0.7)

        # Plot min-max range if requested
        if show_range:
            min_path = np.min(S, axis=1)
            max_path = np.max(S, axis=1)
            ax1.fill_between(time, min_path, max_path, color='grey', alpha=0.3, label='Min-Max Range')

            min_var = np.min(v, axis=1)
            max_var = np.max(v, axis=1)
            ax2.fill_between(time, min_var, max_var, color='grey', alpha=0.3, label='Min-Max Range')

        # Plot all volatility paths
        ax2.plot(time, v[:, :paths_to_show], lw=1, alpha=0.7)

        if show_range:
            ax1.legend()

    ax1.set_title(f'Heston Model {"Returns" if plot_returns else "Asset Prices"}', fontsize=16)
    ax1.set_xlabel('Time', fontsize=16)
    ax1.set_ylabel('Returns' if plot_returns else 'Asset Prices', fontsize=16)

    ax2.set_title('Heston Model Volatility Paths', fontsize=16)
    ax2.set_xlabel('Time', fontsize=16)
    ax2.set_ylabel('Volatility', fontsize=16)

    return fig

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
