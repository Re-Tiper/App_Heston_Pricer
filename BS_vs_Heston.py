import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt


def BSM_pricer(s0, strike, T, r, sigma, option_type="call"):
    """
    Calculate the price of vanilla European options using BSM formula
    """
    d1 = (np.log(s0 / strike) + (r + sigma**2 / 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    price = s0 * norm.cdf(d1) - strike * np.exp(-r * T) * norm.cdf(d2)
    price = price if option_type == "call" else price - s0 + strike * np.exp(-r * T)

    return price


def plot_Heston_vs_BSM(heston_pricer_func, s0, strike, T, r,
                       kappa, theta, sigma, rho, v0,
                       param_name, param_values,
                       option_type="call"):
    """
    Plot the difference between Heston and BSM prices for varying s0 values.

    Parameters:
        heston_pricer_func (function): Function that prices an option using the Heston model.
        s0 (float): Current asset price.
        strike (float): Strike price of the option.
        T (float): Time to maturity.
        r (float): Risk-free interest rate.
        kappa, theta, sigma, rho, v0 (float): Heston model parameters.
        param_name (str): Name of the Heston parameter to vary.
        param_values (list[float]): List of values to use for the parameter being varied (contains 1, 2 or 3 values).
        option_type (str): "call" or "put"

    Returns:
        matplotlib.Figure
    """
    s0_vals = np.linspace(s0 * (1 - 0.4), s0 * (1 + 0.4), 100)  # 40% range around s0
    # Set the BSM vol to match the expected variance of returns under the Heston model
    avg_var = theta + (v0 - theta) * (1 - np.exp(-kappa * T)) / (kappa * T)
    #avg_var = v0 * np.exp(-kappa * T) + theta * (1 - np.exp(-kappa * T))
    sigma_bsm = np.sqrt(avg_var)

    fig = plt.figure(figsize=(16, 8))
    ax = fig.add_subplot(1, 1, 1)

    base_params = {
        "r": r, "kappa": kappa, "theta": theta,
        "sigma": sigma, "rho": rho, "v0": v0
    }

    for val in param_values:
        params = base_params.copy()
        params[param_name] = val

        diffs = [
            heston_pricer_func(params, s0_i, strike, T, option_type)
            - BSM_pricer(s0_i, strike, T, r, sigma_bsm, option_type)
            for s0_i in s0_vals
        ]

        ax.plot(s0_vals, diffs, label=f"{param_name} = {val:.3f}")

    ax.axhline(0, color='gray', linestyle='--', linewidth=1)
    ax.set_title(f"{option_type.capitalize()} Option    ", fontsize=16)
    ax.set_xlabel(r"Initial Asset Price $S_0$", fontsize=16)
    ax.set_ylabel("Price Difference (Heston - BSM)", fontsize=16)
    ax.legend()
    ax.grid(True)

    return fig