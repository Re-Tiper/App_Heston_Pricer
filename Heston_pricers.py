import numpy as np
import time
from scipy.integrate import quad

"""
Heston model prices using Fast Fourier Transform
"""

def HestonCallFFT(s0, strike, T, r, kappa, theta, sigma, rho, v0):
    """
    Computes the European call option price using the Heston model and Fast Fourier Transform.
    Inputs:
     - s0    : current asset price
     - strike: strike price of the option
     - T     : expiry time of simulation
     - r     : risk-free rate
     - kappa : rate of mean reversion in variance process
     - theta : long-term mean of variance process
     - sigma : vol of vol / volatility of variance process
     - rho   : correlation between asset returns and variance
     - v0    : initial variance

     Outputs:
     - Call price
    """
    x0 = np.log(s0)  # Initial log price
    alpha = 1.25
    N = 4096
    c = 600
    eta = c / N
    b = np.pi / eta
    u = np.arange(0, N) * eta
    lambd = 2 * b / N
    position = int(np.round((np.log(strike) + b) / lambd))  # Position of call value in FFT

    # Complex numbers for characteristic function
    v = u - (alpha + 1) * 1j
    zeta = -0.5 * (v ** 2 + 1j * v)
    gamma = kappa - rho * sigma * v * 1j
    PHI = np.sqrt(gamma ** 2 - 2 * sigma ** 2 * zeta)
    A = 1j * v * (x0 + r * T)
    B = v0 * ((2 * zeta * (1 - np.exp(-PHI * T))) /
              (2 * PHI - (PHI - gamma) * (1 - np.exp(-PHI * T))))
    C = -(kappa * theta / sigma ** 2) * (
            2 * np.log((2 * PHI - (PHI - gamma) * (1 - np.exp(-PHI * T))) / (2 * PHI)) +
            (PHI - gamma) * T
    )

    # Characteristic function
    char_func = np.exp(A + B + C)

    # Modified characteristic function
    modified_char_func = (char_func * np.exp(-r * T) /
                          (alpha ** 2 + alpha - u ** 2 + 1j * (2 * alpha + 1) * u))

    # Simpson weights for integration
    simpson_w = (1 / 3) * (3 + (-1) ** np.arange(1, N + 1) - np.append(1, np.zeros(N - 1)))

    # FFT computation
    fft_func = np.exp(1j * b * u) * modified_char_func * eta * simpson_w
    payoff = np.real(np.fft.fft(fft_func))

    # Extract call value
    call_value_m = np.exp(-np.log(strike) * alpha) * payoff / np.pi
    call_value = call_value_m[position]

    return call_value


def HestonPutFFT(s0, strike, T, r, kappa, theta, sigma, rho, v0):
    """
    Computes the European put option price using put-call parity and the Heston FFT call price.
    """
    call_price = HestonCallFFT(s0, strike, T, r, kappa, theta, sigma, rho, v0)
    put_price = call_price - s0 + strike * np.exp(-r * T)
    return put_price


def HestonFFT(s0, strike, T, r, kappa, theta, sigma, rho, v0, option_type="call"):
    call_price = HestonCallFFT(s0, strike, T, r, kappa, theta, sigma, rho, v0)
    if option_type == "call":
        return call_price
    elif option_type == "put":
        return call_price - s0 + strike * np.exp(-r * T)
    else:
        raise ValueError("option_type must be 'call' or 'put'")


"""
Heston model prices using semi-closed form formulas
"""

def HestonCallQuad(s0, strike, T, r, kappa, theta, sigma, rho, v0):
    """Computes the price of a European call option using the Heston model."""
    call = s0 * HestonP(s0, strike, T, r, kappa, theta, sigma, rho, v0, 1) \
         - strike * np.exp(-r * T) * HestonP(s0, strike, T, r, kappa, theta, sigma, rho, v0, 2)
    return call


def HestonP(s0, strike, T, r, kappa, theta, sigma, rho, v0, option_type):
    """Computes the Heston characteristic function using numerical integration."""
    integral_result = quad(HestonPIntegrand, 0, 100,
                           args=(s0, strike, T, r, kappa, theta, sigma, rho, v0, option_type))[0]
    return 0.5 + (1 / np.pi) * integral_result


def HestonPIntegrand(phi, s0, strike, T, r, kappa, theta, sigma, rho, v0, option_type):
    """Evaluates the integrand for the Heston characteristic function."""
    return np.real(np.exp(-1j * phi * np.log(strike)) *
                   HestonCharfun(phi, s0, strike, T, r, kappa, theta, sigma, rho, v0, option_type) / (1j * phi))


def HestonCharfun(phi, s0, strike, T, r, kappa, theta, sigma, rho, v0, option_type):
    """Computes the Heston characteristic function."""
    if option_type == 1:
        u = 0.5
        b = kappa - rho * sigma
    else:
        u = -0.5
        b = kappa

    a = kappa * theta
    x = np.log(s0)
    d = np.sqrt((rho * sigma * phi * 1j - b)**2 - sigma**2 * (2 * u * phi * 1j - phi**2))
    g = (b - rho * sigma * phi * 1j + d) / (b - rho * sigma * phi * 1j - d)

    C = r * phi * 1j * T + (a / sigma**2) * ((b - rho * sigma * phi * 1j + d) * T - 2 * np.log((1 - g * np.exp(d * T)) / (1 - g)))
    D = (b - rho * sigma * phi * 1j + d) / sigma**2 * ((1 - np.exp(d * T)) / (1 - g * np.exp(d * T)))

    return np.exp(C + D * v0 + 1j * phi * x)


def HestonPutQuad(s0, strike, T, r, kappa, theta, sigma, rho, v0):
    """
    Computes the price of a European put option using put-call parity
    and the Heston call price via quadrature.
    """
    call_price = HestonCallQuad(s0, strike, T, r, kappa, theta, sigma, rho, v0)
    put_price = call_price - s0 + strike * np.exp(-r * T)
    return put_price


def HestonQuad(s0, strike, T, r, kappa, theta, sigma, rho, v0, option_type="call"):
    """
    Unified Heston quadrature-based option pricer.
    """
    call_price = HestonCallQuad(s0, strike, T, r, kappa, theta, sigma, rho, v0)
    if option_type.lower() == "call":
        return call_price
    elif option_type.lower() == "put":
        return call_price - s0 + strike * np.exp(-r * T)
    else:
        raise ValueError("option_type must be 'call' or 'put'")


"""
Pricer for american option
"""

from Heston_sim import HestonModelSim
#from numpy.polynomial.laguerre import lagval

def basis_functions(X, k):
    """
    Generate basis functions for the regression.

    Parameters:
    X : array
        Asset prices
    k : int
        Number of basis functions
    Returns:
    numpy.ndarray
        Basis functions evaluated at X
    """
    if k == 1:
        A = np.vstack([np.ones(X.shape), 1 - X]).T
    elif k == 2:
        A = np.vstack([np.ones(X.shape), 1 - X, 0.5 * (2 - 4 * X + X**2)]).T
    elif k == 3:
        A = np.vstack([np.ones(X.shape), 1 - X, 0.5 * (2 - 4 * X + X**2),
                       1/6 * (6 - 18 * X + 9 * X**2 - X**3)]).T
    elif k == 4:
        A = np.vstack([np.ones(X.shape), 1 - X, 0.5 * (2 - 4 * X + X**2),
                       1/6 * (6 - 18 * X + 9 * X**2 - X**3),
                       1/24 * (24 - 96 * X + 72 * X**2 - 16 * X**3 + X**4)]).T
    elif k == 5:
        A = np.vstack([np.ones(X.shape), 1 - X, 0.5 * (2 - 4 * X + X**2),
                       1/6 * (6 - 18 * X + 9 * X**2 - X**3),
                       1/24 * (24 - 96 * X + 72 * X**2 - 16 * X**3 + X**4),
                       1/120 * (120 - 600 * X + 600 * X**2 - 200 * X**3 + 25 * X**4 - X**5)]).T
    else:
        raise ValueError('Too many basis functions requested')
    return A


def LSM_American(s0, strike, T, r, kappa, theta, sigma, rho, v0, option_type="call", num_basis=3):
    """
    Prices an American option using Least Squares Monte Carlo (LSM),
    with custom basis functions.

    Parameters:
    Inputs:
     - s0    : current asset price
     - strike: strike price of the option
     - T     : expiry time of simulation
     - r     : risk-free rate
     - kappa : rate of mean reversion in variance process
     - theta : long-term mean of variance process
     - sigma : vol of vol / volatility of variance process
     - rho   : correlation between asset returns and variance
     - v0    : initial variance
     -  option_type : str Either "call" or "put"
     -  num_basis : int
        Number of basis functions used in regression

    Returns:
    float
        Estimated American option price
    """

    # Simulated asset price paths (shape: [n+1, M])
    S, _ = HestonModelSim(
        s0=s0,
        T=T,
        r=r,
        kappa=kappa,
        theta=theta,
        sigma=sigma,
        rho=rho,
        v0=v0,
        n=int(T*252), # number of time steps in simulation (252 trading days in year
        M=10000       # number of simulations
    )

    if option_type not in ["call", "put"]:
        raise ValueError("option_type must be either 'call' or 'put'")

    n = S.shape[0] - 1
    M = S.shape[1]
    dt = T / n

    if option_type == "call":
        payoff = np.maximum(S[-1] - strike, 0).astype(float)
    else:
        payoff = np.maximum(strike - S[-1], 0).astype(float)

    for t in range(n, 0, -1):
        if option_type == "call":
            itm = np.where(S[t] > strike)[0]
        else:
            itm = np.where(S[t] < strike)[0]

        if len(itm) == 0:
            continue

        X = S[t, itm]
        Y = np.exp(-r * dt) * payoff[itm]

        A = basis_functions(X, num_basis)
        coeffs = np.linalg.lstsq(A, Y, rcond=None)[0]
        continuation_value = A @ coeffs

        if option_type == "call":
            exercise_value = X - strike
        else:
            exercise_value = strike - X

        exercise_now = exercise_value > continuation_value
        ex_indices = itm[exercise_now]

        payoff[ex_indices] = exercise_value[exercise_now]

        hold_indices = np.setdiff1d(np.arange(M), ex_indices)
        payoff[hold_indices] *= np.exp(-r * dt)

    return np.mean(payoff)


"""
# EXAMPLE USAGE
s0 = 100.0             # asset price
strike = 100           # strike price of option
T = 1.0                # time in years
r = 0.03               # risk-free rate
n = 252                # number of time steps in simulation
M = 50000              # number of simulations
# Heston dependent parameters
kappa = 0.5             # rate of mean reversion of variance under risk-neutral dynamics
theta = 0.04            # long-term mean of variance under risk-neutral dynamics
sigma = 0.3             # volatility of volatility
rho = 0.5               # correlation between returns and variances under risk-neutral dynamics
v0 = 0.02               # initial variance under risk-neutral dynamics

call_price = HestonFFT(s0, strike, T, r, kappa, theta, sigma, rho, v0, option_type="put")
print(f"European Call Price (FFT): {call_price:.4f}")

price = HestonQuad(s0, strike, T, r, kappa, theta, sigma, rho, v0, option_type="put")
print(f"European Call Price (Quadrature): {price:.4f}")

start = time.time()
price = LSM_American(s0, strike, T, r, kappa, theta, sigma, rho, v0, option_type="put", num_basis=3)
end = time.time()
print(f"American call price (Heston, LSM): {price:.4f}")
print(f"Execution time: {end - start:.4f} seconds")
"""
