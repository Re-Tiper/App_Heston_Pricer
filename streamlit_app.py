import streamlit as st
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from Heston_pricers import HestonFFT, HestonQuad, LSM_American
from Heston_sim import HestonModelSim, PlotHestonModel
from BS_vs_Heston import plot_Heston_vs_BSM


# To run locally:
# conda activate Stochastics
# streamlit run "/Users/themis/Programming/Python/Heston App/app.py"


# Inject custom CSS for buttons
st.markdown("""
    <style>
    div.stButton > button {
        background-color: #8AA6A3;
        color: #011F26;             
        padding: 0.6em 1.2em;
        font-size: 16px;
        border-radius: 8px;
        border: none;
        transition: background-color 0.3s;
    }
    div.stButton > button:hover {
        background-color: #023535;  
    }
    </style>
""", unsafe_allow_html=True)


# --- Main: Title ---
st.set_page_config(layout="wide")

col_title, col_author = st.columns([0.83, 0.17])

with col_title:
    st.markdown("<h1>Heston Option Pricing Model</h1>", unsafe_allow_html=True)

with col_author:
    st.markdown(
        f"""
            <strong><font color='#0FC2C0'>Created by: </font></strong> 
            <a href="https://github.com/Re-Tiper" target="_blank" style="color: #0FC2C0;">Re-Tiper</a>
        """,
        unsafe_allow_html=True,
    )

with st.expander("**📘 Mathematical Background**"):
    st.markdown(r"""
    ### The Heston Stochastic Volatility Model
    
    One of the most popular stochastic volatility models is the so-called **Heston model**, which is an extension of the **Black-Scholes-Merton model**. It is defined by the following system of stochastic differential equations, which describe the movement of an asset's price when both the price and its volatility follow random stochastic processes based on **Brownian motion**:
      
    $$
    dS(t) = \mu S(t) dt + \sqrt{v(t)} S(t)\, dW_S(t)  
    $$
    
    $$
    dv(t) = \kappa(\theta - v(t)) dt + \sigma \sqrt{v(t)}\, dW_v(t)  
    $$
    
    where the parameters are:
    
    - $\mu$: average rate of return for the underlying asset price.
    - $\theta$: the limit of the expected value of $v(t)$ as $t \to \infty$, i.e. $\lim_{t\to\infty}\mathbb{E}[v(t)]=\theta$.
    - $\kappa$: rate of mean reversion of $v(t)$ towards $\theta$.
    - $\sigma$: volatility of volatility $v(t)$.
    - $W_S(t)$: Brownian motion of the underlying asset price.
    - $W_v(t)$: Brownian motion of the asset's price volatility.
    - $\rho$: correlation between $W_S(t)$ and $W_v(t)$, i.e., $dW_S(t)\, dW_v(t)=\rho dt$.
    
    Additionally, when simulating the above stochastic differential equations, we also need the parameter $v_0 = dv(0)$, which denotes the initial variance.
    
    Under the **risk-neutral** probability measure $\widehat{\mathbb{P}}$ by applying **Girsanov's theorem**, it is shown that the equations become:
    
    $$
    dS(t) = r S(t) dt + \sqrt{v(t)} S(t)\, d\widehat{W}_S(t)
    $$
    
    $$
    dv(t) = \widehat{\kappa} (\widehat{\theta} - v(t)) dt + \sigma \sqrt{v(t)}\, d\widehat{W}_v(t)
    $$
    
    with
    
    $$
    d\widehat{W}_S(t) = dW_S(t) + \alpha_S dt \quad\text{where}\quad \alpha_S=\frac{\mu-r}{\sqrt{v(t)}}
    $$
    
    $$
    d\widehat{W}_v(t) = dW_v(t) + \alpha_v dt \quad\text{where}\quad \alpha_v=\frac{\lambda}{\sigma}\sqrt{v(t)}
    $$
    
    and
    
    $$
    \widehat{\kappa} = \kappa + \lambda\, \quad \widehat{\theta} = \frac{\kappa \theta}{\kappa + \lambda} \, \quad \widehat{\rho}= \rho
    $$
    
    where $\lambda$ is the **risk premium** parameter, which can be estimated using expected returns from positions in options hedged against the risk of changes in the underlying asset. We use these equations when we want to estimate option prices.
    
    
    ### Discretization of the stochastic differential equations using the Euler method
    
    We integrate the stochastic differential equations from $t$ to $t+dt$ and approximate them using the left-point rule.
    
    Following this process, we have,
    
    $$
    v(t+dt) = v(t) + \int_{t}^{t+dt}\kappa(\theta - v(u))du + \int_{t}^{t+dt}\sigma \sqrt{v(u)}\, dW_v(u)
    $$
    
    $$
    \approx v(t) + \kappa(\theta - v(t))dt + \sigma \sqrt{v(t)}(W_v(t+dt) - W_v(t))
    $$
    
    $$
    = v(t) + \kappa(\theta - v(t))dt + \sigma \sqrt{v(t) dt}Z_v
    $$
    
    where $Z_v$ is the standard normal distribution. It becomes apparent that the above discrete process for $v(t)$ can become negative with non-zero probability, making the calculation of $\sqrt{v(t)}$ impossible. Therefore, to avoid negative values, we replace $v(t)$ with $v^+(t)=\max(0,v(t))$ (the volatility $v(t)$ here is a **square root process**).
    
    Similarly, 
    
    $$
    S(t+dt) = S(t) + \int_{t}^{t+dt} \mu S(u)\, du + \int_{t}^{t+dt} \sqrt{v(u)} S(u)\, dW_S(u)
    $$
    
    $$
    \approx S(t) + \mu S(t) dt + \sqrt{v(t)} S(t)(W_S(t+dt) - W_S(t))
    $$
    
    $$
    = S(t) + \mu S(t) dt + \sqrt{v(t) dt} S(t) Z_S
    $$
    
    where $Z_S$ is the standard normal distribution correlated with $Z_v$ by $\rho$.
    
    Following Cholensky's decomposition, to construct $Z_S$ and $Z_v$, we first generate two independent $Z_1$ and $Z_2$ following $\mathcal{N}(0,1)$ and then set $Z_v = Z_1$ and $Z_S = \rho Z_1 + \sqrt{1-\rho^2}Z_2$.
    
    Solving the equation for the price of the underlying asset, leads to the solution:
    
    $$
    S(t+dt) = S(t)\exp \left[ \int_{t}^{t+dt}\left(\mu-\frac{1}{2}v(u)\right)\, du + \int_{t}^{t+dt}\sqrt{v(u)}\, dW(u)\right]
    $$
    
    and applying the **Euler discretization method**, we get:
    
    $$
    S(t+dt) = S(t)\exp\left[ \left(\mu-\frac{1}{2}v(t)\right)\, dt + \sqrt{v(t) dt}Z_S\right]
    $$
    """)

# --- Main: Choose Option Type ---
st.header("Option Type")
option_type = st.radio("**Select the option type:**", ("European Option", "American Option"))

# --- Main: Choose Pricing Method ---
st.header("Pricing Method")

# European Option Methods
if option_type == "European Option":
    col1, col2 = st.columns([0.8, 0.2])

    with col1:
        pricer_choice = st.radio(
            label="**Select the option pricing method:**",
            options=("Fast Fourier Transform Pricer", "Closed-Form solution")
        )

    with col2:
        st.markdown(" ")
        st.markdown(" ")
        st.markdown(
            """
            <a href="https://www.ma.imperial.ac.uk/~ajacquie/IC_Num_Methods/IC_Num_Methods_Docs/Literature/CarrMadan.pdf" 
               target="_blank" style="color: #0FC2C0; font-weight: bold; text-decoration: none;">
               📄 Carr & Madan, 1999
            </a><br>
            <a href="https://www.ma.imperial.ac.uk/~ajacquie/IC_Num_Methods/IC_Num_Methods_Docs/Literature/Heston.pdf" 
               target="_blank" style="color: #0FC2C0; font-weight: bold; text-decoration: none;">
               📄 Heston, 1993
            </a>
            """,
            unsafe_allow_html=True
        )

# American Option Method
elif option_type == "American Option":
    col1, col2 = st.columns([0.75, 0.25])
    with col1:
        pricer_choice = st.radio(
            label="**Only available method for American options:**",
            options=["Least-Squares Monte Carlo (LSM)"]
        )

    with col2:
        st.markdown(" ")
        st.markdown(" ")
        st.markdown(
            """
            <a href="https://people.math.ethz.ch/~hjfurrer/teaching/LongstaffSchwartzAmericanOptionsLeastSquareMonteCarlo.pdf" 
                target="_blank" style="color: #0FC2C0; font-weight: bold; text-decoration: none;">
                📄 Longstaff & Schwartz, 2001
            </a><br>
            <a href="https://www.diva-portal.org/smash/get/diva2:818128/FULLTEXT01.pdf"
                target="_blank" style="color: #0FC2C0; font-weight: bold; text-decoration: none;">
                📄 Gustafsson, 2015
            </a>
            """,
        unsafe_allow_html=True
    )

    # Expander for LSM Explanation
    with st.expander("📘 How the Least-Squares Monte Carlo (LSM) Algorithm Works"):
        st.markdown(r"""

        ## Value of American Options
        
        The value of an American option can be expressed as:
        
        $$
        V(t) = \sup_{t \leq \tau \leq T} \mathbb{E}_{\widehat{\mathbb{P}}}\left[e^{-r(\tau-t)} G(\tau) \mid S(t)\right]
        $$
        
        where 
        - $\tau$ is the exercise time
        - $\widehat{\mathbb{P}}$ is the risk-neutral probability measure
        - $r$ is the risk-free rate
        - $G(\tau)$ is the payoff of the option at time $\tau$
        
        The calculation corresponds to finding the optimal exercise time $\tau$, i.e., the moment when the option should be exercised. Once $\tau$ is determined, we compute the expected value:
        
        $$
        V(t) = \mathbb{E}_{\widehat{\mathbb{P}}}\left[e^{-r(\tau-t)} G(\tau) \mid S(t)\right]
        $$
        
        The present value of the option is then:
        
        $$
        V(0) = \sup_{0 \leq \tau \leq T} \mathbb{E}_{\widehat{\mathbb{P}}}\left[e^{-r\tau} G(\tau) \mid S(0)\right]
        $$
        
        ## Steps of the Longstaff-Schwartz Method (LSM)
        
        The LSM algorithm for pricing American options consists of the following steps:
        
        1. **Path Simulation**:  
           Simulate the price of the underlying asset M times over n equally spaced time steps, according to the geometric Brownian motion (or another stochastic model, i.e Heston's diffusion), using the Monte Carlo method. This generates paths reflecting potential asset prices.
        
        2. **Dynamic Programming**:  
           After simulating the paths, the algorithm works backward from the option's expiration date T to time 0. At each time step:
           - Compute the payoff if the option is exercised at that time.
           - Estimate the continuation value using least squares regression, which approximates the expected payoff if the holder continues to hold the option.
        
        3. **Regression**:  
           The core of the method is the use of least squares regression to estimate the continuation value at each time step:
        
        $$
        V(t_i) = \mathbb{E}_{\widehat{\mathbb{P}}}\left[e^{-r(\tau'-t_i)} G(\tau') \mid S(t_i)\right]
        $$
        
           Here, $\tau'$ is the optimal exercise time in $\{t_{i+1}, t_{i+2}, \dots, t_n = T\} \subseteq [0, T]$. The regression uses a set of basis functions, often polynomials, to approximate these expected values, enabling the decision of whether to exercise the option or continue holding it.
        
        4. **Exercise Decision**:  
           At each time step, for every simulated path, compare the immediate exercise value with the estimated continuation value. If the immediate exercise value is higher, exercise the option, and the optimal stopping time becomes that time step. Otherwise, do not exercise. Repeat this process backward to determine the optimal stopping time.
        
        5. **Option Value Calculation**:  
           After determining the optimal stopping time for all paths, discount the payoffs back to present value using the risk-free rate. The average of these discounted payoffs is the estimated option value.
        
        ## Basis Functions
        
        We will use **Laguerre polynomials** as the basis functions for estimating the expected value \( V(t_i) \). Laguerre polynomials form an orthogonal basis of the space $$ L^2([0,\infty]) $$ with respect to the inner product:
        
        $$
        \langle f, g \rangle = \int_{0}^{\infty} f(x) g(x) e^{-x} \, dx
        $$
        
        The general form of Laguerre polynomials is:
        
        $$
        L_n(x) = \sum_{k=0}^{n} \binom{n}{k} \frac{(-x)^k}{k!}
        $$
        
        We will use the first 5 such polynomials:
        
        $$
        \begin{aligned}
        L_0(x) &= 1 \\
        L_1(x) &= 1 - x \\
        L_2(x) &= \frac{1}{2}(2 - 4x + x^2) \\
        L_3(x) &= \frac{1}{6}(6 - 18x + 9x^2 - x^3) \\
        L_4(x) &= \frac{1}{24}(24 - 96x + 72x^2 - 16x^3 + x^4)
        \end{aligned}
        $$
        
        The expected value $V(t_i)$ can be expressed as a linear combination of these basis functions:
        
        $$
        \widehat{V}(S_l(t_i)) = \sum_{j=0}^{k} \beta_j L_j(S_l(t_i))
        $$
        
        Here, $\beta_j$ are the regression coefficients obtained using least squares.
        
        Specifically:
        
        $$
        Y = A \cdot \hat{\beta}
        $$
        
        where:
        
        - $Y = (y_1, y_2, \dots, y_M)^T$ is the vector of future discounted payoffs (continuation values), with $y_l = e^{-r(\tau' - t_i)} G_l(\tau')$ for $l = 1, \dots, M$, where $\tau'$ is the optimal exercise time in $\{t_{i+1}, t_{i+2}, \dots, t_n = T\}$, and $G_l$ is the payoff of the $l$-th path.
        - $\hat{\beta} = (\hat{\beta}_0, \hat{\beta}_1, \dots, \hat{\beta}_k)^T$ are the regression coefficients.
        - $A_{M \times k}$ is the matrix of basis functions, where $A_{lj} = L_j(S_l(t_i))$, $l = 1, \dots, M$, $j = 0, 1, \dots, k$ (e.g., for a 2nd-degree polynomial, $A = [1, 1 - X, \frac{1}{2}(2 - 4X + X^2)]$).
        
        The least squares solution for $\hat{\beta}$ is:
        
        $$
        \hat{\beta} = (A^{T}A)^{-1}A^{T}Y
        $$
        
        It can be shown that:
        
        $$
        \lim_{k \to \infty} \widehat{V}(S_l(t_i)) = V(S_l(t_i)) \quad \forall l \in \{1, \dots, M\}
        $$
        
        but without knowing the rate of convergence.
        """)


# Pricer function based on choice
def price_option(params, s0_local, K_local, T_local, option_type):
    if pricer_choice == "Fast Fourier Transform Pricer":
        return HestonFFT(
            s0=s0_local,
            strike=K_local,
            T=T_local,
            r=params["r"],
            kappa=params["kappa"],
            theta=params["theta"],
            sigma=params["sigma"],
            rho=params["rho"],
            v0=params["v0"],
            option_type=option_type,
        )
    elif pricer_choice == "Closed-Form solution":
        return HestonQuad(
            s0=s0_local,
            strike=K_local,
            T=T_local,
            r=params["r"],
            kappa=params["kappa"],
            theta=params["theta"],
            sigma=params["sigma"],
            rho=params["rho"],
            v0=params["v0"],
            option_type=option_type,
        )
    else:
        return LSM_American(
            s0=s0_local,
            strike=K_local,
            T=T_local,
            r=params["r"],
            kappa=params["kappa"],
            theta=params["theta"],
            sigma=params["sigma"],
            rho=params["rho"],
            v0=params["v0"],
            option_type=option_type,
            num_basis=4
        )

# --- Sidebar: Option Contract Parameters ---
st.sidebar.header("Option Contract Parameters")

s0 = st.sidebar.number_input(
    "S₀ (Current asset price)",
    min_value=0.001,
    max_value=10000.0,
    value=100.0,
    step=1.0,
)

K = st.sidebar.number_input(
    "K (Strike price)",
    min_value=0.001,
    max_value=10000.0,
    value=100.0,
    step=1.0,
)

T = st.sidebar.number_input(
    "T (Time to maturity in years)",
    min_value=0.001,
    max_value=10.0,
    value=1.0,
    step=0.01,
)

# --- Sidebar: Heston Model Parameters ---
st.sidebar.header("Heston Model Parameters")
st.sidebar.caption(
    "**Note:** To recover the Black-Scholes model, set:\n"
    "- σ = 0  (no volatility of volatility)\n"
    "- θ = v₀  (in order to achieve constant variance)\n"
    "- κ and ρ can be any values\n\n"
    "Keep in mind that both pricing methods for European options do not work when σ = 0 since "
    "then variance becomes deterministic and the assumptions of the Heston model do not hold."
)

params = {
    "kappa": st.sidebar.slider(
        "κ (mean reversion)", min_value=1e-5, max_value=5.0, value=2.0, step=0.01
    ),
    "theta": st.sidebar.slider(
        "θ (long-term variance)", min_value=1e-5, max_value=0.4, value=0.04, step=0.01
    ),
    "sigma": st.sidebar.slider(
        "σ (vol of vol)", min_value=1e-5, max_value=1.0, value=0.3, step=0.01
    ),
    "rho": st.sidebar.slider(
        "ρ (correlation)", min_value=-0.99999, max_value=0.99999, value=0.5, step=0.01
    ),
    "v0": st.sidebar.slider(
        "v₀ (initial variance)", min_value=0.0, max_value=0.4, value=0.02, step=0.01
    ),
    "r": st.sidebar.slider(
        "r (risk-free rate)", min_value=0.0, max_value=0.2, value=0.03, step=0.01
    ),
}

# --- Parameter Input Types and Ranges ---

SLIDER_PARAMS = {
    "kappa": (1e-5, 5.0, 0.01, 2.0),
    "theta": (1e-5, 0.4, 0.01, 0.04),
    "sigma": (1e-5, 1.0, 0.01, 0.3),
    "rho": (-0.99999, 0.99999, 0.01, -0.5),
    "v0": (0.0, 0.1, 0.01, 0.04),
    "r": (0.0, 0.1, 0.01, 0.03),
}

NUMBER_INPUT_PARAMS = {
    "S₀": (0.001, 10000.0, 100.0, 1.0),
    "K": (0.001, 10000.0, 100.0, 1.0),
    "T": (0.001, 10.0, 1.0, 0.01),
}

def input_range(param_key: str, label_min: str, label_max: str, center_value: float, default_range=0.3):
    """
    Create dynamic input widgets for min/max range selection based on param type,
    centered around `center_value` with a relative `default_range` fraction.

    Parameters:
        param_key (str): The parameter name key.
        label_min (str): Label for min input.
        label_max (str): Label for max input.
        center_value (float): The current value (choosen by the user) around which to center the range.
        default_range (float): Fraction of the center_value for the default +/- range (e.g., 0.2 for ±20%).

    Returns:
        (min_val, max_val): The user-selected range.
    """
    if param_key in SLIDER_PARAMS:
        mn, mx, step, _ = SLIDER_PARAMS[param_key]
    else:
        mn, mx, _, step = NUMBER_INPUT_PARAMS[param_key]

    # Calculate default min and max centered around center_value, clipped to allowed range
    default_min = max(mn, center_value * (1 - default_range))
    default_max = min(mx, center_value * (1 + default_range))

    # For parameters where center_value is very close to zero, fallback to a small fixed range
    if center_value == 0 or (default_max - default_min) < step:
        default_min = mn
        default_max = mx

    if param_key in SLIDER_PARAMS:
        min_val, max_val = st.slider(
            f"{label_min} range",
            min_value=mn,
            max_value=mx,
            value=(default_min, default_max),
            step=step,
        )
    else:
        min_val = st.number_input(
            f"{label_min} min",
            min_value=mn,
            max_value=mx,
            value=round(default_min, 6),
            step=step,
            format="%.6f",
        )
        max_val = st.number_input(
            f"{label_max} max",
            min_value=mn,
            max_value=mx,
            value=round(default_max, 6),
            step=step,
            format="%.6f",
        )
        if max_val < min_val:
            st.error(f"Error: max {label_max} must be >= min {label_min}")
            max_val = min_val

    return min_val, max_val


# --- Main: Compute and display the Call and Put price using selected parameters ---

try:
    call_price = price_option(params, s0, K, T, "call")
    put_price = price_option(params, s0, K, T, "put")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(
            f"""
            <div style="background-color:#03A688; padding:10px 12px; border-radius:10px; text-align:center;">
                <h4 style="color:black; margin-top: 0;">Call Price</h4>
                <p style="font-size:18px; margin:0; color:black;"><strong>{call_price:.4f}</strong></p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            f"""
            <div style="background-color:#F2668B; padding:10px 12px; border-radius:10px; text-align:center;">
                <h4 style="color:black; margin-top: 0;">Put Price</h4>
                <p style="font-size:18px; margin:0; color:black;"><strong>{put_price:.4f}</strong></p>
            </div>
            """,
            unsafe_allow_html=True,
        )

except Exception as e:
    st.error(f"Failed to compute option prices: {e}")



# --- Main: Heatmap Title ---

st.header("Interactive Heatmap")

st.markdown(
        """
        <div style="background-color:#025E73; padding:15px; border-radius:10px; margin-top:-10px;">
            <p style="font-size:18px; margin:0; color:#011F26;">
                Explore how call and put option prices vary as you change two model parameters. Select your variables and press the button to generate the heatmaps.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

# --- Heatmap Axis Selection ---
st.markdown("### Select Heatmap Axes")
CHOICES = {
    "Asset Price (S₀)": "S₀",
    "Strike (K)": "K",
    "Maturity (T)": "T",
    "κ (kappa)": "kappa",
    "θ (theta)": "theta",
    "σ (sigma)": "sigma",
    "ρ (rho)": "rho",
    "v₀ (v0)": "v0",
    "Risk-free rate (r)": "r",
}

x_label = st.selectbox("X-axis", list(CHOICES.keys()), index=0)
y_label = st.selectbox("Y-axis", list(CHOICES.keys()), index=1)

x_var = CHOICES[x_label]
y_var = CHOICES[y_label]

st.markdown("### Specify the range of the parameters")

# Pass current values for centering
current_values = {
    "S₀": s0,
    "K": K,
    "T": T,
    **params  # unpack kappa, theta, sigma, rho, v0, r
}

x_min, x_max = input_range(x_var, x_label, x_label, center_value=current_values[x_var])
y_min, y_max = input_range(y_var, y_label, y_label, center_value=current_values[y_var])

generate_heatmap = st.button("Generate Heatmaps", key="heatmap_button")

if generate_heatmap:

    # Adjust grid based on pricer
    if pricer_choice == "Least-Squares Monte Carlo (LSM)":
        x_vals = np.linspace(x_min, x_max, num=10)
        y_vals = np.linspace(y_min, y_max, num=10)
    else:
        x_vals = np.linspace(x_min, x_max, num=15)
        y_vals = np.linspace(y_min, y_max, num=15)

    heatmap_call = np.zeros((len(y_vals), len(x_vals)))
    heatmap_put = np.zeros((len(y_vals), len(x_vals)))

    for i, y_val in enumerate(y_vals):
        for j, x_val in enumerate(x_vals):
            local_params = params.copy()
            K_local, T_local, s0_local = K, T, s0  # Defaults

            if x_var == "K":
                K_local = x_val
            elif x_var == "T":
                T_local = x_val
            elif x_var == "S₀":
                s0_local = x_val
            else:
                local_params[x_var] = x_val

            if y_var == "K":
                K_local = y_val
            elif y_var == "T":
                T_local = y_val
            elif y_var == "S₀":
                s0_local = y_val
            else:
                local_params[y_var] = y_val

            try:
                call_price = price_option(local_params, s0_local, K_local, T_local, "call")
                put_price = price_option(local_params, s0_local, K_local, T_local, "put")
                heatmap_call[i, j] = call_price
                heatmap_put[i, j] = put_price
            except Exception as e:
                heatmap_call[i, j] = np.nan
                heatmap_put[i, j] = np.nan
                print(f"Failed at {x_var}={x_val}, {y_var}={y_val}: {e}")

    # --- Main: Plot Heatmap ---
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Call Option Price Heatmap")
        fig_call, ax_call = plt.subplots(figsize=(20, 16))
        sns.heatmap(
            heatmap_call,
            xticklabels=np.round(x_vals, 2),
            yticklabels=np.round(y_vals, 2),
            cmap="RdYlGn",
            annot=np.round(heatmap_call, 2),
            fmt=".2f",
            annot_kws={"size": 14},
            linewidths=0.2,
            ax=ax_call,
            cbar_kws={"label": "Call Option Price"},
        )
        ax_call.set_title("Call Option Price Heatmap", fontsize=16)
        ax_call.set_xlabel(x_label, fontsize=16)
        ax_call.set_ylabel(y_label, fontsize=16)
        ax_call.tick_params(axis='both', labelsize=14)
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        st.pyplot(fig_call)

    with col2:
        st.markdown("### Put Option Price Heatmap")
        fig_put, ax_put = plt.subplots(figsize=(20, 16))
        sns.heatmap(
            heatmap_put,
            xticklabels=np.round(x_vals, 2),
            yticklabels=np.round(y_vals, 2),
            cmap="RdYlGn",
            annot=np.round(heatmap_put, 2),
            fmt=".2f",
            annot_kws={"size": 14},
            linewidths=0.2,
            ax=ax_put,
            cbar_kws={"label": "Put Option Price"},
        )
        ax_put.set_title("Put Option Price Heatmap", fontsize=16)
        ax_put.set_xlabel(x_label, fontsize=16)
        ax_put.set_ylabel(y_label, fontsize=16)
        ax_put.tick_params(axis='both', labelsize=14)
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        st.pyplot(fig_put)


# --- Main: Plot Option Price Difference ---
st.header("Heston vs Black-Scholes")

st.markdown(
        """
        <div style="background-color:#025E73; padding:15px; border-radius:10px; margin-top:-10px;">
            <p style="font-size:18px; margin:0; color:#011F26;">
                Plot the option (European) price difference from the Heston stochastic volatility model minus the Black-Scholes with a volatility that matches the 
                the (square root of the) expected variance of the spot return over the life of the option. Except from the parameter choosen, all the other parameter values 
                remain the same.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )
st.markdown(" ")

with st.expander("**📘 Matching Heston Variance to Black-Scholes Volatility**"):
    st.markdown(r"""
    # Matching Heston Variance to Black-Scholes Volatility

    To compare the Heston model to the Black-Scholes model fairly, we match the **total variance** over the life of the option. \\
    In the Black-Scholes model, the volatility $\sigma_{\text{BSM}}$ is constant and hence, the variance of log-returns over time $ T $ is simply:
    
    $$
    \text{Var}_{\text{BSM}} = \sigma_{\text{BSM}}^2 \cdot T
    $$
    
    In the Heston model, variance is stochastic. But, using Fubini's theorem, we can compute the **expected total variance** over time $ T $ as:
    
    $$
    \text{Var}_{\text{Heston}} = \mathbb{E}\left[ \int_0^T v_t dt \right] = \int_0^T \mathbb{E}[v_t] dt
    $$
    
    ---
    
    ## Expected Variance in the Heston Model
    
    The variance process follows the Cox-Ingersoll-Ross (CIR) process:
    
    $$
    dv_t = \kappa(\theta - v_t) dt + \sigma \sqrt{v_t} dW_t
    $$
    
    Using the Itô-Doeblin formula with the function $ f(t,x) = e^{\kappa t} x $, we can show that:
    
    $$
    \mathbb{E}[v_t] = v_0 e^{-\kappa t} + \theta (1 - e^{-\kappa t})
    $$
    
    ---
    
    ## Total Expected Variance
    
    Now integrate this over $ [0, T] $:
    
    $$
    \int_0^T \mathbb{E}[v_t] dt = \int_0^T \left( v_0 e^{-\kappa t} + \theta (1 - e^{-\kappa t}) \right) dt
    = v_0 \int_0^T e^{-\kappa t} dt + \theta \int_0^T (1 - e^{-\kappa t}) dt
    $$
    
    We derive:
    
    $$
    \int_0^T \mathbb{E}[v_t] dt = v_0 \cdot \frac{1 - e^{-\kappa T}}{\kappa} + \theta \left( T - \frac{1 - e^{-\kappa T}}{\kappa} \right)
    $$
    
    $$
    = \frac{1 - e^{-\kappa T}}{\kappa} (v_0 - \theta) + \theta T
    $$
    
    ---
    
    ## BSM Matching Volatility
    
    Equating this to $ \sigma_{\text{BSM}}^2 T $, we get:
    
    $$
    \sigma_{\text{BSM}}^2 T = \frac{1 - e^{-\kappa T}}{\kappa} (v_0 - \theta) + \theta T
    $$
    
    Divide both sides by $ T $:
    
    $$
    \sigma_{\text{BSM}}^2 = \frac{1}{T} \left[ \theta T + \frac{1 - e^{-\kappa T}}{\kappa} (v_0 - \theta) \right]
    $$
    
    $$
    = \theta + \frac{(v_0 - \theta)}{\kappa T} (1 - e^{-\kappa T})
    $$
    
    ---
    
    Hence, the **Black-Scholes volatility** that matches the **average variance** of the Heston process over time $ T $ is

    $$
    \boxed{
    \sigma_{\text{BSM}} = \sqrt{ \theta + \frac{(v_0 - \theta)}{\kappa T} (1 - e^{-\kappa T}) }
    }
    $$
    """)

st.subheader("Choose option type")
call_option = st.checkbox("Call Option", value=True)
put_option = st.checkbox("Put Option", value=False)

PAR_CHOICES = {
    "κ (kappa)": "kappa",
    "θ (theta)": "theta",
    "σ (sigma)": "sigma",
    "ρ (rho)": "rho",
    "v₀ (v0)": "v0",
}
x_label_2 = st.selectbox("Choose Parameter", list(PAR_CHOICES.keys()))
x_var_2 = PAR_CHOICES[x_label_2]

st.markdown("### Specify the different values of the parameter")
# Select how many parameter values to compare
num_values = st.selectbox("How many values do you want to compare?", [1, 2, 3], index=1)

# Get bounds for the selected parameter
param_min, param_max, _, param_step = SLIDER_PARAMS[x_var_2]

param_values = []
for i in range(num_values):
    val = st.number_input(
        f"Value {i+1} for {x_label_2}",
        min_value=param_min,
        max_value=param_max,
        value=np.clip(current_values[x_var_2] * (1 + 0.5 * (i - 1)), param_min, param_max), # creates spaced values around the center (30%)
        step=param_step,
        format="%.3f",
        key=f"param_val_{i}_{x_var_2}"
    )
    param_values.append(val)

plot_difference = st.button("Generate Plot", key="diff_plot_button")

if plot_difference:
    try:
        # Only plot for selected option types
        for opt_type in (["call"] if call_option and not put_option else
                        ["put"] if put_option and not call_option else
                         ["call", "put"]):
            fig = plot_Heston_vs_BSM(
                heston_pricer_func=price_option,  # your Heston pricer function
                s0=s0,
                strike=K,
                T=T,
                r=params["r"],
                kappa=params["kappa"],
                theta=params["theta"],
                sigma=params["sigma"],
                rho=params["rho"],
                v0=params["v0"],
                param_name=x_var_2,
                param_values=param_values,
                option_type=opt_type
            )
            st.pyplot(fig)

    except Exception as e:
        st.error(f"Failed to generate price difference plot: {e}")



# --- Main: Plot Asset Price and Volatility ---

st.header("Heston Model Simulation")

st.markdown(
        """
        <div style="background-color:#025E73; padding:15px; border-radius:10px; margin-top:-10px;">
            <p style="font-size:18px; margin:0; color:#011F26;">
                Simulate the asset prices and their volatility as defined in the Heston model. 
                You can also study how the daily and terminal log-returns distribution varies for different values.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )
st.markdown(" ")

# User inputs for simulation
st.subheader("Plots to Show")
plot_prices = st.checkbox("Asset Prices", value=True)
plot_vol = st.checkbox("Volatility", value=True)
plot_returns = st.checkbox("Log-Returns", value=False)
plot_dist = st.checkbox("Distribution of Daily Log-Returns", value=False)
plot_dist_comp = st.checkbox("Distribution of Compounded Log-Returns", value=False)


# Nested checkbox for KDE only if distribution is selected
if plot_dist or plot_dist_comp:
    _, plot_kde = st.columns([0.2, 9.8])
    with plot_kde:
        plot_kde = st.checkbox("Overlay Kernel Density Estimate (KDE)", value=True)
else:
    plot_kde = False


n_steps = st.number_input("Number of time steps (252 days in 1 trading year)", min_value=10, max_value=5000, value=252, step=10)
n_sims = st.number_input("Number of simulation paths", min_value=1, max_value=100000, value=10, step=1)

show_subset = st.selectbox(
    "Select Subset of Paths to Display",
    options=["Show all paths", "First 10 Paths", "First 50 Paths", "First 100 Paths"],
)

show_mean = st.checkbox("Show Mean Only")
show_range = st.checkbox("Show Min-Max Range")

run_sim = st.button("Run Simulation")

if run_sim:
    try:
        S_sim, var_sim = HestonModelSim(
            s0=s0,
            T=T,
            r=params["r"],
            kappa=params["kappa"],
            theta=params["theta"],
            sigma=params["sigma"],
            rho=params["rho"],
            v0=params["v0"],
            n=n_steps,
            M=n_sims
        )

        figs = PlotHestonModel(S_sim, var_sim, T, show_subset, show_mean, show_range, plot_prices, plot_vol, plot_returns, plot_dist, plot_dist_comp, plot_kde)

        if 'prices' in figs:
            st.pyplot(figs['prices'])

        if 'volatility' in figs:
            st.pyplot(figs['volatility'])

        if 'returns' in figs:
            st.pyplot(figs['returns'])

        if 'return_dist' in figs:
            st.pyplot(figs['return_dist'])

        if 'comp_return_dist' in figs:
            st.pyplot(figs['comp_return_dist'])

    except Exception as e:
        st.error(f"Simulation failed: {e}")

