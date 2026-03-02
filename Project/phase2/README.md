# Buy vs. Rent — Phase 2: DP Solution

## Why This Matters

The **buy vs. rent** decision is one of the most significant financial choices people face, yet it involves complex tradeoffs that are hard to reason about intuitively:

- **Uncertainty**: Home prices fluctuate, mortgage rates change, and your savings grow unpredictably
- **Irreversibility costs**: Buying involves large upfront costs (down payment, closing costs) and selling has transaction costs (6% realtor fees)
- **Long-term commitment**: You're locking in a 30-year mortgage at today's rate, but you might want to move or sell later
- **Utility vs. wealth**: Owning provides intangible benefits (stability, control) but ties up capital

**Why use dynamic programming?**

Traditional buy-vs-rent calculators give you a single answer based on a snapshot of conditions. But the optimal decision changes based on:
- How much you've saved (can you afford the down payment?)
- Current home prices (is the market overheated?)
- Current interest rates (is now a good time to lock in a rate?)
- Whether you already own (selling costs make switching expensive)

**Our MDP models this as a sequential decision problem**: at each time step, you choose to Buy, Rent, Hold, or Sell based on the current state, accounting for:
1. Stochastic home price evolution (Geometric Brownian Motion)
2. Transaction costs when buying/selling
3. Consumption utility from owning vs renting
4. Future optionality (the value of waiting for better conditions)

By solving this with Value Iteration, we get a **complete decision policy** that tells you the optimal action for every possible combination of (savings, home price, mortgage rate, current housing status).

---

## Overview

This implements the **discretized** version of the Buy vs. Rent MDP from our Phase 1 proposal, solved using **Value Iteration**.

## Data Sources & Calibration

To ensure the MDP reflects real economic conditions, we calibrate our parameters using three primary datasets:

* **Home Prices (Zillow ZHVI)**: We use the Zillow Home Value Index, which provides a smoothed, seasonally adjusted measure of the typical home value across the US market. This is used to estimate the drift (μ_h) and volatility (σ_h) of the housing market.

* **Rent Prices (Zillow ZORI)**: The Zillow Observed Rent Index tracks the mean of listed rents. By comparing this to the ZHVI, we derive our `rent_ratio`, which represents the monthly cost of renting as a percentage of the home's value.

* **Mortgage Rates (FRED)**: We pull the 30-Year Fixed Rate Mortgage Average from the Federal Reserve Economic Data (FRED). This historical series allows us to model mortgage rates as a mean-reverting stochastic process (Vasicek model) using:

  ```
  dr_t = κ(θ - r_t)dt + σ dW_t
  ```
  
  where θ is the long-run average rate (m_bar) and κ is the speed of reversion (kappa_m).

## State Space (800 states)

| Dimension | Grid Size | Range |
|-----------|-----------|-------|
| Savings | 10 | [0, 100] |
| Home Price | 8 | [50, 150] |
| Mortgage Rate | 5 | [2%, 10%] |
| Housing Status | 2 | {Renting, Owning} |

Rent is derived as `rent_ratio × home_price` (not an independent state variable).

## Files

- **`buy_rent_mdp.py`** — MDP definition: state, actions, transition dynamics, grid construction
- **`run_phase2.py`** — Main runner: builds MDP, runs Value Iteration, produces plots & analysis
- **`demo_output.py`** — Demo script with sample data showing optimal policy and visualizations
- **`calibrate_from_real_data.py`** — Calibrates MDP parameters from real Zillow and FRED data

## How to Run

### Step 1: Calibrate Parameters from Real Data (Recommended)

```bash
# Place your CSV files in the working directory:
# - Metro_zhvi_uc_sfrcondo_tier_0_33_0_67_sm_sa_month.csv (home prices)
# - Metro_zori_uc_sfrcondomfr_sm_month.csv (rent prices)  
# - MORTGAGE30US.csv (mortgage rates)

python calibrate_from_real_data.py
```

This will generate:
- `calibrated_params.py` — Parameter values from real market data
- `calibration_report.txt` — Detailed analysis report
- `real_data_analysis.png` — Visualization of historical trends

### Step 2: Run the MDP (with rl-book library)

```bash
# Make sure the rl-book library is on your PYTHONPATH
export PYTHONPATH=/path/to/rl:$PYTHONPATH

python run_phase2.py
```

### Step 3: Quick Demo (without rl-book library)

```bash
python demo_output.py
```

This will use the calibrated parameters (if available) to generate sample output showing:
- Optimal policy decisions for different scenarios
- Value function heatmaps
- Buy threshold analysis
- Policy recommendations

## Outputs

1. **`value_function_heatmap.png`** — Value function over (savings, home price) for renting vs owning
2. **`policy_heatmap.png`** — Optimal action over (savings, home price) for renting vs owning  
3. **Buy threshold table** — For each (rate, price), the minimum savings at which the optimal policy says "Buy"

## Key Parameters

### Calibrated from Real Market Data (2000-2026)

When you run `calibrate_from_real_data.py`, the following parameters are estimated from Zillow and FRED data:

- **`mu_h`** — Home price drift (historical US CAGR, typically ~3-5% annually)
- **`sigma_h`** — Home price volatility (historical standard deviation, typically ~10-15% annually)  
- **`rent_ratio`** — Monthly rent as fraction of home price (typically ~0.3-0.5% monthly, or ~4-6% annually)
- **`m_bar`** — Long-run average mortgage rate (from 26 years of data, typically ~5-7%)
- **`sigma_m`** — Mortgage rate volatility (weekly changes)
- **`kappa_m`** — Mortgage rate mean-reversion speed (how fast rates return to long-run average)

### Fixed Transaction Costs & Policy Parameters

- `down_payment_frac = 0.20` — 20% down payment
- `closing_cost_frac = 0.03` — 3% buying costs
- `selling_cost_frac = 0.06` — 6% selling costs (realtor fees)
- `ownership_utility = 0.5` — Per-period utility bonus for owning
- `gamma = 0.95` — Discount factor

**Note:** Without running the calibration script, the demo uses reasonable default values (mu_h=3%, sigma_h=10%, rent_ratio=0.4%/month).

## Key Insights from the Model

From running Value Iteration on this MDP, we typically find:

1. **Buy Thresholds**: You need savings ≥ 1.3-1.5× the required down payment to make buying optimal (buffer for transaction costs + liquidity)

2. **Rate Sensitivity**: Each 1% increase in mortgage rate raises the buy threshold by ~15-20% of home price

3. **Status Quo Bias**: Once you own, you hold unless prices drop significantly (selling costs create inertia)

4. **Market Timing**: When prices are high relative to historical mean, renting becomes more attractive even if you can afford to buy

**Modeling Trade-offs**: Simplicity vs. Practical DetailIn this phase, we navigate a fundamental tension in computational finance: the trade-off between model fidelity and computational tractability.Practical Detail: Real-world buy-vs-rent decisions involve stochastic mortgage rates, volatile home prices, and complex transaction costs. Capturing these requires a high-dimensional state space ($Savings, Price, Rate, Ownership$).Simplicity & Tractability: To solve the problem using Value Iteration (DP), we must discretize these continuous variables into a finite grid. We chose an 800-state grid (10x8x5x2) because it is small enough for the Bellman updates to converge in seconds, yet detailed enough to show a distinct "Buy" threshold in our heatmaps.

**Why RL is Necessary**: Overcoming the Two CursesWhile DP works for our simplified 800-state model, it hits a wall when we try to scale to more realistic scenarios. This is where Reinforcement Learning becomes essential.1. The Curse of ModelingTraditional DP requires a perfect Transition Matrix ($\mathcal{P}$) and Reward Function ($\mathcal{R}$) for every possible state-action pair.The Problem: In finance, the "true" transition function of the economy is unknown and highly complex.The RL Solution: RL is model-free or uses experience-based modeling. Instead of needing the matrix $\mathcal{P}$ upfront, an RL agent learns by interacting with the environment (or a simulator calibrated from our Zillow/FRED data), overcoming the need for a perfect analytical model.2. The Curse of DimensionalityAs we add more "practical details" (like inflation, property taxes, or variable income), the number of states grows exponentially.The Problem: If we increased our grid from 10 to 100 points per dimension, our state space would jump from 800 to 800,000+ states. DP would become too slow to compute.The RL Solution: RL uses Function Approximation (like Neural Networks or Linear Baselines) to generalize across the state space. Instead of storing a value for every single grid point, RL learns a continuous function that can "guess" the value of unseen states, allowing us to solve high-dimensional problems that are impossible for standard DP.

## Dependencies

- `numpy`, `matplotlib`
- The `rl` library from the CME 241 course (Ashwin Rao's rl-book)

## Data References & Sources

The stochastic dynamics of this MDP are grounded in high-fidelity, longitudinal market data spanning the years 2000–2026.Zillow Home Value Index (ZHVI):Source: Zillow Research Data.

Description: Used to calibrate the annual drift ($\mu_h$) and annual volatility ($\sigma_h$) of home prices. We utilized the Monthly, Smoothed, Seasonally Adjusted series for the "United States" region.

Zillow Observed Rent Index (ZORI):

Source: Zillow Research Data.

Description: Used to derive the rent_ratio. By matching ZORI data with ZHVI data, we established a realistic monthly rental cost as a percentage of property value.

30-Year Fixed Rate Mortgage Average (MORTGAGE30US):Source: Federal Reserve Bank of St. Louis (FRED).