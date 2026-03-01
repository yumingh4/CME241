# Buy vs. Rent — Phase 2: DP Solution

## Overview

This implements the **discretized** version of the Buy vs. Rent MDP from our Phase 1 proposal, solved using **Value Iteration**.

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

## How to Run

```bash
# Make sure the rl-book library is on your PYTHONPATH
export PYTHONPATH=/path/to/rl:$PYTHONPATH

python run_phase2.py
```

## Outputs

1. **`value_function_heatmap.png`** — Value function over (savings, home price) for renting vs owning
2. **`policy_heatmap.png`** — Optimal action over (savings, home price) for renting vs owning  
3. **Buy threshold table** — For each (rate, price), the minimum savings at which the optimal policy says "Buy"

## Key Parameters (editable in `buy_rent_mdp.py`)

- `mu_h = 0.03` — home price drift
- `sigma_h = 0.10` — home price volatility  
- `down_payment_frac = 0.20` — 20% down payment
- `closing_cost_frac = 0.03` — 3% buying costs
- `selling_cost_frac = 0.06` — 6% selling costs
- `ownership_utility = 0.5` — per-period utility bonus for owning
- `gamma = 0.95` — discount factor

## Dependencies

- `numpy`, `matplotlib`
- The `rl` library from the CME 241 course (Ashwin Rao's rl-book)
