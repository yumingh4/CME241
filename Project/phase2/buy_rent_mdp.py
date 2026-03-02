"""
Buy vs. Rent MDP — Phase 2 (DP-Solvable Discretized Version)

State: (savings_idx, price_idx, rate_idx, owns)
  - savings: ~10 buckets
  - home_price: ~8 levels  
  - mortgage_rate: ~5 levels
  - owns: {False, True}
  - rent is derived from home_price (not independent)

Actions:
  - Renting: {RENT_STAY, BUY}
  - Owning:  {OWN_STAY, SELL}

Total states: 10 * 8 * 5 * 2 = 800

Parameters can be calibrated from real data using calibrate_from_real_data.py
"""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import (
    Dict, Iterable, Iterator, List, Mapping, Sequence, Set, Tuple
)
from collections import defaultdict

import numpy as np

from rl.distribution import Categorical, FiniteDistribution
from rl.markov_decision_process import FiniteMarkovDecisionProcess
from rl.markov_process import NonTerminal, Terminal, State
from rl.policy import FinitePolicy, FiniteDeterministicPolicy


# -----------------------------------------------------------------------
# Load calibrated parameters if available
# -----------------------------------------------------------------------
def load_calibrated_params():
    """Load calibrated parameters from file if available, else return defaults."""
    try:
        import sys
        import os
        # Try to load from outputs directory
        sys.path.insert(0, '/mnt/user-data/outputs')
        from calibrated_params import CALIBRATED_PARAMS
        return CALIBRATED_PARAMS
    except:
        # Return reasonable defaults (conservative estimates)
        return {
            'mu_h': 0.03,      # 3% annual home price appreciation
            'sigma_h': 0.10,   # 10% annual volatility
            'rent_ratio': 0.004,  # 0.4% monthly (4.8% annually)
            'm_bar': 0.06,     # 6% long-run mortgage rate
            'sigma_m': 0.01,   # 1% mortgage rate volatility
            'kappa_m': 0.20,   # Mean reversion speed
        }

CALIB = load_calibrated_params()


# -----------------------------------------------------------------------
# Action enum
# -----------------------------------------------------------------------
class Action(Enum):
    RENT_STAY = 0  # continue renting
    BUY = 1        # purchase home
    OWN_STAY = 2   # continue owning (pay mortgage)
    SELL = 3       # sell home, return to renting


# -----------------------------------------------------------------------
# Parameters
# -----------------------------------------------------------------------
@dataclass
class BuyRentParams:
    """All tunable parameters for the discretized MDP.
    
    Default values are loaded from calibrated_params.py if available,
    otherwise fall back to conservative estimates.
    """

    # --- Dynamics (calibrated from real data) ---
    mu_h: float = CALIB['mu_h']          # home price drift (annual)
    sigma_h: float = CALIB['sigma_h']    # home price volatility
    rent_ratio: float = CALIB['rent_ratio']  # rent = rent_ratio * home_price (monthly)
    m_bar: float = CALIB['m_bar']        # long-run mortgage rate
    sigma_m: float = CALIB['sigma_m']    # mortgage rate volatility
    kappa_m: float = CALIB['kappa_m']    # mortgage rate mean-reversion speed
    
    # --- Other dynamics ---
    income: float = 5.0         # periodic income (per year)
    risk_free: float = 0.03     # return on savings

    # --- Transaction costs (industry standards) ---
    down_payment_frac: float = 0.20
    closing_cost_frac: float = 0.03
    selling_cost_frac: float = 0.06

    # --- Utility ---
    ownership_utility: float = 0.1  # per-period bonus for owning
    gamma: float = 0.9

    # --- Discretization grids ---
    savings_grid: np.ndarray = field(
        default_factory=lambda: np.linspace(0, 100, 10)
    )
    price_grid: np.ndarray = field(
        default_factory=lambda: np.linspace(50, 150, 8)
    )
    rate_grid: np.ndarray = field(
        default_factory=lambda: np.linspace(0.02, 0.10, 5)
    )


# -----------------------------------------------------------------------
# Discrete state representation
# -----------------------------------------------------------------------
@dataclass(frozen=True)
class BuyRentState:
    """Discretized state for Phase 2."""
    savings_idx: int
    price_idx: int
    rate_idx: int
    owns: bool  # True = owning, False = renting


# -----------------------------------------------------------------------
# Helper: snap a continuous value to the nearest grid index
# -----------------------------------------------------------------------
def snap_to_grid(value: float, grid: np.ndarray) -> int:
    """Return index of nearest grid point, clamped to valid range."""
    idx = int(np.argmin(np.abs(grid - value)))
    return idx


# -----------------------------------------------------------------------
# Build the finite MDP mapping
# -----------------------------------------------------------------------
def build_buy_rent_mdp(
    params: BuyRentParams,
    n_samples: int = 500,
) -> Tuple[
    FiniteMarkovDecisionProcess[BuyRentState, Action],
    BuyRentParams,
]:
    """
    Construct a FiniteMarkovDecisionProcess by Monte Carlo estimation
    of transition probabilities.

    For each (state, action) pair we sample `n_samples` next states
    from the continuous dynamics, snap them to the grid, and build
    an empirical distribution.

    Returns:
        (mdp, params)
    """
    p = params
    sg = p.savings_grid
    pg = p.price_grid
    rg = p.rate_grid

    # All non-terminal states
    all_states: List[BuyRentState] = []
    for si in range(len(sg)):
        for pi in range(len(pg)):
            for ri in range(len(rg)):
                for owns in [False, True]:
                    all_states.append(BuyRentState(si, pi, ri, owns))

    # Build the mapping: S -> {A -> FiniteDistribution[(S, reward)]}
    mapping: Dict[
        BuyRentState,
        Dict[Action, FiniteDistribution[Tuple[BuyRentState, float]]]
    ] = {}

    for state in all_states:
        w = sg[state.savings_idx]
        h = pg[state.price_idx]
        m = rg[state.rate_idx]
        rent = p.rent_ratio * h

        # Determine available actions
        actions: List[Action] = []
        if state.owns:
            actions = [Action.OWN_STAY, Action.SELL]
        else:
            actions = [Action.RENT_STAY]
            buy_cost = (p.down_payment_frac + p.closing_cost_frac) * h
            if w >= buy_cost:
                actions.append(Action.BUY)

        action_map: Dict[
            Action, FiniteDistribution[Tuple[BuyRentState, float]]
        ] = {}

        for action in actions:
            # Monte Carlo: sample transitions
            outcomes: Dict[Tuple[BuyRentState, float], float] = defaultdict(float)

            for _ in range(n_samples):
                # --- Evolve stochastic variables ---
                eps_h = np.random.randn()
                eps_m = np.random.randn()

                new_h = h * np.exp(
                    (p.mu_h - 0.5 * p.sigma_h ** 2) + p.sigma_h * eps_h
                )
                new_m = m + p.kappa_m * (p.m_bar - m) + p.sigma_m * eps_m
                new_m = max(rg[0], min(rg[-1], new_m))

                # --- Process action ---
                new_w = w
                reward = 0.0
                new_owns = state.owns

                if action == Action.RENT_STAY:
                    housing_cost = rent
                    new_w = (w - housing_cost + p.income) * (1 + p.risk_free)
                    reward = -housing_cost

                elif action == Action.BUY:
                    down = p.down_payment_frac * h
                    closing = p.closing_cost_frac * h
                    new_w = w - down - closing
                    new_owns = True
                    reward = -closing + p.ownership_utility

                elif action == Action.OWN_STAY:
                    # Simplified: interest-only mortgage payment
                    mortgage_bal = (1 - p.down_payment_frac) * h
                    payment = mortgage_bal * m
                    housing_cost = payment
                    new_w = (w - payment + p.income) * (1 + p.risk_free)
                    reward = -housing_cost + p.ownership_utility

                elif action == Action.SELL:
                    # Simplified equity: assume price appreciation only
                    mortgage_bal = (1 - p.down_payment_frac) * h
                    equity = h - mortgage_bal
                    sell_cost = p.selling_cost_frac * h
                    new_w = w + equity - sell_cost
                    new_owns = False
                    reward = -sell_cost

                # Snap to grid
                new_si = snap_to_grid(new_w, sg)
                new_pi = snap_to_grid(new_h, pg)
                new_ri = snap_to_grid(new_m, rg)

                next_state = BuyRentState(new_si, new_pi, new_ri, new_owns)
                # Round reward to reduce state explosion
                reward_rounded = round(reward, 2)
                outcomes[(next_state, reward_rounded)] += 1.0

            # Normalize to get probabilities
            total = sum(outcomes.values())
            dist = Categorical(
                {k: v / total for k, v in outcomes.items()}
            )
            action_map[action] = dist

        mapping[state] = action_map

    mdp = FiniteMarkovDecisionProcess(mapping)
    return mdp, params


# -----------------------------------------------------------------------
# Pretty-print a policy
# -----------------------------------------------------------------------
def print_policy(
    policy: FiniteDeterministicPolicy[BuyRentState, Action],
    params: BuyRentParams,
):
    """Print the policy in a readable format."""
    sg = params.savings_grid
    pg = params.price_grid
    rg = params.rate_grid

    print("\n=== OPTIMAL POLICY ===")
    for owns in [False, True]:
        status = "OWNING" if owns else "RENTING"
        print(f"\n--- Currently {status} ---")
        print(f"{'Savings':>10} {'Price':>10} {'Rate':>10} {'Action':>15}")
        print("-" * 50)
        for si in range(len(sg)):
            for pi in range(len(pg)):
                for ri in range(len(rg)):
                    state = BuyRentState(si, pi, ri, owns)
                    nt = NonTerminal(state)
                    try:
                        action_dist = policy.act(nt)
                        action = action_dist.sample()
                        print(
                            f"{sg[si]:>10.1f} {pg[pi]:>10.1f} "
                            f"{rg[ri]:>10.3f} {action.name:>15}"
                        )
                    except KeyError:
                        pass