"""
Phase 2 Runner: Build the discretized Buy vs. Rent MDP and solve with DP.

Usage:
    python run_phase2.py
"""

from __future__ import annotations

import time
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Iterator, Mapping, Tuple

from rl.markov_process import NonTerminal
from rl.markov_decision_process import FiniteMarkovDecisionProcess
from rl.dynamic_programming import value_iteration_result, value_iteration
from rl.distribution import FiniteDistribution

from Project.phase2.buy_rent_mdp import (
    Action, BuyRentParams, BuyRentState,
    build_buy_rent_mdp, print_policy,
)


def analyze_policy(
    mdp: FiniteMarkovDecisionProcess[BuyRentState, Action],
    vf: Mapping[NonTerminal[BuyRentState], float],
    params: BuyRentParams,
):
    """Produce analysis plots of the optimal value function and policy."""
    sg = params.savings_grid
    pg = params.price_grid
    rg = params.rate_grid

    # ------------------------------------------------------------------
    # 1. Value function heatmap: savings vs home price
    #    (fix rate at middle index, show for renting and owning)
    # ------------------------------------------------------------------
    mid_rate = len(rg) // 2

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for idx, owns in enumerate([False, True]):
        vals = np.zeros((len(sg), len(pg)))
        for si in range(len(sg)):
            for pi in range(len(pg)):
                state = BuyRentState(si, pi, mid_rate, owns)
                nt = NonTerminal(state)
                vals[si, pi] = vf.get(nt, 0.0)

        im = axes[idx].imshow(
            vals, origin='lower', aspect='auto',
            extent=[pg[0], pg[-1], sg[0], sg[-1]],
            cmap='RdYlGn'
        )
        title = "Owning" if owns else "Renting"
        axes[idx].set_title(f"Value Function ({title}, rate={rg[mid_rate]:.2%})")
        axes[idx].set_xlabel("Home Price")
        axes[idx].set_ylabel("Savings")
        plt.colorbar(im, ax=axes[idx])

    plt.tight_layout()
    plt.savefig("value_function_heatmap.png", dpi=150, bbox_inches='tight')
    plt.show()
    print("Saved: value_function_heatmap.png")

    # ------------------------------------------------------------------
    # 2. Policy heatmap: savings vs home price
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    action_to_num = {
        Action.RENT_STAY: 0,
        Action.BUY: 1,
        Action.OWN_STAY: 2,
        Action.SELL: 3,
    }
    action_labels = ["Rent(stay)", "Buy", "Own(stay)", "Sell"]

    for idx, owns in enumerate([False, True]):
        policy_grid = np.full((len(sg), len(pg)), np.nan)
        for si in range(len(sg)):
            for pi in range(len(pg)):
                state = BuyRentState(si, pi, mid_rate, owns)
                nt = NonTerminal(state)
                if nt not in vf:
                    continue

                # Find greedy action
                action_map = mdp.mapping.get(nt, {})
                best_action = None
                best_val = -np.inf
                for a, dist in action_map.items():
                    q_val = sum(
                        p * (r + params.gamma * vf.get(
                            ns if isinstance(ns, NonTerminal) else nt, 0.0
                        ))
                        for (ns, r), p in dist
                    )
                    if q_val > best_val:
                        best_val = q_val
                        best_action = a

                if best_action is not None:
                    policy_grid[si, pi] = action_to_num[best_action]

        im = axes[idx].imshow(
            policy_grid, origin='lower', aspect='auto',
            extent=[pg[0], pg[-1], sg[0], sg[-1]],
            cmap='tab10', vmin=0, vmax=3,
        )
        title = "Owning" if owns else "Renting"
        axes[idx].set_title(f"Optimal Policy ({title}, rate={rg[mid_rate]:.2%})")
        axes[idx].set_xlabel("Home Price")
        axes[idx].set_ylabel("Savings")

    # Add legend
    from matplotlib.patches import Patch
    colors = plt.cm.tab10(np.linspace(0, 0.3, 4))
    legend_elements = [
        Patch(facecolor=plt.cm.tab10(i / 10), label=action_labels[i])
        for i in range(4)
    ]
    fig.legend(
        handles=legend_elements, loc='lower center',
        ncol=4, fontsize=10, bbox_to_anchor=(0.5, -0.02)
    )
    plt.tight_layout()
    plt.savefig("policy_heatmap.png", dpi=150, bbox_inches='tight')
    plt.show()
    print("Saved: policy_heatmap.png")

    # ------------------------------------------------------------------
    # 3. Buy threshold: for each home price, find min savings to buy
    # ------------------------------------------------------------------
    print("\n=== BUY THRESHOLD (Renting → Buy) ===")
    print(f"{'Rate':>10} {'Home Price':>12} {'Min Savings to Buy':>20}")
    print("-" * 45)
    for ri in range(len(rg)):
        for pi in range(len(pg)):
            threshold = None
            for si in range(len(sg)):
                state = BuyRentState(si, pi, ri, False)
                nt = NonTerminal(state)
                action_map = mdp.mapping.get(nt, {})

                best_action = None
                best_val = -np.inf
                for a, dist in action_map.items():
                    q_val = sum(
                        p * (r + params.gamma * vf.get(
                            ns if isinstance(ns, NonTerminal) else nt, 0.0
                        ))
                        for (ns, r), p in dist
                    )
                    if q_val > best_val:
                        best_val = q_val
                        best_action = a

                if best_action == Action.BUY:
                    threshold = sg[si]
                    break

            if threshold is not None:
                print(f"{rg[ri]:>10.3f} {pg[pi]:>12.1f} {threshold:>20.1f}")


def main():
    np.random.seed(42)

    params = BuyRentParams()

    print("=" * 60)
    print("Phase 2: Buy vs. Rent — Discretized DP")
    print("=" * 60)
    print(f"Grid sizes: savings={len(params.savings_grid)}, "
          f"prices={len(params.price_grid)}, "
          f"rates={len(params.rate_grid)}")
    print(f"Total states: "
          f"{len(params.savings_grid) * len(params.price_grid) * len(params.rate_grid) * 2}")
    print()

    # Build MDP
    print("Building MDP (Monte Carlo transition estimation)...")
    t0 = time.time()
    mdp, params = build_buy_rent_mdp(params, n_samples=200)
    t1 = time.time()
    print(f"  Done in {t1 - t0:.1f}s")
    print(f"  Non-terminal states: {len(mdp.non_terminal_states)}")
    print()

    # Solve with Value Iteration
    print("Running Value Iteration...")
    t0 = time.time()
    opt_vf, opt_policy = value_iteration_result(mdp, gamma=params.gamma)
    t1 = time.time()
    print(f"  Converged in {t1 - t0:.1f}s")
    print()

    # Analyze
    analyze_policy(mdp, opt_vf, params)


if __name__ == "__main__":
    main()
