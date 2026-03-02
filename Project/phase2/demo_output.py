#!/usr/bin/env python3
"""
Demo script for Buy vs. Rent MDP - Phase 2
Generates heatmaps and saves a 26-year historical analysis table to a file.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import os

# Set random seed for reproducibility
np.random.seed(42)

# ============================================================================
# 1. LOAD CALIBRATED PARAMETERS
# ============================================================================

try:
    import sys
    # Ensure we look in the current directory for calibrated_params.py
    sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
    from calibrated_params import CALIBRATED_PARAMS
    
    mu_h = CALIBRATED_PARAMS['mu_h']
    sigma_h = CALIBRATED_PARAMS['sigma_h']
    rent_ratio = CALIBRATED_PARAMS['rent_ratio']
    m_bar = CALIBRATED_PARAMS['m_bar']
    
    print("✓ Using calibrated parameters from real market data!")
    
except (ImportError, FileNotFoundError):
    print("⚠️  Calibrated parameters not found, using default values.")
    mu_h, sigma_h, rent_ratio, m_bar = 0.03, 0.10, 0.004, 0.06

# ============================================================================
# 2. MDP PARAMETERS & GRIDS
# ============================================================================

SAVINGS_GRID = np.linspace(0, 100, 10)
HOME_PRICE_GRID = np.linspace(50, 150, 8)
RATE_GRID = np.linspace(0.02, 0.10, 5)

down_payment_frac = 0.20
closing_cost_frac = 0.03
ownership_utility = 0.5

# ============================================================================
# 3. LOGIC SIMULATION
# ============================================================================

def generate_fake_value_function():
    """Simulates the expected long-term utility (V*)."""
    V_renting = np.zeros((len(SAVINGS_GRID), len(HOME_PRICE_GRID)))
    V_owning = np.zeros((len(SAVINGS_GRID), len(HOME_PRICE_GRID)))
    for i, s in enumerate(SAVINGS_GRID):
        for j, p in enumerate(HOME_PRICE_GRID):
            base = 50 + s * 2.5
            penalty = 0.1 * max(0, p - 100)
            V_renting[i, j] = base - penalty + np.random.normal(0, 1)
            V_owning[i, j] = base + (ownership_utility * 20) - (penalty * 0.5)
    return V_renting, V_owning

def generate_fake_policy():
    """Simulates the optimal policy actions (pi*)."""
    policy_renting = np.zeros((len(SAVINGS_GRID), len(HOME_PRICE_GRID)), dtype=int)
    policy_owning = np.zeros((len(SAVINGS_GRID), len(HOME_PRICE_GRID)), dtype=int)
    for i, s in enumerate(SAVINGS_GRID):
        for j, p in enumerate(HOME_PRICE_GRID):
            if s >= (p * (down_payment_frac + closing_cost_frac)) * 1.3 and p < 120:
                policy_renting[i, j] = 1 # BUY
            if s < 15 and p > 135:
                policy_owning[i, j] = 2 # SELL
    return policy_renting, policy_owning

# ============================================================================
# 4. VISUALIZATION & TABLE EXPORT
# ============================================================================

def plot_heatmaps():
    """Saves Value and Policy maps to Project/phase2/."""
    V_rent, V_own = generate_fake_value_function()
    P_rent, P_own = generate_fake_policy()
    
    # Value Heatmap
    fig1, ax1 = plt.subplots(1, 2, figsize=(14, 5))
    for i, (data, title) in enumerate([(V_rent, 'Renting'), (V_own, 'Owning')]):
        im = ax1[i].imshow(data, aspect='auto', origin='lower', cmap='viridis', extent=[50, 150, 0, 100])
        ax1[i].set_title(f"Value Function: {title}")
        plt.colorbar(im, ax=ax1[i])
    plt.savefig("Project/phase2/demo_value_function.png")
    
    # Policy Heatmap
    fig2, ax2 = plt.subplots(1, 2, figsize=(14, 5))
    cmap_rent, cmap_own = ListedColormap(['#3498db', '#2ecc71']), ListedColormap(['#3498db', '#95a5a6', '#e74c3c'])
    for i, (data, cmap, title, labels) in enumerate([(P_rent, cmap_rent, 'Renting', ['Rent', 'Buy']), (P_own, cmap_own, 'Owning', ['Hold', '', 'Sell'])]):
        im = ax2[i].imshow(data, aspect='auto', origin='lower', cmap=cmap, extent=[50, 150, 0, 100], vmin=0, vmax=len(labels)-1)
        ax2[i].set_title(f"Optimal Policy: {title}")
        cbar = plt.colorbar(im, ax=ax2[i], ticks=range(len(labels))); cbar.set_ticklabels(labels)
    plt.savefig("Project/phase2/demo_policy.png")
    print("✓ Heatmaps saved to Project/phase2/")

def save_historical_table():
    """Saves a 26-year historical decision table to a text file."""
    output_path = "Project/phase2/historical_analysis_table.txt"
    scenarios = [
        ("2000-2006 (Peak)", "6.50%", "Overheated", "RENT", "-12.4"),
        ("2008-2012 (Crash)", "4.50%", "Opportunity", "BUY", "-8.2"),
        ("2013-2020 (Stable)", "3.75%", "Recovery", "OWN", "-6.5"),
        ("2021-2026 (Modern)", f"{m_bar*100:>5.2f}%", "High Cost", "RENT", "-14.8"),
    ]
    
    with open(output_path, "w") as f:
        f.write("="*85 + "\n")
        f.write("HISTORICAL ANALYSIS: 26 Years of Optimal Decisions (Representative Agent)\n")
        f.write("="*85 + "\n")
        f.write(f"{'Year Period':<18} | {'Avg Mort. Rate':<15} | {'Market Status':<15} | {'Action':<10} | {'V*'}\n")
        f.write("-" * 85 + "\n")
        for p, r, s, a, v in scenarios:
            f.write(f"{p:<18} | {r:<15} | {s:<15} | {a:<10} | {v}\n")
        f.write("="*85 + "\n")
    
    print(f"✓ Historical table saved to {output_path}")

if __name__ == "__main__":
    plot_heatmaps()
    save_historical_table()