#!/usr/bin/env python3
"""
Calibrate Buy vs. Rent MDP parameters from real Zillow and FRED data.

This script:
1. Loads home price, rent, and mortgage rate data
2. Calculates historical statistics (drift, volatility, correlations)
3. Updates buy_rent_mdp.py with calibrated parameters
4. Generates visualization showing the historical data

Usage:
    python calibrate_from_real_data.py
    
Required files:
    - Metro_zhvi_uc_sfrcondo_tier_0_33_0_67_sm_sa_month.csv (home prices)
    - Metro_zori_uc_sfrcondomfr_sm_month.csv (rent prices)
    - MORTGAGE30US.csv (mortgage rates)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

print("="*70)
print("CALIBRATING MDP PARAMETERS FROM REAL DATA")
print("="*70)

# ============================================================================
# 1. LOAD DATA
# ============================================================================

print("\n📂 Loading data files...")

# Home prices (ZHVI)
home_df = pd.read_csv('Project/phase2/data/buy_prices.csv')
print(f"   ✓ Loaded {len(home_df)} rows from home prices CSV")

# Rent prices (ZORI)
rent_df = pd.read_csv('Project/phase2/data/rent_prices.csv')
print(f"   ✓ Loaded {len(rent_df)} rows from rent prices CSV")

# Mortgage rates (FRED)
mort_df = pd.read_csv('Project/phase2/data/mortgages.csv')
print(f"   ✓ Loaded {len(mort_df)} rows from mortgage rates CSV")

# ============================================================================
# 2. EXTRACT UNITED STATES DATA
# ============================================================================

print("\n🇺🇸 Extracting US national data...")

# Find United States row
us_home_row = home_df[home_df['RegionName'] == 'United States']
us_rent_row = rent_df[rent_df['RegionName'] == 'United States']

if us_home_row.empty or us_rent_row.empty:
    print("   ⚠️  Could not find 'United States' row. Using first row as fallback.")
    us_home_row = home_df.iloc[[0]]
    us_rent_row = rent_df.iloc[[0]]

# Extract date columns (format: YYYY-MM-DD or YYYY-MM-31)
home_cols = [col for col in home_df.columns if col.count('-') == 2]
rent_cols = [col for col in rent_df.columns if col.count('-') == 2]

# Get time series data
home_prices = us_home_row[home_cols].values.flatten()
rent_prices = us_rent_row[rent_cols].values.flatten()
home_dates = pd.to_datetime(home_cols)
rent_dates = pd.to_datetime(rent_cols)

# Remove NaN values
home_valid = ~np.isnan(home_prices)
home_prices = home_prices[home_valid]
home_dates = home_dates[home_valid]

rent_valid = ~np.isnan(rent_prices)
rent_prices = rent_prices[rent_valid]
rent_dates = rent_dates[rent_valid]

print(f"   ✓ Home prices: {len(home_prices)} months from {home_dates[0].date()} to {home_dates[-1].date()}")
print(f"   ✓ Rent prices: {len(rent_prices)} months from {rent_dates[0].date()} to {rent_dates[-1].date()}")

# Process mortgage rates
mort_df['observation_date'] = pd.to_datetime(mort_df['observation_date'])
mort_df = mort_df.sort_values('observation_date')
mort_df = mort_df.dropna()
mortgage_rates = mort_df['MORTGAGE30US'].values / 100  # Convert to decimal
mortgage_dates = mort_df['observation_date'].values

print(f"   ✓ Mortgage rates: {len(mortgage_rates)} weeks from {pd.to_datetime(mortgage_dates[0]).date()} to {pd.to_datetime(mortgage_dates[-1]).date()}")

# ============================================================================
# 3. CALCULATE HOME PRICE STATISTICS
# ============================================================================

print("\n📊 Calculating home price statistics...")

# Calculate monthly returns
monthly_returns = np.diff(np.log(home_prices))

# Annualize
mu_h_monthly = np.mean(monthly_returns)
sigma_h_monthly = np.std(monthly_returns)

mu_h_annual = mu_h_monthly * 12
sigma_h_annual = sigma_h_monthly * np.sqrt(12)

print(f"   • Mean monthly return: {mu_h_monthly*100:.3f}%")
print(f"   • Monthly volatility: {sigma_h_monthly*100:.3f}%")
print(f"   • Annual drift (μ_h): {mu_h_annual*100:.2f}%")
print(f"   • Annual volatility (σ_h): {sigma_h_annual*100:.2f}%")

# Calculate price changes over different periods
recent_5yr = home_prices[-60:] if len(home_prices) >= 60 else home_prices
recent_return = (recent_5yr[-1] / recent_5yr[0]) ** (1/5) - 1
print(f"   • Recent 5-year CAGR: {recent_return*100:.2f}%")

# ============================================================================
# 4. CALCULATE RENT-TO-PRICE RATIO
# ============================================================================

print("\n🏠 Calculating rent-to-price ratio...")

# Find overlapping dates
common_dates = np.intersect1d(
    home_dates.strftime('%Y-%m'),
    rent_dates.strftime('%Y-%m')
)

if len(common_dates) == 0:
    print("   ⚠️  No overlapping dates found, using separate averages")
    rent_ratio_monthly = np.mean(rent_prices) / np.mean(home_prices)
else:
    # Get matching data
    home_mask = np.isin(home_dates.strftime('%Y-%m'), common_dates)
    rent_mask = np.isin(rent_dates.strftime('%Y-%m'), common_dates)
    
    matched_home = home_prices[home_mask]
    matched_rent = rent_prices[rent_mask]
    
    # Calculate ratio over time
    rent_ratios = matched_rent / matched_home
    rent_ratio_monthly = np.mean(rent_ratios)
    
    print(f"   • Matched {len(common_dates)} months of data")
    print(f"   • Rent-to-price ratio range: {np.min(rent_ratios)*100:.3f}% - {np.max(rent_ratios)*100:.3f}%")

print(f"   • Average monthly rent ratio: {rent_ratio_monthly*100:.3f}%")
print(f"   • Average annual rent ratio: {rent_ratio_monthly*12*100:.2f}%")

# ============================================================================
# 5. CALCULATE MORTGAGE RATE STATISTICS
# ============================================================================

print("\n💰 Calculating mortgage rate statistics...")

# Use monthly average mortgage rates for consistency
mort_monthly = mort_df.set_index('observation_date').resample('ME')['MORTGAGE30US'].mean() / 100
mort_monthly = mort_monthly.dropna()

m_bar = np.mean(mortgage_rates)
sigma_m = np.std(np.diff(mortgage_rates))

# Estimate mean reversion (simple AR(1) model)
rate_changes = np.diff(mortgage_rates)
rate_levels = mortgage_rates[:-1]
rate_deviations = rate_levels - m_bar

# Simple linear regression: Δr = -κ(r - r̄) + ε
if len(rate_deviations) > 0 and np.std(rate_deviations) > 0:
    kappa_m = -np.cov(rate_changes, rate_deviations)[0,1] / np.var(rate_deviations)
    kappa_m = max(0.01, min(kappa_m, 1.0))  # Bound between 0.01 and 1.0
else:
    kappa_m = 0.20  # Default

print(f"   • Long-run mean (m̄): {m_bar*100:.2f}%")
print(f"   • Volatility (σ_m): {sigma_m*100:.3f}%")
print(f"   • Mean reversion speed (κ_m): {kappa_m:.3f}")
print(f"   • Current rate (latest): {mortgage_rates[-1]*100:.2f}%")

# Rate statistics by period
rate_min = np.min(mortgage_rates)
rate_max = np.max(mortgage_rates)
rate_current = mortgage_rates[-1]
rate_5yr_avg = np.mean(mortgage_rates[-260:]) if len(mortgage_rates) >= 260 else np.mean(mortgage_rates)

print(f"   • Historical range: {rate_min*100:.2f}% - {rate_max*100:.2f}%")
print(f"   • 5-year average: {rate_5yr_avg*100:.2f}%")

# ============================================================================
# 6. SUMMARY OF CALIBRATED PARAMETERS
# ============================================================================

print("\n" + "="*70)
print("CALIBRATED MDP PARAMETERS")
print("="*70)

params_summary = f"""
Home Price Dynamics:
  mu_h (annual drift):        {mu_h_annual:.4f}  ({mu_h_annual*100:.2f}%)
  sigma_h (annual volatility): {sigma_h_annual:.4f}  ({sigma_h_annual*100:.2f}%)

Rent:
  rent_ratio (monthly):        {rent_ratio_monthly:.6f}  ({rent_ratio_monthly*100:.3f}% of home price)

Mortgage Rates:
  m_bar (long-run mean):       {m_bar:.4f}  ({m_bar*100:.2f}%)
  sigma_m (weekly volatility): {sigma_m:.6f}  ({sigma_m*100:.3f}%)
  kappa_m (mean reversion):    {kappa_m:.4f}
"""

print(params_summary)

# ============================================================================
# 7. GENERATE VISUALIZATIONS
# ============================================================================

print("\n📈 Generating visualizations...")

fig, axes = plt.subplots(2, 2, figsize=(16, 10))

# Plot 1: Home Prices
ax1 = axes[0, 0]
ax1.plot(home_dates, home_prices / 1000, 'b-', linewidth=2, label='US Home Prices')
ax1.set_xlabel('Year', fontsize=11)
ax1.set_ylabel('Home Price ($1000s)', fontsize=11)
ax1.set_title('US Median Home Prices (Zillow ZHVI)', fontsize=13, fontweight='bold')
ax1.grid(True, alpha=0.3)
ax1.legend()

# Plot 2: Rent Prices
ax2 = axes[0, 1]
ax2.plot(rent_dates, rent_prices, 'g-', linewidth=2, label='US Rent')
ax2.set_xlabel('Year', fontsize=11)
ax2.set_ylabel('Monthly Rent ($)', fontsize=11)
ax2.set_title('US Median Rent (Zillow ZORI)', fontsize=13, fontweight='bold')
ax2.grid(True, alpha=0.3)
ax2.legend()

# Plot 3: Mortgage Rates
ax3 = axes[1, 0]
ax3.plot(pd.to_datetime(mortgage_dates), mortgage_rates * 100, 'r-', linewidth=1.5, alpha=0.7)
ax3.axhline(m_bar * 100, color='k', linestyle='--', linewidth=2, label=f'Long-run mean: {m_bar*100:.2f}%')
ax3.set_xlabel('Year', fontsize=11)
ax3.set_ylabel('Mortgage Rate (%)', fontsize=11)
ax3.set_title('30-Year Fixed Mortgage Rates (Freddie Mac)', fontsize=13, fontweight='bold')
ax3.grid(True, alpha=0.3)
ax3.legend()

# Plot 4: Rent-to-Price Ratio
ax4 = axes[1, 1]
if len(common_dates) > 0:
    matched_dates = pd.to_datetime([d + '-01' for d in common_dates])
    ax4.plot(matched_dates, rent_ratios * 100, 'purple', linewidth=2)
    ax4.axhline(rent_ratio_monthly * 100, color='k', linestyle='--', linewidth=2, 
                label=f'Average: {rent_ratio_monthly*100:.3f}%')
ax4.set_xlabel('Year', fontsize=11)
ax4.set_ylabel('Monthly Rent / Home Price (%)', fontsize=11)
ax4.set_title('Rent-to-Price Ratio Over Time', fontsize=13, fontweight='bold')
ax4.grid(True, alpha=0.3)
ax4.legend()

plt.tight_layout()
plt.savefig('Project/phase2/real_data_analysis.png', dpi=150, bbox_inches='tight')
print("   ✓ Saved: real_data_analysis.png")

# ============================================================================
# 8. SAVE CALIBRATED PARAMETERS TO FILE
# ============================================================================

print("\n💾 Saving calibrated parameters...")

params_dict = {
    'mu_h': mu_h_annual,
    'sigma_h': sigma_h_annual,
    'rent_ratio': rent_ratio_monthly,
    'm_bar': m_bar,
    'sigma_m': sigma_m,
    'kappa_m': kappa_m,
}

# Save as Python dict for easy import
with open('Project/phase2/calibrated_params.py', 'w') as f:
    f.write('"""Calibrated parameters from real market data."""\n\n')
    f.write('CALIBRATED_PARAMS = {\n')
    for key, val in params_dict.items():
        f.write(f'    "{key}": {val:.6f},\n')
    f.write('}\n')

print("   ✓ Saved: calibrated_params.py")

# Also save as text report
with open('Project/phase2/calibration_report.txt', 'w') as f:
    f.write("="*70 + "\n")
    f.write("BUY VS. RENT MDP - PARAMETER CALIBRATION REPORT\n")
    f.write("="*70 + "\n\n")
    f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    
    f.write("DATA SOURCES:\n")
    f.write(f"  - Home Prices: Zillow ZHVI ({len(home_prices)} months)\n")
    f.write(f"  - Rent: Zillow ZORI ({len(rent_prices)} months)\n")
    f.write(f"  - Mortgage Rates: Freddie Mac ({len(mortgage_rates)} weeks)\n\n")
    
    f.write(params_summary)
    
    f.write("\n\nINTERPRETATION:\n")
    f.write(f"  - Home prices have grown at ~{mu_h_annual*100:.1f}% per year historically\n")
    f.write(f"  - Annual volatility of {sigma_h_annual*100:.1f}% means prices can swing ±{sigma_h_annual*100:.0f}% in a typical year\n")
    f.write(f"  - Monthly rent is ~{rent_ratio_monthly*100:.2f}% of home price (~{rent_ratio_monthly*12*100:.1f}% annually)\n")
    f.write(f"  - Mortgage rates have averaged {m_bar*100:.1f}% and revert toward this mean\n")
    f.write(f"  - Higher kappa_m = {kappa_m:.2f} means rates adjust quickly to shocks\n")

print("   ✓ Saved: calibration_report.txt")

print("\n" + "="*70)
print("✅ CALIBRATION COMPLETE!")
print("="*70)
print("\nNext steps:")
print("  1. Review the visualizations in real_data_analysis.png")
print("  2. Check calibrated_params.py for the parameter values")
print("  3. Run demo_output.py to see the updated MDP with real data")
print("="*70)
