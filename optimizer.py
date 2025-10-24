import pandas as pd
import numpy as np
from scipy.optimize import minimize

# Load data
returns = pd.read_csv("asset_returns_monthly.csv", parse_dates=["Date"], index_col="Date")
regimes = pd.read_csv("regime_labels_expanded.csv", parse_dates=["date"], index_col="date")

# Add Period column
returns["Period"] = returns.index.to_period("M")
regimes["Period"] = regimes.index.to_period("M")

# Ensure required columns exist
if "stablecoins" not in returns.columns:
    print("⚠️ Adding synthetic 'stablecoins' column (0% return).")
    returns["stablecoins"] = 0.0

if "cash" not in returns.columns:
    print("⚠️ Adding synthetic 'cash' column (0% return).")
    returns["cash"] = 0.0

# Define asset groups
all_assets = [col for col in returns.columns if col not in ["Period"]]
risky_assets = [a for a in all_assets if a not in ["stablecoins", "cash"]]
full_asset_list = risky_assets + ["stablecoins", "cash"]

# Merge datasets
merged = pd.merge(returns, regimes, on="Period", how="inner")
merged.set_index("Period", inplace=True)

# Objective Function
def negative_sharpe(weights, mean_returns, cov_matrix, risk_free=0.0):
    risky_len = len(mean_returns)
    risky_weights = weights[:risky_len]  # only risky assets are used for Sharpe
    port_return = np.dot(risky_weights, mean_returns)
    port_vol = np.sqrt(np.dot(risky_weights.T, np.dot(cov_matrix, risky_weights)))
    if port_vol == 0:
        return np.inf
    return -(port_return - risk_free) / port_vol

# Constraints & bounds
min_cash = 0.1

def get_constraints(num_assets):
    return [
        {"type": "eq", "fun": lambda w: np.sum(w) - 1},  # full allocation
        {"type": "ineq", "fun": lambda w: w[-1] - min_cash}  # min cash
    ]

# Store results
optimal_allocations = {}

# Optimization loop
for regime in merged["regime"].unique():
    print(f"\n🔍 Optimizing for regime: {regime}")
    subset = merged[merged["regime"] == regime]
    subset_risky = subset[risky_assets].fillna(0)

    if len(subset_risky) < 2:
        print(f"⚠️ Skipping regime {regime}: not enough data.")
        continue

    mean_returns = subset_risky.mean()
    cov_matrix = subset_risky.cov()

    init_guess = np.full(len(full_asset_list), 1 / len(full_asset_list))
    bounds = [(0.0, 1.0)] * len(full_asset_list)

    result = minimize(
        negative_sharpe,
        init_guess,
        args=(mean_returns, cov_matrix),
        method="SLSQP",
        bounds=bounds,
        constraints=get_constraints(len(full_asset_list))
    )

    if result.success:
        allocation = dict(zip(full_asset_list, result.x))
        optimal_allocations[regime] = allocation
        print(f"✅ Success: {regime}")
    else:
        print(f"❌ Failure for {regime}: {result.message}")

# Save results
if optimal_allocations:
    df_opt = pd.DataFrame(optimal_allocations).T
    df_opt.index.name = "regime"
    df_opt.to_csv("optimal_allocations.csv")
    print("✅ Saved optimal_allocations.csv")
else:
    print("⚠️ No successful optimizations. CSV not saved.")

