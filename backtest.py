import yfinance as yf
import pandas as pd
import numpy as np

# === Settings ===
START_DATE = "2010-01-01"
END_DATE = "2024-12-31"
STABLECOIN_MONTHLY_YIELD = 0.05 / 12  # 5% APY

# === Tickers and Proxies ===
TICKERS = {
    "stocks": "SPY",
    "crypto": "BTC-USD",
    "commodities": "GLD",
    "cash": None  # includes both USD & stablecoins
}

# === Load Price Data ===
data = {}
for asset, ticker in TICKERS.items():
    if ticker:
        df = yf.download(ticker, start=START_DATE, end=END_DATE, progress=False)
        df = df["Adj Close"] if "Adj Close" in df.columns else df["Close"]
        data[asset] = df

prices = pd.concat(data.values(), axis=1)
prices.columns = [k for k in data.keys() if data[k] is not None]
prices = prices.dropna()

# === Load Regime Labels ===
regime_df = pd.read_csv("regime_labels_expanded.csv", parse_dates=["date"])
regime_df.set_index("date", inplace=True)
regime_df = regime_df.reindex(prices.index, method="ffill")

# === Load Optimized Allocations ===
opt_alloc_df = pd.read_csv("optimal_allocations.csv")
opt_alloc_df.set_index("regime", inplace=True)
allocations = opt_alloc_df.to_dict(orient="index")

# === Ensure All Regimes Include Stablecoin Allocation ===
for alloc in allocations.values():
    if "stablecoins" not in alloc:
        alloc["stablecoins"] = 0.0  # If missing, set to 0
    if "cash" not in alloc:
        alloc["cash"] = 0.1  # Default to 10% cash if not included
    total_weight = sum(alloc.values())
    for k in alloc:
        alloc[k] = alloc[k] / total_weight  # Normalize to 100%


# === Calculate Daily Returns ===
returns = prices.pct_change().dropna()
# === Calculate Daily Returns ===
returns = prices.pct_change().dropna()

# Add constant yields for synthetic assets
CASH_DAILY_YIELD = 0.045 / 12  # For example, assume 4.5% monthly
returns["cash"] = CASH_DAILY_YIELD


returns["stablecoins"] = STABLECOIN_MONTHLY_YIELD
returns["cash"] = CASH_DAILY_YIELD


# === Compute Portfolio Returns Based on Regime ===
# === Compute Portfolio Returns with Event-Driven Rebalancing ===

portfolio_returns = []
prev_regime = None

# Start with equal weights as default (in case regime is NaN initially)
current_weights = {asset: 0.25 for asset in TICKERS.keys()}

for date in returns.index:
    # Get the current regime
    regime = regime_df.loc[date, "regime"]

    # If we don't know the regime for this date, skip it
    if pd.isna(regime):
        portfolio_returns.append(np.nan)
        continue

    # Rebalance only when the regime changes
    if regime != prev_regime:
        if regime in allocations:
            current_weights = allocations[regime]
            prev_regime = regime
        else:
            # If we somehow get an unknown regime, use previous weights
            print(f"⚠️ Warning: Unknown regime on {date}: {regime}")

    # Compute daily return using current weights
    daily_return = sum(returns.loc[date, asset] * current_weights[asset] for asset in current_weights)
    portfolio_returns.append(daily_return)

# Convert to pandas Series for further analysis
portfolio_returns = pd.Series(portfolio_returns, index=returns.index)


# === Performance Metrics ===
def compute_metrics(rets):
    mean_daily = rets.mean()
    std_daily = rets.std()

    # 🔍 Print debugging values
    print("📈 Mean daily return:", mean_daily)
    print("📉 Std dev daily:", std_daily)

    # Performance metrics
    cagr = (1 + mean_daily) ** 252 - 1
    volatility = std_daily * np.sqrt(252)
    sharpe = (mean_daily / std_daily) * np.sqrt(252)

    # Drawdown
    drawdown = (1 + rets).cumprod().div((1 + rets).cumprod().cummax()) - 1
    max_dd = drawdown.min()

    return {
        "CAGR": cagr,
        "Volatility": volatility,
        "Sharpe": sharpe,
        "Max Drawdown": max_dd
    }


# === Results ===
metrics = compute_metrics(portfolio_returns.dropna())

print("\n📊 Portfolio Performance Based on Dynamic Regime Allocations:")
for k, v in metrics.items():
    if k == "Sharpe":
        print(f"{k}: {v:.2f}")
    else:
        print(f"{k}: {v:.2%}")

