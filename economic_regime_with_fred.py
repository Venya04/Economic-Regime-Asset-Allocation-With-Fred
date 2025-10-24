import pandas as pd
from fredapi import Fred
import os

# === FRED API ===
fred = Fred(api_key="9fc71a0e5a8fb6318c0f031f5b8e5f25")

# === Fetch Data ===
gdp = fred.get_series("GDPC1")
cpi = fred.get_series("CPIAUCSL")
yield_10y = fred.get_series("GS10")
yield_3m = fred.get_series("GS3M")
m2 = fred.get_series("M2SL")
velocity = fred.get_series("M2V")

# === Index Setup ===
gdp.index = pd.to_datetime(gdp.index)
cpi.index = pd.to_datetime(cpi.index)
yield_10y.index = pd.to_datetime(yield_10y.index)
yield_3m.index = pd.to_datetime(yield_3m.index)
m2.index = pd.to_datetime(m2.index)
velocity.index = pd.to_datetime(velocity.index)

# === Resample to Monthly ===
gdp = gdp.resample("ME").ffill()
yield_10y = yield_10y.resample("ME").ffill()
yield_3m = yield_3m.resample("ME").ffill()
m2 = m2.resample("ME").ffill()
velocity = velocity.resample("ME").ffill()
cpi.index = cpi.index + pd.offsets.MonthEnd(0)

# === Yield Curve ===
yield_curve = yield_10y - yield_3m
yield_curve.name = "yield_curve"

# === Build full monthly timeline ===
monthly_index = cpi.index.to_series().asfreq("ME").index

# === Create DataFrame with full date range ===
df = pd.DataFrame(index=monthly_index)
df.index.name = "date"
df = df.reset_index()

# === Merge all indicators ===

# Prepare each series with correct date columns
gdp_growth = gdp.pct_change().rename("gdp_growth").reset_index().rename(columns={"index": "date"})
cpi_inflation = cpi.pct_change().rename("inflation").reset_index().rename(columns={"index": "date"})
yield_curve_df = yield_curve.rename("yield_curve").reset_index().rename(columns={"index": "date"})
m2_growth = m2.pct_change().rename("m2_growth").reset_index().rename(columns={"index": "date"})
velocity_change = velocity.pct_change().rename("velocity_change").reset_index().rename(columns={"index": "date"})

# Merge into main dataframe
df = df.merge(gdp_growth, on="date", how="left")
df = df.merge(cpi_inflation, on="date", how="left")
df = df.merge(yield_curve_df, on="date", how="left")
df = df.merge(m2_growth, on="date", how="left")
df = df.merge(velocity_change, on="date", how="left")

# === Trends ===
df["growth_trend"] = df["gdp_growth"].rolling(window=12).mean()
df["inflation_trend"] = df["inflation"].rolling(window=12).mean()
df["yield_trend"] = df["yield_curve"].rolling(window=12).mean()
df["m2_trend"] = df["m2_growth"].rolling(window=12).mean()
df["velocity_trend"] = df["velocity_change"].rolling(window=12).mean()

# === Classify Regime ===
def classify_regime(row):
    gdp_valid = not pd.isna(row["growth_trend"])
    inflation_valid = not pd.isna(row["inflation_trend"])
    m2_valid = not pd.isna(row["m2_trend"])
    velocity_valid = not pd.isna(row["velocity_trend"])

    if sum([inflation_valid, m2_valid, velocity_valid]) < 2:
        return None

    gdp_up = row["gdp_growth"] > row["growth_trend"] if gdp_valid else None
    inflation_up = row["inflation"] > row["inflation_trend"] if inflation_valid else None
    m2_up = row["m2_growth"] > row["m2_trend"] if m2_valid else None
    velocity_up = row["velocity_change"] > row["velocity_trend"] if velocity_valid else None

    if gdp_up is not None:
        if gdp_up and inflation_up and (m2_up or velocity_up):
            return "Overheating"
        elif gdp_up and not inflation_up and (m2_up or velocity_up):
            return "Recovery"
        elif not gdp_up and inflation_up:
            return "Stagflation"
        elif not gdp_up and not m2_up and not velocity_up:
            return "Contraction"
        else:
            if gdp_up:
                return "Recovery"
            # else:
            #     return "Mixed"


    if inflation_up and (m2_up or velocity_up):
        return "Overheating"
    elif not inflation_up and (m2_up or velocity_up):
        return "Recovery"
    elif inflation_up and not m2_up and not velocity_up:
        return "Stagflation"

df["regime"] = df.apply(classify_regime, axis=1)

# === Portfolio Allocation ===
def allocation_by_regime(regime):
    if regime == "Overheating":
        return {"stocks": 0.39747521865407315, "crypto": 0.07001253172847896, "commodities": 0.1802910736997232, "stable_yield": 0.35222117591}
    elif regime == "Recovery":
        return {"stocks": 0.3928068906738552, "crypto": 0.029166000138361462, "commodities": 0.47369716493452413, "stable_yield": 0.10432994425}
    elif regime == "Stagflation":
        return {"stocks": 0.5441181588630634, "crypto": 0.08262672206168974, "commodities": 0, "stable_yield": 0.37325511907}
    elif regime == "Contraction":
        return {"stocks": 0.0, "crypto": 0.020920205293443395, "commodities": 0.6603384678663021, "stable_yield": 0.31874132684}
    else:
        return None

df["allocation"] = df["regime"].apply(allocation_by_regime)

# === Print Recent Results ===
classified = df[df["regime"].notna()]
classified_recent = classified[classified["date"] >= pd.to_datetime("2020-01-01")]

print("\n🗕 Regime Classification from 2020 to Latest:")
print(classified_recent[["date", "regime"]])

latest = classified_recent.iloc[-1]
print("\n🔞 Latest Classified Month:")
print("Date:", latest["date"].strftime("%Y-%m-%d"))
print("Regime:", latest["regime"])
print("Suggested Allocation:", latest["allocation"])

print("\n📊 Regime Count Since 2020:")
print(classified_recent["regime"].value_counts())

# === Save to Desktop ===
regime_df = df[["date", "regime"]].dropna()
desktop = os.path.join(os.path.expanduser("~"), "Desktop")
save_path = os.path.join(desktop, "regime_labels_expanded.csv")
regime_df.to_csv(save_path, index=False)
print("\n📅 Saved to:", save_path)



