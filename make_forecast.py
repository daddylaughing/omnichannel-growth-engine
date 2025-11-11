# make_forecast.py — robust version (auto-detects date column)
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CLEAN = ROOT / "clean"
SRC = CLEAN / "cleaned_fmcg_omnichannel_sales.csv"
OUT = CLEAN / "forecast_channel_weekly.csv"

# -------- helpers --------
def guess_date_col(cols):
    cands = ["date", "orderdate", "transactiondate", "purchasedate",
             "weekstart", "week_start", "week", "week_of", "weekbegin", "weekstartdate"]
    low = {c: c.lower() for c in cols}
    # exact / common substrings
    for c in cols:
        lc = low[c]
        if lc in cands or "date" in lc or "week" in lc:
            return c
    return None

def guess_col(cols, options, substrings=None):
    low = {c: c.lower() for c in cols}
    for c in cols:
        if low[c] in options:
            return c
    if substrings:
        for c in cols:
            if any(s in low[c] for s in substrings):
                return c
    return None

# -------- load flexibly --------
df0 = pd.read_csv(SRC)  # no parse_dates yet
date_col = guess_date_col(df0.columns)
if not date_col:
    raise ValueError(f"Could not find a date-like column in {SRC}. Columns: {list(df0.columns)}")

chan_col = guess_col(df0.columns, {"channel","saleschannel","platform"})
rev_col  = guess_col(df0.columns, {"revenue","sales","amount","gmv","netsales","totalsales"}, substrings=["rev","sale","amount","gmv"])
if not chan_col or not rev_col:
    raise ValueError(f"Need Channel and Revenue columns. Found channel={chan_col}, revenue={rev_col}. Columns: {list(df0.columns)}")

# now parse dates properly
df0[date_col] = pd.to_datetime(df0[date_col], errors="coerce")
df = df0[[date_col, chan_col, rev_col]].dropna()
df.columns = ["Date", "Channel", "Revenue"]

out = []
for ch, g in df.groupby("Channel"):
    # weekly revenue series
    weekly = g.resample("W-MON", on="Date")["Revenue"].sum().reset_index()
    weekly.columns = ["ds", "y"]
    if len(weekly) < 12:
        continue
    try:
        from prophet import Prophet
        m = Prophet(seasonality_mode="multiplicative")
        m.fit(weekly)
        future = m.make_future_dataframe(periods=8, freq="W-MON")
        fc = m.predict(future)[["ds", "yhat", "yhat_lower", "yhat_upper"]]
    except Exception:
        # ARIMA fallback (no heavy build)
        from statsmodels.tsa.arima.model import ARIMA
        s = weekly.set_index("ds")["y"].asfreq("W-MON").fillna(method="ffill")
        model = ARIMA(s, order=(1,1,1))
        res = model.fit()
        steps = 8
        pred = res.get_forecast(steps=steps).summary_frame()
        fc = pred[["mean", "mean_ci_lower", "mean_ci_upper"]].reset_index()
        fc.columns = ["ds", "yhat", "yhat_lower", "yhat_upper"]
    fc["Channel"] = ch
    out.append(fc)

fc_all = pd.concat(out, ignore_index=True) if out else pd.DataFrame(columns=["ds","yhat","yhat_lower","yhat_upper","Channel"])
fc_all = fc_all.rename(columns={"ds": "Date"})
OUT.parent.mkdir(parents=True, exist_ok=True)
fc_all.to_csv(OUT, index=False)
print(f"✅ Saved forecast to {OUT}")