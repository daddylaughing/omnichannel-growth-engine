# app/app.py — Multi-Tab Omnichannel FMCG Dashboard
# Prepared by Derrick Wong | NTUC LearningHub ADA (Cohort 36) 2025

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go

# -----------------------
# Setup & paths
# -----------------------
st.set_page_config(page_title="Omnichannel FMCG Dashboard", layout="wide")
ROOT = Path(__file__).resolve().parents[1]
CLEAN = ROOT / "clean"

FACT_PATH = CLEAN / "cleaned_fmcg_omnichannel_sales.csv"
CUST_PATH = CLEAN / "fmcg_customer_repeat_summary_clean.csv"
FORE_PATH = CLEAN / "forecast_channel_weekly.csv"  # optional

# -----------------------
# Loaders (robust to column names)
# -----------------------
@st.cache_data
def load_fact():
    df0 = pd.read_csv(FACT_PATH)

    def pick(cols, exact, subs=None):
        for c in cols:
            if c.lower() in exact: return c
        if subs:
            for c in cols:
                if any(s in c.lower() for s in subs): return c
        return None

    def pick_date(cols):
        for c in cols:
            lc = c.lower()
            if lc in {"date","orderdate","week","weekstart","week_start"} or "date" in lc or "week" in lc:
                return c
        return None

    date  = pick_date(df0.columns)
    ch    = pick(df0.columns, {"channel","platform","saleschannel"})
    terr  = pick(df0.columns, {"territory","region","area"})
    cat   = pick(df0.columns, {"category","productcategory","sku_category"}, subs=["category"])
    promo = pick(df0.columns, {"promo","promotion","is_promo","on_promo"})
    rev   = pick(df0.columns, {"revenue","sales","amount","gmv","netsales"}, subs=["rev","sale","amount","gmv"])

    if any(x is None for x in [date, ch, terr, cat, rev]):
        raise ValueError("Missing key columns in fact CSV")

    df0[date] = pd.to_datetime(df0[date], errors="coerce")
    if promo is None:
        df0["Promo"] = False; promo = "Promo"

    df = df0[[date, ch, terr, cat, promo, rev]].dropna()
    df.columns = ["Date","Channel","Territory","Category","Promo","Revenue"]
    df["Promo"] = df["Promo"].astype(bool)
    df["Revenue"] = pd.to_numeric(df["Revenue"], errors="coerce").fillna(0.0)
    df["Channel"] = df["Channel"].astype(str)
    df["Territory"] = df["Territory"].astype(str)
    df["Category"] = df["Category"].astype(str)
    return df

@st.cache_data
def load_customers():
    if not CUST_PATH.exists():
        return None
    c = pd.read_csv(CUST_PATH)

    def pick(options, subs=None):
        for col in c.columns:
            if col.lower() in options: return col
        if subs:
            for col in c.columns:
                if any(s in col.lower() for s in subs): return col
        return None

    cid    = pick({"customer_id","customerid","cust_id"}, subs=["customer"])
    orders = pick({"orders","order_count","repeat_count"}, subs=["order","purchase"])
    rev    = pick({"revenue","sales","amount","gmv"}, subs=["rev","sale","amount","gmv"])

    if cid:    c = c.rename(columns={cid:"Customer_ID"})
    if orders: c = c.rename(columns={orders:"Orders"})
    if rev:    c = c.rename(columns={rev:"Revenue"})

    if "Orders" in c.columns:  c["Orders"]  = pd.to_numeric(c["Orders"], errors="coerce").fillna(0).astype(int)
    if "Revenue" in c.columns: c["Revenue"] = pd.to_numeric(c["Revenue"], errors="coerce").fillna(0.0)
    return c

@st.cache_data
def load_forecast():
    if FORE_PATH.exists():
        f = pd.read_csv(FORE_PATH, parse_dates=["Date"])
        if "Channel" not in f.columns and "channel" in f.columns:
            f = f.rename(columns={"channel":"Channel"})
        return f[["Date","Channel","yhat","yhat_lower","yhat_upper"]]
    return None

# Load data
fact  = load_fact()
cust  = load_customers()
fcast = load_forecast()

# -----------------------
# Sidebar (global filters)
# -----------------------
with st.sidebar:
    st.header("Filters")
    dmin, dmax = fact["Date"].min(), fact["Date"].max()
    date_range = st.date_input("Date Range", (dmin, dmax))
    channels   = st.multiselect("Channel",   sorted(fact["Channel"].unique()))
    territories= st.multiselect("Territory", sorted(fact["Territory"].unique()))
    categories = st.multiselect("Category",  sorted(fact["Category"].unique()))
    pflag      = st.selectbox("Promo Flag", ["All","Promo only","Non-promo"])

# Apply filters
f = fact.copy()
if isinstance(date_range, (tuple, list)) and len(date_range) == 2:
    f = f[(f["Date"] >= pd.to_datetime(date_range[0])) & (f["Date"] <= pd.to_datetime(date_range[1]))]
if channels:    f = f[f["Channel"].isin(channels)]
if territories: f = f[f["Territory"].isin(territories)]
if categories:  f = f[f["Category"].isin(categories)]
if pflag == "Promo only": f = f[f["Promo"] == True]
elif pflag == "Non-promo": f = f[f["Promo"] == False]

# -----------------------
# Tabs
# -----------------------
st.title("Omnichannel FMCG Dashboard")
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["Overview", "Channel & Territory", "Product & Promotion", "Customers", "Forecast"]
)

# ========== OVERVIEW ==========
with tab1:
    k1, k2, k3, k4 = st.columns(4)
    total_rev = f["Revenue"].sum()
    units = len(f)  # using row count as a proxy when quantity not available
    repeat_pct = None
    if cust is not None and {"Customer_ID","Orders"}.issubset(cust.columns):
        total_c = len(cust)
        repeat_c = (cust["Orders"] >= 2).sum()
        repeat_pct = 100 * repeat_c / total_c if total_c > 0 else 0
    promo_eff = None
    if f["Promo"].nunique() > 1:
        base = f[f["Promo"] == False]["Revenue"].mean()
        lift = f[f["Promo"] == True]["Revenue"].mean()
        if base and not np.isnan(base):
            promo_eff = (lift / base - 1) * 100

    k1.metric("Total Revenue", f"${total_rev:,.0f}")
    k2.metric("Total Units (orders proxy)", f"{units:,}")
    k3.metric("Repeat Purchase Rate", "-" if repeat_pct is None else f"{repeat_pct:.1f}%")
    k4.metric("Promo Effectiveness", "-" if promo_eff is None else f"{promo_eff:.1f}%")

    st.markdown("---")
    left, mid, right = st.columns([2, 1.3, 1.7])

    with left:
        st.subheader("Revenue Trend (Weekly)")
        wk = f.groupby(pd.Grouper(key="Date", freq="W-MON"))["Revenue"].sum().reset_index()
        st.plotly_chart(px.line(wk, x="Date", y="Revenue"), use_container_width=True)

    with mid:
        st.subheader("Promo vs Non-Promo Share")
        promo = f.groupby("Promo")["Revenue"].sum().reset_index()
        promo["Label"] = promo["Promo"].map({True: "Promo", False: "Non-Promo"})
        st.plotly_chart(px.pie(promo, names="Label", values="Revenue", hole=0.55),
                        use_container_width=True)

    with right:
        st.subheader("Channel Share (Revenue)")
        ch_mix = f.groupby("Channel")["Revenue"].sum().reset_index().sort_values("Revenue", ascending=False)
        st.plotly_chart(px.bar(ch_mix, x="Revenue", y="Channel", orientation="h"),
                        use_container_width=True)

# ========== CHANNEL & TERRITORY ==========
with tab2:
    c1, c2 = st.columns(2)
    with c1:
    st.subheader("Territory Heatmap")
    terr = f.groupby("Territory")["Revenue"].sum().reset_index().sort_values("Revenue", ascending=False)
    if terr.empty:
        st.info("No data for current filters.")
    else:
        fig = px.bar(
            terr, x="Revenue", y="Territory",
            orientation="h", color="Revenue",
            color_continuous_scale="Reds",
            labels={"Revenue": "Revenue", "Territory": "Territory"},
            title=None
        )
        fig.update_layout(coloraxis_showscale=True, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)


# ========== PRODUCT & PROMOTION ==========
with tab3:
    p1, p2 = st.columns(2)
    with p1:
        st.subheader("Category Revenue Treemap")
        tre = f.groupby(["Channel","Category"])["Revenue"].sum().reset_index()
        st.plotly_chart(px.treemap(tre, path=["Channel","Category"], values="Revenue"),
                        use_container_width=True)
    with p2:
        st.subheader("Promo vs Non-Promo — Avg Revenue")
        g = f.groupby("Promo")["Revenue"].mean().reset_index()
        g["Promo"] = g["Promo"].map({True:"Promo", False:"Non-Promo"})
        st.plotly_chart(px.bar(g, x="Promo", y="Revenue"), use_container_width=True)

# ========== CUSTOMERS ==========
with tab4:
    st.subheader("Revenue vs Orders (Customer Segments)")
    if cust is None or not {"Orders","Revenue"}.issubset(cust.columns):
        st.info("Customer summary file not found or missing columns.")
    else:
        fig = px.scatter(cust, x="Orders", y="Revenue", opacity=0.45,
                         labels={"Orders":"Orders per Customer"},
                         hover_data=["Customer_ID"])
        # lightweight best-fit (no statsmodels)
        try:
            x = cust["Orders"].to_numpy(dtype=float)
            y = cust["Revenue"].to_numpy(dtype=float)
            if len(cust) >= 3 and np.isfinite(x).all() and np.isfinite(y).all():
                m, b = np.polyfit(x, y, 1)
                xline = np.linspace(x.min(), x.max(), 100)
                yline = m * xline + b
                fig.add_trace(go.Scatter(x=xline, y=yline, mode="lines",
                                         name="Best-fit (polyfit)", line=dict(width=2)))
        except Exception:
            pass
        med_rev = float(np.nanmedian(cust["Revenue"]))
        fig.add_hline(y=med_rev, line_dash="dot",
                      annotation_text=f"Median Rev: {med_rev:,.0f}")
        st.plotly_chart(fig, use_container_width=True)

        # quick KPI table
        kpi = pd.DataFrame({
            "Metric": ["Customers","Repeat %","Median Revenue","Mean Orders"],
            "Value": [
                len(cust),
                f"{(cust['Orders']>=2).mean()*100:.1f}%",
                f"${cust['Revenue'].median():,.0f}",
                f"{cust['Orders'].mean():.2f}"
            ]
        })
        st.dataframe(kpi, use_container_width=True, hide_index=True)

# ========== FORECAST ==========
with tab5:
    st.subheader("Channel Demand Forecast (next 8 weeks)")
    if fcast is None or fcast.empty:
        st.info("No forecast file found. Add clean/forecast_channel_weekly.csv to enable this tab.")
    else:
        fc = fcast.copy()
        if channels: fc = fc[fc["Channel"].isin(channels)]
        if fc.empty:
            st.info("No channels available for current filters.")
        else:
            ch_pick = st.selectbox("Channel", sorted(fc["Channel"].unique()), key="fc_pick")
            fc1 = fc[fc["Channel"] == ch_pick].sort_values("Date")

            fig = go.Figure()
            # history
            hist = (f[f["Channel"] == ch_pick]
                    .groupby(pd.Grouper(key="Date", freq="W-MON"))["Revenue"].sum().reset_index())
            if len(hist):
                fig.add_trace(go.Scatter(x=hist["Date"], y=hist["Revenue"],
                                         name="Actual", mode="lines+markers"))
            # forecast band
            if {"yhat","yhat_lower","yhat_upper"}.issubset(fc1.columns):
                fig.add_trace(go.Scatter(x=fc1["Date"], y=fc1["yhat"],
                                         name="Forecast", mode="lines"))
                fig.add_trace(go.Scatter(
                    x=pd.concat([fc1["Date"], fc1["Date"][::-1]]),
                    y=pd.concat([fc1["yhat_upper"], fc1["yhat_lower"][::-1]]),
                    fill="toself", name="Confidence", mode="lines",
                    line=dict(width=0), opacity=0.15
                ))
            st.plotly_chart(fig, use_container_width=True)
