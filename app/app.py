# app/app.py
# Omnichannel Growth Engine – Streamlit Dashboard
# Derrick Wong | NTUC LearningHub ADA (C36, 2025)

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go

# -----------------------------
# Paths & loaders
# -----------------------------
ROOT_DIR = Path(__file__).resolve().parents[1]
CLEAN_DIR = ROOT_DIR / "clean"

FACT_PATH = CLEAN_DIR / "cleaned_fmcg_omnichannel_sales.csv"
CUST_SUM_PATH = CLEAN_DIR / "fmcg_customer_repeat_summary_clean.csv"
FORECAST_PATH = CLEAN_DIR / "forecast_channel_weekly.csv"  # <- Option A precomputed

st.set_page_config(page_title="Omnichannel Growth Engine", layout="wide")

@st.cache_data
def load_fact():
    """Load the main clean fact table with flexible column detection."""
    df0 = pd.read_csv(FACT_PATH)
    # guess columns
    def guess_date(cols):
        for c in cols:
            lc = c.lower()
            if lc in {"date","orderdate","week","weekstart","week_start"} or "date" in lc or "week" in lc:
                return c
        return None
    def guess(cols, exact, subs=None):
        for c in cols:
            if c.lower() in exact:
                return c
        if subs:
            for c in cols:
                if any(s in c.lower() for s in subs):
                    return c
        return None

    date_col = guess_date(df0.columns)
    chan_col = guess(df0.columns, {"channel","platform","saleschannel"})
    terr_col = guess(df0.columns, {"territory","region","area"})
    cat_col  = guess(df0.columns, {"category","productcategory","sku_category"}, subs=["category"])
    promo_col= guess(df0.columns, {"promo","promotion","is_promo","on_promo"})
    rev_col  = guess(df0.columns, {"revenue","sales","amount","gmv","netsales"}, subs=["rev","sale","amount","gmv"])

    # minimal requirements
    must = [date_col, chan_col, terr_col, cat_col, rev_col]
    if any(m is None for m in must):
        raise ValueError(f"Missing expected columns in {FACT_PATH}. Found: {list(df0.columns)}")

    df0[date_col] = pd.to_datetime(df0[date_col], errors="coerce")
    if promo_col is None:
        df0["Promo"] = False
        promo_col = "Promo"

    df = df0[[date_col, chan_col, terr_col, cat_col, promo_col, rev_col]].dropna()
    df.columns = ["Date","Channel","Territory","Category","Promo","Revenue"]
    # sensible types
    df["Channel"]   = df["Channel"].astype("category")
    df["Territory"] = df["Territory"].astype("category")
    df["Category"]  = df["Category"].astype("category")
    df["Promo"]     = df["Promo"].astype(bool)
    df["Revenue"]   = pd.to_numeric(df["Revenue"], errors="coerce").fillna(0.0)
    return df

@st.cache_data
def load_cust_summary():
    if not CUST_SUM_PATH.exists():
        return None
    cs = pd.read_csv(CUST_SUM_PATH)
    # try to standardize common headers
    rename_map = {c.lower(): c for c in cs.columns}
    # find columns
    def pick(options, subs=None):
        for c in cs.columns:
            if c.lower() in options: return c
        if subs:
            for c in cs.columns:
                if any(s in c.lower() for s in subs): return c
        return None
    cust_col = pick({"customer_id","customerid","cust_id"}, subs=["customer"])
    orders_col = pick({"orders","order_count","repeat_count"}, subs=["order","purchase"])
    rev_col = pick({"revenue","sales","amount","gmv"}, subs=["rev","sale","amount","gmv"])

    if cust_col: cs = cs.rename(columns={cust_col:"Customer_ID"})
    if orders_col: cs = cs.rename(columns={orders_col:"Orders"})
    if rev_col: cs = cs.rename(columns={rev_col:"Revenue"})
    if "Orders" in cs.columns:
        cs["Orders"] = pd.to_numeric(cs["Orders"], errors="coerce").fillna(0).astype(int)
    if "Revenue" in cs.columns:
        cs["Revenue"] = pd.to_numeric(cs["Revenue"], errors="coerce").fillna(0.0)
    return cs

@st.cache_data
def load_forecast():
    """Load precomputed channel-weekly forecast if available."""
    if FORECAST_PATH.exists():
        fcast = pd.read_csv(FORECAST_PATH, parse_dates=["Date"])
        # standardize expected cols
        cols = {c.lower(): c for c in fcast.columns}
        fcast = fcast.rename(columns={
            cols.get("yhat","yhat"): "yhat",
            cols.get("yhat_lower","yhat_lower"): "yhat_lower",
            cols.get("yhat_upper","yhat_upper"): "yhat_upper",
            cols.get("channel","Channel"): "Channel",
        })
        return fcast
    return None

# Load
df = load_fact()
cust = load_cust_summary()
fcast = load_forecast()

# -----------------------------
# Sidebar filters
# -----------------------------
st.sidebar.header("Filters")
min_date, max_date = df["Date"].min(), df["Date"].max()
date_range = st.sidebar.date_input("Date range", (min_date, max_date))
channels = st.sidebar.multiselect("Channel", sorted(df["Channel"].cat.categories.tolist()))
territories = st.sidebar.multiselect("Territory", sorted(df["Territory"].cat.categories.tolist()))
categories = st.sidebar.multiselect("Category", sorted(df["Category"].cat.categories.tolist()))
promo_flag = st.sidebar.selectbox("Promo filter", ["All", "Promo only", "Non-promo"])

# apply filters
f = df.copy()
if isinstance(date_range, (list, tuple)) and len(date_range) == 2:
    f = f[(f["Date"] >= pd.to_datetime(date_range[0])) & (f["Date"] <= pd.to_datetime(date_range[1]))]
if channels:   f = f[f["Channel"].isin(channels)]
if territories:f = f[f["Territory"].isin(territories)]
if categories: f = f[f["Category"].isin(categories)]
if promo_flag == "Promo only": f = f[f["Promo"] == True]
elif promo_flag == "Non-promo": f = f[f["Promo"] == False]

# -----------------------------
# KPIs
# -----------------------------
col1, col2, col3, col4 = st.columns(4)
total_rev = f["Revenue"].sum()
aov = f.groupby(pd.Grouper(key="Date", freq="W-MON"))["Revenue"].sum().mean()
promo_uplift = None
if "Promo" in f.columns and f["Promo"].nunique() > 1:
    promo_avg = f[f["Promo"]==True]["Revenue"].mean()
    non_avg   = f[f["Promo"]==False]["Revenue"].mean()
    if non_avg and not np.isnan(non_avg):
        promo_uplift = (promo_avg/non_avg - 1.0) * 100

repeat_pct = None
if cust is not None and {"Customer_ID","Orders"}.issubset(cust.columns):
    total_c = len(cust)
    repeat_c = (cust["Orders"] >= 2).sum()
    if total_c > 0:
        repeat_pct = 100 * repeat_c / total_c

col1.metric("Total Revenue", f"${total_rev:,.0f}")
col2.metric("Avg Weekly Revenue", f"${(0 if pd.isna(aov) else aov):,.0f}")
col3.metric("Promo Uplift", f"{(0 if promo_uplift is None else promo_uplift):.1f}%")
col4.metric("Repeat Customers", f"{(0 if repeat_pct is None else repeat_pct):.1f}%")

st.markdown("---")

# -----------------------------
# Tabs
# -----------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["Overview", "Channel & Territory", "Product & Promo", "Customers", "Forecast"]
)

# ---------------- Overview ----------------
with tab1:
    st.subheader("Channel Sales Trend (Weekly)")
    ch_week = f.groupby([pd.Grouper(key="Date", freq="W-MON"), "Channel"])["Revenue"].sum().reset_index()
    if len(ch_week) == 0:
        st.info("No data for current filter.")
    else:
        fig = px.line(ch_week, x="Date", y="Revenue", color="Channel")
        fig.update_layout(margin=dict(l=10,r=10,t=40,b=10))
        st.plotly_chart(fig, use_container_width=True)

# ------------- Channel & Territory --------
with tab2:
    left, right = st.columns(2)
    with left:
        st.subheader("Territory Sales Leaderboard")
        terr = f.groupby("Territory")["Revenue"].sum().reset_index().sort_values("Revenue", ascending=False)
        st.plotly_chart(px.bar(terr, x="Revenue", y="Territory", orientation="h"), use_container_width=True)
    with right:
        st.subheader("Channel Mix")
        ch_mix = f.groupby("Channel")["Revenue"].sum().reset_index()
        st.plotly_chart(px.pie(ch_mix, names="Channel", values="Revenue", hole=0.35), use_container_width=True)

# ------------- Product & Promo ------------
with tab3:
    left, right = st.columns(2)
    with left:
        st.subheader("Category Contribution (Pareto)")
        cat = f.groupby("Category")["Revenue"].sum().reset_index().sort_values("Revenue", ascending=False)
        cat["cum_share"] = 100 * cat["Revenue"].cumsum() / cat["Revenue"].sum()
        fig = go.Figure()
        fig.add_bar(x=cat["Category"], y=cat["Revenue"], name="Revenue")
        fig.add_trace(go.Scatter(x=cat["Category"], y=cat["cum_share"], yaxis="y2", name="Cum %"))
        fig.update_layout(
            yaxis2=dict(overlaying="y", side="right", range=[0,100], title="Cum %"),
            margin=dict(l=10,r=10,t=40,b=10)
        )
        st.plotly_chart(fig, use_container_width=True)
    with right:
        st.subheader("Promo vs Non-Promo – Average Revenue")
        g = f.groupby("Promo")["Revenue"].mean().reset_index()
        g["Promo"] = g["Promo"].map({True:"Promo", False:"Non-Promo"})
        st.plotly_chart(px.bar(g, x="Promo", y="Revenue"), use_container_width=True)

# ---------------- Customers ----------------
with tab4:
    st.subheader("Revenue vs Orders (Customer Segments)")
    if cust is None or not {"Customer_ID","Orders","Revenue"}.issubset(cust.columns):
        st.info("Customer summary file not found or missing columns.")
    else:
        # Base scatter
        fig = px.scatter(
            cust, x="Orders", y="Revenue",
            opacity=0.45, labels={"Orders":"Orders per Customer"},
            hover_data=["Customer_ID"]
        )

        # Optional lightweight best-fit (no statsmodels)
        try:
            x = cust["Orders"].to_numpy(dtype=float)
            y = cust["Revenue"].to_numpy(dtype=float)
            if len(cust) >= 3 and np.isfinite(x).all() and np.isfinite(y).all():
                m, b = np.polyfit(x, y, 1)  # slope, intercept
                xline = np.linspace(x.min(), x.max(), 100)
                yline = m * xline + b
                fig.add_traces([
                    go.Scatter(x=xline, y=yline, mode="lines", name="Best-fit (polyfit)", line=dict(width=2))
                ])
        except Exception:
            pass

        # Reference medians
        med_rev = float(np.nanmedian(cust["Revenue"]))
        fig.add_hline(y=med_rev, line_dash="dot", annotation_text=f"Median Rev: {med_rev:,.0f}")

        fig.update_layout(margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Top-right cluster = high-value frequent buyers. Build CRM journeys to retain them.")
with tab5:
    st.subheader("Channel Demand Forecast (next 8 weeks)")
    if fcast is None or fcast.empty:
        st.info("No forecast file found. Add clean/forecast_channel_weekly.csv to enable this tab.")
    else:
        # respect channel filter
        fc = fcast.copy()
        if "Channel" in fc.columns and channels:
            fc = fc[fc["Channel"].isin(channels)]
        ch_list = sorted(fc["Channel"].dropna().unique())
        if not ch_list:
            st.info("No channels available for current filters.")
        else:
            ch_pick = st.selectbox("Channel", ch_list, key="fc_channel")
            fc1 = fc[fc["Channel"] == ch_pick].sort_values("Date")

            fig = go.Figure()
            # history line from filtered fact
            hist = (f[f["Channel"] == ch_pick]
                    .groupby(pd.Grouper(key="Date", freq="W-MON"))["Revenue"]
                    .sum().reset_index())
            if len(hist):
                fig.add_trace(go.Scatter(x=hist["Date"], y=hist["Revenue"], name="Actual", mode="lines+markers"))

            if {"yhat","yhat_lower","yhat_upper"}.issubset(fc1.columns):
                fig.add_trace(go.Scatter(x=fc1["Date"], y=fc1["yhat"], name="Forecast", mode="lines"))
                fig.add_trace(go.Scatter(
                    x=pd.concat([fc1["Date"], fc1["Date"][::-1]]),
                    y=pd.concat([fc1["yhat_upper"], fc1["yhat_lower"][::-1]]),
                    fill="toself", name="Confidence", mode="lines", line=dict(width=0), opacity=0.15
                ))
            fig.update_layout(margin=dict(l=10,r=10,t=40,b=10))
            st.plotly_chart(fig, use_container_width=True)

# footer
st.caption("Prepared by Derrick Wong | Graduate NTUC LearningHub Associate Data Analyst (Cohort 36) 2025")
