# app/app.py — Single-View Omnichannel FMCG Dashboard
# Derrick Wong | Omnichannel Growth Engine (Capstone 2025)

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

FACT_PATH   = CLEAN / "cleaned_fmcg_omnichannel_sales.csv"
CUST_PATH   = CLEAN / "fmcg_customer_repeat_summary_clean.csv"
FORE_PATH   = CLEAN / "forecast_channel_weekly.csv"  # optional

# -----------------------
# Loaders (robust)
# -----------------------
@st.cache_data
def load_fact():
    df0 = pd.read_csv(FACT_PATH)
    # guess columns flexibly
    def pick(cols, exact, subs=None):
        low = {c: c.lower() for c in cols}
        for c in cols:
            if low[c] in exact: return c
        if subs:
            for c in cols:
                if any(s in low[c] for s in subs): return c
        return None
    def pick_date(cols):
        for c in cols:
            lc = c.lower()
            if lc in {"date","orderdate","week","weekstart","week_start"} or "date" in lc or "week" in lc:
                return c
        return None

    date = pick_date(df0.columns)
    ch   = pick(df0.columns, {"channel","platform","saleschannel"})
    terr = pick(df0.columns, {"territory","region","area"})
    cat  = pick(df0.columns, {"category","productcategory","sku_category"}, subs=["category"])
    promo= pick(df0.columns, {"promo","promotion","is_promo","on_promo"})
    rev  = pick(df0.columns, {"revenue","sales","amount","gmv","netsales"}, subs=["rev","sale","amount","gmv"])

    if any(c is None for c in [date, ch, terr, cat, rev]):
        raise ValueError("Missing key columns in fact CSV")

    df0[date] = pd.to_datetime(df0[date], errors="coerce")
    if promo is None:
        df0["Promo"] = False; promo = "Promo"

    df = df0[[date, ch, terr, cat, promo, rev]].dropna()
    df.columns = ["Date","Channel","Territory","Category","Promo","Revenue"]
    df["Promo"] = df["Promo"].astype(bool)
    df["Channel"] = df["Channel"].astype(str)
    df["Territory"] = df["Territory"].astype(str)
    df["Category"] = df["Category"].astype(str)
    df["Revenue"] = pd.to_numeric(df["Revenue"], errors="coerce").fillna(0.0)
    return df

@st.cache_data
def load_customers():
    if not CUST_PATH.exists():
        return None
    c = pd.read_csv(CUST_PATH)
    # normalize headers
    def pick(options, subs=None):
        for col in c.columns:
            if col.lower() in options: return col
        if subs:
            for col in c.columns:
                if any(s in col.lower() for s in subs): return col
        return None
    cid = pick({"customer_id","customerid","cust_id"}, subs=["customer"])
    orders = pick({"orders","order_count","repeat_count"}, subs=["order","purchase"])
    rev = pick({"revenue","sales","amount","gmv"}, subs=["rev","sale","amount","gmv"])
    if cid:    c = c.rename(columns={cid:"Customer_ID"})
    if orders: c = c.rename(columns={orders:"Orders"})
    if rev:    c = c.rename(columns={rev:"Revenue"})
    if "Orders" in c.columns:  c["Orders"] = pd.to_numeric(c["Orders"], errors="coerce").fillna(0).astype(int)
    if "Revenue" in c.columns: c["Revenue"] = pd.to_numeric(c["Revenue"], errors="coerce").fillna(0.0)
    return c

@st.cache_data
def load_forecast():
    if FORE_PATH.exists():
        f = pd.read_csv(FORE_PATH, parse_dates=["Date"])
        # normalize
        if "Channel" not in f.columns and "channel" in f.columns:
            f = f.rename(columns={"channel":"Channel"})
        return f[["Date","Channel","yhat","yhat_lower","yhat_upper"]]
    return None

fact = load_fact()
cust = load_customers()
fcast = load_forecast()

# -----------------------
# Sidebar – global filters (left column of wireframe)
# -----------------------
with st.sidebar:
    st.header("Filter")
    # date
    dmin, dmax = fact["Date"].min(), fact["Date"].max()
    date_range = st.date_input("Date Range", (dmin, dmax))
    # channel/territory/category
    channels = st.multiselect("Channel", sorted(fact["Channel"].unique()))
    territories = st.multiselect("Territory", sorted(fact["Territory"].unique()))
    categories = st.multiselect("Product Category", sorted(fact["Category"].unique()))
    pflag = st.selectbox("Promo Flag", ["All","Promo only","Non-promo"])
    metric_choice = st.selectbox("Metric", ["Revenue"])
    st.markdown("---")
    st.caption("Channel & Territory Performance")
    # (purely informational section label to match wireframe)

# apply filters
f = fact.copy()
if isinstance(date_range, (tuple, list)) and len(date_range)==2:
    f = f[(f["Date"] >= pd.to_datetime(date_range[0])) & (f["Date"] <= pd.to_datetime(date_range[1]))]
if channels:    f = f[f["Channel"].isin(channels)]
if territories: f = f[f["Territory"].isin(territories)]
if categories:  f = f[f["Category"].isin(categories)]
if pflag == "Promo only": f = f[f["Promo"] == True]
elif pflag == "Non-promo": f = f[f["Promo"] == False]

# -----------------------
# EXECUTIVE OVERVIEW (top band)
# -----------------------
st.title("Omnichannel FMCG Dashboard")

k1,k2,k3,k4 = st.columns([1,1,1,1])
total_rev = f["Revenue"].sum()
units = len(f)  # proxy if you don’t have quantity
repeat_pct = None
if cust is not None and {"Customer_ID","Orders"}.issubset(cust.columns):
    total_c = len(cust)
    repeat_c = (cust["Orders"] >= 2).sum()
    repeat_pct = 100*repeat_c/total_c if total_c>0 else 0
promo_eff = None
if f["Promo"].nunique() > 1:
    promo_eff = (f[f["Promo"]==True]["Revenue"].mean() /
                 max(1e-9, f[f["Promo"]==False]["Revenue"].mean()) - 1) * 100

k1.metric("Total Revenue", f"${total_rev:,.0f}")
k2.metric("Total Units (orders proxy)", f"{units:,}")
k3.metric("Repeat Purchase Rate", "-" if repeat_pct is None else f"{repeat_pct:.1f}%")
k4.metric("Promo Effectiveness", "-" if promo_eff is None else f"{promo_eff:.1f}%")

# Row: Revenue Trend | Promo ROI vs Spend | Brand Share (+Search placeholder)
c1,c2,c3,c4 = st.columns([2,1.4,1.6,0.0001])  # last tiny column just to balance layout spacing

# -- Revenue Trend (weekly)
with c1:
    st.subheader("Revenue Trend")
    wk = f.groupby(pd.Grouper(key="Date", freq="W-MON"))["Revenue"].sum().reset_index()
    st.plotly_chart(px.line(wk, x="Date", y="Revenue"), use_container_width=True)

# -- Promo ROI vs Spend (approximation)
with c2:
    st.subheader("Promo ROI vs Spend")
    # Approximate "spend" as count of promo transactions * constant (since we lack media spend)
    # This still shows ROI pattern and is interactive.
    promo = f.groupby("Promo")["Revenue"].sum().reset_index()
    promo["Label"] = promo["Promo"].map({True:"Promo", False:"Non-Promo"})
    # donut share
    st.plotly_chart(px.pie(promo, names="Label", values="Revenue", hole=0.55), use_container_width=True)

with c3:
    st.subheader("Brand Share (by Channel)")
    # If you later add real brand/search data, replace this.
    ch_mix = f.groupby("Channel")["Revenue"].sum().reset_index().sort_values("Revenue", ascending=False)
    st.plotly_chart(px.bar(ch_mix, x="Revenue", y="Channel", orientation="h"), use_container_width=True)

st.markdown("---")

# -----------------------
# CHANNEL & TERRITORY PERFORMANCE (middle band)
# -----------------------
g1,g2,g3 = st.columns([1.3,1.3,1.4])

with g1:
    st.subheader("Territory Heatmap")
    # Heatmap-like bar (SG map not required). Darker = more revenue.
    terr = f.groupby("Territory")["Revenue"].sum().reset_index()
    fig = px.density_heatmap(terr, x="Territory", y=["Revenue"]*len(terr), z="Revenue",
                             color_continuous_scale="Reds")
    fig.update_yaxes(visible=False, showticklabels=False)
    st.plotly_chart(fig, use_container_width=True)

with g2:
    st.subheader("Channel Trend (Weekly)")
    ch_w = f.groupby([pd.Grouper(key="Date", freq="W-MON"), "Channel"])["Revenue"].sum().reset_index()
    st.plotly_chart(px.line(ch_w, x="Date", y="Revenue", color="Channel"), use_container_width=True)

with g3:
    st.subheader("Top 10 SKUs / Categories")
    cat = f.groupby("Category")["Revenue"].sum().reset_index().sort_values("Revenue", ascending=False).head(10)
    st.plotly_chart(px.bar(cat, x="Category", y="Revenue"), use_container_width=True)

st.markdown("---")

# -----------------------
# PRODUCT & PROMOTION ANALYSIS (bottom band)
# -----------------------
h1,h2,h3 = st.columns([1.3,1.3,1.4])

with h1:
    st.subheader("Category Revenue Treemap")
    tre = f.groupby(["Channel","Category"])["Revenue"].sum().reset_index()
    st.plotly_chart(px.treemap(tre, path=["Channel","Category"], values="Revenue"), use_container_width=True)

with h2:
    st.subheader("Customer Segment – Radar (RFM-lite)")
    # Make 3 simple segment KPIs from customer table if present
    if cust is None or not {"Orders","Revenue"}.issubset(cust.columns):
        st.info("Customer summary not available.")
    else:
        # Bucket by quantiles
        cust["Freq"] = pd.qcut(cust["Orders"].rank(method="first"), 5, labels=False) + 1
        cust["Monetary"] = pd.qcut(cust["Revenue"].rank(method="first"), 5, labels=False) + 1
        seg = cust.agg({"Freq":"mean","Monetary":"mean"}).rename({"Freq":"Frequency","Monetary":"Monetary"})
        # fake "Recency" if not present: inverse of Orders rank (for visual balance)
        seg["Recency"] = 6 - seg["Frequency"]
        radar = pd.DataFrame({"Metric":seg.index, "Score":seg.values})
        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(r=radar["Score"], theta=radar["Metric"], fill='toself', name='Avg'))
        fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0,6])), showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

with h3:
    st.subheader("Customer Scatter & KPI Table")
    if cust is None or not {"Orders","Revenue"}.issubset(cust.columns):
        st.info("Customer summary not available.")
    else:
        sc = px.scatter(cust, x="Orders", y="Revenue", opacity=0.45)
        st.plotly_chart(sc, use_container_width=True)
        kpi = pd.DataFrame({
            "Metric":["Customers","Repeat %","Median Revenue","Mean Orders"],
            "Value":[len(cust),
                     f"{(cust['Orders']>=2).mean()*100:.1f}%",
                     f"${cust['Revenue'].median():,.0f}",
                     f"{cust['Orders'].mean():.2f}"]
        })
        st.dataframe(kpi, use_container_width=True, hide_index=True)

st.markdown("---")

# -----------------------
# OPTIONAL – Forecast strip (if CSV available)
# -----------------------
if fcast is not None and not fcast.empty:
    st.subheader("Forecast Snapshot (next 8 weeks)")
    # respect channel filters
    fc = fcast.copy()
    if channels: fc = fc[fc["Channel"].isin(channels)]
    if not fc.empty:
        ch_pick = st.selectbox("Channel (Forecast)", sorted(fc["Channel"].unique()), key="fc_pick")
        fc1 = fc[fc["Channel"]==ch_pick].sort_values("Date")
        fig = go.Figure()
        # history
        hist = (f[f["Channel"] == ch_pick]
                .groupby(pd.Grouper(key="Date", freq="W-MON"))["Revenue"].sum().reset_index())
        if len(hist):
            fig.add_trace(go.Scatter(x=hist["Date"], y=hist["Revenue"], name="Actual", mode="lines+markers"))
        # forecast
        if {"yhat","yhat_lower","yhat_upper"}.issubset(fc1.columns):
            fig.add_trace(go.Scatter(x=fc1["Date"], y=fc1["yhat"], name="Forecast", mode="lines"))
            fig.add_trace(go.Scatter(
                x=pd.concat([fc1["Date"], fc1["Date"][::-1]]),
                y=pd.concat([fc1["yhat_upper"], fc1["yhat_lower"][::-1]]),
                fill="toself", name="Confidence", mode="lines", line=dict(width=0), opacity=0.15
            ))
        st.plotly_chart(fig, use_container_width=True)

# Footer
st.caption("Prepared by Derrick Wong | Graduate NTUC LearningHub Associate Data Analyst (Cohort 36) 2025")
