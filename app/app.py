# app/app.py — Omnichannel FMCG Dashboard (Multi-tab)
# Prepared for Derrick Wong | NTUC LearningHub ADA Capstone 2025

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


# ----------------------------------------------------
# Page config
# ----------------------------------------------------
st.set_page_config(
    page_title="Omnichannel FMCG Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ----------------------------------------------------
# Paths & data loading helpers
# ----------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
CLEAN = ROOT / "clean"
DATA_CANDIDATES = [
    CLEAN / "cleaned_fmcg_omnichannel_sales.csv",
    ROOT / "cleaned_fmcg_omnichannel_sales.csv",
]

CUST_CANDIDATES = [
    CLEAN / "fmcg_customer_repeat_summary_clean.csv",
    ROOT / "fmcg_customer_repeat_summary_clean.csv",
    CLEAN / "fmcg_customer_repeat_summary.csv",
    ROOT / "fmcg_customer_repeat_summary.csv",
]

SKU_CANDIDATES = [
    CLEAN / "fmcg_sku_reference_clean.csv",
    ROOT / "fmcg_sku_reference_clean.csv",
    CLEAN / "fmcg_sku_reference.csv",
    ROOT / "fmcg_sku_reference.csv",
]


def _first_existing(paths):
    for p in paths:
        if p.exists():
            return p
    return None


@st.cache_data(show_spinner=False)
def load_fact() -> pd.DataFrame:
    """Load main fact table and normalise columns."""
    src = _first_existing(DATA_CANDIDATES)
    if src is None:
        st.error("Could not find cleaned_fmcg_omnichannel_sales.csv")
        st.stop()

    df = pd.read_csv(src)

    # Standardise column names
    cols = {c.lower(): c for c in df.columns}
    # Date column
    if "week_start" in cols:
        date_col = cols["week_start"]
    elif "date" in cols:
        date_col = cols["date"]
    else:
        raise ValueError("Could not find a date column (Week_Start / Date)")

    df.rename(columns={date_col: "Date"}, inplace=True)
    df["Date"] = pd.to_datetime(df["Date"])

    # Other key columns
    rename_map = {}
    for src_name, std in [
        ("territory", "Territory"),
        ("channel", "Channel"),
        ("category", "Category"),
        ("sku_id", "SKU_ID"),
        ("customer_id", "Customer_ID"),
    ]:
        for c in df.columns:
            if c.lower() == src_name:
                rename_map[c] = std
    if rename_map:
        df.rename(columns=rename_map, inplace=True)

    # Revenue / units
    if "Revenue" not in df.columns:
        # try to construct
        rev = None
        price_col = next((c for c in df.columns if c.lower() == "unit_price"), None)
        units_col = next((c for c in df.columns if c.lower() in ("units", "quantity")), None)
        if price_col and units_col:
            rev = df[price_col] * df[units_col]
        if rev is None:
            raise ValueError("No Revenue column and could not infer from Unit_Price x Units")
        df["Revenue"] = rev

    if "Units" not in df.columns:
        units_col = next((c for c in df.columns if c.lower() in ("units", "quantity")), None)
        if units_col:
            df.rename(columns={units_col: "Units"}, inplace=True)

    # Promo flag -> boolean
    promo_col = None
    for c in df.columns:
        if c.lower() in ("promo_flag", "promo", "is_promo"):
            promo_col = c
            break
    if promo_col is not None:
        df["Promo"] = df[promo_col].astype(int).astype(bool)
    else:
        df["Promo"] = False

    # Orders_6m proxy for repeat behaviour
    orders_col = next((c for c in df.columns if c.lower() in ("orders_6m", "orders6m", "order_count")), None)
    if orders_col and orders_col != "Orders_6m":
        df.rename(columns={orders_col: "Orders_6m"}, inplace=True)

    return df


@st.cache_data(show_spinner=False)
def load_customers():  # returns DataFrame or None
    src = _first_existing(CUST_CANDIDATES)
    if src is None:
        return None
    df = pd.read_csv(src)
    # normalise
    if "Customer_ID" not in df.columns:
        for c in df.columns:
            if c.lower() == "customer_id":
                df.rename(columns={c: "Customer_ID"}, inplace=True)
                break
    return df


@st.cache_data(show_spinner=False)
def load_sku():  # returns DataFrame or None
    src = _first_existing(SKU_CANDIDATES)
    if src is None:
        return None
    df = pd.read_csv(src)
    if "SKU_ID" not in df.columns:
        for c in df.columns:
            if c.lower() == "sku_id":
                df.rename(columns={c: "SKU_ID"}, inplace=True)
                break
    return df


@st.cache_data(show_spinner=False)
def build_forecast(df: pd.DataFrame) -> pd.DataFrame:
    """Simple moving-average forecast per channel (no external libs needed)."""
    # aggregate weekly revenue by channel
    wk = (
        df.groupby(["Date", "Channel"], as_index=False)["Revenue"]
        .sum()
        .sort_values("Date")
    )
    # 4-week moving average forecast shifted forward
    forecasts = []
    for ch, ch_df in wk.groupby("Channel"):
        ch_df = ch_df.sort_values("Date").copy()
        ch_df["yhat"] = ch_df["Revenue"].rolling(window=4, min_periods=2).mean().shift(1)
        ch_df["yhat_lower"] = ch_df["yhat"] * 0.9
        ch_df["yhat_upper"] = ch_df["yhat"] * 1.1
        ch_df["Channel"] = ch
        forecasts.append(ch_df)
    fc = pd.concat(forecasts, ignore_index=True)
    return fc.dropna(subset=["yhat"])


fact = load_fact()
cust = load_customers()
sku_ref = load_sku()
fcast = build_forecast(fact)


# ----------------------------------------------------
# Sidebar filters
# ----------------------------------------------------
with st.sidebar:
    st.header("Filters")

    min_date = fact["Date"].min()
    max_date = fact["Date"].max()

    date_range = st.date_input(
        "Date Range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date,
    )

    ch_all = ["All"] + sorted(fact["Channel"].dropna().unique().tolist())
    channel_sel = st.selectbox("Channel", ch_all)

    terr_all = ["All"] + sorted(fact["Territory"].dropna().unique().tolist())
    territory_sel = st.selectbox("Territory", terr_all)

    cat_all = ["All"] + sorted(fact["Category"].dropna().unique().tolist())
    category_sel = st.selectbox("Category", cat_all)

    promo_opt = ["All", "Promo Only", "Non-Promo Only"]
    promo_sel = st.selectbox("Promo Flag", promo_opt)


# Apply filters
f = fact.copy()
start, end = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
f = f[(f["Date"] >= start) & (f["Date"] <= end)]

if channel_sel != "All":
    f = f[f["Channel"] == channel_sel]
if territory_sel != "All":
    f = f[f["Territory"] == territory_sel]
if category_sel != "All":
    f = f[f["Category"] == category_sel]
if promo_sel == "Promo Only":
    f = f[f["Promo"]]
elif promo_sel == "Non-Promo Only":
    f = f[~f["Promo"]]

if f.empty:
    st.warning("No data for the current filter selection. Please adjust the filters.")
    st.stop()


# ----------------------------------------------------
# Helper metrics
# ----------------------------------------------------
def compute_promo_effectiveness(df: pd.DataFrame):
    g = df.groupby("Promo")["Revenue"].mean()
    if True in g.index and False in g.index and g[False] > 0:
        return (g[True] / g[False] - 1.0) * 100.0
    return None


def compute_repeat_rate(df: pd.DataFrame):
    if "Orders_6m" not in df.columns:
        return None
    cust_grp = df.groupby("Customer_ID")["Orders_6m"].max()
    if cust_grp.empty:
        return None
    repeat = (cust_grp >= 2).mean() * 100.0
    return repeat


promo_eff = compute_promo_effectiveness(f)
repeat_rate = compute_repeat_rate(f)


# ----------------------------------------------------
# Layout – Title & Tabs
# ----------------------------------------------------
st.title("Omnichannel FMCG Dashboard")

tabs = st.tabs(["Overview", "Channel & Territory", "Product & Promotion", "Customers", "Forecast"])

# ----------------------------------------------------
# 1. OVERVIEW TAB
# ----------------------------------------------------
with tabs[0]:
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)

    total_rev = f["Revenue"].sum()
    total_units = f.get("Units", pd.Series(index=f.index, data=np.nan)).sum()

    with kpi1:
        st.metric("Total Revenue", f"${total_rev:,.0f}")
    with kpi2:
        st.metric("Total Units (orders proxy)", f"{total_units:,.0f}")
    with kpi3:
        if repeat_rate is not None:
            st.metric("Repeat Purchase Rate", f"{repeat_rate:,.1f}%")
        else:
            st.metric("Repeat Purchase Rate", "N/A")
    with kpi4:
        if promo_eff is not None:
            st.metric("Promo Effectiveness", f"{promo_eff:,.1f}%")
        else:
            st.metric("Promo Effectiveness", "N/A")

    st.markdown("---")

    # Revenue trend + Promo share + Channel share
    top_row = st.columns([2, 1.5, 1.5])

    with top_row[0]:
        st.subheader("Revenue Trend (Weekly)")
        tr = (
            f.groupby("Date", as_index=False)["Revenue"]
            .sum()
            .sort_values("Date")
        )
        fig = px.line(tr, x="Date", y="Revenue")
        fig.update_traces(mode="lines+markers")
        st.plotly_chart(fig, use_container_width=True)

    with top_row[1]:
        st.subheader("Promo vs Non-Promo Share")

        promo = (
            f.groupby("Promo")["Revenue"]
            .sum()
            .reset_index()
        )

        if promo.empty:
            st.info("No Promo / Non-Promo data for current filters.")
        else:
            promo["Label"] = promo["Promo"].map({True: "Promo", False: "Non-Promo"})

            # If only one class present, add a 0-value other class so the pie renders nicely
            if promo["Label"].nunique() == 1:
                existing = promo["Label"].iloc[0]
                missing = "Promo" if existing == "Non-Promo" else "Non-Promo"
                promo = pd.concat(
                    [promo,
                     pd.DataFrame([{"Promo": None, "Label": missing, "Revenue": 0.0}])],
                    ignore_index=True,
                )

            fig_pie = px.pie(
                promo,
                names="Label",
                values="Revenue",
                hole=0.55,
            )
            fig_pie.update_layout(showlegend=True)
            st.plotly_chart(fig_pie, use_container_width=True)

    with top_row[2]:
        st.subheader("Channel Share (Revenue)")
        ch = (
            f.groupby("Channel", as_index=False)["Revenue"]
            .sum()
            .sort_values("Revenue", ascending=False)
        )
        fig_ch = px.bar(ch, x="Revenue", y="Channel", orientation="h")
        st.plotly_chart(fig_ch, use_container_width=True)


# ----------------------------------------------------
# 2. CHANNEL & TERRITORY TAB
# ----------------------------------------------------
with tabs[1]:
    st.subheader("Territory Performance")

    col_map, col_trend = st.columns([1.2, 2.0])

    with col_map:
        terr = (
            f.groupby("Territory", as_index=False)["Revenue"]
            .sum()
            .sort_values("Revenue", ascending=False)
        )
        if terr.empty:
            st.info("No data by territory for current filters.")
        else:
            fig_terr = px.bar(
                terr,
                x="Territory",
                y="Revenue",
                title="Revenue by Territory",
            )
            st.plotly_chart(fig_terr, use_container_width=True)

    with col_trend:
        st.subheader("Channel Trend (Weekly)")
        ch_wk = (
            f.groupby(["Date", "Channel"], as_index=False)["Revenue"]
            .sum()
            .sort_values(["Date", "Channel"])
        )
        if ch_wk.empty:
            st.info("No channel trend data for current filters.")
        else:
            fig_trend = px.line(
                ch_wk,
                x="Date",
                y="Revenue",
                color="Channel",
            )
            fig_trend.update_traces(mode="lines+markers")
            st.plotly_chart(fig_trend, use_container_width=True)

    st.markdown("### Top 10 SKUs by Revenue")
    top_sku = (
        f.groupby(["SKU_ID", "Category"], as_index=False)["Revenue"]
        .sum()
        .sort_values("Revenue", ascending=False)
        .head(10)
    )
    st.dataframe(top_sku, use_container_width=True)


# ----------------------------------------------------
# 3. PRODUCT & PROMOTION TAB
# ----------------------------------------------------
with tabs[2]:
    st.subheader("Product & Promotion Analysis")

    col_tree, col_bar = st.columns([2.0, 1.5])

    with col_tree:
        st.markdown("#### Category Revenue Treemap")
        tree = (
            f.groupby(["Channel", "Category"], as_index=False)["Revenue"]
            .sum()
        )
        if tree.empty:
            st.info("No product data for current filters.")
        else:
            fig_tree = px.treemap(
                tree,
                path=["Channel", "Category"],
                values="Revenue",
            )
            st.plotly_chart(fig_tree, use_container_width=True)

    with col_bar:
        st.markdown("#### Promo vs Non-Promo – Avg Revenue per Order")
        promo_grp = f.groupby("Promo")["Revenue"].mean().reset_index()
        if promo_grp.empty:
            st.info("No promo data for current filters.")
        else:
            promo_grp["Label"] = promo_grp["Promo"].map({True: "Promo", False: "Non-Promo"})
            fig_bar = px.bar(promo_grp, x="Label", y="Revenue")
            st.plotly_chart(fig_bar, use_container_width=True)


# ----------------------------------------------------
# 4. CUSTOMERS TAB
# ----------------------------------------------------
with tabs[3]:
    st.subheader("Customer Segment Insights")

    if cust is None:
        st.info("Customer repeat summary file not found. Showing simple distribution from fact table.")
        if "Orders_6m" in f.columns:
            hist = f.groupby("Customer_ID")["Orders_6m"].max().reset_index()
            fig_hist = px.histogram(hist, x="Orders_6m", nbins=20)
            fig_hist.update_layout(xaxis_title="Orders per Customer (6m)", yaxis_title="Customers")
            st.plotly_chart(fig_hist, use_container_width=True)
    else:
        c = cust.copy()
        st.markdown("#### Orders vs Revenue (Customer Level)")
        fig_scatter = px.scatter(
            c,
            x="Orders_6m",
            y="Total_Revenue_6m",
            size="Total_Units_6m",
            color="Channels_Used",
            hover_data=["Customer_ID"],
        )
        fig_scatter.update_layout(xaxis_title="Orders (6 months)", yaxis_title="Revenue (6 months)")
        st.plotly_chart(fig_scatter, use_container_width=True)

        st.markdown("#### Repeat Frequency Distribution")
        fig_hist = px.histogram(c, x="Orders_6m", nbins=20)
        fig_hist.update_layout(xaxis_title="Orders per Customer (6m)", yaxis_title="Customers")
        st.plotly_chart(fig_hist, use_container_width=True)


# ----------------------------------------------------
# 5. FORECAST TAB
# ----------------------------------------------------
with tabs[4]:
    st.subheader("Channel Forecast (Moving-Average)")

    ch_all = sorted(fcast["Channel"].unique().tolist())
    ch_sel = st.selectbox("Channel for forecast", ch_all, key="forecast_channel")

    fc_ch = fcast[fcast["Channel"] == ch_sel].copy()
    if fc_ch.empty:
        st.info("No forecast data for this channel.")
    else:
        # Show last 26 weeks
        fc_ch = fc_ch.sort_values("Date").tail(26)

        fig_fc = go.Figure()
        fig_fc.add_trace(go.Scatter(
            x=fc_ch["Date"],
            y=fc_ch["Revenue"],
            mode="lines+markers",
            name="Actual Revenue",
        ))
        fig_fc.add_trace(go.Scatter(
            x=fc_ch["Date"],
            y=fc_ch["yhat"],
            mode="lines",
            name="Moving-Avg Forecast",
        ))
        fig_fc.add_trace(go.Scatter(
            x=pd.concat([fc_ch["Date"], fc_ch["Date"][::-1]]),
            y=pd.concat([fc_ch["yhat_upper"], fc_ch["yhat_lower"][::-1]]),
            fill="toself",
            mode="lines",
            line=dict(width=0),
            opacity=0.2,
            name="Forecast band",
        ))
        fig_fc.update_layout(
            xaxis_title="Week",
            yaxis_title="Revenue",
        )
        st.plotly_chart(fig_fc, use_container_width=True)

        st.markdown(
            "Forecast is based on a simple 4-week moving average per channel. "
            "For production use, this can be upgraded to ARIMA/Prophet."
        )

# ----------------------------------------------------
# Footer
# ----------------------------------------------------
st.caption("Prepared by Derrick Wong | Graduate NTUC LearningHub Associate Data Analyst Course (Cohort 36) 2025")
