# app.py — Streamlit Power BI-style dashboard (clean version, no debug)
# Run:  streamlit run app.py

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from pathlib import Path

# ------------------ APP CONFIG ------------------
st.set_page_config(page_title="Omnichannel Growth Engine", page_icon="📊", layout="wide")
st.title("Omnichannel Growth Engine (Python Dashboard)")
st.caption("Prepared by Derrick Wong | NTUC LearningHub Associate Data Analyst (Cohort 36), 2025")

# ------------------ PATHS ------------------
BASE_DIR = Path(__file__).resolve().parents[1]
CLEAN_DIR = BASE_DIR / "clean"
FACT_PATH = CLEAN_DIR / "cleaned_fmcg_omnichannel_sales.csv"
CUST_SUMMARY_PATH = CLEAN_DIR / "fmcg_customer_repeat_summary_clean.csv"  # optional

# ------------------ HELPERS ------------------
def _normalize(col):
    """lowercase + remove non-alnum for fuzzy matching."""
    return "".join(ch for ch in col.strip().lower() if ch.isalnum())

CANDIDATES = {
    "Date": ["date","orderdate","transactiondate","purchasedate","weekstart","week","weekof","weekbegin","weekstartdate"],
    "Channel": ["channel","saleschannel","platform"],
    "Territory": ["territory","region","area","zone"],
    "Product_Category": ["productcategory","category","skucategory","prodcategory"],
    "Promotion_Flag": ["promotionflag","promo","ispromo","onpromotion","promoflag","promotion"],
    "Revenue": ["revenue","sales","amount","gmv","netsales","totalsales","netrevenue"],
    "Quantity": ["quantity","units","qty","volume","qtty"],
    "Order_ID": ["orderid","orderno","order_id","transactionid","txn"],
    "Customer_ID": ["customerid","custid","buyerid","clientid"],
    "Repeat_Frequency": ["repeatfrequency","repeats","orders","ordercount","repeatcount"]
}

def _auto_rename(df):
    """Build rename map from fuzzy matches."""
    rename = {}
    norm_map = {c: _normalize(c) for c in df.columns}
    for std, options in CANDIDATES.items():
        hit = [c for c in df.columns if norm_map[c] in options]
        if not hit and std == "Date":
            for c in df.columns:
                n = norm_map[c]
                if ("date" in n) or ("week" in n):
                    hit = [c]
                    break
        if not hit and std == "Revenue":
            for c in df.columns:
                n = norm_map[c]
                if any(k in n for k in ["revenue","sales","amount","gmv"]):
                    hit = [c]
                    break
        if hit:
            rename[hit[0]] = std
    return rename

def _to_numeric(df, cols):
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

# ------------------ LOAD DATA ------------------
@st.cache_data
def load_data():
    if not FACT_PATH.exists():
        st.error(f"Missing file: {FACT_PATH}")
        st.stop()
    fact = pd.read_csv(FACT_PATH)
    rename_map = _auto_rename(fact)
    fact = fact.rename(columns=rename_map)

    if "Date" not in fact.columns:
        st.error("No date-like column found. Add a column like Date/OrderDate/WeekStart.")
        st.stop()

    fact["Date"] = pd.to_datetime(fact["Date"], errors="coerce")
    if "Promotion_Flag" in fact.columns:
        fact["Promotion_Flag"] = pd.to_numeric(fact["Promotion_Flag"], errors="coerce").fillna(0).astype(int)
    _to_numeric(fact, ["Revenue","Quantity"])
    fact = fact.dropna(subset=["Date"])

    cust_summary = None
    if CUST_SUMMARY_PATH.exists():
        cust_summary = pd.read_csv(CUST_SUMMARY_PATH)
        if "Last_Order_Date" in cust_summary.columns:
            cust_summary["Last_Order_Date"] = pd.to_datetime(cust_summary["Last_Order_Date"], errors="coerce")

    return fact, cust_summary, rename_map

fact, cust_summary, auto_map = load_data()

# ------------------ COLUMN OVERRIDES ------------------
with st.sidebar.expander("Column overrides", expanded=True):
    cols = list(fact.columns)
    def pick(label, default_name):
        idx = cols.index(default_name) if default_name in cols else 0
        return st.selectbox(label, cols, index=idx)

    date_col  = pick("Date column", "Date")
    chan_col  = pick("Channel column", "Channel")
    terr_col  = pick("Territory column", "Territory")
    cat_col   = pick("Product Category column", "Product_Category")
    promo_col = pick("Promotion Flag column", "Promotion_Flag")
    rev_col   = pick("Revenue column", "Revenue")
    qty_col   = pick("Quantity column", "Quantity")
    cust_col  = pick("Customer ID column", "Customer_ID")
    order_col = pick("Order ID column", "Order_ID")
    synth = st.checkbox("If Customer_ID missing/blank, create synthetic Customer_ID", value=True)

alias = {}
if date_col  != "Date":             alias[date_col]  = "Date"
if chan_col  != "Channel":          alias[chan_col]  = "Channel"
if terr_col  != "Territory":        alias[terr_col]  = "Territory"
if cat_col   != "Product_Category": alias[cat_col]   = "Product_Category"
if promo_col != "Promotion_Flag":   alias[promo_col] = "Promotion_Flag"
if rev_col   != "Revenue":          alias[rev_col]   = "Revenue"
if qty_col   != "Quantity":         alias[qty_col]   = "Quantity"
if cust_col  != "Customer_ID":      alias[cust_col]  = "Customer_ID"
if order_col != "Order_ID":         alias[order_col] = "Order_ID"

df = fact.rename(columns=alias).copy()
df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
if "Promotion_Flag" in df.columns:
    df["Promotion_Flag"] = pd.to_numeric(df["Promotion_Flag"], errors="coerce").fillna(0).astype(int)
_to_numeric(df, ["Revenue","Quantity"])

if ("Customer_ID" not in df.columns) or df["Customer_ID"].isna().all() or (df["Customer_ID"].astype(str).str.strip()=="").all():
    if synth:
        if "Order_ID" in df.columns:
            df["Customer_ID"] = df["Order_ID"].astype(str)
        else:
            df["Customer_ID"] = df.index.astype(str)

# ------------------ FILTERS ------------------
st.sidebar.header("Filters")
date_min, date_max = pd.to_datetime(df["Date"]).min(), pd.to_datetime(df["Date"]).max()
date_range = st.sidebar.date_input("Date range", (date_min, date_max))

def ms(label, name):
    if name in df.columns:
        vals = sorted([x for x in df[name].dropna().unique()])
        return st.sidebar.multiselect(label, vals, default=vals)
    return []

channels   = ms("Channel", "Channel")
territories= ms("Territory", "Territory")
categories = ms("Product Category", "Product_Category")

promo_choice = "All"
if "Promotion_Flag" in df.columns:
    promo_choice = st.sidebar.radio("Promotion", ["All","Promo only","Non-promo"])

f = df[df["Date"].between(pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1]))].copy()
if channels:    f = f[f["Channel"].isin(channels)]
if territories: f = f[f["Territory"].isin(territories)]
if categories:  f = f[f["Product_Category"].isin(categories)]
if "Promotion_Flag" in f.columns:
    if promo_choice == "Promo only":  f = f[f["Promotion_Flag"] == 1]
    elif promo_choice == "Non-promo": f = f[f["Promotion_Flag"] == 0]

st.caption(f"Rows after filters: {len(f):,}")

# ------------------ CUSTOMER SUMMARY ------------------
def build_customer_summary(df_in):
    if "Customer_ID" not in df_in.columns:
        return None
    if "Order_ID" in df_in.columns:
        per_order = df_in.groupby(["Customer_ID","Order_ID"], as_index=False)["Revenue"].sum()
        agg = per_order.groupby("Customer_ID").agg(
            Orders=("Order_ID","nunique"),
            Total_Revenue=("Revenue","sum")
        ).reset_index()
    else:
        agg = df_in.groupby("Customer_ID").agg(
            Orders=("Customer_ID","count"),
            Total_Revenue=("Revenue","sum") if "Revenue" in df_in.columns else ("Customer_ID","count")
        ).reset_index()
    agg["AOV"] = agg["Total_Revenue"] / agg["Orders"].replace(0, np.nan)
    agg["Repeat_Frequency"] = (agg["Orders"] - 1).clip(lower=0)
    return agg

cust_f = build_customer_summary(f)

# ------------------ KPIs ------------------
total_rev = f["Revenue"].sum() if "Revenue" in f.columns else np.nan
orders = f["Order_ID"].nunique() if "Order_ID" in f.columns else len(f)
aov = (total_rev / orders) if orders else np.nan

repeat_pct = np.nan
if cust_f is not None and len(cust_f):
    total_cust = cust_f["Customer_ID"].nunique()
    repeat_cust = (cust_f["Orders"] > 1).sum()
    repeat_pct = (repeat_cust / total_cust) if total_cust else np.nan

promo_uplift = np.nan
if {"Promotion_Flag","Revenue"}.issubset(f.columns):
    promo_rev = f.loc[f["Promotion_Flag"]==1, "Revenue"].sum()
    non_rev   = f.loc[f["Promotion_Flag"]==0, "Revenue"].sum()
    promo_uplift = ((promo_rev / non_rev) - 1) if non_rev else np.nan

c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Revenue", f"${total_rev:,.0f}" if pd.notna(total_rev) else "—")
c2.metric("Average Order Value", f"${aov:,.2f}" if pd.notna(aov) else "—")
c3.metric("Repeat Customers %", f"{repeat_pct*100:.1f}%" if pd.notna(repeat_pct) else "—")
c4.metric("Promo Uplift", f"{promo_uplift*100:.1f}%" if pd.notna(promo_uplift) else "—")

st.divider()

# ------------------ TABS ------------------
tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Channel & Territory", "Product & Promo", "Customers"])

with tab1:
    st.subheader("Weekly Revenue by Channel")
    if {"Date","Channel","Revenue"}.issubset(f.columns):
        weekly = f.groupby([pd.Grouper(key="Date", freq="W-MON"), "Channel"], as_index=False)["Revenue"].sum()
        if len(weekly):
            st.plotly_chart(px.line(weekly, x="Date", y="Revenue", color="Channel", markers=True), use_container_width=True)
        else:
            st.info("No data for selected filters.")

with tab2:
    colA, colB = st.columns(2)
    with colA:
        st.subheader("Revenue by Channel")
        if {"Channel","Revenue"}.issubset(f.columns):
            by_ch = f.groupby("Channel", as_index=False)["Revenue"].sum().sort_values("Revenue", ascending=False)
            st.plotly_chart(px.bar(by_ch, x="Channel", y="Revenue", text_auto=".2s"), use_container_width=True)
    with colB:
        st.subheader("Revenue by Territory")
        if {"Territory","Revenue"}.issubset(f.columns):
            by_ter = f.groupby("Territory", as_index=False)["Revenue"].sum().sort_values("Revenue", ascending=False)
            st.plotly_chart(px.bar(by_ter, x="Territory", y="Revenue", color="Territory", text_auto=".2s"), use_container_width=True)

with tab3:
    colC, colD = st.columns(2)
    with colC:
        st.subheader("Revenue by Product Category")
        if {"Product_Category","Revenue"}.issubset(f.columns):
            by_cat = f.groupby("Product_Category", as_index=False)["Revenue"].sum().sort_values("Revenue", ascending=False)
            st.plotly_chart(px.bar(by_cat, x="Product_Category", y="Revenue", color="Product_Category", text_auto=".2s"), use_container_width=True)
    with colD:
        st.subheader("Avg Order Revenue: Promo vs Non-Promo")
        if {"Promotion_Flag","Revenue"}.issubset(f.columns):
            if "Order_ID" in f.columns:
                aro = (f.groupby(["Promotion_Flag","Order_ID"], as_index=False)["Revenue"].sum()
                         .groupby("Promotion_Flag")["Revenue"].mean()
                         .rename("Avg_Revenue_per_Order").reset_index())
            else:
                aro = (f.groupby("Promotion_Flag", as_index=False)["Revenue"].mean()
                         .rename(columns={"Revenue":"Avg_Revenue_per_Order"}))
            aro["Promo"] = aro["Promotion_Flag"].map({0:"Non-Promo",1:"Promo"})
            st.plotly_chart(px.bar(aro, x="Promo", y="Avg_Revenue_per_Order", text_auto=".2s"), use_container_width=True)

with tab4:
    st.subheader("Customers — Revenue vs Orders (filtered)")
    if cust_f is not None and len(cust_f):
        fig = px.scatter(
            cust_f, x="Orders", y="Total_Revenue",
            size="AOV" if "AOV" in cust_f.columns else None,
            hover_data=["Customer_ID"],
            title="Customer Revenue vs Orders (size=AOV)"
        )
        st.plotly_chart(fig, use_container_width=True)
        topN = st.slider("Show top N customers by Total_Revenue", 5, 50, 10)
        st.dataframe(
            cust_f.sort_values("Total_Revenue", ascending=False)
                  .head(topN)
                  .reset_index(drop=True)
        )
    else:
        st.info("No customers available — set Customer ID in 'Column overrides' or enable synthetic option.")
