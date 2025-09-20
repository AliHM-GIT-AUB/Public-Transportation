# app.py 

import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
from pathlib import Path

st.set_page_config(page_title="Lebanon Transport — Interactive", layout="wide")
st.title("Public Transportation in Lebanon — Interactive Explorer")
st.markdown(
    "Interactively explore town-level data about road conditions, dedicated bus stops, and main transport modes."
)

CSV_NAME = "Public Transportation.csv"
BUS_STOP_COL = "Existence of dedicated bus stops - exists"
STEM_MAP = {
    "main": "State of the main roads - ",
    "secondary": "State of the secondary roads - ",
    "agricultural": "State of agricultural roads - ",
}
MODE_COLS = {
    "taxis": "The main means of public transport - taxis",
    "vans": "The main means of public transport - vans",
    "buses": "The main means of public transport - buses",
}

# ---------- Data loading ----------
@st.cache_data(show_spinner=False)
def load_data() -> pd.DataFrame:
    here = Path(__file__).parent
    for p in (here / CSV_NAME, here.parent / CSV_NAME):
        if p.exists():
            return pd.read_csv(p)
    st.error(f"Data file not found. Place **{CSV_NAME}** next to app.py or in the repo root.")
    st.stop()

df = load_data()

# ---------- Sidebar ----------
st.sidebar.header("Filters")
road_type = st.sidebar.radio("Choose road type", ["main", "secondary", "agricultural"], index=0)
require_bus_stop = st.sidebar.checkbox("Only towns WITH dedicated bus stops", value=False)
mode_filter = st.sidebar.multiselect(
    "Show transport modes", ["taxis", "vans", "buses"], default=["taxis", "vans", "buses"]
)

# ---------- Fast preparation (cached by UI state) ----------
@st.cache_data(show_spinner=False)
def prepare(df: pd.DataFrame, road_type: str, require_bus_stop: bool):
    work = df.copy()

    # Optional filter
    if require_bus_stop and BUS_STOP_COL in work.columns:
        work = work[work[BUS_STOP_COL] == 1]

    # Build vectorized "Condition"
    stem = STEM_MAP[road_type]
    good_col = stem + "good"
    acc_col  = stem + "acceptable"
    bad_col  = stem + "bad"
    for c in (good_col, acc_col, bad_col):
        if c not in work.columns:
            work[c] = 0

    # vectorized mapping: Bad > Acceptable > Good > Unknown
    bad = work[bad_col].to_numpy() == 1
    acc = work[acc_col].to_numpy() == 1
    good = work[good_col].to_numpy() == 1
    cond = np.full(len(work), "Unknown", dtype=object)
    cond = np.where(bad, "Bad", cond)
    cond = np.where(~bad & acc, "Acceptable", cond)
    cond = np.where(~bad & ~acc & good, "Good", cond)

    work = work.assign(Condition=cond)
    return work, good_col, acc_col, bad_col

work, good_col, acc_col, bad_col = prepare(df, road_type, require_bus_stop)

# ---------- KPI strip ----------
c1, c2, c3 = st.columns(3)
with c1:
    st.metric("Towns (after filters)", f"{len(work):,}")
with c2:
    pct_bus = (100 * work[BUS_STOP_COL].mean()) if BUS_STOP_COL in work.columns and len(work) else 0
    st.metric("% with bus stops", f"{pct_bus:.1f}%")
with c3:
    st.metric("Road type", road_type.capitalize())

if len(work) == 0:
    st.warning("No towns match the selected filters.")
    st.stop()

# ---------- Chart 1: Condition composition ----------
st.subheader(f"Road Condition Composition — {road_type.capitalize()} roads")
cond_counts = work["Condition"].value_counts().reindex(["Good", "Acceptable", "Bad"]).fillna(0)
cond_df = cond_counts.rename("Count").reset_index().rename(columns={"index": "Condition"})
cond_df["% of Towns"] = (cond_df["Count"] / len(work) * 100).round(1)

fig1 = px.bar(cond_df, x="Condition", y="% of Towns", text="% of Towns")
fig1.update_traces(texttemplate="%{text:.1f}%", textposition="outside", cliponaxis=False)
fig1.update_yaxes(range=[0, max(100, float(cond_df["% of Towns"].max() or 0) + 10)])
st.plotly_chart(fig1, use_container_width=True)
st.caption("Interaction #1 — Road type & bus-stop filter recalculate this chart.")

# ---------- Chart 2: Mode share ----------
st.subheader("Share of Towns by Main Transport Mode")
mode_long = [(m.capitalize(), int(work[MODE_COLS[m]].sum()))
             for m in mode_filter if MODE_COLS[m] in work.columns]
mode_df = pd.DataFrame(mode_long, columns=["Mode", "Towns reporting mode"])

if mode_df.empty or mode_df["Towns reporting mode"].sum() == 0:
    st.info("No transport modes selected (or none present after filters).")
else:
    total = int(mode_df["Towns reporting mode"].sum())
    mode_df["Share (%)"] = (mode_df["Towns reporting mode"] / total * 100).round(1)
    fig2 = px.pie(mode_df, names="Mode", values="Towns reporting mode", hole=0.45)
    st.plotly_chart(fig2, use_container_width=True)
    st.caption("Interaction #2 — Mode multiselect updates the donut.")

# ---------- Chart 3: With vs without bus stops ----------
st.subheader("Quick Diagnostic: Bus Stops vs Road Condition")
if BUS_STOP_COL in df.columns:
    stem = STEM_MAP[road_type]
    g, a, b = stem + "good", stem + "acceptable", stem + "bad"
    tmp = df.copy()
    for c in (g, a, b):
        if c not in tmp.columns:
            tmp[c] = 0

    bad = tmp[b].to_numpy() == 1
    acc = tmp[a].to_numpy() == 1
    good = tmp[g].to_numpy() == 1
    cond = np.full(len(tmp), "Unknown", dtype=object)
    cond = np.where(bad, "Bad", cond)
    cond = np.where(~bad & acc, "Acceptable", cond)
    cond = np.where(~bad & ~acc & good, "Good", cond)
    tmp["Condition"] = cond

    grp = (
        tmp.groupby(tmp[BUS_STOP_COL].map({1: "With Bus Stops", 0: "No Bus Stops"}))["Condition"]
        .value_counts(normalize=True)
        .rename("%").mul(100).reset_index()
    )
    pivot = (
        grp.pivot(index=BUS_STOP_COL, columns="Condition", values="%")
        .rename_axis("Bus stops").fillna(0)
        .reindex(["With Bus Stops", "No Bus Stops"])
        .reindex(columns=["Bad", "Acceptable", "Good"], fill_value=0)
    )
    st.plotly_chart(px.bar(pivot, barmode="stack", text_auto=".1f"), use_container_width=True)
    st.caption("Compares condition shares for towns with vs without bus stops.")
else:
    st.info("Bus-stop column not found in the dataset.")

st.write("---")
st.caption("Percentages are town-based (each town counted once). Columns are binary indicators; aggregations are computed on the fly.")
