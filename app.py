# app.py
import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path

# ---------- PAGE SETUP ----------
st.set_page_config(page_title="Lebanon Transport — Interactive", layout="wide")
st.title("Public Transportation in Lebanon — Interactive Explorer")
st.markdown(
    "This app builds on the earlier Plotly assignment to let you interactively explore "
    "town-level data about road conditions, dedicated bus stops, and main transport modes."
)

# ---------- DATA LOADING ----------
CSV_NAME = "Public Transportation.csv"

@st.cache_data
def load_data() -> pd.DataFrame:
    """Load CSV from the app folder or its parent (repo root)."""
    here = Path(__file__).parent
    candidates = [here / CSV_NAME, here.parent / CSV_NAME]
    for p in candidates:
        if p.exists():
            df = pd.read_csv(p)
            return df
    st.error(
        f"Data file not found.\nLooked in:\n- {candidates[0]}\n- {candidates[1]}\n\n"
        f"Place **{CSV_NAME}** next to app.py or in the repo root."
    )
    st.stop()

df = load_data()

# ---------- CONSTANTS ----------
stem_map = {
    "main": "State of the main roads - ",
    "secondary": "State of the secondary roads - ",
    "agricultural": "State of agricultural roads - ",
}
mode_cols = {
    "taxis": "The main means of public transport - taxis",
    "vans": "The main means of public transport - vans",
    "buses": "The main means of public transport - buses",
}
bus_stop_col = "Existence of dedicated bus stops - exists"

# ---------- SIDEBAR CONTROLS ----------
st.sidebar.header("Filters")
road_type = st.sidebar.radio(
    "Choose road type", ["main", "secondary", "agricultural"], index=0,
    help="Switch between main, secondary, and agricultural roads"
)
require_bus_stop = st.sidebar.checkbox("Only towns WITH dedicated bus stops", value=False)
mode_filter = st.sidebar.multiselect(
    "Show transport modes", options=["taxis", "vans", "buses"],
    default=["taxis", "vans", "buses"]
)

# ---------- HELPER FUNCTIONS ----------
def make_condition_columns_exist(d: pd.DataFrame, stem: str) -> None:
    """Ensure the three expected columns exist; if not, create zeros."""
    for suffix in ["good", "acceptable", "bad"]:
        col = stem + suffix
        if col not in d.columns:
            d[col] = 0

def pick_condition(row, bad_col, acc_col, good_col) -> str:
    if row.get(bad_col, 0) == 1:
        return "Bad"
    if row.get(acc_col, 0) == 1:
        return "Acceptable"
    if row.get(good_col, 0) == 1:
        return "Good"
    return "Unknown"

# ---------- FILTER + DERIVED FIELDS ----------
work = df.copy()

if require_bus_stop and bus_stop_col in work.columns:
    work = work[work[bus_stop_col] == 1]

# Build condition column from one-hot triples
stem = stem_map[road_type]
make_condition_columns_exist(work, stem)
good_col = stem + "good"
acc_col  = stem + "acceptable"
bad_col  = stem + "bad"
work["Condition"] = work.apply(pick_condition, axis=1, args=(bad_col, acc_col, good_col))

# ---------- KPI STRIP ----------
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Towns (after filters)", f"{len(work):,}")
with col2:
    pct_bus = (
        100 * work[bus_stop_col].mean()
        if bus_stop_col in work.columns and len(work) > 0 else 0
    )
    st.metric("% with bus stops", f"{pct_bus:.1f}%")
with col3:
    st.metric("Road type", road_type.capitalize())

if len(work) == 0:
    st.warning("No towns match the selected filters.")
    st.stop()

# ---------- CHART 1: CONDITION COMPOSITION ----------
st.subheader(f"Road Condition Composition — {road_type.capitalize()} roads")

cond_counts = (
    work["Condition"].value_counts(dropna=False)
    .reindex(["Good", "Acceptable", "Bad", "Unknown"])
    .fillna(0)
)
cond_df = cond_counts.reset_index()
cond_df.columns = ["Condition", "Count"]
cond_df["% of Towns"] = (cond_df["Count"] / len(work) * 100).round(1)

fig1 = px.bar(
    cond_df[cond_df["Condition"].isin(["Good", "Acceptable", "Bad"])],
    x="Condition", y="% of Towns", text="% of Towns"
)
fig1.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
ymax = float(cond_df["% of Towns"].max() or 0) + 10
fig1.update_yaxes(range=[0, max(100, ymax)])
st.plotly_chart(fig1, use_container_width=True)
st.caption(
    "Interaction #1 — The **road type** and **bus-stop filter** recompute this chart from the raw town records."
)

# ---------- CHART 2: TRANSPORT MODE SHARE ----------
st.subheader("Share of Towns by Main Transport Mode")

mode_long = []
for m in mode_filter:
    col = mode_cols[m]
    if col in work.columns:
        mode_long.append((m.capitalize(), int(work[col].sum())))
mode_df = pd.DataFrame(mode_long, columns=["Mode", "Towns reporting mode"])

total_mode = int(mode_df["Towns reporting mode"].sum()) if not mode_df.empty else 0
mode_df["Share (%)"] = (
    (mode_df["Towns reporting mode"] / total_mode * 100).round(1) if total_mode > 0 else 0.0
)

fig2 = px.pie(mode_df, names="Mode", values="Towns reporting mode", hole=0.45)
st.plotly_chart(fig2, use_container_width=True)
st.caption("Interaction #2 — The **Mode multiselect** updates the donut chart.")

# ---------- CHART 3: WITH vs WITHOUT BUS STOPS ----------
st.subheader("Quick Diagnostic: Bus Stops vs Road Condition (selected road type)")

if bus_stop_col in df.columns:
    tmp = df.copy()
    make_condition_columns_exist(tmp, stem)
    tmp["Condition"] = tmp.apply(pick_condition, axis=1, args=(bad_col, acc_col, good_col))
    grp = (
        tmp.groupby(tmp[bus_stop_col].map({1: "With Bus Stops", 0: "No Bus Stops"}))["Condition"]
        .value_counts(normalize=True)
        .rename("%")
        .mul(100)
        .reset_index()
    )
    pivot = (
        grp.pivot(index=bus_stop_col, columns="Condition", values="%")
        .rename_axis("Bus stops")
        .fillna(0)
        .reindex(["With Bus Stops", "No Bus Stops"])
        .reindex(columns=["Bad", "Acceptable", "Good"], fill_value=0)
    )
    fig3 = px.bar(pivot, barmode="stack", text_auto=".1f")
    fig3.update_layout(legend_title_text="Condition")
    st.plotly_chart(fig3, use_container_width=True)
    st.caption("Compares condition shares for towns **with vs without** bus stops.")
else:
    st.info("Bus-stop column not found in the dataset.")

# ---------- FOOTER ----------
st.write("---")
st.markdown(
    "**Notes**: Percentages are town-based (each town counted once). Columns in the CSV are binary indicators. "
    "This app aggregates those indicators on the fly with your chosen filters."
)
