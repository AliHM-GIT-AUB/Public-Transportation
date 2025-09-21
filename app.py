# app.py (polished titles + insights + stacked-bar fix)

import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
from pathlib import Path
import plotly

st.set_page_config(page_title="Lebanon Road & Public Transport Explorer", layout="wide")

# ---------- Header ----------
st.title("Lebanon Road & Public Transport — Interactive Explorer")
st.markdown(
    "Use the controls on the left to: "
    "1) switch **road type** (Main / Secondary / Agricultural), "
    "2) optionally keep **only towns with dedicated bus stops**, and "
    "3) filter the **main transport modes** shown in the donut."
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
            return pd.read_csv(p, encoding="utf-8-sig")
    st.error(f"Data file not found. Place **{CSV_NAME}** next to app.py or in the repo root.")
    st.stop()

df = load_data()

# ---------- Sidebar ----------
st.sidebar.header("Filters")
road_type = st.sidebar.radio("Road type", ["main", "secondary", "agricultural"], index=0)
require_bus_stop = st.sidebar.checkbox("Only towns WITH dedicated bus stops", value=False)
mode_filter = st.sidebar.multiselect("Show transport modes", ["taxis", "vans", "buses"],
                                     default=["taxis", "vans", "buses"])
show_debug = st.sidebar.checkbox("Show debug info", value=False)

# ---------- Prepare (cached by UI state) ----------
@st.cache_data(show_spinner=False)
def prepare(df: pd.DataFrame, road_type: str, require_bus_stop: bool):
    work = df.copy()
    if require_bus_stop and BUS_STOP_COL in work.columns:
        work = work[work[BUS_STOP_COL] == 1]

    stem = STEM_MAP[road_type]
    g, a, b = stem + "good", stem + "acceptable", stem + "bad"
    for c in (g, a, b):
        if c not in work.columns:
            work[c] = 0

    bad = work[b].to_numpy() == 1
    acc = work[a].to_numpy() == 1
    good = work[g].to_numpy() == 1

    cond = np.full(len(work), "Unknown", dtype=object)
    cond = np.where(bad, "Bad", cond)
    cond = np.where(~bad & acc, "Acceptable", cond)
    cond = np.where(~bad & ~acc & good, "Good", cond)

    work = work.assign(Condition=cond)
    return work, g, a, b

work, gcol, acol, bcol = prepare(df, road_type, require_bus_stop)

# ---------- Debug ----------
if show_debug:
    st.info(f"Plotly {plotly.__version__} | Pandas {pd.__version__} | Rows (filtered): {len(work)}")

# ---------- KPI strip ----------
c1, c2, c3 = st.columns(3)
with c1: st.metric("Towns in view", f"{len(work):,}")
with c2:
    pct_bus = (100 * work[BUS_STOP_COL].mean()) if BUS_STOP_COL in work.columns and len(work) else 0
    st.metric("% with dedicated bus stops", f"{pct_bus:.1f}%")
with c3: st.metric("Road type selected", road_type.capitalize())

if len(work) == 0:
    st.warning("No towns match the selected filters.")
    st.stop()

# ---------- Chart 1: Condition composition ----------
st.subheader(f"Condition mix — {road_type.capitalize()} roads (% of towns)")
cond_counts = work["Condition"].value_counts().reindex(["Good", "Acceptable", "Bad"]).fillna(0)
cond_df = cond_counts.rename("Count").reset_index().rename(columns={"index": "Condition"})
cond_df["% of towns"] = (cond_df["Count"] / len(work) * 100).round(1)

fig1 = px.bar(cond_df, x="Condition", y="% of towns", text="% of towns")
fig1.update_traces(texttemplate="%{text:.1f}%", textposition="outside", cliponaxis=False)
fig1.update_yaxes(title="% of towns", range=[0, max(100, float(cond_df["% of towns"].max() or 0) + 10)])
st.plotly_chart(fig1, use_container_width=True)

# quick insight
top_row = cond_df.sort_values("% of towns", ascending=False).iloc[0]
st.caption(f"**Insight:** For {road_type} roads, **{top_row['Condition']}** is most common "
           f"({top_row['% of towns']:.1f}% of towns in view).")

# ---------- Chart 2: Mode share ----------
st.subheader("Main transport mode — share of towns shown")
mode_long = [(m.capitalize(), int(work[MODE_COLS[m]].sum()))
             for m in mode_filter if MODE_COLS[m] in work.columns]
mode_df = pd.DataFrame(mode_long, columns=["Mode", "Towns reporting mode"])

if mode_df.empty or mode_df["Towns reporting mode"].sum() == 0:
    st.info("No transport modes selected (or none present after filters).")
else:
    total = int(mode_df["Towns reporting mode"].sum())
    mode_df["Share (%)"] = (mode_df["Towns reporting mode"] / total * 100).round(1)
    fig2 = px.pie(mode_df, names="Mode", values="Towns reporting mode", hole=0.45)
    fig2.update_traces(textinfo="percent+label")
    st.plotly_chart(fig2, use_container_width=True)

    top_mode = mode_df.sort_values("Towns reporting mode", ascending=False).iloc[0]
    st.caption(f"**Insight:** **{top_mode['Mode']}** is the most reported main mode "
               f"({top_mode['Share (%)']:.1f}% among selected towns).")

# ---------- Chart 3: With vs without bus stops (Unknown excluded) ----------
st.subheader("Do bus stops coincide with better roads?")
if BUS_STOP_COL in df.columns:
    tmp = df.copy()
    for c in (gcol, acol, bcol):
        if c not in tmp.columns:
            tmp[c] = 0

    # label condition
    bad = tmp[bcol].to_numpy() == 1
    acc = tmp[acol].to_numpy() == 1
    good = tmp[gcol].to_numpy() == 1
    cond = np.full(len(tmp), "Unknown", dtype=object)
    cond = np.where(bad, "Bad", cond)
    cond = np.where(~bad & acc, "Acceptable", cond)
    cond = np.where(~bad & ~acc & good, "Good", cond)
    tmp["Condition"] = cond

    # exclude Unknown so stacks add to 100
    tmp = tmp[tmp["Condition"].isin(["Good", "Acceptable", "Bad"])]

    grp = (
        tmp.groupby(tmp[BUS_STOP_COL].map({1: "With bus stops", 0: "No bus stops"}))["Condition"]
        .value_counts(normalize=True).rename("%").mul(100).reset_index()
    )
    pivot = (
        grp.pivot(index=BUS_STOP_COL, columns="Condition", values="%")
        .rename_axis("Bus stops").fillna(0)
        .reindex(["With bus stops", "No bus stops"])
        .reindex(columns=["Bad", "Acceptable", "Good"], fill_value=0)
    )

    fig3 = px.bar(pivot, barmode="stack")
    fig3.update_yaxes(title="% of towns", range=[0, 100])
    fig3.update_traces(texttemplate="%{y:.1f}%", textposition="inside", cliponaxis=False)
    fig3.update_layout(legend_title_text="Condition")
    st.plotly_chart(fig3, use_container_width=True)

    # short diagnostic sentence
    with_bus = pivot.loc["With bus stops", ["Good", "Acceptable"]].sum()
    no_bus = pivot.loc["No bus stops", ["Good", "Acceptable"]].sum()
    delta = with_bus - no_bus
    st.caption(
        f"**Insight:** Towns **with bus stops** have **{with_bus:.1f}% Good+Acceptable** "
        f"vs **{no_bus:.1f}%** without ({delta:+.1f} pp difference)."
    )
else:
    st.info("Bus-stop column not found in the dataset.")

st.write("---")
st.caption(
    "Notes — Each town is counted once. “Condition” uses a worst-case label per town "
    "(Bad > Acceptable > Good). Percentages are recomputed live by your selections."
)

