import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path

st.set_page_config(page_title="Lebanon Transport — Interactive", layout="wide")

@st.cache_data
def load_data():
    # Load CSV from the SAME folder where this app.py lives
    data_path = Path(__file__).parent / "Public Transportation.csv"
    if not data_path.exists():
        st.error(f"Data file not found at: {data_path}")
        st.stop()
    return pd.read_csv(data_path)

df = load_data()

st.title("Public Transportation in Lebanon — Interactive Explorer")
st.markdown(
    "This app builds on the earlier Plotly assignment to let you interactively explore "
    "town-level data about road conditions, dedicated bus stops, and main transport modes."
)

# ---------------------- SIDEBAR CONTROLS ----------------------
st.sidebar.header("Filters")
road_type = st.sidebar.radio(
    "Choose road type",
    ["main", "secondary", "agricultural"],
    index=0,
    help="Switch between main, secondary, and agricultural roads"
)

require_bus_stop = st.sidebar.checkbox(
    "Only towns WITH dedicated bus stops",
    value=False
)

mode_filter = st.sidebar.multiselect(
    "Show transport modes",
    options=["taxis", "vans", "buses"],
    default=["taxis", "vans", "buses"]
)

# ---------------------- PREPARE COLUMNS ----------------------
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

# Filter by bus stop if requested
work = df.copy()
if require_bus_stop and bus_stop_col in work.columns:
    work = work[work[bus_stop_col] == 1]

# Derive one categorical column for the selected road type's condition
good_col = stem_map[road_type] + "good"
acc_col  = stem_map[road_type] + "acceptable"
bad_col  = stem_map[road_type] + "bad"

def pick_condition(row):
    if row.get(bad_col, 0) == 1:
        return "Bad"
    if row.get(acc_col, 0) == 1:
        return "Acceptable"
    if row.get(good_col, 0) == 1:
        return "Good"
    return "Unknown"

work["Condition"] = work.apply(pick_condition, axis=1)

# ---------------------- CHART 1: CONDITION COMPOSITION ----------------------
st.subheader(f"Road Condition Composition — {road_type.capitalize()} roads")

cond_counts = (
    work["Condition"]
    .value_counts(dropna=False)
    .reindex(["Good", "Acceptable", "Bad", "Unknown"])
    .fillna(0)
)
cond_df = cond_counts.reset_index()
cond_df.columns = ["Condition", "Count"]
cond_df["% of Towns"] = (cond_df["Count"] / max(len(work), 1) * 100).round(1)

fig1 = px.bar(
    cond_df[cond_df["Condition"].isin(["Good", "Acceptable", "Bad"])],
    x="Condition",
    y="% of Towns",
    text="% of Towns",
)
fig1.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
fig1.update_yaxes(range=[0, max(100, float(cond_df["% of Towns"].max() or 0) + 10)])
st.plotly_chart(fig1, use_container_width=True)

st.caption(
    "Interaction #1 — The **road type** and **bus-stop filter** in the sidebar recompute this chart from the raw town records."
)

# ---------------------- CHART 2: TRANSPORT MODE SHARE ----------------------
st.subheader("Share of Towns by Main Transport Mode")

mode_long = []
for m in mode_filter:
    col = mode_cols[m]
    if col in work.columns:
        mode_long.append((m.capitalize(), int(work[col].sum())))

mode_df = pd.DataFrame(mode_long, columns=["Mode", "Towns reporting mode"])
total_mode = int(mode_df["Towns reporting mode"].sum()) if not mode_df.empty else 0
mode_df["Share (%)"] = (mode_df["Towns reporting mode"] / total_mode * 100).round(1) if total_mode > 0 else 0.0

fig2 = px.pie(
    mode_df,
    names="Mode",
    values="Towns reporting mode",
    hole=0.45
)
st.plotly_chart(fig2, use_container_width=True)

st.caption(
    "Interaction #2 — The **Mode multiselect** controls which transport modes are shown and updates the pie."
)

# ---------------------- BONUS: BUS STOPS vs BAD ROADS ----------------------
st.subheader("Quick Diagnostic: Bus Stops vs Bad Roads (selected road type)")

if bus_stop_col in df.columns:
    tmp = df.copy()
    tmp["Condition"] = tmp.apply(pick_condition, axis=1)
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
    st.caption("This stacked bar compares condition shares for towns **with vs without** bus stops.")
else:
    st.info("Bus-stop column not found in the dataset.")

st.write("---")
st.markdown(
    "**Notes**: Percentages are town-based (each town counted once). The dataset columns are binary indicators "
    "for conditions and transport modes. This app aggregates those indicators on the fly with your chosen filters."
)
