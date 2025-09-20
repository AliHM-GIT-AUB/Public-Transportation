# app.py — Governorate edition (fast + robust)
# --------------------------------------------------
import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
from pathlib import Path
import re, unicodedata

st.set_page_config(page_title="Lebanon Transport — Interactive", layout="wide")

st.title("Public Transportation in Lebanon — Interactive Explorer")
st.markdown(
    "This app extends the earlier Plotly assignment with **interactive** views and "
    "adds **governorate-level** analysis to mirror the original PPT."
)

# ---------- constants ----------
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
COL_TOWN = "Town"
REF_AREA = "refArea"

# ---------- helpers: area cleaning -> governorate ----------
def strip_accents(s: str) -> str:
    if not isinstance(s, str):
        return s
    return "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))

def last_path_component(uri: str) -> str:
    if not isinstance(uri, str) or "/" not in uri:
        return uri
    return uri.rsplit("/", 1)[-1]

def clean_token(s: str) -> str:
    if not isinstance(s, str):
        return s
    s = (
        s.replace("â", "-")
         .replace("â€“", "-")
         .replace("â€”", "-")
         .replace("ZahlÃ©", "Zahle")
         .replace("MiniyehâDanniyeh", "Miniyeh-Danniyeh")
         .replace("Miniyehâ\x80\x93Danniyeh", "Miniyeh-Danniyeh")
    )
    s = s.replace("_Governorate", "").replace("_", " ").replace(", Lebanon", "")
    s = strip_accents(s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

GOV_ALIAS = {
    "beirut": "Beirut",
    "akkar": "Akkar",
    "baalbek-hermel": "Baalbek-Hermel",
    "bekaa": "Bekaa",
    "beqaa": "Bekaa",
    "mount lebanon": "Mount Lebanon",
    "north": "North Lebanon",
    "north lebanon": "North Lebanon",
    "south": "South Lebanon",
    "south lebanon": "South Lebanon",
    "nabatieh": "Nabatiyeh",
    "nabatiyeh": "Nabatiyeh",
    "keserwan-jbeil": "Mount Lebanon",
}
DISTRICT_TO_GOV = {
    # Mount Lebanon
    "baabda": "Mount Lebanon", "aley": "Mount Lebanon", "chouf": "Mount Lebanon",
    "shouf": "Mount Lebanon", "metn": "Mount Lebanon", "matn": "Mount Lebanon",
    "keserwan": "Mount Lebanon", "jbeil": "Mount Lebanon", "byblos": "Mount Lebanon",
    # North Lebanon
    "tripoli": "North Lebanon", "miniyeh-danniyeh": "North Lebanon",
    "miniyeh": "North Lebanon", "danniyeh": "North Lebanon",
    "zgharta": "North Lebanon", "bsharri": "North Lebanon", "bcharre": "North Lebanon",
    "koura": "North Lebanon", "batroun": "North Lebanon",
    # Akkar
    "akkar": "Akkar",
    # Bekaa
    "zahle": "Bekaa", "zahleh": "Bekaa",
    "western beqaa": "Bekaa", "west beqaa": "Bekaa", "west bekaa": "Bekaa",
    "rashaya": "Bekaa",
    # Baalbek-Hermel
    "baalbek": "Baalbek-Hermel", "hermel": "Baalbek-Hermel",
    # Nabatiyeh
    "nabatieh": "Nabatiyeh", "bint jbeil": "Nabatiyeh", "marjeyoun": "Nabatiyeh", "hasbaya": "Nabatiyeh",
    # South Lebanon
    "sidon": "South Lebanon", "saida": "South Lebanon", "tyre": "South Lebanon", "jezzine": "South Lebanon",
}

def token_to_canonical(token: str) -> str | None:
    if not isinstance(token, str):
        return None
    t = token.lower().strip()
    t = re.sub(r"\s*(district|districts|caza|qada)\s*$", "", t)
    if t in GOV_ALIAS:
        return GOV_ALIAS[t]
    if t in DISTRICT_TO_GOV:
        return DISTRICT_TO_GOV[t]
    first = t.split(" ")[0]
    return DISTRICT_TO_GOV.get(first)

# ---------- load + clean ----------
@st.cache_data(show_spinner=False)
def load_raw() -> pd.DataFrame:
    here = Path(__file__).parent
    for p in (here / CSV_NAME, here.parent / CSV_NAME):
        if p.exists():
            return pd.read_csv(p, encoding="utf-8-sig")
    raise FileNotFoundError(f"Could not find {CSV_NAME} next to app.py or in the repo root.")

@st.cache_data(show_spinner=False)
def prepare_base(df: pd.DataFrame) -> pd.DataFrame:
    if REF_AREA not in df.columns:
        st.error(f"Column '{REF_AREA}' not found in the CSV.")
        st.stop()
    d = df.copy()
    # Governorate
    d["_token"] = d[REF_AREA].astype(str).apply(lambda x: clean_token(last_path_component(x)))
    d["Governorate"] = d["_token"].apply(token_to_canonical)
    d = d.dropna(subset=["Governorate"]).copy()

    # Make sure important columns exist as 0/1
    cond_cols = []
    for stem in STEM_MAP.values():
        for suf in ["good", "acceptable", "bad"]:
            cond_cols.append(stem + suf)
    bin_cols = [BUS_STOP_COL] + list(MODE_COLS.values()) + cond_cols
    for c in bin_cols:
        if c not in d.columns:
            d[c] = 0
        d[c] = pd.to_numeric(d[c], errors="coerce").fillna(0).astype(int)

    return d

raw = load_raw()
base = prepare_base(raw)

# ---------- sidebar ----------
st.sidebar.header("Filters")
road_type = st.sidebar.radio("Choose road type", ["main", "secondary", "agricultural"], index=0)
gov_options = sorted(base["Governorate"].unique().tolist())
selected_govs = st.sidebar.multiselect("Governorates", gov_options, default=gov_options)
require_bus_stop = st.sidebar.checkbox("Only towns WITH dedicated bus stops", value=False)
mode_filter = st.sidebar.multiselect("Show transport modes", ["taxis", "vans", "buses"],
                                     default=["taxis", "vans", "buses"])

# ---------- derive filtered dataframe ----------
@st.cache_data(show_spinner=False)
def make_filtered(base: pd.DataFrame, road_type: str, selected_govs: list[str], require_bus_stop: bool):
    work = base[base["Governorate"].isin(selected_govs)].copy()
    if require_bus_stop:
        work = work[work[BUS_STOP_COL] == 1]

    stem = STEM_MAP[road_type]
    g, a, b = stem + "good", stem + "acceptable", stem + "bad"
    # vectorized "worst-case" condition
    bad = work[b].to_numpy() == 1
    acc = work[a].to_numpy() == 1
    good = work[g].to_numpy() == 1
    cond = np.full(len(work), "Unknown", dtype=object)
    cond = np.where(bad, "Bad", cond)
    cond = np.where(~bad & acc, "Acceptable", cond)
    cond = np.where(~bad & ~acc & good, "Good", cond)
    work = work.assign(Condition=cond)

    return work, (g, a, b)

work, (gcol, acol, bcol) = make_filtered(base, road_type, selected_govs, require_bus_stop)

# ---------- KPI strip ----------
c1, c2, c3 = st.columns(3)
with c1: st.metric("Towns (after filters)", f"{len(work):,}")
with c2:
    pct_bus = (100 * work[BUS_STOP_COL].mean()) if BUS_STOP_COL in work.columns and len(work) else 0
    st.metric("% with bus stops", f"{pct_bus:.1f}%")
with c3: st.metric("Road type", road_type.capitalize())

if len(work) == 0:
    st.warning("No towns match the selected filters.")
    st.stop()

# ---------- tabs ----------
tab_overview, tab_gov = st.tabs(["Overview", "By Governorate"])

# =============== OVERVIEW (keeps your earlier interactions) ===============
with tab_overview:
    st.subheader(f"Road Condition Composition — {road_type.capitalize()} roads")
    cond_counts = work["Condition"].value_counts().reindex(["Good", "Acceptable", "Bad"]).fillna(0)
    cond_df = cond_counts.rename("Count").reset_index().rename(columns={"index": "Condition"})
    cond_df["% of Towns"] = (cond_df["Count"] / len(work) * 100).round(1)
    fig1 = px.bar(cond_df, x="Condition", y="% of Towns", text="% of Towns")
    fig1.update_traces(texttemplate="%{text:.1f}%", textposition="outside", cliponaxis=False)
    fig1.update_yaxes(range=[0, max(100, float(cond_df["% of Towns"].max() or 0) + 10)])
    st.plotly_chart(fig1, use_container_width=True)
    st.caption("Interaction #1 — Road type & bus-stop filter recalculate this chart.")

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

    st.subheader("Quick Diagnostic: Bus Stops vs Road Condition")
    tmp = base.copy()  # show all towns for this diagnostic
    # build condition for the chosen road type on the full dataset
    for c in (gcol, acol, bcol):
        if c not in tmp.columns:
            tmp[c] = 0
    bad = tmp[bcol].to_numpy() == 1
    acc = tmp[acol].to_numpy() == 1
    good = tmp[gcol].to_numpy() == 1
    cond = np.full(len(tmp), "Unknown", dtype=object)
    cond = np.where(bad, "Bad", cond)
    cond = np.where(~bad & acc, "Acceptable", cond)
    cond = np.where(~bad & ~acc & good, "Good", cond)
    tmp["Condition"] = cond

    grp = (tmp.groupby(tmp[BUS_STOP_COL].map({1: "With Bus Stops", 0: "No Bus Stops"}))["Condition"]
              .value_counts(normalize=True).rename("%").mul(100).reset_index())
    pivot = (grp.pivot(index=BUS_STOP_COL, columns="Condition", values="%")
                .rename_axis("Bus stops").fillna(0)
                .reindex(["With Bus Stops", "No Bus Stops"])
                .reindex(columns=["Bad", "Acceptable", "Good"], fill_value=0))
    fig3 = px.bar(pivot, barmode="stack")
    fig3.update_traces(texttemplate="%{y:.1f}%", textposition="inside", cliponaxis=False)
    fig3.update_layout(legend_title_text="Condition")
    st.plotly_chart(fig3, use_container_width=True)
    st.caption("Compares condition shares for towns with vs without bus stops.")

# =============== BY GOVERNORATE (mirrors your PPT) ===============
with tab_gov:
    st.markdown("### Governorate Views")

    # ---- 1) % of towns with bus stops (ranked bar) ----
    gov_bus = (work.groupby("Governorate")[BUS_STOP_COL].mean().mul(100).round(1)
                     .sort_values(ascending=False)).rename("% with bus stops").reset_index()
    fig_bars = px.bar(
        gov_bus, x="Governorate", y="% with bus stops",
        text=gov_bus["% with bus stops"].astype(str) + "%",
        title="% of Towns with Bus Stops (by Governorate)"
    )
    fig_bars.update_traces(textposition="outside", cliponaxis=False)
    st.plotly_chart(fig_bars, use_container_width=True)

    # ---- 2) 100% stacked bars: main road condition shares by governorate ----
    # build shares of Good/Acceptable/Bad by governorate using 'worst-case' logic already in work["Condition"]
    dist = (work.groupby(["Governorate", "Condition"]).size().reset_index(name="towns"))
    pivot = (dist.pivot(index="Governorate", columns="Condition", values="towns").fillna(0))
    # denominator excludes Unknown so it doesn't dilute
    if "Unknown" in pivot.columns:
        denom = pivot[["Good", "Acceptable", "Bad"]].sum(axis=1, min_count=1)
    else:
        denom = pivot.sum(axis=1, min_count=1)
    share = (pivot.div(denom, axis=0)[["Good", "Acceptable", "Bad"]] * 100).fillna(0)

    v2 = (share.reset_index()
                .melt(id_vars="Governorate", var_name="Condition", value_name="Percent"))
    order_bad = (v2[v2["Condition"] == "Bad"]
                    .sort_values("Percent", ascending=False)["Governorate"].tolist())

    fig_stack = px.bar(
        v2, x="Governorate", y="Percent", color="Condition",
        category_orders={"Governorate": order_bad, "Condition": ["Good", "Acceptable", "Bad"]},
        title=f"{road_type.capitalize()} Road Conditions — % of Towns by Governorate (worst-case)"
    )
    fig_stack.update_layout(barmode="stack")
    fig_stack.update_traces(texttemplate="%{y:.1f}%", textposition="inside",
                            insidetextanchor="middle", textfont_size=11, cliponaxis=False)
    st.plotly_chart(fig_stack, use_container_width=True)

    # ---- 3) Heatmap: % Good/Acceptable/Bad by governorate ----
    heat_df = share.copy()
    heat_df = heat_df[["Good", "Acceptable", "Bad"]]
    fig_heat = px.imshow(
        heat_df,
        labels=dict(x="Condition", y="Governorate", color="% of Towns"),
        x=["Good", "Acceptable", "Bad"],
        title="Main Road Condition Distribution (Heatmap)"
    )
    st.plotly_chart(fig_heat, use_container_width=True)

    # ---- 4) Scatter: coverage vs % bad (with simple trend line) ----
    agg = pd.DataFrame({
        "Governorate": share.index,
        "pct_main_bad": share["Bad"].values,
        "pct_bus_stops": gov_bus.set_index("Governorate")["% with bus stops"].reindex(share.index).values
    })
    fig_sc = px.scatter(
        agg, x="pct_bus_stops", y="pct_main_bad", text="Governorate",
        labels={"pct_bus_stops": "% with bus stops", "pct_main_bad": "% with bad main roads"},
        title="Coverage vs Condition"
    )
    # simple OLS using numpy (no statsmodels needed)
    if len(agg) >= 2 and agg["pct_bus_stops"].nunique() > 1:
        x = agg["pct_bus_stops"].to_numpy()
        y = agg["pct_main_bad"].to_numpy()
        m, b = np.polyfit(x, y, 1)
        xs = np.linspace(x.min(), x.max(), 100)
        ys = m * xs + b
        fig_sc.add_scatter(x=xs, y=ys, mode="lines", name="trend")
    st.plotly_chart(fig_sc, use_container_width=True)

# ---------- footer ----------
st.write("---")
st.caption(
    "Notes: Each town is counted once. The 'worst-case' condition encodes **Bad > Acceptable > Good** "
    "per road type. Governorate mapping follows your original cleaning logic."
)
