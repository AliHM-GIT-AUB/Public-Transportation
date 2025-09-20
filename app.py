# app.py — Streamlit cleaner + PPT visuals + downloads
import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
from pathlib import Path
import io, re, unicodedata

st.set_page_config(page_title="Lebanon Transport — Cleaner & Explorer", layout="wide")

st.title("Public Transportation in Lebanon — Cleaner & Explorer")
st.markdown(
    "Upload your raw CSV, optionally upload a **Town→Governorate** mapping if the dataset "
    "doesn’t include `Governorate`/`refArea`, then explore the **governorate visuals** and "
    "**download the cleaned Excel/CSV**."
)

# ----------------------- Helpers -----------------------
def find_col(cols, target: str) -> str | None:
    t = re.sub(r"\s+", "", target).lower()
    for c in cols:
        if re.sub(r"\s+", "", c).lower() == t:
            return c
    return None

def find_any(cols, keywords: list[str]) -> str | None:
    lc = {c.lower(): c for c in cols}
    for k in keywords:
        for c in lc:
            if k in c:
                return lc[c]
    return None

def strip_accents(s: str) -> str:
    if not isinstance(s, str): return s
    return "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))

def last_path_component(uri: str) -> str:
    if not isinstance(uri, str) or "/" not in uri: return uri
    return uri.rsplit("/", 1)[-1]

def clean_token(s: str) -> str:
    if not isinstance(s, str): return s
    s = (s.replace("â","-").replace("â€“","-").replace("â€”","-")
           .replace("ZahlÃ©","Zahle")
           .replace("MiniyehâDanniyeh","Miniyeh-Danniyeh")
           .replace("Miniyeh\u0080\u0093Danniyeh","Miniyeh-Danniyeh"))
    s = s.replace("_Governorate","").replace("_"," ").replace(", Lebanon","")
    s = strip_accents(s)
    return re.sub(r"\s+"," ",s).strip()

gov_alias = {
    "beirut":"Beirut","akkar":"Akkar","baalbek-hermel":"Baalbek-Hermel",
    "bekaa":"Bekaa","beqaa":"Bekaa","mount lebanon":"Mount Lebanon",
    "north":"North Lebanon","north lebanon":"North Lebanon",
    "south":"South Lebanon","south lebanon":"South Lebanon",
    "nabatieh":"Nabatiyeh","nabatiyeh":"Nabatiyeh","keserwan-jbeil":"Mount Lebanon",
}
district_to_gov = {
    "baabda":"Mount Lebanon","aley":"Mount Lebanon","chouf":"Mount Lebanon","shouf":"Mount Lebanon",
    "metn":"Mount Lebanon","matn":"Mount Lebanon","keserwan":"Mount Lebanon","jbeil":"Mount Lebanon","byblos":"Mount Lebanon",
    "tripoli":"North Lebanon","miniyeh-danniyeh":"North Lebanon","miniyeh":"North Lebanon","danniyeh":"North Lebanon",
    "zgharta":"North Lebanon","bsharri":"North Lebanon","bcharre":"North Lebanon","koura":"North Lebanon","batroun":"North Lebanon",
    "akkar":"Akkar",
    "zahle":"Bekaa","zahleh":"Bekaa","western beqaa":"Bekaa","west beqaa":"Bekaa","west bekaa":"Bekaa","rashaya":"Bekaa",
    "baalbek":"Baalbek-Hermel","hermel":"Baalbek-Hermel",
    "nabatieh":"Nabatiyeh","bint jbeil":"Nabatiyeh","marjeyoun":"Nabatiyeh","hasbaya":"Nabatiyeh",
    "sidon":"South Lebanon","saida":"South Lebanon","tyre":"South Lebanon","jezzine":"South Lebanon",
}
def token_to_canonical(token: str) -> str | None:
    if not isinstance(token, str): return None
    t = token.lower().strip()
    t = re.sub(r"\s*(district|districts|caza|qada)\s*$","",t)
    if t in gov_alias: return gov_alias[t]
    if t in district_to_gov: return district_to_gov[t]
    first = t.split(" ")[0]
    return district_to_gov.get(first)

def ensure_binary_columns(d: pd.DataFrame) -> pd.DataFrame:
    STEM_MAP = {
        "main": "State of the main roads - ",
        "secondary": "State of the secondary roads - ",
        "agricultural": "State of agricultural roads - ",
    }
    BUS_STOP_COL = "Existence of dedicated bus stops - exists"
    MODE_COLS = [
        "The main means of public transport - taxis",
        "The main means of public transport - vans",
        "The main means of public transport - buses",
    ]
    d = d.copy()
    need = [BUS_STOP_COL] + MODE_COLS
    for stem in STEM_MAP.values():
        for suf in ["good","acceptable","bad"]:
            need.append(stem + suf)
    for c in need:
        if c not in d.columns:
            d[c] = 0
        d[c] = pd.to_numeric(d[c], errors="coerce").fillna(0).astype(int)
    return d

def build_governorate(df: pd.DataFrame, mapping_df: pd.DataFrame | None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return df with Governorate (if possible) and a frame of unmatched sources."""
    d = df.copy()
    # already present?
    gov_col = find_col(d.columns, "Governorate")
    if gov_col:
        d = d.rename(columns={gov_col:"Governorate"})
        return d, pd.DataFrame(columns=["unmatched_source"])

    # try region-like column
    region_col = (find_col(d.columns,"refArea")
                  or find_any(d.columns, ["refarea","governorate","gov","mohaf","region","area","district","caza","qada"]))
    if region_col:
        tok = d[region_col].astype(str).map(last_path_component).map(clean_token)
        d["Governorate"] = tok.map(token_to_canonical)
        unmatched = d[d["Governorate"].isna()][[region_col]].drop_duplicates()
        unmatched = unmatched.rename(columns={region_col:"unmatched_source"})
        return d, unmatched

    # merge mapping if provided
    if mapping_df is not None:
        cols_l = {c.lower().strip(): c for c in mapping_df.columns}
        if not {"town","governorate"}.issubset(cols_l):
            st.error("Mapping CSV must have columns: **Town, Governorate**.")
            st.stop()
        m = mapping_df.rename(columns={cols_l["town"]:"Town", cols_l["governorate"]:"Governorate"})
        d["Town"] = d["Town"].astype(str).str.strip()
        m["Town"] = m["Town"].astype(str).str.strip()
        d = d.merge(m[["Town","Governorate"]], on="Town", how="left")
        return d, pd.DataFrame(columns=["unmatched_source"])

    # nothing available
    return d.assign(Governorate=pd.NA), pd.DataFrame(columns=["unmatched_source"])

def worst_case_condition(row, g, a, b) -> str:
    if row[b] == 1: return "Bad"
    if row[a] == 1: return "Acceptable"
    if row[g] == 1: return "Good"
    return "Unknown"

def pct(series: pd.Series) -> float:
    return pd.to_numeric(series, errors="coerce").fillna(0).mean() * 100

# ----------------------- Sidebar -----------------------
st.sidebar.header("Data")
src = st.sidebar.file_uploader("Upload raw CSV", type=["csv"])
mapping = st.sidebar.file_uploader("Optional: Town→Governorate mapping (CSV)", type=["csv"])
st.sidebar.divider()
st.sidebar.header("Filters for visuals")
road_type = st.sidebar.radio("Road type", ["main","secondary","agricultural"], index=0)
require_bus_stop = st.sidebar.checkbox("Only towns WITH dedicated bus stops", value=False)
mode_filter = st.sidebar.multiselect("Transport modes", ["taxis","vans","buses"], default=["taxis","vans","buses"])

# Try fallback to local file (for local dev) if nothing uploaded
def try_default_load() -> pd.DataFrame | None:
    here = Path(__file__).parent
    for p in (here / "Public Transportation.csv", here.parent / "Public Transportation.csv"):
        if p.exists():
            return pd.read_csv(p, encoding="utf-8-sig")
    return None

if src is not None:
    raw = pd.read_csv(src, encoding="utf-8-sig")
else:
    raw = try_default_load()
    if raw is None:
        st.info("Upload your **raw CSV** to start.")
        st.stop()

# Drop common metadata and normalize binaries
DROP_COLS = ["Observation URI","publisher","dataset","references"]
raw = raw.drop(columns=[c for c in DROP_COLS if c in raw.columns], errors="ignore")
raw = ensure_binary_columns(raw)

mapping_df = pd.read_csv(mapping, encoding="utf-8-sig") if mapping is not None else None
gov_df, unmatched = build_governorate(raw, mapping_df)

# Template for Town→Governorate if still missing
if gov_df["Governorate"].isna().all():
    towns = pd.DataFrame(sorted(gov_df["Town"].astype(str).unique()), columns=["Town"])
    towns["Governorate"] = ""
    st.warning("No `Governorate` or region column detected. Upload a **Town→Governorate** mapping.")
    st.download_button("Download Town→Governorate template (CSV)",
                       towns.to_csv(index=False).encode("utf-8-sig"),
                       file_name="town_to_governorate_template.csv",
                       mime="text/csv")
    st.stop()

# Keep only rows we could map
df_clean = gov_df.dropna(subset=["Governorate"]).copy()

# ---------- Downloads (cleaned CSV + Excel) ----------
@st.cache_data
def build_downloads(clean_df: pd.DataFrame) -> tuple[bytes, bytes]:
    # CSV
    csv_bytes = clean_df.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")

    # Excel with aggregates + unmatched
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="xlsxwriter") as xw:
        clean_df.to_excel(xw, index=False, sheet_name="clean_rows")
        # aggregates
        COL_TOWN      = "Town"
        COL_BUS_STOP  = "Existence of dedicated bus stops - exists"
        COL_MAIN_GOOD = "State of the main roads - good"
        COL_MAIN_ACC  = "State of the main roads - acceptable"
        COL_MAIN_BAD  = "State of the main roads - bad"
        COL_MODE_VAN  = "The main means of public transport - vans"
        COL_MODE_TAXI = "The main means of public transport - taxis"
        COL_MODE_BUS  = "The main means of public transport - buses"
        have_core = {COL_TOWN, COL_BUS_STOP, COL_MAIN_GOOD, COL_MAIN_ACC, COL_MAIN_BAD}.issubset(clean_df.columns)
        if have_core:
            agg = clean_df.groupby("Governorate").agg(
                towns=(COL_TOWN,"count"),
                pct_bus_stops=(COL_BUS_STOP,pct),
                pct_main_good=(COL_MAIN_GOOD,pct),
                pct_main_acc=(COL_MAIN_ACC,pct),
                pct_main_bad=(COL_MAIN_BAD,pct),
                pct_mode_van=(COL_MODE_VAN,pct) if COL_MODE_VAN in clean_df.columns else ("Governorate","size"),
                pct_mode_taxi=(COL_MODE_TAXI,pct) if COL_MODE_TAXI in clean_df.columns else ("Governorate","size"),
                pct_mode_bus=(COL_MODE_BUS,pct) if COL_MODE_BUS in clean_df.columns else ("Governorate","size"),
            ).reset_index()
        else:
            agg = pd.DataFrame(columns=["Governorate","towns","pct_bus_stops","pct_main_good","pct_main_acc","pct_main_bad",
                                        "pct_mode_van","pct_mode_taxi","pct_mode_bus"])
        agg.to_excel(xw, index=False, sheet_name="agg_governorate")
        # unmatched (from detection step)
        unmatched.to_excel(xw, index=False, sheet_name="unmatched_source")
    return csv_bytes, output.getvalue()

csv_bytes, xlsx_bytes = build_downloads(df_clean)

colA, colB = st.columns(2)
with colA:
    st.download_button("⬇️ Download CLEANED.csv", data=csv_bytes, file_name="Public Transportation - CLEANED.csv",
                       mime="text/csv", use_container_width=True)
with colB:
    st.download_button("⬇️ Download CLEANED.xlsx", data=xlsx_bytes, file_name="Public Transportation - CLEANED.xlsx",
                       mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                       use_container_width=True)

st.write("---")

# ---------- Visuals (match your PPT) ----------
BUS_STOP_COL = "Existence of dedicated bus stops - exists"
MODE_MAP = {
    "taxis": "The main means of public transport - taxis",
    "vans":  "The main means of public transport - vans",
    "buses": "The main means of public transport - buses",
}
STEM_MAP = {
    "main": "State of the main roads - ",
    "secondary": "State of the secondary roads - ",
    "agricultural": "State of agricultural roads - ",
}

# Filter by bus stops if requested
work = df_clean.copy()
if require_bus_stop:
    work = work[work[BUS_STOP_COL] == 1]

# Worst-case condition for MAIN roads (to mirror PPT)
stem = STEM_MAP["main"] if road_type == "main" else STEM_MAP[road_type]
g, a, b = stem + "good", stem + "acceptable", stem + "bad"
for c in (g, a, b):
    if c not in work.columns: work[c] = 0
work["main_condition"] = np.select(
    [work[b].eq(1), work[a].eq(1), work[g].eq(1)],
    ["Bad","Acceptable","Good"], default="Unknown"
)

# 1) Ranked bar — % towns with bus stops by governorate
gov_bus = (work.groupby("Governorate")[BUS_STOP_COL].mean().mul(100).round(1)
                .sort_values(ascending=False)).rename("% with bus stops").reset_index()
fig1 = px.bar(
    gov_bus, x="Governorate", y="% with bus stops",
    text=gov_bus["% with bus stops"].astype(str) + "%",
    title="% of Towns with Bus Stops (by Governorate)"
)
fig1.update_traces(textposition="outside", cliponaxis=False)
st.plotly_chart(fig1, use_container_width=True)

# 2) 100% stacked bars — Main road condition shares per governorate (worst-case)
dist = work.groupby(["Governorate","main_condition"]).size().reset_index(name="towns")
pivot = dist.pivot(index="Governorate", columns="main_condition", values="towns").fillna(0)
if "Unknown" in pivot.columns:
    denom = pivot[["Good","Acceptable","Bad"]].sum(axis=1, min_count=1)
else:
    denom = pivot.sum(axis=1, min_count=1)
share = (pivot.div(denom, axis=0)[["Good","Acceptable","Bad"]] * 100).fillna(0)
v2 = share.reset_index().melt(id_vars="Governorate", var_name="Condition", value_name="Percent")
order_bad = v2[v2["Condition"]=="Bad"].sort_values("Percent", ascending=False)["Governorate"].tolist()
fig2 = px.bar(
    v2, x="Governorate", y="Percent", color="Condition",
    category_orders={"Governorate": order_bad, "Condition": ["Good","Acceptable","Bad"]},
    title=f"{road_type.capitalize()} Road Conditions — % of Towns by Governorate (worst-case)"
)
fig2.update_layout(barmode="stack")
fig2.update_traces(texttemplate="%{y:.1f}%", textposition="inside", insidetextanchor="middle", textfont_size=11, cliponaxis=False)
st.plotly_chart(fig2, use_container_width=True)

# 3) Heatmap — Governorate × Condition
fig3 = px.imshow(
    share[["Good","Acceptable","Bad"]],
    labels=dict(x="Condition", y="Governorate", color="% of Towns"),
    x=["Good","Acceptable","Bad"],
    title="Main Road Condition Distribution (Heatmap)",
    zmin=0, zmax=100, aspect="auto"
)
st.plotly_chart(fig3, use_container_width=True)

# 4) Donut — overall share of towns reporting each main transport mode
mode_cols = [MODE_MAP[m] for m in mode_filter if MODE_MAP[m] in work.columns]
mode_counts = {m.title(): int(work[MODE_MAP[m]].sum()) for m in mode_filter if MODE_MAP[m] in work.columns}
pie_df = pd.DataFrame({"Mode": list(mode_counts.keys()), "Towns reporting mode": list(mode_counts.values())})
if not pie_df.empty and pie_df["Towns reporting mode"].sum() > 0:
    fig4 = px.pie(pie_df, names="Mode", values="Towns reporting mode", hole=0.35,
                  title="Share of Towns Reporting Each Main Transport Mode")
    st.plotly_chart(fig4, use_container_width=True)
else:
    st.info("No transport modes selected (or none present after filters).")

# 5) Scatter — Coverage vs Bad roads with trendline
agg_sc = pd.DataFrame({
    "Governorate": share.index,
    "pct_main_bad": share["Bad"].values,
    "pct_bus_stops": gov_bus.set_index("Governorate")["% with bus stops"].reindex(share.index).values
})
fig5 = px.scatter(
    agg_sc, x="pct_bus_stops", y="pct_main_bad", text="Governorate",
    labels={"pct_bus_stops":"% with bus stops","pct_main_bad":"% with bad main roads"},
    title="Coverage vs Condition"
)
if len(agg_sc) >= 2 and pd.Series(agg_sc["pct_bus_stops"]).nunique() > 1:
    x = agg_sc["pct_bus_stops"].to_numpy(); y = agg_sc["pct_main_bad"].to_numpy()
    m, b_ = np.polyfit(x, y, 1)
    xs = np.linspace(x.min(), x.max(), 100); ys = m*xs + b_
    fig5.add_scatter(x=xs, y=ys, mode="lines", name="trend")
st.plotly_chart(fig5, use_container_width=True)

st.write("---")
st.caption(
    "Notes: Each town is counted once. The 'worst-case' condition encodes **Bad > Acceptable > Good**. "
    "Use the sidebar to filter and to supply a Town→Governorate mapping when needed."
)
