"""Beijing multi-site air quality dashboard (Streamlit).

Run:  streamlit run app.py
Needs Data/processed/ created by preprocess.py
"""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

CLEAN_DATA_PATH = "Data/processed/beijing_air_clean.parquet"
MISSING_REPORT_PATH = "Data/processed/missing_report.csv"
POLLUTANTS = ["PM2.5", "PM10", "SO2", "NO2", "CO", "O3"]
WEATHER = ["TEMP", "PRES", "DEWP", "RAIN", "WSPM"]
UNITS = {
    "PM2.5": "µg/m³",
    "PM10": "µg/m³",
    "SO2": "µg/m³",
    "NO2": "µg/m³",
    "CO": "µg/m³",
    "O3": "µg/m³",
    "TEMP": "°C",
    "PRES": "hPa",
    "DEWP": "°C",
    "RAIN": "mm",
    "WSPM": "m/s",
}
COMPASS = [
    "N",
    "NNE",
    "NE",
    "ENE",
    "E",
    "ESE",
    "SE",
    "SSE",
    "S",
    "SSW",
    "SW",
    "WSW",
    "W",
    "WNW",
    "NW",
    "NNW",
]
CATS = ["Excellent", "Good", "Light", "Moderate", "Heavy", "Severe"]
CAT_COLORS = ["#2e7d32", "#9ccc65", "#fdd835", "#fb8c00", "#e53935", "#6a1b9a"]
SEASONS = ["Spring", "Summer", "Autumn", "Winter"]
SITE_TYPES = ["Urban", "Suburban", "Regional Background"]
SITE_TYPE_COLORS = {"Urban": "#e53935", "Suburban": "#fb8c00", "Regional Background": "#2e7d32"}
STATION_COORDS = {
    "Aotizhongxin": (39.982, 116.397),
    "Changping": (40.217, 116.230),
    "Dingling": (40.292, 116.220),
    "Dongsi": (39.929, 116.417),
    "Guanyuan": (39.929, 116.339),
    "Gucheng": (39.914, 116.184),
    "Huairou": (40.328, 116.628),
    "Nongzhanguan": (39.937, 116.461),
    "Shunyi": (40.127, 116.655),
    "Tiantan": (39.886, 116.407),
    "Wanliu": (39.987, 116.287),
    "Wanshouxigong": (39.878, 116.352),
}
WEEKDAYS = [
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday",
]
THRESHOLDS = {"WHO 2021 (15)": 15, "China Class 1 (35)": 35, "China Class 2 (75)": 75}

st.set_page_config(page_title="Beijing Air Quality", layout="wide")


@st.cache_data
def load() -> pd.DataFrame:
    df = pd.read_parquet(CLEAN_DATA_PATH)
    df["pm25_category"] = pd.Categorical(
        df["pm25_category"], categories=CATS, ordered=True
    )
    df["site_type"] = pd.Categorical(
        df["site_type"], categories=SITE_TYPES, ordered=True
    )
    return df


@st.cache_data
def filtered(stations, start, end, seasons, drop_imputed) -> pd.DataFrame:
    df = load()
    m = (
        df["station"].isin(stations)
        & df["season"].isin(seasons)
        & df["date"].between(pd.Timestamp(start), pd.Timestamp(end))
    )
    if drop_imputed:
        m &= ~df["imputed_any"]
    return df[m]


def daily_table(d: pd.DataFrame) -> pd.DataFrame:
    agg = {c: "mean" for c in POLLUTANTS + WEATHER}
    agg["RAIN"] = "sum"
    return d.groupby(["station", "date"], as_index=False).agg(agg)


# ---------------------------------------------------------------- sidebar
full = load()
st.sidebar.header("Filters")
all_stations = sorted(full["station"].unique())
stations = st.sidebar.multiselect("Stations", all_stations, default=all_stations)
dmin, dmax = full["date"].min().date(), full["date"].max().date()
rng = st.sidebar.date_input("Date range", (dmin, dmax), min_value=dmin, max_value=dmax)
seasons = st.sidebar.multiselect("Seasons", SEASONS, default=SEASONS)
pol = st.sidebar.selectbox("Pollutant", POLLUTANTS, index=0)
drop_imputed = st.sidebar.checkbox("Exclude rows with imputed values", value=False)
st.sidebar.caption("Applies to every tab except Data quality.")

if len(rng) != 2 or not stations or not seasons:
    st.warning("Choose at least one station and season, and a full date range.")
    st.stop()
d = filtered(tuple(stations), rng[0], rng[1], tuple(seasons), drop_imputed)
if d.empty:
    st.warning("No data for this selection.")
    st.stop()
unit = UNITS[pol]
daily = daily_table(d)

st.title("Beijing air quality, 2013–2017")
st.caption(
    "12 monitoring stations, hourly readings, 1 Mar 2013 to 28 Feb 2017 (UCI Beijing Multi-Site Air Quality). "
    f"Showing {len(d):,} hourly records."
)

tab_over, tab_pat, tab_drv, tab_std, tab_dq = st.tabs(
    ["Overview", "Patterns", "Drivers", "Health standards", "Data quality"]
)

# ---------------------------------------------------------------- overview
with tab_over:
    by_station = (
        d.groupby(["station", "site_type"], as_index=False)[pol].mean().sort_values(pol)
    )
    by_station["lat"] = by_station["station"].map(lambda s: STATION_COORDS[s][0])
    by_station["lon"] = by_station["station"].map(lambda s: STATION_COORDS[s][1])
    days_over = (daily["PM2.5"] > 75).mean() * 100
    c1, c2, c3, c4 = st.columns(4)
    c1.metric(f"Mean {pol}", f"{d[pol].mean():.1f} {unit}")
    c2.metric("Days above China limit (PM2.5 > 75)", f"{days_over:.1f}%")
    c3.metric(
        "Highest-mean station",
        by_station.iloc[-1]["station"],
        f"{by_station.iloc[-1][pol]:.1f} {unit}",
        delta_color="off",
    )
    c4.metric(
        "Lowest-mean station",
        by_station.iloc[0]["station"],
        f"{by_station.iloc[0][pol]:.1f} {unit}",
        delta_color="off",
    )

    left, right = st.columns([3, 2])
    with left:
        mode = st.radio("Group lines by", ["Site type", "Station"], horizontal=True)
        key = "site_type" if mode == "Site type" else "station"
        monthly = (
            d.groupby([pd.Grouper(key="datetime", freq="MS"), key])[pol]
            .mean()
            .reset_index()
        )
        color_kwargs = {"color_discrete_map": SITE_TYPE_COLORS} if key == "site_type" else {}
        fig = px.line(
            monthly,
            x="datetime",
            y=pol,
            color=key,
            template="plotly_white",
            category_orders={key: SITE_TYPES if key == "site_type" else sorted(d["station"].unique())},
            labels={"datetime": "Month", pol: f"{pol} ({unit})"},
            **color_kwargs,
        )
        fig.update_layout(title=f"Monthly mean {pol}", legend_title=None, height=380)
        st.plotly_chart(fig, width="stretch")
        st.caption(
            "I used a **line chart** here because tracking continuous monthly averages over time makes "
            "long-term cyclical trends and seasonal spikes obvious at a glance. From this, we learn that "
            "pollution repeatedly surges to dangerous levels every winter and drops significantly in summer. "
            "It also reveals that urban stations face consistently higher baseline pollution than suburban "
            "areas, while the regional background station (Dingling) stays noticeably lower year-round. "
            "*(Note: because 2013 and 2017 are partial years, I focused on month-by-month and seasonal "
            "comparisons rather than calendar-year totals.)*"
        )
    with right:
        view = st.radio("Display stations as", ["Map", "Bar chart"], horizontal=True)
        if view == "Map":
            fig = px.scatter_map(
                by_station,
                lat="lat",
                lon="lon",
                hover_name="station",
                hover_data={"site_type": True, pol: ":.1f", "lat": False, "lon": False},
                color="site_type",
                color_discrete_map=SITE_TYPE_COLORS,
                category_orders={"site_type": SITE_TYPES},
                size=pol,
                size_max=20,
                zoom=8.3,
                center={"lat": 40.08, "lon": 116.40},
                map_style="carto-positron",
                labels={pol: f"Mean {pol} ({unit})", "site_type": "Site type"},
            )
            fig.update_layout(
                title=f"Monitoring stations (size = {pol})",
                legend_title=None,
                height=380,
                margin=dict(l=0, r=0, t=40, b=0),
            )
            st.plotly_chart(fig, width="stretch")
            st.caption(
                "I used an **interactive geographic map** with marker sizes scaled to pollutant concentration "
                "because air quality is strongly governed by local topography and urban density. From this, we "
                "learn that the highest pollution concentrates in the southern and western urban stations (like "
                "Gucheng and Wanshouxigong), while northern suburban and regional background stations (like "
                "Huairou and Dingling) stay noticeably cleaner near the sheltering mountain ranges."
            )
        else:
            fig = px.bar(
                by_station,
                x=pol,
                y="station",
                color="site_type",
                orientation="h",
                template="plotly_white",
                category_orders={"site_type": SITE_TYPES},
                color_discrete_map=SITE_TYPE_COLORS,
                labels={pol: f"Mean {pol} ({unit})", "station": "", "site_type": "Site type"},
            )
            fig.update_layout(title=f"Mean {pol} by station", legend_title=None, height=380)
            st.plotly_chart(fig, width="stretch")
            st.caption(
                "I used a **horizontal bar chart** ranked by concentration because it provides a clear, clean "
                "way to compare all 12 stations without cramped axis labels. From this, we learn that western "
                "and southern urban stations (like Gucheng and Wanshouxigong) experience the highest average "
                "pollution, whereas the regional background site (Dingling) and northern suburban areas (like "
                "Huairou) have the cleanest air. This reflects the intense spatial gradient in traffic, "
                "industry, and population density across Beijing."
            )

# ---------------------------------------------------------------- patterns
with tab_pat:
    pivot = d.groupby(["month", "hour"])[pol].mean().unstack("hour")
    fig = px.imshow(
        pivot,
        aspect="auto",
        color_continuous_scale="YlOrRd",
        template="plotly_white",
        labels=dict(x="Hour of day", y="Month", color=f"{pol}"),
    )
    fig.update_layout(title=f"Mean {pol} by month and hour", height=420)
    st.plotly_chart(fig, width="stretch")
    st.caption(
        "I used a **2D heatmap** because it can compress two distinct cyclical time dimensions (the 12 "
        "months and the 24 hours of the day) into a single visual view, using colour intensity to expose "
        "hot spots. From this, we learn that extreme pollution events concentrate heavily in late evenings "
        "and early mornings (21:00 to 08:00) during the winter months (November to February). This peak "
        "occurs because domestic heating emissions increase right when cold night air creates temperature "
        "inversions that trap pollutants near ground level."
    )

    sm = d.groupby("season")[pol].mean()
    if {"Winter", "Summer"} <= set(sm.index):
        st.info(
            f"Winter mean {pol} is {sm['Winter'] / sm['Summer']:.1f}× the summer mean "
            f"({sm['Winter']:.1f} vs {sm['Summer']:.1f} {unit}) for this selection."
        )

    a, b = st.columns(2)
    with a:
        hs = d.groupby(["season", "hour"], as_index=False)[pol].mean()
        fig = px.line(
            hs,
            x="hour",
            y=pol,
            color="season",
            template="plotly_white",
            category_orders={"season": SEASONS},
            labels={pol: f"{pol} ({unit})"},
        )
        fig.update_layout(title="Daily cycle by season", legend_title=None, height=340)
        st.plotly_chart(fig, width="stretch")
        st.caption(
            "I used a **multi-line chart** to overlay the 24-hour diurnal cycle for all four seasons on the "
            "same axes so their shapes and amplitudes can be compared directly. From this, we learn that all "
            "seasons share a midday dip (around 12:00 to 16:00) when solar heating creates convective mixing "
            "that disperses pollutants, followed by a sharp evening rise. Crucially, we also learn that "
            "the winter baseline is dramatically higher at every single hour of the day compared to any "
            "other season."
        )
    with b:
        wk = d.groupby("weekday")[pol].mean().reindex(WEEKDAYS).reset_index()
        fig = px.bar(
            wk,
            x="weekday",
            y=pol,
            template="plotly_white",
            labels={pol: f"{pol} ({unit})", "weekday": ""},
        )
        fig.update_layout(title="Mean by day of week", height=340)
        st.plotly_chart(fig, width="stretch")
        st.caption(
            "I used a **bar chart** because the days of the week are discrete categorical variables, making "
            "it easy to evaluate differences between weekdays and weekends. From this, we learn that the "
            "day-of-week effect is very modest (averages hover between ~73 µg/m³ on Mondays and ~87 µg/m³ "
            "on Saturdays). This shows that weekly commuter traffic cycles have far less impact on overall "
            "pollution levels than broad seasonal heating demands and regional weather systems."
        )

# ---------------------------------------------------------------- drivers
with tab_drv:
    st.subheader("What moves pollution?")
    cols = POLLUTANTS + WEATHER
    corr = daily[cols].corr(method="spearman")
    fig = px.imshow(
        corr,
        text_auto=".2f",
        zmin=-1,
        zmax=1,
        color_continuous_scale="RdBu_r",
        template="plotly_white",
        aspect="auto",
    )
    fig.update_layout(
        title="Spearman correlation of daily values (RAIN = daily total)", height=520
    )
    st.plotly_chart(fig, width="stretch")
    st.caption(
        "I used a **correlation heatmap** with numeric annotations because it summarizes all pairwise "
        "relationships between pollutants and meteorological variables in a single view. I chose "
        "**Spearman rank correlation** because pollutant concentrations are heavily right-skewed with extreme "
        "outliers, and I aggregated data into daily values to avoid hour-to-hour autocorrelation. From this, "
        "we learn that primary combustion pollutants (PM2.5, PM10, CO, SO2, and NO2) strongly co-move "
        "(r > 0.6–0.8), confirming common sources such as fossil fuel burning. Conversely, ozone (O3) shows "
        "an inverse correlation with NO2 due to photochemical titration, and wind speed (WSPM) shows a "
        "consistent negative relationship with particulate matter, confirming the role of wind in dispersing haze."
    )

    drv = st.selectbox(
        "Weather variable",
        ["WSPM", "TEMP", "DEWP", "PRES"],
        format_func=lambda v: f"{v} ({UNITS[v]})",
    )
    bins = pd.qcut(d[drv], 12, duplicates="drop")
    g = d.groupby(bins, observed=True).agg(
        x=(drv, "mean"),
        med=(pol, "median"),
        q1=(pol, lambda s: s.quantile(0.25)),
        q3=(pol, lambda s: s.quantile(0.75)),
    )
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=g["x"], y=g["q3"], line=dict(width=0), showlegend=False, hoverinfo="skip"
        )
    )
    fig.add_trace(
        go.Scatter(
            x=g["x"],
            y=g["q1"],
            fill="tonexty",
            line=dict(width=0),
            fillcolor="rgba(230,120,40,0.25)",
            name="25th–75th percentile",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=g["x"],
            y=g["med"],
            mode="lines+markers",
            name="Median",
            line=dict(color="#d84315"),
        )
    )
    fig.update_layout(
        template="plotly_white",
        height=380,
        title=f"{pol} by {drv} (12 equal-count bins)",
        xaxis_title=f"{drv} ({UNITS[drv]})",
        yaxis_title=f"{pol} ({unit})",
    )
    st.plotly_chart(fig, width="stretch")
    st.caption(
        "I used a **band chart** (plotting the median line with an interquartile range across 12 "
        "equal-frequency bins) because standard scatter plots become an unreadable blur with over 400,000 "
        "points. From this, we learn how pollution levels respond non-linearly to specific weather conditions: "
        "for example, as wind speed increases from calm to over 4 m/s, median PM2.5 drops sharply from "
        "~79 µg/m³ to ~22 µg/m³. The narrowing IQR band also shows that severe haze episodes are effectively "
        "suppressed once winds pick up."
    )
    if drv == "WSPM" and len(g) > 1:
        st.info(
            f"Median {pol} falls from {g['med'].iloc[0]:.0f} in the calmest bin to "
            f"{g['med'].iloc[-1]:.0f} {unit} in the windiest bin: wind disperses pollution."
        )

    mean_rose = d.groupby("wd")[pol].mean().reindex(COMPASS)
    freq = d["wd"].value_counts(normalize=True).reindex(COMPASS).fillna(0) * 100
    r1, r2 = st.columns(2)
    for col, vals, ttl, scale in [
        (r1, mean_rose, f"Mean {pol} by wind direction", "YlOrRd"),
        (r2, freq, "How often wind blows from each direction (%)", "Blues"),
    ]:
        fig = go.Figure(
            go.Barpolar(
                r=vals.values,
                theta=COMPASS,
                marker={"color": vals.values, "colorscale": scale},
            )
        )
        fig.update_layout(
            template="plotly_white",
            title=ttl,
            height=400,
            polar={
                "angularaxis": {
                    "type": "category",
                    "direction": "clockwise",
                    "rotation": 90,
                    "categoryarray": COMPASS,
                }
            },
        )
        col.plotly_chart(fig, width="stretch")
    st.caption(
        "I used two **polar wind rose charts** side by side because polar coordinates naturally reflect "
        "physical 360° compass directions. Displaying average concentration on the left and wind frequency "
        "on the right lets us contrast pollution intensity against prevailing wind patterns. From reading "
        "them together, we learn that while clean winds blow most frequently from the northwest (originating "
        "over the mountains), the most severe pollution arrives with slower winds from the south and "
        "south-southwest (S, SSW). These southerly winds transport regional industrial emissions from "
        "neighbouring provinces straight into the Beijing basin, where the northern mountains block dispersion."
    )

# ---------------------------------------------------------------- standards
with tab_std:
    st.subheader("How often is the air above the limits?")
    rows = []
    for st_name, g_ in daily.groupby("station"):
        rows.append(
            {
                "station": st_name,
                **{k: (g_["PM2.5"] > v).mean() * 100 for k, v in THRESHOLDS.items()},
            }
        )
    exc = (
        pd.DataFrame(rows)
        .set_index("station")
        .sort_values("China Class 2 (75)", ascending=False)
    )
    fig = px.bar(
        exc.reset_index().melt(
            "station", var_name="Limit", value_name="% of days above"
        ),
        x="station",
        y="% of days above",
        color="Limit",
        barmode="group",
        template="plotly_white",
    )
    fig.update_layout(
        title="Share of days with daily-mean PM2.5 above each 24-hour limit",
        height=380,
        legend_title=None,
        xaxis_title="",
    )
    st.plotly_chart(fig, width="stretch")
    st.caption(
        "I used a **grouped bar chart** so that multiple public health benchmarks could be evaluated "
        "side by side across all 12 monitoring stations. From this, we learn that chronic exposure in Beijing "
        "was remarkably severe during this period: every station exceeded the strict WHO 2021 daily guideline "
        "(15 µg/m³) on over 85% to 95% of all days. Furthermore, around 35% to 45% of days also violated "
        "China's own Class 2 standard (75 µg/m³), underscoring the huge gap between local air quality and "
        "international health targets."
    )

    cat = d.dropna(subset=["pm25_category"])
    split = st.radio(
        "Split categories by", ["season", "station", "site_type"], horizontal=True
    )
    share = (
        cat.groupby([split, "pm25_category"], observed=True)
        .size()
        .groupby(level=0)
        .transform(lambda s: s / s.sum() * 100)
        .reset_index(name="share")
    )
    fig = px.bar(
        share,
        x=split,
        y="share",
        color="pm25_category",
        template="plotly_white",
        category_orders={"pm25_category": CATS, "season": SEASONS, "site_type": SITE_TYPES},
        color_discrete_sequence=CAT_COLORS,
        labels={"share": "% of hours", "pm25_category": "Grade"},
    )
    fig.update_layout(
        title="Air quality grade distribution (rolling 24 h mean PM2.5)",
        height=400,
        xaxis_title="",
    )
    st.plotly_chart(fig, width="stretch")
    st.caption(
        "I used a **100% stacked bar chart** because it allows us to compare the relative proportion of each "
        "air quality grade across seasons, stations, or site types without being distorted by sample size. "
        "From this, we learn that poor air quality is heavily concentrated in winter, where 'Heavy' and "
        "'Severe' categories make up over 30% of all hours, whereas summer is dominated by 'Good' and "
        "'Excellent' air. *(Note: because China's HJ 633-2012 AQI grades are legally defined for 24-hour "
        "exposures, I calculated them using a rolling 24-hour mean per station rather than single hourly spikes.)*"
    )

# ---------------------------------------------------------------- data quality
with tab_dq:
    st.subheader("Preprocessing summary")
    try:
        rep = pd.read_csv(MISSING_REPORT_PATH)
        fig = px.bar(
            rep.sort_values("before", ascending=False),
            x="column",
            y=["before", "after"],
            barmode="group",
            template="plotly_white",
            labels={"value": "Missing values", "variable": ""},
        )
        fig.update_layout(title="Missing values before and after cleaning", height=360)
        st.plotly_chart(fig, width="stretch")
        st.caption(
            "I used a **grouped before-and-after bar chart** to visually audit the impact of my cleaning "
            "pipeline on each individual feature. From this, we learn that missing data was largely limited "
            "to gas sensors like CO (~20,700 missing values, or 4.9%) and O3 (~13,300, or 3.2%), whereas "
            "weather sensors were nearly 100% complete (<0.1% missing). The zero 'after' bars show that my "
            "hybrid imputation strategy (linear interpolation for short gaps <=6 hours and station-month-hour "
            "median matching for larger gaps) successfully resolved all missing values without discarding rows."
        )
    except Exception:
        st.info("Missing report CSV not found.")
    q = full.groupby("station").agg(
        rows=("station", "size"),
        imputed_rows_pct=("imputed_any", "mean"),
        pm25_gt_pm10_pct=("pm_inconsistent", "mean"),
        pm25_above_500_pct=("extreme_pm25", "mean"),
    )
    q[["imputed_rows_pct", "pm25_gt_pm10_pct", "pm25_above_500_pct"]] *= 100
    st.dataframe(q.round(2), width="stretch")
    st.markdown("""
### Data Cleaning Methodology & Preprocessing Decisions

1. **Dataset Unification**: The 12 independent station CSV files (March 2013 to February 2017) were concatenated into a single master dataset of 420,768 hourly records. Timestamps were reconstructed into unified datetime indices and verified for zero duplicates.
2. **Quality Audit & Pre-imputation Flags**: Before replacing missing data, anomaly flags were calculated on raw observations:
   - `imputed_any`: Flags any record where one or more features required imputation (allowing sensitivity testing via the sidebar toggle).
   - `pm_inconsistent`: Flags physical discrepancies where $PM_{2.5} > PM_{10}$ (~5% of observations). These are preserved as genuine instrument calibration artifacts rather than dropped.
   - `extreme_pm25`: Flags extreme haze events where $PM_{2.5} > 500$ µg/m³ (933 records). These represent verified episodic pollution emergencies and were intentionally preserved.
3. **Two-Stage Imputation Pipeline**:
   - *Short-duration gaps (≤ 6 consecutive hours)*: Handled via station-specific linear interpolation (`limit=6, limit_area='inside'`) to respect short-term atmospheric persistence.
   - *Extended gaps (> 6 hours)*: Filled using the median value grouped by station, month, and hour. Medians were chosen over means to avoid distortion from heavy right-skewed extremes while retaining typical diurnal and seasonal baselines.
   - *Wind direction (`wd`)*: Forward-filled for short gaps (≤ 6 h), with remaining gaps imputed using the station-month modal direction.

---

### Environmental Site Classification

To analyze spatial heterogeneity, stations are categorized into three distinct environmental site types:
- **Urban (8 stations)**: Aotizhongxin, Dongsi, Guanyuan, Gucheng, Nongzhanguan, Tiantan, Wanliu, Wanshouxigong (situated within Beijing's central ring roads, dominated by dense traffic and commercial activity).
- **Suburban (3 stations)**: Changping, Huairou, Shunyi (situated in peripheral satellite districts with lower population density).
- **Regional Background (1 station)**: Dingling (located at the Ming Tombs reservoir, surrounded by dense vegetation and mountains, serving as a baseline reference for unpolluted regional air).

*Classification reference:* Beijing Municipal Environmental Monitoring Center (BMEMC) network architecture, as documented by Zhang et al. (2015), *Atmosphere*, 6(11), 1604–1618.
""")
