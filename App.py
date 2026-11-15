"""Beijing multi-site air quality dashboard (Streamlit).

Run:  streamlit run app.py
Needs Data/processed/ created by preprocess.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

DATA = Path(__file__).parent / "Data" / "processed"
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
    df = pd.read_parquet(DATA / "beijing_air_clean.parquet")
    df["pm25_category"] = pd.Categorical(
        df["pm25_category"], categories=CATS, ordered=True
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
        fig = px.line(
            monthly,
            x="datetime",
            y=pol,
            color=key,
            template="plotly_white",
            labels={"datetime": "Month", pol: f"{pol} ({unit})"},
        )
        fig.update_layout(title=f"Monthly mean {pol}", legend_title=None, height=380)
        st.plotly_chart(fig, width="stretch")
        st.caption(
            "A **line chart** is used here because it shows how the monthly average changes over time and "
            "makes seasonal peaks easy to spot. From this we can see a clear repeating winter spike in "
            "PM2.5 each year, with urban stations consistently higher than suburban/rural ones. "
            "Note: the first and last years are partial (Mar 2013 and Feb 2017), so compare months and "
            "seasons rather than calendar-year totals."
        )
    with right:
        fig = px.bar(
            by_station,
            x=pol,
            y="station",
            color="site_type",
            orientation="h",
            template="plotly_white",
            labels={pol: f"Mean {pol} ({unit})", "station": ""},
        )
        fig.update_layout(title=f"Mean {pol} by station", legend_title=None, height=380)
        st.plotly_chart(fig, width="stretch")
        st.caption(
            "A **horizontal bar chart** is used to rank all 12 stations side by side, making it "
            "easy to compare magnitudes at a glance. From this we can see that inner-city (urban) "
            "stations like Gucheng consistently record higher pollution than suburban/rural stations "
            "like Dingling. Site type is an assumed grouping based on station location — see the "
            "Data quality tab for details."
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
        "A **heatmap** is used because it can display two time dimensions (month and hour) simultaneously, "
        "with colour intensity showing concentration. From this we learn that the highest pollution occurs "
        "during winter nights and early mornings (around 22:00–08:00 in Dec–Feb), likely driven by heating "
        "emissions trapped under stable nighttime boundary layers."
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
            "A **multi-line chart** is used to overlay the 24-hour cycle for each season on the same axes, "
            "making the shape and level of daily variation directly comparable. From this we learn that all "
            "seasons show a midday dip (when convective mixing disperses pollutants) and a late-evening rise, "
            "but winter concentrations are consistently higher across all hours."
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
            "A **bar chart** is used for weekday means because each day is a discrete category, and bars "
            "make small differences in height easy to judge. From this we learn that the weekday effect is "
            "small compared to the seasonal differences shown above, suggesting that traffic-related weekly "
            "patterns play a minor role relative to meteorology and heating demand."
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
        "A **correlation heatmap** with annotated coefficients is used because it reveals the strength and "
        "direction of pairwise relationships across all pollutants and weather variables at once. Spearman "
        "rank correlation is chosen over Pearson because pollutant distributions are strongly right-skewed, "
        "and daily averages are used to reduce hour-to-hour autocorrelation. From this we learn that PM2.5, "
        "PM10, SO2, NO2 and CO are all positively correlated (they share emission sources), while O3 is "
        "negatively correlated with the other pollutants (it is consumed by NO₂ in urban photochemistry). "
        "WSPM (wind speed) shows a negative correlation with most pollutants, confirming that wind disperses "
        "pollution. Note: correlation is not causation — temperature and dew point partly proxy for season "
        "(heating demand) and humidity."
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
        "A **band chart** (median line with a shaded interquartile range) is used because it shows both the "
        "central trend and the spread of pollution across equal-count bins of the weather variable. This "
        "reveals whether the relationship is monotonic and how much variability exists at each level."
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
        "**Polar bar charts (wind roses)** are used because they naturally map compass directions to angular "
        "position, making directional patterns intuitive. The left rose shows mean concentration by wind "
        "direction and the right shows how frequently wind blows from each direction. From reading them "
        "together we learn that the highest PM2.5 arrives on southerly winds (S, SSW), which carry polluted "
        "air from industrialised areas south of Beijing, while the most frequent winds blow from the "
        "northwest, bringing cleaner air from the mountains. Wind direction here means the direction the "
        "wind comes from."
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
        "A **grouped bar chart** is used so that the three regulatory thresholds can be compared side by "
        "side for each station. From this we learn that nearly all stations exceed the WHO 2021 guideline "
        "(15 µg/m³) on over 90% of days, while roughly 40% of days exceed China's Class 2 limit "
        "(75 µg/m³). The gap between the WHO bar and the China bars highlights how much stricter the WHO "
        "standard is. Limits shown are 24-hour PM2.5 means: WHO 2021 = 15, China GB 3095-2012 Class 1 = "
        "35, Class 2 = 75 µg/m³."
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
        category_orders={"pm25_category": CATS, "season": SEASONS},
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
        "A **stacked bar chart** is used because it shows both the overall distribution and the share of "
        "each air-quality grade as a proportion of the whole. From this we learn that winter has the "
        "largest share of Heavy and Severe hours, while summer is dominated by Excellent and Good grades. "
        "Grades follow China's HJ 633-2012 PM2.5 breakpoints (35/75/115/150/250 µg/m³), which are defined "
        "on 24-hour averages, so they are applied to a rolling 24 h mean rather than single hours."
    )

# ---------------------------------------------------------------- data quality
with tab_dq:
    st.subheader("Preprocessing summary")
    rep_path = DATA / "missing_report.csv"
    if rep_path.exists():
        rep = pd.read_csv(rep_path)
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
            "A **grouped bar chart** is used to compare missing-value counts before and after cleaning for "
            "each variable. From this we can see that CO had the most missing values (~20,700, about 4.9%), "
            "followed by O3 and NO2, while weather variables had very few gaps. After cleaning, all columns "
            "have zero missing values."
        )
    q = full.groupby("station").agg(
        rows=("station", "size"),
        imputed_rows_pct=("imputed_any", "mean"),
        pm25_gt_pm10_pct=("pm_inconsistent", "mean"),
        pm25_above_500_pct=("extreme_pm25", "mean"),
    )
    q[["imputed_rows_pct", "pm25_gt_pm10_pct", "pm25_above_500_pct"]] *= 100
    st.dataframe(q.round(2), width="stretch")
    st.markdown("""
**Method.** Stations combined into one table (12 x 35,064 hourly rows) and a datetime built from year, month, day, hour.
Gaps of up to 6 hours were linearly interpolated within each station; longer gaps were filled with the median for the same
station, month and hour; wind direction was forward-filled then set to the station-month mode.
Rows with PM2.5 above PM10 or PM2.5 above 500 µg/m³ were **flagged, not removed**: the first can be an instrument artefact,
the second reflects genuine severe haze. Median filling pulls values toward typical levels, so use the sidebar option to
exclude imputed rows as a sensitivity check.

**Assumption.** *Site type* groups Changping, Dingling, Huairou and Shunyi as suburban/rural and the other eight as urban,
based on station location; it is not a field in the original data.
""")
