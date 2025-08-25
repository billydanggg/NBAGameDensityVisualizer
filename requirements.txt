import streamlit as st
import pandas as pd
import numpy as np
import math
from zoneinfo import ZoneInfo

st.set_page_config(layout="wide")

@st.cache_data
def load_data():
    # Load
    sched_full = pd.read_csv("schedule.csv", parse_dates=["gamedate"])
    sched_partial = pd.read_csv("schedule_24_partial.csv", parse_dates=["gamedate"])
    locations = pd.read_csv("locations.csv")  # not used yet, but we’ll need it later

    # Combine: historical (2014–2024 season keys) + 2024-25 partial (OKC & DEN 80 games each)
    schedule = pd.concat([sched_full, sched_partial], ignore_index=True)

    # Light cleaning / helpers
    schedule["team"] = schedule["team"].str.upper()
    schedule["opponent"] = schedule["opponent"].str.upper()
    schedule = schedule.sort_values(["season", "team", "gamedate"]).reset_index(drop=True)
    
    # (Optional) venue label for later plotting
    _map = {1: "Home", 0: "Away", True: "Home", False: "Away", "H": "Home", "A": "Away"}
    schedule["venue"] = schedule["home"].map(_map).fillna(schedule["home"].astype(str))

    # -------- Team vs Opponent density labels --------
    sch = schedule.copy()

    # 1) Per-team rest gaps
    sch["prev_gamedate"] = sch.groupby(["team", "season"])["gamedate"].shift()
    sch["rest_gap_days"] = (sch["gamedate"] - sch["prev_gamedate"]).dt.days.fillna(99).astype(int)

    sch["is_b2b"]    = sch["rest_gap_days"].eq(1)
    sch["is_rest1"]  = sch["rest_gap_days"].eq(2)
    sch["is_rest2p"] = sch["rest_gap_days"].ge(3)

    # 2) 4-in-6 via rolling count over 6 calendar days
    rsize = (
        sch.assign(one=1)
        .set_index("gamedate")
        .groupby(["team", "season"])
        .rolling("6D")["one"]
        .sum()
        .reset_index(level=[0, 1], drop=True)
    )
    sch["in_4in6"] = rsize.ge(4).to_numpy()

    # 3) Priority label for each team/season/date
    density = pd.Series("2+ Days Rest", index=sch.index)
    density = density.mask(sch["is_rest1"], "1 Day Rest")
    density = density.mask(sch["is_b2b"], "B2B")
    density = density.mask(~sch["is_b2b"] & sch["in_4in6"], "4-in-6 Stretch Game")
    density = density.mask(sch["is_b2b"] & sch["in_4in6"], "B2B 4-in-6 Stretch Game")
    sch["Density_Team"] = density

    # 4) Build lookup (season, date, team) -> density
    keys = list(zip(sch["season"], sch["gamedate"].dt.to_pydatetime(), sch["team"]))
    dens_map = dict(zip(keys, sch["Density_Team"]))

    # 5) Attach Team_Density (row's team) and Opponent_Density (row's opponent)
    row_keys_team = list(zip(schedule["season"], schedule["gamedate"].dt.to_pydatetime(), schedule["team"]))
    row_keys_opp  = list(zip(schedule["season"], schedule["gamedate"].dt.to_pydatetime(), schedule["opponent"]))

    schedule["Team_Density"]     = [dens_map.get(k) for k in row_keys_team]
    schedule["Opponent_Density"] = [dens_map.get(k) for k in row_keys_opp]

    # Optional: make partial seasons explicit (e.g., 2024–25 where only OKC/DEN loaded)
    fallback = "Unavailable (opponent not in loaded season data)"
    schedule["Opponent_Density"] = schedule["Opponent_Density"].fillna(fallback)

    # ---- Normalize locations table ----
    loc = locations.copy()
    loc.columns = [c.strip().lower() for c in loc.columns]

    def pick(cols, *cands):
        for c in cands:
            if c in cols: 
                return c
        raise KeyError(f"Missing expected column among {cands}")

    key_col = pick(loc.columns, "team", "abbreviation", "abbr")
    lat_col = pick(loc.columns, "lat", "latitude")
    lon_col = pick(loc.columns, "lon", "lng", "longitude", "long")
    tz_col  = pick(loc.columns, "timezone", "tz")

    loc = loc.rename(columns={key_col: "team", lat_col: "lat", lon_col: "lon", tz_col: "timezone"})
    loc["team"] = loc["team"].str.upper()

    # Quick dicts
    lat_map = dict(zip(loc["team"], loc["lat"]))
    lon_map = dict(zip(loc["team"], loc["lon"]))


    # ---- Determine current and previous game locations (team perspective) ----
    sch = schedule.copy()

    is_home_now = sch["venue"].astype(str).str.title().eq("Home")
    sch["curr_place_team"] = np.where(is_home_now, sch["team"], sch["opponent"])

    prev_home = sch.groupby(["team", "season"])["home"].shift()
    prev_opp  = sch.groupby(["team", "season"])["opponent"].shift()

    prev_place_team = np.where(
        (prev_home.astype(float) == 1.0) if prev_home.notna().any() else False,
        sch["team"],   # previous game was home
        prev_opp       # previous game was away
    )
    sch["prev_place_team"] = prev_place_team

    # ---- Travel distance in miles ----
    R_MILES = 3958.7613
    def haversine_miles(lat1, lon1, lat2, lon2):
        if any(pd.isna([lat1, lon1, lat2, lon2])):
            return np.nan
        phi1, phi2 = math.radians(lat1), math.radians(lat2)
        dphi = math.radians(lat2 - lat1)
        dlambda = math.radians(lon2 - lon1)
        a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2)**2
        return 2 * R_MILES * math.asin(math.sqrt(a))

    sch["prev_lat"] = sch["prev_place_team"].map(lat_map)
    sch["prev_lon"] = sch["prev_place_team"].map(lon_map)
    sch["curr_lat"] = sch["curr_place_team"].map(lat_map)
    sch["curr_lon"] = sch["curr_place_team"].map(lon_map)

    sch["Travel_Miles"] = [
        0.0 if pd.isna(pl) or pd.isna(cl) 
        else round(haversine_miles(plat, plon, clat, clon), 1)
        for pl, cl, plat, plon, clat, clon in zip(
            sch["prev_place_team"], sch["curr_place_team"],
            sch["prev_lat"], sch["prev_lon"], sch["curr_lat"], sch["curr_lon"]
        )
    ]


    # Finalize
    schedule = sch.drop(columns=[
        "curr_place_team","prev_place_team","prev_lat","prev_lon",
        "curr_lat","curr_lon"
    ], errors="ignore")

    # Cleanup helpers (keep rest_gap_days if you want it)
    schedule = schedule.drop(columns=["prev_gamedate", "one"], errors="ignore")



    return schedule, locations

def season_label(s):
    """Turn 2023 -> '2023–24' etc. (purely cosmetic for the selector)."""
    return f"{s}–{str(s+1)[-2:]}"

# ---- App UI ----
st.title("NBA Game Density Visualizer 🏀")
st.write("Visualize NBA team schedules by season and team. Pick a season and team to get started.")

schedule, locations = load_data()

# Season selector (show most recent first)
season_options = sorted(schedule["season"].unique(), reverse=True)
season_display = {season_label(s): s for s in season_options}
season_choice = st.selectbox("Season", list(season_display.keys()))
season_selected = season_display[season_choice]

# Team selector (filtered by chosen season)
teams_this_season = sorted(schedule.loc[schedule["season"] == season_selected, "team"].unique())
# Try to preselect OKC if available (handy for the project)
default_team_index = teams_this_season.index("OKC") if "OKC" in teams_this_season else 0
team_selected = st.selectbox("Team", teams_this_season, index=default_team_index)

# Filtered frame for downstream visuals
view = (
    schedule.loc[(schedule["season"] == season_selected) & (schedule["team"] == team_selected)]
            .sort_values("gamedate")
            .reset_index(drop=True)
)

# Quick sanity preview (you can remove later)
st.caption("Schedule Overview")
st.dataframe(
    view[["season", "gamedate", "team", "opponent", "venue", "Team_Density", "Opponent_Density","Travel_Miles" ]],
    use_container_width=True
)

# ---- 14-day rolling game density (selected team/season) ----
import altair as alt

# Rolling count of games in the previous 14 calendar days (including game day)
view2 = view.copy()
view2["games_last_14d"] = (
    view2.assign(one=1)
         .rolling("14D", on="gamedate")["one"]
         .sum()
         .astype(int)
)

st.subheader("14-Day Rolling Game Density")
st.caption("Each point is a game. Hover to see opponent and exact value. Points colored by Home/Away.")

line = alt.Chart(view2).mark_line().encode(
    x=alt.X("gamedate:T", title="Date"),
    y=alt.Y("games_last_14d:Q", title="Games in prior 14 days"),
)

points = alt.Chart(view2).mark_circle(size=60).encode(
    x="gamedate:T",
    y="games_last_14d:Q",
    color=alt.Color("venue:N", title="Venue"),  # <--- color by Home/Away
    tooltip=[
        alt.Tooltip("gamedate:T", title="Game date"),
        alt.Tooltip("opponent:N", title="Opponent"),
        alt.Tooltip("venue:N", title="Venue"),
        alt.Tooltip("games_last_14d:Q", title="Games (last 14d)")
    ]
)

chart = (line + points).properties(height=320).interactive()

st.altair_chart(chart, use_container_width=True)
