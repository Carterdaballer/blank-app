import math
import json
import urllib.parse
import urllib.request

import pandas as pd
import streamlit as st


# =========================================================
# PAGE SETUP
# =========================================================

st.set_page_config(
    page_title="Matchup Edge V3",
    page_icon="🏈",
    layout="wide",
)

st.title("🏈 Matchup Edge V3")
st.caption("Live college football schedule + matchup analysis + line testing")

st.warning(
    "For research and entertainment only. "
    "Model probabilities are estimates and do not guarantee profitable bets."
)


# =========================================================
# CFBD API
# =========================================================

try:
    CFBD_API_KEY = st.secrets["CFBD_API_KEY"]
except Exception:
    st.error("CFBD API key is not configured in Streamlit Secrets.")
    st.stop()


@st.cache_data(ttl=3600)
def cfbd_get(path, params, api_key):
    query_string = urllib.parse.urlencode(params)

    url = (
        "https://api.collegefootballdata.com"
        + path
        + "?"
        + query_string
    )

    request = urllib.request.Request(
        url,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Accept": "application/json",
        },
    )

    with urllib.request.urlopen(request, timeout=20) as response:
        return json.loads(response.read().decode("utf-8"))


def get_field(data, *names, default=None):
    for name in names:
        if name in data and data[name] is not None:
            return data[name]

    return default


# =========================================================
# BETTING UTILITY FUNCTIONS
# =========================================================

def american_to_decimal(odds):
    if odds > 0:
        return 1 + (odds / 100)

    return 1 + (100 / abs(odds))


def american_to_implied_prob(odds):
    if odds > 0:
        return 100 / (odds + 100)

    return abs(odds) / (abs(odds) + 100)


def normal_cdf(x):
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def spread_cover_probability(
    model_margin_home,
    bet_team,
    line,
    home_team,
    away_team,
    sigma=13.5,
):
    if bet_team == home_team:
        expected_margin = model_margin_home
    else:
        expected_margin = -model_margin_home

    threshold = -line

    return 1 - normal_cdf(
        (threshold - expected_margin) / sigma
    )


def total_probability(
    model_total,
    direction,
    line,
    sigma=14.0,
):
    probability_under = normal_cdf(
        (line - model_total) / sigma
    )

    if direction == "Under":
        return probability_under

    return 1 - probability_under


def moneyline_probability(
    model_margin_home,
    selected_team,
    home_team,
    away_team,
):
    home_probability = 1 / (
        1 + math.exp(-model_margin_home / 6.5)
    )

    if selected_team == home_team:
        return home_probability

    return 1 - home_probability


def expected_value_per_dollar(model_prob, odds):
    decimal_odds = american_to_decimal(odds)

    profit_if_win = decimal_odds - 1

    return (
        model_prob * profit_if_win
        - (1 - model_prob)
    )


def edge_grade(prob_edge):
    if prob_edge >= 0.08:
        return "A"

    if prob_edge >= 0.04:
        return "B"

    if prob_edge >= 0.015:
        return "C"

    return "PASS"


def suggested_units(prob_edge, ev):
    if prob_edge < 0.015 or ev <= 0:
        return 0.0

    if prob_edge < 0.04:
        return 0.5

    if prob_edge < 0.08:
        return 1.0

    if prob_edge < 0.12:
        return 1.5

    return 2.0


# =========================================================
# LIVE SCHEDULE
# =========================================================

st.divider()

st.header("📅 Live 2026 FBS Schedule")

season = st.selectbox(
    "Season",
    [2026],
)

week = st.selectbox(
    "Week",
    list(range(0, 17)),
    index=2,
)


try:
    games = cfbd_get(
        "/games",
        {
            "year": season,
            "week": week,
            "seasonType": "regular",
            "classification": "fbs",
        },
        CFBD_API_KEY,
    )

except Exception as error:
    st.error(
        "We couldn't load the CFBD schedule. "
        "Double-check the API key and try again."
    )

    st.code(str(error))
    st.stop()


if not games:
    st.info(
        f"No FBS games were returned for Week {week}."
    )

    st.stop()


# =========================================================
# NORMALIZE GAME DATA
# =========================================================

normalized_games = []

for game in games:
    away_team = get_field(
        game,
        "awayTeam",
        "away_team",
        default="Away Team",
    )

    home_team = get_field(
        game,
        "homeTeam",
        "home_team",
        default="Home Team",
    )

    start_date = get_field(
        game,
        "startDate",
        "start_date",
        default="",
    )

    venue = get_field(
        game,
        "venue",
        default="",
    )

    home_points = get_field(
        game,
        "homePoints",
        "home_points",
    )

    away_points = get_field(
        game,
        "awayPoints",
        "away_points",
    )

    game_id = get_field(
        game,
        "id",
        default="",
    )

    normalized_games.append(
        {
            "id": game_id,
            "away_team": away_team,
            "home_team": home_team,
            "start_date": start_date,
            "venue": venue,
            "home_points": home_points,
            "away_points": away_points,
            "raw": game,
        }
    )


game_labels = [
    f"{game['away_team']} @ {game['home_team']}"
    for game in normalized_games
]


selected_label = st.selectbox(
    "Select Game",
    game_labels,
)


selected_index = game_labels.index(selected_label)

selected_game = normalized_games[selected_index]

away_team = selected_game["away_team"]
home_team = selected_game["home_team"]


# =========================================================
# SELECTED GAME
# =========================================================

st.subheader("🏟️ Selected Matchup")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Away Team",
        away_team,
    )

with col2:
    st.metric(
        "Home Team",
        home_team,
    )

with col3:
    st.metric(
        "Week",
        week,
    )


if selected_game["start_date"]:
    st.write(
        f"**Kickoff:** {selected_game['start_date']}"
    )


if selected_game["venue"]:
    st.write(
        f"**Venue:** {selected_game['venue']}"
    )


if (
    selected_game["away_points"] is not None
    and selected_game["home_points"] is not None
):
    st.write(
        "**Score:** "
        f"{away_team} {selected_game['away_points']} — "
        f"{home_team} {selected_game['home_points']}"
    )


# =========================================================
# V3A STATUS
# =========================================================

st.divider()

st.header("🧠 Matchup Model")

st.success(
    "CFBD schedule connection is active. "
    "This matchup was loaded automatically from the 2026 schedule."
)

st.info(
    "V3A connects the schedule. "
    "V3B will calculate automatic team ratings from real results "
    "and statistics before we turn the betting model back on for "
    "every matchup."
)


# =========================================================
# MANUAL LINE LAB
# =========================================================

st.divider()

st.header("🧪 Line Lab")

st.write(
    "The sportsbook line-testing engine will use the automatic "
    "matchup rating once V3B is connected."
)

bet_type = st.selectbox(
    "Bet Type",
    [
        "Spread",
        "Total",
        "Moneyline",
    ],
)

american_odds = st.number_input(
    "American Odds",
    min_value=-10000,
    max_value=10000,
    value=-110,
    step=1,
)

if american_odds == 0:
    st.warning(
        "American odds cannot be 0."
    )
else:
    break_even = american_to_implied_prob(
        american_odds
    )

    st.metric(
        "Sportsbook Break-even Probability",
        f"{break_even * 100:.1f}%",
    )


# =========================================================
# DEVELOPMENT ROADMAP
# =========================================================

st.divider()

st.header("🚀 Matchup Edge Roadmap")

st.markdown(
    """
**V3A — LIVE NOW**
- Automatic 2026 FBS schedule
- Week selector
- Game selector
- Secure CFBD connection

**V3B — NEXT**
- Automatic team statistics
- Game results
- Rolling power ratings
- Offensive and defensive efficiency
- Recent form
- Strength of schedule
- Automatic fair spread
- Automatic fair total

**V3C**
- Rosters
- Quarterback evaluation
- Offensive line / defensive front
- Skill-position strength
- Injury adjustments

**V3D**
- Sportsbook odds
- Alternate spreads and totals
- Line movement
- Price comparison

**V3E**
- Prediction history
- Closing line value
- Bet results
- Model calibration
"""
)


st.divider()

st.caption(
    "Matchup Edge V3A • Live Schedule → "
    "Automatic Matchup Data → Line Testing"
)
