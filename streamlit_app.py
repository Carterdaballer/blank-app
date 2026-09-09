import math
import json
import statistics
import urllib.parse
import urllib.request

import pandas as pd
import streamlit as st


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Matchup Edge V5.2",
    page_icon="🏈",
    layout="wide",
)

st.title("🏈 Matchup Edge V5.2")

st.caption(
    "Independent college-football matchup model • "
    "Consensus ratings → matchup → fair line → price → EV"
)


# =========================================================
# SETTINGS
# =========================================================

SEASON = 2026
PREVIOUS_SEASON = 2025

HOME_FIELD_ADVANTAGE = 2.5

SPREAD_SIGMA = 13.5
TOTAL_SIGMA = 14.0
ML_LOGISTIC_SCALE = 6.5


# =========================================================
# SIDEBAR — DIAGNOSTICS & OVERRIDES
# =========================================================

with st.sidebar:
    st.header("Model Diagnostics & Overrides")

    show_diagnostics = st.checkbox(
        "Show prior vs. current rating breakdown",
        value=True,
        help=(
            "Shows exactly how much of each team's power rating comes "
            "from the 2025 prior vs. 2026 current-season data, so you "
            "can see where a suspicious number is coming from."
        ),
    )

    override_season_weight = st.checkbox(
        "Manually override current-season weight",
        value=False,
        help=(
            "By default the model automatically decides how much to trust "
            "2026 data vs. 2025 data based on the week number. Turn this on "
            "to force a specific weight and test how sensitive the fair "
            "spread is to that assumption."
        ),
    )

    manual_season_weight = st.slider(
        "Current-season weight (if overridden)",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.05,
        disabled=not override_season_weight,
    )


# =========================================================
# API KEY
# =========================================================

try:
    CFBD_API_KEY = st.secrets["CFBD_API_KEY"]

except Exception:
    st.error(
        "CFBD_API_KEY is missing from Streamlit Secrets."
    )
    st.stop()


# =========================================================
# CFBD REQUEST FUNCTION
# =========================================================

@st.cache_data(ttl=3600, show_spinner=False)
def cfbd_get(path, params, api_key):
    base_url = "https://api.collegefootballdata.com"

    clean_params = {
        key: value
        for key, value in params.items()
        if value is not None
    }

    query = urllib.parse.urlencode(clean_params)
    url = f"{base_url}{path}"

    if query:
        url += f"?{query}"

    request = urllib.request.Request(
        url,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Accept": "application/json",
        },
    )

    with urllib.request.urlopen(request, timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))


# =========================================================
# BASIC HELPERS
# =========================================================

def safe_float(value, default=None):
    try:
        if value is None:
            return default
        value = float(value)
        if math.isnan(value):
            return default
        return value
    except (TypeError, ValueError):
        return default


def safe_int(value, default=None):
    try:
        if value is None:
            return default
        return int(value)
    except (TypeError, ValueError):
        return default


def first_value(dictionary, *keys, default=None):
    for key in keys:
        if key in dictionary:
            value = dictionary.get(key)
            if value is not None:
                return value
    return default


def normal_cdf(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def logistic(x):
    return 1.0 / (1.0 + math.exp(-x))


def average(values, default=0.0):
    if not values:
        return default
    return sum(values) / len(values)


# =========================================================
# ODDS FUNCTIONS
# =========================================================

def american_to_decimal(odds):
    odds = safe_float(odds)

    if odds is None or odds == 0:
        raise ValueError("American odds cannot be 0 or non-numeric.")

    if odds > 0:
        return 1.0 + odds / 100.0

    return 1.0 + 100.0 / abs(odds)


def implied_probability(odds):
    return 1.0 / american_to_decimal(odds)


def expected_value(probability, odds):
    decimal_odds = american_to_decimal(odds)
    return probability * (decimal_odds - 1.0) - (1.0 - probability)


def edge_grade(probability_edge, ev):
    if ev <= 0:
        return "PASS"
    if probability_edge >= 0.08:
        return "A"
    if probability_edge >= 0.04:
        return "B"
    if probability_edge >= 0.015:
        return "C"
    return "PASS"


def suggested_units(probability_edge, ev, confidence):
    if ev <= 0 or probability_edge < 0.015:
        return 0.0

    if probability_edge < 0.04:
        units = 0.5
    elif probability_edge < 0.08:
        units = 1.0
    elif probability_edge < 0.12:
        units = 1.5
    else:
        units = 2.0

    if confidence < 55:
        units = min(units, 0.5)
    elif confidence < 65:
        units = min(units, 1.0)

    return units


# =========================================================
# TEAM NAME NORMALIZATION
# =========================================================

def normalize_team_name(name):
    if name is None:
        return ""
    return (
        str(name).strip().lower()
        .replace("&", "and")
        .replace(".", "")
        .replace("'", "")
    )


def make_team_lookup(rows):
    lookup = {}
    for row in rows:
        team = first_value(row, "team", "school")
        if team:
            lookup[normalize_team_name(team)] = row
    return lookup


def find_team_row(lookup, team):
    key = normalize_team_name(team)
    if key in lookup:
        return lookup[key]
    for stored_key, row in lookup.items():
        if key == stored_key or key in stored_key or stored_key in key:
            return row
    return None


# =========================================================
# FBS TEAM FILTER
# =========================================================

@st.cache_data(ttl=3600, show_spinner=False)
def get_fbs_teams(year, api_key):
    return cfbd_get("/teams/fbs", {"year": year}, api_key)


def build_fbs_team_set(rows):
    teams = set()
    for row in rows:
        school = first_value(row, "school", "team")
        if school:
            teams.add(normalize_team_name(school))
    return teams


def is_fbs_vs_fbs(game, fbs_team_set):
    away = first_value(game, "awayTeam", "away_team")
    home = first_value(game, "homeTeam", "home_team")
    if not away or not home:
        return False
    return (
        normalize_team_name(away) in fbs_team_set
        and normalize_team_name(home) in fbs_team_set
    )


# =========================================================
# RATING VALUE EXTRACTORS
# =========================================================

def sp_rating(row):
    return safe_float(first_value(row, "rating"))


def fpi_rating(row):
    return safe_float(first_value(row, "fpi"))


def elo_rating(row):
    return safe_float(first_value(row, "elo"))


def core_rating(row):
    return safe_float(first_value(row, "overall"))


# =========================================================
# POINT-SCALE RATING CALIBRATION
# =========================================================

def raw_rating_map(rows, value_getter, allowed_teams=None):
    """Build team -> raw rating map. SP+ and FPI keep their native point scales."""
    rating_map = {}

    for row in rows:
        team = first_value(row, "team", "school")
        value = safe_float(value_getter(row))

        if not team or value is None:
            continue

        team_key = normalize_team_name(team)

        if allowed_teams is not None and team_key not in allowed_teams:
            continue

        rating_map[team_key] = value

    return rating_map


def average_anchor_map(sp_map, fpi_map):
    """SP+ and FPI serve as the primary point-rating anchor."""
    teams = set(sp_map.keys()).union(fpi_map.keys())
    anchor = {}

    for team in teams:
        values = []
        if team in sp_map:
            values.append(sp_map[team])
        if team in fpi_map:
            values.append(fpi_map[team])
        if values:
            anchor[team] = sum(values) / len(values)

    return anchor


def rescale_to_anchor_distribution(source_map, anchor_map, min_teams=20):
    """
    Converts Elo or CORE onto the SP+/FPI point-rating scale by matching
    DISTRIBUTIONS (mean/stdev) rather than fitting an OLS regression.

    Why this matters: a regression fit (intercept + slope * x) always
    shrinks extreme values toward the mean whenever the correlation
    between the two sources isn't perfect (regression dilution). That
    means a genuinely elite or genuinely bad team gets its calibrated
    rating pulled toward average every time this transform runs — which
    quietly compresses fair spreads for lopsided matchups. Standardizing
    each source against its OWN spread and re-expressing it on the
    anchor's spread preserves how extreme a team actually is.
    """

    if len(source_map) < min_teams or len(anchor_map) < min_teams:
        return {}

    source_values = list(source_map.values())
    anchor_values = list(anchor_map.values())

    try:
        source_mean = statistics.mean(source_values)
        source_sd = statistics.stdev(source_values)
        anchor_mean = statistics.mean(anchor_values)
        anchor_sd = statistics.stdev(anchor_values)
    except statistics.StatisticsError:
        return {}

    if source_sd == 0 or anchor_sd == 0:
        return {}

    transformed = {}

    for team, value in source_map.items():
        z = (value - source_mean) / source_sd
        transformed[team] = anchor_mean + z * anchor_sd

    return transformed


# =========================================================
# RATING ENDPOINTS
# =========================================================

@st.cache_data(ttl=3600, show_spinner=False)
def get_sp(year, api_key):
    return cfbd_get("/ratings/sp", {"year": year}, api_key)


@st.cache_data(ttl=3600, show_spinner=False)
def get_fpi(year, api_key):
    return cfbd_get("/ratings/fpi", {"year": year}, api_key)


@st.cache_data(ttl=3600, show_spinner=False)
def get_core(year, api_key):
    return cfbd_get("/ratings/core", {"year": year}, api_key)


@st.cache_data(ttl=3600, show_spinner=False)
def get_elo(year, week, api_key):
    return cfbd_get(
        "/ratings/elo",
        {"year": year, "week": week, "seasonType": "regular"},
        api_key,
    )


# =========================================================
# SCHEDULE
# =========================================================

@st.cache_data(ttl=1800, show_spinner=False)
def get_week_games(year, week, api_key):
    return cfbd_get(
        "/games",
        {
            "year": year,
            "week": week,
            "seasonType": "regular",
            "classification": "fbs",
        },
        api_key,
    )


# =========================================================
# PRIOR GAME HISTORY
# =========================================================

@st.cache_data(ttl=3600, show_spinner=False)
def get_prior_games(year, selected_week, api_key):
    all_games = {}

    for game_week in range(0, selected_week):
        try:
            games = cfbd_get(
                "/games",
                {
                    "year": year,
                    "week": game_week,
                    "seasonType": "regular",
                    "classification": "fbs",
                },
                api_key,
            )

            for game in games:
                game_id = first_value(game, "id", "gameId")
                if game_id is not None:
                    all_games[str(game_id)] = game

        except Exception:
            continue

    return list(all_games.values())


# =========================================================
# TEAM GAME METRICS
# =========================================================

def build_team_game_metrics(games):
    metrics = {}

    def ensure_team(team):
        if team not in metrics:
            metrics[team] = {"games": 0, "points_for": [], "points_against": [], "margins": []}

    for game in games:
        home = first_value(game, "homeTeam", "home_team")
        away = first_value(game, "awayTeam", "away_team")
        home_points = safe_float(first_value(game, "homePoints", "home_points"))
        away_points = safe_float(first_value(game, "awayPoints", "away_points"))

        if not home or not away or home_points is None or away_points is None:
            continue

        ensure_team(home)
        ensure_team(away)

        metrics[home]["games"] += 1
        metrics[home]["points_for"].append(home_points)
        metrics[home]["points_against"].append(away_points)
        metrics[home]["margins"].append(home_points - away_points)

        metrics[away]["games"] += 1
        metrics[away]["points_for"].append(away_points)
        metrics[away]["points_against"].append(home_points)
        metrics[away]["margins"].append(away_points - home_points)

    return metrics


def team_game_summary(metrics, team):
    data = metrics.get(team, {"games": 0, "points_for": [], "points_against": [], "margins": []})
    return {
        "games": data["games"],
        "ppg": average(data["points_for"], 0.0),
        "pa": average(data["points_against"], 0.0),
        "margin": average(data["margins"], 0.0),
    }


# =========================================================
# CORE FRESHNESS
# =========================================================

def valid_core_rows(rows, selected_week):
    valid = []
    for row in rows:
        through_week = safe_int(first_value(row, "throughWeek", "through_week"), default=-1)
        if through_week < selected_week:
            valid.append(row)
    return valid


# =========================================================
# NATIONAL RATING MAPS
# =========================================================

def build_rating_maps(sp_rows, fpi_rows, elo_rows, core_rows, fbs_team_set):
    """
    V5.2 rating system:

    SP+  = native point-like rating
    FPI  = native point-like rating

    Elo and CORE are statistically translated onto the shared SP+/FPI
    scale using distribution matching (not OLS regression — see
    rescale_to_anchor_distribution for why).

    Only FBS programs are used.
    """

    sp_map = raw_rating_map(sp_rows, sp_rating, fbs_team_set)
    fpi_map = raw_rating_map(fpi_rows, fpi_rating, fbs_team_set)
    elo_raw = raw_rating_map(elo_rows, elo_rating, fbs_team_set)
    core_raw = raw_rating_map(core_rows, core_rating, fbs_team_set)

    anchor_map = average_anchor_map(sp_map, fpi_map)

    elo_map = rescale_to_anchor_distribution(elo_raw, anchor_map)
    core_map = rescale_to_anchor_distribution(core_raw, anchor_map)

    return {
        "SP+": sp_map,
        "FPI": fpi_map,
        "Elo": elo_map,
        "CORE": core_map,
    }


# =========================================================
# CONSENSUS TEAM STRENGTH
# =========================================================

def consensus_from_maps(team, maps, source_weights):
    key = normalize_team_name(team)
    components = []

    for source, weight in source_weights.items():
        source_map = maps.get(source, {})
        rating = safe_float(source_map.get(key))

        if rating is not None:
            components.append((source, rating, weight))

    # Missing data must never become a fake 0.0 rating.
    if not components:
        return None, []

    total_weight = sum(component[2] for component in components)
    if total_weight <= 0:
        return None, []

    consensus_rating = sum(
        rating * weight for (source, rating, weight) in components
    ) / total_weight

    return consensus_rating, components


# =========================================================
# SEASON BLEND
# =========================================================

def current_season_weight(week):
    """Early season remains prior-heavy; current info gradually gains influence."""
    weight = 0.15 + 0.10 * max(0, week - 1)
    return min(0.85, max(0.15, weight))


def blend_team_strength(prior_rating, current_rating, current_weight):
    if prior_rating is None and current_rating is None:
        return None
    if prior_rating is None:
        return current_rating
    if current_rating is None:
        return prior_rating
    return prior_rating * (1.0 - current_weight) + current_rating * current_weight


# =========================================================
# DISAGREEMENT / CONFIDENCE
# =========================================================

def source_disagreement(away_components, home_components):
    away_dict = {name: value for name, value, weight in away_components}
    home_dict = {name: value for name, value, weight in home_components}

    common_sources = set(away_dict.keys()).intersection(home_dict.keys())
    differences = [home_dict[s] - away_dict[s] for s in common_sources]

    if len(differences) < 2:
        return 6.0

    try:
        return statistics.stdev(differences)
    except statistics.StatisticsError:
        return 6.0


def calculate_confidence(week, games_away, games_home, disagreement, source_count):
    sample_games = min(games_away, games_home)

    confidence = 50.0
    confidence += min(15.0, sample_games * 4.0)
    confidence += min(10.0, source_count * 2.0)
    confidence -= min(15.0, disagreement * 1.5)

    if week <= 2:
        confidence -= 5.0

    return max(35, min(85, round(confidence)))


# =========================================================
# FAIR TOTAL ENGINE
# =========================================================

def stabilized_scoring(observed, games_played, national_average):
    if games_played <= 0:
        return national_average
    sample_weight = min(0.80, games_played * 0.12)
    return observed * sample_weight + national_average * (1.0 - sample_weight)


def calculate_national_scoring_average(metrics):
    scores = []
    for team_data in metrics.values():
        scores.extend(team_data["points_for"])
    if not scores:
        return 27.5
    return average(scores, 27.5)


# =========================================================
# GAME LABEL
# =========================================================

def game_label(game):
    away = first_value(game, "awayTeam", "away_team", default="Away")
    home = first_value(game, "homeTeam", "home_team", default="Home")
    return f"{away} @ {home}"


# =========================================================
# GAME SELECTION — HOME SCREEN
# =========================================================

st.subheader("Game Selection")

week = st.selectbox("Week", list(range(0, 17)), index=2, key="home_week")


# =========================================================
# LOAD FBS TEAMS + SCHEDULE
# =========================================================

try:
    with st.spinner("Loading 2026 schedule..."):
        week_games = get_week_games(SEASON, week, CFBD_API_KEY)
        current_fbs_rows = get_fbs_teams(SEASON, CFBD_API_KEY)
        previous_fbs_rows = get_fbs_teams(PREVIOUS_SEASON, CFBD_API_KEY)

except Exception as error:
    st.error("Could not load the CFBD schedule or FBS team list.")
    st.code(str(error))
    st.stop()


current_fbs_team_set = build_fbs_team_set(current_fbs_rows)
previous_fbs_team_set = build_fbs_team_set(previous_fbs_rows)

week_games = [g for g in week_games if is_fbs_vs_fbs(g, current_fbs_team_set)]

if not week_games:
    st.warning(f"No FBS vs FBS games were returned for Week {week}.")
    st.stop()

selected_game = st.selectbox("Game", week_games, format_func=game_label, key="home_game")


# =========================================================
# SELECTED GAME
# =========================================================

away_team = first_value(selected_game, "awayTeam", "away_team", default="Away")
home_team = first_value(selected_game, "homeTeam", "home_team", default="Home")

if (
    normalize_team_name(away_team) not in current_fbs_team_set
    or normalize_team_name(home_team) not in current_fbs_team_set
):
    st.error("Matchup Edge currently supports FBS vs FBS matchups only.")
    st.stop()

neutral_site = bool(first_value(selected_game, "neutralSite", "neutral_site", default=False))
venue = first_value(selected_game, "venue", default="Unknown venue")
start_date = first_value(selected_game, "startDate", "start_date", default="Unknown kickoff")
home_points = safe_float(first_value(selected_game, "homePoints", "home_points"))
away_points = safe_float(first_value(selected_game, "awayPoints", "away_points"))

st.header(f"{away_team} at {home_team}")

game_col1, game_col2, game_col3 = st.columns(3)
game_col1.metric("Week", week)
game_col2.metric("Venue", venue)
game_col3.metric("Site", "Neutral" if neutral_site else f"{home_team} home")
st.caption(f"Kickoff: {start_date}")

if home_points is not None and away_points is not None:
    st.warning(
        "This game already has a recorded score. Current-season snapshot ratings "
        "may contain information unavailable before kickoff. Use V5.2 primarily "
        "for upcoming games until historical snapshots are fully implemented."
    )


# =========================================================
# LOAD MODEL DATA
# =========================================================

try:
    with st.spinner("Building V5.2 consensus model..."):

        prior_sp = get_sp(PREVIOUS_SEASON, CFBD_API_KEY)
        prior_fpi = get_fpi(PREVIOUS_SEASON, CFBD_API_KEY)
        prior_elo = get_elo(PREVIOUS_SEASON, 16, CFBD_API_KEY)

        try:
            prior_core_all = get_core(PREVIOUS_SEASON, CFBD_API_KEY)
        except Exception:
            prior_core_all = []

        prior_maps = build_rating_maps(
            prior_sp, prior_fpi, prior_elo, prior_core_all, previous_fbs_team_set,
        )

        current_sp = get_sp(SEASON, CFBD_API_KEY)
        current_fpi = get_fpi(SEASON, CFBD_API_KEY)
        elo_week = max(1, week - 1)
        current_elo = get_elo(SEASON, elo_week, CFBD_API_KEY)

        try:
            current_core_all = get_core(SEASON, CFBD_API_KEY)
        except Exception:
            current_core_all = []

        current_core = valid_core_rows(current_core_all, week)

        current_maps = build_rating_maps(
            current_sp, current_fpi, current_elo, current_core, current_fbs_team_set,
        )

        prior_games = get_prior_games(SEASON, week, CFBD_API_KEY)
        prior_games = [g for g in prior_games if is_fbs_vs_fbs(g, current_fbs_team_set)]
        team_metrics = build_team_game_metrics(prior_games)

except Exception as error:
    st.error("V5.2 could not load one or more model data sources.")
    st.code(str(error))
    st.stop()


# =========================================================
# SOURCE WEIGHTS
# =========================================================

PRIOR_SOURCE_WEIGHTS = {"SP+": 0.40, "FPI": 0.35, "Elo": 0.20, "CORE": 0.05}
CURRENT_SOURCE_WEIGHTS = {"SP+": 0.30, "FPI": 0.30, "Elo": 0.25, "CORE": 0.15}


# =========================================================
# TEAM RATINGS
# =========================================================

away_prior_rating, away_prior_components = consensus_from_maps(away_team, prior_maps, PRIOR_SOURCE_WEIGHTS)
home_prior_rating, home_prior_components = consensus_from_maps(home_team, prior_maps, PRIOR_SOURCE_WEIGHTS)
away_current_rating, away_current_components = consensus_from_maps(away_team, current_maps, CURRENT_SOURCE_WEIGHTS)
home_current_rating, home_current_components = consensus_from_maps(home_team, current_maps, CURRENT_SOURCE_WEIGHTS)

if away_prior_rating is None and away_current_rating is None:
    st.error(f"Insufficient rating data for {away_team}. Fair spread unavailable.")
    st.stop()

if home_prior_rating is None and home_current_rating is None:
    st.error(f"Insufficient rating data for {home_team}. Fair spread unavailable.")
    st.stop()

auto_season_weight = current_season_weight(week)
season_weight = manual_season_weight if override_season_weight else auto_season_weight

away_power = blend_team_strength(away_prior_rating, away_current_rating, season_weight)
home_power = blend_team_strength(home_prior_rating, home_current_rating, season_weight)

if away_power is None or home_power is None:
    st.error("Insufficient rating data to calculate this matchup.")
    st.stop()


# =========================================================
# HOME FIELD
# =========================================================

home_field = 0.0 if neutral_site else HOME_FIELD_ADVANTAGE


# =========================================================
# FAIR SPREAD
# =========================================================

model_home_margin = home_power - away_power + home_field

rating_gap = abs(home_power - away_power)

if rating_gap > 45:
    st.warning(
        "Extremely large team-strength differential detected. "
        "Treat this matchup cautiously."
    )

model_home_margin = max(-35.0, min(35.0, model_home_margin))


# =========================================================
# GAME SUMMARIES
# =========================================================

away_summary = team_game_summary(team_metrics, away_team)
home_summary = team_game_summary(team_metrics, home_team)

national_scoring_average = calculate_national_scoring_average(team_metrics)

away_offense = stabilized_scoring(away_summary["ppg"], away_summary["games"], national_scoring_average)
away_defense_allowed = stabilized_scoring(away_summary["pa"], away_summary["games"], national_scoring_average)
home_offense = stabilized_scoring(home_summary["ppg"], home_summary["games"], national_scoring_average)
home_defense_allowed = stabilized_scoring(home_summary["pa"], home_summary["games"], national_scoring_average)


# =========================================================
# FAIR TOTAL
# =========================================================

expected_away_points = (away_offense + home_defense_allowed) / 2.0
expected_home_points = (home_offense + away_defense_allowed) / 2.0

if not neutral_site:
    expected_home_points += 1.25
    expected_away_points -= 1.25

raw_total = expected_home_points + expected_away_points

minimum_games = min(away_summary["games"], home_summary["games"])
total_sample_weight = min(0.75, minimum_games * 0.15)
national_total_baseline = national_scoring_average * 2.0

model_total = raw_total * total_sample_weight + national_total_baseline * (1.0 - total_sample_weight)
model_total = max(30.0, min(85.0, model_total))


# =========================================================
# CONFIDENCE
# =========================================================

current_disagreement = source_disagreement(away_current_components, home_current_components)

common_sources = len(
    set(item[0] for item in away_current_components).intersection(
        item[0] for item in home_current_components
    )
)

confidence = calculate_confidence(
    week, away_summary["games"], home_summary["games"], current_disagreement, common_sources,
)


# =========================================================
# MODEL DASHBOARD
# =========================================================

st.divider()
st.header("🧠 V5.2 Matchup Model")

col1, col2, col3 = st.columns(3)
col1.metric(f"{away_team} Power", f"{away_power:+.1f}")
col2.metric(f"{home_team} Power", f"{home_power:+.1f}")
col3.metric("Model Confidence", f"{confidence}%")

col4, col5, col6 = st.columns(3)
if model_home_margin >= 0:
    col4.metric("Model Fair Spread", f"{home_team} -{abs(model_home_margin):.1f}")
else:
    col4.metric("Model Fair Spread", f"{away_team} -{abs(model_home_margin):.1f}")
col5.metric("Model Fair Total", f"{model_total:.1f}")
col6.metric("Home Field", f"{home_field:+.1f}")


# =========================================================
# DIAGNOSTICS — WHY DID THE MODEL LAND HERE?
# =========================================================

if show_diagnostics:
    st.subheader("🩺 Why did the model land here?")

    weight_source = "manual override" if override_season_weight else "automatic (by week)"
    st.caption(
        f"Current-season weight in effect: **{season_weight:.2f}** ({weight_source}). "
        f"That means each team's blended power is "
        f"{(1 - season_weight) * 100:.0f}% 2025 prior + {season_weight * 100:.0f}% 2026 current."
    )

    diag_rows = [
        {
            "Team": away_team,
            "2025 Prior Rating": round(away_prior_rating, 1) if away_prior_rating is not None else "—",
            "2026 Current Rating": round(away_current_rating, 1) if away_current_rating is not None else "—",
            "Blended Power": round(away_power, 1),
        },
        {
            "Team": home_team,
            "2025 Prior Rating": round(home_prior_rating, 1) if home_prior_rating is not None else "—",
            "2026 Current Rating": round(home_current_rating, 1) if home_current_rating is not None else "—",
            "Blended Power": round(home_power, 1),
        },
    ]

    st.dataframe(pd.DataFrame(diag_rows), use_container_width=True, hide_index=True)

    if away_prior_rating is not None and home_prior_rating is not None and \
       away_current_rating is not None and home_current_rating is not None:

        prior_gap = home_prior_rating - away_prior_rating
        current_gap = home_current_rating - away_current_rating
        gap_difference = abs(current_gap - prior_gap)

        st.caption(
            f"2025 rating gap: **{prior_gap:+.1f}** • "
            f"2026 rating gap: **{current_gap:+.1f}**"
        )

        if gap_difference >= 10:
            bigger_gap_year = "2026" if abs(current_gap) > abs(prior_gap) else "2025"
            st.warning(
                f"The 2025 and 2026 gaps between these teams disagree by "
                f"{gap_difference:.1f} points, and {bigger_gap_year} shows the "
                f"bigger mismatch. The blended power rating is averaging these "
                f"together, which mutes whatever changed between seasons. If you "
                f"believe the {bigger_gap_year} gap is the real one (e.g. a "
                f"coaching change, major roster turnover, or a big jump/decline "
                f"in one program), try the current-season weight override in the "
                f"sidebar and push it toward the season you trust more."
            )


# =========================================================
# RATING CONSENSUS
# =========================================================

st.subheader("📊 Rating Consensus")

rating_rows = []

for source in ["SP+", "FPI", "Elo", "CORE"]:
    away_rating = current_maps.get(source, {}).get(normalize_team_name(away_team))
    home_rating = current_maps.get(source, {}).get(normalize_team_name(home_team))

    rating_rows.append({
        "Source": source,
        away_team: round(away_rating, 1) if away_rating is not None else None,
        home_team: round(home_rating, 1) if home_rating is not None else None,
    })

rating_df = pd.DataFrame(rating_rows)

st.dataframe(rating_df, use_container_width=True, hide_index=True)

st.caption(
    "SP+ and FPI retain their point-like rating scales. Elo and CORE are "
    "distribution-matched onto that same scale (preserving how extreme a "
    "team's rating is) before consensus strength is calculated."
)


# =========================================================
# MODEL INTERPRETATION
# =========================================================

st.subheader("🔍 Model Interpretation")

if current_disagreement <= 2.5:
    disagreement_text = "The rating systems are in relatively strong agreement on this matchup."
elif current_disagreement <= 5.0:
    disagreement_text = "The rating systems show moderate disagreement on this matchup."
else:
    disagreement_text = "The rating systems disagree materially. Treat the fair spread with extra caution."

st.info(disagreement_text)

if week <= 3:
    st.warning(
        "Early-season model: uncertainty remains elevated. V5.2 limits the "
        "influence of small current-season samples — see the diagnostics "
        "panel above if a spread looks off."
    )


# =========================================================
# LINE LAB
# =========================================================

st.divider()
st.header("🧪 Line Lab")

st.write("Enter the exact line and odds offered by your sportsbook. Test one price at a time.")

bet_type = st.selectbox("Bet Type", ["Spread", "Total", "Moneyline"])


if bet_type == "Spread":
    bet_team = st.selectbox("Bet Team", [away_team, home_team])

    input_col1, input_col2 = st.columns(2)
    spread = input_col1.number_input("Spread", value=-2.5, step=0.5)
    odds = input_col2.number_input("American Odds", value=-110, step=5)

    team_expected_margin = model_home_margin if bet_team == home_team else -model_home_margin
    cover_threshold = -spread
    z = (team_expected_margin - cover_threshold) / SPREAD_SIGMA
    model_probability = normal_cdf(z)
    bet_label = f"{bet_team} {spread:+.1f}"

elif bet_type == "Total":
    total_side = st.selectbox("Side", ["Over", "Under"])

    input_col1, input_col2 = st.columns(2)
    sportsbook_total = input_col1.number_input("Sportsbook Total", value=50.5, step=0.5)
    odds = input_col2.number_input("American Odds", value=-110, step=5)

    z = (model_total - sportsbook_total) / TOTAL_SIGMA
    over_probability = normal_cdf(z)
    model_probability = over_probability if total_side == "Over" else 1.0 - over_probability
    bet_label = f"{total_side} {sportsbook_total:.1f}"

else:
    bet_team = st.selectbox("Bet Team", [away_team, home_team])
    odds = st.number_input("American Odds", value=-110, step=5)

    home_win_probability = logistic(model_home_margin / ML_LOGISTIC_SCALE)
    model_probability = home_win_probability if bet_team == home_team else 1.0 - home_win_probability
    bet_label = f"{bet_team} ML"


# =========================================================
# BET EVALUATION
# =========================================================

if odds == 0:
    st.error("American odds cannot be 0 — enter a real sportsbook price.")
    st.stop()

break_even = implied_probability(odds)
probability_edge = model_probability - break_even
ev = expected_value(model_probability, odds)
grade = edge_grade(probability_edge, ev)
units = suggested_units(probability_edge, ev, confidence)

st.subheader("📊 Bet Evaluation")
st.markdown(f"## {bet_label}")

eval_col1, eval_col2 = st.columns(2)
eval_col1.metric("Model Probability", f"{model_probability * 100:.1f}%")
eval_col2.metric("Sportsbook Break-Even", f"{break_even * 100:.1f}%")

eval_col3, eval_col4 = st.columns(2)
eval_col3.metric("Probability Edge", f"{probability_edge * 100:+.1f}%")
eval_col4.metric("Expected Value", f"{ev * 100:+.1f}%")

eval_col5, eval_col6 = st.columns(2)
eval_col5.metric("Grade", grade)
eval_col6.metric("Suggested Units", f"{units:.1f}u")

if grade == "A":
    st.success(
        "A-grade model edge. Verify injuries, QB status, matchup context and "
        "market information before considering a wager."
    )
elif grade == "B":
    st.success("B-grade model edge. Potentially actionable after matchup and availability checks.")
elif grade == "C":
    st.warning("C-grade edge. Small advantage only.")
else:
    st.info("PASS — the price does not currently clear the model's threshold.")


# =========================================================
# ONE ALTERNATE SPREAD TEST
# =========================================================

if bet_type == "Spread":
    st.subheader("🔀 Alternate Spread Test")
    st.caption("Test one alternate spread and price at a time.")

    alt_col1, alt_col2 = st.columns(2)
    alt_spread = alt_col1.number_input(
        "Alt Spread", value=float(spread + 1.0), step=0.5, key="single_alt_spread",
    )
    alt_odds = alt_col2.number_input("Alt Odds", value=-110, step=5, key="single_alt_odds")

    alt_threshold = -alt_spread
    alt_z = (team_expected_margin - alt_threshold) / SPREAD_SIGMA
    alt_probability = normal_cdf(alt_z)
    alt_break_even = implied_probability(alt_odds)
    alt_edge = alt_probability - alt_break_even
    alt_ev = expected_value(alt_probability, alt_odds)
    alt_grade = edge_grade(alt_edge, alt_ev)
    alt_units = suggested_units(alt_edge, alt_ev, confidence)

    st.markdown(f"### {bet_team} {alt_spread:+.1f}")

    alt_eval1, alt_eval2 = st.columns(2)
    alt_eval1.metric("Model Probability", f"{alt_probability * 100:.1f}%")
    alt_eval2.metric("Break-Even", f"{alt_break_even * 100:.1f}%")

    alt_eval3, alt_eval4 = st.columns(2)
    alt_eval3.metric("Probability Edge", f"{alt_edge * 100:+.1f}%")
    alt_eval4.metric("Expected Value", f"{alt_ev * 100:+.1f}%")

    alt_eval5, alt_eval6 = st.columns(2)
    alt_eval5.metric("Grade", alt_grade)
    alt_eval6.metric("Suggested Units", f"{alt_units:.1f}u")


# =========================================================
# DATA QUALITY PANEL
# =========================================================

st.divider()
st.header("🧾 Model Data Quality")

quality_rows = []
for source in ["SP+", "FPI", "Elo", "CORE"]:
    source_map = current_maps.get(source, {})
    away_available = normalize_team_name(away_team) in source_map
    home_available = normalize_team_name(home_team) in source_map

    quality_rows.append({
        "Source": source,
        away_team: "✅" if away_available else "❌",
        home_team: "✅" if home_available else "❌",
    })

st.dataframe(pd.DataFrame(quality_rows), use_container_width=True, hide_index=True)


# =========================================================
# MODEL STATUS
# =========================================================

st.header("🚧 Model Status")

st.markdown(
    """
**V5.2 adds on top of V5.1**

- Fixed Elo/CORE calibration: replaced OLS regression fit (which
  systematically compresses extreme teams toward the mean) with
  distribution-matched rescaling that preserves how good or bad a
  team's rating really is
- Diagnostics panel showing each team's prior vs. current rating and
  the blend weight actually applied — so a surprising number can be
  traced instead of just trusted
- Sidebar override to manually set the current-season weight and
  stress-test how much the 2025 prior is influencing a spread
- Explicit warning when the 2025 and 2026 rating gaps for a matchup
  disagree by 10+ points, since that's exactly the situation where
  blending mutes a real talent-level change
- `odds = 0` no longer crashes the app

**Still to add**

- Historical spread calibration / closing-line backtesting
- Advanced offensive/defensive efficiency, matchup-specific factors
- Injury / availability layer, weather, rest and travel
- Live sportsbook odds and line movement
- Probability calibration
"""
)

st.warning(
    "V5.2 is still an experimental decision-support model. The diagnostics "
    "panel makes it possible to see WHY a spread came out the way it did, "
    "but that visibility is not the same as the model being calibrated — "
    "backtesting against real closing lines is still the only way to know "
    "if these fair spreads are actually accurate."
)

st.caption("Matchup Edge V5.2 • Prior → Consensus → Matchup → Fair Line → Price → EV")
