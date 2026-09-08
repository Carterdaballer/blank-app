import math
import json
import statistics
import urllib.parse
import urllib.request
from datetime import datetime

import pandas as pd
import streamlit as st


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Matchup Edge V6",
    page_icon="🏈",
    layout="wide",
)

st.title("🏈 Matchup Edge V6")
st.caption(
    "Independent college-football matchup model • "
    "Consensus ratings → matchup → fair line → market line → EV"
)


# =========================================================
# SETTINGS
# =========================================================

with st.sidebar:
    st.header("Settings")

    SEASON = st.number_input(
        "Season",
        min_value=2015,
        max_value=2035,
        value=2026,
        step=1,
        help="Change this each year instead of editing code.",
    )

    PREVIOUS_SEASON = SEASON - 1

    HOME_FIELD_ADVANTAGE = st.slider(
        "Home field advantage (pts)",
        0.0, 5.0, 2.5, 0.25,
    )

    DEFAULT_JUICE = st.number_input(
        "Default spread/total odds (used in Slate Scanner)",
        value=-110,
        step=5,
    )

SPREAD_SIGMA = 13.5
TOTAL_SIGMA = 14.0
ML_LOGISTIC_SCALE = 6.5

# Approximate number of spread-points represented by
# one national standard deviation of team strength.
RATING_SD_POINTS = 7.5


# =========================================================
# API KEY
# =========================================================

try:
    CFBD_API_KEY = st.secrets["CFBD_API_KEY"]
except Exception:
    st.error("CFBD_API_KEY is missing from Streamlit Secrets.")
    st.stop()


# =========================================================
# CFBD REQUEST FUNCTION
# =========================================================

@st.cache_data(ttl=3600, show_spinner=False)
def cfbd_get(path, params, api_key):
    base_url = "https://api.collegefootballdata.com"

    clean_params = {k: v for k, v in params.items() if v is not None}
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
        # 0 is not a legal American price; caller must guard before use.
        raise ValueError("American odds cannot be 0 or non-numeric.")

    if odds > 0:
        return 1.0 + odds / 100.0

    return 1.0 + 100.0 / abs(odds)


def implied_probability(odds):
    return 1.0 / american_to_decimal(odds)


def expected_value(probability, odds):
    decimal_odds = american_to_decimal(odds)
    return probability * (decimal_odds - 1.0) - (1.0 - probability)


def kelly_fraction(probability, odds, fraction_cap=0.25):
    """Quarter-Kelly by default, capped, floored at 0."""
    decimal_odds = american_to_decimal(odds)
    b = decimal_odds - 1.0
    if b <= 0:
        return 0.0
    q = 1.0 - probability
    full_kelly = (b * probability - q) / b
    if full_kelly <= 0:
        return 0.0
    return round(full_kelly * fraction_cap, 4)


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
# STANDARDIZATION
# =========================================================

def zscore_map(rows, value_getter):
    values = []
    team_values = {}

    for row in rows:
        team = first_value(row, "team", "school")
        value = safe_float(value_getter(row))
        if team and value is not None:
            team_values[normalize_team_name(team)] = value
            values.append(value)

    if len(values) < 5:
        return {}

    mean_value = statistics.mean(values)

    try:
        sd_value = statistics.stdev(values)
    except statistics.StatisticsError:
        return {}

    if sd_value == 0:
        return {}

    return {
        team: (value - mean_value) / sd_value
        for team, value in team_values.items()
    }


def z_to_points(z):
    if z is None:
        return None
    return z * RATING_SD_POINTS


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
# SCHEDULE + LINES
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


@st.cache_data(ttl=1800, show_spinner=False)
def get_week_lines(year, week, api_key):
    """Sportsbook lines for every game in a week (all providers)."""
    try:
        return cfbd_get(
            "/lines",
            {"year": year, "week": week, "seasonType": "regular"},
            api_key,
        )
    except Exception:
        return []


def build_lines_lookup(lines_rows):
    """
    Maps normalized 'away|home' -> consensus market numbers.
    Spread convention: negative = home favored (matches CFBD /lines).
    """
    lookup = {}

    for game in lines_rows:
        away = first_value(game, "awayTeam", "away_team")
        home = first_value(game, "homeTeam", "home_team")
        if not away or not home:
            continue

        spreads, totals, home_mls, away_mls = [], [], [], []

        for book in game.get("lines", []) or []:
            s = safe_float(book.get("spread"))
            t = safe_float(book.get("overUnder"))
            hml = safe_float(book.get("homeMoneyline"))
            aml = safe_float(book.get("awayMoneyline"))
            if s is not None:
                spreads.append(s)
            if t is not None:
                totals.append(t)
            if hml is not None:
                home_mls.append(hml)
            if aml is not None:
                away_mls.append(aml)

        key = f"{normalize_team_name(away)}|{normalize_team_name(home)}"
        lookup[key] = {
            "spread": round(average(spreads, default=None), 1) if spreads else None,
            "total": round(average(totals, default=None), 1) if totals else None,
            "home_ml": round(average(home_mls, default=None)) if home_mls else None,
            "away_ml": round(average(away_mls, default=None)) if away_mls else None,
            "book_count": len(game.get("lines", []) or []),
        }

    return lookup


def lines_for_game(lookup, away_team, home_team):
    key = f"{normalize_team_name(away_team)}|{normalize_team_name(home_team)}"
    return lookup.get(key)


# =========================================================
# PRIOR GAME HISTORY
# =========================================================

@st.cache_data(ttl=3600, show_spinner=False)
def get_prior_games(year, selected_week, api_key):
    all_games = {}
    failed_weeks = []

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
            failed_weeks.append(game_week)

    return list(all_games.values()), failed_weeks


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

def build_rating_maps(sp_rows, fpi_rows, elo_rows, core_rows):
    return {
        "SP+": zscore_map(sp_rows, sp_rating),
        "FPI": zscore_map(fpi_rows, fpi_rating),
        "Elo": zscore_map(elo_rows, elo_rating),
        "CORE": zscore_map(core_rows, core_rating),
    }


# =========================================================
# CONSENSUS TEAM STRENGTH
# =========================================================

def consensus_from_maps(team, maps, source_weights):
    key = normalize_team_name(team)
    components = []

    for source, weight in source_weights.items():
        z = maps.get(source, {}).get(key)
        if z is not None:
            components.append((source, z_to_points(z), weight))

    # Never silently interpret missing rating info as 0.
    if not components:
        return None, []

    total_weight = sum(item[2] for item in components)
    if total_weight <= 0:
        return None, []

    rating = sum(item[1] * item[2] for item in components) / total_weight
    return rating, components


# =========================================================
# SEASON BLEND
# =========================================================

def current_season_weight(week):
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

    differences = [
        home_dict[s] - away_dict[s]
        for s in set(away_dict).intersection(home_dict)
    ]

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
# SOURCE WEIGHTS
# =========================================================

PRIOR_SOURCE_WEIGHTS = {"SP+": 0.40, "FPI": 0.35, "Elo": 0.20, "CORE": 0.05}
CURRENT_SOURCE_WEIGHTS = {"SP+": 0.30, "FPI": 0.30, "Elo": 0.25, "CORE": 0.15}


# =========================================================
# CORE MATCHUP EVALUATION (shared by single-game view + slate scanner)
# =========================================================

def evaluate_matchup(
    away_team, home_team, neutral_site,
    prior_maps, current_maps, team_metrics,
    national_scoring_average, season_weight, week,
):
    away_prior_rating, away_prior_components = consensus_from_maps(away_team, prior_maps, PRIOR_SOURCE_WEIGHTS)
    home_prior_rating, home_prior_components = consensus_from_maps(home_team, prior_maps, PRIOR_SOURCE_WEIGHTS)
    away_current_rating, away_current_components = consensus_from_maps(away_team, current_maps, CURRENT_SOURCE_WEIGHTS)
    home_current_rating, home_current_components = consensus_from_maps(home_team, current_maps, CURRENT_SOURCE_WEIGHTS)

    if away_prior_rating is None and away_current_rating is None:
        return {"error": f"Insufficient rating data for {away_team}."}
    if home_prior_rating is None and home_current_rating is None:
        return {"error": f"Insufficient rating data for {home_team}."}

    away_power = blend_team_strength(away_prior_rating, away_current_rating, season_weight)
    home_power = blend_team_strength(home_prior_rating, home_current_rating, season_weight)

    if away_power is None or home_power is None:
        return {"error": "Insufficient rating data to calculate this matchup."}

    home_field = 0.0 if neutral_site else HOME_FIELD_ADVANTAGE
    model_home_margin = max(-35.0, min(35.0, home_power - away_power + home_field))

    away_summary = team_game_summary(team_metrics, away_team)
    home_summary = team_game_summary(team_metrics, home_team)

    away_offense = stabilized_scoring(away_summary["ppg"], away_summary["games"], national_scoring_average)
    away_defense_allowed = stabilized_scoring(away_summary["pa"], away_summary["games"], national_scoring_average)
    home_offense = stabilized_scoring(home_summary["ppg"], home_summary["games"], national_scoring_average)
    home_defense_allowed = stabilized_scoring(home_summary["pa"], home_summary["games"], national_scoring_average)

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

    disagreement = source_disagreement(away_current_components, home_current_components)
    common_sources = len(
        set(i[0] for i in away_current_components).intersection(i[0] for i in home_current_components)
    )
    confidence = calculate_confidence(week, away_summary["games"], home_summary["games"], disagreement, common_sources)

    home_win_probability = logistic(model_home_margin / ML_LOGISTIC_SCALE)

    return {
        "error": None,
        "away_power": away_power,
        "home_power": home_power,
        "model_home_margin": model_home_margin,
        "model_total": model_total,
        "confidence": confidence,
        "disagreement": disagreement,
        "home_win_probability": home_win_probability,
        "away_current_components": away_current_components,
        "home_current_components": home_current_components,
        "away_games_played": away_summary["games"],
        "home_games_played": home_summary["games"],
    }


# =========================================================
# GAME SELECTION — HOME SCREEN
# =========================================================

st.subheader("Game Selection")

week = st.selectbox("Week", list(range(0, 17)), index=2, key="home_week")


# =========================================================
# LOAD SCHEDULE + FBS TEAMS + LINES
# =========================================================

try:
    with st.spinner("Loading schedule, teams, and market lines..."):
        week_games = get_week_games(SEASON, week, CFBD_API_KEY)
        fbs_team_rows = get_fbs_teams(SEASON, CFBD_API_KEY)
        week_lines_raw = get_week_lines(SEASON, week, CFBD_API_KEY)
except Exception as error:
    st.error("Could not load the CFBD schedule.")
    st.code(str(error))
    st.stop()

fbs_team_set = build_fbs_team_set(fbs_team_rows)
lines_lookup = build_lines_lookup(week_lines_raw)

week_games = [g for g in week_games if is_fbs_vs_fbs(g, fbs_team_set)]

if not week_games:
    st.warning(f"No FBS vs FBS games were returned for Week {week}.")
    st.stop()

selected_game = st.selectbox("Game", week_games, format_func=game_label, key="home_game")


# =========================================================
# SELECTED GAME
# =========================================================

away_team = first_value(selected_game, "awayTeam", "away_team", default="Away")
home_team = first_value(selected_game, "homeTeam", "home_team", default="Home")

if normalize_team_name(away_team) not in fbs_team_set or normalize_team_name(home_team) not in fbs_team_set:
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
        "may contain information unavailable before kickoff. Use V6 primarily for "
        "upcoming games until historical snapshots are fully implemented."
    )


# =========================================================
# LOAD MODEL DATA
# =========================================================

try:
    with st.spinner("Building V6 consensus model..."):
        prior_sp = get_sp(PREVIOUS_SEASON, CFBD_API_KEY)
        prior_fpi = get_fpi(PREVIOUS_SEASON, CFBD_API_KEY)
        prior_elo = get_elo(PREVIOUS_SEASON, 16, CFBD_API_KEY)

        try:
            prior_core_all = get_core(PREVIOUS_SEASON, CFBD_API_KEY)
        except Exception:
            prior_core_all = []

        prior_maps = build_rating_maps(prior_sp, prior_fpi, prior_elo, prior_core_all)

        current_sp = get_sp(SEASON, CFBD_API_KEY)
        current_fpi = get_fpi(SEASON, CFBD_API_KEY)
        elo_week = max(1, week - 1)
        current_elo = get_elo(SEASON, elo_week, CFBD_API_KEY)
        current_core_all = get_core(SEASON, CFBD_API_KEY)
        current_core = valid_core_rows(current_core_all, week)

        current_maps = build_rating_maps(current_sp, current_fpi, current_elo, current_core)

        prior_games, failed_weeks = get_prior_games(SEASON, week, CFBD_API_KEY)
        prior_games = [g for g in prior_games if is_fbs_vs_fbs(g, fbs_team_set)]
        team_metrics = build_team_game_metrics(prior_games)

except Exception as error:
    st.error("V6 could not load one or more model data sources.")
    st.code(str(error))
    st.stop()

if failed_weeks:
    st.caption(
        f"⚠️ Could not load prior-game data for week(s) {failed_weeks} — "
        "season-to-date scoring stats may be incomplete."
    )

season_weight = current_season_weight(week)
national_scoring_average = calculate_national_scoring_average(team_metrics)


# =========================================================
# EVALUATE SELECTED GAME
# =========================================================

result = evaluate_matchup(
    away_team, home_team, neutral_site,
    prior_maps, current_maps, team_metrics,
    national_scoring_average, season_weight, week,
)

if result["error"]:
    st.error(result["error"])
    st.stop()

away_power = result["away_power"]
home_power = result["home_power"]
model_home_margin = result["model_home_margin"]
model_total = result["model_total"]
confidence = result["confidence"]
current_disagreement = result["disagreement"]
away_current_components = result["away_current_components"]
home_current_components = result["home_current_components"]


# =========================================================
# MARKET LINE COMPARISON
# =========================================================

market = lines_for_game(lines_lookup, away_team, home_team)

st.divider()
st.header("🧠 V6 Matchup Model")

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
col6.metric("Home Field", f"{0.0 if neutral_site else HOME_FIELD_ADVANTAGE:+.1f}")

st.subheader("📈 Model vs. Market")

if market and (market["spread"] is not None or market["total"] is not None):
    m1, m2, m3 = st.columns(3)

    if market["spread"] is not None:
        market_home_margin = -market["spread"]  # spread is home-team spread; negative = favored
        spread_edge = model_home_margin - market_home_margin
        m1.metric(
            "Market Spread (home)",
            f"{market['spread']:+.1f}",
            delta=f"{spread_edge:+.1f} pts model vs market",
        )
    else:
        m1.metric("Market Spread", "No line found")

    if market["total"] is not None:
        total_edge = model_total - market["total"]
        m2.metric(
            "Market Total",
            f"{market['total']:.1f}",
            delta=f"{total_edge:+.1f} pts model vs market",
        )
    else:
        m2.metric("Market Total", "No line found")

    m3.metric("Books Averaged", market["book_count"])
    st.caption(
        "Market numbers are averaged across every provider CFBD returned for this game. "
        "Use them as a sanity check and as prefilled defaults in Line Lab below — always "
        "confirm the live number at your actual sportsbook before betting."
    )
else:
    st.info("No market lines were found for this game yet (common far in advance of kickoff).")


# =========================================================
# RATING BREAKDOWN
# =========================================================

st.subheader("📊 Rating Consensus")

rating_rows = []
for source in ["SP+", "FPI", "Elo", "CORE"]:
    away_z = current_maps.get(source, {}).get(normalize_team_name(away_team))
    home_z = current_maps.get(source, {}).get(normalize_team_name(home_team))
    rating_rows.append({
        "Source": source,
        away_team: round(z_to_points(away_z), 1) if away_z is not None else None,
        home_team: round(z_to_points(home_z), 1) if home_z is not None else None,
    })

st.dataframe(pd.DataFrame(rating_rows), use_container_width=True, hide_index=True)
st.caption(
    "Displayed source ratings are normalized onto the same internal point scale "
    "before being blended. They are not the raw values published by each rating system."
)


# =========================================================
# MODEL INTERPRETATION
# =========================================================

st.subheader("🔍 Model Interpretation")

if current_disagreement <= 2.5:
    st.info("The rating systems are in relatively strong agreement on this matchup.")
elif current_disagreement <= 5.0:
    st.info("The rating systems show moderate disagreement on this matchup.")
else:
    st.warning("The rating systems disagree materially. Treat the fair spread with extra caution.")

if week <= 3:
    st.warning(
        "Early-season model: uncertainty remains elevated. V6 deliberately limits "
        "the influence of tiny current-season samples."
    )


# =========================================================
# LINE LAB
# =========================================================

st.divider()
st.header("🧪 Line Lab")
st.write("Prefilled from market data when available — always confirm the live number at your book.")

bet_type = st.selectbox("Bet Type", ["Spread", "Total", "Moneyline"])

model_probability = None
bet_label = None
odds = DEFAULT_JUICE

if bet_type == "Spread":
    bet_team = st.selectbox("Bet Team", [away_team, home_team])

    default_spread = -2.5
    if market and market["spread"] is not None:
        default_spread = market["spread"] if bet_team == home_team else -market["spread"]

    input_col1, input_col2 = st.columns(2)
    spread = input_col1.number_input("Spread", value=float(default_spread), step=0.5)
    odds = input_col2.number_input("American Odds", value=int(DEFAULT_JUICE), step=5)

    team_expected_margin = model_home_margin if bet_team == home_team else -model_home_margin
    cover_threshold = -spread
    z = (team_expected_margin - cover_threshold) / SPREAD_SIGMA
    model_probability = normal_cdf(z)
    bet_label = f"{bet_team} {spread:+.1f}"

elif bet_type == "Total":
    total_side = st.selectbox("Side", ["Over", "Under"])

    default_total = market["total"] if (market and market["total"] is not None) else 50.5

    input_col1, input_col2 = st.columns(2)
    sportsbook_total = input_col1.number_input("Sportsbook Total", value=float(default_total), step=0.5)
    odds = input_col2.number_input("American Odds", value=int(DEFAULT_JUICE), step=5)

    z = (model_total - sportsbook_total) / TOTAL_SIGMA
    over_probability = normal_cdf(z)
    model_probability = over_probability if total_side == "Over" else 1.0 - over_probability
    bet_label = f"{total_side} {sportsbook_total:.1f}"

else:
    bet_team = st.selectbox("Bet Team", [away_team, home_team])

    default_ml = -110
    if market:
        default_ml = (market["home_ml"] if bet_team == home_team else market["away_ml"]) or -110

    odds = st.number_input("American Odds", value=int(default_ml), step=5)

    home_win_probability = result["home_win_probability"]
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
kelly = kelly_fraction(model_probability, odds)

st.subheader("📊 Bet Evaluation")
st.markdown(f"## {bet_label}")

eval_col1, eval_col2 = st.columns(2)
eval_col1.metric("Model Probability", f"{model_probability * 100:.1f}%")
eval_col2.metric("Sportsbook Break-Even", f"{break_even * 100:.1f}%")

eval_col3, eval_col4 = st.columns(2)
eval_col3.metric("Probability Edge", f"{probability_edge * 100:+.1f}%")
eval_col4.metric("Expected Value", f"{ev * 100:+.1f}%")

eval_col5, eval_col6, eval_col7 = st.columns(3)
eval_col5.metric("Grade", grade)
eval_col6.metric("Suggested Units", f"{units:.1f}u")
eval_col7.metric("Quarter-Kelly Stake", f"{kelly * 100:.1f}% of bankroll")

if grade == "A":
    st.success("A-grade model edge. Verify injuries, QB status, matchup context and market information before considering a wager.")
elif grade == "B":
    st.success("B-grade model edge. Potentially actionable after matchup and availability checks.")
elif grade == "C":
    st.warning("C-grade edge. Small advantage only.")
else:
    st.info("PASS — the price does not currently clear the model's threshold.")


# =========================================================
# BET LOG (session-only)
# =========================================================

if "bet_log" not in st.session_state:
    st.session_state.bet_log = []

log_col1, log_col2 = st.columns([1, 3])

if log_col1.button("📝 Log this bet"):
    st.session_state.bet_log.append({
        "logged_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "week": week,
        "matchup": f"{away_team} @ {home_team}",
        "bet": bet_label,
        "odds": odds,
        "model_prob": round(model_probability * 100, 1),
        "break_even": round(break_even * 100, 1),
        "edge_pct": round(probability_edge * 100, 1),
        "ev_pct": round(ev * 100, 1),
        "grade": grade,
        "units": units,
        "confidence": confidence,
    })
    st.success("Logged.")

if st.session_state.bet_log:
    st.subheader("🗒️ Session Bet Log")
    log_df = pd.DataFrame(st.session_state.bet_log)
    st.dataframe(log_df, use_container_width=True, hide_index=True)
    st.download_button(
        "Download bet log as CSV",
        data=log_df.to_csv(index=False).encode("utf-8"),
        file_name=f"matchup_edge_bet_log_week{week}.csv",
        mime="text/csv",
    )


# =========================================================
# SLATE SCANNER — every game in the week at once
# =========================================================

st.divider()
st.header("📅 Slate Scanner")
st.caption(
    "Runs the model against the market spread for every FBS game this week, "
    f"assuming {DEFAULT_JUICE} odds on both sides. Ranked by edge so you can "
    "scan a full slate without opening every matchup individually."
)

if st.button("Scan full week"):
    scanner_rows = []

    with st.spinner("Evaluating every game this week..."):
        for game in week_games:
            g_away = first_value(game, "awayTeam", "away_team", default="Away")
            g_home = first_value(game, "homeTeam", "home_team", default="Home")
            g_neutral = bool(first_value(game, "neutralSite", "neutral_site", default=False))

            g_result = evaluate_matchup(
                g_away, g_home, g_neutral,
                prior_maps, current_maps, team_metrics,
                national_scoring_average, season_weight, week,
            )

            if g_result["error"]:
                continue

            g_market = lines_for_game(lines_lookup, g_away, g_home)
            if not g_market or g_market["spread"] is None:
                continue

            market_home_margin = -g_market["spread"]
            model_margin = g_result["model_home_margin"]

            # Evaluate the home side at the market number.
            z = (model_margin - market_home_margin) / SPREAD_SIGMA
            home_cover_prob = normal_cdf(z)
            away_cover_prob = 1.0 - home_cover_prob

            home_be = implied_probability(DEFAULT_JUICE)
            away_be = implied_probability(DEFAULT_JUICE)

            home_edge = home_cover_prob - home_be
            away_edge = away_cover_prob - away_be

            if home_edge >= away_edge:
                pick_team, pick_prob, pick_edge = g_home, home_cover_prob, home_edge
                pick_spread = g_market["spread"]
            else:
                pick_team, pick_prob, pick_edge = g_away, away_cover_prob, away_edge
                pick_spread = -g_market["spread"]

            pick_ev = expected_value(pick_prob, DEFAULT_JUICE)
            pick_grade = edge_grade(pick_edge, pick_ev)

            scanner_rows.append({
                "Matchup": f"{g_away} @ {g_home}",
                "Best Side": f"{pick_team} {pick_spread:+.1f}",
                "Model Prob": round(pick_prob * 100, 1),
                "Edge %": round(pick_edge * 100, 1),
                "EV %": round(pick_ev * 100, 1),
                "Grade": pick_grade,
                "Confidence": g_result["confidence"],
            })

    if scanner_rows:
        scanner_df = pd.DataFrame(scanner_rows).sort_values("Edge %", ascending=False)
        st.dataframe(scanner_df, use_container_width=True, hide_index=True)
    else:
        st.info("No games this week had both a resolvable model rating and a market spread.")


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
**V6 adds on top of V5**

- Season is a sidebar setting instead of hardcoded — works next year without a code edit
- Live sportsbook lines pulled from CFBD `/lines`, averaged across books
- Model vs. Market comparison panel on every game
- Line Lab auto-prefills from real market numbers
- Full-week Slate Scanner ranks every game by edge in one click
- Quarter-Kelly stake shown alongside flat-unit sizing
- Session bet log with CSV export
- `odds = 0` no longer crashes the app
- Prior-game fetch failures are now surfaced instead of silently swallowed

**Still to add**

- Live line movement / closing-line tracking (CLV)
- Multi-book (not just averaged) line shopping
- Injury / availability layer
- Weather, rest, and travel
- Historical backtesting and probability calibration
- Persistent (cross-session) bet history, not just session-local
"""
)

st.warning(
    "V6 is still an experimental decision-support model, not a proven betting system. "
    "Grades and unit suggestions must be validated through historical backtesting and "
    "calibration before being treated as reliable wagering signals."
)

st.caption("Matchup Edge V6 • Prior → Consensus → Matchup → Fair Line → Market Line → EV")
