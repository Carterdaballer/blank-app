import math
import json
import urllib.parse
import urllib.request
from collections import defaultdict

import pandas as pd
import streamlit as st


# =========================================================
# PAGE SETUP
# =========================================================

st.set_page_config(
    page_title="Matchup Edge V4",
    page_icon="🏈",
    layout="wide",
)

st.title("🏈 Matchup Edge V4")
st.caption(
    "College Football Matchup Model • "
    "Power Ratings • Fair Lines • EV Analysis"
)

st.warning(
    "For research and entertainment only. "
    "Model probabilities are estimates, not guarantees."
)


# =========================================================
# CFBD CONNECTION
# =========================================================

try:
    CFBD_API_KEY = st.secrets["CFBD_API_KEY"]
except Exception:
    st.error(
        "CFBD_API_KEY is missing from Streamlit Secrets."
    )
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

    with urllib.request.urlopen(
        request,
        timeout=25,
    ) as response:
        return json.loads(
            response.read().decode("utf-8")
        )


def get_field(data, *names, default=None):
    for name in names:
        if name in data and data[name] is not None:
            return data[name]

    return default


# =========================================================
# ODDS FUNCTIONS
# =========================================================

def american_to_decimal(odds):
    if odds > 0:
        return 1 + odds / 100

    return 1 + 100 / abs(odds)


def implied_probability(odds):
    if odds > 0:
        return 100 / (odds + 100)

    return abs(odds) / (
        abs(odds) + 100
    )


def normal_cdf(x):
    return 0.5 * (
        1 + math.erf(
            x / math.sqrt(2)
        )
    )


def expected_value(model_prob, odds):
    decimal = american_to_decimal(odds)

    profit = decimal - 1

    return (
        model_prob * profit
        - (1 - model_prob)
    )


def edge_grade(prob_edge, ev):
    if ev <= 0:
        return "PASS"

    if prob_edge >= 0.08:
        return "A"

    if prob_edge >= 0.04:
        return "B"

    if prob_edge >= 0.015:
        return "C"

    return "PASS"


def suggested_units(prob_edge, ev):
    if ev <= 0 or prob_edge < 0.015:
        return 0.0

    if prob_edge < 0.04:
        return 0.5

    if prob_edge < 0.08:
        return 1.0

    if prob_edge < 0.12:
        return 1.5

    return 2.0


# =========================================================
# SCHEDULE
# =========================================================

@st.cache_data(ttl=3600)
def get_week_games(
    year,
    week,
    api_key,
):
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


st.divider()
st.header("📅 Select Matchup")

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
    schedule = get_week_games(
        season,
        week,
        CFBD_API_KEY,
    )

except Exception as error:
    st.error(
        "The 2026 schedule could not be loaded."
    )
    st.code(str(error))
    st.stop()


if not schedule:
    st.info(
        f"No games were returned for Week {week}."
    )
    st.stop()


games = []

for game in schedule:
    away = get_field(
        game,
        "awayTeam",
        "away_team",
        default="Away",
    )

    home = get_field(
        game,
        "homeTeam",
        "home_team",
        default="Home",
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

    neutral_site = get_field(
        game,
        "neutralSite",
        "neutral_site",
        default=False,
    )

    games.append(
        {
            "away": away,
            "home": home,
            "start_date": start_date,
            "venue": venue,
            "neutral_site": neutral_site,
        }
    )


game_labels = [
    f"{g['away']} @ {g['home']}"
    for g in games
]

selected_label = st.selectbox(
    "Game",
    game_labels,
)

selected_game = games[
    game_labels.index(selected_label)
]

away_team = selected_game["away"]
home_team = selected_game["home"]

st.subheader(
    f"{away_team} @ {home_team}"
)

if selected_game["start_date"]:
    st.write(
        f"**Kickoff:** "
        f"{selected_game['start_date']}"
    )

if selected_game["venue"]:
    st.write(
        f"**Venue:** "
        f"{selected_game['venue']}"
    )

if selected_game["neutral_site"]:
    st.write("**Site:** Neutral")


# =========================================================
# LOAD ALL PRIOR GAMES
# =========================================================

@st.cache_data(ttl=3600)
def get_prior_games(
    year,
    selected_week,
    api_key,
):
    all_games = []

    for w in range(0, selected_week):
        week_games = cfbd_get(
            "/games",
            {
                "year": year,
                "week": w,
                "seasonType": "regular",
            },
            api_key,
        )

        all_games.extend(
            week_games
        )

    return all_games


try:
    raw_prior_games = get_prior_games(
        season,
        week,
        CFBD_API_KEY,
    )

except Exception as error:
    st.error(
        "Previous game results could not be loaded."
    )
    st.code(str(error))
    st.stop()


# =========================================================
# BUILD TEAM RESULTS
# =========================================================

def build_team_results(raw_games):
    results = defaultdict(list)

    seen_ids = set()

    for game in raw_games:
        game_id = get_field(
            game,
            "id",
            default=None,
        )

        if game_id is not None:
            if game_id in seen_ids:
                continue

            seen_ids.add(game_id)

        game_week = get_field(
            game,
            "week",
            default=0,
        )

        home = get_field(
            game,
            "homeTeam",
            "home_team",
            default="",
        )

        away = get_field(
            game,
            "awayTeam",
            "away_team",
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

        if (
            not home
            or not away
            or home_points is None
            or away_points is None
        ):
            continue

        home_margin = (
            home_points - away_points
        )

        away_margin = -home_margin

        results[home].append(
            {
                "week": game_week,
                "opponent": away,
                "pf": home_points,
                "pa": away_points,
                "margin": home_margin,
                "location": "Home",
            }
        )

        results[away].append(
            {
                "week": game_week,
                "opponent": home,
                "pf": away_points,
                "pa": home_points,
                "margin": away_margin,
                "location": "Away",
            }
        )

    return results


team_results = build_team_results(
    raw_prior_games
)


# =========================================================
# RECENCY WEIGHTING
# =========================================================

def recency_weight(
    game_week,
    selected_week,
):
    weeks_ago = max(
        1,
        selected_week - game_week,
    )

    return 0.88 ** (
        weeks_ago - 1
    )


def weighted_average(
    values,
    weights,
):
    if not values:
        return 0.0

    total_weight = sum(weights)

    if total_weight == 0:
        return 0.0

    return sum(
        value * weight
        for value, weight
        in zip(values, weights)
    ) / total_weight


# =========================================================
# RAW TEAM PERFORMANCE
# =========================================================

def raw_team_metrics(
    team,
    results,
    selected_week,
):
    games_list = results.get(
        team,
        [],
    )

    if not games_list:
        return {
            "games": 0,
            "margin": 0.0,
            "offense": 0.0,
            "defense": 0.0,
        }

    margins = []
    points_for = []
    points_against = []
    weights = []

    for game in games_list:
        weight = recency_weight(
            game["week"],
            selected_week,
        )

        margins.append(
            game["margin"]
        )

        points_for.append(
            game["pf"]
        )

        points_against.append(
            game["pa"]
        )

        weights.append(weight)

    return {
        "games": len(games_list),

        "margin": weighted_average(
            margins,
            weights,
        ),

        "offense": weighted_average(
            points_for,
            weights,
        ),

        "defense": weighted_average(
            points_against,
            weights,
        ),
    }


# =========================================================
# OPPONENT-ADJUSTED RATINGS
# =========================================================

def calculate_adjusted_ratings(
    results,
    selected_week,
):
    ratings = {}

    for team in results:
        metrics = raw_team_metrics(
            team,
            results,
            selected_week,
        )

        ratings[team] = metrics[
            "margin"
        ]

    for _ in range(10):
        new_ratings = {}

        for team, games_list in results.items():
            performances = []
            weights = []

            for game in games_list:
                opponent_rating = ratings.get(
                    game["opponent"],
                    0.0,
                )

                performance = (
                    game["margin"]
                    + opponent_rating
                )

                weight = recency_weight(
                    game["week"],
                    selected_week,
                )

                performances.append(
                    performance
                )

                weights.append(
                    weight
                )

            new_ratings[team] = weighted_average(
                performances,
                weights,
            )

        if new_ratings:
            center = (
                sum(new_ratings.values())
                / len(new_ratings)
            )

            ratings = {
                team: rating - center
                for team, rating
                in new_ratings.items()
            }

    return ratings


adjusted_ratings = (
    calculate_adjusted_ratings(
        team_results,
        week,
    )
)


# =========================================================
# EARLY-SEASON STABILIZATION
# =========================================================

def stabilized_rating(
    team,
    adjusted_ratings,
    results,
):
    current_rating = adjusted_ratings.get(
        team,
        0.0,
    )

    games_played = len(
        results.get(team, [])
    )

    # Early-season shrinkage toward average.
    # As more games are played, the team's actual
    # 2026 results receive progressively more weight.
    current_weight = min(
        0.85,
        0.25 + (
            0.10 * games_played
        ),
    )

    prior_rating = 0.0

    final_rating = (
        current_weight
        * current_rating
        + (
            1 - current_weight
        )
        * prior_rating
    )

    return (
        final_rating,
        current_weight,
    )


away_rating, away_current_weight = (
    stabilized_rating(
        away_team,
        adjusted_ratings,
        team_results,
    )
)

home_rating, home_current_weight = (
    stabilized_rating(
        home_team,
        adjusted_ratings,
        team_results,
    )
)


# =========================================================
# OFFENSE / DEFENSE
# =========================================================

away_metrics = raw_team_metrics(
    away_team,
    team_results,
    week,
)

home_metrics = raw_team_metrics(
    home_team,
    team_results,
    week,
)


all_pf = []

for team, games_list in team_results.items():
    for game in games_list:
        all_pf.append(
            game["pf"]
        )


if all_pf:
    national_scoring_average = (
        sum(all_pf) / len(all_pf)
    )
else:
    national_scoring_average = 27.0


def stabilized_scoring(
    value,
    games_played,
    national_average,
):
    sample_weight = min(
        0.80,
        0.25 + (
            0.10 * games_played
        ),
    )

    return (
        sample_weight * value
        + (1 - sample_weight)
        * national_average
    )


away_offense = stabilized_scoring(
    away_metrics["offense"],
    away_metrics["games"],
    national_scoring_average,
)

away_defense_allowed = stabilized_scoring(
    away_metrics["defense"],
    away_metrics["games"],
    national_scoring_average,
)

home_offense = stabilized_scoring(
    home_metrics["offense"],
    home_metrics["games"],
    national_scoring_average,
)

home_defense_allowed = stabilized_scoring(
    home_metrics["defense"],
    home_metrics["games"],
    national_scoring_average,
)


# =========================================================
# FAIR SPREAD
# =========================================================

if selected_game["neutral_site"]:
    home_field = 0.0
else:
    home_field = 2.5


rating_difference = (
    home_rating - away_rating
)

model_home_margin = (
    rating_difference
    + home_field
)


# Limit extreme early-season projections.
model_home_margin = max(
    -35.0,
    min(
        35.0,
        model_home_margin,
    ),
)


# =========================================================
# FAIR TOTAL
# =========================================================

expected_away_points = (
    away_offense
    + home_defense_allowed
) / 2

expected_home_points = (
    home_offense
    + away_defense_allowed
) / 2


if not selected_game["neutral_site"]:
    expected_home_points += 1.25
    expected_away_points -= 1.25


model_total = (
    expected_home_points
    + expected_away_points
)


model_total = max(
    30.0,
    min(
        85.0,
        model_total,
    ),
)


# =========================================================
# MODEL DASHBOARD
# =========================================================

st.divider()
st.header("🧠 Matchup Edge Model")

col1, col2 = st.columns(2)

with col1:
    st.subheader(away_team)

    st.metric(
        "Power Rating",
        f"{away_rating:+.1f}",
    )

    st.metric(
        "Games in Model",
        away_metrics["games"],
    )

    st.write(
        f"**Scoring:** "
        f"{away_metrics['offense']:.1f} PPG"
    )

    st.write(
        f"**Allowed:** "
        f"{away_metrics['defense']:.1f} PPG"
    )


with col2:
    st.subheader(home_team)

    st.metric(
        "Power Rating",
        f"{home_rating:+.1f}",
    )

    st.metric(
        "Games in Model",
        home_metrics["games"],
    )

    st.write(
        f"**Scoring:** "
        f"{home_metrics['offense']:.1f} PPG"
    )

    st.write(
        f"**Allowed:** "
        f"{home_metrics['defense']:.1f} PPG"
    )


st.caption(
    "Power ratings are opponent-adjusted, "
    "recency-weighted and shrunk toward average "
    "when the current-season sample is small."
)


st.subheader("🎯 Model Fair Lines")

col1, col2 = st.columns(2)

with col1:
    if model_home_margin >= 0:
        fair_spread_text = (
            f"{home_team} "
            f"-{model_home_margin:.1f}"
        )
    else:
        fair_spread_text = (
            f"{away_team} "
            f"-{abs(model_home_margin):.1f}"
        )

    st.metric(
        "Fair Spread",
        fair_spread_text,
    )


with col2:
    st.metric(
        "Fair Total",
        f"{model_total:.1f}",
    )


st.write(
    f"**Home-field adjustment:** "
    f"{home_field:+.1f}"
)


# =========================================================
# MODEL CONFIDENCE
# =========================================================

away_games_count = (
    away_metrics["games"]
)

home_games_count = (
    home_metrics["games"]
)

minimum_sample = min(
    away_games_count,
    home_games_count,
)

confidence = min(
    85,
    45 + minimum_sample * 5,
)

if minimum_sample == 0:
    confidence = 35


st.metric(
    "Model Confidence",
    f"{confidence}/100",
)

if minimum_sample <= 2:
    st.info(
        "Early-season sample is small. "
        "Ratings are intentionally conservative."
    )


# =========================================================
# RECENT RESULTS
# =========================================================

st.divider()
st.header("📋 Games Used by Model")


def history_dataframe(team):
    rows = []

    for game in team_results.get(
        team,
        [],
    ):
        margin = game["margin"]

        if margin > 0:
            result = "W"
        elif margin < 0:
            result = "L"
        else:
            result = "T"

        rows.append(
            {
                "Week": game["week"],
                "Opponent": game["opponent"],
                "Site": game["location"],
                "Result": result,
                "PF": game["pf"],
                "PA": game["pa"],
                "Margin": margin,
            }
        )

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(
        sorted(
            rows,
            key=lambda row: row["Week"],
        )
    )


col1, col2 = st.columns(2)

with col1:
    st.subheader(away_team)

    away_history_df = (
        history_dataframe(
            away_team
        )
    )

    if away_history_df.empty:
        st.write(
            "No completed prior games."
        )
    else:
        st.dataframe(
            away_history_df,
            use_container_width=True,
            hide_index=True,
        )


with col2:
    st.subheader(home_team)

    home_history_df = (
        history_dataframe(
            home_team
        )
    )

    if home_history_df.empty:
        st.write(
            "No completed prior games."
        )
    else:
        st.dataframe(
            home_history_df,
            use_container_width=True,
            hide_index=True,
        )


# =========================================================
# LINE LAB
# =========================================================

st.divider()
st.header("🧪 Sportsbook Line Lab")

st.write(
    "Enter the exact line and odds offered by "
    "your sportsbook. The underlying matchup "
    "evaluation stays fixed."
)

bet_type = st.selectbox(
    "Bet Type",
    [
        "Spread",
        "Total",
        "Moneyline",
    ],
)


model_probability = None
bet_description = ""


# =========================================================
# SPREAD
# =========================================================

if bet_type == "Spread":
    selected_team = st.selectbox(
        "Bet Team",
        [
            away_team,
            home_team,
        ],
    )

    spread_line = st.number_input(
        "Spread",
        value=0.0,
        step=0.5,
        format="%.1f",
    )

    odds = st.number_input(
        "American Odds",
        value=-110,
        step=1,
        key="spread_odds",
    )

    if selected_team == home_team:
        expected_margin = (
            model_home_margin
        )
    else:
        expected_margin = (
            -model_home_margin
        )

    threshold = -spread_line

    model_probability = (
        1 - normal_cdf(
            (
                threshold
                - expected_margin
            ) / 13.5
        )
    )

    bet_description = (
        f"{selected_team} "
        f"{spread_line:+.1f}"
    )


# =========================================================
# TOTAL
# =========================================================

elif bet_type == "Total":
    direction = st.selectbox(
        "Direction",
        [
            "Over",
            "Under",
        ],
    )

    total_line = st.number_input(
        "Total",
        value=50.0,
        step=0.5,
        format="%.1f",
    )

    odds = st.number_input(
        "American Odds",
        value=-110,
        step=1,
        key="total_odds",
    )

    under_probability = normal_cdf(
        (
            total_line
            - model_total
        ) / 14.0
    )

    if direction == "Under":
        model_probability = (
            under_probability
        )
    else:
        model_probability = (
            1 - under_probability
        )

    bet_description = (
        f"{direction} "
        f"{total_line:.1f}"
    )


# =========================================================
# MONEYLINE
# =========================================================

else:
    selected_team = st.selectbox(
        "Moneyline Team",
        [
            away_team,
            home_team,
        ],
    )

    odds = st.number_input(
        "American Odds",
        value=100,
        step=1,
        key="moneyline_odds",
    )

    home_win_probability = (
        1 / (
            1 + math.exp(
                -model_home_margin
                / 6.5
            )
        )
    )

    if selected_team == home_team:
        model_probability = (
            home_win_probability
        )
    else:
        model_probability = (
            1 - home_win_probability
        )

    bet_description = (
        f"{selected_team} ML"
    )


# =========================================================
# BET EVALUATION
# =========================================================

if odds == 0:
    st.error(
        "American odds cannot be zero."
    )

else:
    break_even = implied_probability(
        odds
    )

    probability_edge = (
        model_probability
        - break_even
    )

    ev = expected_value(
        model_probability,
        odds,
    )

    grade = edge_grade(
        probability_edge,
        ev,
    )

    units = suggested_units(
        probability_edge,
        ev,
    )


    st.subheader("📊 Bet Evaluation")

    st.write(
        f"### {bet_description}"
    )

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Model Probability",
            f"{model_probability * 100:.1f}%",
        )

        st.metric(
            "Break-even Probability",
            f"{break_even * 100:.1f}%",
        )

    with col2:
        st.metric(
            "Probability Edge",
            f"{probability_edge * 100:+.1f}%",
        )

        st.metric(
            "EV per $1",
            f"${ev:+.3f}",
        )


    st.metric(
        "Matchup Edge Grade",
        grade,
    )

    st.metric(
        "Suggested Units",
        f"{units:.1f}u",
    )


    if grade == "A":
        st.success(
            "A-grade model edge. "
            "Still verify injuries, quarterback "
            "status and market information."
        )

    elif grade == "B":
        st.success(
            "B-grade model edge. "
            "Potential betting candidate."
        )

    elif grade == "C":
        st.warning(
            "Small model edge. "
            "Price and matchup details matter."
        )

    else:
        st.info(
            "PASS — current price does not "
            "provide enough modeled value."
        )


# =========================================================
# ALTERNATE LINE COMPARISON
# =========================================================

st.divider()
st.header("🔬 Compare Alternate Lines")

comparison_type = st.selectbox(
    "Comparison",
    [
        "Spreads",
        "Totals",
    ],
)


comparison_rows = []


if comparison_type == "Spreads":
    comparison_team = st.selectbox(
        "Team",
        [
            away_team,
            home_team,
        ],
        key="comparison_team",
    )

    default_lines = [
        -2.5,
        -1.5,
        2.5,
        6.5,
    ]

    for i in range(4):
        col1, col2 = st.columns(2)

        with col1:
            line = st.number_input(
                f"Line {i + 1}",
                value=default_lines[i],
                step=0.5,
                key=f"spread_line_{i}",
            )

        with col2:
            line_odds = st.number_input(
                f"Odds {i + 1}",
                value=-110,
                step=1,
                key=f"spread_price_{i}",
            )

        if comparison_team == home_team:
            expected_margin = (
                model_home_margin
            )
        else:
            expected_margin = (
                -model_home_margin
            )

        probability = (
            1 - normal_cdf(
                (
                    -line
                    - expected_margin
                ) / 13.5
            )
        )

        if line_odds != 0:
            break_even_prob = (
                implied_probability(
                    line_odds
                )
            )

            edge = (
                probability
                - break_even_prob
            )

            line_ev = expected_value(
                probability,
                line_odds,
            )

            line_grade = edge_grade(
                edge,
                line_ev,
            )

            comparison_rows.append(
                {
                    "Bet": (
                        f"{comparison_team} "
                        f"{line:+.1f}"
                    ),
                    "Odds": line_odds,
                    "Model Prob": (
                        f"{probability * 100:.1f}%"
                    ),
                    "Break-even": (
                        f"{break_even_prob * 100:.1f}%"
                    ),
                    "Prob Edge": (
                        f"{edge * 100:+.1f}%"
                    ),
                    "EV/$1": (
                        f"${line_ev:+.3f}"
                    ),
                    "Grade": line_grade,
                }
            )


else:
    direction = st.selectbox(
        "Direction",
        [
            "Over",
            "Under",
        ],
        key="comparison_direction",
    )

    default_totals = [
        45.5,
        49.5,
        53.5,
        57.5,
    ]

    for i in range(4):
        col1, col2 = st.columns(2)

        with col1:
            line = st.number_input(
                f"Total {i + 1}",
                value=default_totals[i],
                step=0.5,
                key=f"total_line_{i}",
            )

        with col2:
            line_odds = st.number_input(
                f"Odds {i + 1}",
                value=-110,
                step=1,
                key=f"total_price_{i}",
            )

        under_probability = normal_cdf(
            (
                line - model_total
            ) / 14.0
        )

        if direction == "Under":
            probability = (
                under_probability
            )
        else:
            probability = (
                1 - under_probability
            )

        if line_odds != 0:
            break_even_prob = (
                implied_probability(
                    line_odds
                )
            )

            edge = (
                probability
                - break_even_prob
            )

            line_ev = expected_value(
                probability,
                line_odds,
            )

            line_grade = edge_grade(
                edge,
                line_ev,
            )

            comparison_rows.append(
                {
                    "Bet": (
                        f"{direction} "
                        f"{line:.1f}"
                    ),
                    "Odds": line_odds,
                    "Model Prob": (
                        f"{probability * 100:.1f}%"
                    ),
                    "Break-even": (
                        f"{break_even_prob * 100:.1f}%"
                    ),
                    "Prob Edge": (
                        f"{edge * 100:+.1f}%"
                    ),
                    "EV/$1": (
                        f"${line_ev:+.3f}"
                    ),
                    "Grade": line_grade,
                }
            )


if comparison_rows:
    comparison_df = pd.DataFrame(
        comparison_rows
    )

    st.dataframe(
        comparison_df,
        use_container_width=True,
        hide_index=True,
    )


# =========================================================
# MODEL NOTES
# =========================================================

st.divider()
st.header("🔧 Model Status")

st.markdown(
    """
**Currently automatic**
- 2026 FBS schedule
- Completed results before selected matchup
- Recency weighting
- Opponent-adjusted performance
- Early-season sample stabilization
- Offensive scoring
- Defensive scoring allowed
- Home-field adjustment
- Fair spread
- Fair total
- Spread probability
- Total probability
- Moneyline probability
- Break-even probability
- Expected value
- A/B/C/PASS grades
- Suggested units
- Alternate-line comparison

**Next data layers**
- Preseason team-strength priors
- Advanced offensive efficiency
- Advanced defensive efficiency
- Strength of schedule improvements
- QB performance
- Rosters / returning production
- Offensive line / defensive front
- Injuries
- Weather
- Rest / travel
- Sportsbook odds and line movement
- Prediction history and CLV
- Historical backtesting and calibration
"""
)

st.caption(
    "Matchup Edge V4 • "
    "Data → Matchup → Fair Line → Price → EV"
)


# =========================================================
# V5 API ACCESS DIAGNOSTIC
# =========================================================

st.divider()
st.header("🧪 V5 Data Access Test")

def test_cfbd_endpoint(name, path, params):
    try:
        data = cfbd_get(
            path,
            params,
            CFBD_API_KEY,
        )

        if isinstance(data, list):
            count = len(data)
        else:
            count = 1 if data else 0

        return {
            "Source": name,
            "Status": "✅ WORKING",
            "Records": count,
            "Error": "",
        }

    except Exception as error:
        return {
            "Source": name,
            "Status": "❌ BLOCKED / ERROR",
            "Records": 0,
            "Error": str(error),
        }


tests = []

tests.append(
    test_cfbd_endpoint(
        "CORE",
        "/ratings/core",
        {
            "year": 2026,
        },
    )
)

tests.append(
    test_cfbd_endpoint(
        "SP+",
        "/ratings/sp",
        {
            "year": 2026,
        },
    )
)

tests.append(
    test_cfbd_endpoint(
        "Elo",
        "/ratings/elo",
        {
            "year": 2026,
            "week": max(1, week - 1),
            "seasonType": "regular",
        },
    )
)

tests.append(
    test_cfbd_endpoint(
        "SRS",
        "/ratings/srs",
        {
            "year": 2026,
        },
    )
)

tests.append(
    test_cfbd_endpoint(
        "FPI",
        "/ratings/fpi",
        {
            "year": 2026,
        },
    )
)


test_df = pd.DataFrame(tests)

st.dataframe(
    test_df,
    use_container_width=True,
    hide_index=True,
)


working_count = sum(
    1
    for test in tests
    if test["Status"] == "✅ WORKING"
)


if working_count >= 3:
    st.success(
        f"{working_count}/5 advanced rating "
        "sources are accessible. "
        "We have enough to build V5."
    )

elif working_count >= 1:
    st.warning(
        f"{working_count}/5 advanced rating "
        "sources are accessible. "
        "V5 can use the available sources."
    )

else:
    st.error(
        "None of the advanced rating sources "
        "are accessible with this API key."
    )
