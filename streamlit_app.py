import math
import streamlit as st
import pandas as pd

st.set_page_config(
    page_title="Matchup Edge v2",
    page_icon="🏈",
    layout="wide"
)

st.title("🏈 Matchup Edge v2")
st.caption("Automatic matchup model + line testing + EV analysis")

st.warning(
    "For research and entertainment only. "
    "This model does not guarantee profitable betting results."
)

# -----------------------------
# Utility functions
# -----------------------------

def american_to_decimal(odds):
    if odds > 0:
        return 1 + odds / 100
    return 1 + 100 / abs(odds)


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
    sigma=13.5
):
    if bet_team == home_team:
        threshold = -line
        z = (model_margin_home - threshold) / sigma
        return normal_cdf(z)

    model_margin_away = -model_margin_home
    threshold = -line
    z = (model_margin_away - threshold) / sigma

    return normal_cdf(z)


def total_probability(
    model_total,
    direction,
    line,
    sigma=14.0
):
    if direction == "Over":
        z = (model_total - line) / sigma
        return normal_cdf(z)

    z = (line - model_total) / sigma
    return normal_cdf(z)


def moneyline_probability(
    model_margin_home,
    selected_team,
    home_team,
    away_team
):
    scale = 6.5

    p_home = 1 / (
        1 + math.exp(
            -model_margin_home / scale
        )
    )

    if selected_team == home_team:
        return p_home

    return 1 - p_home


def expected_value_per_dollar(
    model_prob,
    odds
):
    decimal_odds = american_to_decimal(odds)

    return (
        model_prob * (decimal_odds - 1)
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


def suggested_units(
    prob_edge,
    ev
):

    if prob_edge < 0.015 or ev <= 0:
        return 0.0

    if prob_edge < 0.04:
        return 0.5

    if prob_edge < 0.08:
        return 1.0

    if prob_edge < 0.12:
        return 1.5

    return 2.0


def pct(x):
    return f"{100 * x:.1f}%"


# -----------------------------
# MATCHUP DATABASE
# -----------------------------
# This is the part we will eventually
# replace with automatic sports data feeds.

MATCHUPS = {

    "Kansas vs Missouri": {

        "away_team": "Missouri",

        "home_team": "Kansas",

        "model_margin_home": 2.75,

        "model_total": 50.8,

        "confidence": 69,

        "top_edges": [

            ("Missouri QB", -3.8),

            ("Missouri Offensive Line", -3.2),

            ("Kansas Home Field", 2.6),

            (
                "Kansas Coaching / Rivalry Spot",
                2.1
            ),

            (
                "Kansas Defensive Matchup",
                1.8
            )
        ],

        "notes": [
            "Prototype matchup model.",
            "Rerun when injuries or major news changes."
        ]
    },


    "Oregon at Oklahoma State": {

        "away_team": "Oregon",

        "home_team": "Oklahoma State",

        "model_margin_home": -25.4,

        "model_total": 55.8,

        "confidence": 72,

        "top_edges": [

            (
                "Oregon Defensive Front",
                -6.0
            ),

            (
                "Oregon Passing Attack",
                -5.8
            ),

            (
                "Oregon Skill Talent",
                -5.6
            ),

            (
                "Oregon Offensive Line",
                -5.0
            ),

            (
                "Oklahoma State Home Field",
                2.4
            )
        ],

        "notes": [
            "Prototype matchup model.",
            "Users can test unlimited alternate lines against the same evaluation."
        ]
    }
}


# -----------------------------
# Sidebar
# -----------------------------

with st.sidebar:

    st.header("How Matchup Edge Works")

    st.markdown(
        """
        1. Select a matchup.

        2. The model loads a fair spread
        and projected total.

        3. Enter any sportsbook line.

        4. Enter the odds.

        5. The app calculates:
        - Model probability
        - Break-even probability
        - Probability edge
        - Expected value
        - Bet grade
        - Suggested units
        """
    )

    st.info(
        "Future versions will automatically "
        "pull injuries, rosters, statistics, "
        "weather and sportsbook odds."
    )


# -----------------------------
# MATCHUP SELECTION
# -----------------------------

st.header("1. Select Matchup")

matchup_name = st.selectbox(
    "Game",
    list(MATCHUPS.keys())
)

m = MATCHUPS[matchup_name]

away_team = m["away_team"]

home_team = m["home_team"]

model_margin_home = float(
    m["model_margin_home"]
)

model_total = float(
    m["model_total"]
)

confidence = int(
    m["confidence"]
)


if model_margin_home < 0:

    model_spread_text = (
        f"{away_team} "
        f"-{abs(model_margin_home):.1f}"
    )

else:

    model_spread_text = (
        f"{home_team} "
        f"-{abs(model_margin_home):.1f}"
    )


c1, c2, c3 = st.columns(3)

c1.metric(
    "Model Fair Spread",
    model_spread_text
)

c2.metric(
    "Model Fair Total",
    f"{model_total:.1f}"
)

c3.metric(
    "Matchup Confidence",
    f"{confidence}/100"
)


st.subheader("Top Matchup Edges")


for label, value in m["top_edges"]:

    if value > 0:
        side = home_team

    elif value < 0:
        side = away_team

    else:
        side = "Even"


    st.write(
        f"**{label}:** "
        f"{side} "
        f"({value:+.1f})"
    )


for note in m["notes"]:
    st.caption(note)


st.divider()


# -----------------------------
# TEST ONE BET
# -----------------------------

st.header("2. Test a Bet")

bet_type = st.selectbox(
    "Bet Type",
    [
        "Spread",
        "Total",
        "Moneyline"
    ]
)


if bet_type == "Spread":

    team = st.selectbox(
        "Team",
        [
            away_team,
            home_team
        ]
    )

    line = st.number_input(
        "Spread Line",
        value=-3.5,
        step=0.5,
        help=(
            "Examples: -22.5 or +7.5"
        )
    )

    odds = st.number_input(
        "American Odds",
        value=-110,
        step=5
    )


elif bet_type == "Total":

    direction = st.selectbox(
        "Side",
        [
            "Over",
            "Under"
        ]
    )

    line = st.number_input(
        "Total Line",
        value=float(
            round(model_total * 2) / 2
        ),
        step=0.5
    )

    odds = st.number_input(
        "American Odds",
        value=-110,
        step=5
    )


else:

    team = st.selectbox(
        "Team",
        [
            away_team,
            home_team
        ]
    )

    odds = st.number_input(
        "American Odds",
        value=-110,
        step=5
    )


if st.button(
    "🔎 ANALYZE BET",
    type="primary",
    use_container_width=True
):

    implied = american_to_implied_prob(
        int(odds)
    )


    if bet_type == "Spread":

        model_prob = (
            spread_cover_probability(
                model_margin_home,
                team,
                float(line),
                home_team,
                away_team
            )
        )

        label = (
            f"{team} "
            f"{line:+.1f}"
        )

        if team == home_team:

            fair_team_spread = (
                -model_margin_home
            )

        else:

            fair_team_spread = (
                model_margin_home
            )


        point_edge = (
            fair_team_spread
            - float(line)
        )


    elif bet_type == "Total":

        model_prob = (
            total_probability(
                model_total,
                direction,
                float(line)
            )
        )

        label = (
            f"{direction} "
            f"{line:.1f}"
        )


        if direction == "Over":

            point_edge = (
                model_total
                - float(line)
            )

        else:

            point_edge = (
                float(line)
                - model_total
            )


    else:

        model_prob = (
            moneyline_probability(
                model_margin_home,
                team,
                home_team,
                away_team
            )
        )

        label = (
            f"{team} ML"
        )

        point_edge = None


    prob_edge = (
        model_prob
        - implied
    )


    ev = expected_value_per_dollar(
        model_prob,
        int(odds)
    )


    grade = edge_grade(
        prob_edge
    )


    units = suggested_units(
        prob_edge,
        ev
    )


    st.divider()

    st.subheader(
        f"Result: {label}"
    )


    a, b, c, d = st.columns(4)


    a.metric(
        "Model Probability",
        pct(model_prob)
    )


    b.metric(
        "Break-even Probability",
        pct(implied)
    )


    c.metric(
        "Probability Edge",
        f"{100 * prob_edge:+.1f}%"
    )


    d.metric(
        "Grade",
        grade
    )


    e1, e2 = st.columns(2)


    e1.metric(
        "EV per $1",
        f"${ev:+.3f}"
    )


    e2.metric(
        "Suggested Size",
        f"{units:.1f}u"
    )


    if point_edge is not None:

        st.metric(
            "Model Line Cushion",
            f"{point_edge:+.1f} pts"
        )


    if grade == "PASS":

        st.info(
            "PASS — the price does not "
            "create enough estimated edge."
        )


    elif grade == "C":

        st.warning(
            "C-GRADE LEAN — small edge. "
            "Usually wait for a better price."
        )


    elif grade == "B":

        st.success(
            "B-GRADE CANDIDATE — meaningful "
            "model-versus-price edge."
        )


    else:

        st.success(
            "A-GRADE CANDIDATE — strong "
            "model-versus-price discrepancy. "
            "Verify injury news before betting."
        )


    st.caption(
        "Probability estimates use a "
        "football scoring distribution model."
    )


st.divider()


# -----------------------------
# COMPARE ALTERNATE LINES
# -----------------------------

st.header("3. Compare Alternate Lines")

st.write(
    "Enter several sportsbook options and "
    "compare the line/price combinations."
)


compare_type = st.selectbox(
    "Compare",
    [
        "Spreads",
        "Totals"
    ]
)


rows = []


if compare_type == "Spreads":

    compare_team = st.selectbox(
        "Compare Team",
        [
            away_team,
            home_team
        ],
        key="compare_team"
    )


    if (
        compare_team == away_team
        and model_margin_home < 0
    ):

        defaults = [
            (-19.5, -250),
            (-20.5, -190),
            (-21.5, -150),
            (-22.5, -110)
        ]

    else:

        defaults = [
            (3.5, -110),
            (6.5, -150),
            (9.5, -250),
            (13.5, -500)
        ]


    for i in range(4):

        c1, c2 = st.columns(2)


        with c1:

            test_line = (
                st.number_input(
                    f"Line {i + 1}",
                    value=float(
                        defaults[i][0]
                    ),
                    step=0.5,
                    key=f"spread_line_{i}"
                )
            )


        with c2:

            test_odds = (
                st.number_input(
                    f"Odds {i + 1}",
                    value=int(
                        defaults[i][1]
                    ),
                    step=5,
                    key=f"spread_odds_{i}"
                )
            )


        p = spread_cover_probability(
            model_margin_home,
            compare_team,
            test_line,
            home_team,
            away_team
        )


        implied = (
            american_to_implied_prob(
                int(test_odds)
            )
        )


        prob_edge = (
            p - implied
        )


        ev = expected_value_per_dollar(
            p,
            int(test_odds)
        )


        rows.append(
            {
                "Bet":
                    f"{compare_team} "
                    f"{test_line:+.1f}",

                "Odds":
                    int(test_odds),

                "Model Prob":
                    p,

                "Break-even":
                    implied,

                "Prob Edge":
                    prob_edge,

                "EV/$1":
                    ev,

                "Grade":
                    edge_grade(
                        prob_edge
                    )
            }
        )


else:

    compare_direction = (
        st.selectbox(
            "Direction",
            [
                "Over",
                "Under"
            ]
        )
    )


    defaults = [
        (41.5, -400),
        (48.5, -180),
        (54.5, -110),
        (58.5, -110)
    ]


    for i in range(4):

        c1, c2 = st.columns(2)


        with c1:

            test_line = (
                st.number_input(
                    f"Total {i + 1}",
                    value=float(
                        defaults[i][0]
                    ),
                    step=0.5,
                    key=f"total_line_{i}"
                )
            )


        with c2:

            test_odds = (
                st.number_input(
                    f"Odds {i + 1}",
                    value=int(
                        defaults[i][1]
                    ),
                    step=5,
                    key=f"total_odds_{i}"
                )
            )


        p = total_probability(
            model_total,
            compare_direction,
            test_line
        )


        implied = (
            american_to_implied_prob(
                int(test_odds)
            )
        )


        prob_edge = (
            p - implied
        )


        ev = expected_value_per_dollar(
            p,
            int(test_odds)
        )


        rows.append(
            {
                "Bet":
                    f"{compare_direction} "
                    f"{test_line:.1f}",

                "Odds":
                    int(test_odds),

                "Model Prob":
                    p,

                "Break-even":
                    implied,

                "Prob Edge":
                    prob_edge,

                "EV/$1":
                    ev,

                "Grade":
                    edge_grade(
                        prob_edge
                    )
            }
        )


df = pd.DataFrame(rows)


df["Model Prob"] = (
    df["Model Prob"]
    .map(
        lambda x:
        f"{100 * x:.1f}%"
    )
)


df["Break-even"] = (
    df["Break-even"]
    .map(
        lambda x:
        f"{100 * x:.1f}%"
    )
)


df["Prob Edge"] = (
    df["Prob Edge"]
    .map(
        lambda x:
        f"{100 * x:+.1f}%"
    )
)


df["EV/$1"] = (
    df["EV/$1"]
    .map(
        lambda x:
        f"${x:+.3f}"
    )
)


st.dataframe(
    df,
    use_container_width=True,
    hide_index=True
)


st.divider()


# -----------------------------
# ROADMAP
# -----------------------------

st.header("4. Matchup Edge Roadmap")

st.markdown(
    """
    Future automatic features:

    - Live game schedule
    - Sportsbook odds
    - Alternate spreads
    - Alternate totals
    - Injury reports
    - Rosters and depth charts
    - QB efficiency
    - Offensive line metrics
    - Defensive front metrics
    - Explosive-play rates
    - Team efficiency
    - Weather
    - Travel and rest
    - Line movement
    - Closing-line value tracking
    - Historical model performance
    """
)


st.caption(
    "Matchup Edge v2 • "
    "Matchup Model → Test Line → "
    "Calculate EV → Bet / Pass"
)
