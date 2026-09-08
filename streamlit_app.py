import streamlit as st

st.set_page_config(
    page_title="Matchup Edge",
    page_icon="🏈",
    layout="wide"
)

st.title("🏈 Matchup Edge")
st.caption("Football matchup analysis & betting evaluation")

st.warning(
    "For research and entertainment only. No wager is guaranteed to win."
)

st.header("1. Enter the Matchup")

col1, col2 = st.columns(2)

with col1:
    away_team = st.text_input("Away Team", placeholder="Oregon")

with col2:
    home_team = st.text_input("Home Team", placeholder="Oklahoma State")

st.header("2. Sportsbook Market")

col1, col2, col3 = st.columns(3)

with col1:
    spread = st.number_input(
        "Home Team Spread",
        value=0.0,
        step=0.5
    )

with col2:
    total = st.number_input(
        "Game Total",
        value=50.0,
        step=0.5
    )

with col3:
    opening_spread = st.number_input(
        "Opening Home Spread",
        value=0.0,
        step=0.5
    )

st.header("3. Matchup Evaluation")

st.caption(
    "Rate each category from the HOME team's perspective. "
    "-10 = huge away-team advantage, 0 = even, +10 = huge home-team advantage."
)

weights = {
    "QB": 15,
    "Offensive Line": 11,
    "Defensive Front": 11,
    "Secondary": 8,
    "Skill Positions": 8,
    "Run Game": 7,
    "Pass Game": 7,
    "Coaching": 5,
    "Special Teams": 3,
    "Injuries": 8,
    "Home Field": 4,
    "Rest / Travel": 3,
    "Matchup Specific": 7,
    "Recent Form": 3
}

ratings = {}

columns = st.columns(2)

for i, factor in enumerate(weights):
    with columns[i % 2]:
        ratings[factor] = st.slider(
            factor,
            -10.0,
            10.0,
            0.0,
            0.5
        )

if st.button(
    "🏈 RUN MATCHUP EVALUATION",
    type="primary",
    use_container_width=True
):

    if not away_team or not home_team:
        st.error("Please enter both teams.")

    else:

        weighted_score = 0

        for factor, weight in weights.items():
            weighted_score += (
                ratings[factor] / 10
            ) * weight

        raw_model_margin = weighted_score * 0.17

        raw_model_spread = -raw_model_margin

        model_spread = (
            0.65 * raw_model_spread
            + 0.35 * spread
        )

        edge = spread - model_spread

        if edge > 0:
            preferred_team = home_team
        else:
            preferred_team = away_team

        abs_edge = abs(edge)

        if abs_edge >= 4:
            grade = "A"
            units = 1.5

        elif abs_edge >= 2:
            grade = "B"
            units = 1.0

        elif abs_edge >= 1:
            grade = "C"
            units = 0.5

        else:
            grade = "PASS"
            units = 0.0

        confidence = min(
            95,
            round(50 + abs_edge * 8)
        )

        st.divider()

        st.header("📊 Model Results")

        a, b, c, d = st.columns(4)

        a.metric(
            "Model Spread",
            f"{home_team} {model_spread:+.1f}"
        )

        b.metric(
            "Market Spread",
            f"{home_team} {spread:+.1f}"
        )

        c.metric(
            "Model Edge",
            f"{abs_edge:.1f} pts"
        )

        d.metric(
            "Grade",
            grade
        )

        st.subheader(
            f"Preferred Side: {preferred_team}"
        )

        st.write(
            f"**Confidence:** {confidence}/100"
        )

        st.write(
            f"**Suggested Size:** {units:.1f} units"
        )

        line_move = spread - opening_spread

        st.subheader("📈 Market Analysis")

        if abs(line_move) < 0.25:
            st.write(
                "No meaningful spread movement."
            )

        elif line_move < 0:
            st.write(
                f"Market has moved {abs(line_move):.1f} "
                f"points toward {home_team}."
            )

        else:
            st.write(
                f"Market has moved {abs(line_move):.1f} "
                f"points toward {away_team}."
            )

        st.subheader("🔍 Biggest Matchup Edges")

        contributions = []

        for factor, weight in weights.items():

            contribution = (
                ratings[factor] / 10
            ) * weight

            contributions.append(
                (factor, contribution)
            )

        contributions.sort(
            key=lambda x: abs(x[1]),
            reverse=True
        )

        for factor, contribution in contributions[:5]:

            if contribution > 0:
                team = home_team
            elif contribution < 0:
                team = away_team
            else:
                team = "Even"

            st.write(
                f"**{factor}:** {team} "
                f"({contribution:+.1f})"
            )

        st.divider()

        if grade == "PASS":

            st.info(
                "PASS — the model does not see "
                "enough separation from the market "
                "to justify forcing a bet."
            )

        elif grade == "A":

            st.success(
                "A-GRADE CANDIDATE — strong model "
                "disagreement with the sportsbook. "
                "Verify injuries and final price "
                "before betting."
            )

        elif grade == "B":

            st.success(
                "B-GRADE CANDIDATE — meaningful "
                "model edge. Worth deeper research."
            )

        else:

            st.warning(
                "C-GRADE LEAN — small edge. "
                "Usually better to wait for a "
                "better number or pass."
            )

st.divider()

st.caption(
    "Matchup Edge v1 • "
    "Matchup → Model Line → Market Comparison → Bet/Pass"
)
