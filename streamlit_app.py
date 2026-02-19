import streamlit as st
import requests
from requests.exceptions import ReadTimeout, RequestException
import os

st.set_page_config(layout="wide")
st.title("Multi-Sport Prop Analyzer")

def render_table_html(rows, columns):
    if not rows:
        st.write("No data.")
        return
    header = "".join(f"<th>{c}</th>" for c in columns)
    body_rows = []
    for r in rows:
        body_rows.append("<tr>" + "".join(f"<td>{r.get(c, '')}</td>" for c in columns) + "</tr>")
    html = f"<table><thead><tr>{header}</tr></thead><tbody>{''.join(body_rows)}</tbody></table>"
    st.markdown(html, unsafe_allow_html=True)


def render_hit_rate_bars(rows):
    if not rows:
        st.write("No chart data.")
        return
    st.markdown("**Hit Rate Overview**")
    for r in rows:
        label = r.get("Window", "")
        val = float(r.get("Hit Rate (%)", 0))
        val = max(0.0, min(100.0, val))
        st.markdown(
            f"<div style='margin:6px 0;'>"
            f"<div style='font-size:14px;margin-bottom:2px;'>{label}: {val:.2f}%</div>"
            f"<div style='background:#eee;border-radius:6px;overflow:hidden;height:12px;'>"
            f"<div style='width:{val}%;background:#2e86de;height:12px;'></div>"
            f"</div></div>",
            unsafe_allow_html=True,
        )


BACKEND_BASE_URL = os.getenv("BACKEND_BASE_URL", "https://champion-prop-analyzer.onrender.com").rstrip("/")
BACKEND_URL = os.getenv("BACKEND_ANALYZE_URL", f"{BACKEND_BASE_URL}/analyze")

DEFAULT_PRESETS = {
    "Default": {"season_type": "Regular Season", "window_1": 5, "window_2": 10, "hit_operator": "gt",
                "conf_l5_min": 50, "conf_l10_min": 50, "conf_h2h_good": 60, "conf_low_max": 40},
    "Aggressive": {"season_type": "Regular Season", "window_1": 5, "window_2": 10, "hit_operator": "gt",
                   "conf_l5_min": 55, "conf_l10_min": 55, "conf_h2h_good": 65, "conf_low_max": 35},
    "Conservative": {"season_type": "Regular Season", "window_1": 7, "window_2": 15, "hit_operator": "gte",
                     "conf_l5_min": 60, "conf_l10_min": 60, "conf_h2h_good": 70, "conf_low_max": 45},
}

if "custom_presets" not in st.session_state:
    st.session_state["custom_presets"] = {}

all_presets = {**DEFAULT_PRESETS, **st.session_state["custom_presets"]}

if "preset" not in st.session_state:
    st.session_state["preset"] = "Default"
if "last_preset" not in st.session_state:
    st.session_state["last_preset"] = st.session_state["preset"]

st.sidebar.markdown("### Settings")
preset = st.sidebar.selectbox("Preset", list(all_presets.keys()), key="preset")

if preset != st.session_state["last_preset"]:
    p = all_presets[preset]
    st.session_state["season_type"] = p["season_type"]
    st.session_state["window_1"] = p["window_1"]
    st.session_state["window_2"] = p["window_2"]
    st.session_state["hit_operator"] = p["hit_operator"]
    st.session_state["conf_l5_min"] = p["conf_l5_min"]
    st.session_state["conf_l10_min"] = p["conf_l10_min"]
    st.session_state["conf_h2h_good"] = p["conf_h2h_good"]
    st.session_state["conf_low_max"] = p["conf_low_max"]
    st.session_state["last_preset"] = preset

for key in ["season_type", "window_1", "window_2", "hit_operator", "conf_l5_min", "conf_l10_min", "conf_h2h_good", "conf_low_max"]:
    if key not in st.session_state:
        st.session_state[key] = all_presets[preset][key]

season_type = st.sidebar.selectbox(
    "Season Type",
    ["Regular Season", "Playoffs"],
    index=["Regular Season", "Playoffs"].index(st.session_state["season_type"]),
    key="season_type",
)
window_1 = st.sidebar.slider("Last 5 Games (adjustable)", 1, 30, st.session_state["window_1"], key="window_1")
window_2 = st.sidebar.slider("Last 10 Games (adjustable)", 1, 50, st.session_state["window_2"], key="window_2")
hit_operator = st.sidebar.selectbox(
    "Hit Operator",
    ["gt", "gte"],
    index=["gt", "gte"].index(st.session_state["hit_operator"]),
    key="hit_operator",
)
offered_odds_input = st.sidebar.text_input("Offered Odds (American)", value="")
include_injury = st.sidebar.checkbox("Include injury context", value=False)

st.sidebar.markdown("### Model Tuning")
conf_l5_min = st.sidebar.slider("Conf L5 Min", 0, 100, st.session_state["conf_l5_min"], key="conf_l5_min")
conf_l10_min = st.sidebar.slider("Conf L10 Min", 0, 100, st.session_state["conf_l10_min"], key="conf_l10_min")
conf_h2h_good = st.sidebar.slider("Conf H2H Good", 0, 100, st.session_state["conf_h2h_good"], key="conf_h2h_good")
conf_low_max = st.sidebar.slider("Conf Low Max", 0, 100, st.session_state["conf_low_max"], key="conf_low_max")

st.sidebar.markdown("### Save Preset")
preset_name = st.sidebar.text_input("Preset Name", "")
if st.sidebar.button("Save Preset") and preset_name.strip():
    st.session_state["custom_presets"][preset_name.strip()] = {
        "season_type": season_type,
        "window_1": window_1,
        "window_2": window_2,
        "hit_operator": hit_operator,
        "conf_l5_min": conf_l5_min,
        "conf_l10_min": conf_l10_min,
        "conf_h2h_good": conf_h2h_good,
        "conf_low_max": conf_low_max,
    }
    st.session_state["preset"] = preset_name.strip()

NBA_TEAMS = [
    "", "ATL", "BOS", "BKN", "CHA", "CHI", "CLE", "DAL",
    "DEN", "DET", "GSW", "HOU", "IND", "LAC", "LAL",
    "MEM", "MIA", "MIL", "MIN", "NOP", "NYK", "OKC",
    "ORL", "PHI", "PHX", "POR", "SAC", "SAS", "TOR",
    "UTA", "WAS",
]

SPORT_OPTIONS = ["nba", "mlb", "nfl", "soccer", "nhl"]
PROP_OPTIONS_BY_SPORT = {
    "nba": ["points", "rebounds", "assists", "points+assists", "points+rebounds", "rebounds+assists", "pra"],
    "mlb": ["hits", "runs", "rbis", "home_runs", "total_bases", "strikeouts"],
    "nfl": ["passing_yards", "rushing_yards", "receiving_yards", "receptions", "touchdowns"],
    "soccer": ["goals", "assists", "shots", "shots_on_target", "passes"],
    "nhl": ["goals", "assists", "points", "shots", "saves"],
}
OPPONENTS_BY_SPORT = {
    "nba": NBA_TEAMS,
    "mlb": ["", "ARI", "ATL", "BAL", "BOS", "CHC", "CIN", "CLE", "COL", "CWS", "DET", "HOU", "KC", "LAA", "LAD", "MIA", "MIL", "MIN", "NYM", "NYY", "OAK", "PHI", "PIT", "SD", "SEA", "SF", "STL", "TB", "TEX", "TOR", "WSH"],
    "nfl": ["", "ARI", "ATL", "BAL", "BUF", "CAR", "CHI", "CIN", "CLE", "DAL", "DEN", "DET", "GB", "HOU", "IND", "JAX", "KC", "LAC", "LAR", "LV", "MIA", "MIN", "NE", "NO", "NYG", "NYJ", "PHI", "PIT", "SEA", "SF", "TB", "TEN", "WAS"],
    "soccer": ["", "ARS", "AVL", "BAR", "BAY", "CHE", "DOR", "INT", "JUV", "LIV", "MCI", "MUN", "NEW", "PSG", "RMA", "ROM", "TOT"],
    "nhl": ["", "ANA", "BOS", "BUF", "CAR", "CBJ", "CGY", "CHI", "COL", "DAL", "DET", "EDM", "FLA", "LAK", "MIN", "MTL", "NJD", "NSH", "NYI", "NYR", "OTT", "PHI", "PIT", "SEA", "SJS", "STL", "TBL", "TOR", "UTA", "VAN", "VGK", "WPG", "WSH"],
}

sport = st.selectbox("Sport", SPORT_OPTIONS, index=0).lower()
available_props = PROP_OPTIONS_BY_SPORT[sport]
player = st.text_input("Player Name", "LeBron James")
prop = st.selectbox(
    "Prop Type",
    available_props,
)
line = st.number_input("Prop Line", value=25.5)
opponent = st.selectbox("Opponent (for H2H & DvP)", OPPONENTS_BY_SPORT[sport])

if "history" not in st.session_state:
    st.session_state["history"] = []
compare_mode = st.checkbox("Compare multiple props/lines", value=False)
if compare_mode:
    compare_props = st.multiselect(
        "Props to compare",
        available_props,
        default=[prop],
    )
    compare_lines_raw = st.text_input("Lines to compare (comma separated)", "20.5, 25.5")
else:
    compare_props = [prop]
    compare_lines_raw = str(line)

if st.button("Evaluate"):
    with st.spinner("Running analysis..."):
        lines = []
        for part in compare_lines_raw.split(","):
            part = part.strip()
            if not part:
                continue
            try:
                lines.append(float(part))
            except ValueError:
                pass
        if not lines:
            lines = [line]

        results = []
        errors = []
        for p in compare_props:
            for ln in lines:
                params = {
                    "player": player,
                    "sport": sport,
                    "prop": p,
                    "line": ln,
                    "opponent": opponent,
                    "season_type": season_type,
                    "window_1": window_1,
                    "window_2": window_2,
                    "hit_operator": hit_operator,
                    "conf_l5_min": conf_l5_min,
                    "conf_l10_min": conf_l10_min,
                    "conf_h2h_good": conf_h2h_good,
                    "conf_low_max": conf_low_max,
                    "include_injury": str(include_injury).lower(),
                }
                if offered_odds_input.strip():
                    try:
                        params["offered_odds"] = int(offered_odds_input.strip())
                    except ValueError:
                        st.error("Offered Odds must be an integer like -110 or 120.")
                        st.stop()
                resp = None
                req_error = ""
                for attempt in range(2):
                    try:
                        resp = requests.get(BACKEND_URL, params=params, timeout=30)
                        break
                    except ReadTimeout:
                        if attempt == 0:
                            req_error = "Backend request timed out. Retrying once."
                            continue
                        req_error = "Backend timed out twice."
                    except RequestException as exc:
                        req_error = f"Backend request failed: {exc}"
                        break
                if resp is None:
                    errors.append(f"{p} @ {ln}: {req_error or 'Request failed.'}")
                    continue
                try:
                    res = resp.json()
                except Exception:
                    errors.append(f"{p} @ {ln}: Backend returned an invalid response.")
                    continue
                if "error" in res:
                    errors.append(f"{p} @ {ln}: {res['error']}")
                    continue
                if "confidence" not in res:
                    errors.append(f"{p} @ {ln}: Backend response missing expected fields.")
                    continue
                results.append(res)

        if errors:
            st.warning("Some requests failed:\n- " + "\n- ".join(errors[:5]))
        if not results:
            st.error("No results returned from backend. Check backend connectivity and provider status.")
            st.stop()
        best = max(results, key=lambda r: r["confidence"])

        st.subheader(f"Recommendation: {best['recommendation']}")
        opp_label = f" vs {opponent}" if opponent else ""
        st.write(
            f"Best pick ({best.get('sport', sport).upper()}): {best['player']} {best['prop']} at line {best['line']}{opp_label} "
            f"with {best['confidence']}% confidence."
        )

        c1, c2, c3 = st.columns(3)
        c1.metric("Confidence", f"{best['confidence']}%")
        c2.metric("Last 5 Hit Rate", f"{best['last_5_hit_rate']}%")
        c3.metric("Last 10 Hit Rate", f"{best['last_10_hit_rate']}%")

        c4, c5, c6 = st.columns(3)
        c4.metric("H2H Hit Rate", f"{best['h2h_hit_rate']}%")
        c5.metric(best.get("projection_label", "Minutes Projection"), f"{best['minutes_proj']}")
        c6.metric("Defense vs Position", best["dvp"])

        c7, c8, c9 = st.columns(3)
        c7.metric("Projected Prob", f"{best.get('projected_probability', 0)}%")
        c8.metric("Edge", f"{best.get('edge_pct', 'n/a')}%")
        c9.metric("Data Source", best.get("data_source", "unknown"))

        if best.get("fallback_used"):
            st.warning("Fallback model used because live provider data was unavailable.")
        injury_ctx = best.get("injury_context", {})
        if injury_ctx and injury_ctx.get("status") != "not_requested":
            st.info(f"Injury context: {injury_ctx.get('status')} - {injury_ctx.get('detail', '')}")

        st.progress(best["confidence"] / 100)

        st.subheader("Why This Recommendation")
        for reason in best["reasons"]:
            st.write(f"- {reason}")

        chart_data = [
            {"Window": "Last 5", "Hit Rate (%)": float(best["last_5_hit_rate"]), "Avg Stat": float(best["last_5_avg_stat"]),
             "Low": float(best["last_5_ci"][0]), "High": float(best["last_5_ci"][1])},
            {"Window": "Last 10", "Hit Rate (%)": float(best["last_10_hit_rate"]), "Avg Stat": float(best["last_10_avg_stat"]),
             "Low": float(best["last_10_ci"][0]), "High": float(best["last_10_ci"][1])},
            {"Window": "H2H", "Hit Rate (%)": float(best["h2h_hit_rate"]), "Avg Stat": float(best["h2h_avg_stat"]),
             "Low": float(best["h2h_ci"][0]), "High": float(best["h2h_ci"][1])},
        ]

        render_hit_rate_bars(chart_data)

        st.subheader("Comparison Table")
        rows = []
        for r in results:
            rows.append({
                "Prop": r["prop"],
                "Line": r["line"],
                "Confidence": r["confidence"],
                "Rec": r["recommendation"],
                "L5 Hit%": r["last_5_hit_rate"],
                "L10 Hit%": r["last_10_hit_rate"],
                "H2H Hit%": r["h2h_hit_rate"],
                "L5 CI": f"{r['last_5_ci'][0]}-{r['last_5_ci'][1]}",
                "L10 CI": f"{r['last_10_ci'][0]}-{r['last_10_ci'][1]}",
                "H2H CI": f"{r['h2h_ci'][0]}-{r['h2h_ci'][1]}",
            })
        rows = sorted(rows, key=lambda x: x["Confidence"], reverse=True)
        columns = ["Prop", "Line", "Confidence", "Rec", "L5 Hit%", "L10 Hit%", "H2H Hit%", "L5 CI", "L10 CI", "H2H CI"]
        render_table_html(rows, columns)

        st.subheader("Top Picks")
        top_n = 1 if len(rows) <= 1 else st.slider("Number of picks", 1, min(10, len(rows)), min(3, len(rows)))
        render_table_html(rows[:top_n], columns)
