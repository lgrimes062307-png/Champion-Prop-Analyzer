import streamlit as st
import requests
import pandas as pd
import altair as alt
import datetime
import os
import urllib.parse

st.set_page_config(layout="wide")
st.title("NBA Prop Analyzer")


def render_table_html(df: pd.DataFrame):
    # Avoid Arrow serialization (LargeUtf8) by rendering as HTML
    st.markdown(df.to_html(index=False, escape=True), unsafe_allow_html=True)


BACKEND_URL = st.sidebar.text_input("Backend URL", "http://localhost:8000/analyze")
SUBSCRIBE_URL = st.sidebar.text_input("Subscribe URL Endpoint", "http://localhost:8000/create-checkout-session")
STATUS_URL = st.sidebar.text_input("Subscription Status Endpoint", "http://localhost:8000/subscription-status")
PORTAL_URL = st.sidebar.text_input("Manage Subscription Endpoint", "http://localhost:8000/create-portal-session")

DISCORD_CLIENT_ID = os.getenv("DISCORD_CLIENT_ID", "")
DISCORD_CLIENT_SECRET = os.getenv("DISCORD_CLIENT_SECRET", "")
DISCORD_REDIRECT_URI = os.getenv("DISCORD_REDIRECT_URI", "")

DEFAULT_PRESETS = {
    "Default": {
        "season_type": "Regular Season",
        "window_1": 5,
        "window_2": 10,
        "hit_operator": "gt",
        "conf_l5_min": 50,
        "conf_l10_min": 50,
        "conf_h2h_good": 60,
        "conf_low_max": 40,
    },
    "Aggressive": {
        "season_type": "Regular Season",
        "window_1": 5,
        "window_2": 10,
        "hit_operator": "gt",
        "conf_l5_min": 55,
        "conf_l10_min": 55,
        "conf_h2h_good": 65,
        "conf_low_max": 35,
    },
    "Conservative": {
        "season_type": "Regular Season",
        "window_1": 7,
        "window_2": 15,
        "hit_operator": "gte",
        "conf_l5_min": 60,
        "conf_l10_min": 60,
        "conf_h2h_good": 70,
        "conf_low_max": 45,
    },
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

if "season_type" not in st.session_state:
    st.session_state["season_type"] = all_presets[preset]["season_type"]
if "window_1" not in st.session_state:
    st.session_state["window_1"] = all_presets[preset]["window_1"]
if "window_2" not in st.session_state:
    st.session_state["window_2"] = all_presets[preset]["window_2"]
if "hit_operator" not in st.session_state:
    st.session_state["hit_operator"] = all_presets[preset]["hit_operator"]
if "conf_l5_min" not in st.session_state:
    st.session_state["conf_l5_min"] = all_presets[preset]["conf_l5_min"]
if "conf_l10_min" not in st.session_state:
    st.session_state["conf_l10_min"] = all_presets[preset]["conf_l10_min"]
if "conf_h2h_good" not in st.session_state:
    st.session_state["conf_h2h_good"] = all_presets[preset]["conf_h2h_good"]
if "conf_low_max" not in st.session_state:
    st.session_state["conf_low_max"] = all_presets[preset]["conf_low_max"]

season_type = st.sidebar.selectbox(
    "Season Type",
    ["Regular Season", "Playoffs"],
    index=["Regular Season", "Playoffs"].index(st.session_state["season_type"]),
    key="season_type",
)
window_1 = st.sidebar.slider(
    "Last 5 Games (adjustable)",
    min_value=1,
    max_value=30,
    value=st.session_state["window_1"],
    key="window_1",
)
window_2 = st.sidebar.slider(
    "Last 10 Games (adjustable)",
    min_value=1,
    max_value=50,
    value=st.session_state["window_2"],
    key="window_2",
)
hit_operator = st.sidebar.selectbox(
    "Hit Operator",
    ["gt", "gte"],
    index=["gt", "gte"].index(st.session_state["hit_operator"]),
    key="hit_operator",
)

st.sidebar.markdown("### Model Tuning")
conf_l5_min = st.sidebar.slider("Conf L5 Min", min_value=0, max_value=100, value=st.session_state["conf_l5_min"], key="conf_l5_min")
conf_l10_min = st.sidebar.slider("Conf L10 Min", min_value=0, max_value=100, value=st.session_state["conf_l10_min"], key="conf_l10_min")
conf_h2h_good = st.sidebar.slider("Conf H2H Good", min_value=0, max_value=100, value=st.session_state["conf_h2h_good"], key="conf_h2h_good")
conf_low_max = st.sidebar.slider("Conf Low Max", min_value=0, max_value=100, value=st.session_state["conf_low_max"], key="conf_low_max")

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
    "UTA", "WAS"
]

player = st.text_input("Player Name", "LeBron James")

prop = st.selectbox(
    "Prop Type",
    [
        "points",
        "rebounds",
        "assists",
        "points+assists",
        "points+rebounds",
        "rebounds+assists",
        "pra"
    ]
)

line = st.number_input("Prop Line", value=25.5)
opponent = st.selectbox("Opponent (for H2H & DvP)", NBA_TEAMS)

st.subheader("Paid Access")
if "discord_user_id" not in st.session_state:
    st.session_state["discord_user_id"] = ""

query = st.experimental_get_query_params()
code = query.get("code", [None])[0]
if code and DISCORD_CLIENT_ID and DISCORD_CLIENT_SECRET and DISCORD_REDIRECT_URI:
    token_res = requests.post(
        "https://discord.com/api/oauth2/token",
        data={
            "client_id": DISCORD_CLIENT_ID,
            "client_secret": DISCORD_CLIENT_SECRET,
            "grant_type": "authorization_code",
            "code": code,
            "redirect_uri": DISCORD_REDIRECT_URI,
            "scope": "identify",
        },
        headers={"Content-Type": "application/x-www-form-urlencoded"},
    )
    if token_res.status_code == 200:
        token = token_res.json().get("access_token", "")
        user_res = requests.get(
            "https://discord.com/api/users/@me",
            headers={"Authorization": f"Bearer {token}"},
        )
        if user_res.status_code == 200:
            st.session_state["discord_user_id"] = user_res.json().get("id", "")
    st.experimental_set_query_params()

if DISCORD_CLIENT_ID and DISCORD_REDIRECT_URI:
    params = {
        "client_id": DISCORD_CLIENT_ID,
        "redirect_uri": DISCORD_REDIRECT_URI,
        "response_type": "code",
        "scope": "identify",
    }
    auth_url = "https://discord.com/api/oauth2/authorize?" + urllib.parse.urlencode(params)
    st.markdown(f"[Connect Discord]({auth_url})")
else:
    st.caption("Set DISCORD_CLIENT_ID and DISCORD_REDIRECT_URI to enable one-click Discord connect.")

discord_user_id = st.text_input("Discord User ID (for paid access)", st.session_state["discord_user_id"])
st.session_state["discord_user_id"] = discord_user_id

if discord_user_id:
    status_res = requests.get(STATUS_URL, params={"discord_user_id": discord_user_id}).json()
    is_active = status_res.get("active", False)
    st.write(f"Subscription status: {status_res.get('status', 'unknown')}")
else:
    is_active = False

if not is_active:
    st.info("Free preview: upgrade to unlock full analysis, comparisons, top picks, and history.")

if st.button("Generate Checkout Link"):
    if not discord_user_id:
        st.error("Please enter your Discord User ID.")
    else:
        checkout = requests.post(SUBSCRIBE_URL, params={"discord_user_id": discord_user_id}).json()
        if "url" in checkout:
            st.success("Checkout link generated.")
            st.write(checkout["url"])
        else:
            st.error(checkout.get("error", "Unable to create checkout session."))

if st.button("Manage Subscription"):
    if not discord_user_id:
        st.error("Please enter your Discord User ID.")
    else:
        portal = requests.post(PORTAL_URL, params={"discord_user_id": discord_user_id}).json()
        if "url" in portal:
            st.success("Manage subscription link generated.")
            st.write(portal["url"])
        else:
            st.error(portal.get("error", "Unable to create portal session."))

compare_mode = st.checkbox("Compare multiple props/lines", value=False, disabled=not is_active)
if compare_mode:
    compare_props = st.multiselect(
        "Props to compare",
        ["points", "rebounds", "assists", "points+assists", "points+rebounds", "rebounds+assists", "pra"],
        default=[prop],
    )
    compare_lines_raw = st.text_input("Lines to compare (comma separated)", "20.5, 25.5")
else:
    compare_props = [prop]
    compare_lines_raw = str(line)

if st.button("Evaluate"):
    with st.spinner("Running analysis..."):
        if not is_active:
            compare_props = [prop]
            compare_lines_raw = str(line)

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
        for p in compare_props:
            for ln in lines:
                resp = requests.get(
                    BACKEND_URL,
                    params={
                        "player": player,
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
                    },
                    timeout=20,
                )
                try:
                    res = resp.json()
                except Exception:
                    st.error("Backend returned an invalid response. Check your Backend URL and backend logs.")
                    st.stop()
                if "error" in res:
                    st.error(res["error"])
                    st.stop()
                if "confidence" not in res:
                    st.error("Backend response is missing expected fields. Check backend deployment.")
                    st.stop()
                results.append(res)

        if is_active:
            if "history" not in st.session_state:
                st.session_state["history"] = []
            for r in results:
                st.session_state["history"].append({
                    "time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "player": r["player"],
                    "prop": r["prop"],
                    "line": r["line"],
                    "confidence": r["confidence"],
                    "recommendation": r["recommendation"],
                    "last_5_hit_rate": r["last_5_hit_rate"],
                    "last_10_hit_rate": r["last_10_hit_rate"],
                    "h2h_hit_rate": r["h2h_hit_rate"],
                })

        if not results:
            st.error("No results returned from backend.")
            st.stop()
        best = max(results, key=lambda r: r["confidence"])

        st.subheader(f"Recommendation: {best['recommendation']}")
        opp_label = f" vs {opponent}" if opponent else ""
        st.write(
            f"Best pick: {best['player']} {best['prop']} at line {best['line']}{opp_label} "
            f"with {best['confidence']}% confidence."
        )

        c1, c2, c3 = st.columns(3)
        c1.metric("Confidence", f"{best['confidence']}%")
        c2.metric("Last 5 Hit Rate", f"{best['last_5_hit_rate']}%")
        c3.metric("Last 10 Hit Rate", f"{best['last_10_hit_rate']}%")

        c4, c5, c6 = st.columns(3)
        c4.metric("H2H Hit Rate", f"{best['h2h_hit_rate']}%")
        c5.metric("Minutes Projection", f"{best['minutes_proj']} min")
        c6.metric("Defense vs Position", best["dvp"])

        st.progress(best["confidence"] / 100)

        if is_active:
            st.subheader("Why This Recommendation")
            for reason in best["reasons"]:
                st.write(f"- {reason}")
        else:
            st.info("Free preview: upgrade to see full analysis, reasons, and comparisons.")

        chart_data = pd.DataFrame([
            {
                "Window": "Last 5",
                "Hit Rate (%)": best["last_5_hit_rate"],
                "Avg Stat": best["last_5_avg_stat"],
                "Low": best["last_5_ci"][0],
                "High": best["last_5_ci"][1],
            },
            {
                "Window": "Last 10",
                "Hit Rate (%)": best["last_10_hit_rate"],
                "Avg Stat": best["last_10_avg_stat"],
                "Low": best["last_10_ci"][0],
                "High": best["last_10_ci"][1],
            },
            {
                "Window": "H2H",
                "Hit Rate (%)": best["h2h_hit_rate"],
                "Avg Stat": best["h2h_avg_stat"],
                "Low": best["h2h_ci"][0],
                "High": best["h2h_ci"][1],
            },
        ])
        # Avoid Arrow LargeUtf8 serialization issues in Streamlit
        for col in ["Window"]:
            chart_data[col] = chart_data[col].astype(str)
        for col in ["Hit Rate (%)", "Avg Stat", "Low", "High"]:
            chart_data[col] = pd.to_numeric(chart_data[col], errors="coerce").fillna(0.0)

        st.subheader("Hit Rate Overview")
        bars = (
            alt.Chart(chart_data)
            .mark_bar()
            .encode(
                x=alt.X("Window:N", sort=["Last 5", "Last 10", "H2H"]),
                y=alt.Y("Hit Rate (%):Q", scale=alt.Scale(domain=[0, 100])),
                tooltip=[
                    alt.Tooltip("Window:N"),
                    alt.Tooltip("Hit Rate (%):Q", format=".2f"),
                    alt.Tooltip("Avg Stat:Q", format=".2f"),
                    alt.Tooltip("Low:Q", format=".2f"),
                    alt.Tooltip("High:Q", format=".2f"),
                ],
            )
        )
        if is_active:
            error = (
                alt.Chart(chart_data)
                .mark_errorbar()
                .encode(
                    x=alt.X("Window:N", sort=["Last 5", "Last 10", "H2H"]),
                    y=alt.Y("Low:Q"),
                    y2=alt.Y2("High:Q"),
                )
            )
            st.altair_chart(bars + error, use_container_width=True)
        else:
            st.altair_chart(bars, use_container_width=True)

        if is_active:
            st.subheader("Comparison Table")
            table = pd.DataFrame([
                {
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
                }
                for r in results
            ])
            # Force string columns to plain object dtype for Streamlit compatibility
            for col in table.columns:
                if table[col].dtype.name == "string":
                    table[col] = table[col].astype(str)
            table_sorted = table.sort_values("Confidence", ascending=False)
            render_table_html(table_sorted)

            st.subheader("Top Picks")
            if len(table_sorted) <= 1:
                render_table_html(table_sorted.head(1))
            else:
                top_n = st.slider(
                    "Number of picks",
                    min_value=1,
                    max_value=min(10, len(table_sorted)),
                    value=min(3, len(table_sorted)),
                )
                render_table_html(table_sorted.head(top_n))

            with st.expander("History"):
                if st.button("Clear History"):
                    st.session_state["history"] = []
                if st.session_state["history"]:
                    hist = pd.DataFrame(st.session_state["history"])
                    for col in hist.columns:
                        if hist[col].dtype.name == "string":
                            hist[col] = hist[col].astype(str)
                    render_table_html(hist)
                else:
                    st.write("No history yet.")



