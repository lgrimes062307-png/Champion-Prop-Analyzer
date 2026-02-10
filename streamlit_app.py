import streamlit as st
import requests
from requests.exceptions import ReadTimeout, RequestException
import datetime
import os
import urllib.parse
import extra_streamlit_components as stx

st.set_page_config(layout="wide")
st.title("NBA Prop Analyzer")

cookie_manager = stx.CookieManager(key="cookie_manager")


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


BACKEND_URL = "https://champion-prop-analyzer.onrender.com/analyze"
SUBSCRIBE_URL = "https://champion-prop-analyzer.onrender.com/create-checkout-session"
STATUS_URL = "https://champion-prop-analyzer.onrender.com/subscription-status"
PORTAL_URL = "https://champion-prop-analyzer.onrender.com/create-portal-session"
ADMIN_GRANT_URL = st.sidebar.text_input("Admin Grant Endpoint", "http://localhost:8000/grant-access")
ADMIN_REVOKE_URL = st.sidebar.text_input("Admin Revoke Endpoint", "http://localhost:8000/revoke-access")

DISCORD_CLIENT_ID = os.getenv("DISCORD_CLIENT_ID", "")
DISCORD_CLIENT_SECRET = os.getenv("DISCORD_CLIENT_SECRET", "")
DISCORD_REDIRECT_URI = os.getenv("DISCORD_REDIRECT_URI", "")

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

st.sidebar.markdown("### Admin Access")
admin_secret = st.sidebar.text_input("Admin Secret", type="password")
admin_user_id = st.sidebar.text_input("Discord User ID to grant/revoke")
col_a, col_b = st.sidebar.columns(2)
if col_a.button("Grant Access"):
    if not admin_secret or not admin_user_id:
        st.sidebar.error("Enter admin secret and user ID.")
    else:
        resp = requests.post(ADMIN_GRANT_URL, params={"discord_user_id": admin_user_id, "admin_secret": admin_secret}, timeout=30)
        try:
            data = resp.json()
        except Exception:
            st.sidebar.error(f"Grant failed (status {resp.status_code}).")
            st.sidebar.code(resp.text)
        else:
            if data.get("ok"):
                st.sidebar.success(f"Granted access to {admin_user_id}.")
            else:
                st.sidebar.error(data.get("detail", "Grant failed."))
if col_b.button("Revoke Access"):
    if not admin_secret or not admin_user_id:
        st.sidebar.error("Enter admin secret and user ID.")
    else:
        resp = requests.post(ADMIN_REVOKE_URL, params={"discord_user_id": admin_user_id, "admin_secret": admin_secret}, timeout=30)
        try:
            data = resp.json()
        except Exception:
            st.sidebar.error(f"Revoke failed (status {resp.status_code}).")
            st.sidebar.code(resp.text)
        else:
            if data.get("ok"):
                st.sidebar.success(f"Revoked access for {admin_user_id}.")
            else:
                st.sidebar.error(data.get("detail", "Revoke failed."))

NBA_TEAMS = [
    "", "ATL", "BOS", "BKN", "CHA", "CHI", "CLE", "DAL",
    "DEN", "DET", "GSW", "HOU", "IND", "LAC", "LAL",
    "MEM", "MIA", "MIL", "MIN", "NOP", "NYK", "OKC",
    "ORL", "PHI", "PHX", "POR", "SAC", "SAS", "TOR",
    "UTA", "WAS",
]

player = st.text_input("Player Name", "LeBron James")
prop = st.selectbox(
    "Prop Type",
    ["points", "rebounds", "assists", "points+assists", "points+rebounds", "rebounds+assists", "pra"],
)
line = st.number_input("Prop Line", value=25.5)
opponent = st.selectbox("Opponent (for H2H & DvP)", NBA_TEAMS)

st.subheader("Paid Access")
if "discord_user_id" not in st.session_state:
    st.session_state["discord_user_id"] = ""
cookie_discord_id = cookie_manager.get("discord_user_id")
if not st.session_state["discord_user_id"] and cookie_discord_id:
    st.session_state["discord_user_id"] = cookie_discord_id
if "free_eval_date" not in st.session_state:
    st.session_state["free_eval_date"] = ""
if "free_eval_count" not in st.session_state:
    st.session_state["free_eval_count"] = 0
if "history" not in st.session_state:
    st.session_state["history"] = []

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
            if st.session_state["discord_user_id"]:
                cookie_manager.set("discord_user_id", st.session_state["discord_user_id"])
    st.experimental_set_query_params()

if DISCORD_CLIENT_ID and DISCORD_REDIRECT_URI:
    params = {"client_id": DISCORD_CLIENT_ID, "redirect_uri": DISCORD_REDIRECT_URI, "response_type": "code", "scope": "identify"}
    auth_url = "https://discord.com/api/oauth2/authorize?" + urllib.parse.urlencode(params)
    st.markdown(f"[Connect Discord]({auth_url})")
else:
    st.caption("Set DISCORD_CLIENT_ID and DISCORD_REDIRECT_URI to enable one-click Discord connect.")

discord_user_id = st.text_input("Discord User ID (for paid access)", st.session_state["discord_user_id"])
st.session_state["discord_user_id"] = discord_user_id
if discord_user_id:
    cookie_manager.set("discord_user_id", discord_user_id)

if discord_user_id:
    status_res = requests.get(STATUS_URL, params={"discord_user_id": discord_user_id}).json()
    is_active = status_res.get("active", False)
    st.write(f"Subscription status: {status_res.get('status', 'unknown')}")
else:
    is_active = False

if not is_active:
    today = datetime.date.today().isoformat()
    if st.session_state["free_eval_date"] != today:
        st.session_state["free_eval_date"] = today
        st.session_state["free_eval_count"] = 0
    remaining = max(0, 1 - st.session_state["free_eval_count"])
    st.info(f"Free preview: {remaining} evaluation left today. Upgrade to unlock full analysis, comparisons, top picks, and history.")

if st.button("Generate Checkout Link"):
    if not discord_user_id:
        st.error("Please enter your Discord User ID.")
    else:
        resp = requests.post(SUBSCRIBE_URL, params={"discord_user_id": discord_user_id}, timeout=30)
        try:
            checkout = resp.json()
        except Exception:
            st.error(f"Checkout endpoint returned non-JSON (status {resp.status_code}).")
            st.code(resp.text)
            st.stop()
        if "url" in checkout:
            st.success("Checkout link generated.")
            st.write(checkout["url"])
        else:
            st.error(checkout.get("error", "Unable to create checkout session."))

if st.button("Manage Subscription"):
    if not discord_user_id:
        st.error("Please enter your Discord User ID.")
    else:
        resp = requests.post(PORTAL_URL, params={"discord_user_id": discord_user_id}, timeout=30)
        try:
            portal = resp.json()
        except Exception:
            st.error(f"Portal endpoint returned non-JSON (status {resp.status_code}).")
            st.code(resp.text)
            st.stop()
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
            today = datetime.date.today().isoformat()
            if st.session_state["free_eval_date"] != today:
                st.session_state["free_eval_date"] = today
                st.session_state["free_eval_count"] = 0
            if st.session_state["free_eval_count"] >= 1:
                st.error("Free limit reached for today. Please upgrade to continue.")
                st.stop()
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
                params = {
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
                }
                resp = None
                for attempt in range(2):
                    try:
                        resp = requests.get(BACKEND_URL, params=params, timeout=60)
                        break
                    except ReadTimeout:
                        if attempt == 0:
                            continue
                        st.error("Backend timed out twice. Please try again in a moment.")
                        st.stop()
                    except RequestException as exc:
                        st.error(f"Backend request failed: {exc}")
                        st.stop()
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

        if not results:
            st.error("No results returned from backend.")
            st.stop()
        if not is_active:
            st.session_state["free_eval_count"] += 1

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

        chart_data = [
            {"Window": "Last 5", "Hit Rate (%)": float(best["last_5_hit_rate"]), "Avg Stat": float(best["last_5_avg_stat"]),
             "Low": float(best["last_5_ci"][0]), "High": float(best["last_5_ci"][1])},
            {"Window": "Last 10", "Hit Rate (%)": float(best["last_10_hit_rate"]), "Avg Stat": float(best["last_10_avg_stat"]),
             "Low": float(best["last_10_ci"][0]), "High": float(best["last_10_ci"][1])},
            {"Window": "H2H", "Hit Rate (%)": float(best["h2h_hit_rate"]), "Avg Stat": float(best["h2h_avg_stat"]),
             "Low": float(best["h2h_ci"][0]), "High": float(best["h2h_ci"][1])},
        ]

        render_hit_rate_bars(chart_data)

        if is_active:
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

            with st.expander("History"):
                if st.button("Clear History"):
                    st.session_state["history"] = []
                if st.session_state["history"]:
                    render_table_html(st.session_state["history"], list(st.session_state["history"][0].keys()))
                else:
                    st.write("No history yet.")
