import os
import time
from typing import Any, Dict, List, Optional, Tuple

import requests
import streamlit as st
from requests.exceptions import ReadTimeout, RequestException

try:
    import pandas as pd
except Exception:  # pragma: no cover
    pd = None


st.set_page_config(layout="wide", page_title="Multi-Sport Prop Analyzer")
st.title("Multi-Sport Prop Analyzer")


# ---------------- Config ---------------- #

BACKEND_BASE_URL = os.getenv("BACKEND_BASE_URL", "https://champion-prop-analyzer.onrender.com").rstrip("/")
BACKEND_URL = os.getenv("BACKEND_ANALYZE_URL", f"{BACKEND_BASE_URL}/analyze")
BACKEND_V2_URL = os.getenv("BACKEND_ANALYZE_V2_URL", f"{BACKEND_BASE_URL}/v2/analyze")
BACKEND_USE_V2 = os.getenv("BACKEND_USE_V2", "true").strip().lower() in ("1", "true", "yes", "on")
BACKEND_RETRIES = max(1, int(os.getenv("BACKEND_RETRIES", "3")))
BACKEND_CONNECT_TIMEOUT_SECONDS = max(1, int(os.getenv("BACKEND_CONNECT_TIMEOUT_SECONDS", "10")))
BACKEND_READ_TIMEOUT_SECONDS = max(5, int(os.getenv("BACKEND_READ_TIMEOUT_SECONDS", "75")))

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

SPORT_OPTIONS = ["nba", "mlb", "nfl", "soccer", "nhl"]
NBA_TEAMS = [
    "",
    "ATL",
    "BOS",
    "BKN",
    "CHA",
    "CHI",
    "CLE",
    "DAL",
    "DEN",
    "DET",
    "GSW",
    "HOU",
    "IND",
    "LAC",
    "LAL",
    "MEM",
    "MIA",
    "MIL",
    "MIN",
    "NOP",
    "NYK",
    "OKC",
    "ORL",
    "PHI",
    "PHX",
    "POR",
    "SAC",
    "SAS",
    "TOR",
    "UTA",
    "WAS",
]
PROP_OPTIONS_BY_SPORT = {
    "nba": ["points", "rebounds", "assists", "points+assists", "points+rebounds", "rebounds+assists", "pra"],
    "mlb": ["hits", "runs", "rbis", "home_runs", "total_bases", "strikeouts"],
    "nfl": ["passing_yards", "rushing_yards", "receiving_yards", "receptions", "touchdowns"],
    "soccer": ["goals", "assists", "shots", "shots_on_target", "passes"],
    "nhl": ["goals", "assists", "points", "shots", "saves"],
}
OPPONENTS_BY_SPORT = {
    "nba": NBA_TEAMS,
    "mlb": [
        "",
        "ARI",
        "ATL",
        "BAL",
        "BOS",
        "CHC",
        "CIN",
        "CLE",
        "COL",
        "CWS",
        "DET",
        "HOU",
        "KC",
        "LAA",
        "LAD",
        "MIA",
        "MIL",
        "MIN",
        "NYM",
        "NYY",
        "OAK",
        "PHI",
        "PIT",
        "SD",
        "SEA",
        "SF",
        "STL",
        "TB",
        "TEX",
        "TOR",
        "WSH",
    ],
    "nfl": [
        "",
        "ARI",
        "ATL",
        "BAL",
        "BUF",
        "CAR",
        "CHI",
        "CIN",
        "CLE",
        "DAL",
        "DEN",
        "DET",
        "GB",
        "HOU",
        "IND",
        "JAX",
        "KC",
        "LAC",
        "LAR",
        "LV",
        "MIA",
        "MIN",
        "NE",
        "NO",
        "NYG",
        "NYJ",
        "PHI",
        "PIT",
        "SEA",
        "SF",
        "TB",
        "TEN",
        "WAS",
    ],
    "soccer": ["", "ARS", "AVL", "BAR", "BAY", "CHE", "DOR", "INT", "JUV", "LIV", "MCI", "MUN", "NEW", "PSG", "RMA", "ROM", "TOT"],
    "nhl": [
        "",
        "ANA",
        "BOS",
        "BUF",
        "CAR",
        "CBJ",
        "CGY",
        "CHI",
        "COL",
        "DAL",
        "DET",
        "EDM",
        "FLA",
        "LAK",
        "MIN",
        "MTL",
        "NJD",
        "NSH",
        "NYI",
        "NYR",
        "OTT",
        "PHI",
        "PIT",
        "SEA",
        "SJS",
        "STL",
        "TBL",
        "TOR",
        "UTA",
        "VAN",
        "VGK",
        "WPG",
        "WSH",
    ],
}


# ---------------- Helpers ---------------- #

def _history_add(result: Dict[str, Any]) -> None:
    if "history" not in st.session_state:
        st.session_state["history"] = []
    st.session_state["history"].append(
        {
            "time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "sport": result.get("sport", "").upper(),
            "player": result.get("player", ""),
            "prop": result.get("prop", ""),
            "line": result.get("line"),
            "confidence": result.get("confidence"),
            "recommendation": result.get("recommendation"),
            "projected_probability": result.get("projected_probability"),
            "data_source": result.get("data_source"),
            "fallback_used": result.get("fallback_used"),
            "pick_id": result.get("pick_id"),
        }
    )


def _to_df(rows: List[Dict[str, Any]]):
    if pd is None:
        return rows
    return pd.DataFrame(rows)


def _backend_request(method: str, url: str, *, params=None, json_body=None) -> Tuple[Optional[Dict[str, Any]], str]:
    err = ""
    for attempt in range(BACKEND_RETRIES):
        try:
            if method == "GET":
                resp = requests.get(
                    url,
                    params=params,
                    timeout=(BACKEND_CONNECT_TIMEOUT_SECONDS, BACKEND_READ_TIMEOUT_SECONDS),
                )
            else:
                resp = requests.post(
                    url,
                    json=json_body,
                    timeout=(BACKEND_CONNECT_TIMEOUT_SECONDS, BACKEND_READ_TIMEOUT_SECONDS),
                )

            if resp.status_code >= 400:
                detail = ""
                try:
                    body = resp.json()
                    detail = body.get("detail") or body.get("error") or ""
                except Exception:
                    detail = (resp.text or "").strip()[:200]
                if 500 <= resp.status_code < 600 and attempt < BACKEND_RETRIES - 1:
                    err = f"HTTP {resp.status_code}. Retrying."
                    time.sleep(1.1 * (attempt + 1))
                    continue
                return None, f"HTTP {resp.status_code}" + (f": {detail}" if detail else "")

            try:
                return resp.json(), ""
            except Exception:
                return None, "Backend returned invalid JSON."
        except ReadTimeout:
            err = "Request timed out."
            if attempt < BACKEND_RETRIES - 1:
                time.sleep(1.1 * (attempt + 1))
                continue
            return None, f"Timed out after {BACKEND_RETRIES} attempts."
        except RequestException as exc:
            return None, f"Request failed: {exc}"
    return None, err or "Request failed."


def _analyze(payload: Dict[str, Any]) -> Tuple[Optional[Dict[str, Any]], str]:
    if BACKEND_USE_V2:
        raw, err = _backend_request("POST", BACKEND_V2_URL, json_body=payload)
        if raw is None:
            return None, err
        if not raw.get("ok"):
            e = raw.get("error") or {}
            return None, e.get("message") or "Unknown v2 error"
        return raw.get("data") or {}, ""

    raw, err = _backend_request("GET", BACKEND_URL, params=payload)
    if raw is None:
        return None, err
    if "error" in raw:
        return None, str(raw["error"])
    return raw, ""


def _health_data() -> Tuple[Optional[Dict[str, Any]], str]:
    return _backend_request("GET", f"{BACKEND_BASE_URL}/health")


def _performance_data(days: int, sport: str) -> Tuple[Optional[Dict[str, Any]], str]:
    params = {"days": days}
    if sport:
        params["sport"] = sport
    return _backend_request("GET", f"{BACKEND_BASE_URL}/performance", params=params)


def _picks_data(days: int, sport: str, limit: int) -> Tuple[Optional[Dict[str, Any]], str]:
    params = {"days": days, "limit": limit}
    if sport:
        params["sport"] = sport
    return _backend_request("GET", f"{BACKEND_BASE_URL}/picks", params=params)


def _odds_edge_data(sport: str, market: str, bookmaker: str) -> Tuple[Optional[Dict[str, Any]], str]:
    params = {"sport": sport, "market": market, "bookmaker": bookmaker}
    return _backend_request("GET", f"{BACKEND_BASE_URL}/odds-edge", params=params)


# ---------------- Sidebar ---------------- #

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
    selected = all_presets[preset]
    for k, v in selected.items():
        st.session_state[k] = v
    st.session_state["last_preset"] = preset

for key in ["season_type", "window_1", "window_2", "hit_operator", "conf_l5_min", "conf_l10_min", "conf_h2h_good", "conf_low_max"]:
    if key not in st.session_state:
        st.session_state[key] = all_presets[preset][key]

season_type = st.sidebar.selectbox("Season Type", ["Regular Season", "Playoffs"], key="season_type")
window_1 = st.sidebar.slider("Window 1", 1, 30, st.session_state["window_1"], key="window_1")
window_2 = st.sidebar.slider("Window 2", 1, 50, st.session_state["window_2"], key="window_2")
hit_operator = st.sidebar.selectbox("Hit Operator", ["gt", "gte"], key="hit_operator")
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

st.sidebar.markdown("### Backend")
st.sidebar.caption(f"Base: {BACKEND_BASE_URL}")
st.sidebar.caption(f"Mode: {'v2 POST' if BACKEND_USE_V2 else 'v1 GET'}")


# ---------------- Tabs ---------------- #

tab_analyze, tab_health, tab_perf, tab_picks, tab_odds = st.tabs(["Analyze", "Health", "Performance", "Picks", "Odds Edge"])

with tab_analyze:
    c1, c2 = st.columns([1, 1])
    with c1:
        sport = st.selectbox("Sport", SPORT_OPTIONS, index=0).lower()
        player = st.text_input("Player Name", "LeBron James")
        prop = st.selectbox("Prop Type", PROP_OPTIONS_BY_SPORT[sport])
    with c2:
        line = st.number_input("Prop Line", value=25.5)
        opponent = st.selectbox("Opponent (for H2H & DvP)", OPPONENTS_BY_SPORT[sport])
        compare_mode = st.checkbox("Compare multiple props/lines", value=False)

    if compare_mode:
        compare_props = st.multiselect("Props to compare", PROP_OPTIONS_BY_SPORT[sport], default=[prop])
        compare_lines_raw = st.text_input("Lines to compare (comma separated)", "20.5, 25.5")
    else:
        compare_props = [prop]
        compare_lines_raw = str(line)

    if st.button("Evaluate", type="primary"):
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

            results: List[Dict[str, Any]] = []
            errors: List[str] = []

            for p in compare_props:
                for ln in lines:
                    payload: Dict[str, Any] = {
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
                        "include_injury": bool(include_injury),
                    }

                    if offered_odds_input.strip():
                        try:
                            payload["offered_odds"] = int(offered_odds_input.strip())
                        except ValueError:
                            st.error("Offered Odds must be an integer like -110 or 120.")
                            st.stop()

                    res, err = _analyze(payload)
                    if err:
                        errors.append(f"{p} @ {ln}: {err}")
                        continue
                    if not res or "confidence" not in res:
                        errors.append(f"{p} @ {ln}: Missing expected fields.")
                        continue
                    results.append(res)
                    _history_add(res)

            if errors:
                st.warning("Some requests failed:\n- " + "\n- ".join(errors[:8]))
            if not results:
                st.error("No results returned from backend.")
                st.stop()

            best = max(results, key=lambda r: float(r.get("confidence", 0)))

            st.subheader(f"Recommendation: {best.get('recommendation', 'N/A')}")
            st.write(
                f"Best pick ({best.get('sport', sport).upper()}): {best.get('player')} {best.get('prop')} at line {best.get('line')}"
                f" with {best.get('confidence')}% confidence."
            )

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Confidence", f"{best.get('confidence', 0)}%")
            m2.metric("Projected Prob", f"{best.get('projected_probability', 0)}%")
            m3.metric("Edge", f"{best.get('edge_pct', 'n/a')}%")
            m4.metric("Data Source", str(best.get("data_source", "unknown")))

            if best.get("fallback_used"):
                st.warning("Fallback model used because live data was unavailable.")

            injury_ctx = best.get("injury_context", {}) or {}
            if injury_ctx.get("status") and injury_ctx.get("status") != "not_requested":
                st.info(f"Injury: {injury_ctx.get('status')} | {injury_ctx.get('detail', '')}")

            reasons = best.get("reasons", []) or []
            if reasons:
                with st.expander("Why this recommendation", expanded=True):
                    for reason in reasons:
                        st.write(f"- {reason}")

            chart_rows = [
                {"window": "Last 5", "hit_rate": float(best.get("last_5_hit_rate", 0)), "avg_stat": float(best.get("last_5_avg_stat", 0))},
                {"window": "Last 10", "hit_rate": float(best.get("last_10_hit_rate", 0)), "avg_stat": float(best.get("last_10_avg_stat", 0))},
                {"window": "H2H", "hit_rate": float(best.get("h2h_hit_rate", 0)), "avg_stat": float(best.get("h2h_avg_stat", 0))},
            ]
            st.markdown("**Hit Rate by Window**")
            df_chart = _to_df(chart_rows)
            if pd is not None:
                st.bar_chart(df_chart.set_index("window")["hit_rate"])
                st.bar_chart(df_chart.set_index("window")["avg_stat"])
            else:
                st.dataframe(chart_rows, use_container_width=True)

            compare_rows = []
            for r in sorted(results, key=lambda x: float(x.get("confidence", 0)), reverse=True):
                compare_rows.append(
                    {
                        "sport": str(r.get("sport", "")).upper(),
                        "player": r.get("player"),
                        "prop": r.get("prop"),
                        "line": r.get("line"),
                        "confidence": r.get("confidence"),
                        "recommendation": r.get("recommendation"),
                        "l5_hit": r.get("last_5_hit_rate"),
                        "l10_hit": r.get("last_10_hit_rate"),
                        "h2h_hit": r.get("h2h_hit_rate"),
                        "expected_stat": r.get("expected_stat"),
                        "data_source": r.get("data_source"),
                        "fallback_used": r.get("fallback_used"),
                    }
                )

            st.markdown("**Comparison Results**")
            st.dataframe(_to_df(compare_rows), use_container_width=True)

    with st.expander("Local Session History"):
        if st.button("Clear History"):
            st.session_state["history"] = []
        history = st.session_state.get("history", [])
        if history:
            st.dataframe(_to_df(history), use_container_width=True)
        else:
            st.write("No history yet.")

with tab_health:
    st.subheader("Backend Health")
    if st.button("Refresh Health"):
        st.rerun()
    admin_secret_for_reset = st.text_input("Admin Secret (for runtime reset)", value="", type="password")
    if st.button("Reset Runtime State"):
        if not admin_secret_for_reset.strip():
            st.error("Admin secret is required.")
        else:
            reset_resp, reset_err = _backend_request(
                "POST",
                f"{BACKEND_BASE_URL}/admin/reset-runtime",
                params={"admin_secret": admin_secret_for_reset.strip()},
            )
            if reset_err:
                st.error(f"Reset failed: {reset_err}")
            else:
                st.success("Runtime state reset.")
                st.json(reset_resp)
    health, err = _health_data()
    if err:
        st.error(err)
    elif not health:
        st.warning("No health payload returned.")
    else:
        top1, top2, top3 = st.columns(3)
        top1.metric("API OK", str(health.get("ok", False)))
        top2.metric("Model Version", str(health.get("model_version", "unknown")))
        top3.metric("Build", str(health.get("app_build", "n/a")))

        provider_mode = health.get("provider_mode", {})
        if provider_mode:
            st.markdown("**Provider Mode**")
            st.dataframe(_to_df([provider_mode]), use_container_width=True)

        providers = []
        for name, state in (health.get("providers") or {}).items():
            providers.append(
                {
                    "provider": name,
                    "open": state.get("open"),
                    "failures": state.get("failures"),
                    "last_error": state.get("last_error"),
                    "last_success_at": state.get("last_success_at"),
                }
            )
        st.markdown("**Providers**")
        st.dataframe(_to_df(providers), use_container_width=True)

        cache = health.get("cache") or {}
        st.markdown("**Cache**")
        st.dataframe(_to_df([cache]), use_container_width=True)

with tab_perf:
    st.subheader("Performance")
    perf_days = st.slider("Days", 1, 365, 30)
    perf_sport = st.selectbox("Sport Filter", [""] + SPORT_OPTIONS, index=0)

    if st.button("Load Performance"):
        perf, err = _performance_data(perf_days, perf_sport)
        if err:
            st.error(err)
        elif not perf:
            st.warning("No performance payload returned.")
        else:
            a, b, c, d = st.columns(4)
            a.metric("Total Picks", str(perf.get("total_picks", 0)))
            b.metric("Settled", str(perf.get("settled_picks", 0)))
            c.metric("Hit Rate", f"{perf.get('hit_rate', 0)}%")
            d.metric("PnL Units", str(perf.get("pnl_units", 0)))

            e, f, g = st.columns(3)
            e.metric("Avg Edge", f"{perf.get('avg_edge_pct', 0)}%")
            f.metric("Avg Confidence", f"{perf.get('avg_confidence', 0)}")
            g.metric("Sport", str(perf.get("sport", "all")).upper())

            st.dataframe(_to_df([perf]), use_container_width=True)

with tab_picks:
    st.subheader("Recent Picks")
    p_days = st.slider("Pick Days", 1, 365, 14)
    p_limit = st.slider("Row Limit", 10, 1000, 200)
    p_sport = st.selectbox("Pick Sport Filter", [""] + SPORT_OPTIONS, index=0)

    if st.button("Load Picks"):
        picks_payload, err = _picks_data(p_days, p_sport, p_limit)
        if err:
            st.error(err)
        elif not picks_payload:
            st.warning("No picks payload returned.")
        else:
            items = picks_payload.get("items") or []
            st.metric("Returned", str(picks_payload.get("count", len(items))))
            st.dataframe(_to_df(items), use_container_width=True)

            if items and pd is not None:
                # Show quick confidence trend on most recent picks.
                trend = pd.DataFrame(items)
                if "confidence" in trend.columns:
                    trend = trend.sort_values("id")
                    st.markdown("**Confidence Trend**")
                    st.line_chart(trend.set_index("id")["confidence"])
                if "fallback_used" in trend.columns:
                    fallback_rate = round(100.0 * (trend["fallback_used"].astype(float).mean()), 2)
                    st.metric("Fallback Rate", f"{fallback_rate}%")

with tab_odds:
    st.subheader("Odds Edge Feed")
    oe_sport = st.selectbox("Odds Sport", SPORT_OPTIONS, index=0)
    oe_market = st.text_input("Market", value="player_points")
    oe_book = st.text_input("Bookmaker", value="draftkings")
    if st.button("Load Odds Edge"):
        payload, err = _odds_edge_data(oe_sport, oe_market, oe_book)
        if err:
            st.error(err)
        elif not payload:
            st.warning("No odds payload returned.")
        else:
            rows = payload.get("rows") or []
            st.metric("Events", str(payload.get("count", len(rows))))
            st.dataframe(_to_df(rows), use_container_width=True)
