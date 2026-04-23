import os
import time
import html
import json
import datetime
from typing import Any, Dict, List, Optional, Tuple

import requests
import streamlit as st
from requests.exceptions import ReadTimeout, RequestException

try:
    import altair as alt
except Exception:  # pragma: no cover
    alt = None

try:
    import pandas as pd
except Exception:  # pragma: no cover
    pd = None


st.set_page_config(layout="wide", page_title="Multi-Sport Prop Analyzer")

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&display=swap');

html, body, [class*="stApp"] {
  font-family: 'Space Grotesk', sans-serif;
  background: linear-gradient(180deg, #f7f0e6 0%, #eef4f7 100%);
}

:root {
  --ink: #1f2937;
  --muted: #5f6b7a;
  --accent: #0f766e;
  --accent-2: #e76f51;
  --accent-3: #2f80ed;
  --card: #ffffff;
  --card-border: #eadfce;
  --sand: #f8efe3;
}

.data-table {
  width: 100%;
  border-collapse: collapse;
  font-size: 13px;
  border-radius: 12px;
  overflow: hidden;
  border: 1px solid var(--card-border);
  background: var(--card);
}
.data-table thead tr {
  background: linear-gradient(90deg, #efe4d6 0%, #f8efe3 100%);
}
.data-table th, .data-table td {
  padding: 8px 10px;
  border-bottom: 1px solid #efe6d8;
}
.data-table tbody tr:hover {
  background: #fff8ef;
}

.top-banner {
  background: linear-gradient(135deg, #101726 0%, #1f2a44 35%, #2b4a6a 100%);
  border-radius: 22px;
  padding: 20px 26px;
  color: #fef6ea;
  border: 1px solid rgba(255, 255, 255, 0.08);
  box-shadow: 0 18px 30px rgba(20, 24, 40, 0.25);
  margin-bottom: 16px;
}
.top-title {
  font-size: 30px;
  font-weight: 700;
  margin-bottom: 4px;
}
.top-sub {
  font-size: 14px;
  color: #d8e0ef;
}
.top-tags {
  display: flex;
  gap: 8px;
  margin-top: 10px;
  flex-wrap: wrap;
}
.top-tag {
  background: rgba(255, 255, 255, 0.12);
  border: 1px solid rgba(255, 255, 255, 0.2);
  padding: 6px 12px;
  border-radius: 999px;
  font-size: 12px;
  color: #fef6ea;
  font-weight: 600;
}

.hero-card {
  background: linear-gradient(135deg, #ffffff 0%, #f7f0e6 100%);
  border: 1px solid var(--card-border);
  border-radius: 20px;
  padding: 18px 22px;
  box-shadow: 0 16px 30px rgba(33, 24, 11, 0.10);
}
.hero-title {
  font-size: 28px;
  font-weight: 700;
  color: var(--ink);
  margin: 0 0 6px 0;
}
.hero-sub {
  color: var(--muted);
  font-size: 14px;
  margin: 0 0 12px 0;
}
.hero-rec {
  font-size: 32px;
  font-weight: 700;
  letter-spacing: 0.5px;
  margin-top: 6px;
}
.pill-row {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  margin-top: 10px;
}
.pill {
  background: #f3efe9;
  border: 1px solid #eadfce;
  border-radius: 999px;
  padding: 6px 12px;
  font-size: 12px;
  color: var(--ink);
  font-weight: 600;
}
.metric-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
  gap: 12px;
  margin: 16px 0;
}
.metric-card {
  background: var(--card);
  border: 1px solid var(--card-border);
  border-radius: 14px;
  padding: 10px 14px;
  box-shadow: 0 10px 20px rgba(35, 22, 10, 0.06);
}
.metric-label {
  font-size: 12px;
  color: var(--muted);
  font-weight: 600;
}
.metric-value {
  font-size: 20px;
  color: var(--ink);
  font-weight: 700;
  margin-top: 6px;
}
.gauge-card {
  background: var(--card);
  border: 1px solid var(--card-border);
  border-radius: 20px;
  padding: 16px;
  text-align: center;
  box-shadow: 0 16px 30px rgba(33, 24, 11, 0.10);
}

.chart-card {
  background: #ffffff;
  border: 1px solid var(--card-border);
  border-radius: 16px;
  padding: 12px 16px 10px 16px;
  box-shadow: 0 10px 25px rgba(69, 47, 23, 0.06);
}
.chart-row {
  display: grid;
  grid-template-columns: 140px 1fr 66px;
  gap: 12px;
  align-items: center;
  margin: 10px 0;
}
.chart-label {
  font-size: 13px;
  color: var(--muted);
  font-weight: 600;
}
.chart-value {
  font-size: 12px;
  color: var(--ink);
  text-align: right;
  background: #f3efe9;
  border-radius: 999px;
  padding: 4px 10px;
  font-weight: 600;
}
.chart-bar {
  position: relative;
  height: 12px;
  border-radius: 999px;
  background: #f2ede6;
  overflow: hidden;
}
.chart-fill {
  height: 100%;
  border-radius: 999px;
  transition: width 0.35s ease;
}
.chart-dot {
  position: absolute;
  top: 50%;
  width: 12px;
  height: 12px;
  border-radius: 50%;
  transform: translate(-50%, -50%);
  border: 2px solid #ffffff;
  box-shadow: 0 2px 8px rgba(17, 24, 39, 0.25);
}

section[data-testid="stSidebar"] {
  background: #ffffff;
  border-right: 1px solid var(--card-border);
}
section[data-testid="stSidebar"] button {
  border-radius: 12px !important;
  font-weight: 600 !important;
}

.pitch-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
  gap: 14px;
  margin: 14px 0 18px 0;
}
.pitch-card {
  background: linear-gradient(160deg, #ffffff 0%, #f7f1e8 100%);
  border: 1px solid var(--card-border);
  border-radius: 18px;
  padding: 16px;
  box-shadow: 0 16px 30px rgba(33, 24, 11, 0.10);
}
.pitch-head {
  display: flex;
  justify-content: space-between;
  align-items: flex-start;
  gap: 12px;
}
.pitch-name {
  font-size: 22px;
  line-height: 1.1;
  font-weight: 700;
  color: var(--ink);
}
.pitch-sub {
  font-size: 12px;
  color: var(--muted);
  margin-top: 4px;
}
.edge-chip {
  padding: 8px 10px;
  border-radius: 14px;
  font-size: 12px;
  font-weight: 700;
  color: #fffdf8;
  background: linear-gradient(135deg, #0f766e 0%, #2f80ed 100%);
  min-width: 76px;
  text-align: center;
}
.pitch-prop {
  margin-top: 14px;
  font-size: 18px;
  font-weight: 700;
  color: #102a43;
}
.pitch-mini {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  margin-top: 10px;
}
.mini-pill {
  background: #fff8ef;
  border: 1px solid #eadfce;
  border-radius: 999px;
  padding: 5px 10px;
  font-size: 11px;
  color: #304050;
  font-weight: 600;
}
.component-stack {
  margin-top: 12px;
  display: grid;
  gap: 8px;
}
.component-row {
  display: grid;
  grid-template-columns: 92px 1fr 44px;
  gap: 10px;
  align-items: center;
}
.component-label {
  font-size: 11px;
  color: var(--muted);
  font-weight: 700;
  text-transform: uppercase;
}
.component-track {
  height: 10px;
  background: #efe6d8;
  border-radius: 999px;
  overflow: hidden;
}
.component-fill {
  height: 100%;
  border-radius: 999px;
  background: linear-gradient(90deg, #0f766e 0%, #2f80ed 100%);
}
.component-value {
  font-size: 11px;
  color: var(--ink);
  font-weight: 700;
  text-align: right;
}
.section-label {
  font-size: 12px;
  letter-spacing: 0.12em;
  text-transform: uppercase;
  color: #7b6a55;
  font-weight: 700;
  margin-bottom: 8px;
}
</style>
""",
    unsafe_allow_html=True,
)

st.markdown(
    """
<div class="top-banner">
  <div class="top-title">Prop Lab</div>
  <div class="top-sub">AI-graded props with matchup context, sharp signals, and visual edges.</div>
  <div class="top-tags">
    <span class="top-tag">NBA</span>
    <span class="top-tag">MLB</span>
    <span class="top-tag">NFL</span>
    <span class="top-tag">Live Context</span>
    <span class="top-tag">Confidence Grading</span>
  </div>
</div>
""",
    unsafe_allow_html=True,
)


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

SPORT_OPTIONS = ["nba", "ncaab", "mlb", "nfl", "soccer", "nhl", "tennis", "golf", "cs2", "cod"]
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
    "ncaab": ["points", "rebounds", "assists", "points+assists", "points+rebounds", "rebounds+assists", "pra"],
    "mlb": ["hits", "runs", "rbis", "home_runs", "total_bases", "strikeouts"],
    "nfl": ["passing_yards", "rushing_yards", "receiving_yards", "receptions", "touchdowns"],
    "soccer": ["goals", "assists", "shots", "shots_on_target", "passes"],
    "nhl": ["goals", "assists", "points", "shots", "saves"],
    "tennis": ["aces", "double_faults", "first_serve_pct", "break_points_won", "games_won"],
    "golf": ["birdies", "bogeys", "pars", "fairways_hit", "greens_in_regulation"],
    "cs2": ["kills", "deaths", "assists", "headshots", "kd_ratio", "map_wins"],
    "cod": ["kills", "deaths", "assists", "kd_ratio", "objective_kills", "map_wins"],
}
OPPONENTS_BY_SPORT = {
    "nba": NBA_TEAMS,
    "ncaab": [""],
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
    "tennis": ["", "DJOKOVIC", "ALCARAZ", "SINNER", "MEDVEDEV", "RUNE", "ZVEREV"],
    "golf": ["", "SCHEFFLER", "MCILROY", "RAHM", "SCHAUFFELE", "THOMAS", "MORIKAWA"],
    "cs2": ["", "NAVI", "FAZE", "G2", "VITALITY", "MOUZ", "SPIRIT"],
    "cod": ["", "FAZE", "OPTIC", "ULTRA", "SURGE", "THIEVES", "SUBLINERS"],
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


def _cell_to_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (dict, list, tuple)):
        try:
            return json.dumps(value)
        except Exception:
            return str(value)
    return str(value)


def _rows_from_any(data: Any) -> List[Dict[str, Any]]:
    if data is None:
        return []
    if pd is not None and isinstance(data, pd.DataFrame):
        return data.where(data.notna(), None).to_dict(orient="records")
    if isinstance(data, dict):
        return [data]
    if isinstance(data, list):
        if not data:
            return []
        if all(isinstance(x, dict) for x in data):
            return data
        return [{"value": _cell_to_text(x)} for x in data]
    return [{"value": _cell_to_text(data)}]


def _safe_pct(value: Any) -> str:
    if value in (None, "", "n/a", "N/A"):
        return "n/a"
    try:
        val = float(value)
    except Exception:
        return str(value)
    return f"{val:.1f}%"


def _safe_num(value: Any, digits: int = 2) -> str:
    if value in (None, "", "n/a", "N/A"):
        return "n/a"
    try:
        val = float(value)
    except Exception:
        return str(value)
    fmt = f"{{:.{digits}f}}"
    return fmt.format(val)


def _safe_signed_pct(value: Any) -> str:
    if value in (None, "", "n/a", "N/A"):
        return "n/a"
    try:
        val = float(value)
    except Exception:
        return str(value)
    return f"{val:+.1f}%"


def _safe_american(value: Any) -> str:
    if value in (None, "", "n/a", "N/A"):
        return "n/a"
    try:
        odds = int(value)
    except Exception:
        return str(value)
    return f"+{odds}" if odds > 0 else str(odds)


def _parse_optional_float(raw: Any) -> Optional[float]:
    text = str(raw or "").strip()
    if not text:
        return None
    return float(text)


def _grade_from_conf(confidence: Any) -> str:
    try:
        val = float(confidence)
    except Exception:
        return "?"
    if val >= 92:
        return "A+"
    if val >= 88:
        return "A"
    if val >= 83:
        return "B+"
    if val >= 78:
        return "B"
    if val >= 72:
        return "B-"
    if val >= 66:
        return "C+"
    if val >= 60:
        return "C"
    return "C-"


def _render_hero(result: Dict[str, Any], opponent: str, season_type: str) -> None:
    player = str(result.get("player", "")).strip()
    prop = str(result.get("prop", "")).strip().replace("_", " ").upper()
    line = result.get("line")
    rec = str(result.get("recommendation", "N/A"))
    rec_lower = rec.lower()
    rec_color = "#0f766e" if "over" in rec_lower else "#e76f51"
    conf = result.get("confidence")
    grade = _grade_from_conf(conf)
    edge = result.get("edge_pct")
    proj = result.get("projected_probability")
    data_source = str(result.get("data_source", "unknown"))
    line_text = f"{line}" if line is not None else "?"
    opp = opponent.upper() if opponent else "Any Opp"

    pills = [
        f"Edge { _safe_pct(edge) }",
        f"Proj { _safe_pct(proj) }",
        f"Grade {grade}",
        f"Source {html.escape(data_source)}",
    ]
    pills_html = "".join(f"<span class='pill'>{p}</span>" for p in pills)

    hero_html = f"""
    <div class="hero-card">
      <div class="hero-title">{html.escape(player)}</div>
      <div class="hero-sub">{html.escape(prop)} · Line {html.escape(line_text)} · Opp {html.escape(opp)} · {html.escape(season_type)}</div>
      <div class="hero-rec" style="color:{rec_color};">{html.escape(rec)}</div>
      <div class="pill-row">{pills_html}</div>
    </div>
    """
    st.markdown(hero_html, unsafe_allow_html=True)


def _render_metric_cards(items: List[Tuple[str, str]]) -> None:
    cards = ""
    for label, value in items:
        cards += (
            "<div class='metric-card'>"
            f"<div class='metric-label'>{html.escape(label)}</div>"
            f"<div class='metric-value'>{html.escape(value)}</div>"
            "</div>"
        )
    st.markdown(f"<div class='metric-grid'>{cards}</div>", unsafe_allow_html=True)


def _render_gauge(value: Any, label: str, color: str = "#0f766e") -> None:
    try:
        val = float(value)
    except Exception:
        val = 0.0
    val = max(0.0, min(100.0, val))
    radius = 46
    circumference = 2 * 3.1416 * radius
    offset = circumference * (1 - (val / 100.0))
    html_block = f"""
    <div class="gauge-card">
      <svg width="140" height="140" viewBox="0 0 120 120">
        <circle cx="60" cy="60" r="{radius}" stroke="#eee6db" stroke-width="10" fill="none" />
        <circle cx="60" cy="60" r="{radius}" stroke="{color}" stroke-width="10" fill="none"
          stroke-dasharray="{circumference:.2f}" stroke-dashoffset="{offset:.2f}"
          stroke-linecap="round" transform="rotate(-90 60 60)" />
        <text x="60" y="62" text-anchor="middle" font-size="22" font-weight="700" fill="#1f2937">{val:.0f}%</text>
        <text x="60" y="82" text-anchor="middle" font-size="11" fill="#5f6b7a">{html.escape(label)}</text>
      </svg>
    </div>
    """
    st.markdown(html_block, unsafe_allow_html=True)


def _render_table(data: Any, *, max_rows: int = 300) -> None:
    rows = _rows_from_any(data)
    if not rows:
        st.write("No data.")
        return

    columns: List[str] = []
    for row in rows:
        for key in row.keys():
            if key not in columns:
                columns.append(key)

    show_rows = rows[:max_rows]
    header_html = "".join(f"<th>{html.escape(str(c))}</th>" for c in columns)
    body_parts: List[str] = []
    for row in show_rows:
        cells = "".join(f"<td>{html.escape(_cell_to_text(row.get(col, '')))}</td>" for col in columns)
        body_parts.append(f"<tr>{cells}</tr>")

    table_html = (
        "<div style='overflow-x:auto;'>"
        "<table class='data-table'>"
        "<thead><tr>"
        f"{header_html}"
        "</tr></thead>"
        "<tbody>"
        f"{''.join(body_parts)}"
        "</tbody></table></div>"
    )
    st.markdown(table_html, unsafe_allow_html=True)
    if len(rows) > max_rows:
        st.caption(f"Showing {max_rows} of {len(rows)} rows.")


def _hex_to_rgb(hex_color: str) -> Tuple[int, int, int]:
    clean = (hex_color or "").lstrip("#")
    if len(clean) != 6:
        return (46, 134, 222)
    try:
        return tuple(int(clean[i:i+2], 16) for i in (0, 2, 4))
    except Exception:
        return (46, 134, 222)


def _mix_with_white(rgb: Tuple[int, int, int], mix: float = 0.7) -> Tuple[int, int, int]:
    mix = max(0.0, min(1.0, mix))
    return tuple(int(c + (255 - c) * mix) for c in rgb)


def _rgb_to_hex(rgb: Tuple[int, int, int]) -> str:
    return "#%02x%02x%02x" % rgb


def _render_bars(rows: List[Dict[str, Any]], label_key: str, value_key: str, color: str = "#2E86DE") -> None:
    if not rows:
        st.write("No chart data.")
        return
    base_rgb = _hex_to_rgb(color)
    soft_hex = _rgb_to_hex(_mix_with_white(base_rgb, 0.75))
    rows_html = ""
    for row in rows:
        label = _cell_to_text(row.get(label_key, ""))
        try:
            raw_val = float(row.get(value_key, 0))
        except Exception:
            raw_val = 0.0
        val = max(0.0, min(100.0, raw_val))
        dot_left = max(2.0, min(98.0, val))
        rows_html += (
            "<div class='chart-row'>"
            f"<div class='chart-label'>{html.escape(label)}</div>"
            "<div class='chart-bar'>"
            f"<div class='chart-fill' style='width:{val}%; background:linear-gradient(90deg, {color} 0%, {soft_hex} 100%);'></div>"
            f"<div class='chart-dot' style='left:{dot_left}%; background:{color};'></div>"
            "</div>"
            f"<div class='chart-value'>{val:.1f}</div>"
            "</div>"
        )
    st.markdown(f"<div class='chart-card'>{rows_html}</div>", unsafe_allow_html=True)


def _pitch_pick_label(row: Dict[str, Any]) -> str:
    pitcher = str(row.get("pitcher", "")).strip()
    market = str(row.get("market_name") or row.get("market_key") or "").replace("Pitcher ", "")
    side = str(row.get("recommended_side", "")).upper()
    line = row.get("line")
    line_text = "?" if line in (None, "") else str(line)
    return f"{pitcher} {side} {line_text} {market}".strip()


def _render_pitch_edge_hero(best: Dict[str, Any]) -> None:
    hero_html = f"""
    <div class="hero-card">
      <div class="section-label">Top Market Edge</div>
      <div class="hero-title">{html.escape(str(best.get("pitcher", "")))}</div>
      <div class="hero-sub">{html.escape(str(best.get("team", "")))} vs {html.escape(str(best.get("opponent", "")))} • {html.escape(str(best.get("market_name", "")))}</div>
      <div class="hero-rec" style="color:#0f766e;">{html.escape(str(best.get("recommended_side", "")).upper())} {html.escape(str(best.get("line", "")))}</div>
      <div class="pill-row">
        <span class="pill">Edge {html.escape(_safe_signed_pct(best.get("edge_pct")))}</span>
        <span class="pill">Proj {html.escape(_safe_pct(best.get("projected_probability")))}</span>
        <span class="pill">Odds {html.escape(_safe_american(best.get("offered_odds")))}</span>
        <span class="pill">Fair {html.escape(_safe_american(best.get("fair_odds_american")))}</span>
        <span class="pill">EV {html.escape(_safe_num(best.get("ev_units"), 3))}u</span>
      </div>
    </div>
    """
    st.markdown(hero_html, unsafe_allow_html=True)


def _render_pitch_pick_cards(rows: List[Dict[str, Any]], max_cards: int = 6) -> None:
    cards = []
    for row in rows[:max_cards]:
        edge = _safe_signed_pct(row.get("edge_pct"))
        market_name = str(row.get("market_name", ""))
        side = str(row.get("recommended_side", "")).upper()
        line_text = str(row.get("line", ""))
        subtitle = f"{row.get('team', '')} vs {row.get('opponent', '')} • {row.get('lineup_status', '')}"
        pills = [
            f"Proj {_safe_pct(row.get('projected_probability'))}",
            f"Odds {_safe_american(row.get('offered_odds'))}",
            f"Fair {_safe_american(row.get('fair_odds_american'))}",
            f"Conf {_safe_num(row.get('confidence'), 1)}",
        ]
        pills_html = "".join(f"<span class='mini-pill'>{html.escape(p)}</span>" for p in pills)
        component_rows = ""
        for label, value in (row.get("components") or {}).items():
            try:
                val = max(0.0, min(100.0, float(value)))
            except Exception:
                continue
            label_text = label.replace("_", " ")
            component_rows += (
                "<div class='component-row'>"
                f"<div class='component-label'>{html.escape(label_text)}</div>"
                "<div class='component-track'>"
                f"<div class='component-fill' style='width:{val}%;'></div>"
                "</div>"
                f"<div class='component-value'>{val:.0f}</div>"
                "</div>"
            )
        primary_pitch = (row.get("primary_pitch") or {}).get("description") or (row.get("primary_pitch") or {}).get("code") or ""
        if primary_pitch:
            pills_html += f"<span class='mini-pill'>{html.escape(str(primary_pitch))}</span>"
        cards.append(
            "<div class='pitch-card'>"
            "<div class='pitch-head'>"
            "<div>"
            f"<div class='pitch-name'>{html.escape(str(row.get('pitcher', '')))}</div>"
            f"<div class='pitch-sub'>{html.escape(subtitle)}</div>"
            "</div>"
            f"<div class='edge-chip'>{html.escape(edge)}</div>"
            "</div>"
            f"<div class='pitch-prop'>{html.escape(side)} {html.escape(line_text)} {html.escape(market_name)}</div>"
            f"<div class='pitch-mini'>{pills_html}</div>"
            f"<div class='component-stack'>{component_rows}</div>"
            "</div>"
        )
    if cards:
        st.markdown(f"<div class='pitch-grid'>{''.join(cards)}</div>", unsafe_allow_html=True)


def _render_pitch_edge_bar_chart(rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    if pd is None or alt is None:
        _render_bars(
            [{"label": _pitch_pick_label(row), "edge_pct": float(row.get("edge_pct", 0) or 0)} for row in rows[:10]],
            "label",
            "edge_pct",
            color="#0f766e",
        )
        return
    frame = pd.DataFrame(
        [
            {
                "Pick": _pitch_pick_label(row),
                "Edge": float(row.get("edge_pct", 0) or 0),
                "Side": str(row.get("recommended_side", "")).title(),
            }
            for row in rows[:12]
        ]
    )
    chart = (
        alt.Chart(frame)
        .mark_bar(cornerRadiusEnd=6)
        .encode(
            x=alt.X("Edge:Q", title="Edge %"),
            y=alt.Y("Pick:N", sort="-x", title=""),
            color=alt.Color("Side:N", scale=alt.Scale(domain=["Over", "Under"], range=["#0f766e", "#e76f51"])),
            tooltip=["Pick", alt.Tooltip("Edge:Q", format=".2f"), "Side"],
        )
        .properties(height=380)
    )
    st.altair_chart(chart, use_container_width=True)


def _render_pitch_scatter_chart(rows: List[Dict[str, Any]]) -> None:
    if not rows or pd is None or alt is None:
        return
    frame = pd.DataFrame(
        [
            {
                "Pitcher": str(row.get("pitcher", "")),
                "Market": str(row.get("market_name", "")),
                "Edge": float(row.get("edge_pct", 0) or 0),
                "Confidence": float(row.get("confidence", 0) or 0),
                "ProjProb": float(row.get("projected_probability", 0) or 0),
            }
            for row in rows[:24]
        ]
    )
    chart = (
        alt.Chart(frame)
        .mark_circle(opacity=0.85, stroke="#102a43", strokeWidth=0.4)
        .encode(
            x=alt.X("Confidence:Q", title="Confidence"),
            y=alt.Y("Edge:Q", title="Edge %"),
            size=alt.Size("ProjProb:Q", title="Projected Probability", scale=alt.Scale(range=[80, 800])),
            color=alt.Color("Market:N", legend=alt.Legend(title="Market")),
            tooltip=["Pitcher", "Market", alt.Tooltip("Edge:Q", format=".2f"), alt.Tooltip("Confidence:Q", format=".1f"), alt.Tooltip("ProjProb:Q", format=".2f")],
        )
        .properties(height=360)
    )
    st.altair_chart(chart, use_container_width=True)


def _render_pitch_component_heatmap(rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    heat_rows = []
    for row in rows[:8]:
        for component, value in (row.get("components") or {}).items():
            try:
                heat_rows.append(
                    {
                        "Pick": f"{str(row.get('pitcher', ''))} {str(row.get('market_name', '')).replace('Pitcher ', '')}",
                        "Component": component.replace("_", " ").title(),
                        "Score": float(value),
                    }
                )
            except Exception:
                pass
    if not heat_rows:
        return
    if pd is None or alt is None:
        _render_table(heat_rows, max_rows=60)
        return
    frame = pd.DataFrame(heat_rows)
    chart = (
        alt.Chart(frame)
        .mark_rect(cornerRadius=4)
        .encode(
            x=alt.X("Component:N", title=""),
            y=alt.Y("Pick:N", title="", sort=None),
            color=alt.Color("Score:Q", scale=alt.Scale(domain=[40, 100], range=["#f7d9bf", "#2f80ed"])),
            tooltip=["Pick", "Component", alt.Tooltip("Score:Q", format=".1f")],
        )
        .properties(height=max(180, min(420, 36 * len(frame["Pick"].unique()))))
    )
    st.altair_chart(chart, use_container_width=True)

def _color_for_percentile(pct: Optional[float]) -> str:
    if pct is None:
        return "#e0e0e0"
    try:
        value = max(0.0, min(100.0, float(pct)))
    except Exception:
        return "#e0e0e0"
    # 0 = red, 100 = green
    r = int(220 + (46 - 220) * (value / 100.0))
    g = int(85 + (204 - 85) * (value / 100.0))
    b = int(75 + (113 - 75) * (value / 100.0))
    return f"rgb({r},{g},{b})"


def _format_pct(value: Optional[float]) -> str:
    if value is None:
        return "n/a"
    try:
        val = float(value)
    except Exception:
        return "n/a"
    if val <= 1.2:
        val *= 100.0
    return f"{val:.1f}%"


def _render_defense_court(zones: List[Dict[str, Any]]) -> None:
    if not zones:
        st.write("No shot zone data.")
        return
    zone_map = {str(z.get("zone", "")): z for z in zones}
    def zone_color(name: str) -> str:
        return _color_for_percentile(zone_map.get(name, {}).get("percentile"))

    def zone_pct(name: str) -> str:
        return _format_pct(zone_map.get(name, {}).get("fg_pct"))

    svg = f"""
    <svg viewBox="0 0 600 420" width="100%" height="420" xmlns="http://www.w3.org/2000/svg">
      <rect x="0" y="0" width="600" height="420" fill="#fdf7f0" stroke="#c8a165" stroke-width="3"/>
      <rect x="0" y="0" width="120" height="200" fill="{zone_color('Left Corner 3')}" opacity="0.85"/>
      <rect x="480" y="0" width="120" height="200" fill="{zone_color('Right Corner 3')}" opacity="0.85"/>
      <rect x="120" y="0" width="360" height="120" fill="{zone_color('Above the Break 3')}" opacity="0.85"/>
      <rect x="120" y="120" width="360" height="120" fill="{zone_color('Mid-Range')}" opacity="0.85"/>
      <rect x="200" y="220" width="200" height="150" fill="{zone_color('In The Paint (Non-RA)')}" opacity="0.9"/>
      <rect x="240" y="290" width="120" height="80" fill="{zone_color('Restricted Area')}" opacity="0.95"/>
      <rect x="0" y="0" width="600" height="420" fill="none" stroke="#c8a165" stroke-width="2"/>
      <text x="60" y="30" text-anchor="middle" font-size="13" fill="#2d2d2d">L Corner 3</text>
      <text x="60" y="48" text-anchor="middle" font-size="12" fill="#2d2d2d">{zone_pct('Left Corner 3')}</text>
      <text x="540" y="30" text-anchor="middle" font-size="13" fill="#2d2d2d">R Corner 3</text>
      <text x="540" y="48" text-anchor="middle" font-size="12" fill="#2d2d2d">{zone_pct('Right Corner 3')}</text>
      <text x="300" y="30" text-anchor="middle" font-size="13" fill="#2d2d2d">Above Break 3</text>
      <text x="300" y="48" text-anchor="middle" font-size="12" fill="#2d2d2d">{zone_pct('Above the Break 3')}</text>
      <text x="300" y="145" text-anchor="middle" font-size="13" fill="#2d2d2d">Mid-Range</text>
      <text x="300" y="163" text-anchor="middle" font-size="12" fill="#2d2d2d">{zone_pct('Mid-Range')}</text>
      <text x="300" y="250" text-anchor="middle" font-size="13" fill="#2d2d2d">Paint</text>
      <text x="300" y="268" text-anchor="middle" font-size="12" fill="#2d2d2d">{zone_pct('In The Paint (Non-RA)')}</text>
      <text x="300" y="330" text-anchor="middle" font-size="13" fill="#2d2d2d">Restricted</text>
      <text x="300" y="348" text-anchor="middle" font-size="12" fill="#2d2d2d">{zone_pct('Restricted Area')}</text>
    </svg>
    """
    st.markdown(svg, unsafe_allow_html=True)


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


def _mlb_pitching_bets_data(date_value: str, season_year: int, min_score: float, limit: int) -> Tuple[Optional[Dict[str, Any]], str]:
    params = {"date": date_value, "season_year": season_year, "min_score": min_score, "limit": limit}
    return _backend_request("GET", f"{BACKEND_BASE_URL}/mlb/pitching-bets", params=params)


def _mlb_pitching_edges_data(
    date_value: str,
    season_year: int,
    bookmaker: str,
    regions: str,
    markets: str,
    min_edge_pct: float,
    limit: int,
) -> Tuple[Optional[Dict[str, Any]], str]:
    params = {
        "date": date_value,
        "season_year": season_year,
        "bookmaker": bookmaker,
        "regions": regions,
        "markets": markets,
        "min_edge_pct": min_edge_pct,
        "limit": limit,
    }
    return _backend_request("GET", f"{BACKEND_BASE_URL}/mlb/pitching-bets/edges", params=params)


def _team_intel_data(
    team: str,
    season_type: str,
    include_depth: bool,
    include_injuries: bool,
    include_defense: bool,
    include_shot_zones: bool,
) -> Tuple[Optional[Dict[str, Any]], str]:
    params = {
        "team": team,
        "season_type": season_type,
        "include_depth": include_depth,
        "include_injuries": include_injuries,
        "include_defense": include_defense,
        "include_shot_zones": include_shot_zones,
    }
    return _backend_request("GET", f"{BACKEND_BASE_URL}/nba/team-intel", params=params)


def _player_splits_data(player: str, without: str, season_type: str) -> Tuple[Optional[Dict[str, Any]], str]:
    params = {
        "player": player,
        "without": without,
        "season_type": season_type,
    }
    return _backend_request("GET", f"{BACKEND_BASE_URL}/nba/player-splits", params=params)


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

tab_analyze, tab_pitch, tab_research, tab_health, tab_perf, tab_picks, tab_odds = st.tabs(
    ["Analyze", "Pitch Lab", "Research", "Health", "Performance", "Picks", "Odds Edge"]
)

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

    mlb_pitcher_name = ""
    mlb_venue = ""
    mlb_wind_direction = ""
    mlb_wind_mph_raw = ""
    mlb_temp_raw = ""
    mlb_altitude_raw = ""
    if sport == "mlb":
        st.markdown("**MLB Matchup Inputs**")
        m1, m2, m3 = st.columns(3)
        with m1:
            mlb_pitcher_name = st.text_input("Opposing Pitcher (optional)", "")
            mlb_venue = st.text_input("Venue Override (optional)", "")
        with m2:
            mlb_wind_mph_raw = st.text_input("Wind MPH (optional)", "")
            mlb_temp_raw = st.text_input("Temperature F (optional)", "")
        with m3:
            mlb_wind_direction = st.selectbox(
                "Wind Direction",
                ["", "out", "in", "left to right", "right to left", "crosswind"],
                index=0,
            )
            mlb_altitude_raw = st.text_input("Altitude ft (optional)", "")

    if compare_mode:
        compare_props = st.multiselect("Props to compare", PROP_OPTIONS_BY_SPORT[sport], default=[prop])
        compare_lines_raw = st.text_input("Lines to compare (comma separated)", "20.5, 25.5")
    else:
        compare_props = [prop]
        compare_lines_raw = str(line)

    if st.button("Evaluate", type="primary"):
        with st.spinner("Running analysis..."):
            mlb_wind_mph = None
            mlb_temperature_f = None
            mlb_altitude_ft = None
            if sport == "mlb":
                try:
                    mlb_wind_mph = _parse_optional_float(mlb_wind_mph_raw)
                    mlb_temperature_f = _parse_optional_float(mlb_temp_raw)
                    mlb_altitude_ft = _parse_optional_float(mlb_altitude_raw)
                except ValueError:
                    st.error("Wind, temperature, and altitude values must be numeric.")
                    st.stop()

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
                    if sport == "mlb":
                        payload["pitcher_name"] = mlb_pitcher_name.strip()
                        payload["venue"] = mlb_venue.strip()
                        payload["wind_direction"] = mlb_wind_direction.strip()
                        if mlb_wind_mph is not None:
                            payload["wind_mph"] = mlb_wind_mph
                        if mlb_temperature_f is not None:
                            payload["temperature_f"] = mlb_temperature_f
                        if mlb_altitude_ft is not None:
                            payload["altitude_ft"] = mlb_altitude_ft

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

            hero_col, gauge_col = st.columns([2.3, 1])
            with hero_col:
                _render_hero(best, opponent, season_type)
            with gauge_col:
                _render_gauge(best.get("confidence", 0), "Confidence", color="#0f766e")

            _render_metric_cards(
                [
                    ("Projected Prob", _safe_pct(best.get("projected_probability"))),
                    ("Edge", _safe_pct(best.get("edge_pct"))),
                    ("Expected Stat", _safe_num(best.get("expected_stat"), 2)),
                    (str(best.get("projection_label", "Projection")), _safe_num(best.get("minutes_proj"), 1)),
                ]
            )

            if best.get("fallback_used"):
                st.warning("Fallback model used because live data was unavailable.")

            injury_ctx = best.get("injury_context", {}) or {}
            if injury_ctx.get("status") and injury_ctx.get("status") != "not_requested":
                st.info(f"Injury: {injury_ctx.get('status')} | {injury_ctx.get('detail', '')}")

            mlb_ctx = best.get("mlb_context") if isinstance(best, dict) else None
            if mlb_ctx:
                with st.expander("MLB Context"):
                    summary = {
                        "role": mlb_ctx.get("role"),
                        "opponent": mlb_ctx.get("opponent"),
                        "lineup_spot": mlb_ctx.get("lineup_spot"),
                        "ahead_obp": mlb_ctx.get("ahead_obp"),
                        "team_obp": mlb_ctx.get("team_obp"),
                        "log_season_year": mlb_ctx.get("log_season_year"),
                        "log_fallback": mlb_ctx.get("log_fallback"),
                    }
                    st.markdown("**Summary**")
                    _render_table([summary])
                    pitcher = mlb_ctx.get("pitcher") or {}
                    if pitcher.get("name"):
                        st.markdown("**Opposing Pitcher**")
                        _render_table([pitcher])
                    environment = mlb_ctx.get("environment") or {}
                    if environment:
                        st.markdown("**Environment**")
                        _render_table([environment])
                    pitch_matchup = mlb_ctx.get("pitch_matchup") or {}
                    primary_pitch = pitch_matchup.get("primary_pitch") or {}
                    hitter_pitch_stats = pitch_matchup.get("hitter_pitch_stats") or {}
                    if primary_pitch or hitter_pitch_stats:
                        st.markdown("**Pitch Matchup**")
                        _render_table(
                            [
                                {
                                    "pitch": primary_pitch.get("description") or primary_pitch.get("code"),
                                    "pitch_code": primary_pitch.get("code"),
                                    "usage_pct": primary_pitch.get("usage_pct"),
                                    "hitter_avg": hitter_pitch_stats.get("avg"),
                                    "hitter_slg": hitter_pitch_stats.get("slg"),
                                    "hr_rate": hitter_pitch_stats.get("hr_rate"),
                                    "k_rate": hitter_pitch_stats.get("k_rate"),
                                    "at_bats": hitter_pitch_stats.get("at_bats"),
                                    "hitter_pitch_year": pitch_matchup.get("hitter_pitch_year"),
                                    "hitter_pitch_fallback": pitch_matchup.get("hitter_pitch_fallback"),
                                }
                            ]
                        )
                    factors = mlb_ctx.get("factors") or {}
                    if factors:
                        st.markdown("**Context Factors**")
                        _render_table([factors])

            reasons = best.get("reasons", []) or []
            if reasons:
                with st.expander("Why this recommendation", expanded=True):
                    for reason in reasons:
                        st.write(f"- {reason}")

            last_games_detail = best.get("last_games_detail") or []
            h2h_games_detail = best.get("h2h_games_detail") or []
            st.markdown("**Last Games (Prop Results)**")
            if last_games_detail:
                _render_table(last_games_detail, max_rows=20)
            else:
                st.caption("No per-game detail available for this result.")

            st.markdown("**H2H Games (Prop Results)**")
            if h2h_games_detail:
                _render_table(h2h_games_detail, max_rows=20)
            else:
                st.caption("No H2H prop-game detail available for this matchup.")

            chart_rows = [
                {"window": "Last 5", "hit_rate": float(best.get("last_5_hit_rate", 0)), "avg_stat": float(best.get("last_5_avg_stat", 0))},
                {"window": "Last 10", "hit_rate": float(best.get("last_10_hit_rate", 0)), "avg_stat": float(best.get("last_10_avg_stat", 0))},
                {"window": "H2H", "hit_rate": float(best.get("h2h_hit_rate", 0)), "avg_stat": float(best.get("h2h_avg_stat", 0))},
            ]
            st.markdown("**Hit Rate by Window**")
            _render_bars(chart_rows, "window", "hit_rate", color="#0f766e")
            st.markdown("**Average Stat by Window (scaled to line=100 max visual)**")
            scaled_stat_rows = []
            max_stat = max(1.0, max(float(r.get("avg_stat", 0) or 0) for r in chart_rows))
            for row in chart_rows:
                scaled_stat_rows.append(
                    {
                        "window": row["window"],
                        "avg_stat_scaled": (float(row.get("avg_stat", 0) or 0) / max_stat) * 100.0,
                        "avg_stat_raw": float(row.get("avg_stat", 0) or 0),
                    }
                )
            _render_bars(scaled_stat_rows, "window", "avg_stat_scaled", color="#e76f51")

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
            _render_table(compare_rows)

    with st.expander("Local Session History"):
        if st.button("Clear History"):
            st.session_state["history"] = []
        history = st.session_state.get("history", [])
        if history:
            _render_table(history)
        else:
            st.write("No history yet.")

with tab_pitch:
    st.subheader("MLB Pitch Lab")
    st.caption("Ranks daily pitcher props from model-only signals and live market edges.")

    st.session_state.setdefault("pitch_lab_edge_payload", None)
    st.session_state.setdefault("pitch_lab_edge_error", "")
    st.session_state.setdefault("pitch_lab_model_payload", None)
    st.session_state.setdefault("pitch_lab_model_error", "")

    today_local = datetime.datetime.now().date()
    market_options = [
        "pitcher_strikeouts",
        "pitcher_walks",
        "pitcher_earned_runs",
        "pitcher_hits_allowed",
        "pitcher_outs",
    ]

    ctl_a, ctl_b, ctl_c, ctl_d = st.columns([1.1, 1, 1, 1])
    with ctl_a:
        pitch_lab_date = st.date_input("Slate Date", value=today_local, key="pitch_lab_date")
    with ctl_b:
        pitch_lab_season_year = int(
            st.number_input(
                "Season Year",
                min_value=2008,
                max_value=2100,
                value=max(2008, int(getattr(pitch_lab_date, "year", today_local.year))),
                step=1,
                key="pitch_lab_season_year",
            )
        )
    with ctl_c:
        pitch_lab_bookmaker = st.text_input("Bookmaker", value="draftkings", key="pitch_lab_bookmaker")
    with ctl_d:
        pitch_lab_regions = st.text_input("Regions", value="us", key="pitch_lab_regions")

    filter_a, filter_b, filter_c = st.columns([1.2, 1, 1])
    with filter_a:
        pitch_lab_markets = st.multiselect(
            "Markets",
            market_options,
            default=market_options,
            key="pitch_lab_markets",
        )
    with filter_b:
        pitch_lab_limit = int(st.slider("Results", min_value=5, max_value=30, value=12, step=1, key="pitch_lab_limit"))
    with filter_c:
        pitch_lab_min_edge = float(
            st.slider("Min Edge %", min_value=-2.0, max_value=12.0, value=2.0, step=0.5, key="pitch_lab_min_edge")
        )

    model_a, model_b = st.columns([1, 1.2])
    with model_a:
        pitch_lab_min_score = float(
            st.slider("Model Min Score", min_value=40.0, max_value=85.0, value=58.0, step=1.0, key="pitch_lab_min_score")
        )
    with model_b:
        load_pitch_lab = st.button("Load Pitch Lab", use_container_width=True, key="pitch_lab_load")

    if load_pitch_lab:
        date_value = pitch_lab_date.isoformat() if hasattr(pitch_lab_date, "isoformat") else str(pitch_lab_date)
        markets_csv = ",".join(pitch_lab_markets or market_options)
        edge_payload, edge_error = _mlb_pitching_edges_data(
            date_value=date_value,
            season_year=pitch_lab_season_year,
            bookmaker=pitch_lab_bookmaker.strip().lower() or "draftkings",
            regions=pitch_lab_regions.strip().lower() or "us",
            markets=markets_csv,
            min_edge_pct=pitch_lab_min_edge,
            limit=pitch_lab_limit,
        )
        model_payload, model_error = _mlb_pitching_bets_data(
            date_value=date_value,
            season_year=pitch_lab_season_year,
            min_score=pitch_lab_min_score,
            limit=pitch_lab_limit,
        )
        st.session_state["pitch_lab_edge_payload"] = edge_payload
        st.session_state["pitch_lab_edge_error"] = edge_error
        st.session_state["pitch_lab_model_payload"] = model_payload
        st.session_state["pitch_lab_model_error"] = model_error

    edge_payload = st.session_state.get("pitch_lab_edge_payload") or {}
    edge_error = st.session_state.get("pitch_lab_edge_error") or ""
    model_payload = st.session_state.get("pitch_lab_model_payload") or {}
    model_error = st.session_state.get("pitch_lab_model_error") or ""
    edge_rows = edge_payload.get("recommendations") or []
    model_rows = model_payload.get("recommendations") or []

    if edge_error:
        st.warning(f"Market edges unavailable: {edge_error}")
    if model_error:
        st.warning(f"Model board unavailable: {model_error}")
    if not edge_rows and not model_rows and not edge_error and not model_error:
        st.info("Load a slate to see ranked pitcher props, market edges, and matchup visuals.")

    if edge_rows:
        best_edge = edge_rows[0]
        hero_col, gauge_col = st.columns([2.2, 1])
        with hero_col:
            _render_pitch_edge_hero(best_edge)
        with gauge_col:
            _render_gauge(best_edge.get("confidence", 0), "Market Confidence", color="#0f766e")

        edge_values = [float(row.get("edge_pct", 0) or 0) for row in edge_rows]
        confidence_values = [float(row.get("confidence", 0) or 0) for row in edge_rows]
        confirmed_count = sum(1 for row in edge_rows if "confirmed" in str(row.get("lineup_status", "")).lower())
        avg_edge = sum(edge_values) / len(edge_values) if edge_values else 0.0
        avg_conf = sum(confidence_values) / len(confidence_values) if confidence_values else 0.0
        confirmed_rate = (confirmed_count / len(edge_rows) * 100.0) if edge_rows else 0.0
        _render_metric_cards(
            [
                ("Edges Found", str(len(edge_rows))),
                ("Avg Edge", _safe_signed_pct(avg_edge)),
                ("Avg Confidence", _safe_num(avg_conf, 1)),
                ("Confirmed Lineups", _safe_pct(confirmed_rate)),
            ]
        )

        chart_col_a, chart_col_b = st.columns([1.2, 1])
        with chart_col_a:
            st.markdown("**Top Edges**")
            _render_pitch_edge_bar_chart(edge_rows)
        with chart_col_b:
            st.markdown("**Edge vs Confidence**")
            _render_pitch_scatter_chart(edge_rows)

        st.markdown("**Component Heatmap**")
        _render_pitch_component_heatmap(edge_rows)

        st.markdown("**Pick Cards**")
        _render_pitch_pick_cards(edge_rows, max_cards=min(8, len(edge_rows)))

        with st.expander("Top Pick Breakdown", expanded=True):
            reasons = best_edge.get("reasons") or []
            if reasons:
                st.markdown("**Why it rates well**")
                for reason in reasons:
                    st.write(f"- {reason}")
            lineup_players = ((best_edge.get("lineup_summary") or {}).get("players")) or []
            if lineup_players:
                st.markdown("**Projected lineup**")
                _render_table(lineup_players, max_rows=12)
            pitcher_metrics = best_edge.get("pitcher_metrics") or {}
            if pitcher_metrics:
                st.markdown("**Pitcher metrics**")
                _render_table([pitcher_metrics], max_rows=5)
            source_rows = [
                {"source": key, "detail": value}
                for key, value in (edge_payload.get("data_sources") or {}).items()
            ]
            if source_rows:
                st.markdown("**Data sources**")
                _render_table(source_rows, max_rows=10)

        with st.expander("Raw Edge Table"):
            edge_table_rows = []
            for row in edge_rows:
                edge_table_rows.append(
                    {
                        "pitcher": row.get("pitcher"),
                        "team": row.get("team"),
                        "opp": row.get("opponent"),
                        "market": row.get("market_name"),
                        "side": str(row.get("recommended_side", "")).upper(),
                        "line": row.get("line"),
                        "offered_odds": row.get("offered_odds"),
                        "fair_odds": row.get("fair_odds_american"),
                        "edge_pct": row.get("edge_pct"),
                        "proj_prob": row.get("projected_probability"),
                        "confidence": row.get("confidence"),
                        "expected_stat": row.get("expected_stat"),
                        "lineup_status": row.get("lineup_status"),
                        "book": row.get("bookmaker_title") or row.get("bookmaker"),
                    }
                )
            _render_table(edge_table_rows, max_rows=50)
    elif not edge_error and st.session_state.get("pitch_lab_edge_payload") is not None:
        st.info("No priced edges met the current threshold. Lower Min Edge % or check another bookmaker.")

    if model_rows:
        st.markdown("---")
        st.markdown("**Model Board**")
        best_model_score = max(float(row.get("score", 0) or 0) for row in model_rows)
        _render_metric_cards(
            [
                ("Pitchers Ranked", str(len(model_rows))),
                ("Top Score", _safe_num(best_model_score, 1)),
                ("Slate Date", str(model_payload.get("date") or "")),
                ("Model Version", str(model_payload.get("model_version") or "")),
            ]
        )
        model_chart_rows = [
            {
                "pick": f"{row.get('pitcher', '')} {str(row.get('lean', '')).upper()} {str(row.get('bet_type', '')).replace('_', ' ')}",
                "score": float(row.get("score", 0) or 0),
            }
            for row in model_rows[:12]
        ]
        _render_bars(model_chart_rows, "pick", "score", color="#2f80ed")

        model_table_rows = []
        for row in model_rows:
            model_table_rows.append(
                {
                    "pitcher": row.get("pitcher"),
                    "team": row.get("team"),
                    "opp": row.get("opponent"),
                    "bet_type": row.get("bet_type"),
                    "lean": str(row.get("lean", "")).upper(),
                    "score": row.get("score"),
                    "confidence": row.get("confidence"),
                    "lineup_status": row.get("lineup_status"),
                    "k_pct": (row.get("pitcher_metrics") or {}).get("k_pct"),
                    "bb_pct": (row.get("pitcher_metrics") or {}).get("bb_pct"),
                    "swstr_pct": (row.get("pitcher_metrics") or {}).get("swstr_pct"),
                    "csw_pct": (row.get("pitcher_metrics") or {}).get("csw_pct"),
                }
            )
        _render_table(model_table_rows, max_rows=40)
    elif not model_error and st.session_state.get("pitch_lab_model_payload") is not None:
        st.info("No pitchers cleared the current model score threshold for this slate.")

with tab_research:
    st.subheader("NBA Team Intel")
    col_a, col_b, col_c = st.columns([1, 1, 1])
    with col_a:
        intel_team = st.selectbox("Team", NBA_TEAMS, index=0, key="intel_team")
    with col_b:
        intel_season_type = st.selectbox("Season Type", ["Regular Season", "Playoffs"], key="intel_season_type")
    with col_c:
        questionable_only = st.checkbox("Questionable only", value=False)

    include_depth = st.checkbox("Depth chart", value=True)
    include_injuries = st.checkbox("Injury report", value=True)
    include_defense = st.checkbox("Defensive metrics", value=True)
    include_shot_zones = st.checkbox("Shot zone defense (court)", value=True)

    if st.button("Load Team Intel"):
        if not intel_team:
            st.warning("Select a team first.")
        else:
            payload, err = _team_intel_data(
                intel_team,
                intel_season_type,
                include_depth,
                include_injuries,
                include_defense,
                include_shot_zones,
            )
            if err:
                st.error(err)
            elif not payload:
                st.warning("No team intel payload returned.")
            else:
                errors = payload.get("errors") or []
                if errors:
                    st.warning("Some sections failed:\n- " + "\n- ".join(errors[:6]))

                if include_defense:
                    defense = payload.get("defensive_metrics") or {}
                    metrics = defense.get("metrics") or []
                    team_name = defense.get("team_name") or intel_team
                    st.markdown(f"**Defensive Metrics: {team_name}**")
                    if defense.get("stale"):
                        st.warning("Defensive metrics are from cached data (live provider timed out).")
                    if defense and defense.get("ranked") is False:
                        st.caption("Rankings unavailable (team-only data returned).")
                    if metrics:
                        _render_table(metrics)
                    else:
                        st.caption("No defensive metrics returned.")

                if include_shot_zones:
                    shot_profile = payload.get("shot_zones") or {}
                    zones = shot_profile.get("zones") or []
                    st.markdown("**Opponent Shot Map (FG% Allowed)**")
                    if shot_profile.get("stale"):
                        st.warning("Shot zone data is from cached results (live provider timed out).")
                    if shot_profile and shot_profile.get("ranked") is False:
                        st.caption("Rankings unavailable (team-only data returned).")
                    if zones:
                        _render_defense_court(zones)
                        st.markdown("**Shot Zone Table**")
                        _render_table(zones)
                    else:
                        st.caption("No shot zone data returned.")

                if include_depth:
                    st.markdown("**Depth Chart**")
                    depth_rows = payload.get("depth_chart") or []
                    if depth_rows:
                        _render_table(depth_rows)
                    else:
                        st.caption("No depth chart data returned.")

                if include_injuries:
                    st.markdown("**Injury Report**")
                    injuries = payload.get("injuries") or []
                    if questionable_only:
                        injuries = [row for row in injuries if "questionable" in str(row.get("status", "")).lower()]
                    if injuries:
                        _render_table(injuries)
                    else:
                        st.caption("No injury data returned.")

    st.subheader("Player Splits Without Key Teammates")
    split_col1, split_col2 = st.columns([1, 1])
    with split_col1:
        split_player = st.text_input("Player", "", key="split_player")
    with split_col2:
        split_without = st.text_input("Key players (comma-separated)", "", key="split_without")
    split_season_type = st.selectbox("Season Type", ["Regular Season", "Playoffs"], key="split_season_type")

    if st.button("Compute Splits"):
        if not split_player.strip() or not split_without.strip():
            st.warning("Enter a player and at least one key teammate.")
        else:
            payload, err = _player_splits_data(split_player.strip(), split_without.strip(), split_season_type)
            if err:
                st.error(err)
            elif not payload:
                st.warning("No player splits payload returned.")
            else:
                if payload.get("missing"):
                    st.warning("Unknown players: " + ", ".join(payload["missing"]))
                warnings = payload.get("warnings") or []
                if warnings:
                    st.warning("Notes:\n- " + "\n- ".join(warnings[:6]))

                samples = payload.get("samples") or {}
                if samples:
                    st.markdown("**Samples**")
                    _render_table([samples])

                averages = payload.get("averages") or {}
                rows = []
                for key in ("overall", "without_any", "with_all"):
                    avg = averages.get(key) or {}
                    if avg:
                        row = {"split": key}
                        row.update(avg)
                        rows.append(row)
                st.markdown("**Averages**")
                if rows:
                    _render_table(rows)
                else:
                    st.caption("No averages returned.")

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
            _render_table([provider_mode])

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
        _render_table(providers)

        cache = health.get("cache") or {}
        st.markdown("**Cache**")
        _render_table([cache])

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

            _render_table([perf])

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
            _render_table(items)

            if items:
                conf_vals = []
                fb_vals = []
                for row in items:
                    try:
                        conf_vals.append(float(row.get("confidence", 0) or 0))
                    except Exception:
                        pass
                    try:
                        fb_vals.append(1.0 if bool(row.get("fallback_used")) else 0.0)
                    except Exception:
                        pass
                if conf_vals:
                    st.metric("Avg Confidence (loaded picks)", f"{round(sum(conf_vals) / len(conf_vals), 2)}")
                if fb_vals:
                    fallback_rate = round(100.0 * (sum(fb_vals) / len(fb_vals)), 2)
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
            _render_table(rows)
