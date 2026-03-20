from fastapi import FastAPI, Query, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from nba_api.stats.endpoints import leaguedashteamshotlocations, leaguedashteamstats, playergamelog
from nba_api.stats.static import players, teams
from pydantic import BaseModel
import datetime
import hashlib
import os
import random
import time
import threading
import sqlite3
import requests
import uuid
from typing import Any, Dict, List, Optional, Tuple

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------- CONFIG ---------------- #

HIT_OPERATOR = os.getenv("HIT_OPERATOR", "gt").lower()

CONF_L5_MIN = float(os.getenv("CONF_L5_MIN", "50"))
CONF_L10_MIN = float(os.getenv("CONF_L10_MIN", "50"))
CONF_H2H_GOOD = float(os.getenv("CONF_H2H_GOOD", "60"))
CONF_LOW_MAX = float(os.getenv("CONF_LOW_MAX", "40"))

DEFAULT_H2H = float(os.getenv("DEFAULT_H2H", "50"))
DEFAULT_H2H_WITH_OPP = float(os.getenv("DEFAULT_H2H_WITH_OPP", "60"))

REC_WEIGHT_H2H = float(os.getenv("REC_WEIGHT_H2H", "0.50"))
REC_WEIGHT_L10 = float(os.getenv("REC_WEIGHT_L10", "0.30"))
REC_WEIGHT_L5 = float(os.getenv("REC_WEIGHT_L5", "0.20"))
REC_LEAN_BAND = float(os.getenv("REC_LEAN_BAND", "0.25"))

DATA_TTL_SECONDS = int(os.getenv("DATA_TTL_SECONDS", "900"))
TEAM_STATS_TTL_SECONDS = int(os.getenv("TEAM_STATS_TTL_SECONDS", "3600"))
RATE_LIMIT_MAX = int(os.getenv("RATE_LIMIT_MAX", "60"))
RATE_LIMIT_WINDOW_SECONDS = int(os.getenv("RATE_LIMIT_WINDOW_SECONDS", "60"))

DB_PATH = os.getenv("DB_PATH", "app.db")
ADMIN_SECRET = os.getenv("ADMIN_SECRET", "")

SUPPORTED_SPORTS = ("nba", "ncaab", "mlb", "nfl", "soccer", "nhl", "tennis", "golf", "cs2", "cod")
SPORT_ALIASES = {
    "nba": "nba",
    "basketball": "nba",
    "ncaab": "ncaab",
    "collegebasketball": "ncaab",
    "college-basketball": "ncaab",
    "college basketball": "ncaab",
    "ncaa": "ncaab",
    "cbb": "ncaab",
    "mlb": "mlb",
    "baseball": "mlb",
    "nfl": "nfl",
    "football": "nfl",
    "soccer": "soccer",
    "futbol": "soccer",
    "football-soccer": "soccer",
    "nhl": "nhl",
    "hockey": "nhl",
    "tennis": "tennis",
    "golf": "golf",
    "cs2": "cs2",
    "counterstrike": "cs2",
    "counter-strike": "cs2",
    "counter strike": "cs2",
    "cod": "cod",
    "callofduty": "cod",
    "call-of-duty": "cod",
    "call of duty": "cod",
}

PROP_ALIASES_BY_SPORT = {
    "nba": {
        "points": "points",
        "pts": "points",
        "rebounds": "rebounds",
        "reb": "rebounds",
        "assists": "assists",
        "ast": "assists",
        "points+rebounds": "pts+reb",
        "points+assists": "pts+ast",
        "rebounds+assists": "reb+ast",
        "pts+reb": "pts+reb",
        "pts+ast": "pts+ast",
        "reb+ast": "reb+ast",
        "pra": "pts+reb+ast",
        "pts+reb+ast": "pts+reb+ast",
    },
    "ncaab": {
        "points": "points",
        "pts": "points",
        "rebounds": "rebounds",
        "reb": "rebounds",
        "assists": "assists",
        "ast": "assists",
        "points+rebounds": "pts+reb",
        "points+assists": "pts+ast",
        "rebounds+assists": "reb+ast",
        "pts+reb": "pts+reb",
        "pts+ast": "pts+ast",
        "reb+ast": "reb+ast",
        "pra": "pts+reb+ast",
        "pts+reb+ast": "pts+reb+ast",
    },
    "mlb": {
        "hits": "hits",
        "h": "hits",
        "runs": "runs",
        "r": "runs",
        "rbis": "rbis",
        "rbi": "rbis",
        "home_runs": "home_runs",
        "hr": "home_runs",
        "total_bases": "total_bases",
        "tb": "total_bases",
        "strikeouts": "strikeouts",
        "k": "strikeouts",
    },
    "nfl": {
        "passing_yards": "passing_yards",
        "pass_yds": "passing_yards",
        "rushing_yards": "rushing_yards",
        "rush_yds": "rushing_yards",
        "receiving_yards": "receiving_yards",
        "rec_yds": "receiving_yards",
        "receptions": "receptions",
        "rec": "receptions",
        "touchdowns": "touchdowns",
        "tds": "touchdowns",
    },
    "soccer": {
        "goals": "goals",
        "assists": "assists",
        "shots": "shots",
        "shots_on_target": "shots_on_target",
        "sot": "shots_on_target",
        "passes": "passes",
    },
    "nhl": {
        "goals": "goals",
        "assists": "assists",
        "points": "points",
        "shots": "shots",
        "saves": "saves",
    },
    "tennis": {
        "aces": "aces",
        "double_faults": "double_faults",
        "first_serve_pct": "first_serve_pct",
        "break_points_won": "break_points_won",
        "games_won": "games_won",
    },
    "golf": {
        "birdies": "birdies",
        "bogeys": "bogeys",
        "pars": "pars",
        "fairways_hit": "fairways_hit",
        "greens_in_regulation": "greens_in_regulation",
    },
    "cs2": {
        "kills": "kills",
        "deaths": "deaths",
        "assists": "assists",
        "headshots": "headshots",
        "kd_ratio": "kd_ratio",
        "map_wins": "map_wins",
    },
    "cod": {
        "kills": "kills",
        "deaths": "deaths",
        "assists": "assists",
        "kd_ratio": "kd_ratio",
        "objective_kills": "objective_kills",
        "map_wins": "map_wins",
    },
}

NON_NBA_DVP_MAPS = {
    "mlb": {
        "NYY": "Strong",
        "LAD": "Average",
        "ATL": "Weak",
        "HOU": "Strong",
        "BOS": "Average",
    },
    "nfl": {
        "KC": "Average",
        "SF": "Strong",
        "BAL": "Strong",
        "DET": "Weak",
        "DAL": "Average",
    },
    "soccer": {
        "MCI": "Strong",
        "ARS": "Strong",
        "LIV": "Average",
        "RMA": "Strong",
        "BAR": "Weak",
    },
    "nhl": {
        "BOS": "Strong",
        "NYR": "Strong",
        "EDM": "Average",
        "COL": "Weak",
        "VGK": "Average",
    },
    "ncaab": {},
    "tennis": {
        "DJOKOVIC": "Strong",
        "ALCARAZ": "Strong",
        "SINNER": "Average",
        "MEDVEDEV": "Average",
    },
    "golf": {
        "SCHEFFLER": "Strong",
        "MCILROY": "Average",
        "RAHM": "Average",
        "SCHAUFFELE": "Weak",
    },
    "cs2": {
        "NAVI": "Strong",
        "FAZE": "Strong",
        "G2": "Average",
        "VITALITY": "Average",
    },
    "cod": {
        "FAZE": "Strong",
        "OPTIC": "Average",
        "ULTRA": "Average",
        "SURGE": "Weak",
    },
}

# ---------------- HELPERS ---------------- #

_rate_lock = threading.Lock()
_rate_store: Dict[str, list] = {}
_player_log_cache: Dict[tuple, dict] = {}
_team_stats_cache: Dict[tuple, dict] = {}
_team_shot_cache: Dict[tuple, dict] = {}
_external_cache: Dict[tuple, dict] = {}
EXTERNAL_TTL_SECONDS = int(os.getenv("EXTERNAL_TTL_SECONDS", "900"))
EXTERNAL_HTTP_TIMEOUT_SECONDS = float(os.getenv("EXTERNAL_HTTP_TIMEOUT_SECONDS", "8"))
EXTERNAL_HTTP_RETRIES = int(os.getenv("EXTERNAL_HTTP_RETRIES", "2"))
EXTERNAL_RETRY_BACKOFF_SECONDS = float(os.getenv("EXTERNAL_RETRY_BACKOFF_SECONDS", "0.25"))
NBA_HTTP_TIMEOUT_SECONDS = float(os.getenv("NBA_HTTP_TIMEOUT_SECONDS", "10"))
NBA_HTTP_RETRIES = int(os.getenv("NBA_HTTP_RETRIES", "2"))
NBA_RETRY_BACKOFF_SECONDS = float(os.getenv("NBA_RETRY_BACKOFF_SECONDS", "0.5"))
BALDONTLIE_API_BASE_URL = os.getenv("BALDONTLIE_API_BASE_URL", "https://api.balldontlie.io/v1")
BALDONTLIE_API_KEY = os.getenv("BALDONTLIE_API_KEY", "").strip()
BALDONTLIE_ENABLED = os.getenv("BALDONTLIE_ENABLED", "false").strip().lower() in ("1", "true", "yes", "on")
BALDONTLIE_HTTP_TIMEOUT_SECONDS = float(os.getenv("BALDONTLIE_HTTP_TIMEOUT_SECONDS", "10"))
BALDONTLIE_HTTP_RETRIES = int(os.getenv("BALDONTLIE_HTTP_RETRIES", "2"))
BALDONTLIE_RETRY_BACKOFF_SECONDS = float(os.getenv("BALDONTLIE_RETRY_BACKOFF_SECONDS", "0.3"))
PANDASCORE_API_BASE_URL = os.getenv("PANDASCORE_API_BASE_URL", "https://api.pandascore.co")
PANDASCORE_API_KEY = os.getenv("PANDASCORE_API_KEY", "").strip()
PANDASCORE_ENABLED = os.getenv("PANDASCORE_ENABLED", "false").strip().lower() in ("1", "true", "yes", "on")
PANDASCORE_HTTP_TIMEOUT_SECONDS = float(os.getenv("PANDASCORE_HTTP_TIMEOUT_SECONDS", "10"))
PANDASCORE_HTTP_RETRIES = int(os.getenv("PANDASCORE_HTTP_RETRIES", "2"))
PANDASCORE_RETRY_BACKOFF_SECONDS = float(os.getenv("PANDASCORE_RETRY_BACKOFF_SECONDS", "0.3"))
PANDASCORE_COD_GAME = os.getenv("PANDASCORE_COD_GAME", "codmw").strip().lower()
NFL_SEASON_YEAR = int(os.getenv("NFL_SEASON_YEAR", str(datetime.datetime.now().year)))
SOCCER_SEASON_YEAR = int(os.getenv("SOCCER_SEASON_YEAR", str(datetime.datetime.now().year)))
SOCCER_LEAGUE = os.getenv("SOCCER_LEAGUE", "eng.1")
SOCCER_TEAM = os.getenv("SOCCER_TEAM", "")
NCAAB_SEASON_YEAR = int(os.getenv("NCAAB_SEASON_YEAR", "0"))
TENNIS_SEASON_YEAR = int(os.getenv("TENNIS_SEASON_YEAR", str(datetime.datetime.now().year)))
TENNIS_LEAGUE = os.getenv("TENNIS_LEAGUE", "atp")
GOLF_SEASON_YEAR = int(os.getenv("GOLF_SEASON_YEAR", str(datetime.datetime.now().year)))
GOLF_LEAGUE = os.getenv("GOLF_LEAGUE", "pga")
MODEL_VERSION = os.getenv("MODEL_VERSION", "2026.02.multi.v1")
APP_BUILD = os.getenv("APP_BUILD", "2026-02-20.r1")
ODDS_API_KEY = os.getenv("ODDS_API_KEY", "")
ODDS_API_BASE_URL = os.getenv("ODDS_API_BASE_URL", "https://api.the-odds-api.com/v4")
ALERT_DISCORD_WEBHOOK_URL = os.getenv("ALERT_DISCORD_WEBHOOK_URL", "")
ALERT_MIN_EDGE_PCT = float(os.getenv("ALERT_MIN_EDGE_PCT", "3.5"))
NBA_LIVE_DISABLED = os.getenv("NBA_LIVE_DISABLED", "false").strip().lower() in ("1", "true", "yes", "on")
NBA_ESPN_FALLBACK_ENABLED = os.getenv("NBA_ESPN_FALLBACK_ENABLED", "true").strip().lower() in ("1", "true", "yes", "on")
NBA_PRIMARY_SOURCE = os.getenv("NBA_PRIMARY_SOURCE", "espn").strip().lower()

ESPN_HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; prop-analyzer/1.0)",
    "Accept": "application/json",
}
NBA_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.nba.com/",
    "Origin": "https://www.nba.com",
}
_provider_state: Dict[str, dict] = {}
_provider_lock = threading.Lock()
_provider_fail_threshold = int(os.getenv("PROVIDER_FAIL_THRESHOLD", "4"))
_provider_cooldown_seconds = int(os.getenv("PROVIDER_COOLDOWN_SECONDS", "120"))
_balldontlie_runtime_disabled_reason = ""
_pandascore_runtime_disabled_reason = ""


class AnalyzeRequestV2(BaseModel):
    player: str
    sport: str = "nba"
    prop: str
    line: float
    opponent: str = ""
    season_type: str = "Regular Season"
    window_1: int = 5
    window_2: int = 10
    hit_operator: str = ""
    conf_l5_min: Optional[float] = None
    conf_l10_min: Optional[float] = None
    conf_h2h_good: Optional[float] = None
    conf_low_max: Optional[float] = None
    offered_odds: Optional[int] = None
    include_injury: bool = False


def init_db():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS picks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT,
            sport TEXT,
            player TEXT,
            prop TEXT,
            line REAL,
            recommendation TEXT,
            confidence REAL,
            projected_probability REAL,
            offered_odds INTEGER,
            implied_probability REAL,
            edge_pct REAL,
            data_source TEXT,
            fallback_used INTEGER,
            model_version TEXT,
            result TEXT,
            actual_stat REAL,
            pnl_units REAL
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS provider_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT,
            provider TEXT,
            status TEXT,
            detail TEXT
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS app_meta (
            key TEXT PRIMARY KEY,
            value TEXT
        )
        """
    )
    conn.commit()
    conn.close()


def _rate_limit_for_identity(identity: str):
    now = time.time()
    max_hits = RATE_LIMIT_MAX
    with _rate_lock:
        hits = _rate_store.get(identity, [])
        hits = [t for t in hits if now - t < RATE_LIMIT_WINDOW_SECONDS]
        if len(hits) >= max_hits:
            retry_after = max(1, int(RATE_LIMIT_WINDOW_SECONDS - (now - hits[0])))
            return False, retry_after
        hits.append(now)
        _rate_store[identity] = hits
        return True, 0


def _now_iso():
    return datetime.datetime.utcnow().isoformat()


def _provider_name_from_url(url: str):
    if "statsapi.mlb.com" in url:
        return "mlb_statsapi"
    if "espn.com" in url:
        return "espn"
    if "balldontlie.io" in url:
        return "balldontlie"
    if "pandascore.co" in url:
        return "pandascore"
    if "the-odds-api.com" in url:
        return "odds_api"
    if "discord.com" in url:
        return "discord"
    return "generic_http"


def _balldontlie_is_enabled() -> bool:
    if not BALDONTLIE_ENABLED:
        return False
    if not BALDONTLIE_API_KEY:
        return False
    return not bool(_balldontlie_runtime_disabled_reason)


def _pandascore_is_enabled() -> bool:
    if not PANDASCORE_ENABLED:
        return False
    if not PANDASCORE_API_KEY:
        return False
    return not bool(_pandascore_runtime_disabled_reason)


def _provider_is_open(provider: str):
    if provider == "balldontlie" and not _balldontlie_is_enabled():
        return False
    if provider == "pandascore" and not _pandascore_is_enabled():
        return False
    with _provider_lock:
        state = _provider_state.get(provider, {})
        opened_at = state.get("opened_at")
        if not opened_at:
            return False
        return (time.time() - opened_at) < _provider_cooldown_seconds


def _provider_note_success(provider: str):
    with _provider_lock:
        prev = _provider_state.get(provider, {})
        _provider_state[provider] = {
            "failures": 0,
            "opened_at": None,
            "last_success_at": time.time(),
            "last_error": prev.get("last_error", ""),
        }


def _provider_note_failure(provider: str, detail: str):
    with _provider_lock:
        prev = _provider_state.get(provider, {})
        failures = int(prev.get("failures", 0)) + 1
        opened_at = prev.get("opened_at")
        if failures >= _provider_fail_threshold:
            opened_at = time.time()
        _provider_state[provider] = {
            "failures": failures,
            "opened_at": opened_at,
            "last_success_at": prev.get("last_success_at"),
            "last_error": detail,
        }
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        "INSERT INTO provider_events (created_at, provider, status, detail) VALUES (?, ?, ?, ?)",
        (_now_iso(), provider, "error", detail[:400]),
    )
    conn.commit()
    conn.close()


def get_player_id(name: str):
    needle = (name or "").strip().lower()
    if not needle:
        return None
    all_players = players.get_players()
    for p in all_players:
        if p["full_name"].lower() == needle:
            return p["id"]
    for p in all_players:
        if needle in p["full_name"].lower():
            return p["id"]
    return None


def _nba_with_retries(fn):
    provider = "nba_api"
    if _provider_is_open(provider):
        raise HTTPException(status_code=503, detail="NBA data provider is temporarily unavailable")
    retries = max(1, NBA_HTTP_RETRIES)
    last_exc = None
    for attempt in range(retries):
        try:
            result = fn()
            _provider_note_success(provider)
            return result
        except Exception as exc:
            last_exc = exc
            _provider_note_failure(provider, f"{type(exc).__name__}: {exc}")
            if attempt < retries - 1:
                time.sleep(NBA_RETRY_BACKOFF_SECONDS * (attempt + 1))
    raise HTTPException(status_code=503, detail=f"NBA data provider request failed: {type(last_exc).__name__}")


def get_team_id(abbrev: str):
    if not abbrev:
        return None
    abbrev = abbrev.strip().upper()
    for t in teams.get_teams():
        if t["abbreviation"].upper() == abbrev:
            return t["id"]
    return None


def current_season():
    now = datetime.datetime.now()
    if now.month >= 10:
        return f"{now.year}-{str(now.year+1)[2:]}"
    return f"{now.year-1}-{str(now.year)[2:]}"


def normalize_sport(sport: str) -> str:
    if not sport:
        return "nba"
    return SPORT_ALIASES.get(sport.strip().lower(), "")


def normalize_prop(prop: str, sport: str = "nba") -> str:
    aliases = PROP_ALIASES_BY_SPORT.get(sport, {})
    clean = prop.strip().lower()
    return aliases.get(clean, clean)


def supported_props(sport: str):
    aliases = PROP_ALIASES_BY_SPORT.get(sport, {})
    return sorted(set(aliases.values()))


def stat_value(prop: str, g, sport: str = "nba"):
    prop = normalize_prop(prop, sport)
    if sport == "nba":
        if prop == "points":
            return g["PTS"]
        if prop == "rebounds":
            return g["REB"]
        if prop == "assists":
            return g["AST"]
        if prop == "pts+reb":
            return g["PTS"] + g["REB"]
        if prop == "pts+ast":
            return g["PTS"] + g["AST"]
        if prop == "reb+ast":
            return g["REB"] + g["AST"]
        if prop == "pts+reb+ast":
            return g["PTS"] + g["REB"] + g["AST"]
    return None


def _nba_matchup_opponent(matchup: str) -> str:
    text = str(matchup or "").strip()
    if not text:
        return ""
    parts = text.split()
    if not parts:
        return ""
    return parts[-1].upper()


def nba_prop_game_details(df, prop: str, line: float, op: str, limit: int = 10):
    rows = []
    if df is None:
        return rows
    for _, g in df.head(max(1, int(limit))).iterrows():
        val = stat_value(prop, g, "nba")
        if val is None:
            continue
        valf = float(val)
        matchup = str(g.get("MATCHUP", ""))
        rows.append(
            {
                "date": str(g.get("GAME_DATE", "")),
                "opponent": _nba_matchup_opponent(matchup),
                "matchup": matchup,
                "prop_value": round(valf, 2),
                "line": float(line),
                "hit": _compare(valf, float(line), op),
            }
        )
    return rows


def _compare(stat: float, line: float, op: str) -> bool:
    if op == "gte":
        return stat >= line
    return stat > line


def hit_rate_details(df, prop: str, line: float, op: str, sport: str = "nba"):
    hits = 0
    n = 0
    for _, g in df.iterrows():
        val = stat_value(prop, g, sport)
        if val is None:
            continue
        n += 1
        if _compare(val, line, op):
            hits += 1
    rate = round((hits / n) * 100, 2) if n else 0
    return hits, n, rate


def wilson_interval(hits: int, n: int, z: float = 1.96):
    if n == 0:
        return (0.0, 0.0)
    phat = hits / n
    denom = 1 + (z * z) / n
    center = (phat + (z * z) / (2 * n)) / denom
    margin = (z * ((phat * (1 - phat) + (z * z) / (4 * n)) / n) ** 0.5) / denom
    low = max(0.0, (center - margin) * 100)
    high = min(100.0, (center + margin) * 100)
    return (round(low, 2), round(high, 2))


def avg_stat(df, prop: str, sport: str = "nba"):
    total = 0.0
    count = 0
    for _, g in df.iterrows():
        val = stat_value(prop, g, sport)
        if val is None:
            continue
        total += float(val)
        count += 1
    return round(total / count, 2) if count else 0


def confidence(l5: float, l10: float, h2h: float, l5_min: float, l10_min: float, h2h_good: float, low_max: float):
    # Top confidence requires all three signals to agree.
    if l5 >= l5_min and l10 >= l10_min and h2h >= h2h_good:
        return 90
    if l5 >= l5_min and l10 >= l10_min:
        return 80
    if l5 < low_max and l10 < low_max:
        return 50
    return 70


def recommendation(conf: float) -> str:
    if conf >= 85:
        return "High confidence"
    if conf >= 70:
        return "Lean"
    return "Low confidence"


def weighted_expected_stat(avg_l5: float, avg_l10: float, avg_h2h: float, has_h2h: bool) -> float:
    w_h2h = REC_WEIGHT_H2H if has_h2h else 0.0
    w_l10 = REC_WEIGHT_L10
    w_l5 = REC_WEIGHT_L5
    total = w_h2h + w_l10 + w_l5
    if total <= 0:
        return 0.0
    expected = (w_h2h * avg_h2h) + (w_l10 * avg_l10) + (w_l5 * avg_l5)
    return round(expected / total, 2)


def line_recommendation(expected: float, line: float) -> str:
    if abs(expected - line) <= REC_LEAN_BAND:
        return "Lean Over" if expected >= line else "Lean Under"
    return "Over" if expected >= line else "Under"


def _deterministic_rng(*values):
    joined = "|".join(str(v) for v in values)
    seed = int(hashlib.sha256(joined.encode("utf-8")).hexdigest()[:16], 16)
    return random.Random(seed)


def _cached_external(cache_key: tuple):
    cached = _external_cache.get(cache_key)
    if not cached:
        return None
    if time.time() - cached["ts"] >= EXTERNAL_TTL_SECONDS:
        return None
    return cached["value"]


def _set_cached_external(cache_key: tuple, value):
    _external_cache[cache_key] = {"value": value, "ts": time.time()}


def _safe_float(value, default: float = 0.0):
    try:
        if value in (None, ""):
            return default
        return float(value)
    except Exception:
        return default


def _extract_first_number(value: str) -> Optional[float]:
    if not isinstance(value, str):
        return None
    digits = []
    dot_seen = False
    in_num = False
    for ch in value:
        if ch.isdigit():
            digits.append(ch)
            in_num = True
        elif ch == "." and not dot_seen and in_num:
            digits.append(ch)
            dot_seen = True
        elif in_num:
            break
    if not digits:
        return None
    try:
        return float("".join(digits))
    except Exception:
        return None


def _numeric(val):
    if isinstance(val, (int, float)):
        return float(val)
    if isinstance(val, str):
        parsed = _extract_first_number(val)
        if parsed is not None:
            return parsed
    return None


def _mean(values: List[float]) -> float:
    if not values:
        return 0.0
    return round(sum(values) / len(values), 2)


def _ci_from_values(values: List[float], line: float, op: str):
    hits = sum(1 for v in values if _compare(v, line, op))
    n = len(values)
    rate = round((hits / n) * 100, 2) if n else 0.0
    return hits, n, rate, wilson_interval(hits, n)


def _fetch_json(url: str, params: Optional[dict] = None, headers: Optional[dict] = None):
    provider = _provider_name_from_url(url)
    if _provider_is_open(provider):
        raise HTTPException(status_code=503, detail=f"Provider {provider} is temporarily unavailable")
    last_exc = None
    retries = max(1, EXTERNAL_HTTP_RETRIES)
    timeout = max(1.0, EXTERNAL_HTTP_TIMEOUT_SECONDS)
    for attempt in range(retries):
        try:
            resp = requests.get(url, params=params or {}, headers=headers or ESPN_HEADERS, timeout=timeout)
            resp.raise_for_status()
            payload = resp.json()
            _provider_note_success(provider)
            return payload
        except Exception as exc:
            last_exc = exc
            _provider_note_failure(provider, f"{type(exc).__name__}: {exc}")
            if attempt < retries - 1:
                time.sleep(EXTERNAL_RETRY_BACKOFF_SECONDS * (attempt + 1))
    raise HTTPException(status_code=502, detail=f"Provider request failed ({provider}): {last_exc}")


def implied_probability_from_american(american_odds: Optional[int]) -> Optional[float]:
    if american_odds is None:
        return None
    try:
        odds = int(american_odds)
    except Exception:
        return None
    if odds == 0:
        return None
    if odds > 0:
        return round((100.0 / (odds + 100.0)) * 100.0, 2)
    return round(((-odds) / ((-odds) + 100.0)) * 100.0, 2)


def projected_probability(l5: float, l10: float, h2h: float, has_h2h: bool):
    h2h_w = 0.2 if has_h2h else 0.0
    l10_w = 0.5
    l5_w = 0.5 if not has_h2h else 0.3
    total = h2h_w + l10_w + l5_w
    if total <= 0:
        return 50.0
    value = ((h2h * h2h_w) + (l10 * l10_w) + (l5 * l5_w)) / total
    return round(max(1.0, min(99.0, value)), 2)


def calibrate_confidence(conf: float, proj_prob: float, max_gap: float = 8.0) -> float:
    # Keep confidence and projected probability aligned so UI signals do not conflict.
    c = _safe_float(conf, 50.0)
    p = _safe_float(proj_prob, 50.0)
    blended = (0.25 * c) + (0.75 * p)
    high = p + max_gap
    low = p - max_gap
    aligned = min(high, max(low, blended))
    return round(max(1.0, min(99.0, aligned)), 2)


def save_pick(
    sport: str,
    player: str,
    prop: str,
    line: float,
    recommendation_value: str,
    confidence_value: float,
    projected_prob: float,
    offered_odds: Optional[int],
    implied_prob: Optional[float],
    edge_pct: Optional[float],
    data_source: str,
    fallback_used: bool,
    model_version: str,
):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO picks (
            created_at, sport, player, prop, line, recommendation, confidence, projected_probability,
            offered_odds, implied_probability, edge_pct, data_source, fallback_used, model_version, result, actual_stat, pnl_units
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            _now_iso(),
            sport,
            player,
            prop,
            line,
            recommendation_value,
            float(confidence_value),
            float(projected_prob),
            offered_odds,
            implied_prob,
            edge_pct,
            data_source,
            1 if fallback_used else 0,
            model_version,
            "pending",
            None,
            None,
        ),
    )
    pick_id = int(cur.lastrowid)
    conn.commit()
    conn.close()
    return pick_id


def _send_discord_alert(content: str):
    if not ALERT_DISCORD_WEBHOOK_URL:
        return False
    try:
        requests.post(ALERT_DISCORD_WEBHOOK_URL, json={"content": content[:1900]}, timeout=10)
        return True
    except Exception:
        return False


def _season_label_to_year(season: str) -> int:
    # "2024-25" => 2024
    try:
        return int(season.split("-")[0])
    except Exception:
        return datetime.datetime.now().year


def _season_label_to_end_year(season: str) -> int:
    # "2024-25" => 2025
    try:
        return int(season.split("-")[0]) + 1
    except Exception:
        return datetime.datetime.now().year


def _bdl_headers():
    headers = {
        "Accept": "application/json",
        "User-Agent": "prop-analyzer/1.0",
    }
    if BALDONTLIE_API_KEY:
        headers["Authorization"] = f"Bearer {BALDONTLIE_API_KEY}"
    return headers


def _bdl_fetch_json(path: str, params: Optional[dict] = None):
    global _balldontlie_runtime_disabled_reason
    if not _balldontlie_is_enabled():
        raise HTTPException(status_code=503, detail="Provider balldontlie is disabled")
    base = BALDONTLIE_API_BASE_URL.rstrip("/")
    url = f"{base}/{path.lstrip('/')}"
    provider = _provider_name_from_url(url)
    if _provider_is_open(provider):
        raise HTTPException(status_code=503, detail=f"Provider {provider} is temporarily unavailable")

    last_exc = None
    retries = max(1, BALDONTLIE_HTTP_RETRIES)
    timeout = max(1.0, BALDONTLIE_HTTP_TIMEOUT_SECONDS)
    for attempt in range(retries):
        try:
            resp = requests.get(url, params=params or {}, headers=_bdl_headers(), timeout=timeout)
            resp.raise_for_status()
            payload = resp.json()
            _provider_note_success(provider)
            return payload
        except Exception as exc:
            status_code = getattr(getattr(exc, "response", None), "status_code", None)
            if status_code in (401, 403):
                _balldontlie_runtime_disabled_reason = f"auth_error_{status_code}"
            last_exc = exc
            _provider_note_failure(provider, f"{type(exc).__name__}: {exc}")
            if attempt < retries - 1:
                time.sleep(BALDONTLIE_RETRY_BACKOFF_SECONDS * (attempt + 1))
    raise HTTPException(status_code=502, detail=f"Provider request failed ({provider}): {last_exc}")


def _pandascore_headers():
    return {
        "Accept": "application/json",
        "User-Agent": "prop-analyzer/1.0",
        "Authorization": f"Bearer {PANDASCORE_API_KEY}",
    }


def _pandascore_fetch_json(path: str, params: Optional[dict] = None):
    global _pandascore_runtime_disabled_reason
    if not _pandascore_is_enabled():
        raise HTTPException(status_code=503, detail="Provider pandascore is disabled")
    base = PANDASCORE_API_BASE_URL.rstrip("/")
    url = f"{base}/{path.lstrip('/')}"
    provider = _provider_name_from_url(url)
    if _provider_is_open(provider):
        raise HTTPException(status_code=503, detail=f"Provider {provider} is temporarily unavailable")

    last_exc = None
    retries = max(1, PANDASCORE_HTTP_RETRIES)
    timeout = max(1.0, PANDASCORE_HTTP_TIMEOUT_SECONDS)
    for attempt in range(retries):
        try:
            resp = requests.get(url, params=params or {}, headers=_pandascore_headers(), timeout=timeout)
            resp.raise_for_status()
            payload = resp.json()
            _provider_note_success(provider)
            return payload
        except Exception as exc:
            status_code = getattr(getattr(exc, "response", None), "status_code", None)
            if status_code in (401, 403):
                _pandascore_runtime_disabled_reason = f"auth_error_{status_code}"
            last_exc = exc
            _provider_note_failure(provider, f"{type(exc).__name__}: {exc}")
            if attempt < retries - 1:
                time.sleep(PANDASCORE_RETRY_BACKOFF_SECONDS * (attempt + 1))
    raise HTTPException(status_code=502, detail=f"Provider request failed ({provider}): {last_exc}")


def _pandascore_game_slug(sport: str) -> str:
    if sport == "cs2":
        return "csgo"
    if sport == "cod":
        return PANDASCORE_COD_GAME or "codmw"
    return ""


def _pandascore_rows(payload):
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        data = payload.get("data")
        if isinstance(data, list):
            return data
    return []


def _pandascore_find_player(game_slug: str, player: str):
    cache_key = ("pandascore_player", game_slug, player.lower().strip())
    cached = _cached_external(cache_key)
    if cached is not None:
        return cached
    if not game_slug:
        return None

    candidate_queries = [
        {"search[name]": player, "page[size]": 50},
        {"filter[name]": player, "page[size]": 50},
        {"search": player, "page[size]": 50},
    ]
    rows = []
    for params in candidate_queries:
        try:
            payload = _pandascore_fetch_json(f"/{game_slug}/players", params=params)
            rows = _pandascore_rows(payload)
            if rows:
                break
        except HTTPException as exc:
            # Keep trying alternate query styles unless auth/provider is down.
            if exc.status_code in (401, 403, 503):
                raise
            continue

    if not rows:
        _set_cached_external(cache_key, None)
        return None

    needle = player.strip().lower()
    selected = None
    for row in rows:
        name = str(row.get("name") or row.get("slug") or "").strip().lower()
        if name == needle:
            selected = row
            break
    if selected is None:
        selected = rows[0]
    _set_cached_external(cache_key, selected)
    return selected


def _pandascore_player_matches(game_slug: str, player_id: int, player: str):
    cache_key = ("pandascore_matches", game_slug, int(player_id))
    cached = _cached_external(cache_key)
    if cached is not None:
        return cached

    query_candidates = [
        (f"/{game_slug}/players/{player_id}/matches", {"page[size]": 50}),
        (f"/{game_slug}/matches", {"filter[player_id]": player_id, "page[size]": 50}),
        (f"/{game_slug}/matches/past", {"filter[player_id]": player_id, "page[size]": 50}),
        (f"/{game_slug}/matches/past", {"search": player, "page[size]": 50}),
    ]
    rows = []
    for path, params in query_candidates:
        try:
            payload = _pandascore_fetch_json(path, params=params)
            rows = _pandascore_rows(payload)
            if rows:
                break
        except HTTPException as exc:
            if exc.status_code in (401, 403, 503):
                raise
            continue
    _set_cached_external(cache_key, rows)
    return rows


def _pandascore_collect_from_matches(game_slug: str, player_id: int, player: str, prop: str, opponent: str):
    matches = _pandascore_player_matches(game_slug, player_id, player)
    if not matches:
        return [], [], [], [], []

    metric_candidates = _sport_metric_map("cs2" if game_slug == "csgo" else "cod").get(prop, [prop])
    opp_upper = opponent.strip().upper() if opponent else ""
    values = []
    h2h_values = []
    usage_values = []
    details = []
    h2h_details = []
    needle = player.strip().lower()

    def _match_sort_key(row):
        return str(row.get("begin_at") or row.get("scheduled_at") or "")

    for match in sorted(matches, key=_match_sort_key, reverse=True):
        rows = _flatten_dict_for_metrics(match)
        relevant_rows = []
        for row in rows:
            pid = row.get("player_id")
            rid = row.get("id")
            rname = str(row.get("name") or row.get("slug") or "").strip().lower()
            match_player = False
            if pid is not None and str(pid) == str(player_id):
                match_player = True
            elif rid is not None and str(rid) == str(player_id):
                match_player = True
            elif rname and (rname == needle or needle in rname):
                match_player = True
            if match_player:
                relevant_rows.append(row)
        if not relevant_rows:
            relevant_rows = rows

        vals = _collect_metric_series(relevant_rows, metric_candidates)
        if not vals:
            # Try derived metrics for ratios.
            if prop == "kd_ratio":
                kills = _collect_metric_series(relevant_rows, ["kills"])
                deaths = _collect_metric_series(relevant_rows, ["deaths"])
                if kills and deaths and deaths[0] > 0:
                    vals = [round(kills[0] / deaths[0], 3)]
        if not vals:
            continue

        val = float(vals[0])
        values.append(val)

        opps = []
        for block in match.get("opponents", []) if isinstance(match.get("opponents"), list) else []:
            opp = block.get("opponent") if isinstance(block, dict) else {}
            if isinstance(opp, dict):
                name = str(opp.get("acronym") or opp.get("name") or opp.get("slug") or "").upper()
                if name:
                    opps.append(name)
        opponent_label = " vs ".join(opps[:2]) if opps else str(match.get("name") or "")
        detail = {
            "date": str(match.get("begin_at") or match.get("scheduled_at") or "")[:10],
            "opponent": opponent_label,
            "prop_value": round(val, 2),
            "line": None,
            "hit": None,
        }
        details.append(detail)
        if opp_upper:
            hay = f"{opponent_label} {str(match)}".upper()
            if opp_upper in hay:
                h2h_values.append(val)
                h2h_details.append(detail)

    return values, h2h_values, usage_values, details, h2h_details


def _build_live_esports_result_from_pandascore(
    sport: str,
    player: str,
    prop: str,
    line: float,
    opponent: str,
    window_1: int,
    window_2: int,
    op: str,
    conf_l5_min: float,
    conf_l10_min: float,
    conf_h2h_good: float,
    conf_low_max: float,
):
    if not _pandascore_is_enabled():
        return None
    game_slug = _pandascore_game_slug(sport)
    if not game_slug:
        return None
    player_item = _pandascore_find_player(game_slug, player)
    if not player_item:
        return None
    player_id = player_item.get("id")
    if player_id is None:
        return None

    prop_values, h2h_values, usage_values, details, h2h_details = _pandascore_collect_from_matches(
        game_slug, int(player_id), player, prop, opponent
    )
    if not prop_values:
        return None

    last_5_vals = prop_values[:window_1]
    last_10_vals = prop_values[:window_2]
    h2h_vals = h2h_values[:window_2]
    l5_hits, l5_n, l5_rate, l5_ci = _ci_from_values(last_5_vals, line, op)
    l10_hits, l10_n, l10_rate, l10_ci = _ci_from_values(last_10_vals, line, op)
    if h2h_vals:
        h2h_hits, h2h_n, h2h_rate, h2h_ci = _ci_from_values(h2h_vals, line, op)
    else:
        h2h_hits, h2h_n = 0, 0
        h2h_rate = DEFAULT_H2H_WITH_OPP if opponent else DEFAULT_H2H
        h2h_ci = (0.0, 0.0)

    avg_l5 = _mean(last_5_vals)
    avg_l10 = _mean(last_10_vals)
    avg_h2h = _mean(h2h_vals) if h2h_vals else 0.0
    conf = confidence(l5_rate, l10_rate, h2h_rate, conf_l5_min, conf_l10_min, conf_h2h_good, conf_low_max)
    expected_stat = weighted_expected_stat(avg_l5, avg_l10, avg_h2h, bool(h2h_vals))
    rec = line_recommendation(expected_stat, line)
    proj_prob = projected_probability(l5_rate, l10_rate, h2h_rate, bool(h2h_vals))
    conf = calibrate_confidence(conf, proj_prob)
    projection_label = "Map Projection"
    dvp = NON_NBA_DVP_MAPS.get(sport, {}).get(opponent.strip().upper(), "Average") if opponent else "N/A"
    reasons = [
        "Live source: PandaScore esports matches",
        f"L5/L10 hit rates: {l5_rate:.1f}% / {l10_rate:.1f}%",
        f"Expected {prop}: {expected_stat:.2f} vs line {line}",
        f"Opponent context: {opponent.upper() if opponent else 'none'} ({dvp})",
    ]
    detailed = []
    for row in details[:window_2]:
        val = float(row.get("prop_value", 0.0))
        detailed.append(
            {
                "date": row.get("date", ""),
                "opponent": row.get("opponent", ""),
                "prop_value": round(val, 2),
                "line": float(line),
                "hit": _compare(val, float(line), op),
            }
        )
    h2h_detailed = []
    for row in h2h_details[:window_2]:
        val = float(row.get("prop_value", 0.0))
        h2h_detailed.append(
            {
                "date": row.get("date", ""),
                "opponent": row.get("opponent", ""),
                "prop_value": round(val, 2),
                "line": float(line),
                "hit": _compare(val, float(line), op),
            }
        )

    return {
        "sport": sport,
        "player": player,
        "prop": prop,
        "line": line,
        "last_5_hit_rate": l5_rate,
        "last_10_hit_rate": l10_rate,
        "h2h_hit_rate": h2h_rate,
        "last_5_ci": l5_ci,
        "last_10_ci": l10_ci,
        "h2h_ci": h2h_ci,
        "last_5_avg_stat": avg_l5,
        "last_10_avg_stat": avg_l10,
        "h2h_avg_stat": avg_h2h,
        "confidence": conf,
        "projected_probability": proj_prob,
        "recommendation": rec,
        "confidence_label": recommendation(conf),
        "expected_stat": expected_stat,
        "minutes_proj": 0.0,
        "projection_label": projection_label,
        "dvp": dvp,
        "reasons": reasons,
        "data_source": "pandascore",
        "fallback_used": False,
        "source_timestamp": _now_iso(),
        "model_version": MODEL_VERSION,
        "samples": {
            "last_5_games": l5_n,
            "last_10_games": l10_n,
            "h2h_games": h2h_n,
        },
        "last_games_detail": detailed,
        "h2h_games_detail": h2h_detailed,
    }


def _extract_minutes_from_bdl(val) -> float:
    if isinstance(val, (int, float)):
        return float(val)
    if not isinstance(val, str):
        return 0.0
    txt = val.strip()
    if not txt:
        return 0.0
    if ":" in txt:
        head = txt.split(":", 1)[0]
        try:
            return float(head)
        except Exception:
            return 0.0
    try:
        return float(txt)
    except Exception:
        return 0.0


def _bdl_stat_for_prop(prop: str, row: dict) -> Optional[float]:
    pts = _safe_float(row.get("pts"))
    reb = _safe_float(row.get("reb"))
    ast = _safe_float(row.get("ast"))
    if prop == "points":
        return pts
    if prop == "rebounds":
        return reb
    if prop == "assists":
        return ast
    if prop == "pts+reb":
        return pts + reb
    if prop == "pts+ast":
        return pts + ast
    if prop == "reb+ast":
        return reb + ast
    if prop == "pts+reb+ast":
        return pts + reb + ast
    return None


def _build_live_nba_result_from_bdl(
    player: str,
    prop: str,
    line: float,
    opponent: str,
    season_type: str,
    window_1: int,
    window_2: int,
    op: str,
    l5_min: float,
    l10_min: float,
    h2h_good: float,
    low_max: float,
):
    if season_type.lower() != "regular season":
        return None

    cache_key = ("bdl_player", player.lower().strip())
    player_item = _cached_external(cache_key)
    if player_item is None:
        pdata = _bdl_fetch_json("/players", params={"search": player, "per_page": 25})
        items = pdata.get("data", []) if isinstance(pdata, dict) else []
        needle = player.strip().lower()
        player_item = None
        for item in items:
            full = f"{str(item.get('first_name', '')).strip()} {str(item.get('last_name', '')).strip()}".strip().lower()
            if full == needle:
                player_item = item
                break
        if player_item is None and items:
            player_item = items[0]
        _set_cached_external(cache_key, player_item)
    if not player_item:
        return None

    player_id = player_item.get("id")
    if not player_id:
        return None

    season = current_season()
    season_year = _season_label_to_year(season)
    season_years = [season_year, max(2000, season_year - 1)]

    # Team map for opponent abbreviations.
    teams_map_key = ("bdl_teams_map",)
    teams_map = _cached_external(teams_map_key)
    if teams_map is None:
        tdata = _bdl_fetch_json("/teams", params={"per_page": 100})
        trows = tdata.get("data", []) if isinstance(tdata, dict) else []
        teams_map = {int(t.get("id")): str(t.get("abbreviation", "")).upper() for t in trows if t.get("id") is not None}
        _set_cached_external(teams_map_key, teams_map)

    stats_rows: List[dict] = []
    for year in season_years:
        sdata = _bdl_fetch_json(
            "/stats",
            params={
                "player_ids[]": player_id,
                "seasons[]": year,
                "per_page": 100,
                "page": 1,
            },
        )
        rows = sdata.get("data", []) if isinstance(sdata, dict) else []
        stats_rows.extend(rows)
        if len(stats_rows) >= window_2:
            break
    if not stats_rows:
        return None

    def _game_sort_key(row):
        game = row.get("game", {}) if isinstance(row.get("game"), dict) else {}
        return str(game.get("date", ""))

    stats_rows = sorted(stats_rows, key=_game_sort_key, reverse=True)

    prop_values: List[float] = []
    h2h_values: List[float] = []
    usage_values: List[float] = []
    recent_details: List[dict] = []
    h2h_details: List[dict] = []
    opp_target = opponent.strip().upper() if opponent else ""

    for row in stats_rows:
        v = _bdl_stat_for_prop(prop, row)
        if v is None:
            continue
        valf = float(v)
        mins = _extract_minutes_from_bdl(row.get("min"))
        prop_values.append(valf)
        usage_values.append(mins)

        game = row.get("game", {}) if isinstance(row.get("game"), dict) else {}
        team = row.get("team", {}) if isinstance(row.get("team"), dict) else {}
        home_id = game.get("home_team_id")
        away_id = game.get("visitor_team_id")
        player_team_id = team.get("id")
        opp_abbrev = ""
        if player_team_id is not None and home_id is not None and away_id is not None:
            opp_id = away_id if int(player_team_id) == int(home_id) else home_id
            opp_abbrev = str(teams_map.get(int(opp_id), "")).upper()
        detail_row = {
            "date": str(game.get("date", ""))[:10],
            "opponent": opp_abbrev,
            "prop_value": round(valf, 2),
            "line": float(line),
            "hit": _compare(valf, float(line), op),
            "minutes": round(float(mins), 1),
        }
        recent_details.append(detail_row)

        if opp_target:
            if opp_abbrev == opp_target:
                h2h_values.append(valf)
                h2h_details.append(detail_row)

    if not prop_values:
        return None

    last_5_vals = prop_values[:window_1]
    last_10_vals = prop_values[:window_2]
    h2h_vals = h2h_values[:window_2]

    l5_hits, l5_n, l5_rate, l5_ci = _ci_from_values(last_5_vals, line, op)
    l10_hits, l10_n, l10_rate, l10_ci = _ci_from_values(last_10_vals, line, op)
    if h2h_vals:
        h2h_hits, h2h_n, h2h_rate, h2h_ci = _ci_from_values(h2h_vals, line, op)
    else:
        h2h_hits, h2h_n = 0, 0
        h2h_rate = DEFAULT_H2H_WITH_OPP if opponent else DEFAULT_H2H
        h2h_ci = (0.0, 0.0)

    avg_l5 = _mean(last_5_vals)
    avg_l10 = _mean(last_10_vals)
    avg_h2h = _mean(h2h_vals) if h2h_vals else 0.0
    conf = confidence(l5_rate, l10_rate, h2h_rate, l5_min, l10_min, h2h_good, low_max)
    expected_stat = weighted_expected_stat(avg_l5, avg_l10, avg_h2h, bool(h2h_vals))
    rec = line_recommendation(expected_stat, line)
    proj_prob = projected_probability(l5_rate, l10_rate, h2h_rate, bool(h2h_vals))
    conf = calibrate_confidence(conf, proj_prob)
    minutes_proj = round(_mean(usage_values[:window_2]) if usage_values else 0.0, 1)
    dvp = get_team_def_rating(season, season_type, opponent)

    reasons = [
        "Live source: BALldontlie game logs",
        f"L5/L10 hit rates: {l5_rate:.1f}% / {l10_rate:.1f}%",
        f"Expected {prop}: {expected_stat:.2f} vs line {line}",
        f"Opponent context: {opponent.upper() if opponent else 'none'} ({dvp})",
    ]

    return {
        "sport": "nba",
        "player": player,
        "prop": prop,
        "line": line,
        "last_5_hit_rate": l5_rate,
        "last_10_hit_rate": l10_rate,
        "h2h_hit_rate": h2h_rate,
        "last_5_ci": l5_ci,
        "last_10_ci": l10_ci,
        "h2h_ci": h2h_ci,
        "last_5_avg_stat": avg_l5,
        "last_10_avg_stat": avg_l10,
        "h2h_avg_stat": avg_h2h,
        "confidence": conf,
        "projected_probability": proj_prob,
        "recommendation": rec,
        "confidence_label": recommendation(conf),
        "expected_stat": expected_stat,
        "minutes_proj": minutes_proj,
        "projection_label": "Minutes Projection",
        "dvp": dvp,
        "reasons": reasons,
        "data_source": "balldontlie",
        "fallback_used": False,
        "source_timestamp": _now_iso(),
        "model_version": MODEL_VERSION,
        "samples": {
            "last_5_games": l5_n,
            "last_10_games": l10_n,
            "h2h_games": h2h_n,
        },
        "last_games_detail": recent_details[:window_2],
        "h2h_games_detail": h2h_details[:window_2],
    }


def _collect_nba_from_espn_payload(payload, prop: str, opponent: str):
    values = []
    h2h_values = []
    usage_values = []
    game_details = []
    h2h_game_details = []
    opp_upper = opponent.strip().upper() if opponent else ""
    names = payload.get("names", []) if isinstance(payload, dict) else []
    idx_minutes = names.index("minutes") if "minutes" in names else None
    idx_points = names.index("points") if "points" in names else None
    idx_reb = names.index("totalRebounds") if "totalRebounds" in names else None
    idx_ast = names.index("assists") if "assists" in names else None
    event_map = payload.get("events", {}) if isinstance(payload, dict) else {}

    def _pick_value(stats_row: List[str]):
        pts = _numeric(stats_row[idx_points]) if idx_points is not None and idx_points < len(stats_row) else None
        reb = _numeric(stats_row[idx_reb]) if idx_reb is not None and idx_reb < len(stats_row) else None
        ast = _numeric(stats_row[idx_ast]) if idx_ast is not None and idx_ast < len(stats_row) else None
        mins = _numeric(stats_row[idx_minutes]) if idx_minutes is not None and idx_minutes < len(stats_row) else None

        val = None
        if prop == "points":
            val = pts
        elif prop == "rebounds":
            val = reb
        elif prop == "assists":
            val = ast
        elif prop == "pts+reb" and pts is not None and reb is not None:
            val = pts + reb
        elif prop == "pts+ast" and pts is not None and ast is not None:
            val = pts + ast
        elif prop == "reb+ast" and reb is not None and ast is not None:
            val = reb + ast
        elif prop == "pts+reb+ast" and pts is not None and reb is not None and ast is not None:
            val = pts + reb + ast
        return val, mins

    # Preferred parser for ESPN gamelog structure.
    season_types = payload.get("seasonTypes", []) if isinstance(payload, dict) else []
    for season_block in season_types:
        for category in season_block.get("categories", []) or []:
            if str(category.get("type", "")).lower() != "event":
                continue
            for event_entry in category.get("events", []) or []:
                stats_row = event_entry.get("stats", []) or []
                if not isinstance(stats_row, list):
                    continue
                val, mins = _pick_value(stats_row)
                if val is None:
                    continue
                valf = float(val)
                values.append(valf)
                if mins is not None:
                    usage_values.append(float(mins))
                event_id = str(event_entry.get("eventId", ""))
                event_meta = event_map.get(event_id, {}) if isinstance(event_map, dict) else {}
                opp_abbrev = str(((event_meta.get("opponent") or {}).get("abbreviation") or "")).upper()
                detail_row = {
                    "date": str(event_meta.get("gameDate", ""))[:10],
                    "opponent": opp_abbrev,
                    "prop_value": round(valf, 2),
                    "minutes": round(float(mins), 1) if mins is not None else None,
                }
                game_details.append(detail_row)
                if opp_upper:
                    if opp_abbrev == opp_upper:
                        h2h_values.append(valf)
                        h2h_game_details.append(detail_row)

    if values:
        return values, h2h_values, usage_values, game_details, h2h_game_details

    # Fallback parser for older/different ESPN payloads.
    rows = _flatten_dict_for_metrics(payload)
    opponent_keys = ("opponent", "opponentabbrev", "opponentabbr", "opp")
    game_context_keys = set(opponent_keys) | {"date", "gamedate", "event", "eventid", "gameid"}

    for row in rows:
        row_keys = {str(k).lower().replace(" ", "").replace("_", "") for k in row.keys()}
        if row_keys.isdisjoint(game_context_keys):
            continue
        pts = None
        reb = None
        ast = None
        mins = None
        for k, v in row.items():
            key = str(k).lower().replace(" ", "").replace("_", "")
            if key in {"points", "pts"}:
                pts = _numeric(v)
            elif key in {"rebounds", "reb", "totalrebounds"}:
                reb = _numeric(v)
            elif key in {"assists", "ast"}:
                ast = _numeric(v)
            elif key in {"minutes", "min", "minutesplayed"}:
                mins = _numeric(v)
        val = None
        if prop == "points":
            val = pts
        elif prop == "rebounds":
            val = reb
        elif prop == "assists":
            val = ast
        elif prop == "pts+reb" and pts is not None and reb is not None:
            val = pts + reb
        elif prop == "pts+ast" and pts is not None and ast is not None:
            val = pts + ast
        elif prop == "reb+ast" and reb is not None and ast is not None:
            val = reb + ast
        elif prop == "pts+reb+ast" and pts is not None and reb is not None and ast is not None:
            val = pts + reb + ast
        if val is None:
            continue
        valf = float(val)
        values.append(valf)
        if mins is not None:
            usage_values.append(float(mins))
        detail_row = {
            "date": str(row.get("gameDate") or row.get("date") or ""),
            "opponent": "",
            "prop_value": round(valf, 2),
            "minutes": round(float(mins), 1) if mins is not None else None,
        }
        if opp_upper:
            row_opp = ""
            for k in opponent_keys:
                if k in row and row[k]:
                    row_opp = str(row[k]).upper()
                    break
            detail_row["opponent"] = row_opp
            if row_opp == opp_upper:
                h2h_values.append(valf)
                h2h_game_details.append(detail_row)
        game_details.append(detail_row)

    return values, h2h_values, usage_values, game_details, h2h_game_details


def _build_live_nba_result_from_espn(
    player: str,
    prop: str,
    line: float,
    opponent: str,
    season_type: str,
    window_1: int,
    window_2: int,
    op: str,
    l5_min: float,
    l10_min: float,
    h2h_good: float,
    low_max: float,
):
    if season_type.lower() != "regular season":
        return None

    season = current_season()
    # ESPN NBA gamelog season is keyed by season end-year (e.g. 2026 for 2025-26).
    season_year = _season_label_to_end_year(season)
    athlete_id = _espn_find_player_id("basketball", "nba", player)
    if not athlete_id:
        return None

    payload = _espn_gamelog_payload("basketball", "nba", athlete_id, season_year)
    values, h2h_values, usage_values, game_details, h2h_game_details = _collect_nba_from_espn_payload(payload, prop, opponent)
    if not values:
        return None

    last_5_vals = values[:window_1]
    last_10_vals = values[:window_2]
    h2h_vals = h2h_values[:window_2]

    l5_hits, l5_n, l5_rate, l5_ci = _ci_from_values(last_5_vals, line, op)
    l10_hits, l10_n, l10_rate, l10_ci = _ci_from_values(last_10_vals, line, op)
    if h2h_vals:
        h2h_hits, h2h_n, h2h_rate, h2h_ci = _ci_from_values(h2h_vals, line, op)
    else:
        h2h_hits, h2h_n = 0, 0
        h2h_rate = DEFAULT_H2H_WITH_OPP if opponent else DEFAULT_H2H
        h2h_ci = (0.0, 0.0)

    avg_l5 = _mean(last_5_vals)
    avg_l10 = _mean(last_10_vals)
    avg_h2h = _mean(h2h_vals) if h2h_vals else 0.0
    conf = confidence(l5_rate, l10_rate, h2h_rate, l5_min, l10_min, h2h_good, low_max)
    expected_stat = weighted_expected_stat(avg_l5, avg_l10, avg_h2h, bool(h2h_vals))
    rec = line_recommendation(expected_stat, line)
    proj_prob = projected_probability(l5_rate, l10_rate, h2h_rate, bool(h2h_vals))
    conf = calibrate_confidence(conf, proj_prob)
    minutes_proj = round(_mean(usage_values[:window_2]) if usage_values else _mean(last_10_vals), 1)
    dvp = get_team_def_rating(season, season_type, opponent)

    reasons = [
        "Live source: ESPN NBA game logs",
        f"Season feed: {season_year}",
        f"L5/L10 hit rates: {l5_rate:.1f}% / {l10_rate:.1f}%",
        f"Expected {prop}: {expected_stat:.2f} vs line {line}",
        f"Opponent context: {opponent.upper() if opponent else 'none'} ({dvp})",
    ]

    recent_details = []
    for row in game_details[:window_2]:
        valf = float(row.get("prop_value", 0.0))
        recent_details.append(
            {
                "date": row.get("date", ""),
                "opponent": row.get("opponent", ""),
                "prop_value": round(valf, 2),
                "line": float(line),
                "hit": _compare(valf, float(line), op),
                "minutes": row.get("minutes"),
            }
        )
    h2h_details = []
    for row in h2h_game_details[:window_2]:
        valf = float(row.get("prop_value", 0.0))
        h2h_details.append(
            {
                "date": row.get("date", ""),
                "opponent": row.get("opponent", ""),
                "prop_value": round(valf, 2),
                "line": float(line),
                "hit": _compare(valf, float(line), op),
                "minutes": row.get("minutes"),
            }
        )

    return {
        "sport": "nba",
        "player": player,
        "prop": prop,
        "line": line,
        "last_5_hit_rate": l5_rate,
        "last_10_hit_rate": l10_rate,
        "h2h_hit_rate": h2h_rate,
        "last_5_ci": l5_ci,
        "last_10_ci": l10_ci,
        "h2h_ci": h2h_ci,
        "last_5_avg_stat": avg_l5,
        "last_10_avg_stat": avg_l10,
        "h2h_avg_stat": avg_h2h,
        "confidence": conf,
        "projected_probability": proj_prob,
        "recommendation": rec,
        "confidence_label": recommendation(conf),
        "expected_stat": expected_stat,
        "minutes_proj": minutes_proj,
        "projection_label": "Minutes Projection",
        "dvp": dvp,
        "reasons": reasons,
        "data_source": "espn_nba",
        "fallback_used": False,
        "source_timestamp": _now_iso(),
        "source_season": season,
        "source_season_year": season_year,
        "model_version": MODEL_VERSION,
        "samples": {
            "last_5_games": l5_n,
            "last_10_games": l10_n,
            "h2h_games": h2h_n,
        },
        "last_games_detail": recent_details,
        "h2h_games_detail": h2h_details,
    }


def _mlb_find_player_id(player: str) -> Optional[int]:
    cache_key = ("mlb_player", player.lower())
    cached = _cached_external(cache_key)
    if cached is not None:
        return cached
    url = "https://statsapi.mlb.com/api/v1/people/search"
    data = _fetch_json(url, params={"sportId": 1, "names": player})
    people = data.get("people", [])
    player_id = int(people[0]["id"]) if people else None
    _set_cached_external(cache_key, player_id)
    return player_id


def _mlb_game_logs(player: str, season_year: int, season_type: str):
    player_id = _mlb_find_player_id(player)
    if not player_id:
        return []
    game_type = "R" if season_type == "Regular Season" else "P"
    cache_key = ("mlb_logs", player_id, season_year, game_type)
    cached = _cached_external(cache_key)
    if cached is not None:
        return cached
    url = f"https://statsapi.mlb.com/api/v1/people/{player_id}/stats"
    data = _fetch_json(
        url,
        params={"stats": "gameLog", "group": "hitting,pitching", "season": season_year, "gameType": game_type},
    )
    logs = []
    for block in data.get("stats", []):
        for split in block.get("splits", []):
            stat = split.get("stat", {})
            game = split.get("game", {})
            logs.append(
                {
                    "opponent": ((split.get("opponent") or {}).get("abbreviation") or "").upper(),
                    "hits": _safe_float(stat.get("hits")),
                    "runs": _safe_float(stat.get("runs")),
                    "rbis": _safe_float(stat.get("rbi")),
                    "home_runs": _safe_float(stat.get("homeRuns")),
                    "total_bases": _safe_float(stat.get("totalBases")),
                    "strikeouts": _safe_float(stat.get("strikeOuts") if "strikeOuts" in stat else stat.get("strikeouts")),
                    "minutes_proj": _safe_float(stat.get("plateAppearances")),
                    "game_pk": game.get("gamePk", ""),
                }
            )
    _set_cached_external(cache_key, logs)
    return logs


def _espn_find_player_id(sport_path: str, league_path: str, player: str) -> Optional[int]:
    cache_key = ("espn_player", sport_path, league_path, player.lower())
    cached = _cached_external(cache_key)
    if cached is not None:
        return cached
    # NOTE: ESPN's common/v3 athletes listing endpoint currently returns 400 for search queries.
    # Use the search API and parse athlete id from uid format: s:40~l:46~a:<id>.
    query = player.strip()
    athlete_id = None
    slug_expected = league_path.strip().lower()
    sport_expected = sport_path.strip().lower()
    try:
        search_url = "https://site.api.espn.com/apis/search/v2"
        data = _fetch_json(search_url, params={"query": query})
        blocks = data.get("results", []) if isinstance(data, dict) else []
        needle = query.lower()
        player_rows = []
        for block in blocks:
            if str(block.get("type", "")).lower() != "player":
                continue
            player_rows.extend(block.get("contents", []) or [])

        def _athlete_id_from_uid(uid: str) -> Optional[int]:
            if not uid or "~a:" not in uid:
                return None
            try:
                return int(uid.split("~a:")[-1])
            except Exception:
                return None

        best_candidate = None
        for item in player_rows:
            display_name = str(item.get("displayName", "")).strip().lower()
            default_slug = str(item.get("defaultLeagueSlug", "")).strip().lower()
            sport_name = str(item.get("sport", "")).strip().lower()
            uid = str(item.get("uid", ""))
            parsed_id = _athlete_id_from_uid(uid)
            if parsed_id is None:
                continue

            if sport_expected == "soccer":
                league_match = (sport_name == "soccer")
            else:
                league_match = (default_slug == slug_expected) if default_slug else True

            if not league_match:
                continue

            if display_name == needle:
                athlete_id = parsed_id
                break
            if best_candidate is None:
                best_candidate = parsed_id

        if athlete_id is None:
            athlete_id = best_candidate
    except Exception:
        athlete_id = None

    _set_cached_external(cache_key, athlete_id)
    return athlete_id


def _flatten_dict_for_metrics(payload) -> List[dict]:
    rows = []
    if isinstance(payload, dict):
        rows.append(payload)
        for v in payload.values():
            rows.extend(_flatten_dict_for_metrics(v))
    elif isinstance(payload, list):
        for item in payload:
            rows.extend(_flatten_dict_for_metrics(item))
    return rows


def _collect_metric_series(rows: List[dict], metric_candidates: List[str]) -> List[float]:
    out = []
    candidates = [m.lower().replace(" ", "").replace("_", "") for m in metric_candidates]
    for row in rows:
        # Keep at most one value per row to avoid double-counting the same game.
        found = None
        for candidate in candidates:
            for k, v in row.items():
                key = str(k).lower().replace(" ", "").replace("_", "")
                if key == candidate:
                    num = _numeric(v)
                    if num is not None:
                        found = num
                        break
            if found is not None:
                break
        if found is not None:
            out.append(found)
    return out


def _collect_opponents(rows: List[dict]) -> List[str]:
    opponents = []
    for row in rows:
        for k in ("opponent", "opponentabbrev", "opponentabbr", "opp"):
            if k in row and row[k]:
                opponents.append(str(row[k]).upper())
    return opponents


def _espn_gamelog_payload(sport_path: str, league_path: str, athlete_id: int, season_year: int):
    cache_key = ("espn_logs", sport_path, league_path, athlete_id, season_year)
    cached = _cached_external(cache_key)
    if cached is not None:
        return cached
    url = f"https://site.web.api.espn.com/apis/common/v3/sports/{sport_path}/{league_path}/athletes/{athlete_id}/gamelog"
    data = _fetch_json(url, params={"season": season_year})
    _set_cached_external(cache_key, data)
    return data


def _sport_metric_map(sport: str) -> dict:
    if sport == "nfl":
        return {
            "passing_yards": ["passingYards", "passYds", "passing_yards"],
            "rushing_yards": ["rushingYards", "rushYds", "rushing_yards"],
            "receiving_yards": ["receivingYards", "recYds", "receiving_yards"],
            "receptions": ["receptions", "rec"],
            "touchdowns": ["touchdowns", "totalTouchdowns", "tds", "passingTouchdowns", "rushingTouchdowns", "receivingTouchdowns"],
        }
    if sport == "soccer":
        return {
            "goals": ["goals"],
            "assists": ["assists"],
            "shots": ["shots", "totalShots"],
            "shots_on_target": ["shotsOnTarget", "sot"],
            "passes": ["passes", "passesCompleted", "accuratePasses"],
        }
    if sport == "nhl":
        return {
            "goals": ["goals"],
            "assists": ["assists"],
            "points": ["points"],
            "shots": ["shots", "shotsOnGoal"],
            "saves": ["saves"],
        }
    if sport == "tennis":
        return {
            "aces": ["aces"],
            "double_faults": ["doubleFaults", "double_faults"],
            "first_serve_pct": ["firstServePct", "firstServePercentage", "firstServePercent"],
            "break_points_won": ["breakPointsWon", "break_points_won"],
            "games_won": ["gamesWon", "games_won"],
        }
    if sport == "golf":
        return {
            "birdies": ["birdies"],
            "bogeys": ["bogeys"],
            "pars": ["pars"],
            "fairways_hit": ["fairwaysHit", "fairways_hit"],
            "greens_in_regulation": ["greensInRegulation", "greens_in_regulation", "gir"],
        }
    if sport == "cs2":
        return {
            "kills": ["kills", "kill"],
            "deaths": ["deaths", "death"],
            "assists": ["assists", "assist"],
            "headshots": ["headshots", "headshot_kills", "hs"],
            "kd_ratio": ["kd", "kdr", "kd_ratio"],
            "map_wins": ["map_wins", "maps_won", "wins"],
        }
    if sport == "cod":
        return {
            "kills": ["kills", "kill"],
            "deaths": ["deaths", "death"],
            "assists": ["assists", "assist"],
            "kd_ratio": ["kd", "kdr", "kd_ratio"],
            "objective_kills": ["objective_kills", "objectiveKills", "obj_kills"],
            "map_wins": ["map_wins", "maps_won", "wins"],
        }
    return {}


def _collect_from_mlb(logs: List[dict], prop: str, opponent: str):
    prop_vals = []
    h2h_vals = []
    usage = []
    for g in logs:
        v = _numeric(g.get(prop))
        if v is None:
            continue
        prop_vals.append(v)
        usage_val = _numeric(g.get("minutes_proj"))
        if usage_val is not None:
            usage.append(usage_val)
        if opponent and str(g.get("opponent", "")).upper() == opponent.upper():
            h2h_vals.append(v)
    return prop_vals, h2h_vals, usage


def _collect_from_espn_payload(payload, metric_candidates: List[str], opponent: str):
    rows = _flatten_dict_for_metrics(payload)
    values = []
    h2h_values = []
    opp_upper = opponent.strip().upper() if opponent else ""
    opponent_keys = {"opponent", "opponentabbrev", "opponentabbr", "opp"}
    game_context_keys = opponent_keys | {"date", "gamedate", "event", "eventid", "gameid"}

    for row in rows:
        row_keys = {str(k).lower().replace(" ", "").replace("_", "") for k in row.keys()}
        # Skip obvious non-game aggregate rows.
        if row_keys.isdisjoint(game_context_keys):
            continue

        row_values = _collect_metric_series([row], metric_candidates)
        if not row_values:
            continue

        val = row_values[0]
        values.append(val)
        if opp_upper:
            row_opp = ""
            for k in opponent_keys:
                if k in row and row[k]:
                    row_opp = str(row[k]).upper()
                    break
            if row_opp == opp_upper:
                h2h_values.append(val)

    # Fallback for provider payload changes that don't expose game context keys.
    if not values:
        values = _collect_metric_series(rows, metric_candidates)
        opponents = _collect_opponents(rows)
        if opp_upper and values:
            for idx, val in enumerate(values):
                if idx < len(opponents) and opponents[idx] == opp_upper:
                    h2h_values.append(val)

    return values, h2h_values


def get_injury_context(sport: str, player: str):
    if not player.strip():
        return {"status": "unknown", "detail": "Player missing"}
    league_map = {
        "nba": ("basketball", "nba"),
        "ncaab": ("basketball", "mens-college-basketball"),
        "nfl": ("football", "nfl"),
        "soccer": ("soccer", SOCCER_LEAGUE if "." in SOCCER_LEAGUE else "eng.1"),
        "nhl": ("hockey", "nhl"),
        "tennis": ("tennis", TENNIS_LEAGUE),
        "golf": ("golf", GOLF_LEAGUE),
        "mlb": None,
        "cs2": None,
        "cod": None,
    }
    league = league_map.get(sport)
    if league is None:
        return {"status": "unknown", "detail": "Injury feed not configured for this sport"}
    sport_path, league_path = league
    try:
        url = f"https://site.web.api.espn.com/apis/common/v3/sports/{sport_path}/{league_path}/athletes"
        data = _fetch_json(url, params={"limit": 5, "page": 1, "search": player})
        items = data.get("items", []) or data.get("athletes", [])
        if not items:
            return {"status": "unknown", "detail": "No injury data found"}
        item = items[0]
        status = item.get("injuryStatus") or item.get("status", {}).get("type", {}).get("name") or "active"
        desc = item.get("injuries", [{}])[0].get("shortComment") if item.get("injuries") else ""
        return {"status": str(status), "detail": str(desc or "No active injury report")}
    except Exception:
        return {"status": "unknown", "detail": "Injury source unavailable"}


def _build_live_multi_sport_result(
    sport: str,
    player: str,
    prop: str,
    line: float,
    opponent: str,
    window_1: int,
    window_2: int,
    op: str,
    conf_l5_min: float,
    conf_l10_min: float,
    conf_h2h_good: float,
    conf_low_max: float,
    season_type: str,
):
    season = current_season()
    season_year = _season_label_to_year(season)
    prop_values: List[float] = []
    h2h_values: List[float] = []
    usage_values: List[float] = []
    dvp = "Average"
    last_games_detail: List[dict] = []
    h2h_games_detail: List[dict] = []

    if sport == "mlb":
        logs = _mlb_game_logs(player, season_year, season_type)
        prop_values, h2h_values, usage_values = _collect_from_mlb(logs, prop, opponent)
    elif sport == "nfl":
        athlete_id = _espn_find_player_id("football", "nfl", player)
        if athlete_id:
            payload = _espn_gamelog_payload("football", "nfl", athlete_id, NFL_SEASON_YEAR)
            prop_values, h2h_values = _collect_from_espn_payload(payload, _sport_metric_map("nfl").get(prop, []), opponent)
    elif sport == "soccer":
        league = SOCCER_LEAGUE
        if "." in league:
            sport_path, league_path = "soccer", league
        else:
            sport_path, league_path = "soccer", "eng.1"
        athlete_id = _espn_find_player_id(sport_path, league_path, player)
        if athlete_id:
            payload = _espn_gamelog_payload(sport_path, league_path, athlete_id, SOCCER_SEASON_YEAR)
            prop_values, h2h_values = _collect_from_espn_payload(payload, _sport_metric_map("soccer").get(prop, []), opponent)
            if SOCCER_TEAM and not opponent:
                dvp = f"League: {league_path.upper()}"
    elif sport == "nhl":
        athlete_id = _espn_find_player_id("hockey", "nhl", player)
        if athlete_id:
            payload = _espn_gamelog_payload("hockey", "nhl", athlete_id, season_year)
            prop_values, h2h_values = _collect_from_espn_payload(payload, _sport_metric_map("nhl").get(prop, []), opponent)
    elif sport == "ncaab":
        league_path = "mens-college-basketball"
        athlete_id = _espn_find_player_id("basketball", league_path, player)
        if athlete_id:
            season_year_end = NCAAB_SEASON_YEAR if NCAAB_SEASON_YEAR > 0 else _season_label_to_end_year(season)
            payload = _espn_gamelog_payload("basketball", league_path, athlete_id, season_year_end)
            values, h2h_vals, usage_vals, game_details, h2h_game_details = _collect_nba_from_espn_payload(payload, prop, opponent)
            prop_values, h2h_values, usage_values = values, h2h_vals, usage_vals
            for row in game_details[:window_2]:
                valf = float(row.get("prop_value", 0.0))
                last_games_detail.append(
                    {
                        "date": row.get("date", ""),
                        "opponent": row.get("opponent", ""),
                        "prop_value": round(valf, 2),
                        "line": float(line),
                        "hit": _compare(valf, float(line), op),
                        "minutes": row.get("minutes"),
                    }
                )
            for row in h2h_game_details[:window_2]:
                valf = float(row.get("prop_value", 0.0))
                h2h_games_detail.append(
                    {
                        "date": row.get("date", ""),
                        "opponent": row.get("opponent", ""),
                        "prop_value": round(valf, 2),
                        "line": float(line),
                        "hit": _compare(valf, float(line), op),
                        "minutes": row.get("minutes"),
                    }
                )
    elif sport == "tennis":
        league_path = TENNIS_LEAGUE or "atp"
        athlete_id = _espn_find_player_id("tennis", league_path, player)
        if athlete_id:
            payload = _espn_gamelog_payload("tennis", league_path, athlete_id, TENNIS_SEASON_YEAR)
            prop_values, h2h_values = _collect_from_espn_payload(payload, _sport_metric_map("tennis").get(prop, []), opponent)
            if not opponent:
                dvp = f"Tour: {league_path.upper()}"
    elif sport == "golf":
        league_path = GOLF_LEAGUE or "pga"
        athlete_id = _espn_find_player_id("golf", league_path, player)
        if athlete_id:
            payload = _espn_gamelog_payload("golf", league_path, athlete_id, GOLF_SEASON_YEAR)
            prop_values, h2h_values = _collect_from_espn_payload(payload, _sport_metric_map("golf").get(prop, []), opponent)
            if not opponent:
                dvp = f"Tour: {league_path.upper()}"
    elif sport in ("cs2", "cod"):
        esports_result = _build_live_esports_result_from_pandascore(
            sport=sport,
            player=player,
            prop=prop,
            line=line,
            opponent=opponent,
            window_1=window_1,
            window_2=window_2,
            op=op,
            conf_l5_min=conf_l5_min,
            conf_l10_min=conf_l10_min,
            conf_h2h_good=conf_h2h_good,
            conf_low_max=conf_low_max,
        )
        if esports_result:
            return esports_result
        if not _pandascore_is_enabled():
            reason = _pandascore_runtime_disabled_reason or "disabled_or_missing_key"
            raise HTTPException(status_code=503, detail=f"PandaScore provider unavailable: {reason}")

    if not prop_values:
        return None

    last_5_vals = prop_values[:window_1]
    last_10_vals = prop_values[:window_2]
    h2h_vals = h2h_values[:window_2]

    l5_hits, l5_n, l5_rate, l5_ci = _ci_from_values(last_5_vals, line, op)
    l10_hits, l10_n, l10_rate, l10_ci = _ci_from_values(last_10_vals, line, op)
    if h2h_vals:
        h2h_hits, h2h_n, h2h_rate, h2h_ci = _ci_from_values(h2h_vals, line, op)
    else:
        h2h_hits, h2h_n = 0, 0
        h2h_rate = DEFAULT_H2H_WITH_OPP if opponent else DEFAULT_H2H
        h2h_ci = (0.0, 0.0)

    avg_l5 = _mean(last_5_vals)
    avg_l10 = _mean(last_10_vals)
    avg_h2h = _mean(h2h_vals) if h2h_vals else 0.0
    conf = confidence(l5_rate, l10_rate, h2h_rate, conf_l5_min, conf_l10_min, conf_h2h_good, conf_low_max)
    expected_stat = weighted_expected_stat(avg_l5, avg_l10, avg_h2h, bool(h2h_vals))
    rec = line_recommendation(expected_stat, line)
    proj_prob = projected_probability(l5_rate, l10_rate, h2h_rate, bool(h2h_vals))
    conf = calibrate_confidence(conf, proj_prob)
    minutes_proj = round(_mean(usage_values) if usage_values else _mean(last_10_vals), 1)
    projection_label = {
        "mlb": "Plate Appearances",
        "nfl": "Usage Projection",
        "soccer": "Minutes Projection",
        "nhl": "TOI Projection",
        "ncaab": "Minutes Projection",
        "tennis": "Set Projection",
        "golf": "Round Projection",
        "cs2": "Map Projection",
        "cod": "Map Projection",
    }.get(sport, "Projection")

    if not opponent:
        dvp = "N/A"
    elif dvp == "Average":
        dvp = NON_NBA_DVP_MAPS.get(sport, {}).get(opponent.strip().upper(), "Average")

    reasons = [
        f"Live source: {sport.upper()} game logs",
        f"L5/L10 hit rates: {l5_rate:.1f}% / {l10_rate:.1f}%",
        f"Expected {prop}: {expected_stat:.2f} vs line {line}",
        f"Opponent context: {opponent.upper() if opponent else 'none'} ({dvp})",
    ]

    return {
        "sport": sport,
        "player": player,
        "prop": prop,
        "line": line,
        "last_5_hit_rate": l5_rate,
        "last_10_hit_rate": l10_rate,
        "h2h_hit_rate": h2h_rate,
        "last_5_ci": l5_ci,
        "last_10_ci": l10_ci,
        "h2h_ci": h2h_ci,
        "last_5_avg_stat": avg_l5,
        "last_10_avg_stat": avg_l10,
        "h2h_avg_stat": avg_h2h,
        "confidence": conf,
        "projected_probability": proj_prob,
        "recommendation": rec,
        "confidence_label": recommendation(conf),
        "expected_stat": expected_stat,
        "minutes_proj": minutes_proj,
        "projection_label": projection_label,
        "dvp": dvp,
        "reasons": reasons,
        "data_source": "live_external",
        "fallback_used": False,
        "source_timestamp": _now_iso(),
        "model_version": MODEL_VERSION,
        "samples": {
            "last_5_games": l5_n,
            "last_10_games": l10_n,
            "h2h_games": h2h_n,
        },
        "last_games_detail": last_games_detail,
        "h2h_games_detail": h2h_games_detail,
    }


def build_multi_sport_fallback(
    sport: str,
    player: str,
    prop: str,
    line: float,
    opponent: str,
    window_1: int,
    window_2: int,
    conf_l5_min: float,
    conf_l10_min: float,
    conf_h2h_good: float,
    conf_low_max: float,
):
    rng = _deterministic_rng(sport, player, prop, line, opponent, window_1, window_2)
    base_rate = {
        "mlb": 51.0,
        "nfl": 54.0,
        "soccer": 49.0,
        "nhl": 50.0,
        "tennis": 52.0,
        "golf": 50.0,
        "cs2": 53.0,
        "cod": 52.0,
    }.get(sport, 50.0)
    spread = {
        "mlb": 18.0,
        "nfl": 14.0,
        "soccer": 17.0,
        "nhl": 16.0,
        "tennis": 13.0,
        "golf": 15.0,
        "cs2": 14.0,
        "cod": 14.0,
    }.get(sport, 15.0)
    stat_spread = {
        "mlb": 1.6,
        "nfl": 18.0,
        "soccer": 1.2,
        "nhl": 1.4,
        "tennis": 2.4,
        "golf": 1.8,
        "cs2": 6.0,
        "cod": 6.0,
    }.get(sport, 1.5)

    last_5_hit_rate = round(max(5.0, min(95.0, base_rate + rng.uniform(-spread, spread))), 2)
    last_10_hit_rate = round(max(5.0, min(95.0, base_rate + rng.uniform(-spread, spread))), 2)

    if opponent:
        h2h_n = max(2, min(window_2, int(rng.uniform(2, 8))))
        h2h_hit_rate = round(max(5.0, min(95.0, base_rate + rng.uniform(-spread, spread))), 2)
    else:
        h2h_n = 0
        h2h_hit_rate = DEFAULT_H2H

    last_5_avg_stat = round(max(0.0, line + rng.uniform(-stat_spread, stat_spread)), 2)
    last_10_avg_stat = round(max(0.0, line + rng.uniform(-stat_spread, stat_spread)), 2)
    h2h_avg_stat = round(max(0.0, line + rng.uniform(-stat_spread, stat_spread)), 2) if h2h_n else 0.0

    l5_hits = int(round((last_5_hit_rate / 100) * window_1))
    l10_hits = int(round((last_10_hit_rate / 100) * window_2))
    h2h_hits = int(round((h2h_hit_rate / 100) * h2h_n)) if h2h_n else 0

    conf = confidence(last_5_hit_rate, last_10_hit_rate, h2h_hit_rate, conf_l5_min, conf_l10_min, conf_h2h_good, conf_low_max)
    expected_stat = weighted_expected_stat(last_5_avg_stat, last_10_avg_stat, h2h_avg_stat, bool(h2h_n))
    rec = line_recommendation(expected_stat, line)
    proj_prob = projected_probability(last_5_hit_rate, last_10_hit_rate, h2h_hit_rate, bool(h2h_n))
    conf = calibrate_confidence(conf, proj_prob)

    projection_label = {
        "mlb": "Plate Appearances",
        "nfl": "Usage Projection",
        "soccer": "Minutes Projection",
        "nhl": "TOI Projection",
        "ncaab": "Minutes Projection",
        "tennis": "Set Projection",
        "golf": "Round Projection",
        "cs2": "Map Projection",
        "cod": "Map Projection",
    }.get(sport, "Projection")
    projection_range = {
        "mlb": (3.5, 5.0),
        "nfl": (40.0, 85.0),
        "soccer": (70.0, 95.0),
        "nhl": (13.0, 24.0),
        "tennis": (2.0, 3.0),
        "golf": (1.0, 4.0),
        "cs2": (1.0, 3.0),
        "cod": (1.0, 3.0),
    }.get(sport, (0.0, 0.0))
    minutes_proj = round(rng.uniform(projection_range[0], projection_range[1]), 1)

    dvp = NON_NBA_DVP_MAPS.get(sport, {}).get(opponent.strip().upper(), "Average") if opponent else "Average"

    reasons = [
        f"Sport model: {sport.upper()}",
        f"Recent hit rates: L5 {last_5_hit_rate:.1f}% and L10 {last_10_hit_rate:.1f}%",
        f"Expected {prop}: {expected_stat:.2f} versus line {line}",
        f"Opponent context: {opponent.upper() if opponent else 'none'} ({dvp})",
    ]

    return {
        "sport": sport,
        "player": player,
        "prop": prop,
        "line": line,
        "last_5_hit_rate": last_5_hit_rate,
        "last_10_hit_rate": last_10_hit_rate,
        "h2h_hit_rate": h2h_hit_rate,
        "last_5_ci": wilson_interval(l5_hits, window_1),
        "last_10_ci": wilson_interval(l10_hits, window_2),
        "h2h_ci": wilson_interval(h2h_hits, h2h_n) if h2h_n else (0.0, 0.0),
        "last_5_avg_stat": last_5_avg_stat,
        "last_10_avg_stat": last_10_avg_stat,
        "h2h_avg_stat": h2h_avg_stat,
        "confidence": conf,
        "projected_probability": proj_prob,
        "recommendation": rec,
        "confidence_label": recommendation(conf),
        "expected_stat": expected_stat,
        "minutes_proj": minutes_proj,
        "projection_label": projection_label,
        "dvp": dvp,
        "reasons": reasons,
        "data_source": "fallback_model",
        "fallback_used": True,
        "source_timestamp": _now_iso(),
        "model_version": MODEL_VERSION,
        "samples": {
            "last_5_games": window_1,
            "last_10_games": window_2,
            "h2h_games": h2h_n,
        },
        "last_games_detail": [],
        "h2h_games_detail": [],
    }


def filter_h2h(df, opponent: str):
    if not opponent:
        return df
    opp = opponent.strip().upper()
    return df[df["MATCHUP"].str.contains(f" {opp}", case=False, na=False)]


def dvp_label(def_rating: float, percent: float) -> str:
    if percent <= 0.2:
        return f"Elite (Def Rtg {def_rating:.1f})"
    if percent <= 0.4:
        return f"Strong (Def Rtg {def_rating:.1f})"
    if percent <= 0.6:
        return f"Average (Def Rtg {def_rating:.1f})"
    if percent <= 0.8:
        return f"Weak (Def Rtg {def_rating:.1f})"
    return f"Poor (Def Rtg {def_rating:.1f})"


def get_team_stats_df(season: str, season_type: str):
    cache_key = (season, season_type)
    now = time.time()
    cached = _team_stats_cache.get(cache_key)
    if cached and (now - cached["ts"] < TEAM_STATS_TTL_SECONDS):
        return cached["df"]

    def fetch_stats():
        try:
            endpoint = leaguedashteamstats.LeagueDashTeamStats(
                season=season,
                season_type_all_star=season_type,
                timeout=NBA_HTTP_TIMEOUT_SECONDS,
                headers=NBA_HEADERS,
            )
        except TypeError:
            endpoint = leaguedashteamstats.LeagueDashTeamStats(
                season=season,
                season_type_all_star=season_type,
            )
        return endpoint.get_data_frames()[0]

    stats = _nba_with_retries(fetch_stats)
    _team_stats_cache[cache_key] = {"df": stats, "ts": now}
    return stats


def get_team_def_rating(season: str, season_type: str, opponent: str):
    team_id = get_team_id(opponent)
    if not team_id:
        return "Unknown"
    try:
        stats = get_team_stats_df(season, season_type)
    except HTTPException:
        return "Unknown"
    if "DEF_RATING" not in stats.columns:
        return "Unknown"
    stats = stats.sort_values("DEF_RATING").reset_index(drop=True)
    idx = stats.index[stats["TEAM_ID"] == team_id]
    if len(idx) == 0:
        return "Unknown"
    rank = int(idx[0]) + 1
    percent = rank / len(stats)
    def_rating = float(stats.loc[idx[0], "DEF_RATING"])
    return dvp_label(def_rating, percent)


def _espn_team_map():
    cache_key = ("espn_team_map", "nba")
    cached = _cached_external(cache_key)
    if cached is not None:
        return cached
    url = "https://site.web.api.espn.com/apis/site/v2/sports/basketball/nba/teams"
    payload = _fetch_json(url)
    teams_list = []
    if isinstance(payload, dict):
        if isinstance(payload.get("sports"), list) and payload["sports"]:
            leagues = payload["sports"][0].get("leagues", [])
            if leagues and isinstance(leagues[0], dict):
                teams_list = leagues[0].get("teams", []) or []
        if not teams_list and isinstance(payload.get("teams"), list):
            teams_list = payload.get("teams", []) or []
    mapping = {}
    for item in teams_list:
        team = item.get("team") if isinstance(item, dict) else None
        if team is None and isinstance(item, dict):
            team = item
        if not isinstance(team, dict):
            continue
        abbrev = team.get("abbreviation") or team.get("abbr") or team.get("shortName")
        team_id = team.get("id")
        if not abbrev or not team_id:
            continue
        display = team.get("displayName") or team.get("shortDisplayName") or team.get("name") or abbrev
        mapping[str(abbrev).upper()] = {"id": str(team_id), "name": str(display)}
    _set_cached_external(cache_key, mapping)
    return mapping


def _espn_team_id_for_abbrev(abbrev: str) -> Optional[str]:
    if not abbrev:
        return None
    mapping = _espn_team_map()
    entry = mapping.get(abbrev.strip().upper())
    if not entry:
        return None
    return entry.get("id")


def _parse_espn_depth_chart(payload: dict) -> List[dict]:
    if not isinstance(payload, dict):
        return []
    positions = []
    if isinstance(payload.get("positions"), list):
        positions = payload.get("positions", [])
    elif isinstance(payload.get("depthChart"), dict):
        positions = payload["depthChart"].get("positions") or []
    rows = []
    for pos in positions or []:
        if not isinstance(pos, dict):
            continue
        pos_name = pos.get("position") or pos.get("name") or pos.get("displayName") or pos.get("abbreviation")
        athletes = pos.get("athletes") or pos.get("athlete") or []
        if not isinstance(athletes, list):
            continue
        for entry in athletes:
            if not isinstance(entry, dict):
                continue
            athlete = entry.get("athlete") or entry.get("player") or {}
            if not isinstance(athlete, dict):
                athlete = {}
            name = athlete.get("displayName") or athlete.get("fullName") or athlete.get("shortName") or athlete.get("name")
            if not name:
                continue
            status = entry.get("injuryStatus") or entry.get("status") or athlete.get("injuryStatus")
            if isinstance(status, dict):
                status = status.get("type", {}).get("name") or status.get("name")
            if isinstance(status, dict):
                status = None
            rank = entry.get("rank") or entry.get("depth") or entry.get("order") or entry.get("positionRank")
            rows.append(
                {
                    "position": str(pos_name or ""),
                    "slot": rank,
                    "player": str(name),
                    "status": str(status) if status else "",
                }
            )
    return rows


def _espn_team_depth_chart(team_id: str) -> List[dict]:
    if not team_id:
        return []
    cache_key = ("espn_depth_chart", team_id)
    cached = _cached_external(cache_key)
    if cached is not None:
        return cached or []
    url = f"https://site.web.api.espn.com/apis/site/v2/sports/basketball/nba/teams/{team_id}/depthchart"
    payload = _fetch_json(url)
    rows = _parse_espn_depth_chart(payload or {})
    _set_cached_external(cache_key, rows)
    return rows


def _parse_espn_injuries(payload: dict) -> List[dict]:
    if not isinstance(payload, dict):
        return []
    injuries_list = payload.get("injuries")
    if not isinstance(injuries_list, list):
        team_obj = payload.get("team", {}) if isinstance(payload.get("team"), dict) else {}
        injuries_list = team_obj.get("injuries")
    if not isinstance(injuries_list, list):
        return []
    rows = []
    for item in injuries_list:
        if not isinstance(item, dict):
            continue
        athlete = item.get("athlete") or item.get("player") or {}
        if not isinstance(athlete, dict):
            athlete = {}
        name = athlete.get("displayName") or athlete.get("fullName") or athlete.get("shortName") or athlete.get("name")
        status = item.get("status") or item.get("injuryStatus") or item.get("type")
        if isinstance(status, dict):
            status = status.get("name") or status.get("type", {}).get("name")
        if isinstance(status, dict):
            status = None
        detail = item.get("details") or item.get("description") or item.get("injury") or ""
        if isinstance(detail, dict):
            detail = detail.get("detail") or detail.get("description") or detail.get("type") or ""
        date_val = item.get("date") or item.get("startDate") or item.get("updated")
        rows.append(
            {
                "player": str(name or ""),
                "status": str(status or ""),
                "detail": str(detail or ""),
                "date": str(date_val or ""),
            }
        )
    return rows


def _espn_team_injuries(team_id: str) -> List[dict]:
    if not team_id:
        return []
    cache_key = ("espn_team_injuries", team_id)
    cached = _cached_external(cache_key)
    if cached is not None:
        return cached or []
    url = f"https://site.web.api.espn.com/apis/site/v2/sports/basketball/nba/teams/{team_id}/injuries"
    payload = _fetch_json(url)
    rows = _parse_espn_injuries(payload or {})
    _set_cached_external(cache_key, rows)
    return rows


def _rank_metric(stats, team_id: int, column: str, ascending: bool) -> Tuple[Optional[int], Optional[float]]:
    if column not in stats.columns:
        return None, None
    ordered = stats.sort_values(column, ascending=ascending).reset_index(drop=True)
    idx = ordered.index[ordered["TEAM_ID"] == team_id]
    if len(idx) == 0:
        return None, None
    rank = int(idx[0]) + 1
    total = len(ordered)
    if total <= 1:
        pct = 100.0
    else:
        pct = round((1 - ((rank - 1) / (total - 1))) * 100.0, 2)
    return rank, pct


def get_team_defensive_metrics(season: str, season_type: str, team_abbrev: str) -> Dict[str, Any]:
    team_id = get_team_id(team_abbrev)
    if not team_id:
        raise HTTPException(status_code=404, detail="Team not found")
    stats = get_team_stats_df(season, season_type)
    if stats is None or "TEAM_ID" not in stats.columns:
        raise HTTPException(status_code=503, detail="Team stats unavailable")
    row = stats.loc[stats["TEAM_ID"] == team_id]
    if row.empty:
        raise HTTPException(status_code=404, detail="Team not found in stats")
    row = row.iloc[0]
    metrics = []
    candidates = [
        ("DEF_RATING", "Def Rtg", True),
        ("OPP_PTS", "Opp PTS", True),
        ("OPP_FG_PCT", "Opp FG%", True),
        ("OPP_FG2_PCT", "Opp 2P%", True),
        ("OPP_FG3_PCT", "Opp 3P%", True),
        ("OPP_EFG_PCT", "Opp eFG%", True),
        ("OPP_FG3A", "Opp 3PA", True),
        ("OPP_FGA", "Opp FGA", True),
        ("OPP_FT_PCT", "Opp FT%", True),
        ("OPP_FTA", "Opp FTA", True),
        ("OPP_REB", "Opp REB", True),
        ("OPP_OREB", "Opp OREB", True),
        ("OPP_TOV", "Opp TOV", False),
    ]
    for column, label, ascending in candidates:
        if column not in stats.columns:
            continue
        val = _numeric(row.get(column))
        rank, pct = _rank_metric(stats, team_id, column, ascending)
        metrics.append(
            {
                "metric": label,
                "value": round(val, 4) if val is not None else None,
                "rank": rank,
                "percentile": pct,
                "better": "lower" if ascending else "higher",
            }
        )
    return {
        "team_abbrev": team_abbrev.strip().upper(),
        "team_name": str(row.get("TEAM_NAME", "")),
        "season": season,
        "season_type": season_type,
        "metrics": metrics,
    }


def _rank_metric_by_index(stats, team_id: int, col_index: int, ascending: bool) -> Tuple[Optional[int], Optional[float]]:
    if stats is None or col_index < 0 or col_index >= len(stats.columns):
        return None, None
    values = []
    for _, row in stats.iterrows():
        tid = row.get("TEAM_ID")
        val = _numeric(row.iloc[col_index])
        if tid is None or val is None:
            continue
        values.append((int(tid), float(val)))
    if not values:
        return None, None
    values = sorted(values, key=lambda x: x[1], reverse=not ascending)
    total = len(values)
    rank = None
    for idx, (tid, _) in enumerate(values):
        if tid == int(team_id):
            rank = idx + 1
            break
    if rank is None:
        return None, None
    if total <= 1:
        pct = 100.0
    else:
        pct = round((1 - ((rank - 1) / (total - 1))) * 100.0, 2)
    return rank, pct


def get_team_shot_locations_df(season: str, season_type: str, measure_type: str):
    cache_key = ("shot_locations", season, season_type, measure_type)
    now = time.time()
    cached = _team_shot_cache.get(cache_key)
    if cached and (now - cached["ts"] < TEAM_STATS_TTL_SECONDS):
        return cached["df"]

    def fetch_stats():
        try:
            endpoint = leaguedashteamshotlocations.LeagueDashTeamShotLocations(
                season=season,
                season_type_all_star=season_type,
                measure_type_simple=measure_type,
                timeout=NBA_HTTP_TIMEOUT_SECONDS,
                headers=NBA_HEADERS,
            )
        except TypeError:
            endpoint = leaguedashteamshotlocations.LeagueDashTeamShotLocations(
                season=season,
                season_type_all_star=season_type,
                measure_type_simple=measure_type,
            )
        return endpoint.get_data_frames()[0]

    stats = _nba_with_retries(fetch_stats)
    _team_shot_cache[cache_key] = {"df": stats, "ts": now}
    return stats


def get_team_defensive_shot_zones(season: str, season_type: str, team_abbrev: str) -> Dict[str, Any]:
    team_id = get_team_id(team_abbrev)
    if not team_id:
        raise HTTPException(status_code=404, detail="Team not found")
    stats = get_team_shot_locations_df(season, season_type, "Opponent")
    if stats is None or "TEAM_ID" not in stats.columns:
        raise HTTPException(status_code=503, detail="Shot location data unavailable")
    row = stats.loc[stats["TEAM_ID"] == team_id]
    if row.empty:
        raise HTTPException(status_code=404, detail="Team not found in shot locations")
    row = row.iloc[0]
    values = list(row)
    zone_names = [
        "Restricted Area",
        "In The Paint (Non-RA)",
        "Mid-Range",
        "Left Corner 3",
        "Right Corner 3",
        "Above the Break 3",
        "Backcourt",
    ]
    base_idx = 2
    zones = []
    for i, zone in enumerate(zone_names):
        idx = base_idx + (i * 3)
        if idx + 2 >= len(values):
            break
        fgm = _numeric(values[idx])
        fga = _numeric(values[idx + 1])
        fg_pct = _numeric(values[idx + 2])
        rank, pct = _rank_metric_by_index(stats, team_id, idx + 2, True)
        zones.append(
            {
                "zone": zone,
                "fgm": round(fgm, 2) if fgm is not None else None,
                "fga": round(fga, 2) if fga is not None else None,
                "fg_pct": round(fg_pct, 4) if fg_pct is not None else None,
                "rank": rank,
                "percentile": pct,
            }
        )
    return {
        "team_abbrev": team_abbrev.strip().upper(),
        "team_name": str(row.get("TEAM_NAME", "")),
        "season": season,
        "season_type": season_type,
        "zones": zones,
    }


def _avg_stats_from_df(df):
    if df is None or df.empty:
        return {"games": 0}
    pts = [_numeric(v) for v in df.get("PTS", [])]
    reb = [_numeric(v) for v in df.get("REB", [])]
    ast = [_numeric(v) for v in df.get("AST", [])]
    mins = [_numeric(v) for v in df.get("MIN", [])]
    pts_vals = [v for v in pts if v is not None]
    reb_vals = [v for v in reb if v is not None]
    ast_vals = [v for v in ast if v is not None]
    min_vals = [v for v in mins if v is not None]
    pra_vals = []
    for i in range(len(df)):
        p = pts[i] if i < len(pts) else None
        r = reb[i] if i < len(reb) else None
        a = ast[i] if i < len(ast) else None
        if p is None or r is None or a is None:
            continue
        pra_vals.append(p + r + a)
    return {
        "games": int(len(df)),
        "avg_pts": round(sum(pts_vals) / len(pts_vals), 2) if pts_vals else None,
        "avg_reb": round(sum(reb_vals) / len(reb_vals), 2) if reb_vals else None,
        "avg_ast": round(sum(ast_vals) / len(ast_vals), 2) if ast_vals else None,
        "avg_pra": round(sum(pra_vals) / len(pra_vals), 2) if pra_vals else None,
        "avg_min": round(sum(min_vals) / len(min_vals), 2) if min_vals else None,
    }


def get_player_splits_without(player: str, without: List[str], season: str, season_type: str) -> Dict[str, Any]:
    if not player or not without:
        raise HTTPException(status_code=400, detail="Player and without list are required")
    player_id = get_player_id(player)
    if not player_id:
        raise HTTPException(status_code=404, detail="Player not found")
    without_ids = []
    missing = []
    for name in without:
        pid = get_player_id(name)
        if not pid:
            missing.append(name)
        else:
            without_ids.append((name, pid))
    if not without_ids:
        raise HTTPException(status_code=404, detail="No valid key players found")
    df_player = get_player_log(player_id, season, season_type)
    if df_player is None or df_player.empty:
        raise HTTPException(status_code=404, detail="No game logs found for player")

    key_game_sets = {}
    for name, pid in without_ids:
        try:
            df_key = get_player_log(pid, season, season_type)
        except HTTPException:
            df_key = None
        if df_key is None or df_key.empty or "GAME_ID" not in df_key.columns:
            key_game_sets[name] = set()
        else:
            key_game_sets[name] = set(df_key["GAME_ID"].astype(str).tolist())

    if "GAME_ID" not in df_player.columns:
        raise HTTPException(status_code=503, detail="Player logs missing GAME_ID")
    df_player = df_player.copy()
    df_player["GAME_ID"] = df_player["GAME_ID"].astype(str)
    with_all = set.intersection(*[s for s in key_game_sets.values()]) if key_game_sets else set()
    mask_with_all = df_player["GAME_ID"].isin(with_all) if with_all else [False] * len(df_player)
    df_with_all = df_player[mask_with_all] if with_all else df_player.iloc[0:0]
    df_without_any = df_player[~df_player["GAME_ID"].isin(with_all)] if key_game_sets else df_player
    warnings = []
    if any(len(s) == 0 for s in key_game_sets.values()):
        warnings.append("Some key players had no logged games for the selected season.")
    if df_with_all.empty:
        warnings.append("No games found where all key players were active.")

    return {
        "player": player,
        "season": season,
        "season_type": season_type,
        "without": [name for name, _ in without_ids],
        "missing": missing,
        "samples": {
            "total_games": int(len(df_player)),
            "without_any_games": int(len(df_without_any)),
            "with_all_games": int(len(df_with_all)),
        },
        "averages": {
            "overall": _avg_stats_from_df(df_player),
            "without_any": _avg_stats_from_df(df_without_any),
            "with_all": _avg_stats_from_df(df_with_all),
        },
        "warnings": warnings,
    }


def get_player_log(player_id: int, season: str, season_type: str):
    cache_key = (player_id, season, season_type)
    now = time.time()
    cached = _player_log_cache.get(cache_key)
    if cached and (now - cached["ts"] < DATA_TTL_SECONDS):
        return cached["df"]
    def fetch_logs():
        try:
            endpoint = playergamelog.PlayerGameLog(
                player_id=player_id,
                season=season,
                season_type_all_star=season_type,
                timeout=NBA_HTTP_TIMEOUT_SECONDS,
                headers=NBA_HEADERS,
            )
        except TypeError:
            endpoint = playergamelog.PlayerGameLog(
                player_id=player_id,
                season=season,
                season_type_all_star=season_type,
            )
        return endpoint.get_data_frames()[0]
    try:
        df = _nba_with_retries(fetch_logs)
    except HTTPException:
        if cached:
            return cached["df"]
        raise
    _player_log_cache[cache_key] = {"df": df, "ts": now}
    return df


def build_reasons(prop: str, line: float, l5: float, l10: float, h2h: float, avg_l5: float, avg_l10: float, avg_h2h: float, minutes_proj: float, dvp: str, opponent: str):
    reasons = []
    reasons.append(f"Recent hit rates: L5 {l5:.1f}% and L10 {l10:.1f}% vs line {line}.")
    reasons.append(f"Average {prop} over L5/L10: {avg_l5:.1f} / {avg_l10:.1f}.")
    if opponent:
        reasons.append(f"H2H hit rate vs {opponent}: {h2h:.1f}% (avg {avg_h2h:.1f}).")
    reasons.append(f"Estimated minutes: {minutes_proj:.1f}.")
    reasons.append(f"Opponent defense: {dvp}.")
    return reasons


init_db()


@app.get("/")
def root():
    return {
        "ok": True,
        "service": "prop-analyzer-api",
        "model_version": MODEL_VERSION,
        "endpoints": [
            "/health",
            "/analyze",
            "/evaluate",
            "/odds-edge",
            "/performance",
            "/picks",
            "/nba/team-intel",
            "/nba/player-splits",
        ],
    }


@app.get("/evaluate")
def evaluate(
    request: Request,
    player: str = Query(..., min_length=1),
    sport: str = Query("nba", min_length=1),
    prop: str = Query(..., min_length=1),
    line: float = Query(..., gt=0),
    opponent: str = "",
    season_type: str = "Regular Season",
    window_1: int = Query(5, ge=1, le=30),
    window_2: int = Query(10, ge=1, le=50),
    hit_operator: str = "",
    conf_l5_min: float = Query(None, ge=0, le=100),
    conf_l10_min: float = Query(None, ge=0, le=100),
    conf_h2h_good: float = Query(None, ge=0, le=100),
    conf_low_max: float = Query(None, ge=0, le=100),
    offered_odds: Optional[int] = Query(None),
    include_injury: bool = Query(False),
):
    normalized_sport = normalize_sport(sport)
    if not normalized_sport:
        raise HTTPException(status_code=400, detail=f"Unsupported sport '{sport}'. Supported: {', '.join(SUPPORTED_SPORTS)}")

    normalized_prop = normalize_prop(prop, normalized_sport)
    if normalized_prop not in supported_props(normalized_sport):
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported prop '{prop}' for sport '{normalized_sport}'. Supported: {', '.join(supported_props(normalized_sport))}",
        )

    client_ip = request.client.host if request.client else "unknown"
    identity = f"{client_ip}|anon"
    ok, retry_after = _rate_limit_for_identity(identity)
    if not ok:
        raise HTTPException(status_code=429, detail=f"Rate limit exceeded. Try again in {retry_after}s.")

    l5_min = conf_l5_min if conf_l5_min is not None else CONF_L5_MIN
    l10_min = conf_l10_min if conf_l10_min is not None else CONF_L10_MIN
    h2h_good = conf_h2h_good if conf_h2h_good is not None else CONF_H2H_GOOD
    low_max = conf_low_max if conf_low_max is not None else CONF_LOW_MAX

    if normalized_sport != "nba":
        op = hit_operator.strip().lower() if hit_operator else HIT_OPERATOR
        live_result = None
        live_error = ""
        try:
            live_result = _build_live_multi_sport_result(
                sport=normalized_sport,
                player=player,
                prop=normalized_prop,
                line=line,
                opponent=opponent,
                window_1=window_1,
                window_2=window_2,
                op=op,
                conf_l5_min=l5_min,
                conf_l10_min=l10_min,
                conf_h2h_good=h2h_good,
                conf_low_max=low_max,
                season_type=season_type,
            )
        except HTTPException as exc:
            live_error = f"Live provider error ({exc.status_code}): {exc.detail}"
        except Exception as exc:
            live_error = f"Live provider error: {type(exc).__name__}"
        if live_result:
            result = live_result
        else:
            fallback_result = build_multi_sport_fallback(
                sport=normalized_sport,
                player=player,
                prop=normalized_prop,
                line=line,
                opponent=opponent,
                window_1=window_1,
                window_2=window_2,
                conf_l5_min=l5_min,
                conf_l10_min=l10_min,
                conf_h2h_good=h2h_good,
                conf_low_max=low_max,
            )
            fallback_result["reasons"].insert(0, "Live data unavailable; using deterministic fallback model.")
            if live_error:
                fallback_result["reasons"].insert(1, live_error[:180])
            result = fallback_result
        implied_prob = implied_probability_from_american(offered_odds)
        edge_pct = round(result.get("projected_probability", 0.0) - implied_prob, 2) if implied_prob is not None else None
        pick_id = save_pick(
            sport=result["sport"],
            player=result["player"],
            prop=result["prop"],
            line=float(result["line"]),
            recommendation_value=result["recommendation"],
            confidence_value=float(result["confidence"]),
            projected_prob=float(result.get("projected_probability", 50.0)),
            offered_odds=offered_odds,
            implied_prob=implied_prob,
            edge_pct=edge_pct,
            data_source=result.get("data_source", "live_external"),
            fallback_used=bool(result.get("fallback_used", False)),
            model_version=result.get("model_version", MODEL_VERSION),
        )
        result["pick_id"] = pick_id
        result["offered_odds"] = offered_odds
        result["implied_probability"] = implied_prob
        result["edge_pct"] = edge_pct
        result["injury_context"] = get_injury_context(normalized_sport, player) if include_injury else {"status": "not_requested"}
        if edge_pct is not None and edge_pct >= ALERT_MIN_EDGE_PCT and result.get("confidence", 0) >= 80:
            _send_discord_alert(
                f"Edge Alert #{pick_id}: {result['sport'].upper()} {result['player']} {result['prop']} {result['recommendation']} "
                f"line {result['line']} edge {edge_pct:.2f}% confidence {result['confidence']}%"
            )
        return result

    if NBA_LIVE_DISABLED:
        fallback_result = build_multi_sport_fallback(
            sport="nba",
            player=player,
            prop=normalized_prop,
            line=line,
            opponent=opponent,
            window_1=window_1,
            window_2=window_2,
            conf_l5_min=l5_min,
            conf_l10_min=l10_min,
            conf_h2h_good=h2h_good,
            conf_low_max=low_max,
        )
        fallback_result["projection_label"] = "Minutes Projection"
        fallback_result["reasons"].insert(0, "NBA live data disabled; using deterministic fallback model.")
        result = fallback_result
        implied_prob = implied_probability_from_american(offered_odds)
        edge_pct = round(result.get("projected_probability", 0.0) - implied_prob, 2) if implied_prob is not None else None
        pick_id = save_pick(
            sport=result["sport"],
            player=result["player"],
            prop=result["prop"],
            line=float(result["line"]),
            recommendation_value=result["recommendation"],
            confidence_value=float(result["confidence"]),
            projected_prob=float(result.get("projected_probability", 50.0)),
            offered_odds=offered_odds,
            implied_prob=implied_prob,
            edge_pct=edge_pct,
            data_source=result.get("data_source", "fallback_model"),
            fallback_used=bool(result.get("fallback_used", True)),
            model_version=result.get("model_version", MODEL_VERSION),
        )
        result["pick_id"] = pick_id
        result["offered_odds"] = offered_odds
        result["implied_probability"] = implied_prob
        result["edge_pct"] = edge_pct
        result["injury_context"] = get_injury_context(normalized_sport, player) if include_injury else {"status": "not_requested"}
        return result

    if NBA_PRIMARY_SOURCE == "espn":
        espn_primary_error = ""
        try:
            espn_result = _build_live_nba_result_from_espn(
                player=player,
                prop=normalized_prop,
                line=line,
                opponent=opponent,
                season_type=season_type,
                window_1=window_1,
                window_2=window_2,
                op=(hit_operator.strip().lower() if hit_operator else HIT_OPERATOR),
                l5_min=l5_min,
                l10_min=l10_min,
                h2h_good=h2h_good,
                low_max=low_max,
            )
            if espn_result:
                result = espn_result
                implied_prob = implied_probability_from_american(offered_odds)
                edge_pct = round(result.get("projected_probability", 0.0) - implied_prob, 2) if implied_prob is not None else None
                pick_id = save_pick(
                    sport=result["sport"],
                    player=result["player"],
                    prop=result["prop"],
                    line=float(result["line"]),
                    recommendation_value=result["recommendation"],
                    confidence_value=float(result["confidence"]),
                    projected_prob=float(result.get("projected_probability", 50.0)),
                    offered_odds=offered_odds,
                    implied_prob=implied_prob,
                    edge_pct=edge_pct,
                    data_source=result.get("data_source", "espn_nba"),
                    fallback_used=False,
                    model_version=result.get("model_version", MODEL_VERSION),
                )
                result["pick_id"] = pick_id
                result["offered_odds"] = offered_odds
                result["implied_probability"] = implied_prob
                result["edge_pct"] = edge_pct
                result["injury_context"] = get_injury_context(normalized_sport, player) if include_injury else {"status": "not_requested"}
                if edge_pct is not None and edge_pct >= ALERT_MIN_EDGE_PCT and result.get("confidence", 0) >= 80:
                    _send_discord_alert(
                        f"Edge Alert #{pick_id}: {result['sport'].upper()} {result['player']} {result['prop']} {result['recommendation']} "
                        f"line {result['line']} edge {edge_pct:.2f}% confidence {result['confidence']}%"
                    )
                return result
            espn_primary_error = "ESPN NBA primary source returned no data"
        except HTTPException as exc:
            espn_primary_error = f"ESPN NBA primary error ({exc.status_code}): {exc.detail}"
        except Exception as exc:
            espn_primary_error = f"ESPN NBA primary error: {type(exc).__name__}"

        fallback_result = build_multi_sport_fallback(
            sport="nba",
            player=player,
            prop=normalized_prop,
            line=line,
            opponent=opponent,
            window_1=window_1,
            window_2=window_2,
            conf_l5_min=l5_min,
            conf_l10_min=l10_min,
            conf_h2h_good=h2h_good,
            conf_low_max=low_max,
        )
        fallback_result["projection_label"] = "Minutes Projection"
        fallback_result["reasons"].insert(0, "NBA primary source failed; using deterministic fallback model.")
        if espn_primary_error:
            fallback_result["reasons"].insert(1, espn_primary_error[:180])
        result = fallback_result
        implied_prob = implied_probability_from_american(offered_odds)
        edge_pct = round(result.get("projected_probability", 0.0) - implied_prob, 2) if implied_prob is not None else None
        pick_id = save_pick(
            sport=result["sport"],
            player=result["player"],
            prop=result["prop"],
            line=float(result["line"]),
            recommendation_value=result["recommendation"],
            confidence_value=float(result["confidence"]),
            projected_prob=float(result.get("projected_probability", 50.0)),
            offered_odds=offered_odds,
            implied_prob=implied_prob,
            edge_pct=edge_pct,
            data_source=result.get("data_source", "fallback_model"),
            fallback_used=bool(result.get("fallback_used", True)),
            model_version=result.get("model_version", MODEL_VERSION),
        )
        result["pick_id"] = pick_id
        result["offered_odds"] = offered_odds
        result["implied_probability"] = implied_prob
        result["edge_pct"] = edge_pct
        result["injury_context"] = get_injury_context(normalized_sport, player) if include_injury else {"status": "not_requested"}
        return result

    bdl_error = ""
    # Keep nba_api as the default primary path. BALldontlie can be explicitly
    # promoted to primary via NBA_PRIMARY_SOURCE=balldontlie.
    if NBA_PRIMARY_SOURCE == "balldontlie" and BALDONTLIE_ENABLED and BALDONTLIE_API_KEY:
        try:
            bdl_result = _build_live_nba_result_from_bdl(
                player=player,
                prop=normalized_prop,
                line=line,
                opponent=opponent,
                season_type=season_type,
                window_1=window_1,
                window_2=window_2,
                op=(hit_operator.strip().lower() if hit_operator else HIT_OPERATOR),
                l5_min=l5_min,
                l10_min=l10_min,
                h2h_good=h2h_good,
                low_max=low_max,
            )
            if bdl_result:
                result = bdl_result
                implied_prob = implied_probability_from_american(offered_odds)
                edge_pct = round(result.get("projected_probability", 0.0) - implied_prob, 2) if implied_prob is not None else None
                pick_id = save_pick(
                    sport=result["sport"],
                    player=result["player"],
                    prop=result["prop"],
                    line=float(result["line"]),
                    recommendation_value=result["recommendation"],
                    confidence_value=float(result["confidence"]),
                    projected_prob=float(result.get("projected_probability", 50.0)),
                    offered_odds=offered_odds,
                    implied_prob=implied_prob,
                    edge_pct=edge_pct,
                    data_source=result.get("data_source", "balldontlie"),
                    fallback_used=False,
                    model_version=result.get("model_version", MODEL_VERSION),
                )
                result["pick_id"] = pick_id
                result["offered_odds"] = offered_odds
                result["implied_probability"] = implied_prob
                result["edge_pct"] = edge_pct
                result["injury_context"] = get_injury_context(normalized_sport, player) if include_injury else {"status": "not_requested"}
                if edge_pct is not None and edge_pct >= ALERT_MIN_EDGE_PCT and result.get("confidence", 0) >= 80:
                    _send_discord_alert(
                        f"Edge Alert #{pick_id}: {result['sport'].upper()} {result['player']} {result['prop']} {result['recommendation']} "
                        f"line {result['line']} edge {edge_pct:.2f}% confidence {result['confidence']}%"
                    )
                return result
        except HTTPException as exc:
            bdl_error = f"BALldontlie error ({exc.status_code}): {exc.detail}"
        except Exception as exc:
            bdl_error = f"BALldontlie error: {type(exc).__name__}"

    pid = get_player_id(player)
    if not pid:
        return {"error": "Player not found"}

    season = current_season()
    try:
        df = get_player_log(pid, season, season_type)
    except HTTPException as exc:
        espn_error = ""
        if NBA_ESPN_FALLBACK_ENABLED:
            try:
                espn_result = _build_live_nba_result_from_espn(
                    player=player,
                    prop=normalized_prop,
                    line=line,
                    opponent=opponent,
                    season_type=season_type,
                    window_1=window_1,
                    window_2=window_2,
                    op=(hit_operator.strip().lower() if hit_operator else HIT_OPERATOR),
                    l5_min=l5_min,
                    l10_min=l10_min,
                    h2h_good=h2h_good,
                    low_max=low_max,
                )
                if espn_result:
                    result = espn_result
                    implied_prob = implied_probability_from_american(offered_odds)
                    edge_pct = round(result.get("projected_probability", 0.0) - implied_prob, 2) if implied_prob is not None else None
                    pick_id = save_pick(
                        sport=result["sport"],
                        player=result["player"],
                        prop=result["prop"],
                        line=float(result["line"]),
                        recommendation_value=result["recommendation"],
                        confidence_value=float(result["confidence"]),
                        projected_prob=float(result.get("projected_probability", 50.0)),
                        offered_odds=offered_odds,
                        implied_prob=implied_prob,
                        edge_pct=edge_pct,
                        data_source=result.get("data_source", "espn_nba"),
                        fallback_used=False,
                        model_version=result.get("model_version", MODEL_VERSION),
                    )
                    result["pick_id"] = pick_id
                    result["offered_odds"] = offered_odds
                    result["implied_probability"] = implied_prob
                    result["edge_pct"] = edge_pct
                    result["injury_context"] = get_injury_context(normalized_sport, player) if include_injury else {"status": "not_requested"}
                    if edge_pct is not None and edge_pct >= ALERT_MIN_EDGE_PCT and result.get("confidence", 0) >= 80:
                        _send_discord_alert(
                            f"Edge Alert #{pick_id}: {result['sport'].upper()} {result['player']} {result['prop']} {result['recommendation']} "
                            f"line {result['line']} edge {edge_pct:.2f}% confidence {result['confidence']}%"
                        )
                    return result
            except HTTPException as espn_exc:
                espn_error = f"ESPN NBA error ({espn_exc.status_code}): {espn_exc.detail}"
            except Exception as espn_exc:
                espn_error = f"ESPN NBA error: {type(espn_exc).__name__}"

        fallback_result = build_multi_sport_fallback(
            sport="nba",
            player=player,
            prop=normalized_prop,
            line=line,
            opponent=opponent,
            window_1=window_1,
            window_2=window_2,
            conf_l5_min=l5_min,
            conf_l10_min=l10_min,
            conf_h2h_good=h2h_good,
            conf_low_max=low_max,
        )
        fallback_result["projection_label"] = "Minutes Projection"
        fallback_result["reasons"].insert(0, "NBA live data unavailable; using deterministic fallback model.")
        fallback_result["reasons"].insert(1, f"Live provider error ({exc.status_code}): {exc.detail}"[:180])
        if espn_error:
            fallback_result["reasons"].insert(1, espn_error[:180])
        if bdl_error:
            fallback_result["reasons"].insert(1, bdl_error[:180])
        result = fallback_result
        implied_prob = implied_probability_from_american(offered_odds)
        edge_pct = round(result.get("projected_probability", 0.0) - implied_prob, 2) if implied_prob is not None else None
        pick_id = save_pick(
            sport=result["sport"],
            player=result["player"],
            prop=result["prop"],
            line=float(result["line"]),
            recommendation_value=result["recommendation"],
            confidence_value=float(result["confidence"]),
            projected_prob=float(result.get("projected_probability", 50.0)),
            offered_odds=offered_odds,
            implied_prob=implied_prob,
            edge_pct=edge_pct,
            data_source=result.get("data_source", "fallback_model"),
            fallback_used=bool(result.get("fallback_used", True)),
            model_version=result.get("model_version", MODEL_VERSION),
        )
        result["pick_id"] = pick_id
        result["offered_odds"] = offered_odds
        result["implied_probability"] = implied_prob
        result["edge_pct"] = edge_pct
        result["injury_context"] = get_injury_context(normalized_sport, player) if include_injury else {"status": "not_requested"}
        return result
    if df is None or df.empty:
        espn_error = ""
        if NBA_ESPN_FALLBACK_ENABLED:
            try:
                espn_result = _build_live_nba_result_from_espn(
                    player=player,
                    prop=normalized_prop,
                    line=line,
                    opponent=opponent,
                    season_type=season_type,
                    window_1=window_1,
                    window_2=window_2,
                    op=(hit_operator.strip().lower() if hit_operator else HIT_OPERATOR),
                    l5_min=l5_min,
                    l10_min=l10_min,
                    h2h_good=h2h_good,
                    low_max=low_max,
                )
                if espn_result:
                    result = espn_result
                    implied_prob = implied_probability_from_american(offered_odds)
                    edge_pct = round(result.get("projected_probability", 0.0) - implied_prob, 2) if implied_prob is not None else None
                    pick_id = save_pick(
                        sport=result["sport"],
                        player=result["player"],
                        prop=result["prop"],
                        line=float(result["line"]),
                        recommendation_value=result["recommendation"],
                        confidence_value=float(result["confidence"]),
                        projected_prob=float(result.get("projected_probability", 50.0)),
                        offered_odds=offered_odds,
                        implied_prob=implied_prob,
                        edge_pct=edge_pct,
                        data_source=result.get("data_source", "espn_nba"),
                        fallback_used=False,
                        model_version=result.get("model_version", MODEL_VERSION),
                    )
                    result["pick_id"] = pick_id
                    result["offered_odds"] = offered_odds
                    result["implied_probability"] = implied_prob
                    result["edge_pct"] = edge_pct
                    result["injury_context"] = get_injury_context(normalized_sport, player) if include_injury else {"status": "not_requested"}
                    if edge_pct is not None and edge_pct >= ALERT_MIN_EDGE_PCT and result.get("confidence", 0) >= 80:
                        _send_discord_alert(
                            f"Edge Alert #{pick_id}: {result['sport'].upper()} {result['player']} {result['prop']} {result['recommendation']} "
                            f"line {result['line']} edge {edge_pct:.2f}% confidence {result['confidence']}%"
                        )
                    return result
            except HTTPException as espn_exc:
                espn_error = f"ESPN NBA error ({espn_exc.status_code}): {espn_exc.detail}"
            except Exception as espn_exc:
                espn_error = f"ESPN NBA error: {type(espn_exc).__name__}"

        fallback_result = build_multi_sport_fallback(
            sport="nba",
            player=player,
            prop=normalized_prop,
            line=line,
            opponent=opponent,
            window_1=window_1,
            window_2=window_2,
            conf_l5_min=l5_min,
            conf_l10_min=l10_min,
            conf_h2h_good=h2h_good,
            conf_low_max=low_max,
        )
        fallback_result["projection_label"] = "Minutes Projection"
        fallback_result["reasons"].insert(0, "NBA game logs were empty; using deterministic fallback model.")
        fallback_result["reasons"].insert(1, f"No NBA game logs available for {player} in season {season} ({season_type})."[:180])
        if espn_error:
            fallback_result["reasons"].insert(1, espn_error[:180])
        if bdl_error:
            fallback_result["reasons"].insert(1, bdl_error[:180])
        result = fallback_result
        implied_prob = implied_probability_from_american(offered_odds)
        edge_pct = round(result.get("projected_probability", 0.0) - implied_prob, 2) if implied_prob is not None else None
        pick_id = save_pick(
            sport=result["sport"],
            player=result["player"],
            prop=result["prop"],
            line=float(result["line"]),
            recommendation_value=result["recommendation"],
            confidence_value=float(result["confidence"]),
            projected_prob=float(result.get("projected_probability", 50.0)),
            offered_odds=offered_odds,
            implied_prob=implied_prob,
            edge_pct=edge_pct,
            data_source=result.get("data_source", "fallback_model"),
            fallback_used=bool(result.get("fallback_used", True)),
            model_version=result.get("model_version", MODEL_VERSION),
        )
        result["pick_id"] = pick_id
        result["offered_odds"] = offered_odds
        result["implied_probability"] = implied_prob
        result["edge_pct"] = edge_pct
        result["injury_context"] = get_injury_context(normalized_sport, player) if include_injury else {"status": "not_requested"}
        return result

    last_1 = df.head(window_1)
    last_2 = df.head(window_2)

    op = hit_operator.strip().lower() if hit_operator else HIT_OPERATOR
    l5_hits, l5_n, l5 = hit_rate_details(last_1, normalized_prop, line, op, normalized_sport)
    l10_hits, l10_n, l10 = hit_rate_details(last_2, normalized_prop, line, op, normalized_sport)
    avg_l5 = avg_stat(last_1, normalized_prop, normalized_sport)
    avg_l10 = avg_stat(last_2, normalized_prop, normalized_sport)
    if l5_n == 0 or l10_n == 0:
        raise HTTPException(
            status_code=422,
            detail=(
                f"Insufficient NBA samples for {player} {normalized_prop}. "
                f"Got L5={l5_n}, L10={l10_n}. Try another player/prop or verify data source."
            ),
        )

    h2h_df = filter_h2h(df, opponent)
    if opponent:
        if len(h2h_df):
            h2h_hits, h2h_n, h2h = hit_rate_details(h2h_df, normalized_prop, line, op, normalized_sport)
        else:
            h2h_hits, h2h_n = 0, 0
            h2h = DEFAULT_H2H_WITH_OPP
    else:
        h2h_hits, h2h_n = 0, 0
        h2h = DEFAULT_H2H
    has_h2h = opponent and len(h2h_df) > 0
    avg_h2h = avg_stat(h2h_df, normalized_prop, normalized_sport) if has_h2h else 0

    conf = confidence(l5, l10, h2h, l5_min, l10_min, h2h_good, low_max)
    expected_stat = weighted_expected_stat(avg_l5, avg_l10, avg_h2h, has_h2h)
    rec = line_recommendation(expected_stat, line)
    proj_prob = projected_probability(l5, l10, h2h, has_h2h)
    conf = calibrate_confidence(conf, proj_prob)
    minutes_proj = round(float(last_2["MIN"].mean()), 1) if len(last_2) else 0
    dvp = get_team_def_rating(season, season_type, opponent)
    reasons = build_reasons(normalized_prop, line, l5, l10, h2h, avg_l5, avg_l10, avg_h2h, minutes_proj, dvp, opponent)
    recent_details = nba_prop_game_details(last_2, normalized_prop, line, op, window_2)
    h2h_details = nba_prop_game_details(h2h_df, normalized_prop, line, op, window_2) if opponent else []

    result = {
        "sport": normalized_sport,
        "player": player,
        "prop": normalized_prop,
        "line": line,
        "last_5_hit_rate": l5,
        "last_10_hit_rate": l10,
        "h2h_hit_rate": h2h,
        "last_5_ci": wilson_interval(l5_hits, l5_n),
        "last_10_ci": wilson_interval(l10_hits, l10_n),
        "h2h_ci": wilson_interval(h2h_hits, h2h_n) if opponent else (0.0, 0.0),
        "last_5_avg_stat": avg_l5,
        "last_10_avg_stat": avg_l10,
        "h2h_avg_stat": avg_h2h,
        "confidence": conf,
        "projected_probability": proj_prob,
        "recommendation": rec,
        "confidence_label": recommendation(conf),
        "expected_stat": expected_stat,
        "minutes_proj": minutes_proj,
        "projection_label": "Minutes Projection",
        "dvp": dvp,
        "reasons": reasons,
        "data_source": "nba_api",
        "fallback_used": False,
        "source_timestamp": _now_iso(),
        "model_version": MODEL_VERSION,
        "samples": {
            "last_5_games": l5_n,
            "last_10_games": l10_n,
            "h2h_games": h2h_n,
        },
        "last_games_detail": recent_details,
        "h2h_games_detail": h2h_details,
    }
    implied_prob = implied_probability_from_american(offered_odds)
    edge_pct = round(result.get("projected_probability", 0.0) - implied_prob, 2) if implied_prob is not None else None
    pick_id = save_pick(
        sport=result["sport"],
        player=result["player"],
        prop=result["prop"],
        line=float(result["line"]),
        recommendation_value=result["recommendation"],
        confidence_value=float(result["confidence"]),
        projected_prob=float(result.get("projected_probability", 50.0)),
        offered_odds=offered_odds,
        implied_prob=implied_prob,
        edge_pct=edge_pct,
        data_source=result.get("data_source", "nba_api"),
        fallback_used=False,
        model_version=result.get("model_version", MODEL_VERSION),
    )
    result["pick_id"] = pick_id
    result["offered_odds"] = offered_odds
    result["implied_probability"] = implied_prob
    result["edge_pct"] = edge_pct
    result["injury_context"] = get_injury_context(normalized_sport, player) if include_injury else {"status": "not_requested"}
    if edge_pct is not None and edge_pct >= ALERT_MIN_EDGE_PCT and result.get("confidence", 0) >= 80:
        _send_discord_alert(
            f"Edge Alert #{pick_id}: {result['sport'].upper()} {result['player']} {result['prop']} {result['recommendation']} "
            f"line {result['line']} edge {edge_pct:.2f}% confidence {result['confidence']}%"
        )
    return result


def _analyze_safe(
    request: Request,
    player: str,
    sport: str,
    prop: str,
    line: float,
    opponent: str,
    season_type: str,
    window_1: int,
    window_2: int,
    hit_operator: str,
    conf_l5_min: Optional[float],
    conf_l10_min: Optional[float],
    conf_h2h_good: Optional[float],
    conf_low_max: Optional[float],
    offered_odds: Optional[int],
    include_injury: bool,
):
    try:
        return evaluate(
            request=request,
            player=player,
            sport=sport,
            prop=prop,
            line=line,
            opponent=opponent,
            season_type=season_type,
            window_1=window_1,
            window_2=window_2,
            hit_operator=hit_operator,
            conf_l5_min=conf_l5_min,
            conf_l10_min=conf_l10_min,
            conf_h2h_good=conf_h2h_good,
            conf_low_max=conf_low_max,
            offered_odds=offered_odds,
            include_injury=include_injury,
        )
    except HTTPException:
        raise
    except Exception as exc:
        normalized_sport = normalize_sport(sport) or "nba"
        normalized_prop = normalize_prop(prop, normalized_sport)
        l5_min = conf_l5_min if conf_l5_min is not None else CONF_L5_MIN
        l10_min = conf_l10_min if conf_l10_min is not None else CONF_L10_MIN
        h2h_good = conf_h2h_good if conf_h2h_good is not None else CONF_H2H_GOOD
        low_max = conf_low_max if conf_low_max is not None else CONF_LOW_MAX
        fallback_result = build_multi_sport_fallback(
            sport=normalized_sport,
            player=player,
            prop=normalized_prop,
            line=line,
            opponent=opponent,
            window_1=window_1,
            window_2=window_2,
            conf_l5_min=l5_min,
            conf_l10_min=l10_min,
            conf_h2h_good=h2h_good,
            conf_low_max=low_max,
        )
        fallback_result["reasons"].insert(0, "Server error during live analysis; using deterministic fallback model.")
        fallback_result["reasons"].insert(1, f"Internal error: {type(exc).__name__}")
        implied_prob = implied_probability_from_american(offered_odds)
        edge_pct = round(fallback_result.get("projected_probability", 0.0) - implied_prob, 2) if implied_prob is not None else None
        try:
            pick_id = save_pick(
                sport=fallback_result["sport"],
                player=fallback_result["player"],
                prop=fallback_result["prop"],
                line=float(fallback_result["line"]),
                recommendation_value=fallback_result["recommendation"],
                confidence_value=float(fallback_result["confidence"]),
                projected_prob=float(fallback_result.get("projected_probability", 50.0)),
                offered_odds=offered_odds,
                implied_prob=implied_prob,
                edge_pct=edge_pct,
                data_source=fallback_result.get("data_source", "fallback_model"),
                fallback_used=bool(fallback_result.get("fallback_used", True)),
                model_version=fallback_result.get("model_version", MODEL_VERSION),
            )
            fallback_result["pick_id"] = pick_id
        except Exception:
            fallback_result["pick_id"] = None
            fallback_result["reasons"].append("Could not persist pick to database.")
        fallback_result["offered_odds"] = offered_odds
        fallback_result["implied_probability"] = implied_prob
        fallback_result["edge_pct"] = edge_pct
        fallback_result["injury_context"] = get_injury_context(normalized_sport, player) if include_injury else {"status": "not_requested"}
        return fallback_result


@app.get("/analyze")
def analyze(
    request: Request,
    player: str = Query(..., min_length=1),
    sport: str = Query("nba", min_length=1),
    prop: str = Query(..., min_length=1),
    line: float = Query(..., gt=0),
    opponent: str = "",
    season_type: str = "Regular Season",
    window_1: int = Query(5, ge=1, le=30),
    window_2: int = Query(10, ge=1, le=50),
    hit_operator: str = "",
    conf_l5_min: float = Query(None, ge=0, le=100),
    conf_l10_min: float = Query(None, ge=0, le=100),
    conf_h2h_good: float = Query(None, ge=0, le=100),
    conf_low_max: float = Query(None, ge=0, le=100),
    offered_odds: Optional[int] = Query(None),
    include_injury: bool = Query(False),
):
    return _analyze_safe(
        request=request,
        player=player,
        sport=sport,
        prop=prop,
        line=line,
        opponent=opponent,
        season_type=season_type,
        window_1=window_1,
        window_2=window_2,
        hit_operator=hit_operator,
        conf_l5_min=conf_l5_min,
        conf_l10_min=conf_l10_min,
        conf_h2h_good=conf_h2h_good,
        conf_low_max=conf_low_max,
        offered_odds=offered_odds,
        include_injury=include_injury,
    )


@app.post("/v2/analyze")
def analyze_v2(request: Request, payload: AnalyzeRequestV2):
    request_id = str(uuid.uuid4())
    try:
        result = _analyze_safe(
            request=request,
            player=payload.player,
            sport=payload.sport,
            prop=payload.prop,
            line=payload.line,
            opponent=payload.opponent,
            season_type=payload.season_type,
            window_1=payload.window_1,
            window_2=payload.window_2,
            hit_operator=payload.hit_operator,
            conf_l5_min=payload.conf_l5_min,
            conf_l10_min=payload.conf_l10_min,
            conf_h2h_good=payload.conf_h2h_good,
            conf_low_max=payload.conf_low_max,
            offered_odds=payload.offered_odds,
            include_injury=payload.include_injury,
        )
        return {"ok": True, "request_id": request_id, "data": result, "error": None}
    except HTTPException as exc:
        detail = exc.detail if isinstance(exc.detail, str) else str(exc.detail)
        code = {
            400: "bad_request",
            401: "unauthorized",
            404: "not_found",
            422: "validation_error",
            429: "rate_limited",
            502: "provider_error",
            503: "provider_unavailable",
        }.get(exc.status_code, "api_error")
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "ok": False,
                "request_id": request_id,
                "data": None,
                "error": {
                    "code": code,
                    "message": detail[:240],
                    "retryable": exc.status_code in (429, 502, 503),
                },
            },
        )


def _odds_sport_key(sport: str):
    return {
        "nba": "basketball_nba",
        "ncaab": "basketball_ncaab",
        "mlb": "baseball_mlb",
        "nfl": "americanfootball_nfl",
        "soccer": "soccer_epl",
        "nhl": "icehockey_nhl",
    }.get(sport, "")


def _parse_market_price(book: dict, market_key: str, side: str):
    for market in book.get("markets", []):
        if market.get("key") != market_key:
            continue
        for outcome in market.get("outcomes", []):
            name = str(outcome.get("name", "")).lower()
            if side.lower() in name:
                price = outcome.get("price")
                point = outcome.get("point")
                try:
                    return int(price), point
                except Exception:
                    return None, point
    return None, None


def _result_side(result: dict):
    rec = str(result.get("recommendation", "")).lower()
    return "over" if "over" in rec else "under"


@app.get("/odds-edge")
def odds_edge(
    request: Request,
    sport: str = Query("nba"),
    market: str = Query("player_points"),
    bookmaker: str = Query("draftkings"),
):
    normalized_sport = normalize_sport(sport)
    if not normalized_sport:
        raise HTTPException(status_code=400, detail="Unsupported sport")
    if not ODDS_API_KEY:
        raise HTTPException(status_code=400, detail="ODDS_API_KEY not configured")
    identity = f"odds|{request.client.host if request.client else 'unknown'}|anon"
    ok, retry_after = _rate_limit_for_identity(identity)
    if not ok:
        raise HTTPException(status_code=429, detail=f"Rate limit exceeded. Try again in {retry_after}s.")
    sport_key = _odds_sport_key(normalized_sport)
    if not sport_key:
        raise HTTPException(status_code=400, detail=f"No odds mapping for sport {normalized_sport}")
    url = f"{ODDS_API_BASE_URL}/sports/{sport_key}/odds/"
    data = _fetch_json(
        url,
        params={
            "apiKey": ODDS_API_KEY,
            "regions": "us",
            "markets": market,
            "oddsFormat": "american",
            "bookmakers": bookmaker,
        },
    )
    rows = []
    for event in data if isinstance(data, list) else []:
        books = event.get("bookmakers", [])
        if not books:
            continue
        books = sorted(books, key=lambda b: b.get("last_update", ""), reverse=True)
        over_price, over_point = _parse_market_price(books[0], market, "over")
        under_price, under_point = _parse_market_price(books[0], market, "under")
        rows.append(
            {
                "event_id": event.get("id"),
                "sport": normalized_sport,
                "home_team": event.get("home_team"),
                "away_team": event.get("away_team"),
                "bookmaker": books[0].get("key"),
                "market": market,
                "over_odds": over_price,
                "under_odds": under_price,
                "line": over_point if over_point is not None else under_point,
            }
        )
    return {"count": len(rows), "rows": rows}


@app.get("/nba/team-intel")
def nba_team_intel(
    team: str = Query(..., min_length=2),
    season: str = Query(""),
    season_type: str = "Regular Season",
    include_depth: bool = Query(True),
    include_injuries: bool = Query(True),
    include_defense: bool = Query(True),
    include_shot_zones: bool = Query(True),
):
    team_abbrev = team.strip().upper()
    if not team_abbrev:
        raise HTTPException(status_code=400, detail="Team is required")
    season_label = season.strip() if season else current_season()
    errors = []
    team_name = ""
    depth_chart = []
    injuries = []
    defense = None
    shot_zones = None

    espn_team_id = None
    try:
        espn_map = _espn_team_map()
        entry = espn_map.get(team_abbrev, {})
        espn_team_id = entry.get("id")
        team_name = entry.get("name", "") if entry else ""
    except HTTPException as exc:
        errors.append(f"ESPN team lookup failed ({exc.status_code}): {exc.detail}")
    except Exception as exc:
        errors.append(f"ESPN team lookup failed: {type(exc).__name__}")

    if include_depth:
        if not espn_team_id:
            errors.append("Depth chart unavailable: ESPN team id not found.")
        else:
            try:
                depth_chart = _espn_team_depth_chart(espn_team_id)
            except HTTPException as exc:
                errors.append(f"Depth chart error ({exc.status_code}): {exc.detail}")
            except Exception as exc:
                errors.append(f"Depth chart error: {type(exc).__name__}")

    if include_injuries:
        if not espn_team_id:
            errors.append("Injury report unavailable: ESPN team id not found.")
        else:
            try:
                injuries = _espn_team_injuries(espn_team_id)
            except HTTPException as exc:
                errors.append(f"Injury report error ({exc.status_code}): {exc.detail}")
            except Exception as exc:
                errors.append(f"Injury report error: {type(exc).__name__}")

    if include_defense:
        try:
            defense = get_team_defensive_metrics(season_label, season_type, team_abbrev)
        except HTTPException as exc:
            errors.append(f"Defensive metrics error ({exc.status_code}): {exc.detail}")
        except Exception as exc:
            errors.append(f"Defensive metrics error: {type(exc).__name__}")

    if include_shot_zones:
        try:
            shot_zones = get_team_defensive_shot_zones(season_label, season_type, team_abbrev)
        except HTTPException as exc:
            errors.append(f"Shot zone metrics error ({exc.status_code}): {exc.detail}")
        except Exception as exc:
            errors.append(f"Shot zone metrics error: {type(exc).__name__}")

    return {
        "team": team_abbrev,
        "team_name": team_name,
        "season": season_label,
        "season_type": season_type,
        "depth_chart": depth_chart,
        "injuries": injuries,
        "defensive_metrics": defense,
        "shot_zones": shot_zones,
        "errors": errors,
        "source_timestamp": _now_iso(),
    }


@app.get("/nba/player-splits")
def nba_player_splits(
    player: str = Query(..., min_length=1),
    without: str = Query(..., min_length=1),
    season: str = Query(""),
    season_type: str = "Regular Season",
):
    names = [n.strip() for n in (without or "").split(",") if n.strip()]
    if not names:
        raise HTTPException(status_code=400, detail="Without list is required")
    season_label = season.strip() if season else current_season()
    result = get_player_splits_without(player, names, season_label, season_type)
    result["source_timestamp"] = _now_iso()
    return result


@app.get("/health")
def health():
    with _provider_lock:
        providers = {}
        for provider, state in _provider_state.items():
            open_state = bool(state.get("opened_at")) and (time.time() - state.get("opened_at", 0) < _provider_cooldown_seconds)
            providers[provider] = {
                "open": open_state,
                "failures": int(state.get("failures", 0)),
                "last_error": state.get("last_error", ""),
                "last_success_at": state.get("last_success_at"),
            }
    return {
        "ok": True,
        "model_version": MODEL_VERSION,
        "app_build": APP_BUILD,
        "provider_mode": {
            "nba_live_disabled": NBA_LIVE_DISABLED,
            "nba_primary_source": NBA_PRIMARY_SOURCE,
            "balldontlie_enabled": _balldontlie_is_enabled(),
            "balldontlie_config_enabled": BALDONTLIE_ENABLED,
            "balldontlie_has_key": bool(BALDONTLIE_API_KEY),
            "balldontlie_runtime_disabled_reason": _balldontlie_runtime_disabled_reason,
            "pandascore_enabled": _pandascore_is_enabled(),
            "pandascore_config_enabled": PANDASCORE_ENABLED,
            "pandascore_has_key": bool(PANDASCORE_API_KEY),
            "pandascore_runtime_disabled_reason": _pandascore_runtime_disabled_reason,
            "nba_espn_fallback_enabled": NBA_ESPN_FALLBACK_ENABLED,
        },
        "providers": providers,
        "cache": {
            "player_log_cache_size": len(_player_log_cache),
            "team_stats_cache_size": len(_team_stats_cache),
            "team_shot_cache_size": len(_team_shot_cache),
            "external_cache_size": len(_external_cache),
        },
    }


@app.post("/admin/reset-runtime")
def admin_reset_runtime(admin_secret: str = Query(..., min_length=1)):
    global _balldontlie_runtime_disabled_reason, _pandascore_runtime_disabled_reason
    if not ADMIN_SECRET or admin_secret != ADMIN_SECRET:
        raise HTTPException(status_code=401, detail="Unauthorized.")
    with _provider_lock:
        provider_count = len(_provider_state)
        _provider_state.clear()
    with _rate_lock:
        rate_identity_count = len(_rate_store)
        _rate_store.clear()
    external_cache_size = len(_external_cache)
    player_cache_size = len(_player_log_cache)
    team_cache_size = len(_team_stats_cache)
    team_shot_cache_size = len(_team_shot_cache)
    _external_cache.clear()
    _player_log_cache.clear()
    _team_stats_cache.clear()
    _team_shot_cache.clear()
    _balldontlie_runtime_disabled_reason = ""
    _pandascore_runtime_disabled_reason = ""
    return {
        "ok": True,
        "reset_at": _now_iso(),
        "cleared": {
            "providers": provider_count,
            "rate_limit_identities": rate_identity_count,
            "external_cache": external_cache_size,
            "player_log_cache": player_cache_size,
            "team_stats_cache": team_cache_size,
            "team_shot_cache": team_shot_cache_size,
        },
    }


@app.get("/performance")
def performance(days: int = Query(30, ge=1, le=365), sport: str = Query("")):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    since = (datetime.datetime.utcnow() - datetime.timedelta(days=days)).isoformat()
    if sport:
        cur.execute(
            "SELECT result, pnl_units, edge_pct, confidence FROM picks WHERE created_at >= ? AND sport = ?",
            (since, sport.lower()),
        )
    else:
        cur.execute(
            "SELECT result, pnl_units, edge_pct, confidence FROM picks WHERE created_at >= ?",
            (since,),
        )
    rows = cur.fetchall()
    conn.close()
    total = len(rows)
    settled = [r for r in rows if r[0] in ("win", "loss", "push")]
    wins = sum(1 for r in settled if r[0] == "win")
    losses = sum(1 for r in settled if r[0] == "loss")
    pushes = sum(1 for r in settled if r[0] == "push")
    pnl = round(sum(float(r[1]) for r in settled if r[1] is not None), 3)
    avg_edge = round(sum(float(r[2]) for r in rows if r[2] is not None) / max(1, sum(1 for r in rows if r[2] is not None)), 3)
    avg_conf = round(sum(float(r[3]) for r in rows if r[3] is not None) / max(1, sum(1 for r in rows if r[3] is not None)), 3)
    hit_rate = round((wins / max(1, (wins + losses))) * 100.0, 2)
    return {
        "days": days,
        "sport": sport.lower() if sport else "all",
        "total_picks": total,
        "settled_picks": len(settled),
        "wins": wins,
        "losses": losses,
        "pushes": pushes,
        "hit_rate": hit_rate,
        "pnl_units": pnl,
        "avg_edge_pct": avg_edge,
        "avg_confidence": avg_conf,
    }


@app.get("/picks")
def picks(
    days: int = Query(7, ge=1, le=365),
    sport: str = Query(""),
    limit: int = Query(200, ge=1, le=1000),
):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    since = (datetime.datetime.utcnow() - datetime.timedelta(days=days)).isoformat()
    if sport:
        cur.execute(
            """
            SELECT id, created_at, sport, player, prop, line, recommendation, confidence,
                   projected_probability, offered_odds, implied_probability, edge_pct,
                   data_source, fallback_used, model_version, result, actual_stat, pnl_units
            FROM picks
            WHERE created_at >= ? AND sport = ?
            ORDER BY id DESC
            LIMIT ?
            """,
            (since, sport.lower(), limit),
        )
    else:
        cur.execute(
            """
            SELECT id, created_at, sport, player, prop, line, recommendation, confidence,
                   projected_probability, offered_odds, implied_probability, edge_pct,
                   data_source, fallback_used, model_version, result, actual_stat, pnl_units
            FROM picks
            WHERE created_at >= ?
            ORDER BY id DESC
            LIMIT ?
            """,
            (since, limit),
        )
    rows = cur.fetchall()
    conn.close()
    columns = [
        "id", "created_at", "sport", "player", "prop", "line", "recommendation", "confidence",
        "projected_probability", "offered_odds", "implied_probability", "edge_pct",
        "data_source", "fallback_used", "model_version", "result", "actual_stat", "pnl_units",
    ]
    items = [dict(zip(columns, row)) for row in rows]
    return {"count": len(items), "items": items}


@app.post("/settle-pick")
def settle_pick(
    pick_id: int = Query(..., ge=1),
    actual_stat: float = Query(...),
    admin_secret: str = Query(..., min_length=1),
):
    if not ADMIN_SECRET or admin_secret != ADMIN_SECRET:
        raise HTTPException(status_code=401, detail="Unauthorized.")
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT line, recommendation, offered_odds, result FROM picks WHERE id = ?", (pick_id,))
    row = cur.fetchone()
    if not row:
        conn.close()
        raise HTTPException(status_code=404, detail="Pick not found")
    if row[3] in ("win", "loss", "push"):
        conn.close()
        return {"ok": False, "detail": "Pick already settled"}
    line, rec, offered_odds, _ = row
    rec_lower = str(rec).lower()
    is_over = "over" in rec_lower
    if abs(actual_stat - float(line)) < 1e-9:
        result = "push"
        pnl = 0.0
    else:
        win = actual_stat > float(line) if is_over else actual_stat < float(line)
        result = "win" if win else "loss"
        if offered_odds is None:
            pnl = 1.0 if win else -1.0
        else:
            implied = implied_probability_from_american(int(offered_odds))
            if implied is None:
                pnl = 1.0 if win else -1.0
            else:
                # Risk 1 unit stake.
                if win:
                    if int(offered_odds) > 0:
                        pnl = round(int(offered_odds) / 100.0, 3)
                    else:
                        pnl = round(100.0 / abs(int(offered_odds)), 3)
                else:
                    pnl = -1.0
    cur.execute(
        "UPDATE picks SET result = ?, actual_stat = ?, pnl_units = ? WHERE id = ?",
        (result, float(actual_stat), float(pnl), pick_id),
    )
    conn.commit()
    conn.close()
    return {"ok": True, "pick_id": pick_id, "result": result, "pnl_units": pnl}
