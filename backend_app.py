from fastapi import FastAPI, Query, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from nba_api.stats.endpoints import leaguedashteamshotlocations, leaguedashteamstats, playergamelog
from nba_api.stats.static import players, teams
from pydantic import BaseModel
import datetime
import hashlib
from html.parser import HTMLParser
from io import StringIO
import math
import os
import random
import re
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
NBA_HTTP_TIMEOUT_SECONDS = float(os.getenv("NBA_HTTP_TIMEOUT_SECONDS", "20"))
NBA_HTTP_RETRIES = int(os.getenv("NBA_HTTP_RETRIES", "3"))
NBA_RETRY_BACKOFF_SECONDS = float(os.getenv("NBA_RETRY_BACKOFF_SECONDS", "0.5"))
NBA_CIRCUIT_BREAKER_ENABLED = os.getenv("NBA_CIRCUIT_BREAKER_ENABLED", "false").strip().lower() in ("1", "true", "yes", "on")
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
MLB_SEASON_YEAR = int(os.getenv("MLB_SEASON_YEAR", str(datetime.datetime.now().year)))
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
WEB_SCRAPE_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}
MLB_PITCH_PROP_WEIGHTS = {
    "strikeouts_over": {
        "pitcher_skill": 0.36,
        "pitch_quality": 0.18,
        "opponent_profile": 0.22,
        "pitch_matchup": 0.14,
        "context": 0.10,
    },
    "walks_under": {
        "pitcher_skill": 0.40,
        "pitch_quality": 0.10,
        "opponent_profile": 0.28,
        "pitch_matchup": 0.07,
        "context": 0.15,
    },
    "earned_runs_under": {
        "pitcher_skill": 0.32,
        "pitch_quality": 0.08,
        "opponent_profile": 0.30,
        "pitch_matchup": 0.12,
        "context": 0.18,
    },
    "hits_allowed_under": {
        "pitcher_skill": 0.30,
        "pitch_quality": 0.12,
        "opponent_profile": 0.28,
        "pitch_matchup": 0.15,
        "context": 0.15,
    },
    "outs_recorded_over": {
        "pitcher_skill": 0.34,
        "pitch_quality": 0.10,
        "opponent_profile": 0.28,
        "pitch_matchup": 0.10,
        "context": 0.18,
    },
}
MLB_PITCH_ODDS_MARKETS = {
    "pitcher_strikeouts": {
        "prop_key": "strikeouts",
        "display_name": "Pitcher Strikeouts",
        "preferred_bet_type": "strikeouts_over",
        "std_dev": 1.85,
    },
    "pitcher_walks": {
        "prop_key": "walks",
        "display_name": "Pitcher Walks",
        "preferred_bet_type": "walks_under",
        "std_dev": 1.05,
    },
    "pitcher_earned_runs": {
        "prop_key": "earned_runs",
        "display_name": "Pitcher Earned Runs",
        "preferred_bet_type": "earned_runs_under",
        "std_dev": 1.25,
    },
    "pitcher_hits_allowed": {
        "prop_key": "hits_allowed",
        "display_name": "Pitcher Hits Allowed",
        "preferred_bet_type": "hits_allowed_under",
        "std_dev": 1.65,
    },
    "pitcher_outs": {
        "prop_key": "outs_recorded",
        "display_name": "Pitcher Outs Recorded",
        "preferred_bet_type": "outs_recorded_over",
        "std_dev": 2.4,
    },
}
MLB_DEFAULT_PITCH_MARKETS = ",".join(MLB_PITCH_ODDS_MARKETS.keys())
ROTOWIRE_LINEUP_STATUSES = {"Confirmed Lineup", "Expected Lineup", "Unknown Lineup"}
ROTOWIRE_POSITION_TOKENS = {"C", "1B", "2B", "3B", "SS", "LF", "CF", "RF", "DH", "P"}
ROTOWIRE_HAND_TOKENS = {"L", "R", "S"}
MLB_PITCH_CODE_ALIASES = {
    "FA": "FF",
    "FO": "FS",
    "EP": "CS",
    "SC": "ST",
}
SAVANT_PITCH_QUERY_TEMPLATE = {
    "batter_stands": "",
    "chk_pitch_type": "on",
    "chk_stats_release_extension": "on",
    "chk_stats_spin_rate": "on",
    "chk_stats_velocity": "on",
    "game_date_gt": "",
    "game_date_lt": "",
    "group_by": "name-year",
    "hfAB": "",
    "hfBBL": "",
    "hfBBT": "",
    "hfC": "",
    "hfFlag": "",
    "hfGT": "R|",
    "hfInfield": "",
    "hfInn": "",
    "hfMo": "",
    "hfNewZones": "",
    "hfOpponent": "",
    "hfOutfield": "",
    "hfOuts": "",
    "hfPR": "",
    "hfPull": "",
    "hfRO": "",
    "hfSA": "",
    "hfSit": "",
    "hfStadium": "",
    "hfTeam": "",
    "hfZ": "",
    "home_road": "",
    "metric_1": "",
    "min_pas": 0,
    "min_pitches": 0,
    "min_results": 0,
    "pitcher_throws": "",
    "player_event_sort": "api_p_release_speed",
    "player_type": "pitcher",
    "position": "",
    "sort_col": "velocity",
    "sort_order": "desc",
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
    pitcher_name: str = ""
    venue: str = ""
    wind_mph: Optional[float] = None
    wind_direction: str = ""
    temperature_f: Optional[float] = None
    altitude_ft: Optional[float] = None


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
    if "fangraphs.com" in url:
        return "fangraphs"
    if "rotowire.com" in url:
        return "rotowire"
    if "baseballsavant.mlb.com" in url:
        return "baseballsavant"
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
    if provider == "nba_api" and not NBA_CIRCUIT_BREAKER_ENABLED:
        return False
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


def _percent_value(value) -> Optional[float]:
    if value in (None, ""):
        return None
    if isinstance(value, str):
        text = value.strip()
        if not text or text in ("--", "nan"):
            return None
        if text.endswith("%"):
            return _safe_float(text[:-1], None)
    return _safe_float(value, None)


def _score_band(value: Optional[float], low: float, high: float, higher_is_better: bool = True, default: float = 0.5) -> float:
    if value is None:
        return default
    if high < low:
        low, high = high, low
    if high <= low:
        return default
    scaled = _clamp((float(value) - low) / (high - low), 0.0, 1.0)
    return scaled if higher_is_better else 1.0 - scaled


def _avg_score(*values: Optional[float], default: float = 0.5) -> float:
    valid = [float(v) for v in values if v is not None]
    if not valid:
        return default
    return sum(valid) / len(valid)


def _name_key(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(name or "").lower())


def _team_name_key(name: str) -> str:
    text = str(name or "").lower()
    text = text.replace(".", "").replace("'", "")
    text = text.replace("d-backs", "diamondbacks")
    text = re.sub(r"[^a-z0-9]+", "", text)
    return text


def _team_short_name_key(name: str) -> str:
    tokens = [t for t in re.split(r"[\s-]+", str(name or "").lower()) if t]
    if not tokens:
        return ""
    return re.sub(r"[^a-z0-9]+", "", tokens[-1])


def _parse_baseball_innings(value: Any) -> Optional[float]:
    if value in (None, ""):
        return None
    text = str(value).strip()
    if not text:
        return None
    if "." not in text:
        return _safe_float(text, None)
    whole, frac = text.split(".", 1)
    try:
        whole_int = int(whole)
    except Exception:
        return _safe_float(text, None)
    if frac == "1":
        return whole_int + (1.0 / 3.0)
    if frac == "2":
        return whole_int + (2.0 / 3.0)
    return _safe_float(text, None)


def _player_name_tokens(name: str) -> Tuple[str, str, str]:
    tokens = [t for t in re.split(r"\s+", str(name or "").replace(".", " ").strip()) if t]
    if not tokens:
        return "", "", ""
    first = tokens[0].lower()
    last = tokens[-1].lower()
    return first, first[:1], last


def _player_name_matches(candidate: str, target: str) -> bool:
    c_full = _name_key(candidate)
    t_full = _name_key(target)
    if c_full == t_full:
        return True
    c_first, c_initial, c_last = _player_name_tokens(candidate)
    t_first, t_initial, t_last = _player_name_tokens(target)
    if not c_last or not t_last:
        return False
    if c_last != t_last:
        return False
    if c_first == t_first and c_first:
        return True
    return bool(c_initial and t_initial and c_initial == t_initial)


def _normal_cdf(x: float, mean: float, std_dev: float) -> float:
    if std_dev <= 0:
        return 0.5 if x == mean else (1.0 if x > mean else 0.0)
    z = (float(x) - float(mean)) / (float(std_dev) * math.sqrt(2.0))
    return 0.5 * (1.0 + math.erf(z))


def _american_from_probability(prob_pct: Optional[float]) -> Optional[int]:
    if prob_pct is None:
        return None
    prob = max(0.0001, min(99.9999, float(prob_pct))) / 100.0
    if prob >= 0.5:
        return int(round(-100.0 * prob / (1.0 - prob)))
    return int(round(100.0 * (1.0 - prob) / prob))


def _ev_units_for_american(prob_pct: Optional[float], american_odds: Optional[int]) -> Optional[float]:
    if prob_pct is None or american_odds in (None, 0):
        return None
    prob = max(0.0, min(1.0, float(prob_pct) / 100.0))
    odds = int(american_odds)
    if odds > 0:
        return round((prob * (odds / 100.0)) - (1.0 - prob), 4)
    return round((prob * (100.0 / abs(odds))) - (1.0 - prob), 4)


def _fetch_text(url: str, params: Optional[dict] = None, headers: Optional[dict] = None) -> str:
    provider = _provider_name_from_url(url)
    if _provider_is_open(provider):
        raise HTTPException(status_code=503, detail=f"Provider {provider} is temporarily unavailable")
    last_exc = None
    retries = max(1, EXTERNAL_HTTP_RETRIES)
    timeout = max(1.0, EXTERNAL_HTTP_TIMEOUT_SECONDS)
    for attempt in range(retries):
        try:
            resp = requests.get(url, params=params or {}, headers=headers or WEB_SCRAPE_HEADERS, timeout=timeout)
            resp.raise_for_status()
            _provider_note_success(provider)
            return resp.text
        except Exception as exc:
            last_exc = exc
            _provider_note_failure(provider, f"{type(exc).__name__}: {exc}")
            if attempt < retries - 1:
                time.sleep(EXTERNAL_RETRY_BACKOFF_SECONDS * (attempt + 1))
    raise HTTPException(status_code=502, detail=f"Provider request failed ({provider}): {last_exc}")


def _read_html_table_records(html: str, required_columns: Tuple[str, ...]) -> List[Dict[str, Any]]:
    try:
        tables = pd.read_html(StringIO(html))
    except Exception:
        return []
    target = None
    normalized_required = {str(col).strip() for col in required_columns}
    for df in tables:
        df.columns = [str(col).strip() for col in df.columns]
        if normalized_required.issubset(set(df.columns)):
            target = df
            break
    if target is None:
        return []
    records: List[Dict[str, Any]] = []
    for row in target.to_dict(orient="records"):
        cleaned = {}
        for key, value in row.items():
            key_text = str(key).strip()
            if isinstance(value, str):
                value = value.replace("\xa0", " ").strip()
            cleaned[key_text] = value
        if cleaned.get("Name") in ("Name", "", None) or cleaned.get("Player") in ("Player", "", None):
            continue
        records.append(cleaned)
    return records


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

    result = {
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


def _mlb_game_logs(player: str, season_year: int, season_type: str, player_id: Optional[int] = None):
    if player_id is None:
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


def _mlb_team_map():
    cache_key = ("mlb_team_map",)
    cached = _cached_external(cache_key)
    if cached is not None:
        return cached
    url = "https://statsapi.mlb.com/api/v1/teams"
    data = _fetch_json(url, params={"sportId": 1})
    mapping = {}
    for team in data.get("teams", []) if isinstance(data, dict) else []:
        abbrev = team.get("abbreviation") or team.get("fileCode") or team.get("teamCode")
        if not abbrev:
            continue
        mapping[str(abbrev).upper()] = {
            "id": int(team.get("id")),
            "name": team.get("name") or team.get("teamName") or str(abbrev).upper(),
        }
    _set_cached_external(cache_key, mapping)
    return mapping


def _mlb_team_id(abbrev: str) -> Optional[int]:
    if not abbrev:
        return None
    mapping = _mlb_team_map()
    entry = mapping.get(abbrev.strip().upper())
    if not entry:
        return None
    return entry.get("id")


def _mlb_team_abbrev(team_id: Optional[int]) -> str:
    if not team_id:
        return ""
    for abbrev, entry in _mlb_team_map().items():
        if int(entry.get("id", 0)) == int(team_id):
            return abbrev
    return ""


def _mlb_player_profile(player: str) -> Dict[str, Any]:
    player_id = _mlb_find_player_id(player)
    if not player_id:
        return {}
    cache_key = ("mlb_profile", player_id)
    cached = _cached_external(cache_key)
    if cached is not None:
        return cached or {}
    url = f"https://statsapi.mlb.com/api/v1/people/{player_id}"
    data = _fetch_json(url)
    person = (data.get("people") or [{}])[0] if isinstance(data, dict) else {}
    pos = person.get("primaryPosition") or {}
    team = person.get("currentTeam") or {}
    profile = {
        "id": int(player_id),
        "name": person.get("fullName") or person.get("name") or player,
        "position_code": str(pos.get("code") or ""),
        "position_abbrev": str(pos.get("abbreviation") or ""),
        "position_name": str(pos.get("name") or ""),
        "team_id": team.get("id"),
        "team_name": team.get("name") or team.get("teamName") or "",
        "bat_side": (person.get("batSide") or {}).get("code"),
        "pitch_hand": (person.get("pitchHand") or {}).get("code"),
    }
    _set_cached_external(cache_key, profile)
    return profile


def _mlb_player_stats(player_id: int, group: str, season_year: int) -> Dict[str, Any]:
    cache_key = ("mlb_player_stats", player_id, group, season_year)
    cached = _cached_external(cache_key)
    if cached is not None:
        return cached or {}
    url = f"https://statsapi.mlb.com/api/v1/people/{player_id}/stats"
    data = _fetch_json(url, params={"stats": "season", "group": group, "season": season_year})
    stat = {}
    for block in data.get("stats", []) if isinstance(data, dict) else []:
        splits = block.get("splits") or []
        if not splits:
            continue
        stat = splits[0].get("stat") or {}
        break
    _set_cached_external(cache_key, stat)
    return stat


def _mlb_player_stats_with_fallback(player_id: int, group: str, season_year: int) -> Tuple[Dict[str, Any], int, bool]:
    stats = _mlb_player_stats(player_id, group, season_year)
    if stats:
        return stats, season_year, False
    if season_year > 0:
        prev = _mlb_player_stats(player_id, group, season_year - 1)
        if prev:
            return prev, season_year - 1, True
    return {}, season_year, False


def _mlb_team_stats(team_id: int, group: str, season_year: int) -> Dict[str, Any]:
    cache_key = ("mlb_team_stats", team_id, group, season_year)
    cached = _cached_external(cache_key)
    if cached is not None:
        return cached or {}
    url = f"https://statsapi.mlb.com/api/v1/teams/{team_id}/stats"
    data = _fetch_json(url, params={"stats": "season", "group": group, "season": season_year})
    stat = {}
    for block in data.get("stats", []) if isinstance(data, dict) else []:
        splits = block.get("splits") or []
        if not splits:
            continue
        stat = splits[0].get("stat") or {}
        break
    _set_cached_external(cache_key, stat)
    return stat


def _mlb_team_stats_with_fallback(team_id: int, group: str, season_year: int) -> Tuple[Dict[str, Any], int, bool]:
    stats = _mlb_team_stats(team_id, group, season_year)
    if stats:
        return stats, season_year, False
    if season_year > 0:
        prev = _mlb_team_stats(team_id, group, season_year - 1)
        if prev:
            return prev, season_year - 1, True
    return {}, season_year, False


def _mlb_find_schedule_game(team_id: int, opponent_id: int, season_year: int) -> Optional[dict]:
    if not team_id or not opponent_id:
        return None
    cache_key = ("mlb_schedule", team_id, opponent_id, season_year)
    cached = _cached_external(cache_key)
    if cached is not None:
        return cached or None
    today = datetime.date.today()
    end_date = today + datetime.timedelta(days=7)
    url = "https://statsapi.mlb.com/api/v1/schedule"
    data = _fetch_json(
        url,
        params={
            "sportId": 1,
            "teamId": team_id,
            "opponentId": opponent_id,
            "season": season_year,
            "gameTypes": "R,P",
            "startDate": today.isoformat(),
            "endDate": end_date.isoformat(),
        },
    )
    games = []
    for date_block in data.get("dates", []) if isinstance(data, dict) else []:
        for game in date_block.get("games", []) or []:
            game_date = game.get("gameDate") or ""
            games.append((game_date, game))
    if not games:
        _set_cached_external(cache_key, None)
        return None
    games.sort(key=lambda x: x[0])
    selected = games[0][1]
    _set_cached_external(cache_key, selected)
    return selected


def _mlb_schedule_games_for_date(game_date: datetime.date) -> List[Dict[str, Any]]:
    cache_key = ("mlb_schedule_daily", game_date.isoformat())
    cached = _cached_external(cache_key)
    if cached is not None:
        return cached or []
    data = _fetch_json(
        "https://statsapi.mlb.com/api/v1/schedule",
        params={"sportId": 1, "date": game_date.isoformat(), "gameTypes": "R,P"},
    )
    games: List[Dict[str, Any]] = []
    for date_block in data.get("dates", []) if isinstance(data, dict) else []:
        for game in date_block.get("games", []) or []:
            teams = game.get("teams") or {}
            away = teams.get("away") or {}
            home = teams.get("home") or {}
            away_team = away.get("team") or {}
            home_team = home.get("team") or {}
            game_row = {
                "game_pk": game.get("gamePk"),
                "game_date": game.get("gameDate"),
                "status": ((game.get("status") or {}).get("detailedState") or ""),
                "venue": ((game.get("venue") or {}).get("name") or ""),
                "away": {
                    "id": away_team.get("id"),
                    "name": away_team.get("name") or away_team.get("teamName") or "",
                    "abbr": _mlb_team_abbrev(away_team.get("id")),
                    "probable_pitcher": away.get("probablePitcher") or {},
                },
                "home": {
                    "id": home_team.get("id"),
                    "name": home_team.get("name") or home_team.get("teamName") or "",
                    "abbr": _mlb_team_abbrev(home_team.get("id")),
                    "probable_pitcher": home.get("probablePitcher") or {},
                },
            }
            games.append(game_row)
    _set_cached_external(cache_key, games)
    return games


def _mlb_probable_pitcher(game: dict, opponent_id: int) -> Optional[dict]:
    if not game or not opponent_id:
        return None
    teams = game.get("teams") or {}
    for side in ("home", "away"):
        team = (teams.get(side) or {}).get("team") or {}
        if int(team.get("id", 0)) == int(opponent_id):
            pitcher = (teams.get(side) or {}).get("probablePitcher") or {}
            if pitcher:
                return {"id": pitcher.get("id"), "name": pitcher.get("fullName") or pitcher.get("name")}
    return None


def _mlb_lineup_context(game_pk: int, team_id: int, player_id: int) -> Dict[str, Any]:
    if not game_pk or not team_id or not player_id:
        return {}
    cache_key = ("mlb_lineup", game_pk, team_id, player_id)
    cached = _cached_external(cache_key)
    if cached is not None:
        return cached or {}
    url = f"https://statsapi.mlb.com/api/v1/game/{game_pk}/boxscore"
    payload = _fetch_json(url)
    teams = payload.get("teams") if isinstance(payload, dict) else {}
    target = None
    for side in ("home", "away"):
        block = teams.get(side) or {}
        team = block.get("team") or {}
        if int(team.get("id", 0)) == int(team_id):
            target = block
            break
    if not target:
        _set_cached_external(cache_key, None)
        return {}
    batting_order = target.get("battingOrder") or []
    lineup_spot = None
    ahead_ids = []
    if isinstance(batting_order, list) and batting_order:
        try:
            idx = [int(pid) for pid in batting_order].index(int(player_id))
            lineup_spot = idx + 1
            ahead_ids = [int(pid) for pid in batting_order[:idx]]
        except ValueError:
            lineup_spot = None
    if lineup_spot is None:
        players = target.get("players") or {}
        for key, pdata in players.items():
            pid = pdata.get("person", {}).get("id")
            if not pid or int(pid) != int(player_id):
                continue
            raw = pdata.get("battingOrder")
            if raw is None:
                continue
            try:
                spot = int(raw) // 100
            except Exception:
                continue
            if spot:
                lineup_spot = spot
            break
    result = {"lineup_spot": lineup_spot, "ahead_ids": ahead_ids}
    _set_cached_external(cache_key, result)
    return result


MLB_PARK_ALTITUDE_FT = {
    "Coors Field": 5183,
    "Chase Field": 1090,
    "Kauffman Stadium": 900,
    "Truist Park": 1026,
    "Great American Ball Park": 490,
    "Globe Life Field": 551,
    "Wrigley Field": 594,
    "Fenway Park": 20,
    "Yankee Stadium": 55,
    "Citi Field": 25,
    "Citizens Bank Park": 39,
    "Oriole Park at Camden Yards": 33,
    "Nationals Park": 25,
    "loanDepot park": 8,
    "Tropicana Field": 11,
    "Minute Maid Park": 43,
    "T-Mobile Park": 52,
    "Oracle Park": 13,
    "Petco Park": 62,
    "Dodger Stadium": 340,
    "Angel Stadium": 160,
    "Oakland Coliseum": 14,
    "Sutter Health Park": 35,
    "American Family Field": 635,
    "Target Field": 840,
    "Guaranteed Rate Field": 595,
    "Comerica Park": 585,
    "Progressive Field": 653,
    "PNC Park": 725,
    "Busch Stadium": 466,
    "Rogers Centre": 249,
}


def _mlb_parse_wind_text(text: str) -> Tuple[Optional[float], str]:
    if not text:
        return None, ""
    lowered = str(text).strip().lower()
    mph = _extract_first_number(lowered)
    direction = ""
    if "out to" in lowered:
        direction = "out"
    elif "in from" in lowered:
        direction = "in"
    elif "left to right" in lowered:
        direction = "left_to_right"
    elif "right to left" in lowered:
        direction = "right_to_left"
    elif "calm" in lowered or "none" in lowered:
        direction = "calm"
    return mph, direction


def _mlb_game_environment(game_pk: Optional[int]) -> Dict[str, Any]:
    if not game_pk:
        return {}
    cache_key = ("mlb_game_env", int(game_pk))
    cached = _cached_external(cache_key)
    if cached is not None:
        return cached or {}
    url = f"https://statsapi.mlb.com/api/v1.1/game/{int(game_pk)}/feed/live"
    payload = _fetch_json(url)
    game_data = payload.get("gameData", {}) if isinstance(payload, dict) else {}
    venue = (game_data.get("venue") or {}).get("name") or ""
    weather = game_data.get("weather") or {}
    temp_raw = weather.get("temp")
    temp_f = _extract_first_number(str(temp_raw)) if temp_raw is not None else None
    wind_text = str(weather.get("wind") or "")
    wind_mph, wind_direction = _mlb_parse_wind_text(wind_text)
    altitude_ft = MLB_PARK_ALTITUDE_FT.get(str(venue))
    out = {
        "venue": str(venue or ""),
        "temperature_f": temp_f,
        "wind_mph": wind_mph,
        "wind_direction": wind_direction,
        "altitude_ft": altitude_ft,
        "raw_weather": weather,
    }
    _set_cached_external(cache_key, out)
    return out


def _mlb_player_pitch_arsenal(player_id: int, group: str, season_year: int) -> List[dict]:
    cache_key = ("mlb_pitch_arsenal", player_id, group, season_year)
    cached = _cached_external(cache_key)
    if cached is not None:
        return cached or []
    url = f"https://statsapi.mlb.com/api/v1/people/{player_id}"
    hydrate = f"stats(group=[{group}],type=[pitchArsenal],season={season_year})"
    payload = _fetch_json(url, params={"hydrate": hydrate})
    people = payload.get("people", []) if isinstance(payload, dict) else []
    if not people:
        _set_cached_external(cache_key, [])
        return []
    stats_blocks = people[0].get("stats") or []
    if not stats_blocks:
        _set_cached_external(cache_key, [])
        return []
    splits = stats_blocks[0].get("splits") or []
    rows = []
    for split in splits:
        stat = split.get("stat") or {}
        ptype = stat.get("type") or {}
        rows.append(
            {
                "code": str(ptype.get("code") or ""),
                "description": str(ptype.get("description") or ""),
                "percentage": _safe_float(stat.get("percentage")),
                "count": int(_safe_float(stat.get("count"))) if stat.get("count") is not None else None,
                "total_pitches": int(_safe_float(stat.get("totalPitches"))) if stat.get("totalPitches") is not None else None,
                "avg_speed": _safe_float(stat.get("averageSpeed"), None),
            }
        )
    _set_cached_external(cache_key, rows)
    return rows


def _mlb_player_pitch_arsenal_with_fallback(player_id: int, group: str, season_year: int) -> Tuple[List[dict], int, bool]:
    rows = _mlb_player_pitch_arsenal(player_id, group, season_year)
    if rows:
        return rows, season_year, False
    if season_year > 0:
        prev = _mlb_player_pitch_arsenal(player_id, group, season_year - 1)
        if prev:
            return prev, season_year - 1, True
    return [], season_year, False


def _mlb_primary_pitch(player_id: int, season_year: int) -> Dict[str, Any]:
    rows, used_year, fallback = _mlb_player_pitch_arsenal_with_fallback(player_id, "pitching", season_year)
    if not rows:
        return {}
    rows_sorted = sorted(rows, key=lambda x: float(x.get("percentage") or 0.0), reverse=True)
    top = rows_sorted[0]
    return {
        "code": top.get("code"),
        "description": top.get("description"),
        "usage_pct": round(float(top.get("percentage") or 0.0) * 100.0, 2),
        "avg_speed": round(float(top.get("avg_speed")), 2) if top.get("avg_speed") is not None else None,
        "season_year": used_year,
        "fallback": fallback,
    }


def _mlb_hitter_pitch_profile(player_id: int, season_year: int) -> Tuple[Dict[str, dict], int, bool]:
    cache_key = ("mlb_hitter_pitch_profile", player_id, season_year)
    cached = _cached_external(cache_key)
    if cached is not None:
        return cached.get("profile", {}), int(cached.get("season_year", season_year)), bool(cached.get("fallback", False))

    def _build(year: int) -> Dict[str, dict]:
        url = f"https://statsapi.mlb.com/api/v1/people/{player_id}"
        hydrate = f"stats(group=[hitting],type=[pitchLog],season={year})"
        payload = _fetch_json(url, params={"hydrate": hydrate})
        people = payload.get("people", []) if isinstance(payload, dict) else []
        if not people:
            return {}
        stats_blocks = people[0].get("stats") or []
        if not stats_blocks:
            return {}
        splits = stats_blocks[0].get("splits") or []
        profile: Dict[str, dict] = {}
        for split in splits:
            stat = split.get("stat", {}) or {}
            play = stat.get("play", {}) or {}
            details = play.get("details", {}) or {}
            pitch_type = details.get("type", {}) or {}
            code = str(pitch_type.get("code") or "").upper()
            if not code:
                continue
            is_at_bat = bool(details.get("isAtBat"))
            if not is_at_bat:
                continue
            event_type = str(details.get("eventType") or "").lower()
            base_hit = bool(details.get("isBaseHit"))
            rec = profile.setdefault(code, {"pitch_code": code, "at_bats": 0, "hits": 0, "home_runs": 0, "total_bases": 0, "strikeouts": 0})
            rec["at_bats"] += 1
            if base_hit:
                rec["hits"] += 1
            if event_type == "home_run":
                rec["home_runs"] += 1
            if event_type in ("strikeout", "strikeout_double_play"):
                rec["strikeouts"] += 1
            if event_type == "single":
                rec["total_bases"] += 1
            elif event_type == "double":
                rec["total_bases"] += 2
            elif event_type == "triple":
                rec["total_bases"] += 3
            elif event_type == "home_run":
                rec["total_bases"] += 4
        for code, rec in profile.items():
            ab = max(1, int(rec["at_bats"]))
            rec["avg"] = round(rec["hits"] / ab, 3)
            rec["slg"] = round(rec["total_bases"] / ab, 3)
            rec["hr_rate"] = round(rec["home_runs"] / ab, 3)
            rec["k_rate"] = round(rec["strikeouts"] / ab, 3)
        return profile

    profile = _build(season_year)
    used_year = season_year
    fallback = False
    if not profile and season_year > 0:
        used_year = season_year - 1
        profile = _build(used_year)
        fallback = bool(profile)
    _set_cached_external(cache_key, {"profile": profile, "season_year": used_year, "fallback": fallback})
    return profile, used_year, fallback


class _HTMLTextCollector(HTMLParser):
    def __init__(self):
        super().__init__()
        self.tokens: List[str] = []

    def handle_data(self, data: str):
        text = str(data or "").replace("\xa0", " ")
        text = re.sub(r"\s+", " ", text).strip()
        if text:
            self.tokens.append(text)


def _token_is_time(value: str) -> bool:
    return bool(re.match(r"^\d{1,2}:\d{2}\s+(AM|PM)\s+ET$", str(value or "").strip()))


def _token_is_team_abbrev(value: str) -> bool:
    text = str(value or "").strip().upper()
    if text in {"LINE", "AL", "NL", "O/U"}:
        return False
    return bool(re.match(r"^[A-Z]{2,3}$", text))


def _parse_rotowire_team_section(tokens: List[str], status_idx: int) -> Dict[str, Any]:
    status = tokens[status_idx]
    pitcher_name = ""
    pitcher_hand = ""
    for idx in range(max(1, status_idx - 8), status_idx):
        if idx + 1 < status_idx and tokens[idx + 1] in ROTOWIRE_HAND_TOKENS:
            pitcher_name = tokens[idx]
            pitcher_hand = tokens[idx + 1]
    players = []
    i = status_idx + 1
    while i < len(tokens):
        tok = tokens[i]
        if tok in ROTOWIRE_LINEUP_STATUSES or tok.startswith("Umpire:") or _token_is_time(tok) or tok in ("LINE", "O/U", "Home Run Odds", "Starting Pitcher Intel"):
            break
        if tok in ROTOWIRE_POSITION_TOKENS and i + 2 < len(tokens):
            name = tokens[i + 1]
            hand = tokens[i + 2] if tokens[i + 2] in ROTOWIRE_HAND_TOKENS else ""
            players.append({"position": tok, "name": name, "hand": hand})
            i += 3
            while i < len(tokens) and (tokens[i].startswith("$") or tokens[i] in ("Home Run Odds", "Starting Pitcher Intel")):
                if tokens[i] in ("Home Run Odds", "Starting Pitcher Intel"):
                    break
                i += 1
            continue
        i += 1
    return {
        "status": status,
        "confirmed": status == "Confirmed Lineup",
        "pitcher": {"name": pitcher_name, "hand": pitcher_hand},
        "players": players[:9],
    }


def _parse_rotowire_daily_lineups(html: str) -> List[Dict[str, Any]]:
    parser = _HTMLTextCollector()
    parser.feed(html or "")
    tokens = parser.tokens
    time_indices = [idx for idx, token in enumerate(tokens) if _token_is_time(token)]
    games: List[Dict[str, Any]] = []
    for block_idx, start in enumerate(time_indices):
        end = time_indices[block_idx + 1] if block_idx + 1 < len(time_indices) else len(tokens)
        block = tokens[start:end]
        if len(block) < 10:
            continue
        teams = [token for token in block[:20] if _token_is_team_abbrev(token)]
        if len(teams) < 2:
            continue
        status_indices = [idx for idx, token in enumerate(block) if token in ROTOWIRE_LINEUP_STATUSES]
        if len(status_indices) < 2:
            continue
        away = _parse_rotowire_team_section(block, status_indices[0])
        home = _parse_rotowire_team_section(block, status_indices[1])
        block_text = " ".join(block)
        umpire_name = ""
        umpire_runs = None
        umpire_ks = None
        weather_match = re.search(r"(\d+)%\s+(\d+)°\s+Wind\s+(\d+)\s+mph\s+([A-Za-z-]+)", block_text)
        umpire_match = re.search(r"Umpire:\s*(.*?)\s+([0-9.]+)\s+R/G\s+([0-9.]+)\s+K/G", block_text)
        if umpire_match:
            umpire_name = umpire_match.group(1).strip()
            umpire_runs = _safe_float(umpire_match.group(2), None)
            umpire_ks = _safe_float(umpire_match.group(3), None)
        weather = {
            "summary": "Dome In Domed Stadium" if "Dome In Domed Stadium" in block_text else "",
            "temp_f": _safe_float(weather_match.group(2), None) if weather_match else None,
            "wind_mph": _safe_float(weather_match.group(3), None) if weather_match else None,
            "wind_direction": weather_match.group(4).strip() if weather_match else "",
        }
        games.append(
            {
                "time": block[0],
                "away": {"team": teams[0], **away},
                "home": {"team": teams[1], **home},
                "umpire": {"name": umpire_name, "runs_per_game": umpire_runs, "strikeouts_per_game": umpire_ks},
                "weather": weather,
            }
        )
    return games


def _rotowire_lineups_for_date(game_date: datetime.date) -> List[Dict[str, Any]]:
    if game_date != datetime.date.today():
        return []
    cache_key = ("rotowire_lineups", game_date.isoformat())
    cached = _cached_external(cache_key)
    if cached is not None:
        return cached or []
    html = _fetch_text("https://www.rotowire.com/baseball/daily-lineups.php", params={"site": "Yahoo"}, headers=WEB_SCRAPE_HEADERS)
    games = _parse_rotowire_daily_lineups(html)
    _set_cached_external(cache_key, games)
    return games


def _rotowire_lineup_index(game_date: datetime.date) -> Dict[str, Dict[str, Any]]:
    index: Dict[str, Dict[str, Any]] = {}
    for game in _rotowire_lineups_for_date(game_date):
        for side in ("away", "home"):
            team_block = game.get(side) or {}
            team = str(team_block.get("team") or "").upper()
            if not team:
                continue
            index[team] = {
                "game": {"time": game.get("time"), "umpire": game.get("umpire") or {}, "weather": game.get("weather") or {}},
                **team_block,
            }
    return index


def _fangraphs_pitching_table(season_year: int, table_type: int) -> List[Dict[str, Any]]:
    cache_key = ("fangraphs_pitching_table", season_year, table_type)
    cached = _cached_external(cache_key)
    if cached is not None:
        return cached or []
    html = _fetch_text(
        "https://www.fangraphs.com/leaders-legacy.aspx",
        params={"lg": "all", "pos": "all", "qual": 0, "season": season_year, "stats": "pit", "team": 0, "type": table_type},
        headers=WEB_SCRAPE_HEADERS,
    )
    required = {
        1: ("Name", "Team", "K%", "BB%"),
        5: ("Name", "Team", "SwStr%", "CStr%", "CSW%"),
        24: ("Name", "Team", "EV", "HardHit%", "xERA"),
    }.get(table_type, ("Name", "Team"))
    rows = _read_html_table_records(html, required)
    _set_cached_external(cache_key, rows)
    return rows


def _find_named_metric_row(rows: List[Dict[str, Any]], player_name: str, team_abbrev: str = "") -> Dict[str, Any]:
    key = _name_key(player_name)
    team = str(team_abbrev or "").upper()
    candidates = [row for row in rows if _name_key(row.get("Name") or row.get("Player")) == key]
    if not candidates:
        return {}
    if team:
        for row in candidates:
            if str(row.get("Team") or "").upper() == team:
                return row
    return candidates[0]


def _fangraphs_pitcher_snapshot(player_name: str, team_abbrev: str, season_year: int) -> Dict[str, Any]:
    advanced = _find_named_metric_row(_fangraphs_pitching_table(season_year, 1), player_name, team_abbrev)
    plate = _find_named_metric_row(_fangraphs_pitching_table(season_year, 5), player_name, team_abbrev)
    statcast = _find_named_metric_row(_fangraphs_pitching_table(season_year, 24), player_name, team_abbrev)
    return {
        "advanced": advanced,
        "plate_discipline": plate,
        "statcast": statcast,
    }


def _baseballsavant_pitch_table(pitch_code: str, season_year: int) -> List[Dict[str, Any]]:
    normalized_code = MLB_PITCH_CODE_ALIASES.get(str(pitch_code or "").upper(), str(pitch_code or "").upper())
    if not normalized_code:
        return []
    cache_key = ("baseballsavant_pitch_table", normalized_code, season_year)
    cached = _cached_external(cache_key)
    if cached is not None:
        return cached or []
    params = dict(SAVANT_PITCH_QUERY_TEMPLATE)
    params["hfPT"] = f"{normalized_code}|"
    params["hfSea"] = f"{season_year}|"
    html = _fetch_text("https://baseballsavant.mlb.com/statcast_search", params=params, headers=WEB_SCRAPE_HEADERS)
    rows = _read_html_table_records(html, ("Player", "Year", "Pitches"))
    _set_cached_external(cache_key, rows)
    return rows


def _baseballsavant_primary_pitch_snapshot(player_name: str, pitch_code: str, season_year: int) -> Dict[str, Any]:
    row = _find_named_metric_row(_baseballsavant_pitch_table(pitch_code, season_year), player_name, "")
    if not row:
        return {}
    return {
        "pitch_code": str(pitch_code or "").upper(),
        "velocity": _safe_float(row.get("Pitch (MPH)") or row.get("Pitch (MPH) "), None),
        "spin_rate": _safe_float(row.get("Spin Rate") or row.get("Spin Rate (RPM)") or row.get("Spin (RPM)"), None),
        "extension": _safe_float(row.get("Extension (ft)") or row.get("Extension"), None),
        "pitch_pct": _safe_float(row.get("Pitch %"), None),
        "pitches": _safe_float(row.get("Pitches"), None),
    }

def _mlb_pitch_match_factor(prop: str, pitch_stats: Dict[str, Any]) -> float:
    if not pitch_stats:
        return 0.0
    avg = _safe_float(pitch_stats.get("avg"), None)
    slg = _safe_float(pitch_stats.get("slg"), None)
    hr_rate = _safe_float(pitch_stats.get("hr_rate"), None)
    k_rate = _safe_float(pitch_stats.get("k_rate"), None)
    if prop == "home_runs":
        if hr_rate is None:
            return 0.0
        return _clamp((hr_rate - 0.035) * 1.8, -0.06, 0.06)
    if prop == "total_bases":
        if slg is None:
            return 0.0
        return _clamp((slg - 0.390) * 0.35, -0.06, 0.06)
    if prop == "strikeouts":
        if k_rate is None:
            return 0.0
        return _clamp((k_rate - 0.225) * 0.6, -0.06, 0.06)
    if avg is None:
        return 0.0
    return _clamp((avg - 0.245) * 0.8, -0.06, 0.06)


def _mlb_environment_factor(
    prop: str,
    role: str,
    altitude_ft: Optional[float],
    wind_mph: Optional[float],
    wind_direction: str,
    temperature_f: Optional[float],
) -> float:
    factor = 0.0
    offense_boost = 0.0
    if altitude_ft is not None:
        offense_boost += _clamp((float(altitude_ft) - 500.0) / 7000.0, -0.04, 0.08)
    if temperature_f is not None:
        offense_boost += _clamp((float(temperature_f) - 70.0) / 200.0, -0.04, 0.04)
    if wind_mph is not None and wind_direction:
        mph = max(0.0, float(wind_mph))
        if wind_direction in ("out", "out_to_center", "out_to_left", "out_to_right"):
            offense_boost += _clamp(mph / 180.0, 0.0, 0.08)
        elif wind_direction in ("in", "in_from_center", "in_from_left", "in_from_right"):
            offense_boost -= _clamp(mph / 180.0, 0.0, 0.08)
    if role == "pitcher":
        offense_boost *= -1.0
    if prop == "home_runs":
        factor = offense_boost * 1.5
    elif prop == "total_bases":
        factor = offense_boost * 1.2
    elif prop in ("hits", "runs", "rbis"):
        factor = offense_boost
    elif prop == "strikeouts":
        factor = offense_boost * -0.8
    return _clamp(factor, -0.10, 0.10)


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _mlb_stat_float(stats: Dict[str, Any], keys: Tuple[str, ...]) -> Optional[float]:
    for key in keys:
        if key in stats:
            raw = stats.get(key)
            if raw in (None, ""):
                continue
            val = _safe_float(raw, None)
            if val is not None:
                return val
    return None


def _team_stat_per_game(stats: Dict[str, Any], total_keys: Tuple[str, ...]) -> Optional[float]:
    games = _mlb_stat_float(stats, ("gamesPlayed", "games", "g"))
    total = _mlb_stat_float(stats, total_keys)
    if not games or total is None:
        return None
    if float(games) <= 0:
        return None
    return float(total) / float(games)


def _lineup_summary_for_pitcher(lineup_info: Dict[str, Any], primary_pitch: Dict[str, Any], season_year: int) -> Dict[str, Any]:
    players = lineup_info.get("players") or []
    hitter_rows = []
    k_rates = []
    bb_rates = []
    obp_values = []
    avg_vs_pitch = []
    k_vs_pitch = []
    hand_counts = {"L": 0, "R": 0, "S": 0}
    for hitter in players:
        hand = str(hitter.get("hand") or "").upper()
        if hand in hand_counts:
            hand_counts[hand] += 1
        player_name = str(hitter.get("name") or "").strip()
        if not player_name:
            continue
        pid = _mlb_find_player_id(player_name)
        pitch_stats = {}
        if pid:
            stats, _, _ = _mlb_player_stats_with_fallback(int(pid), "hitting", season_year)
            pa = _mlb_stat_float(stats, ("plateAppearances",))
            strikeouts = _mlb_stat_float(stats, ("strikeOuts", "strikeouts"))
            walks = _mlb_stat_float(stats, ("baseOnBalls", "walks"))
            obp = _mlb_stat_float(stats, ("obp", "onBasePercentage"))
            hitter_k_pct = (float(strikeouts) / float(pa) * 100.0) if pa and strikeouts is not None else None
            hitter_bb_pct = (float(walks) / float(pa) * 100.0) if pa and walks is not None else None
            if hitter_k_pct is not None:
                k_rates.append(hitter_k_pct)
            if hitter_bb_pct is not None:
                bb_rates.append(hitter_bb_pct)
            if obp is not None:
                obp_values.append(obp)
            if primary_pitch.get("code"):
                pitch_profile, _, _ = _mlb_hitter_pitch_profile(int(pid), season_year)
                pitch_stats = pitch_profile.get(str(primary_pitch.get("code")).upper(), {})
                pitch_avg = _safe_float(pitch_stats.get("avg"), None)
                pitch_k = _safe_float(pitch_stats.get("k_rate"), None)
                if pitch_avg is not None:
                    avg_vs_pitch.append(float(pitch_avg))
                if pitch_k is not None:
                    k_vs_pitch.append(float(pitch_k) * 100.0)
        hitter_rows.append(
            {
                "name": player_name,
                "hand": hand,
                "position": hitter.get("position"),
                "avg_vs_primary_pitch": round(_safe_float(pitch_stats.get("avg"), None), 3) if pitch_stats and pitch_stats.get("avg") is not None else None,
                "k_rate_vs_primary_pitch": round(_safe_float(pitch_stats.get("k_rate"), None) * 100.0, 2) if pitch_stats and pitch_stats.get("k_rate") is not None else None,
            }
        )
    return {
        "status": lineup_info.get("status") or "",
        "confirmed": bool(lineup_info.get("confirmed")),
        "player_count": len(players),
        "avg_hitter_k_pct": round(sum(k_rates) / len(k_rates), 2) if k_rates else None,
        "avg_hitter_bb_pct": round(sum(bb_rates) / len(bb_rates), 2) if bb_rates else None,
        "avg_obp": round(sum(obp_values) / len(obp_values), 3) if obp_values else None,
        "avg_vs_primary_pitch": round(sum(avg_vs_pitch) / len(avg_vs_pitch), 3) if avg_vs_pitch else None,
        "avg_k_vs_primary_pitch": round(sum(k_vs_pitch) / len(k_vs_pitch), 2) if k_vs_pitch else None,
        "handedness": hand_counts,
        "players": hitter_rows,
    }


def _metric_bundle_for_pitcher(
    pitcher_name: str,
    pitcher_team: str,
    pitcher_profile: Dict[str, Any],
    statsapi_stats: Dict[str, Any],
    fangraphs: Dict[str, Any],
    primary_pitch: Dict[str, Any],
    savant_pitch: Dict[str, Any],
) -> Dict[str, Any]:
    advanced = fangraphs.get("advanced") or {}
    plate = fangraphs.get("plate_discipline") or {}
    statcast = fangraphs.get("statcast") or {}
    batters_faced = _mlb_stat_float(statsapi_stats, ("battersFaced",))
    strikeouts = _mlb_stat_float(statsapi_stats, ("strikeOuts", "strikeouts"))
    walks = _mlb_stat_float(statsapi_stats, ("baseOnBalls", "walks"))
    innings_pitched = _parse_baseball_innings(statsapi_stats.get("inningsPitched"))
    games_started = _mlb_stat_float(statsapi_stats, ("gamesStarted", "gamesStartedAsStarter"))
    k_pct = _percent_value(advanced.get("K%"))
    if k_pct is None and batters_faced and strikeouts is not None:
        k_pct = round((float(strikeouts) / float(batters_faced)) * 100.0, 2)
    bb_pct = _percent_value(advanced.get("BB%"))
    if bb_pct is None and batters_faced and walks is not None:
        bb_pct = round((float(walks) / float(batters_faced)) * 100.0, 2)
    k_bb_pct = _percent_value(advanced.get("K-BB%"))
    if k_bb_pct is None and k_pct is not None and bb_pct is not None:
        k_bb_pct = round(k_pct - bb_pct, 2)
    return {
        "name": pitcher_name,
        "team": pitcher_team,
        "pitch_hand": pitcher_profile.get("pitch_hand"),
        "k_pct": k_pct,
        "bb_pct": bb_pct,
        "k_bb_pct": k_bb_pct,
        "bb_per_9": _safe_float(advanced.get("BB/9"), _mlb_stat_float(statsapi_stats, ("baseOnBallsPer9Inn", "walksPer9Inn"))),
        "k_per_9": _safe_float(advanced.get("K/9"), _mlb_stat_float(statsapi_stats, ("strikeoutsPer9Inn", "strikeOutsPer9Inn"))),
        "hits_per_9": _mlb_stat_float(statsapi_stats, ("hitsPer9Inn",)),
        "siera": _safe_float(advanced.get("SIERA"), None),
        "swstr_pct": _percent_value(plate.get("SwStr%")),
        "cstr_pct": _percent_value(plate.get("CStr%")),
        "csw_pct": _percent_value(plate.get("CSW%")),
        "fstrike_pct": _percent_value(plate.get("F-Strike%")),
        "era": _safe_float(advanced.get("ERA"), _mlb_stat_float(statsapi_stats, ("era",))),
        "whip": _safe_float(advanced.get("WHIP"), _mlb_stat_float(statsapi_stats, ("whip",))),
        "hr_per_9": _safe_float(advanced.get("HR/9"), _mlb_stat_float(statsapi_stats, ("homeRunsPer9", "homeRunsPer9Inn"))),
        "xera": _safe_float(statcast.get("xERA"), None),
        "ev": _safe_float(statcast.get("EV"), None),
        "hard_hit_pct": _percent_value(statcast.get("HardHit%")),
        "innings_pitched": innings_pitched,
        "games_started": games_started,
        "ip_per_start": round(float(innings_pitched) / float(games_started), 2) if innings_pitched and games_started else None,
        "primary_pitch": {
            "code": primary_pitch.get("code"),
            "description": primary_pitch.get("description"),
            "usage_pct": primary_pitch.get("usage_pct"),
            "velocity": savant_pitch.get("velocity") if savant_pitch.get("velocity") is not None else primary_pitch.get("avg_speed"),
            "spin_rate": savant_pitch.get("spin_rate"),
            "extension": savant_pitch.get("extension"),
        },
        "source_flags": {
            "fangraphs": bool(advanced or plate or statcast),
            "baseballsavant": bool(savant_pitch),
            "statsapi": bool(statsapi_stats),
        },
    }


def _context_score_from_environment(prop: str, altitude_ft: Optional[float], wind_mph: Optional[float], wind_direction: str, temperature_f: Optional[float]) -> float:
    raw = _mlb_environment_factor(prop, "pitcher", altitude_ft, wind_mph, wind_direction, temperature_f)
    return _clamp((raw + 0.10) / 0.20, 0.0, 1.0)


def _score_pitcher_prop_candidates(snapshot: Dict[str, Any]) -> List[Dict[str, Any]]:
    pitcher = snapshot.get("pitcher") or {}
    lineup = snapshot.get("lineup") or {}
    opponent = snapshot.get("opponent_team") or {}
    context = snapshot.get("context") or {}
    primary_pitch = pitcher.get("primary_pitch") or {}
    confirmed_boost = 1.0 if lineup.get("confirmed") else 0.58
    opponent_k_pg = opponent.get("strikeouts_per_game")
    opponent_bb_pg = opponent.get("walks_per_game")
    opponent_runs_pg = opponent.get("runs_per_game")
    opponent_ops = opponent.get("ops")
    umpire_ks = (context.get("umpire") or {}).get("strikeouts_per_game")
    umpire_runs = (context.get("umpire") or {}).get("runs_per_game")
    env = context.get("environment") or {}

    component_scores = {
        "strikeouts_over": {
            "pitcher_skill": _avg_score(
                _score_band(pitcher.get("k_pct"), 14.0, 32.0),
                _score_band(pitcher.get("k_bb_pct"), 7.0, 26.0),
                _score_band(pitcher.get("swstr_pct"), 8.0, 18.0),
                _score_band(pitcher.get("csw_pct"), 26.0, 35.0),
                _score_band(pitcher.get("bb_pct"), 11.0, 4.0, higher_is_better=False),
            ),
            "pitch_quality": _avg_score(
                _score_band(primary_pitch.get("velocity"), 88.0, 99.0),
                _score_band(primary_pitch.get("spin_rate"), 1900.0, 2900.0),
                _score_band(primary_pitch.get("extension"), 5.5, 7.2),
                _score_band(primary_pitch.get("usage_pct"), 18.0, 45.0),
            ),
            "opponent_profile": _avg_score(
                _score_band(lineup.get("avg_hitter_k_pct"), 18.0, 28.0),
                _score_band(opponent_k_pg, 6.8, 10.5),
                _score_band(umpire_ks, 15.5, 19.0),
                confirmed_boost,
            ),
            "pitch_matchup": _avg_score(
                _score_band(lineup.get("avg_k_vs_primary_pitch"), 18.0, 35.0),
                _score_band(lineup.get("avg_vs_primary_pitch"), 0.310, 0.210, higher_is_better=False),
            ),
            "context": _avg_score(
                _context_score_from_environment("strikeouts", env.get("altitude_ft"), env.get("wind_mph"), env.get("wind_direction", ""), env.get("temperature_f")),
                _score_band(pitcher.get("xera"), 5.3, 2.8, higher_is_better=False),
                _score_band(pitcher.get("hard_hit_pct"), 47.0, 30.0, higher_is_better=False),
            ),
        },
        "walks_under": {
            "pitcher_skill": _avg_score(
                _score_band(pitcher.get("bb_pct"), 11.0, 3.0, higher_is_better=False),
                _score_band(pitcher.get("bb_per_9"), 4.4, 1.5, higher_is_better=False),
                _score_band(pitcher.get("fstrike_pct"), 54.0, 67.0),
                _score_band(pitcher.get("csw_pct"), 25.0, 35.0),
            ),
            "pitch_quality": _avg_score(
                _score_band(primary_pitch.get("extension"), 5.5, 7.1),
                _score_band(primary_pitch.get("velocity"), 88.0, 99.0),
            ),
            "opponent_profile": _avg_score(
                _score_band(lineup.get("avg_hitter_bb_pct"), 10.5, 5.5, higher_is_better=False),
                _score_band(opponent_bb_pg, 4.5, 2.4, higher_is_better=False),
                _score_band(opponent.get("ops"), 0.830, 0.650, higher_is_better=False),
                confirmed_boost,
            ),
            "pitch_matchup": _avg_score(
                _score_band(lineup.get("avg_vs_primary_pitch"), 0.310, 0.215, higher_is_better=False),
                _score_band(lineup.get("avg_k_vs_primary_pitch"), 16.0, 33.0),
            ),
            "context": _avg_score(
                _score_band(pitcher.get("whip"), 1.45, 0.95, higher_is_better=False),
                _context_score_from_environment("strikeouts", env.get("altitude_ft"), env.get("wind_mph"), env.get("wind_direction", ""), env.get("temperature_f")),
                confirmed_boost,
            ),
        },
        "earned_runs_under": {
            "pitcher_skill": _avg_score(
                _score_band(pitcher.get("xera"), 5.5, 2.8, higher_is_better=False),
                _score_band(pitcher.get("siera"), 5.3, 2.9, higher_is_better=False),
                _score_band(pitcher.get("era"), 5.5, 2.7, higher_is_better=False),
                _score_band(pitcher.get("hard_hit_pct"), 47.0, 30.0, higher_is_better=False),
                _score_band(pitcher.get("ev"), 91.5, 85.0, higher_is_better=False),
            ),
            "pitch_quality": _avg_score(
                _score_band(primary_pitch.get("spin_rate"), 1900.0, 2900.0),
                _score_band(primary_pitch.get("velocity"), 88.0, 99.0),
            ),
            "opponent_profile": _avg_score(
                _score_band(opponent_runs_pg, 5.6, 3.5, higher_is_better=False),
                _score_band(opponent_ops, 0.820, 0.650, higher_is_better=False),
                _score_band(lineup.get("avg_obp"), 0.355, 0.295, higher_is_better=False),
                confirmed_boost,
            ),
            "pitch_matchup": _avg_score(
                _score_band(lineup.get("avg_vs_primary_pitch"), 0.310, 0.210, higher_is_better=False),
                _score_band(lineup.get("avg_k_vs_primary_pitch"), 16.0, 33.0),
            ),
            "context": _avg_score(
                _context_score_from_environment("runs", env.get("altitude_ft"), env.get("wind_mph"), env.get("wind_direction", ""), env.get("temperature_f")),
                _score_band(umpire_runs, 10.2, 8.0, higher_is_better=False),
                confirmed_boost,
            ),
        },
        "hits_allowed_under": {
            "pitcher_skill": _avg_score(
                _score_band(pitcher.get("whip"), 1.45, 0.95, higher_is_better=False),
                _score_band(pitcher.get("hard_hit_pct"), 47.0, 30.0, higher_is_better=False),
                _score_band(pitcher.get("ev"), 91.5, 85.0, higher_is_better=False),
                _score_band(pitcher.get("xera"), 5.4, 2.8, higher_is_better=False),
            ),
            "pitch_quality": _avg_score(
                _score_band(primary_pitch.get("spin_rate"), 1900.0, 2900.0),
                _score_band(primary_pitch.get("extension"), 5.5, 7.2),
                _score_band(primary_pitch.get("velocity"), 88.0, 99.0),
            ),
            "opponent_profile": _avg_score(
                _score_band(opponent_ops, 0.820, 0.650, higher_is_better=False),
                _score_band(opponent_runs_pg, 5.6, 3.5, higher_is_better=False),
                _score_band(lineup.get("avg_obp"), 0.355, 0.295, higher_is_better=False),
                confirmed_boost,
            ),
            "pitch_matchup": _avg_score(
                _score_band(lineup.get("avg_vs_primary_pitch"), 0.315, 0.210, higher_is_better=False),
                _score_band(lineup.get("avg_k_vs_primary_pitch"), 16.0, 33.0),
            ),
            "context": _avg_score(
                _context_score_from_environment("hits", env.get("altitude_ft"), env.get("wind_mph"), env.get("wind_direction", ""), env.get("temperature_f")),
                _score_band(umpire_runs, 10.2, 8.0, higher_is_better=False),
                confirmed_boost,
            ),
        },
        "outs_recorded_over": {
            "pitcher_skill": _avg_score(
                _score_band(pitcher.get("whip"), 1.45, 0.95, higher_is_better=False),
                _score_band(pitcher.get("bb_pct"), 11.0, 3.5, higher_is_better=False),
                _score_band(pitcher.get("xera"), 5.5, 2.8, higher_is_better=False),
                _score_band(pitcher.get("csw_pct"), 26.0, 35.0),
            ),
            "pitch_quality": _avg_score(
                _score_band(primary_pitch.get("velocity"), 88.0, 99.0),
                _score_band(primary_pitch.get("extension"), 5.5, 7.2),
                _score_band(primary_pitch.get("usage_pct"), 18.0, 45.0),
            ),
            "opponent_profile": _avg_score(
                _score_band(opponent_runs_pg, 5.6, 3.5, higher_is_better=False),
                _score_band(opponent_ops, 0.820, 0.650, higher_is_better=False),
                _score_band(opponent_bb_pg, 4.5, 2.5, higher_is_better=False),
                confirmed_boost,
            ),
            "pitch_matchup": _avg_score(
                _score_band(lineup.get("avg_vs_primary_pitch"), 0.315, 0.210, higher_is_better=False),
                _score_band(lineup.get("avg_k_vs_primary_pitch"), 16.0, 33.0),
            ),
            "context": _avg_score(
                _context_score_from_environment("runs", env.get("altitude_ft"), env.get("wind_mph"), env.get("wind_direction", ""), env.get("temperature_f")),
                _score_band(umpire_runs, 10.2, 8.0, higher_is_better=False),
                confirmed_boost,
            ),
        },
    }

    candidates: List[Dict[str, Any]] = []
    for bet_type, components in component_scores.items():
        weights = MLB_PITCH_PROP_WEIGHTS.get(bet_type, {})
        score = 0.0
        for component_name, component_score in components.items():
            score += float(weights.get(component_name, 0.0)) * float(component_score)
        candidates.append(
            {
                "bet_type": bet_type,
                "lean": "under" if bet_type.endswith("_under") else "over",
                "score": round(score * 100.0, 1),
                "components": {k: round(v * 100.0, 1) for k, v in components.items()},
            }
        )
    candidates.sort(key=lambda item: item["score"], reverse=True)
    return candidates


def _pitch_prop_confidence_label(score: float) -> str:
    if score >= 74:
        return "A"
    if score >= 66:
        return "B"
    if score >= 58:
        return "C"
    return "D"


def _pitch_prop_reasons(snapshot: Dict[str, Any], candidate: Dict[str, Any]) -> List[str]:
    pitcher = snapshot.get("pitcher") or {}
    lineup = snapshot.get("lineup") or {}
    opponent = snapshot.get("opponent_team") or {}
    primary_pitch = pitcher.get("primary_pitch") or {}
    reasons = []
    skill_bits = []
    if pitcher.get("k_pct") is not None:
        skill_bits.append(f"K% {pitcher.get('k_pct'):.1f}")
    if pitcher.get("bb_pct") is not None:
        skill_bits.append(f"BB% {pitcher.get('bb_pct'):.1f}")
    if pitcher.get("csw_pct") is not None:
        skill_bits.append(f"CSW% {pitcher.get('csw_pct'):.1f}")
    if pitcher.get("swstr_pct") is not None:
        skill_bits.append(f"SwStr% {pitcher.get('swstr_pct'):.1f}")
    if skill_bits:
        reasons.append("Pitcher skill: " + " | ".join(skill_bits) + ".")
    pitch_bits = []
    if primary_pitch.get("description") or primary_pitch.get("code"):
        pitch_bits.append(str(primary_pitch.get("description") or primary_pitch.get("code")))
    if primary_pitch.get("usage_pct") is not None:
        pitch_bits.append(f"{float(primary_pitch.get('usage_pct')):.1f}% usage")
    if primary_pitch.get("velocity") is not None:
        pitch_bits.append(f"{float(primary_pitch.get('velocity')):.1f} mph")
    if primary_pitch.get("spin_rate") is not None:
        pitch_bits.append(f"{float(primary_pitch.get('spin_rate')):.0f} rpm")
    if pitch_bits:
        reasons.append("Primary pitch: " + " | ".join(pitch_bits) + ".")
    matchup_bits = []
    if lineup.get("avg_hitter_k_pct") is not None:
        matchup_bits.append(f"lineup K% {lineup.get('avg_hitter_k_pct'):.1f}")
    if lineup.get("avg_vs_primary_pitch") is not None:
        matchup_bits.append(f"AVG vs primary pitch {lineup.get('avg_vs_primary_pitch'):.3f}")
    if lineup.get("avg_k_vs_primary_pitch") is not None:
        matchup_bits.append(f"K% vs primary pitch {lineup.get('avg_k_vs_primary_pitch'):.1f}")
    if matchup_bits:
        reasons.append("Projected lineup: " + " | ".join(matchup_bits) + ".")
    opponent_bits = []
    if opponent.get("runs_per_game") is not None:
        opponent_bits.append(f"R/G {opponent.get('runs_per_game'):.2f}")
    if opponent.get("ops") is not None:
        opponent_bits.append(f"OPS {opponent.get('ops'):.3f}")
    if opponent.get("strikeouts_per_game") is not None:
        opponent_bits.append(f"K/G {opponent.get('strikeouts_per_game'):.2f}")
    if opponent_bits:
        reasons.append(f"Opponent {snapshot.get('opponent')} team profile: " + " | ".join(opponent_bits) + ".")
    reasons.append(f"{candidate.get('bet_type').replace('_', ' ')} score {candidate.get('score'):.1f} with {lineup.get('status') or 'team-stat fallback'} data.")
    return reasons[:4]


def _empty_pitcher_lineup_summary() -> Dict[str, Any]:
    return {
        "status": "",
        "confirmed": False,
        "player_count": 0,
        "avg_hitter_k_pct": None,
        "avg_hitter_bb_pct": None,
        "avg_obp": None,
        "avg_vs_primary_pitch": None,
        "avg_k_vs_primary_pitch": None,
        "handedness": {"L": 0, "R": 0, "S": 0},
        "players": [],
    }


def _build_mlb_pitcher_daily_snapshot(game: Dict[str, Any], side: str, season_year: int, rotowire_index: Dict[str, Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    opponent_side = "home" if side == "away" else "away"
    team_block = game.get(side) or {}
    opponent_block = game.get(opponent_side) or {}
    probable = team_block.get("probable_pitcher") or {}
    pitcher_name = str(probable.get("fullName") or probable.get("name") or "").strip()
    pitcher_id = probable.get("id")
    if not pitcher_name or not pitcher_id:
        return None
    opponent_abbr = str(opponent_block.get("abbr") or "").upper()
    pitcher_profile = _mlb_player_profile(pitcher_name)
    statsapi_stats, _, _ = _mlb_player_stats_with_fallback(int(pitcher_id), "pitching", season_year)
    fangraphs = _fangraphs_pitcher_snapshot(pitcher_name, str(team_block.get("abbr") or "").upper(), season_year)
    primary_pitch = _mlb_primary_pitch(int(pitcher_id), season_year)
    savant_pitch = _baseballsavant_primary_pitch_snapshot(pitcher_name, str(primary_pitch.get("code") or ""), season_year) if primary_pitch.get("code") else {}
    pitcher_metrics = _metric_bundle_for_pitcher(
        pitcher_name=pitcher_name,
        pitcher_team=str(team_block.get("abbr") or "").upper(),
        pitcher_profile=pitcher_profile,
        statsapi_stats=statsapi_stats,
        fangraphs=fangraphs,
        primary_pitch=primary_pitch,
        savant_pitch=savant_pitch,
    )
    opponent_team_stats, opponent_stats_year, opponent_stats_fallback = _mlb_team_stats_with_fallback(int(opponent_block.get("id") or 0), "hitting", season_year) if opponent_block.get("id") else ({}, season_year, False)
    lineup_info = rotowire_index.get(opponent_abbr, {})
    lineup_summary = _lineup_summary_for_pitcher(lineup_info, primary_pitch, season_year) if lineup_info else _empty_pitcher_lineup_summary()
    game_env = _mlb_game_environment(game.get("game_pk"))
    if lineup_info:
        weather = (lineup_info.get("game") or {}).get("weather") or {}
        if weather.get("temp_f") is not None:
            game_env["temperature_f"] = weather.get("temp_f")
        if weather.get("wind_mph") is not None:
            game_env["wind_mph"] = weather.get("wind_mph")
        if weather.get("wind_direction"):
            game_env["wind_direction"] = str(weather.get("wind_direction")).lower().replace("-", "_")
    opponent_summary = {
        "team": opponent_abbr,
        "runs_per_game": _mlb_stat_float(opponent_team_stats, ("runsPerGame",)),
        "ops": _mlb_stat_float(opponent_team_stats, ("ops", "onBasePlusSlugging")),
        "strikeouts_per_game": _team_stat_per_game(opponent_team_stats, ("strikeOuts", "strikeouts")),
        "walks_per_game": _team_stat_per_game(opponent_team_stats, ("baseOnBalls", "walks")),
        "stats_year": opponent_stats_year,
        "stats_fallback": opponent_stats_fallback,
    }
    return {
        "pitcher_id": pitcher_id,
        "team": str(team_block.get("abbr") or "").upper(),
        "opponent": opponent_abbr,
        "game": game,
        "pitcher": pitcher_metrics,
        "lineup": lineup_summary,
        "opponent_team": opponent_summary,
        "context": {
            "game_pk": game.get("game_pk"),
            "game_date": game.get("game_date"),
            "venue": game.get("venue") or game_env.get("venue"),
            "umpire": ((lineup_info.get("game") or {}).get("umpire") if lineup_info else {}) or {},
            "environment": game_env,
        },
        "sources": {
            "fangraphs": pitcher_metrics.get("source_flags", {}).get("fangraphs", False),
            "baseballsavant": pitcher_metrics.get("source_flags", {}).get("baseballsavant", False),
            "rotowire": bool(lineup_info),
            "statsapi": True,
        },
    }


def _side_model_score(score_map: Dict[str, Any], market_key: str, side: str) -> float:
    market_meta = MLB_PITCH_ODDS_MARKETS.get(market_key) or {}
    preferred = str(market_meta.get("preferred_bet_type") or "")
    preferred_score = _safe_float(score_map.get(preferred), 50.0)
    preferred_side = "under" if preferred.endswith("_under") else "over"
    if str(side).lower() == preferred_side:
        return round(_clamp(preferred_score, 1.0, 99.0), 2)
    return round(_clamp(100.0 - preferred_score, 1.0, 99.0), 2)


def _project_pitcher_market_means(snapshot: Dict[str, Any], prop_candidates: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Dict[str, Any]]:
    pitcher = snapshot.get("pitcher") or {}
    lineup = snapshot.get("lineup") or {}
    opponent = snapshot.get("opponent_team") or {}
    context = snapshot.get("context") or {}
    primary_pitch = pitcher.get("primary_pitch") or {}
    env = context.get("environment") or {}
    umpire = context.get("umpire") or {}
    confirmed_factor = 1.0 if lineup.get("confirmed") else 0.62

    ip_per_start = _safe_float(pitcher.get("ip_per_start"), 5.5)
    base_outs = _clamp(ip_per_start * 3.0, 12.0, 21.0)
    outs_score = _avg_score(
        _score_band(pitcher.get("ip_per_start"), 4.8, 6.4),
        _score_band(pitcher.get("whip"), 1.45, 0.95, higher_is_better=False),
        _score_band(pitcher.get("xera"), 5.4, 2.8, higher_is_better=False),
        _score_band(opponent.get("ops"), 0.820, 0.650, higher_is_better=False),
        _score_band(opponent.get("runs_per_game"), 5.6, 3.5, higher_is_better=False),
        confirmed_factor,
        _context_score_from_environment("runs", env.get("altitude_ft"), env.get("wind_mph"), env.get("wind_direction", ""), env.get("temperature_f")),
    )
    projected_outs = _clamp(base_outs + (6.0 * (outs_score - 0.5)), 10.5, 21.5)
    projected_ip = projected_outs / 3.0

    k_per_9 = _safe_float(pitcher.get("k_per_9"), None)
    if k_per_9 is None and pitcher.get("k_pct") is not None:
        k_per_9 = float(pitcher.get("k_pct")) * 0.38
    bb_per_9 = _safe_float(pitcher.get("bb_per_9"), None)
    if bb_per_9 is None and pitcher.get("bb_pct") is not None:
        bb_per_9 = float(pitcher.get("bb_pct")) * 0.38
    hits_per_9 = _safe_float(pitcher.get("hits_per_9"), None)
    if hits_per_9 is None and pitcher.get("whip") is not None:
        whip = float(pitcher.get("whip"))
        hits_per_9 = max(5.5, (whip * 9.0) - float(bb_per_9 or 0.0))

    strikeouts_base = max(1.0, projected_ip * float(k_per_9 or 8.2) / 9.0)
    strikeouts_factor = 1.0
    strikeouts_factor += 0.42 * (_score_band(lineup.get("avg_hitter_k_pct"), 18.0, 28.0) - 0.5)
    strikeouts_factor += 0.28 * (_score_band(lineup.get("avg_k_vs_primary_pitch"), 18.0, 35.0) - 0.5)
    strikeouts_factor += 0.24 * (_score_band(pitcher.get("swstr_pct"), 8.0, 18.0) - 0.5)
    strikeouts_factor += 0.16 * (_score_band(umpire.get("strikeouts_per_game"), 15.5, 19.0) - 0.5)
    strikeouts_factor += 0.10 * (_score_band(primary_pitch.get("velocity"), 88.0, 99.0) - 0.5)
    strikeouts_factor += 0.10 * (_context_score_from_environment("strikeouts", env.get("altitude_ft"), env.get("wind_mph"), env.get("wind_direction", ""), env.get("temperature_f")) - 0.5)
    projected_strikeouts = _clamp(strikeouts_base * max(0.55, strikeouts_factor), 0.8, 12.5)

    walks_base = max(0.2, projected_ip * float(bb_per_9 or 2.8) / 9.0)
    walks_factor = 1.0
    walks_factor += 0.34 * (_score_band(lineup.get("avg_hitter_bb_pct"), 5.5, 10.5) - 0.5)
    walks_factor += 0.22 * (_score_band(opponent.get("walks_per_game"), 2.4, 4.5) - 0.5)
    walks_factor += 0.28 * (_score_band(pitcher.get("bb_pct"), 3.0, 11.0) - 0.5)
    walks_factor += 0.10 * ((_score_band(primary_pitch.get("extension"), 5.5, 7.1) - 0.5) * -1.0)
    projected_walks = _clamp(walks_base * max(0.55, walks_factor), 0.2, 5.5)

    earned_runs_base = max(0.3, projected_ip * float(pitcher.get("xera") or pitcher.get("era") or 4.1) / 9.0)
    earned_runs_factor = 1.0
    earned_runs_factor += 0.34 * (_score_band(opponent.get("ops"), 0.650, 0.820) - 0.5)
    earned_runs_factor += 0.24 * (_score_band(lineup.get("avg_obp"), 0.295, 0.355) - 0.5)
    earned_runs_factor += 0.22 * (_score_band(lineup.get("avg_vs_primary_pitch"), 0.210, 0.315) - 0.5)
    earned_runs_factor += 0.18 * ((_context_score_from_environment("runs", env.get("altitude_ft"), env.get("wind_mph"), env.get("wind_direction", ""), env.get("temperature_f")) * -1.0) + 0.5)
    earned_runs_factor += 0.14 * (_score_band(pitcher.get("hard_hit_pct"), 30.0, 47.0) - 0.5)
    earned_runs_factor -= 0.12 * (_score_band(pitcher.get("k_pct"), 14.0, 32.0) - 0.5)
    projected_earned_runs = _clamp(earned_runs_base * max(0.55, earned_runs_factor), 0.2, 6.5)

    hits_base = max(2.0, projected_ip * float(hits_per_9 or 7.6) / 9.0)
    if pitcher.get("whip") is not None:
        hits_base = max(hits_base, (float(pitcher.get("whip")) * projected_ip) - projected_walks)
    hits_factor = 1.0
    hits_factor += 0.30 * (_score_band(opponent.get("ops"), 0.650, 0.820) - 0.5)
    hits_factor += 0.24 * (_score_band(lineup.get("avg_obp"), 0.295, 0.355) - 0.5)
    hits_factor += 0.22 * (_score_band(lineup.get("avg_vs_primary_pitch"), 0.210, 0.315) - 0.5)
    hits_factor += 0.18 * ((_context_score_from_environment("hits", env.get("altitude_ft"), env.get("wind_mph"), env.get("wind_direction", ""), env.get("temperature_f")) * -1.0) + 0.5)
    hits_factor += 0.16 * (_score_band(pitcher.get("hard_hit_pct"), 30.0, 47.0) - 0.5)
    projected_hits = _clamp(hits_base * max(0.55, hits_factor), 1.5, 12.5)

    if prop_candidates is None:
        prop_candidates = _score_pitcher_prop_candidates(snapshot)
    score_map = {item["bet_type"]: item["score"] for item in prop_candidates}

    return {
        "pitcher_strikeouts": {
            "expected_stat": round(projected_strikeouts, 2),
            "std_dev": MLB_PITCH_ODDS_MARKETS["pitcher_strikeouts"]["std_dev"],
            "model_score": _side_model_score(score_map, "pitcher_strikeouts", "over"),
        },
        "pitcher_walks": {
            "expected_stat": round(projected_walks, 2),
            "std_dev": MLB_PITCH_ODDS_MARKETS["pitcher_walks"]["std_dev"],
            "model_score": _side_model_score(score_map, "pitcher_walks", "under"),
        },
        "pitcher_earned_runs": {
            "expected_stat": round(projected_earned_runs, 2),
            "std_dev": MLB_PITCH_ODDS_MARKETS["pitcher_earned_runs"]["std_dev"],
            "model_score": _side_model_score(score_map, "pitcher_earned_runs", "under"),
        },
        "pitcher_hits_allowed": {
            "expected_stat": round(projected_hits, 2),
            "std_dev": MLB_PITCH_ODDS_MARKETS["pitcher_hits_allowed"]["std_dev"],
            "model_score": _side_model_score(score_map, "pitcher_hits_allowed", "under"),
        },
        "pitcher_outs": {
            "expected_stat": round(projected_outs, 2),
            "std_dev": MLB_PITCH_ODDS_MARKETS["pitcher_outs"]["std_dev"],
            "model_score": _side_model_score(score_map, "pitcher_outs", "over"),
        },
    }


def _pitch_market_probabilities(expected_stat: float, line: float, std_dev: float) -> Tuple[float, float]:
    over_prob = (1.0 - _normal_cdf(float(line), float(expected_stat), float(std_dev))) * 100.0
    over_prob = _clamp(over_prob, 1.0, 99.0)
    under_prob = _clamp(100.0 - over_prob, 1.0, 99.0)
    return round(over_prob, 2), round(under_prob, 2)


def _odds_events_for_sport(sport_key: str) -> List[Dict[str, Any]]:
    cache_key = ("odds_events", sport_key)
    cached = _cached_external(cache_key)
    if cached is not None:
        return cached or []
    url = f"{ODDS_API_BASE_URL}/sports/{sport_key}/events"
    data = _fetch_json(url, params={"apiKey": ODDS_API_KEY, "dateFormat": "iso"})
    rows = data if isinstance(data, list) else []
    _set_cached_external(cache_key, rows)
    return rows


def _odds_event_index(sport_key: str) -> Dict[Tuple[str, str, str, str], List[Dict[str, Any]]]:
    index: Dict[Tuple[str, str, str, str], List[Dict[str, Any]]] = {}
    for event in _odds_events_for_sport(sport_key):
        home_full = _team_name_key(event.get("home_team"))
        away_full = _team_name_key(event.get("away_team"))
        home_short = _team_short_name_key(event.get("home_team"))
        away_short = _team_short_name_key(event.get("away_team"))
        key = (home_full, away_full, home_short, away_short)
        index.setdefault(key, []).append(event)
    return index


def _parse_iso_datetime(text: str) -> Optional[datetime.datetime]:
    if not text:
        return None
    try:
        return datetime.datetime.fromisoformat(str(text).replace("Z", "+00:00"))
    except Exception:
        return None


def _match_odds_event_for_game(game: Dict[str, Any], sport_key: str) -> Optional[Dict[str, Any]]:
    index = _odds_event_index(sport_key)
    home = game.get("home") or {}
    away = game.get("away") or {}
    home_full = _team_name_key(home.get("name"))
    away_full = _team_name_key(away.get("name"))
    home_short = _team_short_name_key(home.get("name"))
    away_short = _team_short_name_key(away.get("name"))
    candidates: List[Dict[str, Any]] = []
    for key, items in index.items():
        if key[0] == home_full and key[1] == away_full:
            candidates.extend(items)
        elif key[2] == home_short and key[3] == away_short:
            candidates.extend(items)
    if not candidates:
        return None
    if len(candidates) == 1:
        return candidates[0]
    game_dt = _parse_iso_datetime(game.get("game_date"))
    if not game_dt:
        return sorted(candidates, key=lambda item: str(item.get("commence_time") or ""))[0]
    candidates = sorted(
        candidates,
        key=lambda item: abs(((_parse_iso_datetime(item.get("commence_time")) or game_dt) - game_dt).total_seconds()),
    )
    return candidates[0]


def _odds_event_market_payload(sport_key: str, event_id: str, bookmaker: str, regions: str, markets: str) -> Dict[str, Any]:
    cache_key = ("odds_event_market_payload", sport_key, event_id, bookmaker, regions, markets)
    cached = _cached_external(cache_key)
    if cached is not None:
        return cached or {}
    url = f"{ODDS_API_BASE_URL}/sports/{sport_key}/events/{event_id}/odds"
    payload = _fetch_json(
        url,
        params={
            "apiKey": ODDS_API_KEY,
            "regions": regions,
            "markets": markets,
            "oddsFormat": "american",
            "bookmakers": bookmaker,
            "dateFormat": "iso",
        },
    )
    _set_cached_external(cache_key, payload)
    return payload if isinstance(payload, dict) else {}


def _parse_pitcher_market_rows(event_payload: Dict[str, Any], pitcher_name: str = "") -> List[Dict[str, Any]]:
    books = event_payload.get("bookmakers") or []
    rows: List[Dict[str, Any]] = []
    for book in books:
        for market in book.get("markets", []) or []:
            market_key = str(market.get("key") or "")
            if market_key not in MLB_PITCH_ODDS_MARKETS:
                continue
            grouped: Dict[Tuple[str, float], Dict[str, Any]] = {}
            for outcome in market.get("outcomes", []) or []:
                desc = str(outcome.get("description") or "").strip()
                if not desc:
                    continue
                if pitcher_name and not _player_name_matches(desc, pitcher_name):
                    continue
                line = _safe_float(outcome.get("point"), None)
                if line is None:
                    continue
                row = grouped.setdefault(
                    (desc, float(line)),
                    {
                        "player_name": desc,
                        "market_key": market_key,
                        "line": float(line),
                        "bookmaker": book.get("key"),
                        "bookmaker_title": book.get("title"),
                        "last_update": market.get("last_update"),
                        "over_odds": None,
                        "under_odds": None,
                    },
                )
                side = str(outcome.get("name") or "").strip().lower()
                try:
                    price = int(outcome.get("price"))
                except Exception:
                    price = None
                if side == "over":
                    row["over_odds"] = price
                elif side == "under":
                    row["under_odds"] = price
            rows.extend(grouped.values())
    rows.sort(key=lambda item: (item.get("market_key", ""), item.get("line", 0.0)))
    return rows


def build_mlb_pitcher_market_edges(
    game_date: datetime.date,
    season_year: int,
    bookmaker: str = "draftkings",
    regions: str = "us",
    markets: str = MLB_DEFAULT_PITCH_MARKETS,
    min_edge_pct: float = 2.0,
    limit: int = 20,
) -> Dict[str, Any]:
    sport_key = _odds_sport_key("mlb")
    market_keys = [m.strip() for m in str(markets or MLB_DEFAULT_PITCH_MARKETS).split(",") if m.strip() in MLB_PITCH_ODDS_MARKETS]
    if not market_keys:
        raise HTTPException(status_code=400, detail="No supported MLB pitcher markets requested")
    if not ODDS_API_KEY:
        return {
            "sport": "mlb",
            "date": game_date.isoformat(),
            "season_year": season_year,
            "bookmaker": bookmaker,
            "regions": regions,
            "markets": market_keys,
            "model_version": MODEL_VERSION,
            "pitchers_scanned": 0,
            "market_rows_scanned": 0,
            "count": 0,
            "recommendations": [],
            "odds_enabled": False,
            "message": "No ODDS_API_KEY configured, so market-edge pricing is disabled. Use the Model Board for no-key pitcher rankings.",
            "source_timestamp": _now_iso(),
            "data_sources": {
                "odds_api": "disabled because ODDS_API_KEY is not configured",
            },
        }

    rotowire_index = _rotowire_lineup_index(game_date)
    schedule_games = _mlb_schedule_games_for_date(game_date)
    recommendations: List[Dict[str, Any]] = []
    pitchers_scanned = 0
    market_rows_scanned = 0

    for game in schedule_games:
        event = _match_odds_event_for_game(game, sport_key)
        if not event:
            continue
        event_payload = _odds_event_market_payload(sport_key, str(event.get("id") or ""), bookmaker, regions, ",".join(market_keys))
        for side in ("away", "home"):
            snapshot = _build_mlb_pitcher_daily_snapshot(game, side, season_year, rotowire_index)
            if not snapshot:
                continue
            pitchers_scanned += 1
            prop_candidates = _score_pitcher_prop_candidates(snapshot)
            score_map = {item["bet_type"]: item["score"] for item in prop_candidates}
            candidate_map = {item["bet_type"]: item for item in prop_candidates}
            projections = _project_pitcher_market_means(snapshot, prop_candidates=prop_candidates)
            market_rows = _parse_pitcher_market_rows(event_payload, pitcher_name=(snapshot.get("pitcher") or {}).get("name", ""))
            for row in market_rows:
                market_rows_scanned += 1
                market_key = str(row.get("market_key") or "")
                projection = projections.get(market_key) or {}
                if not projection:
                    continue
                line = _safe_float(row.get("line"), None)
                expected_stat = _safe_float(projection.get("expected_stat"), None)
                std_dev = _safe_float(projection.get("std_dev"), None)
                if line is None or expected_stat is None or std_dev is None:
                    continue
                over_prob, under_prob = _pitch_market_probabilities(expected_stat, line, std_dev)
                over_implied = implied_probability_from_american(row.get("over_odds"))
                under_implied = implied_probability_from_american(row.get("under_odds"))
                over_edge = round(over_prob - float(over_implied or 0.0), 2) if over_implied is not None else None
                under_edge = round(under_prob - float(under_implied or 0.0), 2) if under_implied is not None else None
                selected_side = ""
                selected_odds = None
                projected_probability = None
                implied_probability = None
                edge_pct = None
                if over_edge is not None and (under_edge is None or over_edge >= under_edge):
                    selected_side = "over"
                    selected_odds = row.get("over_odds")
                    projected_probability = over_prob
                    implied_probability = over_implied
                    edge_pct = over_edge
                elif under_edge is not None:
                    selected_side = "under"
                    selected_odds = row.get("under_odds")
                    projected_probability = under_prob
                    implied_probability = under_implied
                    edge_pct = under_edge
                if not selected_side or edge_pct is None or edge_pct < float(min_edge_pct):
                    continue
                model_score = _side_model_score(score_map, market_key, selected_side)
                preferred_bet_type = str((MLB_PITCH_ODDS_MARKETS.get(market_key) or {}).get("preferred_bet_type") or "")
                confidence = calibrate_confidence(model_score, projected_probability)
                recommendations.append(
                    {
                        "pitcher": (snapshot.get("pitcher") or {}).get("name"),
                        "pitcher_id": snapshot.get("pitcher_id"),
                        "team": snapshot.get("team"),
                        "opponent": snapshot.get("opponent"),
                        "event_id": event.get("id"),
                        "commence_time": event.get("commence_time"),
                        "market_key": market_key,
                        "market_name": MLB_PITCH_ODDS_MARKETS[market_key]["display_name"],
                        "line": line,
                        "recommended_side": selected_side,
                        "offered_odds": selected_odds,
                        "projected_probability": projected_probability,
                        "implied_probability": implied_probability,
                        "edge_pct": edge_pct,
                        "expected_stat": expected_stat,
                        "line_delta": round(expected_stat - float(line), 2),
                        "model_score": round(model_score, 2),
                        "components": (candidate_map.get(preferred_bet_type) or {}).get("components", {}),
                        "confidence": confidence,
                        "fair_odds_american": _american_from_probability(projected_probability),
                        "ev_units": _ev_units_for_american(projected_probability, selected_odds),
                        "over_odds": row.get("over_odds"),
                        "under_odds": row.get("under_odds"),
                        "over_probability": over_prob,
                        "under_probability": under_prob,
                        "bookmaker": row.get("bookmaker"),
                        "bookmaker_title": row.get("bookmaker_title"),
                        "market_last_update": row.get("last_update"),
                        "primary_pitch": (snapshot.get("pitcher") or {}).get("primary_pitch"),
                        "lineup_status": (snapshot.get("lineup") or {}).get("status") or "team-stat fallback",
                        "lineup_summary": {
                            "avg_hitter_k_pct": (snapshot.get("lineup") or {}).get("avg_hitter_k_pct"),
                            "avg_hitter_bb_pct": (snapshot.get("lineup") or {}).get("avg_hitter_bb_pct"),
                            "avg_obp": (snapshot.get("lineup") or {}).get("avg_obp"),
                            "avg_vs_primary_pitch": (snapshot.get("lineup") or {}).get("avg_vs_primary_pitch"),
                            "avg_k_vs_primary_pitch": (snapshot.get("lineup") or {}).get("avg_k_vs_primary_pitch"),
                            "handedness": (snapshot.get("lineup") or {}).get("handedness"),
                            "players": (snapshot.get("lineup") or {}).get("players", [])[:9],
                        },
                        "pitcher_metrics": {
                            "k_pct": (snapshot.get("pitcher") or {}).get("k_pct"),
                            "bb_pct": (snapshot.get("pitcher") or {}).get("bb_pct"),
                            "k_bb_pct": (snapshot.get("pitcher") or {}).get("k_bb_pct"),
                            "swstr_pct": (snapshot.get("pitcher") or {}).get("swstr_pct"),
                            "csw_pct": (snapshot.get("pitcher") or {}).get("csw_pct"),
                            "whip": (snapshot.get("pitcher") or {}).get("whip"),
                            "xera": (snapshot.get("pitcher") or {}).get("xera"),
                            "siera": (snapshot.get("pitcher") or {}).get("siera"),
                            "ip_per_start": (snapshot.get("pitcher") or {}).get("ip_per_start"),
                            "hits_per_9": (snapshot.get("pitcher") or {}).get("hits_per_9"),
                        },
                        "all_prop_scores": score_map,
                        "reasons": _pitch_prop_reasons(snapshot, {"bet_type": f"{market_key}_{selected_side}", "score": model_score}),
                        "sources": snapshot.get("sources"),
                    }
                )

    recommendations.sort(key=lambda item: (float(item.get("edge_pct") or 0.0), float(item.get("confidence") or 0.0)), reverse=True)
    return {
        "sport": "mlb",
        "date": game_date.isoformat(),
        "season_year": season_year,
        "bookmaker": bookmaker,
        "regions": regions,
        "markets": market_keys,
        "model_version": MODEL_VERSION,
        "pitchers_scanned": pitchers_scanned,
        "market_rows_scanned": market_rows_scanned,
        "count": len(recommendations[: max(1, int(limit))]),
        "recommendations": recommendations[: max(1, int(limit))],
        "source_timestamp": _now_iso(),
        "data_sources": {
            "odds_api": "event odds endpoint with official MLB pitcher player-prop markets",
            "fangraphs": "legacy leaderboards types 1, 5, and 24",
            "baseballsavant": "statcast_search grouped pitch traits by primary pitch",
            "rotowire": "daily-lineups page for projected/confirmed lineups, weather, and umpire context",
            "statsapi": "schedule, probable pitchers, pitch arsenal, team stats, and hitter pitch logs",
        },
    }


def build_mlb_daily_pitching_bets(game_date: datetime.date, season_year: int, min_score: float = 55.0, limit: int = 10) -> Dict[str, Any]:
    rotowire_index = _rotowire_lineup_index(game_date)
    games = _mlb_schedule_games_for_date(game_date)
    all_recommendations: List[Dict[str, Any]] = []
    analyzed_pitchers = 0

    for game in games:
        for side in ("away", "home"):
            snapshot = _build_mlb_pitcher_daily_snapshot(game, side, season_year, rotowire_index)
            if not snapshot:
                continue
            analyzed_pitchers += 1
            prop_candidates = _score_pitcher_prop_candidates(snapshot)
            if not prop_candidates:
                continue
            best = prop_candidates[0]
            pitcher_metrics = snapshot.get("pitcher") or {}
            lineup_summary = snapshot.get("lineup") or {}
            projections = _project_pitcher_market_means(snapshot, prop_candidates=prop_candidates)
            model_market_key = next(
                (
                    market_key
                    for market_key, market_meta in MLB_PITCH_ODDS_MARKETS.items()
                    if market_meta.get("preferred_bet_type") == best.get("bet_type")
                ),
                "",
            )
            model_projection = projections.get(model_market_key, {}) if model_market_key else {}
            all_recommendations.append(
                {
                    "pitcher": pitcher_metrics.get("name"),
                    "pitcher_id": snapshot.get("pitcher_id"),
                    "team": snapshot.get("team"),
                    "opponent": snapshot.get("opponent"),
                    "bet_type": best.get("bet_type"),
                    "lean": best.get("lean"),
                    "score": best.get("score"),
                    "confidence": _pitch_prop_confidence_label(float(best.get("score") or 0.0)),
                    "market_key": model_market_key,
                    "market_name": (MLB_PITCH_ODDS_MARKETS.get(model_market_key) or {}).get("display_name") or str(best.get("bet_type") or "").replace("_", " ").title(),
                    "expected_stat": model_projection.get("expected_stat"),
                    "projection_std_dev": model_projection.get("std_dev"),
                    "components": best.get("components"),
                    "all_prop_scores": {item["bet_type"]: item["score"] for item in prop_candidates},
                    "lineup_status": lineup_summary.get("status") or "team-stat fallback",
                    "primary_pitch": pitcher_metrics.get("primary_pitch"),
                    "pitcher_metrics": {
                        "k_pct": pitcher_metrics.get("k_pct"),
                        "bb_pct": pitcher_metrics.get("bb_pct"),
                        "k_bb_pct": pitcher_metrics.get("k_bb_pct"),
                        "swstr_pct": pitcher_metrics.get("swstr_pct"),
                        "csw_pct": pitcher_metrics.get("csw_pct"),
                        "siera": pitcher_metrics.get("siera"),
                        "xera": pitcher_metrics.get("xera"),
                        "hard_hit_pct": pitcher_metrics.get("hard_hit_pct"),
                        "whip": pitcher_metrics.get("whip"),
                    },
                    "lineup_summary": {
                        "avg_hitter_k_pct": lineup_summary.get("avg_hitter_k_pct"),
                        "avg_hitter_bb_pct": lineup_summary.get("avg_hitter_bb_pct"),
                        "avg_obp": lineup_summary.get("avg_obp"),
                        "avg_vs_primary_pitch": lineup_summary.get("avg_vs_primary_pitch"),
                        "avg_k_vs_primary_pitch": lineup_summary.get("avg_k_vs_primary_pitch"),
                        "handedness": lineup_summary.get("handedness"),
                        "players": lineup_summary.get("players", [])[:9],
                    },
                    "context": snapshot.get("context"),
                    "reasons": _pitch_prop_reasons(snapshot, best),
                    "sources": snapshot.get("sources"),
                }
            )

    all_recommendations.sort(key=lambda item: float(item.get("score") or 0.0), reverse=True)
    qualified_recommendations = [row for row in all_recommendations if float(row.get("score") or 0.0) >= float(min_score)]
    returned_recommendations = qualified_recommendations or all_recommendations
    return {
        "sport": "mlb",
        "date": game_date.isoformat(),
        "season_year": season_year,
        "model_version": MODEL_VERSION,
        "weights": MLB_PITCH_PROP_WEIGHTS,
        "analyzed_pitchers": analyzed_pitchers,
        "min_score": min_score,
        "qualified_count": len(qualified_recommendations),
        "fallback_used": not bool(qualified_recommendations) and bool(all_recommendations),
        "message": "Showing the top available model-ranked pitchers for this slate." if all_recommendations else "",
        "recommendations": returned_recommendations[: max(1, int(limit))],
        "data_sources": {
            "fangraphs": "legacy leaderboards types 1, 5, and 24",
            "baseballsavant": "statcast_search grouped pitch traits by primary pitch",
            "rotowire": "daily-lineups page for projected/confirmed lineups, weather, and umpire context",
            "statsapi": "schedule, probable pitchers, pitch arsenal, team stats, and hitter pitch logs",
        },
        "source_timestamp": _now_iso(),
    }


def _mlb_lineup_factor(prop: str, spot: Optional[int]) -> float:
    if not spot:
        return 0.0
    if prop == "rbis":
        if spot in (3, 4, 5):
            return 0.06
        if spot == 2:
            return 0.02
        if spot in (6, 7):
            return 0.01
        if spot in (8, 9):
            return -0.03
        if spot == 1:
            return -0.02
        return 0.0
    if prop == "runs":
        if spot in (1, 2):
            return 0.05
        if spot in (3, 4, 5):
            return 0.03
        if spot in (8, 9):
            return -0.04
        if spot in (6, 7):
            return -0.01
        return 0.0
    if spot in (1, 2, 3, 4, 5):
        return 0.02
    if spot in (8, 9):
        return -0.02
    return 0.0


def _mlb_pitcher_factor(prop: str, stats: Dict[str, Any]) -> float:
    if not stats:
        return 0.0
    era = _mlb_stat_float(stats, ("era",))
    whip = _mlb_stat_float(stats, ("whip",))
    k9 = _mlb_stat_float(stats, ("strikeoutsPer9Inn", "strikeOutsPer9Inn"))
    hr9 = _mlb_stat_float(stats, ("homeRunsPer9", "homeRunsPer9Inn"))
    if prop == "strikeouts":
        if k9 is None:
            return 0.0
        return _clamp((k9 - 8.5) * 0.015, -0.06, 0.06)
    factor = 0.0
    if era is not None:
        factor += _clamp((era - 4.1) * 0.02, -0.05, 0.05)
    if whip is not None:
        factor += _clamp((whip - 1.28) * 0.15, -0.05, 0.05)
    if prop in ("home_runs", "total_bases") and hr9 is not None:
        factor += _clamp((hr9 - 1.1) * 0.04, -0.04, 0.04)
    return _clamp(factor, -0.08, 0.08)


def _mlb_opponent_team_factor(prop: str, stats: Dict[str, Any]) -> float:
    if not stats:
        return 0.0
    games = _mlb_stat_float(stats, ("gamesPlayed", "games", "g"))
    strikeouts = _mlb_stat_float(stats, ("strikeOuts", "strikeouts"))
    runs_per_game = _mlb_stat_float(stats, ("runsPerGame",))
    ops = _mlb_stat_float(stats, ("ops", "onBasePlusSlugging"))
    k_per_game = None
    if games and strikeouts:
        k_per_game = strikeouts / max(1.0, games)
    if prop == "strikeouts":
        if k_per_game is None:
            return 0.0
        return _clamp((k_per_game - 8.0) * 0.02, -0.06, 0.06)
    factor = 0.0
    if runs_per_game is not None:
        factor += _clamp((runs_per_game - 4.4) * -0.015, -0.05, 0.05)
    if ops is not None:
        factor += _clamp((ops - 0.730) * -0.25, -0.05, 0.05)
    return _clamp(factor, -0.08, 0.08)


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
    pitcher_name: str = "",
    venue: str = "",
    wind_mph: Optional[float] = None,
    wind_direction: str = "",
    temperature_f: Optional[float] = None,
    altitude_ft: Optional[float] = None,
):
    season = current_season()
    season_year = _season_label_to_year(season)
    prop_values: List[float] = []
    h2h_values: List[float] = []
    usage_values: List[float] = []
    dvp = "Average"
    last_games_detail: List[dict] = []
    h2h_games_detail: List[dict] = []
    mlb_context = None
    mlb_context_delta = 0.0

    if sport == "mlb":
        season_year = MLB_SEASON_YEAR
        profile = _mlb_player_profile(player)
        player_id = profile.get("id") if isinstance(profile, dict) else None
        logs = _mlb_game_logs(player, season_year, season_type, player_id=player_id)
        prop_values, h2h_values, usage_values = _collect_from_mlb(logs, prop, opponent)
        log_season_year = season_year
        log_fallback = False
        if not prop_values and season_year > 0:
            prev_year = season_year - 1
            logs = _mlb_game_logs(player, prev_year, season_type, player_id=player_id)
            prop_values, h2h_values, usage_values = _collect_from_mlb(logs, prop, opponent)
            if prop_values:
                log_season_year = prev_year
                log_fallback = True
        try:
            mlb_context = {}
            role = "pitcher" if str(profile.get("position_code")) == "1" or str(profile.get("position_abbrev")).upper() == "P" else "batter"
            mlb_context["role"] = role
            mlb_context["team_id"] = profile.get("team_id")
            mlb_context["team_name"] = profile.get("team_name")
            mlb_context["opponent"] = opponent.upper() if opponent else ""
            mlb_context["log_season_year"] = log_season_year
            mlb_context["log_fallback"] = log_fallback
            opp_id = _mlb_team_id(opponent) if opponent else None
            mlb_context["opponent_id"] = opp_id
            game = _mlb_find_schedule_game(profile.get("team_id"), opp_id, season_year) if opp_id else None
            if game:
                mlb_context["game_pk"] = game.get("gamePk")
                mlb_context["game_date"] = game.get("gameDate")
            pitcher_info = None
            lineup_spot = None
            ahead_obp = None
            team_obp = None
            pitcher_stats = {}
            opponent_team_stats = {}
            pitcher_stats_year = None
            opponent_stats_year = None
            team_stats_year = None
            game_env = _mlb_game_environment(game.get("gamePk")) if game else {}
            selected_venue = str(venue or game_env.get("venue") or "")
            selected_temp = temperature_f if temperature_f is not None else game_env.get("temperature_f")
            selected_wind_mph = wind_mph if wind_mph is not None else game_env.get("wind_mph")
            selected_wind_direction = (wind_direction or game_env.get("wind_direction") or "").strip().lower()
            selected_altitude = altitude_ft if altitude_ft is not None else game_env.get("altitude_ft")
            if selected_altitude is None and selected_venue:
                selected_altitude = MLB_PARK_ALTITUDE_FT.get(selected_venue)
            selected_pitcher_name = pitcher_name.strip()
            if role == "batter" and opp_id and game:
                if selected_pitcher_name:
                    override_pid = _mlb_find_player_id(selected_pitcher_name)
                    pitcher_info = {"id": override_pid, "name": selected_pitcher_name}
                else:
                    pitcher_info = _mlb_probable_pitcher(game, opp_id)
                if pitcher_info and pitcher_info.get("id"):
                    pitcher_stats, pitcher_stats_year, _ = _mlb_player_stats_with_fallback(int(pitcher_info["id"]), "pitching", season_year)
                lineup_ctx = _mlb_lineup_context(game.get("gamePk"), profile.get("team_id"), player_id) if player_id else {}
                lineup_spot = lineup_ctx.get("lineup_spot")
                ahead_ids = lineup_ctx.get("ahead_ids") or []
                if ahead_ids:
                    obps = []
                    for pid in ahead_ids[:3]:
                        stats, _, _ = _mlb_player_stats_with_fallback(int(pid), "hitting", season_year)
                        obp = _mlb_stat_float(stats, ("obp", "onBasePercentage"))
                        if obp is not None:
                            obps.append(obp)
                    if obps:
                        ahead_obp = sum(obps) / len(obps)
                team_stats, team_stats_year, _ = _mlb_team_stats_with_fallback(profile.get("team_id"), "hitting", season_year) if profile.get("team_id") else ({}, None, False)
                team_obp = _mlb_stat_float(team_stats, ("obp", "onBasePercentage"))
            elif role == "batter" and selected_pitcher_name:
                override_pid = _mlb_find_player_id(selected_pitcher_name)
                if override_pid:
                    pitcher_info = {"id": override_pid, "name": selected_pitcher_name}
                    pitcher_stats, pitcher_stats_year, _ = _mlb_player_stats_with_fallback(int(override_pid), "pitching", season_year)
            if role == "pitcher" and opp_id:
                opponent_team_stats, opponent_stats_year, _ = _mlb_team_stats_with_fallback(opp_id, "hitting", season_year)
            pitcher_factor = _mlb_pitcher_factor(prop, pitcher_stats) if role == "batter" else 0.0
            lineup_factor = _mlb_lineup_factor(prop, lineup_spot) if role == "batter" else 0.0
            onbase_factor = 0.0
            obp_ref = ahead_obp if ahead_obp is not None else team_obp
            if role == "batter" and obp_ref is not None:
                if prop == "rbis":
                    onbase_factor = _clamp((obp_ref - 0.320) * 1.2, -0.05, 0.05)
                elif prop == "runs":
                    onbase_factor = _clamp((obp_ref - 0.320) * 0.8, -0.04, 0.04)
            team_factor = _mlb_opponent_team_factor(prop, opponent_team_stats) if role == "pitcher" else 0.0
            primary_pitch = {}
            hitter_pitch_stats = {}
            hitter_pitch_year = None
            hitter_pitch_fallback = False
            pitch_match_factor = 0.0
            if role == "batter" and pitcher_info and pitcher_info.get("id") and player_id:
                primary_pitch = _mlb_primary_pitch(int(pitcher_info["id"]), season_year)
                hitter_pitch_profile, hitter_pitch_year, hitter_pitch_fallback = _mlb_hitter_pitch_profile(int(player_id), season_year)
                if primary_pitch.get("code"):
                    hitter_pitch_stats = hitter_pitch_profile.get(str(primary_pitch.get("code")).upper(), {})
                    pitch_match_factor = _mlb_pitch_match_factor(prop, hitter_pitch_stats)
            env_factor = _mlb_environment_factor(
                prop=prop,
                role=role,
                altitude_ft=_safe_float(selected_altitude, None),
                wind_mph=_safe_float(selected_wind_mph, None),
                wind_direction=selected_wind_direction,
                temperature_f=_safe_float(selected_temp, None),
            )
            mlb_context_delta = _clamp(pitcher_factor + lineup_factor + onbase_factor + team_factor + pitch_match_factor + env_factor, -0.16, 0.16)
            mlb_context.update(
                {
                    "role": role,
                    "lineup_spot": lineup_spot,
                    "ahead_obp": round(ahead_obp, 3) if ahead_obp is not None else None,
                    "team_obp": round(team_obp, 3) if team_obp is not None else None,
                    "pitcher": {
                        "id": pitcher_info.get("id") if pitcher_info else None,
                        "name": pitcher_info.get("name") if pitcher_info else None,
                        "era": _mlb_stat_float(pitcher_stats, ("era",)),
                        "whip": _mlb_stat_float(pitcher_stats, ("whip",)),
                        "k_per_9": _mlb_stat_float(pitcher_stats, ("strikeoutsPer9Inn", "strikeOutsPer9Inn")),
                        "hr_per_9": _mlb_stat_float(pitcher_stats, ("homeRunsPer9", "homeRunsPer9Inn")),
                        "season_year": pitcher_stats_year,
                    },
                    "opponent_team_stats": {
                        "runs_per_game": _mlb_stat_float(opponent_team_stats, ("runsPerGame",)),
                        "ops": _mlb_stat_float(opponent_team_stats, ("ops", "onBasePlusSlugging")),
                        "strikeouts": _mlb_stat_float(opponent_team_stats, ("strikeOuts", "strikeouts")),
                        "games_played": _mlb_stat_float(opponent_team_stats, ("gamesPlayed", "games", "g")),
                        "season_year": opponent_stats_year,
                    },
                    "team_stats_year": team_stats_year,
                    "environment": {
                        "venue": selected_venue,
                        "temperature_f": _safe_float(selected_temp, None),
                        "wind_mph": _safe_float(selected_wind_mph, None),
                        "wind_direction": selected_wind_direction,
                        "altitude_ft": _safe_float(selected_altitude, None),
                    },
                    "pitch_matchup": {
                        "primary_pitch": primary_pitch,
                        "hitter_pitch_stats": hitter_pitch_stats,
                        "hitter_pitch_year": hitter_pitch_year,
                        "hitter_pitch_fallback": hitter_pitch_fallback,
                        "pitch_match_factor": round(pitch_match_factor, 4),
                    },
                    "factors": {
                        "pitcher_factor": round(pitcher_factor, 4),
                        "lineup_factor": round(lineup_factor, 4),
                        "onbase_factor": round(onbase_factor, 4),
                        "team_factor": round(team_factor, 4),
                        "environment_factor": round(env_factor, 4),
                        "pitch_match_factor": round(pitch_match_factor, 4),
                        "context_delta": round(mlb_context_delta, 4),
                    },
                }
            )
        except Exception as exc:
            mlb_context = {"error": f"MLB context unavailable: {type(exc).__name__}"}
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
    conf_base = confidence(l5_rate, l10_rate, h2h_rate, conf_l5_min, conf_l10_min, conf_h2h_good, conf_low_max)
    expected_stat = weighted_expected_stat(avg_l5, avg_l10, avg_h2h, bool(h2h_vals))
    if sport == "mlb" and mlb_context_delta:
        expected_stat = round(expected_stat * (1 + mlb_context_delta), 2)
    rec = line_recommendation(expected_stat, line)
    proj_prob = projected_probability(l5_rate, l10_rate, h2h_rate, bool(h2h_vals))
    if sport == "mlb" and mlb_context_delta:
        proj_prob = round(_clamp(proj_prob + (mlb_context_delta * 20.0), 1.0, 99.0), 2)
    conf = calibrate_confidence(conf_base, proj_prob)
    minutes_proj = round(_mean(usage_values) if usage_values else _mean(last_10_vals), 1)
    if sport == "mlb" and mlb_context and isinstance(mlb_context, dict):
        spot = mlb_context.get("lineup_spot")
        if spot in (1, 2):
            minutes_proj = round(minutes_proj + 0.4, 1)
        elif spot in (8, 9):
            minutes_proj = round(max(0.0, minutes_proj - 0.3), 1)
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
    if sport == "mlb" and isinstance(mlb_context, dict):
        if mlb_context.get("log_fallback") and mlb_context.get("log_season_year"):
            reasons.append(f"No current-season logs yet; using {mlb_context.get('log_season_year')} season logs.")
        pitcher = mlb_context.get("pitcher") or {}
        if mlb_context.get("role") == "batter" and pitcher.get("name"):
            era = pitcher.get("era")
            whip = pitcher.get("whip")
            k9 = pitcher.get("k_per_9")
            pitch_bits = []
            if era is not None:
                pitch_bits.append(f"ERA {era}")
            if whip is not None:
                pitch_bits.append(f"WHIP {whip}")
            if k9 is not None:
                pitch_bits.append(f"K/9 {k9}")
            detail = " | ".join(pitch_bits) if pitch_bits else "stats unavailable"
            reasons.append(f"Opposing pitcher: {pitcher.get('name')} ({detail}).")
        pitch_match = mlb_context.get("pitch_matchup") or {}
        primary_pitch = pitch_match.get("primary_pitch") or {}
        hitter_pitch_stats = pitch_match.get("hitter_pitch_stats") or {}
        if primary_pitch.get("code"):
            pitch_desc = primary_pitch.get("description") or primary_pitch.get("code")
            usage = primary_pitch.get("usage_pct")
            if usage is not None:
                reasons.append(f"Pitch profile: {pitch_desc} is primary at {usage:.1f}% usage.")
            else:
                reasons.append(f"Pitch profile: {pitch_desc} is primary.")
            if hitter_pitch_stats.get("avg") is not None:
                reasons.append(
                    f"Hitter vs {primary_pitch.get('code')}: AVG {float(hitter_pitch_stats.get('avg')):.3f}"
                    f" | SLG {float(hitter_pitch_stats.get('slg', 0.0)):.3f}."
                )
        if mlb_context.get("role") == "pitcher" and mlb_context.get("opponent"):
            reasons.append(f"Opponent lineup: {mlb_context.get('opponent')}.")
        if mlb_context.get("lineup_spot"):
            reasons.append(f"Projected lineup spot: {mlb_context.get('lineup_spot')}.")
        obp_note = mlb_context.get("ahead_obp") if mlb_context.get("ahead_obp") is not None else mlb_context.get("team_obp")
        if obp_note is not None:
            reasons.append(f"On-base context: {obp_note:.3f} OBP ahead.")
        env = mlb_context.get("environment") or {}
        if env.get("venue"):
            reasons.append(
                f"Park/weather: {env.get('venue')} | wind {env.get('wind_mph')} mph {env.get('wind_direction') or 'n/a'}"
                f" | temp {env.get('temperature_f')}F | altitude {env.get('altitude_ft')} ft."
            )
        if mlb_context.get("factors", {}).get("context_delta"):
            delta = float(mlb_context.get("factors", {}).get("context_delta") or 0)
            reasons.append(f"Context adjustment: {delta:+.2%}.")

    result = {
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
    if sport == "mlb":
        result["mlb_context"] = mlb_context
    return result


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


def get_team_stats_df(
    season: str,
    season_type: str,
    team_id: Optional[int] = None,
    per_mode: str = "Totals",
    with_meta: bool = False,
):
    cache_key = ("team_stats", season, season_type, team_id or "all", per_mode)
    now = time.time()
    cached = _team_stats_cache.get(cache_key)
    if cached and (now - cached["ts"] < TEAM_STATS_TTL_SECONDS):
        if with_meta:
            return cached["df"], {"stale": False, "cached_at": cached["ts"]}
        return cached["df"]

    def fetch_stats():
        try:
            endpoint = leaguedashteamstats.LeagueDashTeamStats(
                season=season,
                season_type_all_star=season_type,
                team_id_nullable=str(team_id or ""),
                per_mode_detailed=per_mode,
                timeout=NBA_HTTP_TIMEOUT_SECONDS,
                headers=NBA_HEADERS,
            )
        except TypeError:
            endpoint = leaguedashteamstats.LeagueDashTeamStats(
                season=season,
                season_type_all_star=season_type,
                team_id_nullable=str(team_id or ""),
                per_mode_detailed=per_mode,
            )
        return endpoint.get_data_frames()[0]

    try:
        stats = _nba_with_retries(fetch_stats)
        _team_stats_cache[cache_key] = {"df": stats, "ts": now}
        if with_meta:
            return stats, {"stale": False, "cached_at": now}
        return stats
    except HTTPException:
        if cached:
            if with_meta:
                return cached["df"], {"stale": True, "cached_at": cached["ts"]}
            return cached["df"]
        raise


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
    stats, meta = get_team_stats_df(season, season_type, per_mode="PerGame", with_meta=True)
    rank_available = True
    if stats is None or "TEAM_ID" not in stats.columns:
        stats, meta = get_team_stats_df(season, season_type, team_id=team_id, per_mode="PerGame", with_meta=True)
        rank_available = False
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
        if rank_available:
            rank, pct = _rank_metric(stats, team_id, column, ascending)
        else:
            rank, pct = None, None
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
        "stale": bool(meta.get("stale")) if isinstance(meta, dict) else False,
        "cached_at": meta.get("cached_at") if isinstance(meta, dict) else None,
        "ranked": rank_available,
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


def get_team_shot_locations_df(
    season: str,
    season_type: str,
    measure_type: str,
    team_id: Optional[int] = None,
    per_mode: str = "PerGame",
    with_meta: bool = False,
):
    cache_key = ("shot_locations", season, season_type, measure_type, team_id or "all", per_mode)
    now = time.time()
    cached = _team_shot_cache.get(cache_key)
    if cached and (now - cached["ts"] < TEAM_STATS_TTL_SECONDS):
        if with_meta:
            return cached["df"], {"stale": False, "cached_at": cached["ts"]}
        return cached["df"]

    def fetch_stats():
        try:
            endpoint = leaguedashteamshotlocations.LeagueDashTeamShotLocations(
                season=season,
                season_type_all_star=season_type,
                measure_type_simple=measure_type,
                team_id_nullable=str(team_id or ""),
                per_mode_detailed=per_mode,
                timeout=NBA_HTTP_TIMEOUT_SECONDS,
                headers=NBA_HEADERS,
            )
        except TypeError:
            endpoint = leaguedashteamshotlocations.LeagueDashTeamShotLocations(
                season=season,
                season_type_all_star=season_type,
                measure_type_simple=measure_type,
                team_id_nullable=str(team_id or ""),
                per_mode_detailed=per_mode,
            )
        return endpoint.get_data_frames()[0]

    try:
        stats = _nba_with_retries(fetch_stats)
        _team_shot_cache[cache_key] = {"df": stats, "ts": now}
        if with_meta:
            return stats, {"stale": False, "cached_at": now}
        return stats
    except HTTPException:
        if cached:
            if with_meta:
                return cached["df"], {"stale": True, "cached_at": cached["ts"]}
            return cached["df"]
        raise


def get_team_defensive_shot_zones(season: str, season_type: str, team_abbrev: str) -> Dict[str, Any]:
    team_id = get_team_id(team_abbrev)
    if not team_id:
        raise HTTPException(status_code=404, detail="Team not found")
    stats, meta = get_team_shot_locations_df(season, season_type, "Opponent", with_meta=True)
    rank_available = True
    if stats is None or "TEAM_ID" not in stats.columns:
        stats, meta = get_team_shot_locations_df(season, season_type, "Opponent", team_id=team_id, with_meta=True)
        rank_available = False
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
        if rank_available:
            rank, pct = _rank_metric_by_index(stats, team_id, idx + 2, True)
        else:
            rank, pct = None, None
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
        "stale": bool(meta.get("stale")) if isinstance(meta, dict) else False,
        "cached_at": meta.get("cached_at") if isinstance(meta, dict) else None,
        "ranked": rank_available,
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
    pitcher_name: str = "",
    venue: str = "",
    wind_mph: Optional[float] = Query(None),
    wind_direction: str = "",
    temperature_f: Optional[float] = Query(None),
    altitude_ft: Optional[float] = Query(None),
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
                pitcher_name=pitcher_name,
                venue=venue,
                wind_mph=wind_mph,
                wind_direction=wind_direction,
                temperature_f=temperature_f,
                altitude_ft=altitude_ft,
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
    pitcher_name: str = "",
    venue: str = "",
    wind_mph: Optional[float] = None,
    wind_direction: str = "",
    temperature_f: Optional[float] = None,
    altitude_ft: Optional[float] = None,
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
            pitcher_name=pitcher_name,
            venue=venue,
            wind_mph=wind_mph,
            wind_direction=wind_direction,
            temperature_f=temperature_f,
            altitude_ft=altitude_ft,
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
    pitcher_name: str = "",
    venue: str = "",
    wind_mph: Optional[float] = Query(None),
    wind_direction: str = "",
    temperature_f: Optional[float] = Query(None),
    altitude_ft: Optional[float] = Query(None),
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
        pitcher_name=pitcher_name,
        venue=venue,
        wind_mph=wind_mph,
        wind_direction=wind_direction,
        temperature_f=temperature_f,
        altitude_ft=altitude_ft,
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
            pitcher_name=payload.pitcher_name,
            venue=payload.venue,
            wind_mph=payload.wind_mph,
            wind_direction=payload.wind_direction,
            temperature_f=payload.temperature_f,
            altitude_ft=payload.altitude_ft,
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


@app.get("/mlb/pitching-bets")
def mlb_pitching_bets(
    date: str = Query("", description="Defaults to today in local runtime time."),
    season_year: int = Query(MLB_SEASON_YEAR, ge=2008, le=2100),
    min_score: float = Query(55.0, ge=0.0, le=100.0),
    limit: int = Query(10, ge=1, le=50),
):
    if date:
        try:
            game_date = datetime.date.fromisoformat(date)
        except Exception:
            raise HTTPException(status_code=400, detail="Date must be YYYY-MM-DD")
    else:
        game_date = datetime.date.today()
    return build_mlb_daily_pitching_bets(game_date=game_date, season_year=season_year, min_score=min_score, limit=limit)


@app.get("/mlb/pitching-bets/edges")
def mlb_pitching_bet_edges(
    date: str = Query("", description="Defaults to today in local runtime time."),
    season_year: int = Query(MLB_SEASON_YEAR, ge=2008, le=2100),
    bookmaker: str = Query("draftkings"),
    regions: str = Query("us"),
    markets: str = Query(MLB_DEFAULT_PITCH_MARKETS),
    min_edge_pct: float = Query(2.0, ge=-50.0, le=100.0),
    limit: int = Query(20, ge=1, le=100),
):
    if date:
        try:
            game_date = datetime.date.fromisoformat(date)
        except Exception:
            raise HTTPException(status_code=400, detail="Date must be YYYY-MM-DD")
    else:
        game_date = datetime.date.today()
    return build_mlb_pitcher_market_edges(
        game_date=game_date,
        season_year=season_year,
        bookmaker=bookmaker.strip().lower(),
        regions=regions.strip().lower() or "us",
        markets=markets,
        min_edge_pct=min_edge_pct,
        limit=limit,
    )


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
