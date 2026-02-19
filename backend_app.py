from fastapi import FastAPI, Query, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from nba_api.stats.endpoints import leaguedashteamstats, playergamelog
from nba_api.stats.static import players, teams
import datetime
import hashlib
import os
import random
import time
import threading
import sqlite3
import requests
from typing import Dict, List, Optional, Tuple

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

SUPPORTED_SPORTS = ("nba", "mlb", "nfl", "soccer", "nhl")
SPORT_ALIASES = {
    "nba": "nba",
    "basketball": "nba",
    "mlb": "mlb",
    "baseball": "mlb",
    "nfl": "nfl",
    "football": "nfl",
    "soccer": "soccer",
    "futbol": "soccer",
    "football-soccer": "soccer",
    "nhl": "nhl",
    "hockey": "nhl",
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
}

# ---------------- HELPERS ---------------- #

_rate_lock = threading.Lock()
_rate_store: Dict[str, list] = {}
_player_log_cache: Dict[tuple, dict] = {}
_team_stats_cache: Dict[tuple, dict] = {}
_external_cache: Dict[tuple, dict] = {}
EXTERNAL_TTL_SECONDS = int(os.getenv("EXTERNAL_TTL_SECONDS", "900"))
EXTERNAL_HTTP_TIMEOUT_SECONDS = float(os.getenv("EXTERNAL_HTTP_TIMEOUT_SECONDS", "8"))
EXTERNAL_HTTP_RETRIES = int(os.getenv("EXTERNAL_HTTP_RETRIES", "2"))
EXTERNAL_RETRY_BACKOFF_SECONDS = float(os.getenv("EXTERNAL_RETRY_BACKOFF_SECONDS", "0.25"))
NFL_SEASON_YEAR = int(os.getenv("NFL_SEASON_YEAR", str(datetime.datetime.now().year)))
SOCCER_SEASON_YEAR = int(os.getenv("SOCCER_SEASON_YEAR", str(datetime.datetime.now().year)))
SOCCER_LEAGUE = os.getenv("SOCCER_LEAGUE", "eng.1")
SOCCER_TEAM = os.getenv("SOCCER_TEAM", "")
MODEL_VERSION = os.getenv("MODEL_VERSION", "2026.02.multi.v1")
ODDS_API_KEY = os.getenv("ODDS_API_KEY", "")
ODDS_API_BASE_URL = os.getenv("ODDS_API_BASE_URL", "https://api.the-odds-api.com/v4")
ALERT_DISCORD_WEBHOOK_URL = os.getenv("ALERT_DISCORD_WEBHOOK_URL", "")
ALERT_MIN_EDGE_PCT = float(os.getenv("ALERT_MIN_EDGE_PCT", "3.5"))
NBA_LIVE_DISABLED = os.getenv("NBA_LIVE_DISABLED", "false").strip().lower() in ("1", "true", "yes", "on")

ESPN_HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; prop-analyzer/1.0)",
    "Accept": "application/json",
}
_provider_state: Dict[str, dict] = {}
_provider_lock = threading.Lock()
_provider_fail_threshold = int(os.getenv("PROVIDER_FAIL_THRESHOLD", "4"))
_provider_cooldown_seconds = int(os.getenv("PROVIDER_COOLDOWN_SECONDS", "120"))


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
    if "the-odds-api.com" in url:
        return "odds_api"
    if "discord.com" in url:
        return "discord"
    return "generic_http"


def _provider_is_open(provider: str):
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
    for p in players.get_players():
        if p["full_name"].lower() == name.lower():
            return p["id"]
    return None


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
    if l5 >= l5_min and l10 >= l10_min:
        return 80
    if h2h >= h2h_good:
        return 90
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
    url = f"https://site.web.api.espn.com/apis/common/v3/sports/{sport_path}/{league_path}/athletes"
    data = _fetch_json(url, params={"limit": 10, "page": 1, "search": player})
    items = data.get("items", []) or data.get("athletes", [])
    athlete_id = None
    needle = player.strip().lower()
    for item in items:
        display = str(item.get("displayName", "")).strip().lower()
        full_name = str(item.get("fullName", "")).strip().lower()
        if display == needle or full_name == needle:
            athlete_id = int(item.get("id"))
            break
    if athlete_id is None and items:
        candidate = items[0].get("id")
        athlete_id = int(candidate) if candidate is not None else None
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
        "nfl": ("football", "nfl"),
        "soccer": ("soccer", SOCCER_LEAGUE if "." in SOCCER_LEAGUE else "eng.1"),
        "nhl": ("hockey", "nhl"),
        "mlb": None,
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
    minutes_proj = round(_mean(usage_values) if usage_values else _mean(last_10_vals), 1)
    projection_label = {
        "mlb": "Plate Appearances",
        "nfl": "Usage Projection",
        "soccer": "Minutes Projection",
        "nhl": "TOI Projection",
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
    base_rate = {"mlb": 51.0, "nfl": 54.0, "soccer": 49.0, "nhl": 50.0}.get(sport, 50.0)
    spread = {"mlb": 18.0, "nfl": 14.0, "soccer": 17.0, "nhl": 16.0}.get(sport, 15.0)
    stat_spread = {"mlb": 1.6, "nfl": 18.0, "soccer": 1.2, "nhl": 1.4}.get(sport, 1.5)

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

    projection_label = {
        "mlb": "Plate Appearances",
        "nfl": "Usage Projection",
        "soccer": "Minutes Projection",
        "nhl": "TOI Projection",
    }.get(sport, "Projection")
    projection_range = {
        "mlb": (3.5, 5.0),
        "nfl": (40.0, 85.0),
        "soccer": (70.0, 95.0),
        "nhl": (13.0, 24.0),
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


def get_team_def_rating(season: str, season_type: str, opponent: str):
    team_id = get_team_id(opponent)
    if not team_id:
        return "Unknown"
    cache_key = (season, season_type)
    now = time.time()
    cached = _team_stats_cache.get(cache_key)
    if cached and (now - cached["ts"] < TEAM_STATS_TTL_SECONDS):
        stats = cached["df"]
    else:
        stats = leaguedashteamstats.LeagueDashTeamStats(
            season=season,
            season_type_all_star=season_type
        ).get_data_frames()[0]
        _team_stats_cache[cache_key] = {"df": stats, "ts": now}
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


def get_player_log(player_id: int, season: str, season_type: str):
    cache_key = (player_id, season, season_type)
    now = time.time()
    cached = _player_log_cache.get(cache_key)
    if cached and (now - cached["ts"] < DATA_TTL_SECONDS):
        return cached["df"]
    df = playergamelog.PlayerGameLog(
        player_id=player_id,
        season=season,
        season_type_all_star=season_type
    ).get_data_frames()[0]
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

    pid = get_player_id(player)
    if not pid:
        return {"error": "Player not found"}

    season = current_season()
    df = get_player_log(pid, season, season_type)
    if df is None or df.empty:
        raise HTTPException(
            status_code=503,
            detail=f"No NBA game logs available for {player} in season {season} ({season_type}).",
        )

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
    minutes_proj = round(float(last_2["MIN"].mean()), 1) if len(last_2) else 0
    dvp = get_team_def_rating(season, season_type, opponent)
    reasons = build_reasons(normalized_prop, line, l5, l10, h2h, avg_l5, avg_l10, avg_h2h, minutes_proj, dvp, opponent)

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


def _odds_sport_key(sport: str):
    return {
        "nba": "basketball_nba",
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
        "providers": providers,
        "cache": {
            "player_log_cache_size": len(_player_log_cache),
            "team_stats_cache_size": len(_team_stats_cache),
            "external_cache_size": len(_external_cache),
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
