from fastapi import FastAPI, Query, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from nba_api.stats.endpoints import leaguedashteamstats, playergamelog
from nba_api.stats.static import players, teams
import datetime
import os
import time
import threading
import sqlite3
import stripe
import requests
from typing import Dict

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

DATA_TTL_SECONDS = int(os.getenv("DATA_TTL_SECONDS", "900"))
TEAM_STATS_TTL_SECONDS = int(os.getenv("TEAM_STATS_TTL_SECONDS", "3600"))
RATE_LIMIT_MAX = int(os.getenv("RATE_LIMIT_MAX", "60"))
RATE_LIMIT_WINDOW_SECONDS = int(os.getenv("RATE_LIMIT_WINDOW_SECONDS", "60"))

DB_PATH = os.getenv("DB_PATH", "app.db")
STRIPE_SECRET_KEY = os.getenv("STRIPE_SECRET_KEY", "")
STRIPE_WEBHOOK_SECRET = os.getenv("STRIPE_WEBHOOK_SECRET", "")
STRIPE_PRICE_ID = os.getenv("STRIPE_PRICE_ID", "")
STRIPE_SUCCESS_URL = os.getenv("STRIPE_SUCCESS_URL", "https://example.com/success")
STRIPE_CANCEL_URL = os.getenv("STRIPE_CANCEL_URL", "https://example.com/cancel")
STRIPE_PORTAL_RETURN_URL = os.getenv("STRIPE_PORTAL_RETURN_URL", "https://example.com/account")
DISCORD_BOT_TOKEN = os.getenv("DISCORD_BOT_TOKEN", "")
DISCORD_GUILD_ID = os.getenv("DISCORD_GUILD_ID", "")
DISCORD_ROLE_ID = os.getenv("DISCORD_ROLE_ID", "")
ADMIN_SECRET = os.getenv("ADMIN_SECRET", "")

stripe.api_key = STRIPE_SECRET_KEY

PROP_ALIASES = {
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
}

# ---------------- HELPERS ---------------- #

_rate_lock = threading.Lock()
_rate_store: Dict[str, list] = {}
_player_log_cache: Dict[tuple, dict] = {}
_team_stats_cache: Dict[tuple, dict] = {}


def init_db():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS subscriptions (
            discord_user_id TEXT PRIMARY KEY,
            stripe_customer_id TEXT,
            status TEXT,
            updated_at TEXT
        )
        """
    )
    conn.commit()
    conn.close()


def upsert_subscription(discord_user_id: str, customer_id: str, status: str):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO subscriptions (discord_user_id, stripe_customer_id, status, updated_at)
        VALUES (?, ?, ?, ?)
        ON CONFLICT(discord_user_id) DO UPDATE SET
            stripe_customer_id=excluded.stripe_customer_id,
            status=excluded.status,
            updated_at=excluded.updated_at
        """,
        (discord_user_id, customer_id, status, datetime.datetime.utcnow().isoformat()),
    )
    conn.commit()
    conn.close()


def update_status_by_customer(customer_id: str, status: str):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute(
        """
        UPDATE subscriptions
        SET status=?, updated_at=?
        WHERE stripe_customer_id=?
        """,
        (status, datetime.datetime.utcnow().isoformat(), customer_id),
    )
    conn.commit()
    conn.close()


def get_subscription_status(discord_user_id: str):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute(
        "SELECT status FROM subscriptions WHERE discord_user_id=?",
        (discord_user_id,),
    )
    row = cur.fetchone()
    conn.close()
    return row[0] if row else "none"


def get_customer_id(discord_user_id: str):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute(
        "SELECT stripe_customer_id FROM subscriptions WHERE discord_user_id=?",
        (discord_user_id,),
    )
    row = cur.fetchone()
    conn.close()
    return row[0] if row else ""


def list_subscriptions():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT discord_user_id, status FROM subscriptions")
    rows = cur.fetchall()
    conn.close()
    return rows


def _rate_limit_ok(ip: str):
    now = time.time()
    with _rate_lock:
        hits = _rate_store.get(ip, [])
        hits = [t for t in hits if now - t < RATE_LIMIT_WINDOW_SECONDS]
        if len(hits) >= RATE_LIMIT_MAX:
            retry_after = max(1, int(RATE_LIMIT_WINDOW_SECONDS - (now - hits[0])))
            return False, retry_after
        hits.append(now)
        _rate_store[ip] = hits
        return True, 0


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


def normalize_prop(prop: str) -> str:
    return PROP_ALIASES.get(prop.strip().lower(), prop.strip().lower())


def stat_value(prop: str, g):
    prop = normalize_prop(prop)
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


def hit_rate_details(df, prop: str, line: float, op: str):
    hits = 0
    n = 0
    for _, g in df.iterrows():
        val = stat_value(prop, g)
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


def avg_stat(df, prop: str):
    total = 0.0
    count = 0
    for _, g in df.iterrows():
        val = stat_value(prop, g)
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


def assign_discord_role(discord_user_id: str, add: bool):
    if not (DISCORD_BOT_TOKEN and DISCORD_GUILD_ID and DISCORD_ROLE_ID):
        return False
    headers = {
        "Authorization": f"Bot {DISCORD_BOT_TOKEN}",
        "Content-Type": "application/json",
    }
    url = f"https://discord.com/api/v10/guilds/{DISCORD_GUILD_ID}/members/{discord_user_id}/roles/{DISCORD_ROLE_ID}"
    if add:
        resp = requests.put(url, headers=headers, json={})
    else:
        resp = requests.delete(url, headers=headers)
    return resp.status_code in (200, 201, 204)


init_db()


@app.get("/evaluate")
def evaluate(
    request: Request,
    player: str = Query(..., min_length=1),
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
):
    client_ip = request.client.host if request.client else "unknown"
    ok, retry_after = _rate_limit_ok(client_ip)
    if not ok:
        raise HTTPException(status_code=429, detail=f"Rate limit exceeded. Try again in {retry_after}s.")

    pid = get_player_id(player)
    if not pid:
        return {"error": "Player not found"}

    season = current_season()
    df = get_player_log(pid, season, season_type)

    last_1 = df.head(window_1)
    last_2 = df.head(window_2)

    op = hit_operator.strip().lower() if hit_operator else HIT_OPERATOR
    l5_hits, l5_n, l5 = hit_rate_details(last_1, prop, line, op)
    l10_hits, l10_n, l10 = hit_rate_details(last_2, prop, line, op)
    avg_l5 = avg_stat(last_1, prop)
    avg_l10 = avg_stat(last_2, prop)

    h2h_df = filter_h2h(df, opponent)
    if opponent:
        if len(h2h_df):
            h2h_hits, h2h_n, h2h = hit_rate_details(h2h_df, prop, line, op)
        else:
            h2h_hits, h2h_n = 0, 0
            h2h = DEFAULT_H2H_WITH_OPP
    else:
        h2h_hits, h2h_n = 0, 0
        h2h = DEFAULT_H2H
    avg_h2h = avg_stat(h2h_df, prop) if opponent else 0

    l5_min = conf_l5_min if conf_l5_min is not None else CONF_L5_MIN
    l10_min = conf_l10_min if conf_l10_min is not None else CONF_L10_MIN
    h2h_good = conf_h2h_good if conf_h2h_good is not None else CONF_H2H_GOOD
    low_max = conf_low_max if conf_low_max is not None else CONF_LOW_MAX

    conf = confidence(l5, l10, h2h, l5_min, l10_min, h2h_good, low_max)
    minutes_proj = round(float(last_2["MIN"].mean()), 1) if len(last_2) else 0
    dvp = get_team_def_rating(season, season_type, opponent)
    reasons = build_reasons(normalize_prop(prop), line, l5, l10, h2h, avg_l5, avg_l10, avg_h2h, minutes_proj, dvp, opponent)

    return {
        "player": player,
        "prop": normalize_prop(prop),
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
        "recommendation": recommendation(conf),
        "minutes_proj": minutes_proj,
        "dvp": dvp,
        "reasons": reasons,
        "samples": {
            "last_5_games": l5_n,
            "last_10_games": l10_n,
            "h2h_games": h2h_n,
        },
    }


@app.get("/analyze")
def analyze(
    request: Request,
    player: str = Query(..., min_length=1),
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
):
    return evaluate(
        request=request,
        player=player,
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
    )


@app.post("/create-checkout-session")
def create_checkout_session(discord_user_id: str = Query(..., min_length=1)):
    if not STRIPE_SECRET_KEY or not STRIPE_PRICE_ID:
        return {"error": "Stripe is not configured."}
    session = stripe.checkout.Session.create(
        mode="subscription",
        line_items=[{"price": STRIPE_PRICE_ID, "quantity": 1}],
        success_url=STRIPE_SUCCESS_URL,
        cancel_url=STRIPE_CANCEL_URL,
        metadata={"discord_user_id": discord_user_id},
    )
    return {"url": session.url}


@app.get("/subscription-status")
def subscription_status(discord_user_id: str = Query(..., min_length=1)):
    status = get_subscription_status(discord_user_id)
    active = status in ("active", "trialing")
    return {"status": status, "active": active}


@app.post("/create-portal-session")
def create_portal_session(discord_user_id: str = Query(..., min_length=1)):
    if not STRIPE_SECRET_KEY:
        return {"error": "Stripe is not configured."}
    customer_id = get_customer_id(discord_user_id)
    if not customer_id:
        return {"error": "No customer found for this user."}
    session = stripe.billing_portal.Session.create(
        customer=customer_id,
        return_url=STRIPE_PORTAL_RETURN_URL,
    )
    return {"url": session.url}


@app.post("/stripe-webhook")
async def stripe_webhook(request: Request):
    if not STRIPE_WEBHOOK_SECRET:
        raise HTTPException(status_code=500, detail="Stripe webhook secret not configured.")
    payload = await request.body()
    sig_header = request.headers.get("stripe-signature", "")
    try:
        event = stripe.Webhook.construct_event(payload, sig_header, STRIPE_WEBHOOK_SECRET)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    event_type = event["type"]
    data_object = event["data"]["object"]

    if event_type == "checkout.session.completed":
        discord_user_id = data_object.get("metadata", {}).get("discord_user_id", "")
        customer_id = data_object.get("customer", "")
        if discord_user_id:
            upsert_subscription(discord_user_id, customer_id, "active")
            assign_discord_role(discord_user_id, True)

    if event_type in ("customer.subscription.updated", "customer.subscription.deleted"):
        customer_id = data_object.get("customer", "")
        status = data_object.get("status", "canceled")
        if customer_id:
            update_status_by_customer(customer_id, status)

    return {"received": True}


@app.post("/sync-roles")
def sync_roles(admin_secret: str = Query(..., min_length=1)):
    if not ADMIN_SECRET or admin_secret != ADMIN_SECRET:
        raise HTTPException(status_code=401, detail="Unauthorized.")
    updated = 0
    rows = list_subscriptions()
    for discord_user_id, status in rows:
        is_active = status in ("active", "trialing")
        ok = assign_discord_role(discord_user_id, is_active)
        if ok:
            updated += 1
    return {"updated": updated, "total": len(rows)}


