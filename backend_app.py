from fastapi import FastAPI, Query, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import os
import json
import sqlite3
import stripe
import requests
import random
import datetime

app = FastAPI()

# ---- CORS ----
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---- ENV ----
STRIPE_SECRET_KEY = os.getenv("STRIPE_SECRET_KEY", "")
STRIPE_PRICE_ID = os.getenv("STRIPE_PRICE_ID", "")
STRIPE_WEBHOOK_SECRET = os.getenv("STRIPE_WEBHOOK_SECRET", "")

STRIPE_SUCCESS_URL = os.getenv("STRIPE_SUCCESS_URL", "")
STRIPE_CANCEL_URL = os.getenv("STRIPE_CANCEL_URL", "")
STRIPE_PORTAL_RETURN_URL = os.getenv("STRIPE_PORTAL_RETURN_URL", "")

DISCORD_BOT_TOKEN = os.getenv("DISCORD_BOT_TOKEN", "")
DISCORD_GUILD_ID = os.getenv("DISCORD_GUILD_ID", "")
DISCORD_ROLE_ID = os.getenv("DISCORD_ROLE_ID", "")

ADMIN_SECRET = os.getenv("ADMIN_SECRET", "")

if STRIPE_SECRET_KEY:
    stripe.api_key = STRIPE_SECRET_KEY

DB_PATH = "subscriptions.db"


# ---- DB ----
def db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS subscriptions (
            discord_user_id TEXT PRIMARY KEY,
            status TEXT,
            customer_id TEXT,
            updated_at TEXT
        )
        """
    )
    conn.commit()
    return conn


def set_subscription(discord_user_id: str, status: str, customer_id: str = ""):
    conn = db()
    conn.execute(
        """
        INSERT INTO subscriptions (discord_user_id, status, customer_id, updated_at)
        VALUES (?, ?, ?, ?)
        ON CONFLICT(discord_user_id) DO UPDATE SET
          status=excluded.status,
          customer_id=excluded.customer_id,
          updated_at=excluded.updated_at
        """,
        (discord_user_id, status, customer_id, datetime.datetime.utcnow().isoformat()),
    )
    conn.commit()
    conn.close()


def get_subscription(discord_user_id: str):
    conn = db()
    cur = conn.execute(
        "SELECT discord_user_id, status, customer_id, updated_at FROM subscriptions WHERE discord_user_id = ?",
        (discord_user_id,),
    )
    row = cur.fetchone()
    conn.close()
    return row


# ---- DISCORD ROLE ----
def set_discord_role(discord_user_id: str, grant: bool):
    if not (DISCORD_BOT_TOKEN and DISCORD_GUILD_ID and DISCORD_ROLE_ID):
        return False, "Missing Discord env vars"

    headers = {"Authorization": f"Bot {DISCORD_BOT_TOKEN}"}
    url = f"https://discord.com/api/guilds/{DISCORD_GUILD_ID}/members/{discord_user_id}/roles/{DISCORD_ROLE_ID}"

    if grant:
        r = requests.put(url, headers=headers)
    else:
        r = requests.delete(url, headers=headers)

    if r.status_code in [204, 201, 200]:
        return True, "ok"
    return False, f"Discord role update failed ({r.status_code})"


# ---- ANALYZE ----
@app.get("/analyze")
def analyze_prop(
    player: str = Query(...),
    prop: str = Query(...),
    line: float = Query(...),
    opponent: str = Query(...),
    season_type: str = Query("Regular Season"),
    window_1: int = Query(5),
    window_2: int = Query(10),
    hit_operator: str = Query("gt"),
    conf_l5_min: int = Query(50),
    conf_l10_min: int = Query(50),
    conf_h2h_good: int = Query(60),
    conf_low_max: int = Query(40),
):
    last_5_hit_rate = random.choice([40, 50, 60, 70, 80])
    last_10_hit_rate = random.choice([45, 55, 65, 75])
    h2h_hit_rate = random.choice([30, 40, 50, 60])

    last_5_avg_stat = round(line + random.uniform(-4, 4), 1)
    last_10_avg_stat = round(line + random.uniform(-3, 3), 1)
    h2h_avg_stat = round(line + random.uniform(-5, 5), 1)

    last_5_ci = [round(last_5_avg_stat - 4, 1), round(last_5_avg_stat + 4, 1)]
    last_10_ci = [round(last_10_avg_stat - 3, 1), round(last_10_avg_stat + 3, 1)]
    h2h_ci = [round(h2h_avg_stat - 5, 1), round(h2h_avg_stat + 5, 1)]

    minutes_proj = random.choice([28, 30, 32, 34, 36, 38])

    dvp_map = {
        "ATL": "Weak",
        "BOS": "Strong",
        "MIA": "Below Average",
        "LAL": "Average",
        "DEN": "Strong",
    }
    dvp = dvp_map.get(opponent.upper(), "Average")

    confidence = int((last_5_hit_rate * 0.4) + (last_10_hit_rate * 0.4) + (h2h_hit_rate * 0.2))
    recommendation = "Over" if last_5_avg_stat > line else "Under"

    reasons = [
        f"L5 hit rate {last_5_hit_rate}%",
        f"L10 hit rate {last_10_hit_rate}%",
        f"H2H hit rate {h2h_hit_rate}%",
        f"Minutes projection {minutes_proj} min",
    ]

    return {
        "player": player,
        "prop": prop,
        "line": line,
        "recommendation": recommendation,
        "confidence": confidence,
        "last_5_hit_rate": last_5_hit_rate,
        "last_10_hit_rate": last_10_hit_rate,
        "h2h_hit_rate": h2h_hit_rate,
        "last_5_avg_stat": last_5_avg_stat,
        "last_10_avg_stat": last_10_avg_stat,
        "h2h_avg_stat": h2h_avg_stat,
        "last_5_ci": last_5_ci,
        "last_10_ci": last_10_ci,
        "h2h_ci": h2h_ci,
        "minutes_proj": minutes_proj,
        "dvp": dvp,
        "reasons": reasons,
    }


# ---- STRIPE ----
@app.post("/create-checkout-session")
def create_checkout_session(discord_user_id: str = Query(...)):
    if not STRIPE_SECRET_KEY:
        raise HTTPException(status_code=500, detail="Stripe secret key not set.")
    if not STRIPE_PRICE_ID:
        raise HTTPException(status_code=500, detail="Stripe price id not set.")
    if not STRIPE_SUCCESS_URL or not STRIPE_CANCEL_URL:
        raise HTTPException(status_code=500, detail="Stripe success/cancel URL not set.")

    session = stripe.checkout.Session.create(
        mode="subscription",
        line_items=[{"price": STRIPE_PRICE_ID, "quantity": 1}],
        success_url=STRIPE_SUCCESS_URL,
        cancel_url=STRIPE_CANCEL_URL,
        client_reference_id=discord_user_id,
        metadata={"discord_user_id": discord_user_id},
    )
    return {"url": session.url}


@app.post("/create-portal-session")
def create_portal_session(discord_user_id: str = Query(...)):
    if not STRIPE_SECRET_KEY:
        raise HTTPException(status_code=500, detail="Stripe secret key not set.")
    if not STRIPE_PORTAL_RETURN_URL:
        raise HTTPException(status_code=500, detail="Stripe portal return URL not set.")

    sub = get_subscription(discord_user_id)
    if not sub or not sub[2]:
        raise HTTPException(status_code=404, detail="Customer not found for this user.")

    session = stripe.billing_portal.Session.create(
        customer=sub[2],
        return_url=STRIPE_PORTAL_RETURN_URL,
    )
    return {"url": session.url}


@app.get("/subscription-status")
def subscription_status(discord_user_id: str = Query(...)):
    row = get_subscription(discord_user_id)
    if not row:
        return {"status": "none", "active": False}
    status = row[1] or "none"
    return {"status": status, "active": status == "active"}


@app.post("/stripe-webhook")
async def stripe_webhook(request: Request):
    payload = await request.body()
    sig_header = request.headers.get("stripe-signature")

    if not STRIPE_WEBHOOK_SECRET:
        raise HTTPException(status_code=500, detail="Webhook secret not set.")

    try:
        event = stripe.Webhook.construct_event(payload, sig_header, STRIPE_WEBHOOK_SECRET)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid webhook signature")

    # subscription created/updated/canceled
    if event["type"] in ["checkout.session.completed", "customer.subscription.updated", "customer.subscription.deleted"]:
        obj = event["data"]["object"]
        discord_user_id = obj.get("metadata", {}).get("discord_user_id") or obj.get("client_reference_id", "")
        customer_id = obj.get("customer", "")

        if event["type"] == "customer.subscription.deleted":
            set_subscription(discord_user_id, "canceled", customer_id)
            set_discord_role(discord_user_id, False)
        else:
            set_subscription(discord_user_id, "active", customer_id)
            set_discord_role(discord_user_id, True)

    return {"ok": True}


@app.post("/sync-roles")
def sync_roles(discord_user_id: str = Query(...)):
    row = get_subscription(discord_user_id)
    if not row:
        return {"ok": False, "detail": "No subscription found"}
    active = row[1] == "active"
    ok, msg = set_discord_role(discord_user_id, active)
    return {"ok": ok, "detail": msg}


# ---- ADMIN ----
@app.post("/grant-access")
def grant_access(discord_user_id: str = Query(...), admin_secret: str = Query(...)):
    if not ADMIN_SECRET or admin_secret != ADMIN_SECRET:
        raise HTTPException(status_code=401, detail="Invalid admin secret")
    set_subscription(discord_user_id, "active", "manual")
    set_discord_role(discord_user_id, True)
    return {"ok": True}


@app.post("/revoke-access")
def revoke_access(discord_user_id: str = Query(...), admin_secret: str = Query(...)):
    if not ADMIN_SECRET or admin_secret != ADMIN_SECRET:
        raise HTTPException(status_code=401, detail="Invalid admin secret")
    set_subscription(discord_user_id, "canceled", "manual")
    set_discord_role(discord_user_id, False)
    return {"ok": True}


