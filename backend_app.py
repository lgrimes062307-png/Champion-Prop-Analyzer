from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
import random

app = FastAPI()

# Allow Streamlit to talk to backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/analyze")
def analyze_prop(
    player: str = Query(...),
    prop: str = Query(...),
    line: float = Query(...),
    opponent: str = Query(...)
):
    """
    Temporary deterministic mock logic.
    This WILL NOT crash and always returns full JSON.
    """

    # ---- MOCK ANALYTICS (replace later with real NBA data) ----
    last_5_hit_rate = random.choice([40, 50, 60, 70, 80])
    last_10_hit_rate = random.choice([45, 55, 65, 75])
    h2h_hit_rate = random.choice([30, 40, 50, 60])

    minutes_proj = random.choice([28, 30, 32, 34, 36, 38])

    dvp_map = {
        "ATL": "Weak",
        "BOS": "Strong",
        "MIA": "Below Average",
        "LAL": "Average",
        "DEN": "Strong"
    }
    dvp = dvp_map.get(opponent.upper(), "Average")

    # Confidence calculation (simple but stable)
    confidence = int(
        (last_5_hit_rate * 0.4)
        + (last_10_hit_rate * 0.4)
        + (h2h_hit_rate * 0.2)
    )

    return {
        "confidence": confidence,
        "last_5_hit_rate": last_5_hit_rate,
        "last_10_hit_rate": last_10_hit_rate,
        "h2h_hit_rate": h2h_hit_rate,
        "minutes_proj": minutes_proj,
        "dvp": dvp
    }
