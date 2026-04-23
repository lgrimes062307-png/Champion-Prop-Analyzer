import pandas as pd

import backend_app as app


def make_df():
    return pd.DataFrame(
        [
            {"PTS": 20, "REB": 5, "AST": 7, "MATCHUP": "LAL vs BOS", "MIN": 34},
            {"PTS": 28, "REB": 8, "AST": 10, "MATCHUP": "LAL @ BOS", "MIN": 36},
            {"PTS": 15, "REB": 11, "AST": 4, "MATCHUP": "LAL vs NYK", "MIN": 30},
        ]
    )


def test_stat_value_aliases():
    df = make_df()
    row = df.iloc[0]
    assert app.stat_value("points", row) == 20
    assert app.stat_value("rebounds", row) == 5
    assert app.stat_value("assists", row) == 7
    assert app.stat_value("pts+reb", row) == 25
    assert app.stat_value("pts+ast", row) == 27
    assert app.stat_value("reb+ast", row) == 12
    assert app.stat_value("pts+reb+ast", row) == 32


def test_hit_rate_details_and_ci():
    df = make_df()
    hits, n, rate = app.hit_rate_details(df, "points", 18, "gt")
    assert n == 3
    assert hits == 2
    assert rate == round((2 / 3) * 100, 2)
    low, high = app.wilson_interval(hits, n)
    assert 0 <= low <= high <= 100


def test_avg_stat():
    df = make_df()
    avg = app.avg_stat(df, "rebounds")
    assert avg == round((5 + 8 + 11) / 3, 2)


def test_season_label_end_year():
    assert app._season_label_to_end_year("2025-26") == 2026
    assert app._season_label_to_end_year("2024-25") == 2025


def test_nba_prop_game_details():
    df = make_df()
    rows = app.nba_prop_game_details(df, "points", 19.5, "gt", limit=3)
    assert len(rows) == 3
    assert rows[0]["opponent"] == "BOS"
    assert rows[0]["prop_value"] == 20.0
    assert rows[0]["hit"] is True


def test_confidence():
    # 90 only when all three (L5, L10, H2H) pass thresholds.
    assert app.confidence(55, 60, 65, 50, 50, 60, 40) == 90
    assert app.confidence(55, 60, 50, 50, 50, 60, 40) == 80
    assert app.confidence(35, 38, 70, 50, 50, 60, 40) == 50


def test_calibrate_confidence_aligns_to_projected_prob():
    aligned = app.calibrate_confidence(90, 56)
    assert 48 <= aligned <= 64
    aligned2 = app.calibrate_confidence(50, 82)
    assert 74 <= aligned2 <= 90


def test_sport_and_prop_normalization():
    assert app.normalize_sport("basketball") == "nba"
    assert app.normalize_sport("hockey") == "nhl"
    assert app.normalize_sport("counter-strike") == "cs2"
    assert app.normalize_sport("call of duty") == "cod"
    assert app.normalize_prop("pts", "nba") == "points"
    assert app.normalize_prop("hr", "mlb") == "home_runs"
    assert app.normalize_prop("rec_yds", "nfl") == "receiving_yards"
    assert app.normalize_prop("kd_ratio", "cs2") == "kd_ratio"


def test_pandascore_game_slug_mapping():
    assert app._pandascore_game_slug("cs2") == "csgo"
    assert app._pandascore_game_slug("cod") == app.PANDASCORE_COD_GAME
    assert app._pandascore_game_slug("nba") == ""


def test_esports_metric_map_has_core_fields():
    cs2_map = app._sport_metric_map("cs2")
    cod_map = app._sport_metric_map("cod")
    assert "kills" in cs2_map
    assert "kd_ratio" in cs2_map
    assert "objective_kills" in cod_map


def test_multi_sport_fallback_shape():
    res = app.build_multi_sport_fallback(
        sport="nfl",
        player="Patrick Mahomes",
        prop="passing_yards",
        line=275.5,
        opponent="BAL",
        window_1=5,
        window_2=10,
        conf_l5_min=50,
        conf_l10_min=50,
        conf_h2h_good=60,
        conf_low_max=40,
    )
    assert res["sport"] == "nfl"
    assert res["prop"] == "passing_yards"
    assert "recommendation" in res
    assert "confidence" in res
    assert "projection_label" in res


def test_numeric_parsers():
    assert app._extract_first_number("12-8") == 12.0
    assert app._extract_first_number("7.5 attempts") == 7.5
    assert app._numeric("19") == 19.0
    assert app._numeric(4) == 4.0
    assert app._numeric("N/A") is None


def test_collect_metric_series():
    rows = [
        {"passingYards": "310", "opponent": "BAL"},
        {"passing_yards": 280, "opponent": "KC"},
        {"other": 1},
    ]
    vals = app._collect_metric_series(rows, ["passingYards", "passing_yards"])
    assert vals == [310.0, 280.0]


def test_collect_metric_series_does_not_double_count_row():
    rows = [{"passingYards": "310", "passing_yards": 311, "opponent": "BAL"}]
    vals = app._collect_metric_series(rows, ["passingYards", "passing_yards"])
    assert vals == [310.0]


def test_collect_from_espn_payload_keeps_values_aligned_to_opponent():
    payload = {
        "events": [
            {"opponent": "BAL", "passingYards": "300", "gameDate": "2025-09-01"},
            {"opponent": "KC", "passingYards": "250", "gameDate": "2025-09-08"},
        ],
        "seasonTotals": {"passingYards": "550"},
    }
    vals, h2h = app._collect_from_espn_payload(payload, ["passingYards"], "BAL")
    assert vals == [300.0, 250.0]
    assert h2h == [300.0]


def test_implied_probability_from_american():
    assert app.implied_probability_from_american(-110) == 52.38
    assert app.implied_probability_from_american(120) == 45.45


def test_collect_nba_from_espn_payload_points_and_combo():
    payload = {
        "events": [
            {"opponent": "BOS", "points": 30, "rebounds": 8, "assists": 7, "minutes": 36, "gameDate": "2026-01-01"},
            {"opponent": "NYK", "points": 22, "rebounds": 10, "assists": 9, "minutes": 35, "gameDate": "2026-01-03"},
        ]
    }
    vals, h2h, usage, details, h2h_details = app._collect_nba_from_espn_payload(payload, "points", "BOS")
    assert vals == [30.0, 22.0]
    assert h2h == [30.0]
    assert usage == [36.0, 35.0]
    assert details[0]["opponent"] == "BOS"
    assert h2h_details[0]["opponent"] == "BOS"

    vals_combo, _, _, _, _ = app._collect_nba_from_espn_payload(payload, "pts+reb+ast", "")
    assert vals_combo == [45.0, 41.0]


def test_parse_rotowire_daily_lineups_extracts_teams_pitchers_and_status():
    html = """
    <div>12:10 PM ET</div>
    <div>Alerts</div>
    <div>STL</div>
    <div>MIA</div>
    <div>Cardinals (14-10) Marlins (12-13)</div>
    <div>Kyle Leahy</div>
    <div>R</div>
    <div>2-2 5.21 ERA</div>
    <div>Confirmed Lineup</div>
    <div>2B</div><div>J. Wetherholt</div><div>L</div>
    <div>DH</div><div>Ivan Herrera</div><div>R</div>
    <div>Home Run Odds</div>
    <div>Starting Pitcher Intel</div>
    <div>Janson Junk</div>
    <div>R</div>
    <div>0-2 4.50 ERA</div>
    <div>Expected Lineup</div>
    <div>CF</div><div>Jakob Marsee</div><div>L</div>
    <div>DH</div><div>X. Edwards</div><div>S</div>
    <div>Home Run Odds</div>
    <div>Starting Pitcher Intel</div>
    <div>Umpire: Alex Tosi 9.0 R/G 17.5 K/G</div>
    <div>0% 57° Wind 6 mph L-R</div>
    """
    games = app._parse_rotowire_daily_lineups(html)
    assert len(games) == 1
    game = games[0]
    assert game["away"]["team"] == "STL"
    assert game["home"]["team"] == "MIA"
    assert game["away"]["pitcher"]["name"] == "Kyle Leahy"
    assert game["home"]["pitcher"]["name"] == "Janson Junk"
    assert game["away"]["status"] == "Confirmed Lineup"
    assert game["home"]["status"] == "Expected Lineup"
    assert game["umpire"]["strikeouts_per_game"] == 17.5
    assert game["weather"]["wind_mph"] == 6.0
    assert game["away"]["players"][0]["name"] == "J. Wetherholt"
    assert game["home"]["players"][1]["hand"] == "S"


def test_score_pitcher_prop_candidates_gives_high_strikeout_score_when_whiff_profile_is_elite():
    snapshot = {
        "pitcher": {
            "k_pct": 31.2,
            "bb_pct": 4.8,
            "k_bb_pct": 26.4,
            "swstr_pct": 15.9,
            "csw_pct": 33.4,
            "fstrike_pct": 65.1,
            "xera": 3.08,
            "hard_hit_pct": 31.2,
            "whip": 1.01,
            "siera": 3.02,
            "era": 2.94,
            "ev": 85.8,
            "primary_pitch": {
                "code": "FF",
                "description": "Four-Seam Fastball",
                "usage_pct": 37.5,
                "velocity": 97.6,
                "spin_rate": 2475.0,
                "extension": 6.8,
            },
        },
        "lineup": {
            "confirmed": True,
            "status": "Confirmed Lineup",
            "avg_hitter_k_pct": 26.3,
            "avg_hitter_bb_pct": 6.8,
            "avg_obp": 0.303,
            "avg_vs_primary_pitch": 0.214,
            "avg_k_vs_primary_pitch": 31.1,
        },
        "opponent_team": {
            "strikeouts_per_game": 9.7,
            "walks_per_game": 2.8,
            "runs_per_game": 3.8,
            "ops": 0.664,
        },
        "context": {
            "umpire": {"strikeouts_per_game": 18.5, "runs_per_game": 8.1},
            "environment": {"altitude_ft": 250.0, "wind_mph": 8.0, "wind_direction": "in", "temperature_f": 58.0},
        },
        "opponent": "MIA",
    }
    candidates = app._score_pitcher_prop_candidates(snapshot)
    score_map = {item["bet_type"]: item["score"] for item in candidates}
    assert score_map["strikeouts_over"] > 80
    assert score_map["strikeouts_over"] > score_map["walks_under"]
    assert candidates[0]["score"] >= candidates[-1]["score"]


def test_player_name_matches_handles_initials():
    assert app._player_name_matches("P. Skenes", "Paul Skenes") is True
    assert app._player_name_matches("Tarik Skubal", "Tarik Skubal") is True
    assert app._player_name_matches("T. Skubal", "Paul Skenes") is False


def test_parse_pitcher_market_rows_groups_over_under_pairs():
    payload = {
        "bookmakers": [
            {
                "key": "draftkings",
                "title": "DraftKings",
                "markets": [
                    {
                        "key": "pitcher_strikeouts",
                        "last_update": "2026-04-22T15:00:00Z",
                        "outcomes": [
                            {"name": "Over", "description": "Paul Skenes", "price": -115, "point": 7.5},
                            {"name": "Under", "description": "Paul Skenes", "price": -105, "point": 7.5},
                            {"name": "Over", "description": "Tarik Skubal", "price": -120, "point": 6.5},
                            {"name": "Under", "description": "Tarik Skubal", "price": 100, "point": 6.5},
                        ],
                    }
                ],
            }
        ]
    }
    rows = app._parse_pitcher_market_rows(payload, pitcher_name="Paul Skenes")
    assert len(rows) == 1
    row = rows[0]
    assert row["market_key"] == "pitcher_strikeouts"
    assert row["line"] == 7.5
    assert row["over_odds"] == -115
    assert row["under_odds"] == -105


def test_project_pitcher_market_means_returns_main_props():
    snapshot = {
        "pitcher": {
            "name": "Paul Skenes",
            "k_pct": 31.2,
            "bb_pct": 4.8,
            "k_bb_pct": 26.4,
            "swstr_pct": 15.9,
            "csw_pct": 33.4,
            "fstrike_pct": 65.1,
            "xera": 3.08,
            "hard_hit_pct": 31.2,
            "whip": 1.01,
            "siera": 3.02,
            "era": 2.94,
            "ev": 85.8,
            "k_per_9": 11.3,
            "bb_per_9": 2.1,
            "hits_per_9": 6.9,
            "ip_per_start": 6.1,
            "primary_pitch": {
                "code": "FF",
                "description": "Four-Seam Fastball",
                "usage_pct": 37.5,
                "velocity": 97.6,
                "spin_rate": 2475.0,
                "extension": 6.8,
            },
        },
        "lineup": {
            "confirmed": True,
            "status": "Confirmed Lineup",
            "avg_hitter_k_pct": 26.3,
            "avg_hitter_bb_pct": 6.8,
            "avg_obp": 0.303,
            "avg_vs_primary_pitch": 0.214,
            "avg_k_vs_primary_pitch": 31.1,
        },
        "opponent_team": {
            "strikeouts_per_game": 9.7,
            "walks_per_game": 2.8,
            "runs_per_game": 3.8,
            "ops": 0.664,
        },
        "context": {
            "umpire": {"strikeouts_per_game": 18.5, "runs_per_game": 8.1},
            "environment": {"altitude_ft": 250.0, "wind_mph": 8.0, "wind_direction": "in", "temperature_f": 58.0},
        },
    }
    projections = app._project_pitcher_market_means(snapshot)
    assert projections["pitcher_outs"]["expected_stat"] > 16
    assert projections["pitcher_strikeouts"]["expected_stat"] > projections["pitcher_walks"]["expected_stat"]
    over_prob, under_prob = app._pitch_market_probabilities(projections["pitcher_strikeouts"]["expected_stat"], 7.5, projections["pitcher_strikeouts"]["std_dev"])
    assert over_prob > under_prob
