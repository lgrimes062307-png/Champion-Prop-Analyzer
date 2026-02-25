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
    conf = app.confidence(55, 60, 50, 50, 50, 60, 40)
    assert conf in {70, 80, 90, 50}


def test_sport_and_prop_normalization():
    assert app.normalize_sport("basketball") == "nba"
    assert app.normalize_sport("hockey") == "nhl"
    assert app.normalize_prop("pts", "nba") == "points"
    assert app.normalize_prop("hr", "mlb") == "home_runs"
    assert app.normalize_prop("rec_yds", "nfl") == "receiving_yards"


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

