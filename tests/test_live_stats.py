"""In-season box scores: fantasy-week assignment, replacing a pull, and the per-week player totals."""
import sqlite3

import pandas as pd

from src.data_ingestion import live_stats as L

SPANS = [("2026-10-19", "2026-10-25"), ("2026-10-26", "2026-11-01")]          # two fantasy weeks
GAMES = pd.DataFrame({"game_id": ["g1", "g2", "g3", "g4"], "date": ["2026-10-20", "2026-10-25", "2026-10-26", "2026-11-05"],
                      "home_team_id": [1, 1, 2, 1], "away_team_id": [2, 3, 3, 2]})


def box(rows):
    cols = ["game_id", "player_id", "name", "team_id", "min", "fga", "fgm", "tpa", "tpm", "fta", "ftm", "oreb", "dreb", "ast", "stl", "blk", "tov"]
    return pd.DataFrame(rows, columns=cols)


def test_week_assignment_uses_the_matchup_periods_and_leaves_outside_dates_empty():
    assert L.week_of(GAMES["date"], SPANS) == [1, 1, 2, None]
    assert L.week_of(pd.Series(["2026-10-19", "2026-11-01", "2026-10-18"]), SPANS) == [1, 2, None]    # period edges are inclusive


def test_player_weeks_sum_games_into_the_leagues_categories_and_skip_dnp_rows():
    conn = sqlite3.connect(":memory:")
    b = box([("g1", 7, "A", 1, 30.0, 10, 5, 4, 2, 4, 3, 1, 4, 5, 2, 1, 3),
             ("g2", 7, "A", 1, 20.0, 6, 3, 2, 1, 2, 2, 0, 2, 1, 0, 0, 1),
             ("g3", 7, "A", 1, 0.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),             # listed but did not play: no game, no minutes
             ("g3", 8, "B", 2, 25.0, 8, 4, 0, 0, 2, 2, 2, 3, 6, 1, 2, 2)])
    res = L.save_live(conn, GAMES, b, SPANS, "2026-27")
    assert res == {"games": 4, "player_games": 4, "last_date": "2026-11-05"}
    pw = L.player_weeks(conn).set_index(["week", "player_id"])
    a = pw.loc[(1, 7)]
    assert (a["games"], a["min"], a["pts"], a["tpm"], a["reb"], a["ast"], a["tov"]) == (2, 50.0, (2 * 5 + 2 + 3) + (2 * 3 + 1 + 2), 3, (1 + 4) + (0 + 2), 5 + 1, 3 + 1)
    assert a["fgm"] == 8 and a["fga"] == 16 and a["ftm"] == 5 and a["fta"] == 6
    assert (2, 7) not in pw.index and pw.loc[(2, 8), "games"] == 1               # the DNP line does not count as a week-2 game for player 7


def test_a_new_pull_replaces_the_old_one_instead_of_adding_to_it():
    conn = sqlite3.connect(":memory:")
    L.save_live(conn, GAMES, box([("g1", 7, "A", 1, 30.0, 10, 5, 4, 2, 4, 3, 1, 4, 5, 2, 1, 3)]), SPANS, "2026-27")
    L.save_live(conn, GAMES.iloc[:2], box([("g1", 7, "A", 1, 30.0, 10, 5, 4, 2, 4, 3, 1, 4, 5, 2, 1, 3), ("g2", 7, "A", 1, 20.0, 6, 3, 2, 1, 2, 2, 0, 2, 1, 0, 0, 1)]), SPANS, "2026-27")
    assert conn.execute("SELECT COUNT(*) FROM live_game").fetchone()[0] == 2
    assert conn.execute("SELECT COUNT(*) FROM live_box_score").fetchone()[0] == 2
    assert dict(conn.execute("SELECT key, value FROM live_meta"))["last_game_date"] == "2026-10-25"


def test_box_lines_for_unknown_games_are_dropped():
    conn = sqlite3.connect(":memory:")
    res = L.save_live(conn, GAMES.iloc[:1], box([("g1", 7, "A", 1, 30.0, 10, 5, 4, 2, 4, 3, 1, 4, 5, 2, 1, 3), ("zz", 7, "A", 1, 30.0, 10, 5, 4, 2, 4, 3, 1, 4, 5, 2, 1, 3)]), SPANS, "2026-27")
    assert res["player_games"] == 1
