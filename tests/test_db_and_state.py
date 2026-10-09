import sqlite3
from pathlib import Path

import pandas as pd
import pytest

from src.data_ingestion import load_to_db as db
from src.draft import store
from src.draft.state import DraftState, LeagueSettings, snake_order, team_for_pick
from src.features.stints import compute_player_stints

SCHEMA = Path(__file__).resolve().parents[1] / "schema.sql"
BOX_COLS = "(game_id, player_id, team_id, min, fga, fgm, tpa, tpm, fta, ftm, oreb, dreb, ast, stl, blk, tov)"


@pytest.fixture
def conn():
    c = sqlite3.connect(":memory:")
    c.execute("PRAGMA foreign_keys = ON;")
    db.init_schema(c, str(SCHEMA))
    return c


def test_points_is_generated_from_makes(conn):
    conn.execute("INSERT INTO team VALUES (1, 'A'), (2, 'B')")
    conn.execute("INSERT INTO player (player_id, name) VALUES (10, 'x')")
    conn.execute("INSERT INTO game VALUES ('g1', '2024-01-01', 1, 2, '2023-24', 'regular')")
    conn.execute(f"INSERT INTO box_score {BOX_COLS} VALUES ('g1', 10, 1, 30, 20, 10, 8, 4, 6, 5, 1, 4, 5, 1, 0, 2)")
    assert conn.execute("SELECT points FROM box_score").fetchone()[0] == 2 * 10 + 4 + 5
    with pytest.raises(sqlite3.OperationalError):
        conn.execute("INSERT INTO box_score (game_id, player_id, team_id, points) VALUES ('g1', 10, 1, 99)")


def test_stints_split_on_trade_and_rebuild_is_idempotent(conn):
    conn.execute("INSERT INTO team VALUES (1, 'A'), (2, 'B')")
    conn.execute("INSERT INTO player (player_id, name) VALUES (10, 'x')")
    dates = ["2024-01-01", "2024-01-03", "2024-01-05", "2024-01-07"]
    teams = [1, 1, 2, 2]
    for i, (d, t) in enumerate(zip(dates, teams)):
        conn.execute("INSERT INTO game VALUES (?, ?, 1, 2, '2023-24', 'regular')", (f"g{i}", d))
        conn.execute(f"INSERT INTO box_score {BOX_COLS} VALUES (?, 10, ?, 20, 5, 2, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0)",
                     (f"g{i}", t))
    assert db.rebuild_player_stints(conn) == 2
    assert db.rebuild_player_stints(conn) == 2  # no duplication on re-run
    rows = conn.execute("SELECT team_id, start_date, end_date FROM player_stint ORDER BY start_date").fetchall()
    assert rows == [(1, "2024-01-01", "2024-01-03"), (2, "2024-01-05", "2024-01-07")]


def test_compute_stints_handles_multiple_players():
    df = pd.DataFrame({"player_id": [1, 1, 2, 2, 2], "team_id": [5, 6, 5, 5, 5],
                       "date": ["2024-01-01", "2024-01-02", "2024-01-01", "2024-01-02", "2024-01-03"]})
    out = compute_player_stints(df)
    assert len(out) == 3 and set(out["player_id"]) == {1, 2}


def test_snake_order():
    assert snake_order(4, 3) == [1, 2, 3, 4, 4, 3, 2, 1, 1, 2, 3, 4]
    assert team_for_pick(11, 10) == 10 and team_for_pick(20, 10) == 1


def test_draft_state_picks_undo_and_clock():
    s = DraftState(LeagueSettings(n_teams=4, roster_size=2, active_slots=2, my_slot=4))
    assert s.team_on_clock == 1 and not s.my_turn and s.picks_until_my_turn() == 3
    for pid in (101, 102, 103):
        s.make_pick(pid)
    assert s.my_turn and s.my_next_pick_no() == 4 and s.my_pick_numbers() == [4, 5]
    with pytest.raises(ValueError):
        s.make_pick(101)
    s.make_pick(104)
    assert s.team_on_clock == 4  # snake: team 4 picks again
    assert s.roster(4) == [104] and s.undo() == (4, 4, 104) and s.next_pick_no == 4


def test_store_roundtrip_undo_and_exclusion(conn):
    for pid in range(1, 6):
        conn.execute("INSERT INTO player (player_id, name) VALUES (?, ?)", (pid, f"p{pid}"))
    store.save_league(conn, LeagueSettings(name="t", n_teams=3, roster_size=2, active_slots=2, my_slot=2))
    st = store.load_state(conn, "t")
    store.record_pick(conn, st, 1)
    store.record_pick(conn, st, 2)
    store.set_excluded(conn, st, 3, True)
    again = store.load_state(conn, "t")
    assert again.picks == [(1, 1, 1), (2, 2, 2)] and again.excluded == {3}
    store.undo_pick(conn, again)
    assert store.load_state(conn, "t").picks == [(1, 1, 1)]
    store.reset_draft(conn, again)
    assert store.load_state(conn, "t").picks == []
