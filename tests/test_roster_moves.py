"""Roster moves after the draft (adds, drops, injured reserve) sit on top of the draft picks, and league bans keep a player out of everything."""
import sqlite3

import pandas as pd

from src.draft import store
from src.draft.league_rules import banned_player_ids
from src.draft.state import DraftState, LeagueSettings

S2 = LeagueSettings(name="L", n_teams=2, roster_size=2, active_slots=2)
PICKS = [(1, 1, 10), (2, 2, 20), (3, 2, 21), (4, 1, 11)]          # snake: team 1 has 10 and 11, team 2 has 20 and 21


def test_moves_are_applied_on_top_of_the_picks():
    st = DraftState(S2, list(PICKS), moves=[(1, 11, "drop"), (1, 30, "add"), (2, 21, "ir"), (1, 20, "add")])
    assert st.roster(1) == [10, 30]                                  # 11 dropped, 30 added; 20 is on team 2, so that add is ignored
    assert st.roster(2) == [20] and st.ir(2) == [21]                 # IR players are off the active roster
    assert st.drafted == {10, 30, 20, 21}                            # but still taken; the dropped 11 is a free agent again
    assert not st.can_draft(21, slot=1) and st.can_draft(11, slot=2)
    st.moves.append((2, 21, "activate"))
    assert st.roster(2) == [20, 21] and st.ir(2) == []


def test_no_moves_changes_nothing():
    st = DraftState(S2, list(PICKS))
    assert st.roster(1) == [10, 11] and st.rosters() == {1: [10, 11], 2: [20, 21]} and st.ir(1) == [] and st.drafted == {10, 11, 20, 21}


def test_moves_are_saved_with_the_league_and_cleared_by_a_reset():
    conn = sqlite3.connect(":memory:")
    store.ensure_schema(conn)
    for pid in (10, 11, 20, 21, 30):
        conn.execute("INSERT INTO player (player_id, name) VALUES (?, ?)", (pid, f"P{pid}"))
    store.save_league(conn, S2)
    state = store.load_state(conn, "L")
    for pk in PICKS:
        conn.execute("INSERT INTO draft_pick (league_id, pick_no, team_slot, player_id) VALUES (1, ?, ?, ?)", pk)
    state = store.load_state(conn, "L")
    store.record_move(conn, state, 1, 11, "drop"); store.record_move(conn, state, 1, 30, "add"); store.record_move(conn, state, 2, 21, "ir", "hurt")
    again = store.load_state(conn, "L")
    assert again.moves == [(1, 11, "drop"), (1, 30, "add"), (2, 21, "ir")] and again.roster(1) == [10, 30] and again.ir(2) == [21]
    try:
        store.record_move(conn, again, 1, 10, "trade")
        raise AssertionError("an unknown action should be refused")
    except ValueError:
        pass
    store.reset_draft(conn, again)
    assert store.load_state(conn, "L").moves == []


def test_banned_players_come_from_the_league_file(tmp_path):
    f = tmp_path / "banned.csv"
    pd.DataFrame({"player_id": [1630166], "name": ["Deni Avdija"], "reason": ["banned"]}).to_csv(f, index=False)
    assert banned_player_ids(f) == {1630166}
    assert banned_player_ids(tmp_path / "missing.csv") == set()
    assert 1630166 in banned_player_ids()                            # the real list: Deni Avdija is banned from the league
