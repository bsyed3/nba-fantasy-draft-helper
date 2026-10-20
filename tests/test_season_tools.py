"""In-season tools: the league table agrees with the engine's own scoring, swaps and trades are scored sensibly, and transactions are validated and undoable."""
import sqlite3

import numpy as np
import pytest

from src.draft import season, store
from src.draft.engine import DraftEngine
from src.draft.state import LeagueSettings
from src.draft.values import CATEGORIES
from src.simulation.copula import build_library
from tests.conftest import make_projection

T = 4


@pytest.fixture
def league(cal, schedule):
    """Four teams of five from the best 20 synthetic players (a snake split), the other 20 are free agents."""
    proj = make_projection(40)
    lib = build_library(proj, cal, schedule, sims_per_week=50, seed=0)               # 4 periods x 50 draws
    s = LeagueSettings(n_teams=T, roster_size=5, active_slots=4, my_slot=2, max_centers=3, reg_season_weeks=3, playoff_weeks=1)
    eng = DraftEngine(lib, proj, s)
    order = list(np.argsort(eng.rank))[:20]
    rosters = {t: [int(order[k]) for k in range(20) if (k % 8 if (k // 4) % 2 == 0 else 7 - k % 8) % 4 == t - 1][:5] for t in range(1, T + 1)}
    assert [len(v) for v in rosters.values()] == [5] * T and len({i for v in rosters.values() for i in v}) == 20
    free = [int(i) for i in np.argsort(eng.rank) if all(int(i) not in v for v in rosters.values())]
    return eng, s, rosters, free, None


def _table(eng, rosters):
    return season.league_table(eng, rosters, n_periods=4, n_regular=3)


def test_league_table_matches_the_engines_own_scoring(league):
    eng, s, rosters, _, _ = league
    tbl, ev = _table(eng, rosters), eng.evaluate(rosters)
    np.testing.assert_allclose(tbl.win, ev.win_prob)
    np.testing.assert_allclose(tbl.cat_win, ev.cat_win_prob)
    np.testing.assert_allclose(tbl.by_week.mean(1), tbl.win)                          # equal-sized periods: the weeks average to the whole
    np.testing.assert_allclose(tbl.win_reg * 3 / 4 + tbl.win_po / 4, tbl.win)         # 3 regular weeks + 1 playoff week
    h = tbl.h2h
    off = ~np.eye(T, dtype=bool)
    np.testing.assert_allclose((h + h.T)[off], 1.0)                                  # a matchup is won by one side (ties split)
    assert np.isnan(np.diag(h)).all()


def test_category_ranks_are_a_ranking_with_turnovers_lowest_first(league):
    eng, _, rosters, _, _ = league
    tbl = _table(eng, rosters)
    for k in range(len(CATEGORIES)):
        assert sorted(tbl.cat_rank[:, k]) == list(range(1, T + 1))
    k = CATEGORIES.index("tov")
    assert tbl.cat_rank[tbl.totals[:, k].argmin(), k] == 1                            # fewest turnovers is rank 1
    k = CATEGORIES.index("reb")
    assert tbl.cat_rank[tbl.totals[:, k].argmax(), k] == 1
    assert sorted(tbl.power_rank) == list(range(1, T + 1)) and tbl.power_rank[tbl.win.argmax()] == 1


def test_two_week_periods_are_scaled_to_a_week_in_the_totals(league):
    eng, _, rosters, _, _ = league
    plain = season.league_table(eng, rosters, 4, 3)
    scaled = season.league_table(eng, rosters, 4, 3, weeks=np.array([2.0, 1.0, 1.0, 1.0]))
    k = CATEGORIES.index("pts")
    assert (scaled.totals[:, k] < plain.totals[:, k]).all()
    kf = CATEGORIES.index("fg_pct")
    np.testing.assert_allclose(scaled.totals[:, kf], plain.totals[:, kf])             # shares are not scaled
    np.testing.assert_allclose(scaled.win, plain.win)                                # scoring is unaffected


def test_swap_grid_scores_each_swap_like_a_full_rescore(league):
    eng, s, rosters, free, _ = league
    tbl = _table(eng, rosters)
    me = s.my_slot
    opp = season.field_cats(tbl, me)
    roster = rosters[me]
    adds = [free[0], free[-1]]
    grid = season.swap_grid(eng, roster, opp, adds, max_centers=None)
    assert grid.base_win == pytest.approx(tbl.win[me - 1])
    d = 0
    manual = season.score_against(season.team_cats(eng, [x for x in roster if x != roster[d]] + [adds[0]], cache=False), opp)[0].mean()
    assert grid.win[0, d] == pytest.approx(manual)
    assert (grid.gain()[0] >= grid.gain()[1] - 1e-9).all()                            # the better free agent is worth at least as much as the worst
    best = grid.best()
    assert list(best["add"]) == adds and np.isfinite(best["gain"]).all()
    assert (best["gain"].to_numpy() >= grid.gain().max(axis=1) - 1e-9).all()


def test_swap_grid_open_roster_spot_and_center_cap(league):
    eng, s, rosters, free, _ = league
    tbl = _table(eng, rosters)
    me = s.my_slot
    opp = season.field_cats(tbl, me)
    roster = rosters[me]
    grid = season.swap_grid(eng, roster, opp, [free[0]], roster + [None])
    assert grid.drops[-1] is None and np.isfinite(grid.win[0, -1])
    assert type(grid.best()["drop"].iloc[0]) in (int, type(None), np.int64)           # a None must survive as None, not become NaN
    centers = [i for i in free if eng.is_center[i]]
    assert centers and not all(eng.is_center[d] for d in roster) and any(eng.is_center[d] for d in roster)
    cap = season.n_centers(eng, roster)
    capped = season.swap_grid(eng, roster, opp, [centers[0]], roster, max_centers=cap)
    for di, d in enumerate(roster):
        expect_blocked = (not eng.is_center[d])                                      # a non-center out, a center in: one more center than the cap allows
        assert bool(np.isnan(capped.win[0, di])) == expect_blocked


def test_removing_the_best_player_costs_more_than_removing_the_worst(league):
    eng, s, rosters, _, _ = league
    tbl = _table(eng, rosters)
    me = s.my_slot
    cost = season.removal_cost(eng, rosters[me], season.field_cats(tbl, me))
    best, worst = min(rosters[me], key=lambda i: eng.rank[i]), max(rosters[me], key=lambda i: eng.rank[i])
    assert cost[best] > cost[worst]


def test_apply_trade_swaps_players_and_releases_the_drops():
    r = {1: [1, 2, 3], 2: [4, 5, 6]}
    new = season.apply_trade(r, 1, 2, [1, 2], [4], drop_a=[], drop_b=[6])
    assert sorted(new[1]) == [3, 4] and sorted(new[2]) == [1, 2, 5]
    assert r == {1: [1, 2, 3], 2: [4, 5, 6]}                                         # the input is left alone


def test_trade_verdicts():
    assert season.verdict(1.0, 1.0)[0] == "good"
    assert season.verdict(1.0, -0.3)[0] == "ok"
    level, msg = season.verdict(1.2, -2.0)
    assert level == "warn" and "unlikely" in msg
    assert season.verdict(0.1, 1.5)[0] == "warn"
    assert season.verdict(0.1, 0.1)[0] == "ok"
    assert season.verdict(-1.0, 1.0)[0] == "bad" and season.verdict(-1.0, -1.0)[0] == "bad"


# ---------------------------------------------------------------------------
# transactions in the database
# ---------------------------------------------------------------------------
S2 = LeagueSettings(name="L", n_teams=2, roster_size=3, active_slots=2)


@pytest.fixture
def db():
    conn = sqlite3.connect(":memory:")
    store.ensure_schema(conn)
    for pid in range(10, 60, 10):
        conn.execute("INSERT INTO player (player_id, name) VALUES (?, ?)", (pid, f"P{pid}"))
    store.save_league(conn, S2)
    for pk in [(1, 1, 10), (2, 2, 20), (3, 2, 30), (4, 1, 40)]:                       # team 1: 10, 40; team 2: 20, 30; 50 is a free agent
        conn.execute("INSERT INTO draft_pick (league_id, pick_no, team_slot, player_id) VALUES (1, ?, ?, ?)", pk)
    return conn, store.load_state(conn, "L")


def test_ensure_league_creates_once_and_leaves_an_existing_league_alone():
    conn = sqlite3.connect(":memory:")
    store.ensure_schema(conn)
    store.ensure_league(conn, LeagueSettings(name="X", my_slot=3))
    store.ensure_league(conn, LeagueSettings(name="X", my_slot=7))
    assert store.list_leagues(conn) == ["X"] and store.load_state(conn, "X").settings.my_slot == 3


def test_a_pickup_with_its_drop_is_one_transaction_and_undoes_together(db):
    conn, state = db
    store.record_transaction(conn, state, [(1, 50, "add"), (1, 40, "drop")], "waiver")          # the add is listed first; the drop still goes first
    again = store.load_state(conn, "L")
    assert sorted(again.roster(1)) == [10, 50] and again.drafted == {10, 20, 30, 50}
    log = store.moves_log(conn, "L")
    assert [m["action"] for m in log] == ["drop", "add"] and log[0]["note"] == log[1]["note"] and log[0]["note"].startswith("waiver [")
    assert store.undo_move(conn, state) == 2
    assert sorted(store.load_state(conn, "L").roster(1)) == [10, 40] and state.moves == []


def test_a_trade_moves_players_between_teams_and_undoes_in_one_step(db):
    conn, state = db
    store.record_move(conn, state, 2, 30, "ir", "old")                                           # an earlier, separate move
    store.record_trade(conn, state, 1, 2, [40], [20])
    again = store.load_state(conn, "L")
    assert sorted(again.roster(1)) == [10, 20] and sorted(again.roster(2)) == [40] and again.ir(2) == [30]
    assert store.undo_move(conn, state) == 4                                                     # two drops and two adds
    again = store.load_state(conn, "L")
    assert sorted(again.roster(1)) == [10, 40] and sorted(again.roster(2)) == [20] and again.ir(2) == [30]
    assert store.undo_move(conn, state) == 1                                                     # the earlier move had no transaction id: taken back on its own
    assert store.undo_move(conn, state) == 0


def test_legacy_moves_with_the_same_note_are_not_undone_together(db):
    conn, state = db
    store.record_move(conn, state, 1, 40, "drop", "free-agent pickup")
    store.record_move(conn, state, 1, 50, "add", "free-agent pickup")
    assert store.undo_move(conn, state) == 1


def test_an_impossible_transaction_is_refused_and_writes_nothing(db):
    conn, state = db
    for bad in ([(1, 20, "drop")],                       # 20 is on team 2
                [(1, 10, "add")],                        # 10 is already rostered
                [(1, 50, "add"), (2, 50, "add")]):       # the same free agent twice
        with pytest.raises(ValueError):
            store.record_transaction(conn, state, bad, "bad")
    assert store.moves_log(conn, "L") == [] and state.moves == []
