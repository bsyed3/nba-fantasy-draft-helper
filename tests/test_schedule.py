import sqlite3

import numpy as np
import pandas as pd

from src.simulation.library import STAT_IDX, STAT_ORDER
from src.simulation.schedule import actual_library, season_periods


def synthetic_season(thin_cup_week: bool = True) -> sqlite3.Connection:
    """
    4 teams play Tue/Thu/Sat all season, Oct 20 2026 - Apr 11 2027, with the two thin stretches of a real NBA calendar:
    the NBA Cup knockout week (Dec 7-13: one game day) and the All-Star break (Feb 15-28: one game day per week).
    """
    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE game (game_id TEXT, date TEXT, home_team_id INT, away_team_id INT, season TEXT, game_type TEXT)")
    conn.execute("CREATE TABLE box_score (game_id TEXT, player_id INT, team_id INT, min REAL, fgm INT, fga INT, ftm INT, fta INT,"
                 " tpm INT, oreb INT, dreb INT, ast INT, stl INT, blk INT, tov INT)")
    n = 0
    for d in pd.date_range("2026-10-20", "2027-04-11"):
        if d.dayofweek not in (1, 3, 5):
            continue
        if pd.Timestamp("2027-02-15") <= d <= pd.Timestamp("2027-02-28") and d not in (pd.Timestamp("2027-02-18"), pd.Timestamp("2027-02-25")):
            continue  # All-Star break
        if thin_cup_week and pd.Timestamp("2026-12-07") <= d <= pd.Timestamp("2026-12-13") and d != pd.Timestamp("2026-12-10"):
            continue  # Cup knockout week
        for h, a in ((1, 2), (3, 4)):
            n += 1
            conn.execute("INSERT INTO game VALUES (?, ?, ?, ?, '2026-27', 'regular')", (f"g{n}", d.strftime("%Y-%m-%d"), h, a))
    return conn


def test_matchup_periods_match_the_leagues_calendar():
    sc = season_periods(synthetic_season(), "2026-27", n_regular=18, n_playoff=3)
    spans = [(s.strftime("%m-%d"), e.strftime("%m-%d")) for s, e in sc.spans]
    assert sc.n_weeks == 21 and sc.n_days == 14
    assert spans[0] == ("10-19", "10-25")          # week 1 is a normal (partial) week
    assert spans[1] == ("10-26", "11-01")          # week 2
    assert spans[5] == ("11-23", "11-29")
    assert spans[6] == ("11-30", "12-13")          # week 7: the NBA Cup week is merged with the week before it
    assert spans[7] == ("12-14", "12-20")
    assert spans[16] == ("02-15", "02-28")         # week 17: All-Star, two weeks
    assert spans[17] == ("03-01", "03-07")         # week 18 ends the regular season
    assert spans[20] == ("03-22", "03-28")         # three playoff weeks follow
    assert season_periods(synthetic_season(), "2026-27").n_weeks == 18  # default = the 18 regular-season weeks


def test_masks_put_games_on_the_right_offsets():
    sc = season_periods(synthetic_season(), "2026-27")
    t = sc.team_index()[1]
    assert np.flatnonzero(sc.masks[0, t]).tolist() == [1, 3, 5]            # week 1: Tue 20, Thu 22, Sat 24
    assert np.flatnonzero(sc.masks[16, t]).tolist() == [3, 10]             # week 17 (Feb 15-28): Feb 18 and Feb 25


def test_unscheduled_games_are_spread_over_random_days_of_the_cup_week_only(monkeypatch):
    conn = synthetic_season()
    published = conn.execute("SELECT COUNT(*) FROM game WHERE home_team_id = 1 OR away_team_id = 1").fetchone()[0]
    monkeypatch.setattr("src.simulation.schedule.SEASON_GAMES", published + 2)    # like the real feed: 80 of 82 games published
    default = season_periods(conn, "2026-27")                               # default: unscheduled games are ignored
    assert set(default.missing.values()) == {2} and default.flex_period is None
    assert (default.team_day_masks(default.n_weeks * 5, np.random.default_rng(0)) == default.masks[np.arange(default.n_weeks * 5) % default.n_weeks]).all()
    sc = season_periods(conn, "2026-27", fill_unscheduled=True)
    assert set(sc.missing.values()) == {2}
    assert sc.flex_period == 6 and set(sc.missing) == set(sc.team_ids) and min(sc.missing.values()) > 0
    rng = np.random.default_rng(0)
    S = sc.n_weeks * 30
    tm = sc.team_day_masks(S, rng)
    wk = np.arange(S) % sc.n_weeks
    added = tm & ~sc.masks[wk]
    assert added[wk != sc.flex_period].sum() == 0                          # no change outside the Cup week
    cup = added[wk == sc.flex_period]
    assert cup.sum() > 0
    assert (cup.sum(axis=(1, 2)) == 4 * 2).all()                           # exactly 2 extra games per team, every sim
    assert cup[:, :, :7].sum() == 0                                        # only the period's final week (Dec 7-13) gets the games
    assert (cup.sum(axis=(1, 2)) > 0).all()
    assert len({c.tobytes() for c in cup}) > 1                             # the draw differs from sim to sim: genuine uncertainty
    assert (sc.expected_games()[6] > sc.masks[6].sum(axis=1)).all()        # expected games include the missing ones


def test_one_game_day_mask_per_sim_and_team_so_teammates_share_days():
    sc = season_periods(synthetic_season(), "2026-27", fill_unscheduled=True)
    tm = sc.team_day_masks(sc.n_weeks * 20, np.random.default_rng(3))
    assert tm.shape == (sc.n_weeks * 20, 4, sc.n_days)                     # indexed by team, not player: teammates share the row


def test_a_season_that_was_played_is_never_filled_in():
    conn = synthetic_season()
    first = conn.execute("SELECT game_id FROM game LIMIT 1").fetchone()[0]
    conn.execute("INSERT INTO box_score VALUES (?, 1, 1, 30, 5, 10, 2, 3, 1, 1, 4, 6, 1, 0, 2)", (first,))   # mark as already played
    sc = season_periods(conn, "2026-27")
    assert sc.flex_period is None and sc.missing == {}
    assert sc.masks[6].sum(axis=1).tolist() == [4, 4, 4, 4]                # the light Cup week stays exactly as published/played


def test_actual_library_places_box_scores_in_their_period_and_day():
    conn = synthetic_season()
    gid = conn.execute("SELECT game_id FROM game WHERE date = '2026-10-22' AND home_team_id = 1").fetchone()[0]
    conn.execute("INSERT INTO box_score VALUES (?, 7, 1, 30, 5, 10, 2, 3, 1, 1, 4, 6, 1, 0, 2)", (gid,))
    sc = season_periods(conn, "2026-27")
    lib = actual_library(conn, "2026-27", np.array([7, 8]), sc)
    assert lib.played[0].sum() == 1 and lib.played[0, 0, 3]            # Oct 22 = offset 3 of week 1
    assert lib.stats[0, 0, 3, STAT_IDX["reb"]] == 5 and lib.stats[0, 0, 3, STAT_IDX["ast"]] == 6
    assert lib.stats[1].sum() == 0                                       # a player with no games has no stats
    assert lib.stats.shape == (2, 18, 14, len(STAT_ORDER))
