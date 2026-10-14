import sqlite3

import numpy as np
import pandas as pd

from src.features.team_context import known_absent_ids, team_context

G_, F_, C_ = "G", "F", "C"


def _league():
    """Two teams, two seasons of 12 games. 2023: A has P1 (G, 30 min) + P2 (F, 20); B has P3 (C, 25) + P4 (G, 28).
    2024: P2 moves A -> B, P3 moves B -> A; P1 and P4 stay."""
    conn = sqlite3.connect(":memory:")
    conn.executescript("""
        CREATE TABLE game (game_id TEXT, date TEXT, home_team_id INT, away_team_id INT, season TEXT, game_type TEXT);
        CREATE TABLE box_score (game_id TEXT, player_id INT, team_id INT, min REAL, fga INT, fta INT, tov INT);
        CREATE TABLE player (player_id INT, name TEXT, position_1 TEXT);
    """)
    conn.executemany("INSERT INTO player VALUES (?,?,?)", [(1, "P One", G_), (2, "P Two", F_), (3, "P Three", C_), (4, "P Four", G_)])
    rosters = {"2023-24": {1: [(1, 30), (2, 20)], 2: [(3, 25), (4, 28)]}, "2024-25": {1: [(1, 30), (3, 25)], 2: [(2, 20), (4, 28)]}}
    for season, teams in rosters.items():
        for i in range(12):
            gid = f"{season}-{i}"
            conn.execute("INSERT INTO game VALUES (?,?,?,?,?,?)", (gid, f"{season[:4]}-11-{i + 1:02d}", 1, 2, season, "regular"))
            for tid, plist in teams.items():
                for pid, mins in plist:
                    conn.execute("INSERT INTO box_score VALUES (?,?,?,?,?,?,?)", (gid, pid, tid, float(mins), 10, 2, 2))
    return conn


def test_role_based_context_measures_minutes_while_playing_and_tapers_short_seasons():
    ctx = team_context(_league(), [2024]).set_index("player_id")                    # default: role-based
    taper = 12 / 40                                                                 # the fixture's seasons are only 12 games long
    assert abs(ctx.loc[1, "ctx_vac_min"] - 20.0 * taper) < 1e-9 and abs(ctx.loc[1, "ctx_inc_min"] - 25.0 * taper) < 1e-9
    # a star who missed most of last year still leaves a big hole: per-team-game minutes would have understated it
    conn = _league()
    late = ",".join(f"'2023-24-{i}'" for i in range(4, 12))
    conn.execute(f"DELETE FROM box_score WHERE player_id = 2 AND game_id IN ({late})")          # P2 plays only 4 of 12 games in 2023
    per_team_game = team_context(conn, [2024], role_based=False).set_index("player_id").loc[1, "ctx_vac_min"]
    role = team_context(conn, [2024], role_based=True).set_index("player_id").loc[1, "ctx_vac_min"]
    assert abs(per_team_game - 20.0 * 4 / 12) < 1e-9                                # 6.67: minutes per TEAM game
    assert abs(role - 20.0 * (4 / 40)) < 1e-9                                       # 2.0: 20 min per game he played, tapered for 4 games


def test_vacated_and_incoming_minutes_exclude_the_players_own_contribution():
    ctx = team_context(_league(), [2024], role_based=False).set_index("player_id")
    # team A in 2024: lost P2 (20 min/game, a forward), gained P3 (25 min/game last year, a centre)
    p1 = ctx.loc[1]
    assert (p1["ctx_vac_min"], p1["ctx_inc_min"]) == (20.0, 25.0) and p1["ctx_moved"] == 0
    assert (p1["ctx_vac_min_pos"], p1["ctx_inc_min_pos"]) == (0.0, 0.0)            # nobody left / arrived at his position (guard)
    p3 = ctx.loc[3]                                                                 # the arrival: his own incoming minutes are removed
    assert p3["ctx_moved"] == 1 and p3["ctx_inc_min"] == 0.0 and p3["ctx_vac_min"] == 20.0
    p2 = ctx.loc[2]                                                                 # moved to B, where P3 left (25) and he arrived (20)
    assert p2["ctx_moved"] == 1 and p2["ctx_vac_min"] == 25.0 and p2["ctx_inc_min"] == 0.0
    p4 = ctx.loc[4]
    assert (p4["ctx_vac_min"], p4["ctx_inc_min"], p4["ctx_moved"]) == (25.0, 20.0, 0)
    assert abs(p1["ctx_vac_poss"] - (10 + 0.44 * 2 + 2)) < 1e-9                    # possessions used per game by the departed forward
    assert ctx.loc[2:3, "ctx_vac_min_pos"].tolist() == [0.0, 0.0]


def test_projected_season_uses_current_rosters_and_known_absences_vacate_minutes():
    conn = _league()
    cur = pd.Series({1: 1, 3: 1, 2: 2, 4: 2})                                      # 2025: nobody moves
    still = team_context(conn, [2025], current_team=cur, role_based=False).set_index("player_id")
    assert (still.loc[1, "ctx_vac_min"], still.loc[1, "ctx_inc_min"]) == (0.0, 0.0) and still.loc[1, "ctx_moved"] == 0
    out = team_context(conn, [2025], current_team=cur, absent_ids={3}, role_based=False).set_index("player_id")
    assert out.loc[1, "ctx_vac_min"] == 25.0                                       # P3 is out for the start: his minutes are up for grabs
    assert out.loc[3, "ctx_vac_min"] == 0.0 and out.loc[3, "ctx_moved"] == 0       # ... but not for himself
    assert team_context(conn, [2025]).empty                                        # no current roster -> nothing to say about a future year


def test_known_absent_ids_resolve_names_and_later_files_win(tmp_path):
    players = pd.DataFrame({"player_id": [1, 2, 3], "name": ["P One", "Jaren Jackson Jr.", "P Three"]})
    a = tmp_path / "a.csv"; b = tmp_path / "b.csv"
    a.write_text("player_id,name,games_out\n1,P One,40\n3,P Three,2\n")
    b.write_text("name,games_out\njaren jackson,30\nP One,3\n")                      # b shortens P One's absence below the opening-roster cutoff
    assert known_absent_ids([a, b, tmp_path / "missing.csv"], players, min_games=10) == {2}
