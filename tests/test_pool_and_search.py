"""The draft pool and player search: nobody who should be draftable is missing, and plain spelling finds accented / punctuated names."""
import sqlite3

import pandas as pd

from src.draft import store
from src.draft.state import LeagueSettings
from src.features.names import search_key
from src.models.projection import add_recent_players


def test_team_names_are_saved_with_the_league_and_default_to_team_n():
    conn = sqlite3.connect(":memory:")
    store.ensure_schema(conn)
    store.save_league(conn, LeagueSettings(name="L", n_teams=4, roster_size=5, active_slots=3, my_slot=1, season="2026-27"))
    assert store.get_team_names(conn, "L", 4) == ["Team 1", "Team 2", "Team 3", "Team 4"]
    store.set_team_names(conn, "L", ["Bilal", "Patrick", "", "Chanuth"])
    assert store.get_team_names(conn, "L", 4) == ["Bilal", "Patrick", "Team 3", "Chanuth"]        # a blank name falls back to its slot
    assert store.get_team_names(conn, "L", 5)[4] == "Team 5"                                       # a longer league is padded
    store.save_league(conn, LeagueSettings(name="L", n_teams=4, roster_size=5, active_slots=3, my_slot=2, season="2026-27"))
    assert store.get_team_names(conn, "L", 4)[0] == "Bilal"                                        # editing other settings keeps the names


def test_search_key_is_the_plain_letter_spelling():
    assert search_key("Day'Ron Sharpe") == "dayron sharpe"          # the reported miss: typing 'Dayron' must find him
    assert search_key("Nikola Jokić") == "nikola jokic"
    assert search_key("Luka Dončić") == "luka doncic"
    assert search_key("P.J. Washington") == "pj washington"
    assert search_key("Shai Gilgeous-Alexander") == "shai gilgeous alexander"
    assert search_key("Ernest Udeh, Jr.") == "ernest udeh jr"        # suffixes stay: 'jr' is something people type
    assert search_key("  Karl-Anthony   Towns ") == "karl anthony towns"
    assert "dayron" in search_key("Day'Ron Sharpe") and "dayron" not in "day'ron sharpe"


def test_recent_rotation_players_stay_in_the_pool():
    players = pd.DataFrame({"player_id": [1, 2, 3, 4], "last_year": [2026, 2025, 2025, 2024]})
    st = pd.DataFrame({"year": [2025, 2025, 2025, 2024], "player_id": [1, 2, 3, 4], "min": [2000.0, 900.0, 120.0, 1500.0]})
    out, ids = add_recent_players(players, st, 2026)
    assert ids == {2}                                                  # on no current roster, but played 900 minutes last season
    last = out.set_index("player_id")["last_year"]
    assert last[1] == 2026 and last[2] == 2026                         # the rostered player is untouched; the free agent is in the pool
    assert last[3] == 2025                                             # 120 minutes: not a draftable rotation player
    assert last[4] == 2024                                             # last played two seasons ago
    assert players.set_index("player_id").loc[2, "last_year"] == 2025  # the input table is not modified
