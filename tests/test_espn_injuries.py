import datetime as dt
import sqlite3

import pandas as pd

from src.data_ingestion import espn_injuries as E

HTML = """
<div class="ResponsiveTable Table__league-injuries"><div class="Table__Title"><span class="injuries__teamName ml2">Atlanta Hawks</span></div>
<table class="Table"><thead><tr><th>NAME</th><th>POS</th><th>EST. RETURN DATE</th><th>STATUS</th><th>COMMENT</th></tr></thead><tbody>
<tr><td class="col-name"><a href="x">Henri Veesaar</a></td><td>C</td><td>Jul 1</td><td><span>Out</span></td><td>Sep 21: torn ACL, expected to miss the 2026-27 season.</td></tr>
<tr><td class="col-name"><a href="x">Dyson Daniels</a></td><td>G</td><td>Oct 16</td><td><span>Day-To-Day</span></td><td>Oct 12: Daniels (rest) is out for Monday's preseason game.</td></tr>
</tbody></table></div>
<div class="ResponsiveTable Table__league-injuries"><div class="Table__Title"><span class="injuries__teamName ml2">LA Clippers</span></div>
<table class="Table"><tbody>
<tr><td class="col-name"><a href="x">Brandon Ingram</a></td><td>F</td><td>Nov 1</td><td><span>Out</span></td><td>Sep 28: Ingram won't be ready for the start of the season.</td></tr>
</tbody></table></div>
"""


def test_parse_html_reads_every_team_table():
    df = E.parse_html(HTML)
    assert list(df["name"]) == ["Henri Veesaar", "Dyson Daniels", "Brandon Ingram"]
    assert list(df["team"]) == ["Atlanta Hawks", "Atlanta Hawks", "LA Clippers"]
    assert list(df["status"]) == ["Out", "Day-To-Day", "Out"] and df["est_return"].iloc[2] == "Nov 1"


def test_return_dates_are_resolved_relative_to_the_dated_comment():
    today = dt.date(2026, 10, 14)
    assert E.resolve_return_date("Oct 23", "Oct 9: out at least two weeks", today) == dt.date(2026, 10, 23)
    assert E.resolve_return_date("Jan 2", "Oct 2: back early in 2027", today) == dt.date(2027, 1, 2)
    assert E.resolve_return_date("Oct 1", "Oct 6: season-ending Achilles", today) == dt.date(2027, 10, 1)       # earlier than the comment -> next year
    assert E.resolve_return_date("Jan 1", "Feb 9: torn ACL", today) == dt.date(2027, 1, 1)                      # comment from last February
    assert E.resolve_return_date("Oct 16", "", today) == dt.date(2026, 10, 16)                                   # undated comment -> relative to today
    assert E.resolve_return_date("", "Oct 9: ...", today) is None and E.resolve_return_date("TBD", "Oct 9: ...", today) is None


def _db():
    conn = sqlite3.connect(":memory:")
    conn.executescript("""
        CREATE TABLE team (team_id INT, name TEXT);
        CREATE TABLE player (player_id INT, name TEXT, team_id INT, last_year INT);
        CREATE TABLE game (game_id TEXT, date TEXT, home_team_id INT, away_team_id INT, season TEXT, game_type TEXT);
    """)
    conn.executemany("INSERT INTO team VALUES (?,?)", [(1, "Atlanta Hawks"), (2, "Los Angeles Clippers"), (3, "Boston Celtics")])   # ESPN writes "LA Clippers"
    conn.executemany("INSERT INTO player VALUES (?,?,?,?)", [
        (10, "Henri Veesaar", 1, 2026), (11, "Brandon Ingram", 2, 2026), (12, "Dyson Daniels", 1, 2026),
        (20, "Mark Williams", 1, 2026), (21, "Mark Williams", 3, 2026),                                          # two players, one name
        (30, "Nikola Jokić", 3, 2026), (31, "Retired Guy", 3, 2024)])
    for i in range(20):                                                                                          # team 1 plays 20 games, one every two days
        conn.execute("INSERT INTO game VALUES (?,?,?,?,?,?)", (f"g{i}", f"2026-10-{20 + i}" if i < 12 else f"2026-11-{i - 11:02d}", 1 if i % 2 == 0 else 2, 2 if i % 2 == 0 else 1, "2026-27", "regular"))
    return conn


def test_known_absences_counts_team_games_before_the_return_date():
    listings = pd.DataFrame([
        dict(team="Atlanta Hawks", name="Henri Veesaar", pos="C", est_return="Jul 1", status="Out", comment="Sep 21: torn ACL"),
        dict(team="LA Clippers", name="Brandon Ingram", pos="F", est_return="Oct 30", status="Out", comment="Sep 28: partially torn Achilles"),
        dict(team="Atlanta Hawks", name="Dyson Daniels", pos="G", est_return="Oct 16", status="Day-To-Day", comment="Oct 12: rest"),
        dict(team="Atlanta Hawks", name="Mark Williams", pos="C", est_return="Oct 25", status="Out", comment="Oct 10: ankle"),
        dict(team="Atlanta Hawks", name="Nobody Known", pos="C", est_return="Oct 25", status="Out", comment="Oct 10: ankle"),
    ], columns=E.COLUMNS)
    table, warnings = E.known_absences(listings, _db(), today=dt.date(2026, 10, 14))
    t = table.set_index("name")
    assert t.loc["Henri Veesaar", "games_out"] == 82                    # returns after the schedule ends -> out for the season
    assert "Dyson Daniels" not in t.index                               # Day-To-Day is filtered out by default
    assert t.loc["Mark Williams", "player_id"] == 20                    # duplicate name resolved by team
    assert t.loc["Mark Williams", "games_out"] == round(82 * 5 / 20)    # 5 of the team's 20 games (Oct 20-24) come before Oct 25
    assert 0 < t.loc["Brandon Ingram", "games_out"] < 82 and t.loc["Brandon Ingram", "games_out"] == round(82 * 10 / 20)
    assert any("Nobody Known" in w for w in warnings)
    both, _ = E.known_absences(listings, _db(), today=dt.date(2026, 10, 14), statuses=("Out", "Day-To-Day"))
    assert "Dyson Daniels" in set(both["name"])
