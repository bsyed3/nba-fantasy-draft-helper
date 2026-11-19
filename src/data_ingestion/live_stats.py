"""
In-season box scores, kept apart from the training data.

The games of the season being projected go in their own database (data/processed/live_<year>.db), NOT in nba.db's `game` / `box_score`:
build_projections.py takes "the season after the latest one with box scores" as its target, and schedule.season_periods treats a season
with any box score as already played, so loading live games there would silently make the next rebuild project the wrong season.
Here they can be refreshed at will (one full pull replaces the tables) and used by whatever is built on top.

Each game is tagged with its fantasy matchup week (the periods of the published schedule, week 1 = the Monday-Sunday containing the opener),
and the `live_player_week` view sums a player's games per week in the league's categories.
"""
from __future__ import annotations

import datetime as dt
import sqlite3

import pandas as pd

SCHEMA = """
CREATE TABLE IF NOT EXISTS live_game (
    game_id TEXT PRIMARY KEY,
    date DATE NOT NULL,
    home_team_id INTEGER NOT NULL,
    away_team_id INTEGER NOT NULL,
    week INTEGER                       -- fantasy matchup week (1-based); NULL if the date is outside the league's weeks
);
CREATE TABLE IF NOT EXISTS live_box_score (
    game_id TEXT NOT NULL REFERENCES live_game(game_id),
    player_id INTEGER NOT NULL,
    name TEXT,
    team_id INTEGER NOT NULL,
    min REAL,
    fga INTEGER, fgm INTEGER, tpa INTEGER, tpm INTEGER, fta INTEGER, ftm INTEGER,
    oreb INTEGER, dreb INTEGER, ast INTEGER, stl INTEGER, blk INTEGER, tov INTEGER,
    points INTEGER GENERATED ALWAYS AS (2 * fgm + tpm + ftm) VIRTUAL,
    PRIMARY KEY (game_id, player_id)
);
CREATE TABLE IF NOT EXISTS live_meta (key TEXT PRIMARY KEY, value TEXT);
DROP VIEW IF EXISTS live_player_week;
CREATE VIEW live_player_week AS
    SELECT g.week AS week, b.player_id AS player_id, MAX(b.name) AS name, b.team_id AS team_id,
           SUM(b.min > 0) AS games, SUM(b.min) AS min,
           SUM(b.points) AS pts, SUM(b.tpm) AS tpm, SUM(b.oreb + b.dreb) AS reb, SUM(b.ast) AS ast, SUM(b.stl) AS stl, SUM(b.blk) AS blk,
           SUM(b.fgm) AS fgm, SUM(b.fga) AS fga, SUM(b.ftm) AS ftm, SUM(b.fta) AS fta, SUM(b.tov) AS tov
    FROM live_box_score b JOIN live_game g ON g.game_id = b.game_id
    WHERE b.min > 0
    GROUP BY g.week, b.player_id, b.team_id;
"""
BOX_COLS = ["game_id", "player_id", "name", "team_id", "min", "fga", "fgm", "tpa", "tpm", "fta", "ftm", "oreb", "dreb", "ast", "stl", "blk", "tov"]


def week_of(dates: pd.Series, spans: list[tuple]) -> list:
    """Matchup week (1-based) of each date given the periods as [(first day, last day), ...]; None when a date is in no period."""
    bounds = [(pd.Timestamp(a).normalize(), pd.Timestamp(b).normalize()) for a, b in spans]
    out = []
    for d in pd.to_datetime(dates):
        d = d.normalize()
        out.append(next((i + 1 for i, (a, b) in enumerate(bounds) if a <= d <= b), None))
    return out


def save_live(conn: sqlite3.Connection, games: pd.DataFrame, box: pd.DataFrame, spans: list[tuple], season: str,
              fetched_at: dt.datetime | None = None) -> dict:
    """
    Replace the live tables with a full pull: `games` (game_id, date, home_team_id, away_team_id) and `box` (BOX_COLS). Box lines for a game that is
    not in `games` are dropped. Returns {'games', 'player_games', 'last_date'}.
    """
    games = games[["game_id", "date", "home_team_id", "away_team_id"]].copy()
    games["date"] = pd.to_datetime(games["date"]).dt.strftime("%Y-%m-%d")
    games["week"] = week_of(games["date"], spans)
    box = box[box["game_id"].isin(set(games["game_id"]))][BOX_COLS]
    conn.executescript(SCHEMA)
    with conn:                                                       # one transaction: a failed pull leaves the previous data in place
        conn.execute("DELETE FROM live_box_score")
        conn.execute("DELETE FROM live_game")
        conn.executemany("INSERT INTO live_game (game_id, date, home_team_id, away_team_id, week) VALUES (?, ?, ?, ?, ?)",
                         [(r.game_id, r.date, int(r.home_team_id), int(r.away_team_id), None if pd.isna(r.week) else int(r.week)) for r in games.itertuples()])
        conn.executemany(f"INSERT INTO live_box_score ({', '.join(BOX_COLS)}) VALUES ({', '.join('?' * len(BOX_COLS))})",
                         box.astype(object).where(box.notna(), None).itertuples(index=False, name=None))
        last = games["date"].max() if len(games) else None
        meta = {"season": season, "fetched_at": (fetched_at or dt.datetime.now()).isoformat(timespec="seconds"), "last_game_date": last or "", "games": str(len(games))}
        conn.executemany("INSERT OR REPLACE INTO live_meta (key, value) VALUES (?, ?)", list(meta.items()))
    return {"games": len(games), "player_games": len(box), "last_date": last}


def player_weeks(conn: sqlite3.Connection) -> pd.DataFrame:
    """One row per player per fantasy week: games, minutes and the category totals (pts tpm reb ast stl blk tov, fgm/fga, ftm/fta)."""
    return pd.read_sql_query("SELECT * FROM live_player_week ORDER BY week, pts DESC", conn)
