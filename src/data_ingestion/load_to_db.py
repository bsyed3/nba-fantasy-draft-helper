"""
Load pulled DataFrames into the SQLite database (schema.sql). Each
function does one thing: take a DataFrame shaped like nba_api_client.py's
output and load it into the matching table.

Team/player/game/box_score loads use INSERT OR IGNORE, so re-running
ingestion for a season you've already pulled is safe (won't crash on
duplicate primary keys, won't duplicate rows either). player_stint does
NOT have this protection yet — it has no natural unique key from the
source data, so re-running compute_player_stints + load_player_stints for
a season you've already loaded will duplicate stint rows. Either clear
player_stint for that player/date-range first, or add a uniqueness check,
before re-running (flagging this rather than solving it — a fine TODO for
later, not a blocker for getting started).
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd

BOX_SCORE_COLUMNS = [
    "game_id", "player_id", "team_id", "min", "fga", "fgm", "tpa", "tpm",
    "fta", "ftm", "oreb", "dreb", "ast", "stl", "blk", "tov",
]


def get_connection(db_path: str = "data/processed/nba.db") -> sqlite3.Connection:
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn


def init_schema(conn: sqlite3.Connection, schema_path: str = "schema.sql") -> None:
    with open(schema_path, encoding="utf-8") as f:
        conn.executescript(f.read())
    migrate(conn)
    conn.commit()


def migrate(conn: sqlite3.Connection) -> None:
    """Add columns introduced after a database was first created (CREATE IF NOT EXISTS won't)."""
    wanted = {
        "player": {"team_id": "INTEGER", "rookie_year": "INTEGER", "last_year": "INTEGER"},
        "league": {"max_centers": "INTEGER NOT NULL DEFAULT 3"},
    }
    for table, cols in wanted.items():
        have = {r[1] for r in conn.execute(f"PRAGMA table_info({table})")}
        for col, ddl in cols.items():
            if have and col not in have:
                conn.execute(f"ALTER TABLE {table} ADD COLUMN {col} {ddl}")


def _insert_or_ignore(conn: sqlite3.Connection, table: str, df: pd.DataFrame) -> int:
    """Idempotent load. Returns the number of rows attempted (not necessarily inserted)."""
    if df.empty:
        return 0
    cols = list(df.columns)
    placeholders = ", ".join(["?"] * len(cols))
    sql = f"INSERT OR IGNORE INTO {table} ({', '.join(cols)}) VALUES ({placeholders})"
    conn.executemany(sql, [tuple(_py(v) for v in r) for r in df[cols].itertuples(index=False, name=None)])
    conn.commit()
    return len(df)


def load_teams(conn: sqlite3.Connection, teams_df: pd.DataFrame) -> None:
    _insert_or_ignore(conn, "team", teams_df)


def load_players(conn: sqlite3.Connection, players_df: pd.DataFrame) -> None:
    _insert_or_ignore(conn, "player", players_df)


PLAYER_BIO_COLUMNS = [
    "player_id", "name", "position_1", "position_2", "draft_year",
    "draft_position", "rookie_year", "last_year", "team_id",
]


def _py(v):
    """numpy/pandas scalar -> plain Python (None for NA/NaN) so sqlite3 can bind it."""
    if v is None or v is pd.NA or (isinstance(v, float) and v != v):
        return None
    return v.item() if hasattr(v, "item") else v


def upsert_player_index(conn: sqlite3.Connection, index_df: pd.DataFrame) -> None:
    """Insert/refresh bio fields from nba_api PlayerIndex. Keeps birthdate."""
    rows = [tuple(_py(v) for v in r) for r in index_df[PLAYER_BIO_COLUMNS].itertuples(index=False, name=None)]
    conn.executemany(
        """
        INSERT INTO player (player_id, name, position_1, position_2, draft_year,
                            draft_position, rookie_year, last_year, team_id)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(player_id) DO UPDATE SET
            name = excluded.name,
            position_1 = excluded.position_1,
            position_2 = excluded.position_2,
            draft_year = excluded.draft_year,
            draft_position = excluded.draft_position,
            rookie_year = excluded.rookie_year,
            last_year = excluded.last_year,
            team_id = excluded.team_id
        """,
        rows,
    )
    conn.commit()


def load_player_seasons(conn: sqlite3.Connection, ages_df: pd.DataFrame) -> None:
    """Idempotent: replaces age/team for a (player, season)."""
    conn.executemany(
        "INSERT OR REPLACE INTO player_season (player_id, season, age, team_id) VALUES (?, ?, ?, ?)",
        ages_df[["player_id", "season", "age", "team_id"]].itertuples(index=False, name=None),
    )
    conn.commit()


def load_games(conn: sqlite3.Connection, games_df: pd.DataFrame) -> None:
    _insert_or_ignore(conn, "game", games_df)


def load_box_scores(conn: sqlite3.Connection, box_score_df: pd.DataFrame) -> None:
    _insert_or_ignore(conn, "box_score", box_score_df[BOX_SCORE_COLUMNS])


def load_player_stints(conn: sqlite3.Connection, stints_df: pd.DataFrame) -> None:
    """Append stints. Prefer rebuild_player_stints(), which is idempotent."""
    if stints_df.empty:
        return
    cols = ["player_id", "team_id", "start_date", "end_date"]
    stints_df[cols].to_sql("player_stint", conn, if_exists="append", index=False)
    conn.commit()


def rebuild_player_stints(conn: sqlite3.Connection) -> int:
    """Delete and recompute every player_stint from box_score + game."""
    from src.features.stints import compute_player_stints

    box = pd.read_sql_query(
        """
        SELECT bs.player_id, bs.team_id, g.date
        FROM box_score bs JOIN game g ON g.game_id = bs.game_id
        WHERE g.game_type = 'regular'
        """,
        conn,
    )
    stints = compute_player_stints(box)
    conn.execute("DELETE FROM player_stint")
    conn.commit()
    load_player_stints(conn, stints)
    return len(stints)


def update_player_bio(conn: sqlite3.Connection, bio: dict) -> None:
    conn.execute(
        """
        UPDATE player
        SET name = :name,
            birthdate = :birthdate,
            position_1 = :position_1,
            position_2 = :position_2,
            draft_year = :draft_year,
            draft_position = :draft_position
        WHERE player_id = :player_id
        """,
        bio,
    )
    conn.commit()
