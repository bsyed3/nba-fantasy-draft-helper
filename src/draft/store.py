"""SQLite persistence for leagues and draft state (tables defined in schema.sql)."""
from __future__ import annotations

import sqlite3
from pathlib import Path

from src.draft.state import DraftState, LeagueSettings

SCHEMA_PATH = Path(__file__).resolve().parents[2] / "schema.sql"


def ensure_schema(conn: sqlite3.Connection) -> None:
    from src.data_ingestion.load_to_db import migrate

    conn.executescript(SCHEMA_PATH.read_text(encoding="utf-8"))
    migrate(conn)
    conn.commit()


def list_leagues(conn: sqlite3.Connection) -> list[str]:
    return [r[0] for r in conn.execute("SELECT name FROM league ORDER BY league_id")]


def save_league(conn: sqlite3.Connection, s: LeagueSettings) -> int:
    """Create the league, or update its settings if the name exists. Returns league_id."""
    conn.execute(
        """
        INSERT INTO league (name, n_teams, roster_size, active_slots, my_slot, season, max_centers)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(name) DO UPDATE SET n_teams = excluded.n_teams, roster_size = excluded.roster_size,
            active_slots = excluded.active_slots, my_slot = excluded.my_slot, season = excluded.season,
            max_centers = excluded.max_centers
        """,
        (s.name, s.n_teams, s.roster_size, s.active_slots, s.my_slot, s.season, s.max_centers),
    )
    conn.commit()
    return conn.execute("SELECT league_id FROM league WHERE name = ?", (s.name,)).fetchone()[0]


def load_state(conn: sqlite3.Connection, name: str) -> DraftState:
    row = conn.execute(
        "SELECT league_id, n_teams, roster_size, active_slots, my_slot, season, max_centers FROM league WHERE name = ?", (name,)
    ).fetchone()
    if row is None:
        raise KeyError(name)
    lid = row[0]
    settings = LeagueSettings(name=name, n_teams=row[1], roster_size=row[2], active_slots=row[3],
                              my_slot=row[4], season=row[5], max_centers=row[6])
    picks = conn.execute(
        "SELECT pick_no, team_slot, player_id FROM draft_pick WHERE league_id = ? ORDER BY pick_no", (lid,)
    ).fetchall()
    excl = {r[0] for r in conn.execute("SELECT player_id FROM draft_exclusion WHERE league_id = ?", (lid,))}
    return DraftState(settings, [tuple(p) for p in picks], excl)


def _lid(conn: sqlite3.Connection, name: str) -> int:
    return conn.execute("SELECT league_id FROM league WHERE name = ?", (name,)).fetchone()[0]


def record_pick(conn: sqlite3.Connection, state: DraftState, player_id: int) -> tuple[int, int, int]:
    """Apply a pick to the in-memory state and persist it."""
    pick = state.make_pick(player_id)
    lid = _lid(conn, state.settings.name)
    conn.execute("INSERT INTO draft_pick (league_id, pick_no, team_slot, player_id) VALUES (?, ?, ?, ?)", (lid, *pick))
    conn.execute("DELETE FROM draft_exclusion WHERE league_id = ? AND player_id = ?", (lid, pick[2]))
    conn.commit()
    return pick


def undo_pick(conn: sqlite3.Connection, state: DraftState) -> tuple[int, int, int] | None:
    pick = state.undo()
    if pick is not None:
        conn.execute("DELETE FROM draft_pick WHERE league_id = ? AND pick_no = ?", (_lid(conn, state.settings.name), pick[0]))
        conn.commit()
    return pick


def set_excluded(conn: sqlite3.Connection, state: DraftState, player_id: int, excluded: bool) -> None:
    lid = _lid(conn, state.settings.name)
    if excluded:
        state.excluded.add(int(player_id))
        conn.execute("INSERT OR IGNORE INTO draft_exclusion (league_id, player_id) VALUES (?, ?)", (lid, int(player_id)))
    else:
        state.excluded.discard(int(player_id))
        conn.execute("DELETE FROM draft_exclusion WHERE league_id = ? AND player_id = ?", (lid, int(player_id)))
    conn.commit()


def reset_draft(conn: sqlite3.Connection, state: DraftState) -> None:
    lid = _lid(conn, state.settings.name)
    conn.execute("DELETE FROM draft_pick WHERE league_id = ?", (lid,))
    conn.execute("DELETE FROM draft_exclusion WHERE league_id = ?", (lid,))
    conn.commit()
    state.picks.clear()
    state.excluded.clear()
