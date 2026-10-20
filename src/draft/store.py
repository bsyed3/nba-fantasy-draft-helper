"""SQLite persistence for leagues and draft state (tables defined in schema.sql)."""
from __future__ import annotations

import json
import re
import sqlite3
import uuid
from pathlib import Path

from src.draft.state import DraftState, LeagueSettings

SCHEMA_PATH = Path(__file__).resolve().parents[2] / "schema.sql"
TXN_NOTE = re.compile(r" \[[0-9a-f]{6}\]$")   # the id record_transaction appends to every move of one transaction


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


def get_team_names(conn: sqlite3.Connection, name: str, n_teams: int) -> list[str]:
    """The league's team names in draft-slot order; a missing or blank name falls back to 'Team <slot>'."""
    row = conn.execute("SELECT team_names FROM league WHERE name = ?", (name,)).fetchone()
    try:
        saved = json.loads(row[0]) if row and row[0] else []
    except (TypeError, ValueError):
        saved = []
    return [(str(saved[i]).strip() if i < len(saved) and str(saved[i]).strip() else f"Team {i + 1}") for i in range(n_teams)]


def set_team_names(conn: sqlite3.Connection, name: str, names: list[str]) -> None:
    conn.execute("UPDATE league SET team_names = ? WHERE name = ?", (json.dumps([str(n).strip() for n in names], ensure_ascii=False), name))
    conn.commit()


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
    moves = [tuple(m) for m in conn.execute("SELECT team_slot, player_id, action FROM roster_move WHERE league_id = ? ORDER BY move_no", (lid,))]
    return DraftState(settings, [tuple(p) for p in picks], excl, moves=moves)


def record_move(conn: sqlite3.Connection, state: DraftState, slot: int, player_id: int, action: str, note: str = "") -> None:
    """Record a roster move after the draft: 'add' a free agent, 'drop', 'ir' (to injured reserve) or 'activate' (back from IR)."""
    if action not in ("add", "drop", "ir", "activate"):
        raise ValueError(f"unknown roster action {action!r}")
    conn.execute("INSERT INTO roster_move (league_id, team_slot, player_id, action, note) VALUES (?, ?, ?, ?, ?)",
                 (_lid(conn, state.settings.name), int(slot), int(player_id), action, note))
    conn.commit()
    state.moves.append((int(slot), int(player_id), action))


def record_transaction(conn: sqlite3.Connection, state: DraftState, moves: list[tuple[int, int, str]], label: str) -> str:
    """
    Record several roster moves as one transaction (a pickup with its drop, a trade). Drops go first, so a player released by one team
    can be added by another. The moves share a note ('<label> [<id>]'), which `undo_move` uses to take the whole transaction back.
    Raises ValueError, writing nothing, if a drop names someone who is not on that team or an add names someone already on a roster.
    """
    ordered = [m for m in moves if m[2] != "add"] + [m for m in moves if m[2] == "add"]
    active, ir = state._by_team()
    on_team = {t: set(active[t]) | set(ir[t]) for t in active}
    owner = {p for v in on_team.values() for p in v}
    for slot, pid, action in ordered:
        if action == "drop":
            if pid not in on_team[slot]:
                raise ValueError(f"player {pid} is not on team {slot}")
            on_team[slot].discard(pid); owner.discard(pid)
        elif action == "add":
            if pid in owner:
                raise ValueError(f"player {pid} is already on a roster")
            on_team[slot].add(pid); owner.add(pid)
    note = f"{label} [{uuid.uuid4().hex[:6]}]"
    for slot, pid, action in ordered:
        record_move(conn, state, slot, pid, action, note)
    return note


def record_trade(conn: sqlite3.Connection, state: DraftState, a: int, b: int, give_a: list[int], give_b: list[int],
                 drop_a: list[int] = (), drop_b: list[int] = (), label: str = "trade") -> str:
    """Team `a` sends `give_a` to team `b` for `give_b`; `drop_a` / `drop_b` are released to make room. One undoable transaction."""
    moves = [(a, p, "drop") for p in [*give_a, *drop_a]] + [(b, p, "drop") for p in [*give_b, *drop_b]]
    moves += [(b, p, "add") for p in give_a] + [(a, p, "add") for p in give_b]
    return record_transaction(conn, state, moves, label)


def undo_move(conn: sqlite3.Connection, state: DraftState) -> int:
    """Take back the latest roster transaction: the last move plus any directly before it that share its note. Returns how many moves were removed."""
    lid = _lid(conn, state.settings.name)
    rows = conn.execute("SELECT move_no, note FROM roster_move WHERE league_id = ? ORDER BY move_no DESC", (lid,)).fetchall()
    if not rows:
        return 0
    take = [rows[0][0]]
    if rows[0][1] and TXN_NOTE.search(rows[0][1]):                       # only transactions recorded together are undone together
        for no, note in rows[1:]:
            if note != rows[0][1]:
                break
            take.append(no)
    conn.executemany("DELETE FROM roster_move WHERE league_id = ? AND move_no = ?", [(lid, n) for n in take])
    conn.commit()
    del state.moves[len(state.moves) - len(take):]
    return len(take)


def ensure_league(conn: sqlite3.Connection, settings: LeagueSettings) -> None:
    """Create the league row if the database does not have it yet (a fresh database); an existing league is left untouched."""
    if conn.execute("SELECT 1 FROM league WHERE name = ?", (settings.name,)).fetchone() is None:
        save_league(conn, settings)


def moves_log(conn: sqlite3.Connection, name: str) -> list[dict]:
    """Every post-draft roster move, oldest first: {move_no, team_slot, player_id, action, note}."""
    cur = conn.execute("SELECT move_no, team_slot, player_id, action, note FROM roster_move WHERE league_id = ? ORDER BY move_no", (_lid(conn, name),))
    return [dict(zip(("move_no", "team_slot", "player_id", "action", "note"), r)) for r in cur]


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
    conn.execute("DELETE FROM roster_move WHERE league_id = ?", (lid,))
    conn.commit()
    state.picks.clear()
    state.excluded.clear()
    state.moves.clear()
