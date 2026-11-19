"""
When each input of the app was last refreshed, for the "Data freshness" box in the app's sidebar.

The refresh scripts call `record` when they finish (ESPN injuries, projections, simulation library); the date the game stats run through
comes from the live database itself (`live_meta`, written by scripts/fetch_live_stats.py). Everything lives in
data/processed/data_status.json: {"injuries": {"at": "2026-10-27T11:37:44", "listings": 83, "out": 61}, "projections": {...}, "library": {...}}.
Times are local and naive; this is a note to the user, not a clock the code depends on.
"""
from __future__ import annotations

import datetime as dt
import json
import os
from pathlib import Path

DEFAULT_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"
FILE = "data_status.json"


def read(out_dir: str | Path | None = None) -> dict:
    p = Path(out_dir or DEFAULT_DIR) / FILE
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def record(key: str, out_dir: str | Path | None = None, at: dt.datetime | None = None, **info) -> None:
    """Note that `key` was refreshed now (or at `at`), with any facts worth showing. The file is replaced in one step."""
    d = Path(out_dir or DEFAULT_DIR)
    status = read(d)
    status[key] = {"at": (at or dt.datetime.now()).isoformat(timespec="seconds"), **info}
    d.mkdir(parents=True, exist_ok=True)
    tmp = d / (FILE + ".tmp")
    tmp.write_text(json.dumps(status, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, d / FILE)


def _day(d: dt.date) -> str:
    return f"{d:%b} {d.day}"


def _when(iso: str) -> dt.datetime | None:
    try:
        return dt.datetime.fromisoformat(iso)
    except (TypeError, ValueError):
        return None


def _stamp(t: dt.datetime) -> str:
    return f"{_day(t.date())}, {t:%I:%M %p}".replace(" 0", " ")


def _ago(d: dt.date, today: dt.date) -> str:
    n = (today - d).days
    return "today" if n == 0 else "yesterday" if n == 1 else f"{n} days ago" if n > 1 else ""


def describe(status: dict, live: dict | None, today: dt.date | None = None) -> list[tuple[str, str]]:
    """
    (label, text) lines for the sidebar. `live` is the live database's meta ({'last_game_date', 'fetched_at', 'games'}) or None when it has not
    been pulled. A warning line (label 'Warning') is added if the simulation library is older than the projections it should have been built from.
    """
    today = today or dt.date.today()
    lines: list[tuple[str, str]] = []

    last = _when((live or {}).get("last_game_date", "") + "T00:00:00") if live else None
    if live and last:
        pulled = _when(live.get("fetched_at", ""))
        text = f"through {_day(last.date())} ({_ago(last.date(), today)}), {live.get('games', '?')} games"
        lines.append(("Game stats", text + (f" · pulled {_stamp(pulled)}" if pulled else "")))
    elif live:
        lines.append(("Game stats", "pulled, but no games have been played yet"))
    else:
        lines.append(("Game stats", "not pulled yet (python scripts/fetch_live_stats.py)"))

    inj = status.get("injuries")
    t = _when(inj.get("at")) if inj else None
    if t:
        lines.append(("Injuries (ESPN)", f"as of {_stamp(t)} ({_ago(t.date(), today)})" + (f" · {inj['out']} players out" if "out" in inj else "")))
    else:
        lines.append(("Injuries (ESPN)", "no refresh recorded (python scripts/fetch_espn_injuries.py)"))

    proj, lib = status.get("projections"), status.get("library")
    tp, tl = (_when(proj.get("at")) if proj else None), (_when(lib.get("at")) if lib else None)
    if tp or tl:
        built = max(x for x in (tp, tl) if x)
        lines.append(("Projections", f"built {_stamp(built)} from the preseason model and the injury list; game stats are not used in them yet"))
        if tp and tl and tl < tp:
            lines.append(("Warning", "the simulation library is older than the projections: run python scripts/build_library.py"))
    else:
        lines.append(("Projections", "no build recorded"))
    return lines
