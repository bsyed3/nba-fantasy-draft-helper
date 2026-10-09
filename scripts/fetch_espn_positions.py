"""
Fetch ESPN's exact fantasy position eligibility (and average draft position)
and write data/external/positions.csv, which the pipeline then prefers over
its own coarse-label approximation.

    python scripts/fetch_espn_positions.py                 # 2026-27 (ESPN season id 2027)
    python scripts/fetch_espn_positions.py --season 2027 --out data/external/positions.csv
    python scripts/fetch_espn_positions.py --from-json espn_players.json   # parse a saved API response

Then re-run:  python scripts/build_projections.py && python scripts/build_library.py

Uses ESPN's public fantasy API (no login). Run it from a network that can
reach lm-api-reads.fantasy.espn.com; certificate verification is left ON.
Output columns: player_id, name, eligibility ("PG,SG"), espn_adp (may be blank).

NOTE: parse_players() is covered by a unit test on a synthetic payload, but the
live request has not been exercised from the development environment (ESPN is
blocked there) -- eyeball the printed sample against ESPN once.
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding="utf-8")

from src.features.names import normalize_name

# ESPN basketball lineup-slot ids; 0-4 are the dedicated position slots
SLOT_NAMES = {0: "PG", 1: "SG", 2: "SF", 3: "PF", 4: "C"}
URL = "https://lm-api-reads.fantasy.espn.com/apis/v3/games/fba/seasons/{season}/segments/0/leaguedefaults/3"


def fetch(season: int, limit: int = 900) -> dict:
    import requests

    filt = {"players": {"filterStatus": {"value": ["FREEAGENT", "WAIVERS", "ONTEAM"]}, "limit": limit,
                        "sortDraftRanks": {"sortPriority": 100, "sortAsc": True, "value": "STANDARD"}}}
    r = requests.get(URL.format(season=season), params={"view": "kona_player_info"},
                     headers={"x-fantasy-filter": json.dumps(filt), "User-Agent": "Mozilla/5.0"}, timeout=60)
    r.raise_for_status()
    return r.json()


def parse_players(payload: dict) -> pd.DataFrame:
    """ESPN payload -> DataFrame[espn_name, eligibility, espn_adp]."""
    rows = []
    for item in payload.get("players", []):
        p = item.get("player", {})
        slots = [SLOT_NAMES[s] for s in p.get("eligibleSlots", []) if s in SLOT_NAMES]
        if not slots and p.get("defaultPositionId") in (1, 2, 3, 4, 5):
            slots = [SLOT_NAMES[p["defaultPositionId"] - 1]]
        if not p.get("fullName") or not slots:
            continue
        adp = (p.get("ownership") or {}).get("averageDraftPosition")
        rows.append({"espn_name": p["fullName"], "eligibility": ",".join(slots),
                     "espn_adp": adp if adp and adp > 0 else None})
    return pd.DataFrame(rows, columns=["espn_name", "eligibility", "espn_adp"])


def match_to_players(espn: pd.DataFrame, players: pd.DataFrame) -> pd.DataFrame:
    """Join on normalised name. players needs player_id, name."""
    players = players.assign(key=players["name"].map(normalize_name))
    espn = espn.assign(key=espn["espn_name"].map(normalize_name)).drop_duplicates("key")
    dup = players["key"].duplicated(keep=False)
    m = players[~dup].merge(espn, on="key", how="inner")
    return m[["player_id", "name", "eligibility", "espn_adp"]].sort_values("espn_adp", na_position="last")


def main(season: int, out: str, from_json: str | None, db_path: str) -> None:
    payload = json.loads(Path(from_json).read_text(encoding="utf-8")) if from_json else fetch(season)
    espn = parse_players(payload)
    print(f"ESPN returned {len(espn)} players with eligibility")
    conn = sqlite3.connect(db_path)
    players = pd.read_sql_query("SELECT player_id, name FROM player WHERE last_year >= ?", conn, params=[season - 1])
    m = match_to_players(espn, players)
    proj_path = ROOT / "data/processed" / f"projections_{season - 1}.parquet"
    if proj_path.exists():  # report unmatched among players we actually project highly
        top = pd.read_parquet(proj_path).head(200)
        missing = top[~top["player_id"].isin(m["player_id"])]
        if len(missing):
            print(f"WARNING: {len(missing)} of our top-200 players had no ESPN match (heuristic eligibility is used for them):")
            print("  " + ", ".join(missing["name"].head(25)))
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    m.to_csv(out, index=False)
    print(f"Wrote {len(m)} players -> {out}\nSample:")
    print(m.head(12).to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--season", type=int, default=2027, help="ESPN season id = the year the season ENDS (2027 = 2026-27)")
    ap.add_argument("--out", default=str(ROOT / "data/external/positions.csv"))
    ap.add_argument("--from-json", default=None, help="parse a saved API response instead of fetching")
    ap.add_argument("--db-path", default=str(ROOT / "data/processed/nba.db"))
    a = ap.parse_args()
    main(a.season, a.out, a.from_json, a.db_path)
