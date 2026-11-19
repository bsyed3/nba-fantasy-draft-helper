"""
Pull this season's box scores so far from nba_api and store them, with each game's fantasy matchup week, in data/processed/live_<year>.db.

    python scripts/fetch_live_stats.py                    # 2026-27 to date (one pull of every game played; re-run any time, it replaces the tables)
    python scripts/fetch_live_stats.py --season 2026-27 --week 1       # also print that week's leaders

The live games are deliberately NOT written to nba.db (see src/data_ingestion/live_stats.py): the projection pipeline treats nba.db's latest
season with box scores as history, so they would corrupt the next rebuild. Nothing here changes nba.db.
Run it locally: nba_api calls stats.nba.com directly.
"""
from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(ROOT))

from src.data_ingestion import live_stats as L
from src.data_ingestion import nba_api_client as nba
from src.simulation.schedule import season_periods


def main(season: str, live_db: str, nba_db: str, week: int | None) -> None:
    year = int(season[:4])
    games = nba.get_season_games(season)
    print(f"[{season}] {len(games)} games played ({games['date'].min()} .. {games['date'].max()})")
    box = nba.get_season_game_logs(season)
    box = box[box["game_id"].isin(set(games["game_id"]))]                 # same filter as ingest.py: regular-season games only
    print(f"[{season}] {len(box)} player-games, {box['player_id'].nunique()} players")

    ro = sqlite3.connect(f"file:{nba_db}?mode=ro", uri=True)               # read-only: the schedule gives the fantasy week boundaries
    spans = season_periods(ro, season, n_regular=18, n_playoff=3).spans
    scheduled = pd.read_sql_query("SELECT game_id, date FROM game WHERE season = ? AND game_type = 'regular'", ro, params=(season,))
    ro.close()

    Path(live_db).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(live_db)
    res = L.save_live(conn, games, box, spans, season)
    print(f"saved {res['games']} games / {res['player_games']} player-games -> {live_db}")

    # completeness: every game the published schedule has on or before the last date pulled should be there (games still being played are the exception)
    due = scheduled[scheduled["date"] <= res["last_date"]]
    missing = sorted(set(due["game_id"]) - set(games["game_id"]))
    print(f"schedule check: {len(due)} games scheduled through {res['last_date']}, {len(due) - len(missing)} pulled" +
          (f"; not yet in the logs: {', '.join(missing)}" if missing else " (complete)"))

    wk = pd.read_sql_query("SELECT g.week, COUNT(DISTINCT g.game_id) AS games, MIN(g.date) AS first, MAX(g.date) AS last FROM live_game g GROUP BY g.week ORDER BY g.week", conn)
    print("\nby fantasy week:\n" + wk.to_string(index=False))
    if week is not None:
        lead = pd.read_sql_query(
            "SELECT name, team_id, games, ROUND(min / games, 1) AS mpg, pts, tpm, reb, ast, stl, blk, tov FROM live_player_week WHERE week = ? ORDER BY pts DESC LIMIT 15",
            conn, params=(week,))
        print(f"\nweek {week} points leaders:\n" + lead.to_string(index=False))
    conn.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--season", default="2026-27")
    ap.add_argument("--live-db", default=None, help="default data/processed/live_<year>.db")
    ap.add_argument("--nba-db", default=str(ROOT / "data/processed/nba.db"))
    ap.add_argument("--week", type=int, default=None, help="also print this fantasy week's points leaders")
    a = ap.parse_args()
    main(a.season, a.live_db or str(ROOT / f"data/processed/live_{a.season[:4]}.db"), a.nba_db, a.week)
