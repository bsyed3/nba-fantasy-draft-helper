"""
Pull NBA data from nba_api into the local SQLite database.

Usage:
    python scripts/ingest.py                         # default: 2015-16 .. 2025-26
    python scripts/ingest.py --seasons 2023-24:2025-26
    python scripts/ingest.py --seasons 2025-26 --refresh   # re-pull a season already loaded
    python scripts/ingest.py --schedule 2026-27            # (default: the season after the latest) load the published schedule

Per season: games, every player-game box score, player ages (3 API calls).
Once, up front: team list + the PlayerIndex bio table (position, draft slot,
rookie year -- includes the incoming rookie class). Stints are rebuilt from
scratch at the end, so re-running is always safe.

Seasons already present in the database are skipped unless --refresh.
Run it locally: nba_api calls stats.nba.com directly.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.stdout.reconfigure(encoding="utf-8")  # player names have accents; Windows consoles default to cp1252
sys.path.insert(0, str(ROOT))

from src.data_ingestion import load_to_db as db
from src.data_ingestion import nba_api_client as nba

DEFAULT_SEASONS = "2015-16:2025-26"


def season_label(start_year: int) -> str:
    return f"{start_year}-{str(start_year + 1)[-2:]}"


def parse_seasons(spec: str) -> list[str]:
    if ":" not in spec:
        return [s.strip() for s in spec.split(",")]
    first, last = spec.split(":")
    return [season_label(y) for y in range(int(first[:4]), int(last[:4]) + 1)]


def ingest_season(conn, season: str) -> None:
    print(f"[{season}] games...")
    games_df = nba.get_season_games(season)
    db.load_games(conn, games_df)
    time.sleep(nba.REQUEST_DELAY_SECONDS)

    print(f"[{season}] box scores...")
    box_df = nba.get_season_game_logs(season)
    # Safety net: any player missing from PlayerIndex still needs a row for the FK.
    db.load_players(conn, box_df[["player_id", "name"]].drop_duplicates("player_id"))
    box_df = box_df[box_df["game_id"].isin(games_df["game_id"])]
    db.load_box_scores(conn, box_df)
    print(f"[{season}]   {len(games_df)} games, {len(box_df)} player-games")
    time.sleep(nba.REQUEST_DELAY_SECONDS)

    print(f"[{season}] ages...")
    db.load_player_seasons(conn, nba.get_season_player_ages(season))
    time.sleep(nba.REQUEST_DELAY_SECONDS)


def ingest_schedule(conn, season: str) -> None:
    """Future games (no box scores) -- the simulation plays players on the real schedule."""
    sched = nba.get_season_schedule(season)
    conn.execute("DELETE FROM game WHERE season = ? AND game_id NOT IN (SELECT game_id FROM box_score)", (season,))
    db.load_games(conn, sched)
    print(f"[{season}] schedule: {len(sched)} games ({sched['date'].min()} .. {sched['date'].max()})")


def main(seasons: list[str], db_path: str, refresh: bool, schedule: str | None) -> None:
    conn = db.get_connection(db_path)
    db.init_schema(conn, str(ROOT / "schema.sql"))

    print("Teams + PlayerIndex...")
    db.load_teams(conn, nba.get_teams())
    index_df = nba.get_player_index()
    db.upsert_player_index(conn, index_df)
    print(f"  {len(index_df)} players in index")

    loaded = {r[0] for r in conn.execute("SELECT DISTINCT season FROM game")}
    for season in seasons:
        if season in loaded and not refresh:
            print(f"[{season}] already loaded, skipping (use --refresh to re-pull)")
            continue
        ingest_season(conn, season)

    if schedule is None:  # default: the season after the latest one pulled
        schedule = season_label(int(seasons[-1][:4]) + 1)
    try:
        ingest_schedule(conn, schedule)
    except Exception as exc:  # the schedule may not be published yet
        print(f"[{schedule}] schedule not loaded ({exc.__class__.__name__}: {str(exc)[:80]})")

    n = db.rebuild_player_stints(conn)
    print(f"Rebuilt {n} player stints")
    print(f"Done -- {db_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seasons", default=DEFAULT_SEASONS, help='"2015-16:2025-26" or comma list')
    parser.add_argument("--db-path", default=str(ROOT / "data/processed/nba.db"))
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--schedule", default=None, help='season whose published schedule to load, e.g. "2026-27"')
    args = parser.parse_args()
    main(parse_seasons(args.seasons), args.db_path, args.refresh, args.schedule)
