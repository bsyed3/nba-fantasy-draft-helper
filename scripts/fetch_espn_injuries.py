"""
Turn ESPN's NBA injuries page into data/external/known_absences.csv: the players labelled Out and how many games they are
expected to miss at the start of the season (82 = out for the season). Re-run it as injuries change, then rebuild:

    python scripts/fetch_espn_injuries.py                          # download https://www.espn.com/nba/injuries (needs a network that can reach ESPN)
    python scripts/fetch_espn_injuries.py --html espn_injuries.html     # parse a saved copy of the page (File > Save page as...)
    python scripts/fetch_espn_injuries.py --json rows.json              # rows [{team, name, pos, est_return, status, comment}, ...]
    python scripts/fetch_espn_injuries.py --status Out "Day-To-Day"     # also include day-to-day listings (default: Out only)

    python scripts/build_projections.py && python scripts/build_library.py

Games out = the team's regular-season games before the estimated return date (schedule table) scaled to 82. Hand-edit the CSV
afterwards if you know better (a listing with "re-evaluated in four weeks" is only a guess); anything you add keeps working.
The model projects everyone else as healthy at tip-off.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sqlite3
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding="utf-8")

from src.data_ingestion import espn_injuries as E


def main(db_path: str, html: str | None, rows_json: str | None, statuses: list[str], season: str, today: dt.date, out: str) -> None:
    if rows_json:
        listings = pd.DataFrame(json.loads(Path(rows_json).read_text(encoding="utf-8")), columns=E.COLUMNS)
    else:
        text = Path(html).read_text(encoding="utf-8") if html else E.fetch_html()
        listings = E.parse_html(text)
    print(f"{len(listings)} listings; statuses: {listings['status'].value_counts().to_dict()}")
    conn = sqlite3.connect(db_path)
    table, warnings = E.known_absences(listings, conn, season=season, today=today, statuses=tuple(statuses))
    table = table.sort_values("games_out", ascending=False).reset_index(drop=True)
    for w in warnings:
        print("  warning:", w)
    pd.set_option("display.width", 200)
    print(table.to_string(index=False))
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out, index=False)
    print(f"\nwrote {len(table)} known absences -> {out}\nNext: python scripts/build_projections.py && python scripts/build_library.py")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db-path", default=str(ROOT / "data/processed/nba.db"))
    ap.add_argument("--html", default=None, help="saved copy of the ESPN injuries page")
    ap.add_argument("--json", dest="rows_json", default=None, help="listings as a JSON array instead of HTML")
    ap.add_argument("--status", nargs="+", default=["Out"], help="statuses to keep (default: Out)")
    ap.add_argument("--season", default="2026-27")
    ap.add_argument("--today", type=dt.date.fromisoformat, default=dt.date.today(), help="YYYY-MM-DD; dates on the page are resolved relative to it")
    ap.add_argument("--out", default=str(ROOT / "data/external/known_absences.csv"))
    a = ap.parse_args()
    main(a.db_path, a.html, a.rows_json, a.status, a.season, a.today, a.out)
