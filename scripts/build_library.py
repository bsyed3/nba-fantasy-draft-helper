"""
Phase 2: calibrate the simulation from history and build the weekly
simulation library for a projection set.

    python scripts/build_library.py                  # latest projections_<year>.parquet
    python scripts/build_library.py --year 2026 --sims-per-week 60

Needs the outputs of scripts/build_projections.py (projections, backtest
residuals, rookie profiles). Writes data/processed/calibration.json and
library_<year>.npz (~6 MB on disk, ~62 MB in memory for 614 players x ~720 simulated matchups).
The simulated matchups are the real matchup periods of the season's schedule (the published
future schedule when `scripts/ingest.py` has loaded it) -- Mon-Sun weeks, with the opening and
All-Star stretches as two-week periods (2026-27: week 17 = Feb 15-28) -- played game by game.
"""
from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding="utf-8")

from src.simulation.calibration import Calibration, calibrate
from src.simulation.copula import build_library
from src.simulation.schedule import season_periods


def latest_year(out: Path) -> int:
    years = [int(p.stem.split("_")[1]) for p in out.glob("projections_*.parquet")]
    if not years:
        raise SystemExit("No projections found -- run scripts/build_projections.py first.")
    return max(years)


def main(db_path: str, year: int | None, sims: int, seed: int, out_dir: str, playoffs: bool, fill_cup: bool = False) -> None:
    out = Path(out_dir)
    year = year or latest_year(out)
    conn = sqlite3.connect(db_path)

    print("Calibrating from game logs + backtest residuals...")
    oos = pd.read_parquet(out / "backtest_oos.parquet")
    cal = calibrate(conn, oos, out / "rookie_profiles.csv")
    cal.save(out / "calibration.json")
    print(f"  NB dispersion r: { {k: round(v, 1) for k, v in cal.dispersion.items()} }")
    print(f"  shooting rho/game: { {k: round(v, 4) for k, v in cal.rho.items()} }")
    print(f"  games/week pmf: { {k: round(v, 3) for k, v in cal.games_pmf.items()} }, whole-week-absence share c={cal.absence_c:.2f}")

    proj = pd.read_parquet(out / f"projections_{year}.parquet")
    season = f"{year}-{str(year + 1)[-2:]}"
    sched = season_periods(conn, season, n_regular=18, n_playoff=3 if playoffs else 0, fill_unscheduled=fill_cup)
    spans = ", ".join(f"week {i + 1}: {s:%b %d}-{e:%b %d}" for i, (s, e) in enumerate(sched.spans) if (e - s).days > 7)
    print(f"Matchup periods: {sched.n_weeks} ({'18 regular + 3 playoff' if playoffs else '18 regular-season'}); two-week periods -> {spans}")
    if sched.missing:
        n_missing = sum(sched.missing.values()) // 2
        if sched.flex_period is not None:
            wd = ", ".join(f"{d} {w:.0%}" for d, w in zip(["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"], sched.flex_weights) if w > 0.005)
            print(f"  note: {n_missing} NBA Cup games are not yet on the published schedule ({set(sched.missing.values())} missing per team); "
                  f"placed on random days of week {sched.flex_period + 1} in every simulated matchup (past Cup weeks' weekday pattern: {wd})")
        else:
            print(f"  note: {n_missing} NBA Cup games are not yet on the published schedule; they are ignored -- week 7 uses only the "
                  f"published games (pass --fill-cup-games to place the missing ones at random instead)")
    print(f"Simulating {sched.n_weeks} real {season} matchup periods x {sims} draws x {len(proj)} players, game by game...")
    lib = build_library(proj, cal, sched, sims_per_week=sims, seed=seed)
    lib.save(out / f"library_{year}.npz")
    print(f"  saved {out / f'library_{year}.npz'} ({lib.stats.nbytes / 1e6:.0f} MB uncompressed)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db-path", default=str(ROOT / "data/processed/nba.db"))
    ap.add_argument("--year", type=int, default=None)
    ap.add_argument("--sims-per-week", dest="sims", type=int, default=40, help="draws per real matchup period")
    ap.add_argument("--include-playoffs", action="store_true", help="also simulate the 3 playoff matchup periods")
    ap.add_argument("--fill-cup-games", action="store_true",
                    help="place each team's not-yet-scheduled NBA Cup games on random days of the Cup week (default: ignore them)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", default=str(ROOT / "data/processed"))
    a = ap.parse_args()
    main(a.db_path, a.year, a.sims, a.seed, a.out_dir, a.include_playoffs, a.fill_cup_games)
