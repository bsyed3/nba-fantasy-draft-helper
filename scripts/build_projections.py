"""
Phase 1: fit + backtest the projection models and write next-season
per-game projections.

    python scripts/build_projections.py                 # target = season after the latest in the DB
    python scripts/build_projections.py --target-year 2026

Outputs (data/processed/):
    projections_<year>.parquet / .csv   one row per player
    backtest_summary.csv                stat x model out-of-sample error
    backtest_oos.parquet                out-of-sample per-game predictions vs actuals (Phase 2 input)
    rookie_profiles.csv                 draft-tier means + coefficients of variation

Optional hand-curated overrides: data/external/projection_overrides.csv
with a `player_id` or `name` column plus any of
mpg, avail, fga, fgm, tpa, tpm, fta, ftm, reb, ast, stl, blk, tov
(per-game values, e.g. an ESPN/Hashtag projection for a rookie).

Optional exact ESPN eligibility: data/external/positions.csv with a `player_id`
or `name` column and `eligibility` like "PG,SG" (see src/features/positions.py).
"""
from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.stdout.reconfigure(encoding="utf-8")  # player names have accents; Windows consoles default to cp1252
sys.path.insert(0, str(ROOT))

from src.draft.values import z_scores
from src.features.player_seasons import load_player_season_table
from src.features.positions import add_eligibility
from src.models import projection as P

OVERRIDE_COLS = ["mpg", "avail", "fga", "fgm", "tpa", "tpm", "fta", "ftm", "reb", "ast", "stl", "blk", "tov"]


def apply_overrides(proj: pd.DataFrame, path: Path) -> pd.DataFrame:
    if not path.exists():
        return proj
    ov = pd.read_csv(path)
    proj = proj.copy()
    key = "player_id" if "player_id" in ov.columns else "name"
    lookup = proj[key].str.lower() if key == "name" else proj[key]
    n = 0
    for _, row in ov.iterrows():
        k = str(row[key]).lower() if key == "name" else row[key]
        idx = proj.index[lookup == k]
        if len(idx) != 1:
            print(f"  override skipped (matched {len(idx)} players): {row[key]}")
            continue
        for c in OVERRIDE_COLS:
            if c in ov.columns and pd.notna(row[c]):
                proj.loc[idx, c] = row[c]
        proj.loc[idx, "source"] = "override"
        n += 1
    proj["pts"] = 2 * proj["fgm"] + proj["tpm"] + proj["ftm"]
    proj["fg_pct"] = proj["fgm"] / proj["fga"].where(proj["fga"] > 0)
    proj["ft_pct"] = proj["ftm"] / proj["fta"].where(proj["fta"] > 0)
    print(f"  applied {n} override rows")
    return proj


def main(db_path: str, target_year: int | None, folds: int, out_dir: str) -> None:
    out = Path(out_dir)
    conn = sqlite3.connect(db_path)
    st = load_player_season_table(conn)
    players = pd.read_sql_query(
        "SELECT player_id, name, position_1, position_2, draft_year, draft_position, rookie_year, last_year FROM player", conn)
    target_year = target_year or int(st["year"].max()) + 1
    fold_years = list(range(target_year - folds, target_year))
    print(f"Backtesting folds {fold_years} (train on everything before each)...")
    bt = P.backtest(st, fold_years)
    pd.set_option("display.width", 200)
    print("\nOut-of-sample change in weighted MSE vs the shrunken-average baseline (negative = better):")
    print(bt.summary.pivot(index="stat", columns="model", values="vs_baseline_pct").round(1).to_string())
    print("\nSelected per stat:", bt.selection)
    bt.summary.to_csv(out / "backtest_summary.csv", index=False)
    bt.oos.to_parquet(out / "backtest_oos.parquet")

    means, cvs = P.rookie_profiles(st)
    rp = means.add_prefix("mean_").join(cvs.add_prefix("cv_"))
    rp.to_csv(out / "rookie_profiles.csv")
    print("\nRookie draft-tier profiles:")
    print(means[["n", "mpg", "avail"]].round(2).to_string())

    print(f"\nProjecting {target_year}-{str(target_year + 1)[-2:]}...")
    proj = P.project_season(st, players, target_year, bt.selection)
    proj = apply_overrides(proj, ROOT / "data/external/projection_overrides.csv")
    teams = pd.read_sql_query("SELECT player_id, team_id FROM player", conn)  # CURRENT team (PlayerIndex)
    proj = proj.merge(teams, on="player_id", how="left")
    proj = add_eligibility(proj, ROOT / "data/external/positions.csv")
    z = z_scores(proj)
    proj = pd.concat([proj, z], axis=1).sort_values("z_total", ascending=False).reset_index(drop=True)
    proj["rank"] = proj.index + 1
    proj.to_parquet(out / f"projections_{target_year}.parquet")
    proj.to_csv(out / f"projections_{target_year}.csv", index=False)
    cols = ["rank", "name", "elig", "age_t", "source", "mpg", "avail", "pts", "tpm", "reb", "ast",
            "stl", "blk", "fg_pct", "ft_pct", "tov", "z_total"]
    print(proj[cols].head(40).round(2).to_string(index=False))
    print(f"\n{len(proj)} players projected -> {out}/projections_{target_year}.parquet")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db-path", default=str(ROOT / "data/processed/nba.db"))
    ap.add_argument("--target-year", type=int, default=None, help="start year of the season to project (2026 = 2026-27)")
    ap.add_argument("--folds", type=int, default=5, help="number of backtest seasons")
    ap.add_argument("--out-dir", default=str(ROOT / "data/processed"))
    a = ap.parse_args()
    main(a.db_path, a.target_year, a.folds, a.out_dir)
