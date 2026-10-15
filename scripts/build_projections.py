"""
Phase 1: fit + backtest the projection models and write next-season
per-game projections.

    python scripts/build_projections.py                 # target = season after the latest in the DB
    python scripts/build_projections.py --target-year 2026

Outputs (data/processed/):
    projections_<year>.parquet / .csv   one row per player (avail = MEAN availability; gp_p25/gp_median/gp_p75 = games-played range, display only)
    gp_quantiles.json                   the mean -> P25/median/P75 games-played mapping behind those columns
    backtest_summary.csv                stat x model out-of-sample error
    backtest_oos.parquet                out-of-sample per-game predictions vs actuals (Phase 2 input)
    rookie_profiles.csv                 draft-tier means + coefficients of variation

Optional hand-curated overrides: data/external/projection_overrides.csv
with a `player_id` or `name` column plus any of
mpg, avail, fga, fgm, tpa, tpm, fta, ftm, reb, ast, stl, blk, tov
(per-game values, e.g. an ESPN/Hashtag projection for a rookie).

Optional known absences: data/external/known_absences.csv (generated from ESPN by scripts/fetch_espn_injuries.py -- overwritten on each
run) and data/external/known_absences_manual.csv (hand-edited, applied after it so it wins, never overwritten), each with a `player_id`
or `name` column and `games_out` (games he is already known to miss at the start of the season; 82 = out for the season) -- see
known_absences.example.csv. Everyone else is projected as healthy at tip-off; the model's availability is for in-season games after
his first appearance.

Optional exact ESPN eligibility: data/external/positions.csv with a `player_id`
or `name` column and `eligibility` like "PG,SG" (see src/features/positions.py).
"""
from __future__ import annotations

import argparse
import json
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
from src.features.team_context import known_absent_ids, team_context
from src.models import projection as P

OVERRIDE_COLS = ["mpg", "avail", "fga", "fgm", "tpa", "tpm", "fta", "ftm", "reb", "ast", "stl", "blk", "tov"]
COUNT_COLS = [c for c in OVERRIDE_COLS if c not in ("mpg", "avail")]      # per-game totals that scale with minutes


def apply_overrides(proj: pd.DataFrame, path: Path) -> pd.DataFrame:
    """
    Hand-set values from data/external/projection_overrides.csv (player_id or name, plus any of OVERRIDE_COLS, per game).
    Overriding `mpg` also rescales every per-game total that the row does not set itself, keeping his per-minute rates: the
    simulation and the board read the per-game totals, so changing only the minutes would change nothing.
    """
    if not path.exists():
        return proj
    ov = pd.read_csv(path)
    proj = proj.copy()
    key = "player_id" if "player_id" in ov.columns else "name"
    lookup = proj[key].str.lower() if key == "name" else proj[key]
    n = 0
    touched = []
    for _, row in ov.iterrows():
        k = str(row[key]).lower() if key == "name" else row[key]
        idx = proj.index[lookup == k]
        if len(idx) != 1:
            print(f"  override skipped (matched {len(idx)} players): {row[key]}")
            continue
        old_mpg = float(proj.loc[idx[0], "mpg"])
        if "mpg" in ov.columns and pd.notna(row["mpg"]) and old_mpg > 0:
            scale = float(row["mpg"]) / old_mpg
            for c in COUNT_COLS:
                if not (c in ov.columns and pd.notna(row[c])):
                    proj.loc[idx, c] = proj.loc[idx, c] * scale
        for c in OVERRIDE_COLS:
            if c in ov.columns and pd.notna(row[c]):
                proj.loc[idx, c] = row[c]
        proj.loc[idx, "source"] = "override"
        touched.extend(idx)
        n += 1
    P.sync_rates_from_per_game(proj, touched)         # the simulation reads fga2 / fga3 / r_p2 / r_p3 / r_ft, not fga / tpm / ftm
    proj["pts"] = 2 * proj["fgm"] + proj["tpm"] + proj["ftm"]
    proj["fg_pct"] = proj["fgm"] / proj["fga"].where(proj["fga"] > 0)
    proj["ft_pct"] = proj["ftm"] / proj["fta"].where(proj["fta"] > 0)
    print(f"  applied {n} override rows")
    return proj


ABSENCE_FILES = [ROOT / "data/external/known_absences.csv",            # generated from ESPN (fetch_espn_injuries.py)
                 ROOT / "data/external/known_absences_manual.csv"]    # hand-edited; wins, never overwritten


def reapply_overrides(out: Path, target_year: int) -> None:
    """
    Re-apply data/external/projection_overrides.csv to the SAVED projections and refresh the value scores and ranks, without
    re-running the backtest. Overrides are the last thing the projection step does to a player's stats, so this gives the same
    table as a full run (and applying the same file twice changes nothing). Rebuild the simulation library afterwards.
    """
    path = out / f"projections_{target_year}.parquet"
    proj = pd.read_parquet(path)
    proj = apply_overrides(proj, ROOT / "data/external/projection_overrides.csv")
    proj = proj.drop(columns=[c for c in proj.columns if c.startswith("z_") or c == "rank"])      # old value scores and ranks
    proj = pd.concat([proj, z_scores(proj)], axis=1)
    proj = proj.sort_values("z_total", ascending=False).reset_index(drop=True)
    proj["rank"] = proj.index + 1
    proj.to_parquet(path)
    proj.to_csv(out / f"projections_{target_year}.csv", index=False)
    print(f"re-applied overrides and re-ranked {len(proj)} players -> {path}")


def main(db_path: str, target_year: int | None, folds: int, out_dir: str, team_ctx: bool = True, overrides_only: bool = False) -> None:
    out = Path(out_dir)
    if overrides_only:
        year = target_year or max(int(p.stem.split("_")[1]) for p in out.glob("projections_*.parquet"))
        return reapply_overrides(out, year)
    conn = sqlite3.connect(db_path)
    st = load_player_season_table(conn)
    players = pd.read_sql_query(
        "SELECT player_id, name, position_1, position_2, draft_year, draft_position, rookie_year, last_year FROM player", conn)
    target_year = target_year or int(st["year"].max()) + 1
    fold_years = list(range(target_year - folds, target_year))
    ctx = None
    if team_ctx:
        # who left / joined each player's team (vacated and incoming minutes and possessions); the projected season uses today's rosters
        # minus anyone already known to miss the opening stretch, whose minutes are up for grabs
        cur_team = pd.read_sql_query("SELECT player_id, team_id FROM player WHERE last_year >= ?", conn, params=(target_year,)).set_index("player_id")["team_id"]
        ctx = team_context(conn, list(range(int(st["year"].min()) + 1, target_year + 1)), current_team=cur_team,
                           absent_ids=known_absent_ids(ABSENCE_FILES, players))
        print(f"Team context: {len(ctx)} player-seasons ({int((ctx['year'] == target_year).sum())} for {target_year})")
    print(f"Backtesting folds {fold_years} (train on everything before each)...")
    bt = P.backtest(st, fold_years, context=ctx)
    pd.set_option("display.width", 200)
    print("\nOut-of-sample change in weighted MSE vs the shrunken-average baseline (negative = better):")
    print(bt.summary.pivot(index="stat", columns="model", values="vs_baseline_pct").round(1).to_string())
    print("\nSelected per stat:", bt.selection)
    bt.summary.to_csv(out / "backtest_summary.csv", index=False)
    bt.oos.to_parquet(out / "backtest_oos.parquet")
    gp_q = P.fit_gp_quantiles(bt.avail_eval)
    (out / "gp_quantiles.json").write_text(json.dumps({str(q): {"intercept": a, "slope": b} for q, (a, b) in gp_q.items()}, indent=2))
    print("Typical-GP map (actual availability = a + b * projected mean availability, rotation players, backtest):",
          {q: (round(a, 2), round(b, 2)) for q, (a, b) in gp_q.items()})

    means, cvs = P.rookie_profiles(st)
    rp = means.add_prefix("mean_").join(cvs.add_prefix("cv_"))
    rp.to_csv(out / "rookie_profiles.csv")
    print("\nRookie draft-tier profiles:")
    print(means[["n", "mpg", "avail"]].round(2).to_string())

    print(f"\nProjecting {target_year}-{str(target_year + 1)[-2:]}...")
    ignore = None
    ig_path = ROOT / "data/external/ignore_seasons.csv"            # seasons to treat as if they never happened (e.g. rested to force a trade)
    if ig_path.exists():
        ignore = pd.read_csv(ig_path)
        print(f"  ignoring {len(ignore)} season(s) in the projected players' histories: " + ", ".join(f"{r['name']} {int(r['year'])}-{str(int(r['year']) + 1)[-2:]}" for _, r in ignore.iterrows()))
    no_ctx = None
    nc_path = ROOT / "data/external/no_team_context.csv"           # players whose team change should not move their projection
    if nc_path.exists() and ctx is not None:
        nc = pd.read_csv(nc_path)
        no_ctx = set(int(p) for p in nc["player_id"])
        print(f"  no team-context adjustment for: " + ", ".join(nc["name"]))
    proj = P.project_season(st, players, target_year, bt.selection, context=ctx, ignore_seasons=ignore, no_context_ids=no_ctx)
    rk_path = ROOT / "data/external/rookie_projections_yahoo.csv"   # outside rookie projections replace the draft-slot profile
    if rk_path.exists():
        proj = P.apply_external_rookies(proj, pd.read_csv(rk_path))
    proj = apply_overrides(proj, ROOT / "data/external/projection_overrides.csv")
    proj = P.apply_known_absences(proj, ABSENCE_FILES)
    q = P.apply_gp_quantiles(proj["avail_active"], gp_q).mul((82 - proj["games_out"]).clip(lower=0) / 82, axis=0)
    proj = pd.concat([proj, q], axis=1)                                          # display only: P25 / median / P75 games played
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
    ap.add_argument("--no-team-context", action="store_true", help="project from each player's own history only (the model before team context)")
    ap.add_argument("--overrides-only", action="store_true",
                    help="only re-apply data/external/projection_overrides.csv to the saved projections and re-rank (seconds; no backtest)")
    a = ap.parse_args()
    main(a.db_path, a.target_year, a.folds, a.out_dir, team_ctx=not a.no_team_context, overrides_only=a.overrides_only)
