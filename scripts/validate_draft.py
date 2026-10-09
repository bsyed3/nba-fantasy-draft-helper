"""
Out-of-sample validation of the whole pipeline on past seasons.

For each season Y:
  1. project Y using ONLY seasons before Y (Phase 1),
  2. build the simulation library from those projections (Phase 2),
  3. run mock drafts with noisy bots, drafting my team two ways with the
     same bots: (a) the recommendation engine, (b) best available by
     static rank,
  4. score the final rosters on what REALLY happened: season Y's actual
     day-by-day box scores through the same daily-lineup / position-slot /
     category / matchup rules (real weeks are the only "simulations").

The headline number is the paired difference in actual weekly matchup
win probability, engine minus best-available.

    python scripts/validate_draft.py --years 2023 2024 2025 --drafts 20
"""
from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding="utf-8")

from src.draft.engine import DraftEngine
from src.draft.mock import best_available, engine_pick, run_mock_draft
from src.draft.state import LeagueSettings
from src.draft.values import z_scores
from src.features.player_seasons import load_player_season_table
from src.features.positions import add_eligibility
from src.models import projection as P
from src.simulation.calibration import Calibration
from src.simulation.copula import build_library
from src.simulation.schedule import actual_library, season_periods


def team_at_season_start(conn, season: str) -> pd.Series:
    """player_id -> NBA team of his first game that season (the team he was drafted-for)."""
    df = pd.read_sql_query(
        """SELECT bs.player_id, bs.team_id, g.date FROM box_score bs JOIN game g ON g.game_id = bs.game_id
           WHERE g.season = ? AND g.game_type = 'regular' AND bs.min > 0""", conn, params=[season])
    return df.sort_values("date").drop_duplicates("player_id").set_index("player_id")["team_id"]


def validate_year(conn, st_all, players, cal, year: int, n_drafts: int, sims_per_week: int, settings_kw: dict, top_k: int, rollouts: int,
                  margin: float = 0.0):
    season = f"{year}-{str(year + 1)[-2:]}"
    st = st_all[st_all["year"] < year]
    bt = P.backtest(st, folds=list(range(year - 3, year)), verbose=False)
    proj = P.project_season(st, players, year, bt.selection)
    proj["team_id"] = proj["player_id"].map(team_at_season_start(conn, season))
    proj = add_eligibility(proj)  # heuristic eligibility for every strategy alike
    proj = pd.concat([proj, z_scores(proj)], axis=1)
    sched = season_periods(conn, season)   # 18 regular-season matchup periods (All-Star stretch = one 2-week matchup)
    lib = build_library(proj, cal, sched, sims_per_week=sims_per_week, seed=year)
    actual = actual_library(conn, season, lib.player_ids, sched)   # what REALLY happened, day by day

    rows = []
    for d in range(n_drafts):
        slot = d % settings_kw["n_teams"] + 1
        s = LeagueSettings(my_slot=slot, season=season, **settings_kw)
        eng = DraftEngine(lib, proj, s, seed=d)
        truth = DraftEngine(actual, proj, s)
        res = {}
        for name, strat in (("engine", engine_pick(top_k, rollouts, margin)), ("static", best_available)):
            rosters = run_mock_draft(eng, s, strat, bot_seed=1000 * year + d)
            sim_ev = eng.evaluate(rosters)          # what the model believed
            real_ev = truth.evaluate(rosters)       # what actually happened
            res[name] = (sim_ev.win_prob[slot - 1], real_ev.win_prob[slot - 1])
        rows.append(dict(year=year, draft=d, slot=slot, sim_engine=res["engine"][0], sim_static=res["static"][0],
                         real_engine=res["engine"][1], real_static=res["static"][1]))
        print(f"  {season} draft {d + 1}/{n_drafts} slot {slot}: real win% engine {res['engine'][1]:.3f} vs static {res['static'][1]:.3f}")
    return pd.DataFrame(rows)


def main(years, n_drafts, sims_per_week, top_k, rollouts, out_dir, margin=0.0, tag=""):
    out = Path(out_dir)
    conn = sqlite3.connect(out / "nba.db")
    st_all = load_player_season_table(conn)
    players = pd.read_sql_query(
        "SELECT player_id, name, position_1, position_2, draft_year, draft_position, rookie_year, last_year FROM player", conn)
    cal = Calibration.load(out / "calibration.json")
    kw = dict(n_teams=10, roster_size=13, active_slots=10, max_centers=3)
    frames = [validate_year(conn, st_all, players, cal, y, n_drafts, sims_per_week, kw, top_k, rollouts, margin) for y in years]
    res = pd.concat(frames, ignore_index=True)
    res["gain_sim"] = res["sim_engine"] - res["sim_static"]
    res["gain_real"] = res["real_engine"] - res["real_static"]
    res.to_csv(out / (("validation_results" if margin == 0 else f"validation_results_margin{margin:g}") + tag + ".csv"), index=False)

    def summ(g):
        n = len(g)
        return pd.Series({
            "drafts": n,
            "real_win_engine": g["real_engine"].mean(), "real_win_static": g["real_static"].mean(),
            "real_gain": g["gain_real"].mean(), "real_gain_se": g["gain_real"].std(ddof=1) / np.sqrt(n),
            "engine_wins_pct": (g["gain_real"] > 0).mean(),
            "sim_gain": g["gain_sim"].mean(),
        })
    table = res.groupby("year").apply(summ, include_groups=False)
    table.loc["ALL"] = summ(res)
    print("\nActual-weekly-results validation (weekly matchup win probability):")
    print(table.round(4).to_string())
    print(f"\nWrote {out / 'validation_results.csv'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--years", type=int, nargs="+", default=[2023, 2024, 2025], help="start years of seasons to validate on")
    ap.add_argument("--drafts", type=int, default=20)
    ap.add_argument("--sims-per-week", dest="sims", type=int, default=30)
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument("--rollouts", type=int, default=2)
    ap.add_argument("--out-dir", default=str(ROOT / "data/processed"))
    ap.add_argument("--margin", type=float, default=0.0,
                    help="engine only overrides the value ranking when ahead by more than this win-probability margin (e.g. 0.01)")
    ap.add_argument("--tag", default="", help="suffix for the results file (lets seasons run as parallel processes)")
    a = ap.parse_args()
    main(a.years, a.drafts, a.sims, a.top_k, a.rollouts, a.out_dir, a.margin, a.tag)
