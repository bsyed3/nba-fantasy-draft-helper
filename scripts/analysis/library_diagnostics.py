"""
Phase 2 sanity check: does the simulation library reproduce the projections it was built from?

For every player the library's mean games per matchup period should equal projected availability x his team's games per period,
and its mean per-game line (while playing) should equal the projection (the talent and count noise are mean-preserving).

    python scripts/analysis/library_diagnostics.py [--year 2026]
"""
from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding="utf-8")

from src.simulation.copula import weekly_means
from src.simulation.library import Library
from src.simulation.schedule import season_periods

DATA = ROOT / "data" / "processed"


def main(year: int) -> None:
    proj = pd.read_parquet(DATA / f"projections_{year}.parquet")
    lib = Library.load(DATA / f"library_{year}.npz")
    conn = sqlite3.connect(DATA / "nba.db")
    sched = season_periods(conn, f"{year}-{str(year + 1)[-2:]}", n_regular=18, n_playoff=0, fill_unscheduled=False)
    team_games = dict(zip(sched.team_ids, sched.masks.sum(axis=2).mean(axis=0)))     # games per matchup period, by NBA team
    # library columns carry a _lib suffix (weekly totals); projection columns are per game
    wm = weekly_means(lib).rename(columns=lambda c: c if c == "player_id" else c + "_lib").merge(proj, on="player_id")
    wm["exp_games"] = wm["avail"].clip(0.03, 0.995) * wm["team_id"].map(team_games).fillna(0)
    ok = wm["exp_games"] > 0.5
    print(f"{len(lib.player_ids)} players, {int(ok.sum())} with a real chance to play; {lib.stats.shape[1]} simulated periods")

    r = (wm.games_lib / wm.exp_games)[ok]
    print(f"\ngames per period / (availability x team games): median {r.median():.3f}, middle 90% of players {r.quantile(.05):.2f}-{r.quantile(.95):.2f}")
    by = {}
    for lo, hi in ((0, .4), (.4, .6), (.6, .8), (.8, 1.01)):
        m = ok & wm.avail.between(lo, hi, inclusive="left")
        by[f"avail {lo:.1f}-{min(hi, 1):.1f} (n={int(m.sum())})"] = round(float(wm.loc[m, "games_lib"].sum() / wm.loc[m, "exp_games"].sum()), 3)
    print("  total simulated / expected games by projected availability:", by)

    g = wm[ok & (wm.games_lib > 0.3)].copy()
    n = g.games_lib
    lib_line = pd.DataFrame({"pts": (2 * g.fgm_lib + g.tpm_lib + g.ftm_lib) / n, "tpm": g.tpm_lib / n, "reb": g.reb_lib / n, "ast": g.ast_lib / n,
                             "stl": g.stl_lib / n, "blk": g.blk_lib / n, "tov": g.tov_lib / n})
    rows = [dict(stat=c.upper(), projection=g[c].mean(), library=lib_line[c].mean(), ratio=lib_line[c].mean() / g[c].mean(),
                 median_player_ratio=(lib_line[c] / g[c].replace(0, np.nan)).median()) for c in lib_line]
    for label, num, den, proj_col in (("FG%", "fgm_lib", "fga_lib", "fg_pct"), ("FT%", "ftm_lib", "fta_lib", "ft_pct")):
        pct = g[num] / g[den]
        rows.append(dict(stat=label, projection=g[proj_col].mean(), library=pct.mean(), ratio=pct.mean() / g[proj_col].mean(),
                         median_player_ratio=(pct / g[proj_col]).median()))
    print("\nper-game line while playing, library vs projection (n=%d players):" % len(g))
    print(pd.DataFrame(rows).round(3).to_string(index=False))
    gap = (lib_line.pts / g.pts - 1).abs().sort_values(ascending=False).head(5).index
    print("\nlargest points-per-game gaps (name, library, projection, games per period):",
          [(g.loc[i, "name"], round(float(lib_line.pts[i]), 1), round(float(g.pts[i]), 1), round(float(g.games_lib[i]), 2)) for i in gap])
    for nm in ("Brandon Ingram", "Kristaps Porziņģis", "Giannis Antetokounmpo", "Kyrie Irving"):
        w = wm[wm.name == nm]
        if len(w):
            w = w.iloc[0]
            print(f"{nm:24s} availability {w.avail:.2f}; games per period: library {w.games_lib:.2f} vs expected {w.exp_games:.2f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--year", type=int, default=2026)
    main(ap.parse_args().year)
