"""
Rotation players who played ZERO games in the previous season (lost season) and came back: availability, minutes change,
second-absence rate, by season. Evidence behind 'Haliburton-type' projections (avail ~0.48, ~-5 to -8 mpg).

Diagnostic script (not part of the pipeline). Run from anywhere:  python scripts/analysis/returner_gap.py
Needs data/processed/{nba.db, projections_2026.parquet, backtest_oos.parquet, backtest_summary.csv} (see docs/HANDOFF.md section 4).
"""
import os
import sys
from pathlib import Path

ROOT = str(Path(__file__).resolve().parents[2])
sys.stdout.reconfigure(encoding="utf-8")
import sqlite3, sys
import numpy as np, pandas as pd

sys.path.insert(0, ROOT)
from src.features.player_seasons import load_player_season_table, league_priors
from src.models import projection as P

pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40)
conn = sqlite3.connect(os.path.join(ROOT, "data", "processed", "nba.db"))
st = load_player_season_table(conn)
players = pd.read_sql_query("SELECT player_id, name, last_year FROM player", conn).set_index("player_id")
priors = league_priors(st)
sg = st.groupby("year")["sg"].first().to_dict()
first = int(st["year"].min())
rows = []
for T in range(first + 2, 2026):
    w = P.build_wide(st, priors, T, sg)
    w["T"] = T
    rows.append(w)
d = pd.concat(rows, ignore_index=True)
d["mp2"] = d["min_2"] / d["gp_2"]


def returner_mask(w):
    """Rotation player (>=25 mpg over >=30 GP two seasons ago) with NO games last season = a lost season, not a fringe player."""
    return (w["gp_1"].isna() & (w["gp_2"] >= 30) & (w["mp2"] >= 25)).to_numpy()


f = d[returner_mask(d)].copy()
f["name"] = f.player_id.map(players["name"])
print(f"Rotation players (>=25 mpg, >=30 GP at T-2) with NO games at T-1, 2017-2025: {len(f)} rows, {f.player_id.nunique()} players")
have = f[f["y_avail"].notna()]
print(f"   still in the league at T (present or later): {len(have)}; of those who actually played at T: {(have.y_avail > 0).sum()}  ({(have.y_avail > 0).mean():.0%}); stayed out again: {(have.y_avail == 0).sum()}")
print(f"   availability at T: mean over ALL still-in-league {have.y_avail.mean():.2f} | conditional on playing {have[have.y_avail > 0].y_avail.mean():.2f}")
pl = have[have.y_mpg.notna()]
print(f"   minutes when they played: T-2 {pl.mp2.mean():.1f} -> T {pl.y_mpg.mean():.1f} (diff {(pl.y_mpg - pl.mp2).mean():+.1f}, median {(pl.y_mpg - pl.mp2).median():+.1f})")
heavy = pl[pl.mp2 >= 28]
print(f"   heavy (>=28 mpg at T-2) subset n={len(heavy)}: availability {have[have.player_id.isin(heavy.player_id) & have.T.isin(heavy['T'])].y_avail.mean():.2f}, minutes diff {(heavy.y_mpg - heavy.mp2).mean():+.1f}")
print("\nBy season T (n, played, mean avail all / if played, mpg diff):")
for T, g in have.groupby("T"):
    pg = g[g.y_avail > 0]
    print(f"  {T}: n={len(g):2d} played={len(pg):2d} avail_all={g.y_avail.mean():.2f} avail_if_played={pg.y_avail.mean():.2f} mpg_diff={(pg.y_mpg - pg.mp2).mean():+.1f}")
print("\nThe ones who stayed out a second year:")
print(have[have.y_avail == 0][["name", "T", "age_t", "mp2"]].round(1).to_string(index=False))
print("\nModel's current projection for them in 2026 is based on training mean avail:", round(have.y_avail.mean(), 2))
