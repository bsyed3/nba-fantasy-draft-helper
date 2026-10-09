"""
(1) How widely do the mock-draft bots spread players around the value rank (300 simulated drafts)? (2) Why does the value board
disagree with the market for Cade / Edwards / Siakam (per-game vs availability-discounted rank, z by category).

Diagnostic script (not part of the pipeline). Run from anywhere:  python scripts/analysis/bot_spread_and_board.py
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
from src.features.player_seasons import load_player_season_table
from src.reporting import analysis as A
from src.draft.values import CATEGORIES, CAT_LABELS

proj = pd.read_parquet(os.path.join(ROOT, "data", "processed", "projections_2026.parquet"))
conn = sqlite3.connect(os.path.join(ROOT, "data", "processed", "nba.db"))
st = load_player_season_table(conn)
apg = A.actual_per_game(st)

# ---------- (1) how do the mock-draft bots spread players around the value ranking?
rank = np.arange(len(proj))
P = len(proj)
sd = 6.0 + 0.2 * rank
pos = {r: [] for r in range(0, 40)}
rng = np.random.default_rng(0)
for d in range(300):
    orders = [np.argsort(rank + rng.standard_normal(P) * sd, kind="stable") for _ in range(10)]
    ptr = [0] * 10
    taken = np.zeros(P, bool)
    pick = 0
    for rnd in range(13):
        seq = range(10) if rnd % 2 == 0 else range(9, -1, -1)
        for t in seq:
            o, k = orders[t], ptr[t]
            while taken[o[k]]:
                k += 1
            ptr[t] = k + 1
            taken[o[k]] = True
            pick += 1
            if o[k] < 40:
                pos[o[k]].append(pick)
print("Value rank -> where the BOTS take him (300 mock drafts): mean pick, 10th-90th pct pick")
for r in (4, 9, 15, 22, 29, 39):
    a = np.array(pos[r]); print(f"  rank {r + 1:2d}: mean pick {a.mean():5.1f}, 10-90% = {np.percentile(a, 10):.0f}-{np.percentile(a, 90):.0f}, P(still there after pick 25) = {(a > 25).mean():.0%}, after 45 = {(a > 45).mean():.0%}")

# ---------- (2) why does our value board differ from the market for these players?
last = 2025
mv_total = A.movers(proj, apg, last)
mv_pg = A.movers(proj, apg, last, hold_avail=True)
for name in ["Cade Cunningham", "Anthony Edwards", "Pascal Siakam"]:
    r = proj[proj.name == name].iloc[0]
    z = {CAT_LABELS[c]: round(float(r[f"z_{c}"]), 1) for c in CATEGORIES}
    a = apg[(apg.year == last) & (apg.player_id == r.player_id)].iloc[0]
    m = mv_total[mv_total.name == name]
    mp = mv_pg[mv_pg.name == name]
    print(f"\n{name}: projected value rank {int(r['rank'])} (age {r.age_t:.0f}, mpg {r.mpg:.1f}, avail {r.avail:.2f})")
    print("   projected z by category:", z)
    print(f"   2025-26 actual: {a.pts:.1f} pts {a.reb:.1f} reb {a.ast:.1f} ast {a.tov:.1f} to, gp {int(a.gp)}, mpg {a.mpg:.1f}")
    if len(m):
        print(f"   rank among 1000+ min players: 2025-26 actual {int(m.rank_last.iloc[0])} -> projected {int(m.rank_proj.iloc[0])}"
              f" | per-game only (games played held equal): {int(mp.rank_last.iloc[0])} -> {int(mp.rank_proj.iloc[0])}")
