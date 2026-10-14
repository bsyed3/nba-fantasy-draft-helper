"""
Is projected games played biased -- and how much does knowing who is out at tip-off matter?

Availability is modelled as games played / team games AFTER a player's first appearance of the season. Games missed before
it (a torn Achilles in the playoffs, an injury in camp: known at tip-off) are not forecastable from box scores, so they are
excluded from the model's targets and supplied separately (data/external/known_absences.csv). The earlier definition
(games / all team games, never-played seasons counted as 0) taught the model that good players sometimes play zero games,
which pulled every healthy player's projection down ~2.3 GP and made "actual" look worse than projected whenever a star
was known to be out (2025-26: Haliburton, Lillard, Kyrie, VanVleet, Tatum). See docs/HANDOFF.md section 6.1.

Prints, over the 2021-2025 backtest folds: calibration of in-season availability for rotation players; games played for
projected starters split into "active at tip-off" and "missed 20+ opening games"; the same with each player's opening
absence supplied (what a known-absences list does); and the blend weight for players who missed all of last season.

Diagnostic script (not part of the pipeline). Run from anywhere:  python scripts/analysis/availability_calibration.py   (~3 min)
Needs data/processed/{nba.db, backtest_summary.csv} (see docs/HANDOFF.md section 4).
"""
import os
import sqlite3
import sys
from pathlib import Path

ROOT = str(Path(__file__).resolve().parents[2])
sys.stdout.reconfigure(encoding="utf-8")
import numpy as np, pandas as pd

sys.path.insert(0, ROOT)
from src.features.player_seasons import load_player_season_table, league_priors
from src.models import projection as P

pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40)
conn = sqlite3.connect(os.path.join(ROOT, "data", "processed", "nba.db"))
st = load_player_season_table(conn)
priors = league_priors(st)
sg = st.groupby("year")["sg"].first().to_dict()
first = int(st["year"].min())
sel = pd.read_csv(os.path.join(ROOT, "data", "processed", "backtest_summary.csv"))
sel = sel[sel["selected"]].set_index("stat")["model"].to_dict()
wides = {t: P.build_wide(st, priors, t, sg) for t in range(first + 1, 2027)}
current_w = P.RETURNER_BASELINE_WEIGHT
P.RETURNER_BASELINE_WEIGHT = 0.0          # raw selected model; the returner blend is evaluated explicitly below

rows = []
for T in range(2021, 2026):
    train = pd.concat([wides[t] for t in range(first + 1, T)], ignore_index=True)
    test = wides[T].copy()
    preds = P.predict_all(train, test)
    test["raw"] = preds["avail"][sel["avail"]]
    test["base"] = preds["avail"]["baseline"]
    test["p_mpg"] = preds["mpg"][sel["mpg"]]
    rows.append(test)
D = pd.concat(rows, ignore_index=True)
D["last_min"] = D.min_1.fillna(D.min_2).fillna(D.min_3)
D["last_gp"] = D.gp_1.fillna(D.gp_2).fillna(D.gp_3)
D["pred"] = np.where(D.gap > 0, current_w * D.base + (1 - current_w) * D.raw, D.raw)         # what the pipeline now projects
D["smiss_t"] = D.smiss_t.fillna(82.0)                                                         # never played = out from game one
D["act_gp"] = D.gp_t
rot = (D.last_min >= 500) & (D.last_min / D.last_gp >= 24)

print(f"=== in-season availability (games after his first appearance), rotation players who played last season (n={int((rot & (D.gap == 0) & D.y_avail.notna()).sum())})")
d = D[rot & (D.gap == 0) & D.y_avail.notna()]
print(f"projected {d.pred.mean():.3f}  actual {d.y_avail.mean():.3f}  miss {(d.y_avail.mean() - d.pred.mean()) * 82:+.1f} GP (se {(d.y_avail - d.pred).std() / np.sqrt(len(d)) * 82:.1f})")
for y, g in d.groupby("year"):
    print(f"  {y}: n={len(g)} projected {g.pred.mean():.3f} actual {g.y_avail.mean():.3f} ({(g.y_avail.mean() - g.pred.mean()) * 82:+.1f} GP)")

print("\n=== games played for projected starters (projected MPG >= 25), healthy-at-tip-off assumption vs known absences supplied")
S = D[(D.p_mpg >= 25) & D.pred.notna()].copy()
S["gp_healthy"] = S.pred * 82
S["gp_known"] = S.pred * (82 - S.smiss_t).clip(lower=0)
for label, g in (("all", S), ("active at tip-off (missed < 20 opening games)", S[S.smiss_t < 20]), ("out at tip-off (missed 20+ opening games)", S[S.smiss_t >= 20])):
    print(f"\n{label}: n={len(g)}  actual mean {g.act_gp.mean():.1f}  median {g.act_gp.median():.1f}")
    for nm, c in (("assume healthy at tip-off", "gp_healthy"), ("known opening absences supplied", "gp_known")):
        e = g.act_gp - g[c]
        print(f"   {nm:34s} projected mean {g[c].mean():5.1f}  miss {e.mean():+5.1f}  MAE {np.abs(e).mean():5.1f}  RMSE {np.sqrt((e ** 2).mean()):5.1f}")
print("\nby season, active at tip-off: projected mean / actual mean / actual median")
for y, g in S[S.smiss_t < 20].groupby("year"):
    print(f"  {y}: n={len(g)}  {g.gp_healthy.mean():.1f} / {g.act_gp.mean():.1f} / {g.act_gp.median():.1f}   | share of projected starters out 20+ opening games: {(S[S.year == y].smiss_t >= 20).mean():.1%}")

r = D[rot & (D.gap > 0) & D.y_avail.notna()]
print(f"\n=== players who missed all of last season (n={len(r)}): in-season availability, blend weight on the baseline (0 = selected model)")
for wt in (0, 0.25, 0.5, 0.75, 1.0):
    bl = wt * r.base + (1 - wt) * r.raw
    print(f"  w={wt:.2f}: projected {bl.mean():.3f} vs actual {r.y_avail.mean():.3f}   MSE {np.mean((r.y_avail - bl) ** 2):.4f}")
print(f"  (pipeline uses RETURNER_BASELINE_WEIGHT = {current_w})")
