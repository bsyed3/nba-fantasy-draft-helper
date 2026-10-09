"""
Is availability/minutes biased for 'usually healthy, one bad year' / 'recovered' / 'consistently healthy' groups?
Selection-free (absent players count as 0 games). Also prints named players' histories and a leave-one-season-out test of a 'recovered' bump.

Diagnostic script (not part of the pipeline). Run from anywhere:  python scripts/analysis/availability_bias.py
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
priors = league_priors(st)
sg = st.groupby("year")["sg"].first().to_dict()
first = int(st["year"].min())
sel = pd.read_csv(os.path.join(ROOT, "data", "processed", "backtest_summary.csv"))
sel = sel[sel["selected"]].set_index("stat")["model"].to_dict()
wides = {t: P.build_wide(st, priors, t, sg) for t in range(first + 1, 2027)}

rows = []
for T in range(2021, 2026):
    train = pd.concat([wides[t] for t in range(first + 1, T)], ignore_index=True)
    test = wides[T].copy()
    preds = P.predict_all(train, test)
    test["avail_pred"] = preds["avail"][sel["avail"]]
    test["mpg_pred"] = preds["mpg"][sel["mpg"]]
    for l in (1, 2, 3):
        test[f"av{l}"] = test[f"gp_{l}"] / test[f"sg_{l}"]
        test[f"mp{l}"] = test[f"min_{l}"] / test[f"gp_{l}"]
    rows.append(test)
d = pd.concat(rows, ignore_index=True)
d = d[d["min_1"].fillna(0) >= 500]                          # established players (same filter the avail backtest uses); absent players stay in as avail = 0
d["avail_act"] = d["y_avail"]
d = d[d["avail_act"].notna()]
print(f"established player-seasons: {len(d)}  | overall availability miss (actual - projected): {(d.avail_act - d.avail_pred).mean():+.3f}")

healthy_one_bad = d[(d.av1 < 0.60) & (d.av2 >= 0.75) & (d.av3.fillna(d.av2) >= 0.70) & (d.mp2 >= 25)]
chronic = d[(d.av1 < 0.60) & (d.av2 < 0.65)]
normal = d[(d.av1 >= 0.60)]
for name, g in (("USUALLY HEALTHY, one bad year (av1<.6; av2>=.75, av3>=.70; rotation)", healthy_one_bad),
                ("repeatedly missing games (av1<.6 and av2<.65)", chronic), ("everyone else (av1>=.6)", normal)):
    se = (g.avail_act - g.avail_pred).std() / np.sqrt(max(len(g), 1))
    m = g[g.mpg_pred.notna() & g.y_mpg.notna()]
    print(f"\n{name}: n={len(g)}")
    print(f"   availability: last yr {g.av1.mean():.2f}, earlier {g.av2.mean():.2f} | projected {g.avail_pred.mean():.2f} | actual {g.avail_act.mean():.2f} | miss {(g.avail_act - g.avail_pred).mean():+.3f} ({(g.avail_act - g.avail_pred).mean() / se:+.1f} SE)")
    print(f"   minutes (those who played): last yr {m.mp1.mean():.1f}, earlier {m.mp2.mean():.1f} | projected {m.mpg_pred.mean():.1f} | actual {m.y_mpg.mean():.1f} | miss {(m.y_mpg - m.mpg_pred).mean():+.2f}")

# named players: history + current projection
proj = pd.read_parquet(os.path.join(ROOT, "data", "processed", "projections_2026.parquet")).set_index("name")
w26 = wides[2026].set_index("player_id")
print("\nNamed players (availability = GP / team games; minutes per game):")
print(f"{'player':22s} {'2022-23':>8s} {'2023-24':>8s} {'2024-25':>8s} {'2025-26':>8s} | {'proj avail':>10s} {'proj mpg':>9s}")
for nm in ["Giannis Antetokounmpo", "Kevin Durant", "Paul George", "Jamal Murray", "Chet Holmgren", "Victor Wembanyama", "Kawhi Leonard", "Joel Embiid"]:
    if nm not in proj.index:
        print(nm, "not found"); continue
    pid = proj.loc[nm, "player_id"]
    s_ = st[st.player_id == pid].set_index("year")
    cells = []
    for y in (2022, 2023, 2024, 2025):
        cells.append(f"{s_.loc[y, 'gp'] / s_.loc[y, 'sg']:.2f}/{s_.loc[y, 'min'] / s_.loc[y, 'gp']:.0f}" if y in s_.index else "   -   ")
    print(f"{nm:22s} {cells[0]:>8s} {cells[1]:>8s} {cells[2]:>8s} {cells[3]:>8s} | {proj.loc[nm, 'avail']:10.2f} {proj.loc[nm, 'mpg']:9.1f}")

print("\n=== calibration at the healthy end (selection-free: absent players count as 0) ===")
groups = {
    "RECOVERED: healthy last year (av1>=.80) after an injury-wrecked year in the prior two (min(av2,av3)<.60), rotation player":
        d[(d.av1 >= 0.80) & (d[["av2", "av3"]].min(axis=1) < 0.60) & (d.mp1 >= 25)],
    "CONSISTENTLY HEALTHY: av>=.80 in each of the last 2-3 years, rotation player":
        d[(d.av1 >= 0.80) & (d.av2 >= 0.80) & (d.av3.fillna(0.8) >= 0.80) & (d.mp1 >= 25)],
    "HEALTHY LAST YEAR (av1>=.85), any history, rotation player": d[(d.av1 >= 0.85) & (d.mp1 >= 25)],
}
for name, g in groups.items():
    r = g.avail_act - g.avail_pred
    se = r.std() / np.sqrt(len(g))
    print(f"\n{name}\n   n={len(g)} | last yr {g.av1.mean():.2f} | projected {g.avail_pred.mean():.2f} | actual {g.avail_act.mean():.2f} | miss {r.mean():+.3f} ({r.mean() / se:+.1f} SE) | share who played >=75% of games: {(g.avail_act >= .75).mean():.0%}")
    print("   by season:", {int(y): f"n={len(x)} miss {(x.avail_act - x.avail_pred).mean():+.2f}" for y, x in g.groupby("year")})

print("\n=== would a simple 'recovered' availability bump help on HELD-OUT seasons? (leave-one-season-out) ===")
rec = groups["RECOVERED: healthy last year (av1>=.80) after an injury-wrecked year in the prior two (min(av2,av3)<.60), rotation player"].copy()
rec["miss"] = rec.avail_act - rec.avail_pred
sse_old = sse_new = 0.0
for y in sorted(rec.year.unique()):
    delta = rec.loc[rec.year != y, "miss"].mean()
    t = rec[rec.year == y]
    sse_old += ((t.avail_act - t.avail_pred) ** 2).sum()
    sse_new += ((t.avail_act - (t.avail_pred + delta)) ** 2).sum()
print(f"flagged rows: {len(rec)} | MSE old {sse_old / len(rec):.4f} -> new {sse_new / len(rec):.4f} ({(sse_new / sse_old - 1) * 100:+.1f}%)")
tot = ((d.avail_act - d.avail_pred) ** 2).sum()
print(f"effect on ALL {len(d)} established player-seasons: total squared error changes by {(sse_new - sse_old) / tot * 100:+.2f}%")
