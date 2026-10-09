"""
Past top-60 per-game players who lost most of a season (<=41 GP): how did the next season go vs healthy stars,
by age / durability / severity, and how did our backtest projections do on those cases? Also Haliburton/Maxey history.

Diagnostic script (not part of the pipeline). Run from anywhere:  python scripts/analysis/injury_recovery.py
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

pd.set_option("display.width", 260); pd.set_option("display.max_columns", 40)
conn = sqlite3.connect(os.path.join(ROOT, "data", "processed", "nba.db"))
st = load_player_season_table(conn)
players = pd.read_sql_query("SELECT player_id, name FROM player", conn).set_index("player_id")["name"]
apg = A.actual_per_game(st).merge(st[["player_id", "year", "age"]], on=["player_id", "year"])

# per-game value (availability ignored) and rank within each season, among players with >= 25 games
cols = ["player_id", "avail", "mpg"] + A.CAT_COLS + ["fgm", "fga", "ftm", "fta"]
rows = []
for y, g in apg.groupby("year"):
    g2 = g[g.gp >= 25].copy()
    g2["avail"] = 1.0
    z = A.with_z(g2[cols].reset_index(drop=True))[["player_id", "z_total", "rank"]]
    z["year"] = y
    rows.append(z)
zdf = pd.concat(rows).rename(columns={"z_total": "z_pg", "rank": "rank_pg"})
apg = apg.merge(zdf, on=["player_id", "year"], how="left")
by = {(r.player_id, r.year): r for r in apg.itertuples()}
years = sorted(apg.year.unique())

events, controls = [], []
for pid in apg.player_id.unique():
    for Y in range(min(years) + 1, max(years)):
        pre, rec = by.get((pid, Y - 1)), by.get((pid, Y + 1))
        if pre is None or rec is None or pre.gp < 50 or not (pre.rank_pg <= 60) or rec.gp < 10:
            continue
        mid = by.get((pid, Y))
        mid_gp = 0 if mid is None else mid.gp
        rec_row = dict(pid=pid, name=players.get(pid, str(pid)), pre_year=Y - 1, rec_year=Y + 1, age_rec=rec.age,
                       pre_gp=pre.gp, mid_gp=mid_gp, rec_gp=rec.gp, pre_mpg=pre.mpg, rec_mpg=rec.mpg, pre_ppg=pre.pts, rec_ppg=rec.pts,
                       pre_z=pre.z_pg, rec_z=rec.z_pg, rec_avail=rec.avail, pre_avail=pre.avail)
        if mid_gp <= 41:
            events.append(rec_row)
        elif mid_gp >= 62:                                      # played >= ~75% of games: the healthy-star control
            controls.append(rec_row)
ev, ct = pd.DataFrame(events), pd.DataFrame(controls)
for d in (ev, ct):
    d["ppg_ratio"] = d.rec_ppg / d.pre_ppg
    d["mpg_diff"] = d.rec_mpg - d.pre_mpg
    d["z_diff"] = d.rec_z - d.pre_z
    d["age_grp"] = pd.cut(d.age_rec, [0, 28, 31, 99], labels=["<=28", "29-31", "32+"])

print(f"Top-60 per-game players (>=50 GP the year before) who then lost most of a season (<=41 GP, incl. 0): {len(ev)} cases, {ev.name.nunique()} players")
print(f"Healthy top-60 controls (>=62 GP in the middle year): {len(ct)} cases\n")


def summ(d, label):
    print(f"{label}: n={len(d)}")
    print(f"   games played in the recovery year (availability): mean {d.rec_avail.mean():.2f}, median {d.rec_avail.median():.2f}   (pre-injury year {d.pre_avail.mean():.2f})")
    print(f"   minutes: pre {d.pre_mpg.mean():.1f} -> recovery {d.rec_mpg.mean():.1f}  (diff {d.mpg_diff.mean():+.1f}, median {d.mpg_diff.median():+.1f})")
    print(f"   PPG:     pre {d.pre_ppg.mean():.1f} -> recovery {d.rec_ppg.mean():.1f}  (ratio median {d.ppg_ratio.median():.2f})")
    print(f"   per-game 9-cat value: pre {d.pre_z.mean():+.1f} -> recovery {d.rec_z.mean():+.1f}  (diff mean {d.z_diff.mean():+.2f}, median {d.z_diff.median():+.2f})")


summ(ev, "INJURY-HIT STARS (2-year change, pre-year -> recovery year)")
summ(ct, "HEALTHY STARS (same 2-year window)")
print("\nBy age at the recovery year:")
for g in ["<=28", "29-31", "32+"]:
    a, b = ev[ev.age_grp == g], ct[ct.age_grp == g]
    if len(a) >= 3:
        print(f"  {g:6s} injured n={len(a):2d}: avail {a.rec_avail.mean():.2f}, mpg diff {a.mpg_diff.mean():+.1f}, per-game value diff {a.z_diff.mean():+.2f}, PPG ratio {a.ppg_ratio.median():.2f}"
              f"   | healthy n={len(b):3d}: avail {b.rec_avail.mean():.2f}, mpg diff {b.mpg_diff.mean():+.1f}, value diff {b.z_diff.mean():+.2f}, PPG ratio {b.ppg_ratio.median():.2f}")
print("\nBy severity of the injury year:")
for lab, m in (("lost the WHOLE season (0 GP)", ev.mid_gp == 0), ("played 1-20 GP", (ev.mid_gp >= 1) & (ev.mid_gp <= 20)), ("played 21-41 GP", ev.mid_gp >= 21)):
    a = ev[m]
    if len(a):
        print(f"  {lab:30s} n={len(a):2d}: avail {a.rec_avail.mean():.2f}, mpg diff {a.mpg_diff.mean():+.1f}, per-game value diff {a.z_diff.mean():+.2f}, PPG ratio median {a.ppg_ratio.median():.2f}")

print("\nPrime-age (<=30 at recovery) injury-hit stars:")
pr = ev[ev.age_rec <= 30].sort_values("rec_year")
print(pr[["name", "pre_year", "pre_gp", "mid_gp", "rec_gp", "pre_mpg", "rec_mpg", "pre_ppg", "rec_ppg", "pre_z", "rec_z"]].round(1).to_string(index=False))

# how did OUR model project these recovery seasons? (backtest folds 2021-2025 only)
oos = pd.read_parquet(os.path.join(ROOT, "data", "processed", "backtest_oos.parquet"))
m = ev.merge(oos[["player_id", "year", "avail", "mpg", "pts"]].rename(columns={"player_id": "pid", "year": "rec_year", "avail": "p_avail", "mpg": "p_mpg", "pts": "p_ppg"}), on=["pid", "rec_year"])
print(f"\nOur backtest projections for {len(m)} of these recovery seasons (2021-2025 folds, played >=500 min):")
if len(m):
    print(f"   availability: projected {m.p_avail.mean():.2f} vs actual {m.rec_avail.mean():.2f}")
    print(f"   minutes:      projected {m.p_mpg.mean():.1f} vs actual {m.rec_mpg.mean():.1f}   (pre-injury {m.pre_mpg.mean():.1f})")
    print(f"   PPG:          projected {m.p_ppg.mean():.1f} vs actual {m.rec_ppg.mean():.1f}   (pre-injury {m.pre_ppg.mean():.1f})")
    print(m[["name", "rec_year", "pre_mpg", "p_mpg", "rec_mpg", "pre_ppg", "p_ppg", "rec_ppg", "p_avail", "rec_avail"]].round(2).to_string(index=False))

print("\nTyrese Haliburton / Maxey history:")
for nm in ("Tyrese Haliburton", "Tyrese Maxey"):
    pid = players[players == nm].index
    if len(pid):
        h = apg[apg.player_id == pid[0]].sort_values("year").tail(5)
        print(nm); print(h[["year", "gp", "mpg", "pts", "reb", "ast", "avail", "rank_pg"]].round(2).to_string(index=False))

print("\n" + "=" * 100)
print("USUALLY-HEALTHY stars: played >=75% of games in BOTH of the two seasons before the injury year")
def prior_av(row, k):
    r = by.get((row.pid, row.pre_year - k))
    return np.nan if r is None else r.avail
for d in (ev, ct):
    d["pre2_avail"] = [prior_av(r, 1) for r in d.itertuples()]
dur_ev = ev[(ev.pre_avail >= 0.75) & (ev.pre2_avail >= 0.75)]
dur_ct = ct[(ct.pre_avail >= 0.75) & (ct.pre2_avail >= 0.75)]
summ(dur_ev, "DURABLE stars who then lost most of a season")
summ(dur_ct, "DURABLE stars who stayed healthy (control)")
for lab, d in (("durable, injured", dur_ev), ("durable, healthy", dur_ct), ("ALL injured", ev), ("ALL healthy", ct)):
    print(f"   {lab:18s} keep >=95% of PPG: {(d.ppg_ratio >= 0.95).mean():.0%} | keep >=90%: {(d.ppg_ratio >= 0.90).mean():.0%} | PPG ratio 25th/50th/75th pct: "
          f"{d.ppg_ratio.quantile(.25):.2f}/{d.ppg_ratio.median():.2f}/{d.ppg_ratio.quantile(.75):.2f} | played >=70% of games: {(d.rec_avail >= 0.70).mean():.0%}")
print("\nDurable, prime-age (<=30) injured stars:")
x = dur_ev[dur_ev.age_rec <= 30].sort_values("rec_year")
print(x[["name", "pre_year", "mid_gp", "rec_gp", "pre_mpg", "rec_mpg", "pre_ppg", "rec_ppg", "rec_avail"]].round(2).to_string(index=False))
print(f"   -> n={len(x)}: mean avail {x.rec_avail.mean():.2f}, mpg diff {x.mpg_diff.mean():+.1f}, PPG ratio median {x.ppg_ratio.median():.2f}, value diff {x.z_diff.mean():+.2f}")
