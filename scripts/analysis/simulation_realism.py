"""
Phase 2 realism check: is the spread of the simulated weeks realistic?

For a past season Y the pipeline sees only information from before Y: pre-season projections -> simulated library. Realistic
10-team rosters are mock-drafted (best available by the board) and every team is then scored two ways with the SAME daily-lineup
rules: on the simulated weeks and on the weeks that really happened. Two tests:

  1. Category totals. For each team, week and category, where does the real total fall inside the simulated distribution for that
     same week? That position (the probability integral transform, PIT, in [0, 1]) is uniform if the simulation is calibrated:
     mean 0.5, 50% of values inside [0.25, 0.75], and the normal scores have variance 1 (>1 = the simulation is too narrow).
     Normal scores of different categories should be uncorrelated (positive = real categories move together more than simulated).
  2. Matchups. For every pair of teams and week the simulation gives P(A beats B) (more than 4.5 of 9 categories). Those
     probabilities are compared with who really won (reliability by decile, calibration slope, Brier score).

    python scripts/analysis/simulation_realism.py run --year 2024 --out exports/realism_2024.npz
    python scripts/analysis/simulation_realism.py report exports/realism_*.npz

Caveat: calibration.json (count dispersion, talent spread, ...) is fit on all seasons, so the comparison is not strictly out of sample.
"""
from __future__ import annotations

import argparse
import glob
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding="utf-8")

from scripts.validate_draft import prepare_year, project_past_season
from src.draft.engine import DraftEngine, category_matrix
from src.draft.mock import best_available, run_mock_draft
from src.draft.state import LeagueSettings
from src.draft.values import CATEGORIES
from src.features.player_seasons import load_player_season_table
from src.simulation.calibration import Calibration

DATA = ROOT / "data" / "processed"
KW = dict(n_teams=10, roster_size=13, active_slots=10, max_centers=3)
LOWER = np.array([c == "tov" for c in CATEGORIES])


def run_year(year: int, drafts: int, sims_per_week: int, playoffs: bool, out: str) -> None:
    conn = sqlite3.connect(DATA / "nba.db")
    st_all = load_player_season_table(conn)
    players = pd.read_sql_query(
        "SELECT player_id, name, position_1, position_2, draft_year, draft_position, rookie_year, last_year FROM player", conn)
    cal = Calibration.load(DATA / "calibration.json")
    cache = Path(out).with_name(f"realism_proj_{year}.parquet")          # the slow projection step, reused between runs
    proj = pd.read_parquet(cache) if cache.exists() else project_past_season(conn, st_all, players, year)
    proj.to_parquet(cache)
    season, proj, sched, lib, actual = prepare_year(conn, st_all, players, cal, year, sims_per_week, playoffs, proj=proj)
    W, S = sched.n_weeks, lib.stats.shape[1]
    n = S // W
    print(f"{season}: {W} weeks x {n} simulated draws, {len(proj)} players")
    pit_rows, sim_pit_rows, pm_rows, po_rows, wk_rows = [], [], [], [], []
    for d in range(drafts):
        s = LeagueSettings(my_slot=d % KW["n_teams"] + 1, season=season, **KW)
        eng, truth = DraftEngine(lib, proj, s, seed=d), DraftEngine(actual, proj, s)
        ros = run_mock_draft(eng, s, best_available, bot_seed=1000 * year + d)
        T = KW["n_teams"]
        sim = category_matrix(np.stack([eng.team_totals(ros[t]) for t in range(1, T + 1)]))      # [T, S, 9]
        real = category_matrix(np.stack([truth.team_totals(ros[t]) for t in range(1, T + 1)]))   # [T, W, 9]
        cs = sim.reshape(T, n, W, len(CATEGORIES))                                              # sim s = i * W + w plays week w
        less, eq = (cs < real[:, None]).sum(1), (cs == real[:, None]).sum(1)
        pit_rows.append(((less + 0.5 * eq) / n).reshape(-1, len(CATEGORIES)))                   # [T*W, 9]
        # benchmark: treat one simulated draw as if it were reality and place it among the others (leave-one-out PIT). A calibrated
        # simulation gives these the same spread AND the same cross-category correlation as the real ones.
        x0, rest = cs[:, 0], cs[:, 1:]
        sim_pit_rows.append((((rest < x0[:, None]).sum(1) + 0.5 * (rest == x0[:, None]).sum(1)) / (n - 1)).reshape(-1, len(CATEGORIES)))
        for a in range(T):
            for b in range(a + 1, T):
                ca, cb = cs[a][:, None], cs[b][None, :]                                          # [n,1,W,9] vs [1,n,W,9]: every pairing of draws
                wins = (np.where(LOWER, ca < cb, ca > cb) + 0.5 * (ca == cb)).sum(-1)            # [n, n, W]
                p = ((wins > 4.5) + 0.5 * (wins == 4.5)).mean(axis=(0, 1))                        # [W]
                rw = (np.where(LOWER, real[a] < real[b], real[a] > real[b]) + 0.5 * (real[a] == real[b])).sum(-1)
                pm_rows.append(p)
                po_rows.append((rw > 4.5) + 0.5 * (rw == 4.5))
                wk_rows.append(np.arange(W))
        print(f"  draft {d + 1}/{drafts} done", flush=True)
    np.savez_compressed(out, pit=np.concatenate(pit_rows), sim_pit=np.concatenate(sim_pit_rows), p=np.concatenate(pm_rows), o=np.concatenate(po_rows),
                        week=np.concatenate(wk_rows), year=year, n_draws=n, n_weeks=W)
    print(f"wrote {out}")


def report(files: list[str]) -> None:
    parts = [np.load(f) for f in files]
    years = [int(p["year"]) for p in parts]
    n_draws = int(parts[0]["n_draws"])
    pit = np.concatenate([p["pit"] for p in parts])
    ny = np.concatenate([np.full(len(p["pit"]), int(p["year"])) for p in parts])
    pm, po = np.concatenate([p["p"] for p in parts]), np.concatenate([p["o"] for p in parts])
    py = np.concatenate([np.full(len(p["p"]), int(p["year"])) for p in parts])
    print(f"seasons {years}; {len(pit)} team-weeks; {len(pm)} team-pair-weeks; {n_draws} simulated draws per week\n")

    clip = lambda u: np.clip(u, 0.5 / n_draws, 1 - 0.5 / n_draws)
    z = norm.ppf(clip(pit))
    rows = []
    for k, c in enumerate(CATEGORIES):
        u = pit[:, k]
        rows.append(dict(category=c, mean_pit=u.mean(), in_50=((u > .25) & (u < .75)).mean(), in_80=((u > .1) & (u < .9)).mean(),
                         in_90=((u > .05) & (u < .95)).mean(), z_var=z[:, k].var(), z_mean=z[:, k].mean()))
    allu = pit.ravel()
    rows.append(dict(category="ALL", mean_pit=allu.mean(), in_50=((allu > .25) & (allu < .75)).mean(), in_80=((allu > .1) & (allu < .9)).mean(),
                     in_90=((allu > .05) & (allu < .95)).mean(), z_var=z.var(), z_mean=z.mean()))
    t = pd.DataFrame(rows)
    print("1. Where the real weekly category total falls in the simulated distribution (ideal: mean 0.50, inside 50/80/90% = 0.50/0.80/0.90, z var 1.00)")
    print(t.round(3).to_string(index=False))
    by = []
    for y in sorted(set(ny)):
        zz, uu = z[ny == y], pit[ny == y]
        by.append(dict(season=y, mean_pit=uu.mean(), in_50=((uu > .25) & (uu < .75)).mean(), in_80=((uu > .1) & (uu < .9)).mean(), z_var=zz.var()))
    print("\n   by season:\n" + pd.DataFrame(by).round(3).to_string(index=False))
    # categories legitimately move together within a week (more games = more of everything), so the benchmark is the simulation's own
    # correlation, measured by placing one simulated draw among the others
    sim_pit = np.concatenate([p["sim_pit"] for p in parts])
    zs = norm.ppf(clip(sim_pit))
    mask = ~np.eye(len(CATEGORIES), dtype=bool)
    cm, cs_ = np.corrcoef(z.T), np.corrcoef(zs.T)
    print(f"\n   spread check on the benchmark (a calibrated simulation placed among itself): z variance {zs.var():.3f}, mean PIT {sim_pit.mean():.3f}")
    print(f"   cross-category correlation of normal scores: REAL mean {cm[mask].mean():+.3f} vs SIMULATED benchmark {cs_[mask].mean():+.3f}")
    pairs = sorted(((cm[i, j] - cs_[i, j], cm[i, j], cs_[i, j], CATEGORIES[i], CATEGORIES[j]) for i in range(9) for j in range(i + 1, 9)), reverse=True)
    print("   largest real-minus-simulated gaps: " + ", ".join(f"{a}/{b} real {r:+.2f} vs sim {s:+.2f}" for _, r, s, a, b in pairs[:4]))
    print("   smallest: " + ", ".join(f"{a}/{b} real {r:+.2f} vs sim {s:+.2f}" for _, r, s, a, b in pairs[-3:]))

    print("\n2. Matchups: simulated P(team A beats team B in a week) against who really won")
    bins = np.linspace(0, 1, 11)
    idx = np.clip(np.digitize(pm, bins) - 1, 0, 9)
    rel = pd.DataFrame([dict(predicted=f"{bins[i]:.1f}-{bins[i + 1]:.1f}", n=int((idx == i).sum()), mean_predicted=pm[idx == i].mean(),
                             actual_win_rate=po[idx == i].mean()) for i in range(10) if (idx == i).sum() > 0])
    print(rel.round(3).to_string(index=False))
    from sklearn.linear_model import LogisticRegression
    x = np.log(np.clip(pm, 0.01, 0.99) / (1 - np.clip(pm, 0.01, 0.99)))[:, None]
    X, Y, Wt = np.vstack([x, x]), np.r_[np.ones(len(x)), np.zeros(len(x))], np.r_[po, 1 - po]
    lr = LogisticRegression(C=1e6, max_iter=1000).fit(X, Y, sample_weight=Wt)
    brier = np.mean((pm - po) ** 2)
    print(f"\n   calibration slope {lr.coef_[0][0]:.3f} (1.0 = calibrated, <1 = the simulation is overconfident), intercept {lr.intercept_[0]:+.3f}; "
          f"mean predicted {pm.mean():.3f} vs actual {po.mean():.3f}; Brier {brier:.4f} vs {np.mean((0.5 - po) ** 2):.4f} for always saying 50%")
    print(f"   simulated favourite (P > 0.5) wins {np.mean(po[pm > 0.55] > 0.5):.1%} of {int((pm > 0.55).sum())} pairs where P > 0.55")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--year", type=int, required=True)
    r.add_argument("--drafts", type=int, default=6)
    r.add_argument("--sims-per-week", type=int, default=100)
    r.add_argument("--no-playoffs", action="store_true")
    r.add_argument("--out", required=True)
    q = sub.add_parser("report")
    q.add_argument("files", nargs="+")
    a = ap.parse_args()
    if a.cmd == "run":
        Path(a.out).parent.mkdir(exist_ok=True)
        run_year(a.year, a.drafts, a.sims_per_week, not a.no_playoffs, a.out)
    else:
        report(sorted(f for pat in a.files for f in glob.glob(pat)))
