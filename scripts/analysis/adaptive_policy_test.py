"""
Does a category-aware drafting policy beat drafting by the board?

Paired mock drafts on the 2026-27 projections: for every draft slot and bot seed, my team is drafted twice with the
same bots -- (a) best available by the board, (b) src.draft.policy.AdaptivePolicy -- and the final rosters are scored by
the full daily-lineup evaluation on TWO libraries: the one the policy used (in-sample) and a fresh library with a
different seed (held out, so the policy cannot be exploiting simulation noise).

    python scripts/analysis/adaptive_policy_test.py --slots 1 5 10 --seeds 10
"""
from __future__ import annotations

import argparse
import sqlite3
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding="utf-8")

from src.draft.engine import DraftEngine
from src.draft.mock import best_available, run_mock_draft
from src.draft.policy import AdaptivePolicy, adaptive_pick, calibrate_rank_weights
from src.draft.state import LeagueSettings
from src.draft.values import CATEGORIES
from src.simulation.calibration import Calibration
from src.simulation.copula import build_library
from src.simulation.library import Library
from src.simulation.schedule import season_periods

DATA = ROOT / "data" / "processed"
KW = dict(n_teams=10, roster_size=13, active_slots=10, max_centers=3)


def main(slots, seeds, candidates, margin, stride, holdout_sims, out, seed_offset=500):
    proj = pd.read_parquet(DATA / "projections_2026.parquet")
    lib = Library.load(DATA / "library_2026.npz")
    proj = proj[proj["player_id"].isin(lib.player_ids)].reset_index(drop=True)
    t0 = time.time()
    conn = sqlite3.connect(DATA / "nba.db")
    sched = season_periods(conn, "2026-27", n_regular=18, n_playoff=0, fill_unscheduled=False)
    lib_b = build_library(proj, Calibration.load(DATA / "calibration.json"), sched, sims_per_week=holdout_sims, seed=987654)
    print(f"held-out library built in {time.time() - t0:.0f}s ({lib_b.stats.shape[1]} sims)")

    base_eng = DraftEngine(lib, proj, LeagueSettings(my_slot=5, **KW))
    pol0 = AdaptivePolicy(base_eng, n_candidates=candidates, margin=margin, sim_stride=stride)
    # measure how much of each roster rank's production the real daily lineup uses, from board-drafted rosters
    cal_rosters = []
    for sd in range(4):
        drafted = run_mock_draft(base_eng, base_eng.settings, best_available, bot_seed=sd)
        cal_rosters += [drafted[t] for t in (1, 4, 7, 10)]
    w = calibrate_rank_weights(base_eng, cal_rosters)
    print("lineup usage by roster rank (1 = best on the roster):", np.round(w, 2).tolist())
    pol0.rank_weights = w.astype(np.float32)

    rows = []
    for slot in slots:
        s = LeagueSettings(my_slot=slot, **KW)
        eng, truth = DraftEngine(lib, proj, s), DraftEngine(lib_b, proj, s)
        pol = pol0.for_engine(eng)
        for b in range(seeds):
            t1 = time.time()
            ros_board = run_mock_draft(eng, s, best_available, bot_seed=seed_offset + b)
            ros_ad = run_mock_draft(eng, s, adaptive_pick(pol), bot_seed=seed_offset + b)
            r = dict(slot=slot, seed=b)
            for name, ros in (("board", ros_board), ("adaptive", ros_ad)):
                ev_in, ev_out = eng.evaluate(ros), truth.evaluate(ros)
                r[f"in_{name}"], r[f"out_{name}"] = ev_in.win_prob[slot - 1], ev_out.win_prob[slot - 1]
                for k, c in enumerate(CATEGORIES):
                    r[f"cat_{c}_{name}"] = ev_out.cat_win_prob[slot - 1][k]
            r["changed"] = len(set(ros_board[slot]) - set(ros_ad[slot]))
            rows.append(r)
            print(f"slot {slot:2d} seed {b:2d}: held-out win% board {r['out_board']:.3f} vs adaptive {r['out_adaptive']:.3f} "
                  f"(in-sample {r['in_board']:.3f} / {r['in_adaptive']:.3f}); {r['changed']} of 13 players differ  [{time.time() - t1:.0f}s]")
    df = pd.DataFrame(rows)
    df.to_csv(out, index=False)

    def line(label, g):
        d_in, d_out = g.in_adaptive - g.in_board, g.out_adaptive - g.out_board
        se = lambda x: x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else float("nan")
        return dict(group=label, drafts=len(g), board_out=g.out_board.mean(), adaptive_out=g.out_adaptive.mean(), gain_out=d_out.mean(), se_out=se(d_out),
                    gain_in=d_in.mean(), se_in=se(d_in), adaptive_better=(d_out > 0).mean(), players_changed=g.changed.mean())
    tab = pd.DataFrame([line(f"slot {k}", g) for k, g in df.groupby("slot")] + [line("ALL", df)])
    print("\nWeekly matchup win probability, adaptive minus board (held-out library unless marked in):")
    print(tab.round(4).to_string(index=False))
    cats = pd.DataFrame({c: [df[f"cat_{c}_adaptive"].mean() - df[f"cat_{c}_board"].mean()] for c in CATEGORIES}, index=["adaptive - board"])
    print("\nPer-category win probability change (held-out):")
    print(cats.round(3).to_string())
    print(f"\nWrote {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--slots", type=int, nargs="+", default=[1, 5, 10])
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--candidates", type=int, default=12)
    ap.add_argument("--margin", type=float, default=0.003)
    ap.add_argument("--sim-stride", dest="stride", type=int, default=1)
    ap.add_argument("--holdout-sims", type=int, default=40)
    ap.add_argument("--seed-offset", type=int, default=500, help="first bot seed (tune on one range, confirm on another)")
    ap.add_argument("--out", default=str(ROOT / "exports" / "adaptive_policy_test.csv"))
    a = ap.parse_args()
    Path(a.out).parent.mkdir(exist_ok=True)
    main(a.slots, a.seeds, a.candidates, a.margin, a.stride, a.holdout_sims, a.out, a.seed_offset)
