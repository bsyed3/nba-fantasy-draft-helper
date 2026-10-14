"""
Does evaluating EVERY player up to a board rank (e.g. 200) pick better than evaluating a short list?

Paired mock drafts on the 2026-27 projections: for each draft slot and bot seed, my team is drafted with the same bots in
several ways, and the final rosters are scored on a HELD-OUT simulation library (different seed from the one the engine
searched), so extra candidates cannot win by fitting the engine's own simulation noise.

    python scripts/analysis/engine_candidates_test.py --arm engine:max_rank=200:margin=0.01 --seeds 6

Arms: board | adaptive | engine[:top_k=N][:max_rank=N][:margin=X][:rollouts=N]
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
from src.draft.mock import best_available, engine_pick, run_mock_draft
from src.draft.policy import AdaptivePolicy, adaptive_pick
from src.draft.state import LeagueSettings
from src.simulation.calibration import Calibration
from src.simulation.copula import build_library
from src.simulation.library import Library
from src.simulation.schedule import season_periods

DATA = ROOT / "data" / "processed"
KW = dict(n_teams=10, roster_size=13, active_slots=10, max_centers=3)


def make_strategy(spec: str, eng: DraftEngine, pol0: AdaptivePolicy):
    name, *opts = spec.split(":")
    o = {k: float(v) if "." in v else int(v) for k, v in (x.split("=") for x in opts)}
    if name == "board":
        return best_available
    if name == "adaptive":
        return adaptive_pick(pol0.for_engine(eng))
    if name == "engine":
        return engine_pick(top_k=o.get("top_k", 20), rollouts=o.get("rollouts", 2), margin=o.get("margin", 0.0), max_rank=o.get("max_rank"))
    raise SystemExit(f"unknown arm {spec}")


def main(arm, slots, seeds, seed_offset, out, holdout_sims):
    proj = pd.read_parquet(DATA / "projections_2026.parquet")
    lib = Library.load(DATA / "library_2026.npz")
    proj = proj[proj["player_id"].isin(lib.player_ids)].reset_index(drop=True)
    conn = sqlite3.connect(DATA / "nba.db")
    sched = season_periods(conn, "2026-27", n_regular=18, n_playoff=0, fill_unscheduled=False)
    lib_b = build_library(proj, Calibration.load(DATA / "calibration.json"), sched, sims_per_week=holdout_sims, seed=987654)
    pol0 = AdaptivePolicy(DraftEngine(lib, proj, LeagueSettings(my_slot=5, **KW)))
    rows = []
    for slot in slots:
        s = LeagueSettings(my_slot=slot, **KW)
        eng, truth = DraftEngine(lib, proj, s), DraftEngine(lib_b, proj, s)
        for b in range(seeds):
            t0 = time.time()
            r = dict(slot=slot, seed=b)
            for label, spec in (("board", "board"), ("arm", arm)):
                ros = run_mock_draft(eng, s, make_strategy(spec, eng, pol0), bot_seed=seed_offset + b)
                r[f"in_{label}"], r[f"out_{label}"] = eng.evaluate(ros).win_prob[slot - 1], truth.evaluate(ros).win_prob[slot - 1]
                r[f"players_{label}"] = ",".join(str(int(eng.pid[i])) for i in ros[slot])
            r["changed"] = len(set(r["players_board"].split(",")) - set(r["players_arm"].split(",")))
            rows.append(r)
            print(f"slot {slot:2d} seed {b}: held-out win% board {r['out_board']:.3f} vs {arm} {r['out_arm']:.3f}; {r['changed']} of 13 differ [{time.time() - t0:.0f}s]", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(out, index=False)
    d_out, d_in = df.out_arm - df.out_board, df.in_arm - df.in_board
    se = lambda x: x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else float("nan")
    print(f"\n{arm}: {len(df)} drafts; held-out gain over board {d_out.mean():+.4f} (se {se(d_out):.4f}); in-sample gain {d_in.mean():+.4f} (se {se(d_in):.4f}); "
          f"better in {(d_out > 0).mean():.0%}; players changed {df.changed.mean():.1f}/13")
    print(f"Wrote {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", required=True)
    ap.add_argument("--slots", type=int, nargs="+", default=list(range(1, 11)))
    ap.add_argument("--seeds", type=int, default=6)
    ap.add_argument("--seed-offset", type=int, default=7000)
    ap.add_argument("--holdout-sims", type=int, default=40)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    main(a.arm, a.slots, a.seeds, a.seed_offset, a.out, a.holdout_sims)
