"""
Project every remaining regular-season week and each team's final record.

    python scripts/project_season.py                     # from the last ESPN download (data/espn_cache/latest)
    python scripts/project_season.py --refresh           # check the network and download a fresh copy first
    python scripts/project_season.py --sims 20000 --seed 3
    python scripts/project_season.py --through 12        # stop after week 12 instead of week 18

Runs the engine of scripts/project_week.py (read its docstring for the assumptions) week after week, in the same simulations: each player keeps
the same true-rate draw all season, and a random injury carries on across weeks. Every simulation plays out the current week from ESPN's banked
totals and all later weeks from scratch against the ESPN schedule, adds the results to the current records, and the spread across simulations
gives each team's expected record and its 5th-95th percentile range (a 90% interval). Ties count as half a win.

Then it plays the playoffs, weeks 19-21 (6 teams, seeds 1-2 get byes, 3v6 and 4v5 in week 19, no reseeding, final in week 21; seeding ties are
broken by total categories won, then at random; a tied playoff matchup goes to the higher seed) unless --no-playoffs is given.

Prints the projected final records (with the chance of finishing top 6), the playoff odds, each team's week-by-week schedule, and writes
data/external/season_projection_records.csv and season_projection_matchups.csv (category win chances and score distribution for every matchup).

The ranges are narrower than real life: rosters stay as they are on ESPN today (no waiver pickups, trades or lineup decisions), and beyond the
known injuries only the random injury spells add uncertainty. Treat the later weeks as a rough guide.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding="utf-8")

from scripts.project_week import (CAT_NAMES, TALENT_SD, cats_won, final_totals, finishes, make_plays_fn, setup, simulate_period, week_days)
from scripts.sync_espn import EXT, standings_table

N_WEEKS = 18


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1], formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--refresh", action="store_true", help="check the network and download a fresh ESPN copy first")
    ap.add_argument("--sims", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--through", type=int, default=N_WEEKS, help="last week to project (default 18, the end of the regular season)")
    ap.add_argument("--no-csv", action="store_true", help="do not write the two CSV files")
    ap.add_argument("--no-playoffs", action="store_true", help="stop after the regular season (the playoff weeks 19-21 are simulated by default)")
    ap.add_argument("--year", type=int, default=2027)
    ap.add_argument("--db", default=str(Path(ROOT / "data/processed/nba.db")))
    args = ap.parse_args()
    E = setup(args)
    cx, mt, S, today, P = E["cx"], E["mt"], E["S"], E["today"], E["P"]
    cur_week = int(E["data"]["league"]["status"]["currentMatchupPeriod"])
    weeks = list(range(cur_week, min(args.through, N_WEEKS) + 1))
    st = standings_table(cx, mt).set_index("slot")
    names = cx.names
    rng = np.random.default_rng(args.seed)
    sims = args.sims
    print(f"Season projection from week {cur_week} to week {weeks[-1]}: {sims:,} simulations. ESPN download from {E['fetched']:%b %d %H:%M}; records so far: "
          + ", ".join(f"{names[s - 1]} {int(st.loc[s, 'W'])}-{int(st.loc[s, 'L'])}" + (f"-{int(st.loc[s, 'T'])}" if st.loc[s, "T"] else "") for s in range(1, 11)))
    for r in E["overrides"].itertuples():
        back = getattr(r, "return_date", None)
        if back is not None and pd.notna(back) and str(back) > today.isoformat():
            print(f"manual absence: {r.name} out until {back}")

    talent = {i: rng.lognormal(0, TALENT_SD, sims) for i in P.index}              # a player's true rate is one draw for the whole season
    health = {i: np.zeros(sims, dtype=bool) for i in P.index}                     # random injuries carry over from one week to the next
    wins = {s: np.full(sims, float(st.loc[s, "W"]) + 0.5 * float(st.loc[s, "T"])) for s in range(1, 11)}
    catw = {s: np.full(sims, float(st.loc[s, "cat_W"])) for s in range(1, 11)}      # total categories won: the seeding tiebreaker
    matchup_rows, schedule = [], {s: [] for s in range(1, 11)}
    t0 = time.time()
    for week in weeks:
        live = mt[mt.week == week].set_index("slot").copy()
        for c in CAT_NAMES + ["fgm", "fga", "ftm", "fta"]:
            live[c] = live[c].fillna(0) if c in live else 0
        first = today if week == cur_week else S.spans[week - 1][0].date()
        days = week_days(S, week, first)
        plays_fn = make_plays_fn(S, week - 1, sims, rng)
        rem, used = simulate_period(P, plays_fn, days, sims, rng, today, talent, health)
        fin = {s: final_totals(live.loc[s], rem[s]) for s in live.index}
        for s, r in live.iterrows():
            if r.side != "HOME":
                continue
            a, b = int(s), int(r.opp_slot)
            wa, wb = cats_won(fin[a], fin[b]), cats_won(fin[b], fin[a])
            ca, cb = wa.sum(1), wb.sum(1)
            win_a, tie = ca > cb, ca == cb
            wins[a] += win_a + 0.5 * tie
            wins[b] += (cb > ca) + 0.5 * tie
            catw[a] += ca
            catw[b] += cb
            row = dict(week=week, dates=f"{S.spans[week - 1][0]:%b %d}-{S.spans[week - 1][1]:%b %d}", team_a=names[a - 1], team_b=names[b - 1],
                       a_wins=win_a.mean(), b_wins=(cb > ca).mean(), tie=tie.mean(), a_exp_cats=ca.mean(), b_exp_cats=cb.mean(),
                       a_lineup_games=used[a].mean(), b_lineup_games=used[b].mean())
            for k, cat in enumerate(CAT_NAMES):
                row[f"a_win_{cat}"] = wa[:, k].mean()
            row["finishes_a_b"] = finishes(ca, cb)
            matchup_rows.append(row)
            schedule[a].append((week, names[b - 1], win_a.mean() + 0.5 * tie.mean()))
            schedule[b].append((week, names[a - 1], (cb > ca).mean() + 0.5 * tie.mean()))
        print(f"  week {week:2d} done ({len(days)} day(s), {time.time() - t0:.0f}s elapsed)", flush=True)

    # ---- records
    W = np.stack([wins[s] for s in range(1, 11)])                                  # 10 x sims, final wins with ties as half
    C = np.stack([catw[s] for s in range(1, 11)])
    key = W * 1000 + C + rng.random(W.shape) * 0.5                                 # seeding: wins, then total categories won, then luck
    order = (-key).argsort(axis=0)                                                 # order[k, sim] = team (0-based) holding seed k + 1
    seedpos = order.argsort(axis=0)                                                # seedpos[team, sim] = 0-based seed
    rank = seedpos + 1
    po = {t: dict(make_playoffs=(seedpos[t] < 6).mean(), bye=(seedpos[t] < 2).mean(), semis=np.nan, final=np.nan, champion=np.nan) for t in range(10)}
    bracket = None
    if weeks[-1] == N_WEEKS and not args.no_playoffs:
        zero = pd.Series({c: 0.0 for c in CAT_NAMES + ["fgm", "fga", "ftm", "fta"]})
        totals = {}
        for week in (19, 20, 21):                                                  # playoff rounds: nothing banked, one week each
            days = week_days(S, week, S.spans[week - 1][0].date())
            plays_fn = make_plays_fn(S, week - 1, sims, rng)
            rem, _ = simulate_period(P, plays_fn, days, sims, rng, today, talent, health)
            totals[week] = np.stack([final_totals(zero, rem[s]) for s in range(1, 11)])                      # 10 x sims x 9
            print(f"  playoff week {week} done ({time.time() - t0:.0f}s elapsed)", flush=True)
        ar = np.arange(sims)

        def play(hi, lo, T):                                                       # hi = the higher seed; a tied matchup goes to the higher seed
            a, b = T[hi, ar], T[lo, ar]
            return np.where(cats_won(a, b).sum(1) >= cats_won(b, a).sum(1), hi, lo)
        s1, s2, s3, s4, s5, s6 = order[:6]
        w36, w45 = play(s3, s6, totals[19]), play(s4, s5, totals[19])               # round 1: 3 v 6 and 4 v 5; seeds 1 and 2 have byes
        sf1, sf2 = play(s1, w36, totals[20]), play(s2, w45, totals[20])             # round 2: no reseeding, the winner of 3 v 6 plays seed 1
        hi = np.where(seedpos[sf1, ar] < seedpos[sf2, ar], sf1, sf2)
        champ = play(hi, np.where(hi == sf1, sf2, sf1), totals[21])                 # final
        for t in range(10):
            po[t].update(semis=((s1 == t) | (s2 == t) | (w36 == t) | (w45 == t)).mean(), final=((sf1 == t) | (sf2 == t)).mean(), champion=(champ == t).mean())
        bracket = dict(champ=champ)
    rec = []
    for s in range(1, 11):
        w = W[s - 1]
        lo, med, hi = np.percentile(w, [5, 50, 95])
        rec.append(dict(team=names[s - 1], now=f"{int(st.loc[s, 'W'])}-{int(st.loc[s, 'L'])}" + (f"-{int(st.loc[s, 'T'])}" if st.loc[s, "T"] else ""),
                        exp_wins=w.mean(), exp_losses=N_WEEKS - w.mean(), low_wins=lo, median_wins=med, high_wins=hi, exp_rank=rank[s - 1].mean(),
                        top6_chance=(rank[s - 1] <= 6).mean(), first_chance=(rank[s - 1] == 1).mean(),
                        best_finish=np.percentile(rank[s - 1], 5), worst_finish=np.percentile(rank[s - 1], 95), **po[s - 1]))
    R = pd.DataFrame(rec).sort_values("exp_wins", ascending=False).reset_index(drop=True)
    g = lambda v: f"{v:g}"
    show = pd.DataFrame({"team": R.team, "now": R.now, "expected": [f"{a:.1f}-{b:.1f}" for a, b in zip(R.exp_wins, R.exp_losses)],
                         "low (5%)": [f"{g(a)}-{g(N_WEEKS - a)}" for a in R.low_wins], "median": [f"{g(a)}-{g(N_WEEKS - a)}" for a in R.median_wins],
                         "high (95%)": [f"{g(a)}-{g(N_WEEKS - a)}" for a in R.high_wins], "avg seed": R.exp_rank.round(1),
                         "finish (90%)": [f"{a:.0f}-{b:.0f}" for a, b in zip(R.best_finish, R.worst_finish)], "top 6": [f"{100 * x:.0f}%" for x in R.top6_chance],
                         "1st": [f"{100 * x:.0f}%" for x in R.first_chance]})
    print(f"\nPROJECTED FINAL RECORDS after week {weeks[-1]} (90% range = 5th to 95th percentile of wins; ties count half):")
    print(show.to_string(index=False))
    if bracket is not None:
        P_ = R.sort_values(["champion", "final"], ascending=False)
        pct = lambda x: f"{100 * x:.0f}%"
        print("\nPLAYOFFS (weeks 19-21): top 6 qualify, seeds 1-2 get byes, 3v6 and 4v5 in week 19, no reseeding, final in week 21:")
        print(pd.DataFrame({"team": P_.team, "make playoffs": P_.make_playoffs.map(pct), "bye (top 2)": P_.bye.map(pct), "reach semis": P_.semis.map(pct),
                            "reach final": P_.final.map(pct), "champion": P_.champion.map(pct)}).to_string(index=False))
        me = names.index("Bilal")
        seed_probs = [(seedpos[me] == k).mean() for k in range(10)]
        print("\nyour seed odds: " + "  ".join(f"#{k + 1} {100 * p:.0f}%" for k, p in enumerate(seed_probs) if p >= 0.005)
              + "\nyour title chance by seed: " + "  ".join(f"#{k + 1} {100 * (bracket['champ'][seedpos[me] == k] == me).mean():.0f}%" for k in range(6) if (seedpos[me] == k).sum() > 50))
    print("\nWEEK BY WEEK (each team's chance to win its matchup):")
    for s in [1] + [x for x in range(2, 11)]:
        print(f"  {names[s - 1]:8s} " + "  ".join(f"{w}:{o[:3]} {100 * p:.0f}%" for w, o, p in schedule[s]))
    if not args.no_csv:
        M = pd.DataFrame(matchup_rows)
        R.to_csv(EXT / "season_projection_records.csv", index=False)
        M.to_csv(EXT / "season_projection_matchups.csv", index=False)
        print(f"\nwrote data/external/season_projection_records.csv ({len(R)} teams) and season_projection_matchups.csv ({len(M)} matchups)")


if __name__ == "__main__":
    main()
