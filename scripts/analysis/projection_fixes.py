"""
Projection fixes, judged by the walk-forward backtest (folds 2021-2025: train on earlier seasons, predict the next).

Questions, from the players the board ranks far below the market (Kyrie, Murray, Castle, Powell, Kessler, Siakam, Haliburton):
  1. Games played: do the team-context features help or hurt the availability model (esp. players who changed teams, and 33+)?
  2. Returners' minutes: for players with no season last year (Kyrie, Haliburton) does the model cut minutes too much? Is a blend with the
     recency baseline (as already done for availability) more accurate?
  3. Third-year players: does adding season-of-career indicators remove the model's under-projection of 3rd-season players?

    python scripts/analysis/projection_fixes.py run --variant base            # also: noctx_avail, noctx_all, expflags, expflags_noctx_avail
    python scripts/analysis/projection_fixes.py report
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

from src.features.player_seasons import load_player_season_table
from src.features.team_context import team_context
from src.models import projection as P
from src.reporting import analysis as A

DATA = ROOT / "data" / "processed"
OUT = ROOT / "exports"
FOLDS = [2021, 2022, 2023, 2024, 2025]
CTX_DROP = ("ctx_", "usage_pg")
VARIANTS = {
    "base": dict(),
    "noctx_avail": dict(drop={"avail": CTX_DROP}),
    "noctx_all": dict(drop={s: CTX_DROP for s in P.ALL_STATS}),
    "expflags": dict(exp_flags=True),
    "expflags_noctx_avail": dict(exp_flags=True, drop={"avail": CTX_DROP}),
}


def run(variant: str) -> None:
    conn = sqlite3.connect(DATA / "nba.db")
    st = load_player_season_table(conn)
    ctx = team_context(conn, list(range(int(st["year"].min()) + 1, max(FOLDS) + 1)))
    bt = P.backtest(st, FOLDS, verbose=False, context=ctx, **VARIANTS[variant])
    OUT.mkdir(exist_ok=True)
    bt.summary.to_parquet(OUT / f"projfix_{variant}_summary.parquet")
    bt.avail_eval.to_parquet(OUT / f"projfix_{variant}_avail.parquet")
    bt.oos.to_parquet(OUT / f"projfix_{variant}_oos.parquet")
    print(f"{variant}: selection {bt.selection}")


def returners() -> None:
    """Players with no season last year (gap > 0) who then played >= 500 minutes: minutes and games, model vs recency baseline vs blends."""
    conn = sqlite3.connect(DATA / "nba.db")
    st = load_player_season_table(conn)
    first = int(st["year"].min())
    ctx = team_context(conn, list(range(first + 1, max(FOLDS) + 1)))
    priors, sg = P.league_priors(st), st.groupby("year")["sg"].first().to_dict()
    wides = {t: P.build_wide(st, priors, t, sg, ctx) for t in range(first + 1, max(FOLDS) + 1)}
    sel = pd.read_parquet(OUT / "projfix_base_summary.parquet")
    sel = sel[sel.selected].set_index("stat")["model"].to_dict()
    rows = []
    for T in FOLDS:
        train = pd.concat([wides[t] for t in range(first + 1, T)], ignore_index=True)
        test = wides[T]
        preds = P.predict_all(train, test)
        last = pd.Series(np.nan, index=test.index)                  # his minutes per game the last time he played
        for l in (3, 2, 1):
            last = (test[f"min_{l}"] / test[f"gp_{l}"]).where(test[f"gp_{l}"] > 0, last)
        ok = (test["gap"] > 0).to_numpy()
        rows.append(pd.DataFrame({"year": T, "player_id": test.loc[ok, "player_id"].to_numpy(), "age": test.loc[ok, "age_t"].to_numpy(),
                                  "gap": test.loc[ok, "gap"].to_numpy(), "last_mpg": last[ok].to_numpy(), "min_t": test.loc[ok, "min_t"].to_numpy(),
                                  "act_mpg": test.loc[ok, "y_mpg"].to_numpy(), "base_mpg": preds["mpg"]["baseline"][ok], "model_mpg": preds["mpg"][sel["mpg"]][ok],
                                  "act_avail": test.loc[ok, "y_avail"].to_numpy(), "w_avail": test.loc[ok, "w_avail"].to_numpy(),
                                  "base_avail": preds["avail"]["baseline"][ok], "model_avail": preds["avail"][sel["avail"]][ok]}))
        print(f"  fold {T} done ({int(ok.sum())} returner rows)", flush=True)
    d = pd.concat(rows, ignore_index=True)
    d.to_parquet(OUT / "projfix_returners.parquet")
    # minutes: judged on returners who then played >= 500 minutes (as everywhere else); games: on every returner who appeared in at least one game
    # (the availability target is "games after his first appearance", so a returner who never plays again is not part of it) -- NOT on the >= 500 minute
    # survivors, who are healthy by construction
    mins, gms = d[d.act_mpg.notna() & (d.min_t >= 500)], d[d.act_avail.notna() & (d.w_avail > 0)]
    print(f"\n{len(mins)} returner seasons with >= 500 minutes (minutes test) and {len(gms)} who appeared in a game (games test), 2021-2025")
    for label, mask in (("all returners", lambda g: g.last_mpg > 0), ("former rotation stars (last mpg >= 28)", lambda g: g.last_mpg >= 28),
                        ("last mpg < 28", lambda g: g.last_mpg < 28), ("age 30+", lambda g: g.age >= 30), ("age < 30", lambda g: g.age < 30)):
        g, ga = mins[mask(mins)], gms[mask(gms)]
        if len(g) < 5:
            continue
        print(f"\n  {label}: minutes n={len(g)} (average last mpg {g.last_mpg.mean():.1f} -> actual {g.act_mpg.mean():.1f}); games n={len(ga)}")
        for name, pred in (("model (selected, as shipped)", g.model_mpg), ("recency baseline", g.base_mpg), ("blend 25% baseline", .25 * g.base_mpg + .75 * g.model_mpg),
                           ("blend 50% baseline", .5 * g.base_mpg + .5 * g.model_mpg), ("blend 75% baseline", .75 * g.base_mpg + .25 * g.model_mpg)):
            print(f"     minutes: {name:30s} {_stats(g.act_mpg - pred)}")
        if len(ga):
            print(f"     games (share of season; x82 for games): shipped model (50/50 blend) {_stats(ga.act_avail - ga.model_avail)} | recency baseline {_stats(ga.act_avail - ga.base_avail)}")


def _stats(err: pd.Series) -> str:
    return f"n={len(err):4d} MAE {err.abs().mean():.3f} bias {err.mean():+.3f}"


def report() -> None:
    names = [v for v in VARIANTS if (OUT / f"projfix_{v}_summary.parquet").exists()]
    S = {v: pd.read_parquet(OUT / f"projfix_{v}_summary.parquet") for v in names}
    E = {v: pd.read_parquet(OUT / f"projfix_{v}_avail.parquet") for v in names}
    O = {v: pd.read_parquet(OUT / f"projfix_{v}_oos.parquet") for v in names}
    print("variants:", names)

    print("\n=== 1. Pooled out-of-sample error of the SELECTED model per stat (weighted MSE; % change vs base, negative = better)")
    sel = {v: S[v][S[v].selected].set_index("stat")["wmse"] for v in names}
    tab = pd.DataFrame(sel)
    pct = tab.div(tab["base"], axis=0).sub(1).mul(100)
    key = [s for s in ("avail", "mpg", "fga2", "fga3", "fta", "reb", "ast", "stl", "blk", "tov", "p2", "p3", "ft") if s in pct.index]
    print(pct.loc[key].round(2).to_string())
    print("mean over all stats:", pct.mean().round(2).to_dict())

    print("\n=== 2. Games played (availability), established players, by group: out-of-sample error of the model (act - pred, as share of games)")
    for v in names:
        e = E[v].assign(err=lambda d: d.act_avail - d.pred_avail)
        grp = {"all": e, "moved teams": e[e.moved == 1], "returners (no season last year)": e[e.gap > 0], "age 33+": e[e.age >= 33],
               "age 30-32": e[(e.age >= 30) & (e.age < 33)], "stayed, age < 30": e[(e.moved == 0) & (e.age < 30)]}
        print(f"  [{v}]")
        for k, g in grp.items():
            if len(g):
                print(f"     {k:34s} {_stats(g.err)}   (x82 = games: MAE {g.err.abs().mean() * 82:.1f}, bias {g.err.mean() * 82:+.1f})")
    base = E["base"].assign(err=lambda d: d.act_avail - d.base_avail)
    print("  [recency baseline alone, no model]")
    for k, g in {"all": base, "moved teams": base[base.moved == 1], "returners": base[base.gap > 0], "age 33+": base[base.age >= 33]}.items():
        if len(g):
            print(f"     {k:34s} {_stats(g.err)}")

    print("\n=== 3. Returners' minutes per game (players with no season last year, who then played): model vs recency baseline vs blends")
    e = E["base"]
    for label, g in (("returners (gap>0)", e[e.gap > 0]), ("everyone else (control)", e[e.gap == 0])):
        print(f"  {label}, n={len(g)}")
        for name, pred in (("model (selected)", g.pred_mpg), ("recency baseline", g.base_mpg),
                           ("blend 25% baseline", .25 * g.base_mpg + .75 * g.pred_mpg), ("blend 50% baseline", .5 * g.base_mpg + .5 * g.pred_mpg),
                           ("blend 75% baseline", .75 * g.base_mpg + .25 * g.pred_mpg)):
            print(f"     {name:20s} {_stats(g.act_mpg - pred)}")

    print("\n=== 4. Third-season players (exp == 2) vs everyone: actual minus projected per game, selected models")
    apg = A.actual_per_game(load_player_season_table(sqlite3.connect(DATA / "nba.db")))
    for v in names:
        o = O[v].merge(apg[["player_id", "year", "pts", "reb", "ast", "mpg"]].rename(columns=lambda c: f"act_{c}" if c not in ("player_id", "year") else c),
                       on=["player_id", "year"], how="left")
        rows = []
        for lab, g in (("2nd year (exp 1)", o[o.exp == 1]), ("3rd year (exp 2)", o[o.exp == 2]), ("4th year (exp 3)", o[o.exp == 3]), ("5+ years", o[o.exp >= 4]), ("everyone", o)):
            rows.append(dict(group=lab, n=len(g), MPG=(g.act_mpg - g.mpg).mean(), PPG=(g.act_pts - g.pts).mean(), REB=(g.act_reb - g.reb).mean(),
                             AST=(g.act_ast - g.ast).mean(), MAE_PPG=(g.act_pts - g.pts).abs().mean()))
        print(f"  [{v}]\n" + pd.DataFrame(rows).round(2).to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--variant", choices=list(VARIANTS), required=True)
    sub.add_parser("report")
    sub.add_parser("returners")
    a = ap.parse_args()
    {"run": lambda: run(a.variant), "report": report, "returners": returners}[a.cmd]()
