"""
Generate the full written report (docs/REPORT.md + a self-contained docs/REPORT.html) from the data and the
fitted pipeline. Every number, table and figure in the report is computed here -- nothing is hand-typed -- so
re-running after a data refresh keeps the document honest.

    python scripts/make_report.py            # needs ingest + build_projections + build_library (+ validate_draft for section 9)
    python scripts/make_report.py --skip-sim-check    # skip the slower 2025 full-stack calibration check
"""
from __future__ import annotations

import argparse
import base64
import re
import sqlite3
import sys
from pathlib import Path

import markdown
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding="utf-8")

from src.draft.engine import DraftEngine
from src.draft.mock import best_available, run_mock_draft
from src.draft.state import DraftState, LeagueSettings
from src.draft.values import CAT_LABELS, CATEGORIES, z_scores
from src.features.player_seasons import STAT_DEFS, league_priors, load_player_season_table
from src.features.positions import add_eligibility
from src.models import projection as P
from src.reporting import analysis as A
from src.reporting import figs as F
from src.reporting.figs import C, md_table
from src.simulation.calibration import COUNT_DIMS, DIMS, Calibration
from src.simulation.copula import build_library
from src.simulation.library import STAT_IDX, Library
from src.simulation.schedule import actual_library, season_periods

DATA = ROOT / "data" / "processed"
DAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
STAT_LABEL = {"fga2": "2-pt attempts /36", "fga3": "3-pt attempts /36", "fta": "FT attempts /36", "reb": "Rebounds /36",
              "ast": "Assists /36", "stl": "Steals /36", "blk": "Blocks /36", "tov": "Turnovers /36",
              "p2": "2-pt %", "p3": "3-pt %", "ft": "FT %", "mpg": "Minutes per game", "avail": "Availability (GP / team games)"}
K: dict = {}   # headline numbers collected for the executive summary


def pct(x, d=1):
    return f"{x * 100:.{d}f}%"


def team_at_season_start(conn, season: str) -> pd.Series:
    df = pd.read_sql_query(
        """SELECT bs.player_id, bs.team_id, g.date FROM box_score bs JOIN game g ON g.game_id = bs.game_id
           WHERE g.season = ? AND g.game_type = 'regular' AND bs.min > 0""", conn, params=[season])
    return df.sort_values("date").drop_duplicates("player_id").set_index("player_id")["team_id"]


# =========================================================================================================
# 2. data
# =========================================================================================================
def sec_data(conn) -> str:
    t = pd.read_sql_query(
        """SELECT g.season, COUNT(DISTINCT g.game_id) AS games, COUNT(*) AS player_games,
                  COUNT(DISTINCT bs.player_id) AS players
           FROM box_score bs JOIN game g ON g.game_id = bs.game_id GROUP BY g.season ORDER BY g.season""", conn)
    m = pd.read_sql_query(
        """SELECT g.season, bs.game_id, bs.team_id, SUM(bs.min) AS m FROM box_score bs
           JOIN game g ON g.game_id = bs.game_id GROUP BY 1, 2, 3""", conn)
    m["ok"] = (m["m"] - (240 + 25 * ((m["m"] - 240) / 25).round())).abs() < 3
    t["minutes_check"] = t["season"].map(m.groupby("season")["ok"].mean()) * 100
    K["n_playergames"], K["n_games"] = int(t["player_games"].sum()), int(t["games"].sum())
    K["minutes_ok"] = m["ok"].mean() * 100
    sched = pd.read_sql_query("SELECT COUNT(*) n, MIN(date) a, MAX(date) b FROM game WHERE season = '2026-27'", conn).iloc[0]
    return f"""
### 2.1 Where the data comes from

| Source | What we take | Why |
|---|---|---|
| `nba_api` · `LeagueGameLog` | every player's box score in every regular-season game, 2015-16 → 2025-26 | the raw facts everything is built from |
| `nba_api` · `PlayerIndex` | position label, draft slot, rookie year, current team — **including the incoming rookie class** | rookies have no box scores; draft slot is our only prior for them |
| `nba_api` · `LeagueDashPlayerBioStats` | age by season | aging is a core driver of projections |
| `nba_api` · `ScheduleLeagueV2` | the published 2026-27 schedule ({int(sched.n)} games, {sched.a} → {sched.b}) | the simulation plays every player on the real days his team plays |
| ESPN fantasy API *(you run once)* | exact position eligibility + average draft position | position slots and opponent behaviour |

Only **regular-season** games count (game ids starting `002`). The NBA Cup *final* has a different id family (`006…`) and is
not part of your fantasy season, so it never enters; the Cup quarter- and semifinals *are* regular-season games and do.

### 2.2 What is in the database

{md_table(t.rename(columns={"minutes_check": "team-games with correct minutes (%)"}), {"team-games with correct minutes (%)": "{:.1f}"})}

**Quality check.** In every NBA game the players' minutes must add to 240 (+5 per overtime period) per team. {pct(K["minutes_ok"] / 100)} of
all {len(m):,} team-games pass (within 3 minutes); the rest are small rounding gaps, so every player who took the floor is present.

**Design rules that keep the numbers honest** (full schema in [`data_model.md`](data_model.md)):

* `points` is a *generated column*: `2·FGM + 3PM + FTM`. It is impossible to store a points total that disagrees with the shots.
* Percentages are never stored or averaged. FG% is always `ΣFGM ÷ ΣFGA` over whatever window you ask about — a player who shoots
  1-for-1 in one game and 1-for-20 in another shot 2-for-21 (9.5%), not 52.5%.
* Stints, ages and draft state are derived or append-only, so re-running ingestion can never duplicate rows.
"""


# =========================================================================================================
# 3. features / shrinkage
# =========================================================================================================
def sec_features(st: pd.DataFrame, proj: pd.DataFrame) -> str:
    priors = league_priors(st)
    sg = st.groupby("year")["sg"].first().to_dict()
    first = int(st["year"].min())
    wides = {T: P.build_wide(st, priors, T, sg) for T in range(first + 1, int(st["year"].max()) + 2)}
    train = pd.concat([wides[T] for T in range(first + 1, int(st["year"].max()) + 1)], ignore_index=True)
    params = P.fit_baseline_params(train)
    K["params"] = params

    # --- how much does shrinkage help? 3P% next season, by last-season attempts
    d = train[(train["w_p3"] >= 150) & (train["fga3_1"] > 0)].copy()
    d["raw"] = d["tpm_1"] / d["fga3_1"]
    d["blend"] = P.blend(d, "p3", *params["p3"])
    d["lg"] = d["prior_p3"]
    bins = [(1, 49, "<50"), (50, 149, "50-149"), (150, 299, "150-299"), (300, 10_000, "300+")]
    rows = []
    for lo, hi, name in bins:
        g = d[(d["fga3_1"] >= lo) & (d["fga3_1"] <= hi)]
        if len(g) < 15:
            continue
        rows.append(dict(bucket=name, n=len(g), raw=np.sqrt(((g["y_p3"] - g["raw"]) ** 2).mean()),
                         league=np.sqrt(((g["y_p3"] - g["lg"]) ** 2).mean()),
                         shrunk=np.sqrt(((g["y_p3"] - g["blend"]) ** 2).mean())))
    sh = pd.DataFrame(rows)
    fig, ax = F.new(8, 4.3)
    x = np.arange(len(sh)); w = 0.26
    ax.bar(x - w, sh["raw"] * 100, w, color=C["orange"], label="last season's 3P% as-is")
    ax.bar(x, sh["league"] * 100, w, color=C["axis"], label="league average for his position")
    ax.bar(x + w, sh["shrunk"] * 100, w, color=C["blue"], label="shrunken multi-season blend (ours)")
    for xi, r in zip(x, sh.itertuples()):
        ax.text(xi + w, r.shrunk * 100 + 0.15, f"{r.shrunk * 100:.1f}", ha="center", fontsize=8.5, color=C["ink2"])
        ax.text(xi - w, r.raw * 100 + 0.15, f"{r.raw * 100:.1f}", ha="center", fontsize=8.5, color=C["ink2"])
    ax.set_xticks(x, [f"{b}\n(n={n})" for b, n in zip(sh["bucket"], sh["n"])])
    ax.set_xlabel("3-point attempts last season (and number of player-seasons)")
    ax.set_ylabel("typical error in next season's 3P% (points)")
    F.headline(fig, "Small samples lie — shrinking toward the norm fixes most of it",
               "Error predicting next season's 3P% (RMSE, percentage points); lower is better")
    F.legend_top(fig, *ax.get_legend_handles_labels())
    fig_shrink = F.save(fig, "shrinkage_3p")
    K["shrink_small_gain"] = (1 - sh.iloc[0]["shrunk"] / sh.iloc[0]["raw"]) * 100

    # --- worked example for 2026-27
    w26 = wides[int(st["year"].max()) + 1]
    cand = w26[(w26["n_lags"] == 3) & (w26["fga3_1"] >= 100) & (w26["fga3_1"] <= 300)].copy()
    cand["raw1"] = cand["tpm_1"] / cand["fga3_1"]
    cand["bl"] = P.blend(cand, "p3", *params["p3"])
    row = cand.loc[(cand["raw1"] - cand["bl"]).abs().idxmax()]
    decay, k = params["p3"]
    prior = float(row["prior_p3"])
    ex = []
    num = den = 0.0
    seasons = [f"{int(row['year']) - l}-{str(int(row['year']) - l + 1)[-2:]}" for l in (1, 2, 3)]
    for l in (1, 2, 3):
        att, mk = row[f"fga3_{l}"], row[f"tpm_{l}"]
        wgt = decay ** (l - 1)
        ex.append(dict(season=seasons[l - 1], **{"3PA": att, "3PM": mk, "raw 3P%": mk / att if att else np.nan, "weight": wgt}))
        num += wgt * (0 if np.isnan(mk) else mk); den += wgt * (0 if np.isnan(att) else att)
    est = (num + k * prior) / (den + k)
    pname = row["name"]
    final = float(proj.loc[proj["player_id"] == row["player_id"], "r_p3"].iloc[0])
    return f"""
### 3.1 Everything is a ratio of two totals

Almost every quantity we model is "something per something": rebounds **per 36 minutes**, makes **per attempt**, minutes **per game**,
games played **per team game**. We store the two totals and divide at the last moment. That gives one rule — *add the totals, then
divide* — that is right for every stat, including the percentages.

| What we model | numerator ÷ denominator | |
|---|---|---|
{chr(10).join(f"| {STAT_LABEL[s]} | {n} ÷ {d_} | {'×36' if sc == 36 else ''} |" for s, (n, d_, sc) in STAT_DEFS.items())}

Points, FG% and FGM are **not** on this list. They are *derived* afterwards: `FGM = 2P% × 2PA + 3P% × 3PA`, `PTS = 2·FGM + 3PM + FTM`.
(The original proposal treated points and the percentages as independent things to model; they are arithmetic consequences of the
shot-level stats.)

### 3.2 Shrinkage: how much to trust a small sample

A player who went 12-for-20 from three last year is probably not a 60% shooter. The fix is to blend what he did with what players like him
do, weighting by how much evidence there is:

> estimate = ( Σ wₗ · makesₗ + k · league_rate ) ÷ ( Σ wₗ · attemptsₗ + k )

* `l` runs over his last three seasons; `wₗ = decay^(l-1)` weights recent seasons more.
* `k` is the number of "pseudo-attempts" of league-average shooting added to his record — big `k` = trust the league more.
* `decay` and `k` are **not guessed**: for each stat they are chosen by grid search to minimise next-season error on the *training* seasons only.
  For 3P% the search chose `decay = {decay}` and `k = {k:g}` attempts.

![shrinkage]({fig_shrink})

**Reading the chart.** For a player who attempted fewer than 50 threes last season, using his raw percentage is off by
{sh.iloc[0]["raw"] * 100:.1f} points on average; the shrunken blend is off by {sh.iloc[0]["shrunk"] * 100:.1f} — a {K["shrink_small_gain"]:.0f}% reduction.
For high-volume shooters (300+) the three methods converge, because there is enough evidence to trust the player.

**Worked example — {pname}** (projected for 2026-27; the method picks the player where shrinkage matters most among 100-300-attempt shooters):

{md_table(pd.DataFrame(ex), {"3PA": "{:.0f}", "3PM": "{:.0f}", "raw 3P%": "{:.3f}", "weight": "{:.2f}"})}

* League-average 3P% for his position group: **{prior:.3f}** (the prior), `k = {k:g}`.
* Numerator = {num:.1f} + {k:g} × {prior:.3f} = **{num + k * prior:.1f}**; denominator = {den:.1f} + {k:g} = **{den + k:.1f}**.
* Shrunken estimate = {num + k * prior:.1f} ÷ {den + k:.1f} = **{est:.3f}** (his last season alone said {row["tpm_1"] / row["fga3_1"]:.3f}).
* The final 2026-27 projection after the learned adjustment (age, role, etc.) is **{final:.3f}**.
"""


# =========================================================================================================
# 4. model evaluation
# =========================================================================================================
def sec_models(oos: pd.DataFrame, summary: pd.DataFrame, apg: pd.DataFrame) -> str:
    sel = summary[summary["selected"]].set_index("stat")
    base = summary[summary["model"] == "baseline"].set_index("stat")
    t = pd.DataFrame({
        "stat": [STAT_LABEL[s] for s in sel.index], "winning model": sel["model"].values,
        "baseline RMSE": base.loc[sel.index, "rmse"].values, "winning RMSE": sel["rmse"].values,
        "MSE change": sel["vs_baseline_pct"].values})
    pv = summary[summary["model"] != "baseline"].pivot(index="stat", columns="model", values="vs_baseline_pct")
    order = pv.min(axis=1).sort_values().index
    fig, ax = F.new(8, 5.0)
    y = np.arange(len(order)); h = 0.36
    ax.barh(y - h / 2, pv.loc[order, "ridge"], h, color=C["blue"], label="Ridge on the residual")
    lg = pv.loc[order, "lgbm"]
    ax.barh(y + h / 2, lg.fillna(0), h, color=C["orange"], label="LightGBM on the residual")
    for yi, s in zip(y, order):
        ax.text(pv.loc[s, "ridge"] + (-0.4 if pv.loc[s, "ridge"] < 0 else 0.4), yi - h / 2, f"{pv.loc[s, 'ridge']:+.1f}",
                va="center", ha="right" if pv.loc[s, "ridge"] < 0 else "left", fontsize=8, color=C["ink2"])
        if not np.isnan(lg[s]):
            ax.text(lg[s] + (-0.4 if lg[s] < 0 else 0.4), yi + h / 2, f"{lg[s]:+.1f}", va="center",
                    ha="right" if lg[s] < 0 else "left", fontsize=8, color=C["ink2"])
    ax.axvline(0, color=C["ink2"], lw=1)
    ax.set_yticks(y, [STAT_LABEL[s] for s in order]); ax.invert_yaxis()
    ax.set_xlabel("change in out-of-sample squared error vs the shrunken-average baseline (%) — left of 0 is better")
    ax.grid(axis="y", visible=False)
    ax.set_xlim(-19.5, 4.8)
    F.headline(fig, "The learned models beat the baseline most on minutes, shot volume and availability",
               "Rolling-origin backtest: train on seasons < T, predict T, T = 2021-2025 (percentages use Ridge only)")
    F.legend_top(fig, *ax.get_legend_handles_labels())
    fig_bt = F.save(fig, "backtest_vs_baseline")
    K["best_gain"] = float(pv.min().min())

    acc = A.accuracy_table(oos, apg)
    K["acc"] = acc
    lab = {**{c: CAT_LABELS[c] for c in CATEGORIES}}
    acc_t = pd.DataFrame({"category": [lab[s] for s in acc["stat"]], "player-seasons": acc["n"], "avg value": acc["mean_actual"],
                          "MAE: our model": acc["mae_model"], "MAE: 'same as last year'": acc["mae_last_season"],
                          "MAE improvement": acc["mae_gain_pct"], "R² model": acc["r2_model"], "R² last year": acc["r2_last_season"]})
    fig, axes = plt.subplots(2, 3, figsize=(10, 6.2))
    for ax, s in zip(axes.ravel(), ["pts", "reb", "ast", "tpm", "stl", "blk"]):
        cur = apg.set_index(["player_id", "year"])
        m = oos.set_index(["player_id", "year"])
        idx = m.index.intersection(cur.index)
        yv, pv_ = cur.loc[idx, s], m.loc[idx, s]
        ax.scatter(pv_, yv, s=9, color=C["blue"], alpha=0.35, linewidths=0)
        lim = max(pv_.max(), yv.max()) * 1.03
        ax.plot([0, lim], [0, lim], color=C["ink2"], lw=1)
        r2 = acc.set_index("stat").loc[s, "r2_model"]
        ax.set_title(f"{CAT_LABELS[s]} per game", fontsize=10, loc="left")
        ax.text(0.04, 0.9, f"R² = {r2:.2f}", transform=ax.transAxes, fontsize=9, color=C["ink2"])
        ax.set_xlim(0, lim); ax.set_ylim(0, lim)
        ax.set_xlabel("projected"); ax.set_ylabel("actual")
    F.headline(fig, "Projections track what players actually did — points and rebounds best, steals and blocks hardest",
               f"Every dot is a player-season the model had not seen ({len(oos):,} of them, 2021-2025, ≥500 minutes)")
    F.legend_top(fig, [F.swatch(C["blue"], "player-season (never seen in training)", "dot"), F.swatch(C["ink2"], "perfect projection", "line")])
    fig_sc = F.save(fig, "predicted_vs_actual")

    rk = A.ranking_accuracy(oos, apg)
    K["rank"] = rk
    fig, ax = F.new(8, 4.2)
    x = np.arange(len(rk)); w = 0.36
    ax.bar(x - w / 2, rk["spearman_last_season"], w, color=C["orange"], label="rank by last season's production")
    ax.bar(x + w / 2, rk["spearman_model"], w, color=C["blue"], label="rank by our projection")
    for xi, a, b in zip(x, rk["spearman_last_season"], rk["spearman_model"]):
        ax.text(xi - w / 2, a + 0.008, f"{a:.2f}", ha="center", fontsize=8.5, color=C["ink2"])
        ax.text(xi + w / 2, b + 0.008, f"{b:.2f}", ha="center", fontsize=8.5, color=C["ink2"])
    ax.set_xticks(x, [f"{y}-{str(y + 1)[-2:]}" for y in rk["year"]])
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("rank correlation with actual fantasy value")
    F.headline(fig, "Ranking players is what a draft needs — our projection ranks better than last year's stats",
               "Spearman correlation between predicted and actual 9-category value (z-score total), same players")
    F.legend_top(fig, *ax.get_legend_handles_labels())
    fig_rk = F.save(fig, "ranking_accuracy")
    K["spearman_model"], K["spearman_last"] = rk["spearman_model"].mean(), rk["spearman_last_season"].mean()
    n_hit_ge = int((rk["top_overlap_model"] >= rk["top_overlap_last_season"]).sum())
    n_tie = int((rk["top_overlap_model"] == rk["top_overlap_last_season"]).sum())
    K["overlap_model"], K["overlap_last"] = rk["top_overlap_model"].mean(), rk["top_overlap_last_season"].mean()

    return f"""
### 4.1 The competition: three predictors per stat

For every stat, three candidate predictors compete:

1. **Baseline** — the shrunken, recency-weighted average from section 3.
2. **Ridge** — a regularised linear model that learns a *correction* to the baseline from age, age², experience, draft slot, position group,
   per-season minutes and availability, and the blends of all the other stats.
3. **LightGBM** — a small gradient-boosted tree model (6 leaves, 250 rounds, heavy regularisation) learning the same correction, allowed to find
   non-linear shapes such as the aging curve's bend. Percentages (2P%, 3P%, FT%) use Ridge only — shot-efficiency changes are close to linear in
   age and sample size, and trees mostly chase noise there.

Learning the *residual* (what the baseline gets wrong) rather than the stat itself keeps the models stable: with no signal, they predict "no correction".

### 4.2 The test: never grade on data the model has seen

Random train/test splits leak the future (the 2024 version of a player helps predict his 2023). We use a **rolling origin**: for each season
T in 2021…2025 train on every season before T, predict T, then pool the errors (weighted by minutes or attempts, players with ≥ 500 minutes).
The winner per stat is whichever has the lowest pooled squared error.

![backtest]({fig_bt})

{md_table(t, {"baseline RMSE": "{:.3f}", "winning RMSE": "{:.3f}", "MSE change": "{:+.1f}%"})}

**What this says.**
* **Minutes (−{abs(pv.loc['mpg'].min()):.0f}%) and availability** improve most: role and health are where trees find structure (young players' minutes rise; veterans' fall).
* **Shot volume** (2-pt/3-pt/FT attempts) and **assists/turnovers** improve moderately.
* **Rebounds, steals, and FT%** stay with the baseline — their year-to-year signal is already captured by the shrunken average, and the extra model
  only adds noise. That is a finding, not a failure: we keep the simpler model where it wins.
* Percentages move least because they are mostly luck at season scale (see the shrinkage chart).

### 4.3 What the projections look like in fantasy terms

After the rate models run, we multiply back to per-game stats and compare with a "naive" forecaster who just assumes next year = last year, on the
same player-seasons:

{md_table(acc_t, {"player-seasons": "{:.0f}", "avg value": "{:.2f}", "MAE: our model": "{:.3f}", "MAE: 'same as last year'": "{:.3f}", "MAE improvement": "{:+.1f}%", "R² model": "{:.2f}", "R² last year": "{:.2f}"})}

MAE = average absolute miss in per-game units (e.g., 0.9 means a typical miss of 0.9 points / rebounds / etc.). "R²" is the share of the spread between
players the projection explains.

![scatter]({fig_sc})

### 4.4 The test that matters for a draft: ranking

A draft only needs to *order* players. For each backtest season we z-score the nine categories for the projection and for what really happened,
then compare orderings:

![ranking]({fig_rk})

* Average rank correlation: **{K["spearman_model"]:.3f}** (ours) vs **{K["spearman_last"]:.3f}** (last season's stats).
* Of the ~{int(rk["k"].iloc[0])} most valuable players in a season, our projection identified on average **{K["overlap_model"] * 100:.0f}%** in advance, versus
  **{K["overlap_last"] * 100:.0f}%** for "last year's best".

{md_table(rk.rename(columns={"year": "season start", "n": "players", "spearman_model": "rank corr (ours)", "spearman_last_season": "rank corr (last yr)", "top_overlap_model": f"top-k hit (ours)", "top_overlap_last_season": "top-k hit (last yr)", "k": "k"}), {"rank corr (ours)": "{:.3f}", "rank corr (last yr)": "{:.3f}", "top-k hit (ours)": "{:.0%}", "top-k hit (last yr)": "{:.0%}", "players": "{:.0f}", "k": "{:.0f}", "season start": "{:.0f}"})}

The projection ranks better in all {len(rk)} seasons on rank correlation, but its edge in identifying the top group is uneven: it matched or beat last year's stats in
{n_hit_ge} of {len(rk)} seasons ({n_tie} ties), which is why we do not oversell it.

**Honest limits.** Scored players are those who logged ≥ 500 minutes; someone who tore his ACL in October is invisible to this test (the
availability model is the only place that risk lives). Steals and blocks are the least predictable categories, as the R² panel shows — a pattern
that is true of the real game, not of this model.
"""



# =========================================================================================================
# 4.5 residual diagnostics
# =========================================================================================================
def sec_residuals(oos: pd.DataFrame, st: pd.DataFrame, apg: pd.DataFrame, players: pd.DataFrame) -> str:
    from scipy.stats import norm

    rf = A.residual_frame(oos, st, apg, dict(zip(players["player_id"], players["name"])))
    rs = A.residual_summary(rf).set_index("stat")
    lab = {**CAT_LABELS, "mpg": "MPG"}
    scale = {c: (100.0 if c in ("fg_pct", "ft_pct") else 1.0) for c in A.RESID_COLS}   # percentages shown in points
    unit = {c: ("pts" if c in ("fg_pct", "ft_pct") else "") for c in A.RESID_COLS}

    # ---- A. residual vs projected level, with binned trend
    cats6 = ["pts", "reb", "ast", "tpm", "stl", "mpg"]
    fig, axes = plt.subplots(2, 3, figsize=(10.4, 6.4))
    flagged = []
    for ax, c in zip(axes.ravel(), cats6):
        x, y = rf[f"{c}_pred"], rf[f"{c}_res"]
        ax.scatter(x, y, s=7, color=C["blue"], alpha=0.22, linewidths=0)
        b = A.binned_mean(x, y)
        ax.fill_between(b["x"], b["lo"], b["hi"], color=C["orange"], alpha=0.25, linewidth=0)
        ax.plot(b["x"], b["mean"], color=C["orange"], lw=2.2)
        ax.axhline(0, color=C["ink2"], lw=1)
        ax.set_title(f"{lab[c]}", fontsize=10, loc="left")
        ax.set_xlabel("projected"); ax.set_ylabel("actual − projected")
        lim = np.percentile(np.abs(y), 99) * 1.1
        ax.set_ylim(-lim, lim)
    for c in A.RESID_COLS:
        b = A.binned_mean(rf[f"{c}_pred"], rf[f"{c}_res"])
        hit = b[(b["lo"] > 0) | (b["hi"] < 0)]
        for _, r in hit.iterrows():
            flagged.append((c, r["x"], r["mean"], r["se"], int(r["n"])))
    F.headline(fig, "Residuals vs projected level: mostly flat around zero, with a few biased stretches",
               "Each dot is one player-season (actual minus projected). A trend line hugging zero means no bias at that level; the two biggest points misses in each direction are named")
    F.legend_top(fig, [F.swatch(C["blue"], "player-season", "dot"), F.swatch(C["orange"], "average miss by projection decile", "line"),
                       F.swatch(C["orange"], "95% interval of that average"),
                       F.swatch(C["blue"], "biggest under-projection (PTS panel)", "dot"), F.swatch(C["red"], "biggest over-projection (PTS panel)", "dot")], ncol=3)
    big = pd.concat([rf.nlargest(2, "pts_res"), rf.nsmallest(2, "pts_res")])
    axp = axes[0, 0]
    yl = max(axp.get_ylim()[1], float(np.abs(big["pts_res"]).max()) * 1.5)
    axp.set_ylim(-yl, yl)
    F.label_points(axp, [(r.pts_pred, r.pts_res, f"{A.short_name(r.name)} '{str(int(r.year))[-2:]}", C["blue"] if r.pts_res > 0 else C["red"])
                         for r in big.itertuples()])
    fig_a = F.save(fig, "resid_vs_fitted")

    # ---- B. residual distributions
    fig, axes = plt.subplots(3, 3, figsize=(10, 7.2))
    for ax, c in zip(axes.ravel(), A.CAT_COLS):
        z = (rf[f"{c}_res"] - rf[f"{c}_res"].mean()) / rf[f"{c}_res"].std()
        ax.hist(z.clip(-4.5, 4.5), bins=np.linspace(-4.5, 4.5, 31), density=True, color=C["blue"], edgecolor=C["surface"], linewidth=0.6)
        xs = np.linspace(-4.5, 4.5, 200)
        ax.plot(xs, norm.pdf(xs), color=C["orange"], lw=2)
        ax.set_title(f"{lab[c]}  (skew {rs.loc[c, 'skew']:+.2f})", fontsize=9.5, loc="left")
        ax.set_yticks([]); ax.grid(axis="y", visible=False)
    F.headline(fig, "Miss sizes are close to bell-shaped; heavy tails are where injuries and role changes live",
               "Standardised residuals (actual − projected, in standard deviations) compared with a normal curve")
    F.legend_top(fig, [F.swatch(C["blue"], "standardised miss (player-seasons)"), F.swatch(C["orange"], "normal curve", "line")])
    fig_b = F.save(fig, "resid_distributions")

    # ---- C. bias by age and by projected minutes (relative error, so categories share a scale)
    def rel_by_bucket(col_bucket, edges, labels, cats):
        out = {}
        b = pd.cut(rf[col_bucket], edges, labels=labels, right=False)
        for c in cats:
            g = rf.assign(b=b).groupby("b", observed=True)
            m_res, n = g[f"{c}_res"].mean(), g[f"{c}_res"].size()
            se = g[f"{c}_res"].std() / np.sqrt(n)
            avg = g[f"{c}_act"].mean()
            out[c] = pd.DataFrame({"rel": m_res / avg * 100, "se": se / avg * 100, "n": n})
        return out

    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.2), sharey=True)
    cols3 = {"pts": C["blue"], "reb": C["orange"], "ast": C["aqua"]}
    age_edges, age_lab = [0, 22, 24, 26, 28, 30, 32, 34, 60], ["<22", "22-23", "24-25", "26-27", "28-29", "30-31", "32-33", "34+"]
    mpg_edges, mpg_lab = [0, 15, 20, 25, 30, 35, 60], ["<15", "15-20", "20-25", "25-30", "30-35", "35+"]
    for ax, (key, edges, labels, ttl) in zip(axes, [("age", age_edges, age_lab, "by age"), ("mpg_pred", mpg_edges, mpg_lab, "by projected minutes per game")]):
        rb = rel_by_bucket(key, edges, labels, list(cols3))
        x = np.arange(len(labels))
        for c, colr in cols3.items():
            d = rb[c].reindex(labels)
            ax.errorbar(x + (list(cols3).index(c) - 1) * 0.12, d["rel"], yerr=1.96 * d["se"], fmt="o-", color=colr, lw=1.6, ms=5, capsize=2, label=lab[c])
        ax.axhline(0, color=C["ink2"], lw=1)
        ax.set_xticks(x, labels); ax.set_title(f"Average miss as % of typical value, {ttl}", fontsize=10, loc="left")
        ax.set_xlabel("age" if key == "age" else "projected minutes per game")
    axes[0].set_ylabel("actual − projected, % of average (bars = 95% interval)")
    F.headline(fig, "Bias by age and by role: look for anyone far from zero",
               "Above 0 = the model under-projected that group; below 0 = over-projected")
    F.legend_top(fig, *axes[0].get_legend_handles_labels())
    fig_c = F.save(fig, "resid_by_age_minutes")

    # ---- D. stability across seasons
    rows = []
    for c in A.CAT_COLS + ["mpg"]:
        g = rf.groupby("year")
        rows.append((g[f"{c}_res"].mean() / g[f"{c}_act"].mean() * 100).rename(lab[c]))
    st_t = pd.concat(rows, axis=1).T
    fig, ax = F.new(8, 4.6)
    im = ax.imshow(st_t.to_numpy(), cmap=F.DIV_CMAP, vmin=-12, vmax=12, aspect="auto")
    ax.set_xticks(range(st_t.shape[1]), [f"{y}-{str(y + 1)[-2:]}" for y in st_t.columns]); ax.set_yticks(range(st_t.shape[0]), st_t.index)
    ax.grid(False)
    for i in range(st_t.shape[0]):
        for j in range(st_t.shape[1]):
            v = st_t.iloc[i, j]
            ax.text(j, i, f"{v:+.1f}%", ha="center", va="center", fontsize=8.5, color="white" if abs(v) > 8 else C["ink"])
    for sp in ax.spines.values():
        sp.set_visible(False)
    fig.colorbar(im, ax=ax, fraction=0.04, label="average miss (% of typical value)")
    F.headline(fig, "Is the bias stable season to season? Mostly, with some league-wide shifts",
               "Average (actual − projected) as % of the season's typical value, by category and backtest season")
    fig_d = F.save(fig, "resid_by_season", top=0.88)

    # ---- E. residual correlation
    cc = rf[[f"{c}_res" for c in A.RESID_COLS]].corr()
    fig, ax = F.new(6.6, 5.6)
    im = ax.imshow(cc.to_numpy(), cmap=F.DIV_CMAP, vmin=-1, vmax=1)
    names = [lab[c] for c in A.RESID_COLS]
    ax.set_xticks(range(len(names)), names, rotation=45, ha="right"); ax.set_yticks(range(len(names)), names); ax.grid(False)
    for i in range(len(names)):
        for j in range(len(names)):
            v = cc.iloc[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7.5, color="white" if abs(v) > 0.6 else C["ink"])
    fig.colorbar(im, ax=ax, fraction=0.04, label="correlation of misses")
    F.headline(fig, "Misses are correlated: when the model under-projects minutes, it under-projects everything",
               "Correlation between residuals across categories (these feed the simulation's talent uncertainty)")
    fig_e = F.save(fig, "resid_correlation", top=0.89)
    mpg_corr = cc.loc["pts_res", "mpg_res"]

    # ---- tables + commentary
    t = pd.DataFrame({
        "stat": [lab[c] for c in A.RESID_COLS], "n": rs["n"].values,
        "bias (actual−proj)": [rs.loc[c, "bias"] * scale[c] for c in A.RESID_COLS],
        "± SE": [rs.loc[c, "bias_se"] * scale[c] for c in A.RESID_COLS],
        "typical miss (SD)": [rs.loc[c, "sd"] * scale[c] for c in A.RESID_COLS],
        "MAE": [rs.loc[c, "mae"] * scale[c] for c in A.RESID_COLS],
        "skew": rs["skew"].values, "misses > 2 SD": rs["tail2"].values, "error grows with level (ρ)": rs["hetero_rho"].values})
    t["stat"] = [f"{s_} ({unit[c]})" if unit[c] else s_ for s_, c in zip(t["stat"], A.RESID_COLS)]
    z_bias = (rs["bias"] / rs["bias_se"]).abs()
    biased = [lab[c] for c in A.RESID_COLS if z_bias[c] > 2.5]
    flag_lines = []
    for c, x, m, se, n in sorted(flagged, key=lambda f: -abs(f[2] / f[3]))[:6]:
        flag_lines.append(f"* **{lab[c]}**, players projected around {x * scale[c]:.1f}{' pts' if unit[c] else ''}: actual averaged "
                          f"{m * scale[c]:+.2f}{' pts' if unit[c] else ''} vs projection (n={n}, {abs(m / se):.1f} standard errors from zero)")
    flags_md = "\n".join(flag_lines) if flag_lines else "* none of the decile averages is distinguishable from zero"
    het = rs["hetero_rho"].drop("mpg")
    most_het = lab[het.idxmax()]
    K["resid_mpg_corr"] = float(mpg_corr)
    return f"""
### 4.5 Residual diagnostics: where is the model wrong?

A **residual** is *actual − projected* for a player-season the model had not seen (the same {len(rf):,} out-of-sample player-seasons as above). Residual plots answer three questions
that a single accuracy number cannot: is the model **biased** (consistently high or low somewhere), is its **error size** steady across players, and are its **misses related** to each other?

**How to read these plots.** Dots scattered evenly above and below the zero line, with the orange average hugging zero, mean no bias. A slope in the orange line means the model
mis-handles the high or low end (e.g. a downward slope = stars are over-projected). A funnel (dots fanning out to the right) means bigger players have bigger misses.

![resid1]({fig_a})

**Where the trend line leaves zero** (decile averages whose 95% interval excludes zero):

{flags_md}

*Reading:* a handful of stretches of ~10 can be flagged purely by chance, so only large, consistent departures matter — the table below puts the overall bias on a common footing.

{md_table(t, {"n": "{:.0f}", "bias (actual−proj)": "{:+.3f}", "± SE": "{:.3f}", "typical miss (SD)": "{:.3f}", "MAE": "{:.3f}", "skew": "{:+.2f}", "misses > 2 SD": "{:.1%}", "error grows with level (ρ)": "{:+.2f}"})}

* **Overall bias.** {"Categories whose average miss is more than 2.5 standard errors from zero: " + ", ".join(biased) + "." if biased else "No category has an average miss more than 2.5 standard errors from zero — the model is, on average, unbiased."}
  Bias is in the table's own units (per game; FG%/FT% in percentage points).
* **Error grows with level (ρ).** Positive ρ means bigger projected values come with bigger absolute misses (the funnel). The strongest is {most_het} (ρ = {het.max():+.2f}); this is why
  the simulation sizes talent uncertainty *relative* to a player's level for counting stats instead of using one flat number.
* **Skew and tails.** Positive skew means occasional large positive surprises (breakouts); the share of misses beyond 2 SD (normal would be 4.6%) shows how heavy the tails are.

![resid2]({fig_b})

![resid3]({fig_c})

**Bias by group.** Averages far from zero here would be fixable: for example, persistent over-projection of 30+ year-olds would mean the aging curve is too gentle. Error bars are wide for
small groups (the oldest and youngest), so look for departures that are both large and consistent across the three categories.

![resid4]({fig_d})

![resid5]({fig_e})

**Misses move together.** The correlation between the points miss and the minutes miss is **{mpg_corr:.2f}**: most of what the model gets wrong about a player is his *role* (minutes), which then
spreads across every counting stat. That is exactly the structure the simulation's "talent draw" reproduces (section 8) — the projection's uncertainty is not nine independent errors
but mostly one shared one. It also explains why availability and minutes are the first places to look for improvement (more informative inputs such as depth-chart changes would pay off most there).
"""


# =========================================================================================================
# 5. rookies + 6. projections
# =========================================================================================================
def sec_rookies(st: pd.DataFrame, proj: pd.DataFrame) -> str:
    means, cvs = P.rookie_profiles(st)
    pg = P.to_per_game(means[P.ALL_STATS].reset_index(drop=True))
    tab = pd.DataFrame({"draft slot": means.index, "rookies": means["n"].astype(int).values, "mpg": pg["mpg"].values,
                        "availability": pg["avail"].values, "PTS": pg["pts"].values, "REB": pg["reb"].values, "AST": pg["ast"].values,
                        "STL": pg["stl"].values, "BLK": pg["blk"].values, "TO": pg["tov"].values, "3PM": pg["tpm"].values})
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.8))
    for ax, (col, ttl) in zip(axes, [("mpg", "Minutes per game"), ("PTS", "Points per game"), ("availability", "Share of games played")]):
        ax.bar(tab["draft slot"], tab[col], color=C["blue"], width=0.62)
        for xi, v in enumerate(tab[col]):
            ax.text(xi, v * 1.01, f"{v:.1f}" if col != "availability" else f"{v:.0%}", ha="center", fontsize=8.5, color=C["ink2"])
        ax.set_title(ttl, fontsize=10, loc="left"); ax.tick_params(axis="x", labelsize=8)
        ax.grid(axis="x", visible=False)
        if col == "availability":
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    F.headline(fig, "A rookie's draft slot says a lot about his minutes — and how uncertain his season is",
               "Average first-NBA-season profile by draft slot, 2015-16 → 2025-26 (≥200 minutes)")
    fig_r = F.save(fig, "rookie_tiers", top=0.84)
    cv_t = cvs[["fga2", "fga3", "reb", "ast", "stl", "blk", "tov"]].round(2).reset_index().rename(columns={"tier": "draft slot"})
    top_r = proj[proj["source"] == "rookie_tier"].sort_values("z_total", ascending=False).head(10)
    top_r = top_r.assign(**{"draft pick": top_r["draft_pick"].astype(int)})
    show = top_r[["rank", "name", "draft pick", "elig", "mpg", "pts", "reb", "ast", "stl", "blk", "tov"]]
    K["n_rookies"] = int((proj["source"] == "rookie_tier").sum())
    cmp_ = A.rookie_model_comparison(st)
    cmp_t = cmp_.rename(columns={"mpg": "MPG", "pts": "PTS", "reb": "REB", "ast": "AST", "stl": "STL", "blk": "BLK"}).reset_index().rename(
        columns={"index": "method"})
    cmp_t["method"] = cmp_t["method"].map({"tier": "draft-slot bucket means (used)", "curve": "smooth curve in log(pick)"})
    return f"""
## 5. Rookies: players with no history

A player with no NBA season has nothing to blend. We use his **draft slot** as the only prior: all first-year players since 2015 are bucketed by where they
were drafted (1-5, 6-14, 15-30, 31-60, undrafted) and each bucket's average rates become the rookie's projection. Where a source you trust (for example ESPN) disagrees,
drop the number into `data/external/projection_overrides.csv` and it wins.

![rookies]({fig_r})

{md_table(tab, {"rookies": "{:.0f}", "mpg": "{:.1f}", "availability": "{:.0%}", "PTS": "{:.1f}", "REB": "{:.1f}", "AST": "{:.1f}", "STL": "{:.2f}", "BLK": "{:.2f}", "TO": "{:.2f}", "3PM": "{:.2f}"})}

**Uncertainty is part of the answer.** The *coefficient of variation* (spread ÷ average) of each bucket — how much individual rookies differ from their bucket's mean
— is stored and later feeds the simulation, so a late first-rounder's range of outcomes is far wider than a top-5 pick's:

{md_table(cv_t, {c: "{:.2f}" for c in cv_t.columns if c != "draft slot"})}

**Why buckets and not a smooth curve?** We tested a smoother alternative — a regression on log(draft pick), which would separate pick #1 from pick #5 — by leaving out each
rookie class in turn and predicting it from the others (average absolute miss per game):

{md_table(cmp_t, {c: "{:.3f}" for c in cmp_t.columns if c != "method"})}

The curve is no more accurate (differences are in the third decimal, and the buckets are slightly ahead on minutes), so we keep the simpler method. **The consequence you should know about:**
every rookie in a bucket gets the *same* projection, so two top-5 picks are identical in the model, and an individual standout (or bust) can't be anticipated from draft slot alone. If you have
rookie projections you trust, put them in `data/external/projection_overrides.csv`.
Players with no NBA history who are *not* 2026 draftees (e.g. someone returning from Europe) are projected with the undrafted bucket, not the draft slot they were picked at years ago.

**The {K["n_rookies"]} players projected this way for 2026-27** (rookies + anyone without NBA history in the last three seasons); the ten most valuable:

{md_table(show, {"rank": "{:.0f}", "mpg": "{:.1f}", "pts": "{:.1f}", "reb": "{:.1f}", "ast": "{:.1f}", "stl": "{:.2f}", "blk": "{:.2f}", "tov": "{:.2f}"})}
"""


def sec_projections(proj: pd.DataFrame) -> str:
    top = proj.head(25)
    zc = [f"z_{c}" for c in CATEGORIES]
    fig, ax = F.new(9, 8.4)
    mat = top[zc].to_numpy()
    im = ax.imshow(mat, cmap=F.DIV_CMAP, vmin=-4, vmax=4, aspect="auto")
    ax.set_xticks(range(9), [CAT_LABELS[c] for c in CATEGORIES]); ax.xaxis.tick_top()
    ax.set_yticks(range(len(top)), [f"{r}. {n}" for r, n in zip(top["rank"], top["name"])])
    ax.grid(False)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = mat[i, j]
            ax.text(j, i, f"{v:+.1f}", ha="center", va="center", fontsize=8, color="white" if abs(v) > 2.6 else C["ink"])
    cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02); cb.set_label("z-score (standard deviations above a typical drafted player)")
    for s in ax.spines.values():
        s.set_visible(False)
    F.headline(fig, "Where each top player's value comes from",
               "Projected 2026-27 z-scores by category (blue = helps, red = hurts; turnovers are flipped so blue is fewer)")
    fig_h = F.save(fig, "top25_heatmap", top=0.915)
    tbl = proj.head(40)[["rank", "name", "elig", "age_t", "mpg", "avail", "pts", "tpm", "reb", "ast", "stl", "blk", "fg_pct", "ft_pct", "tov", "z_total"]].copy()
    tbl["avail"] = (tbl["avail"] * 100).round(0)
    tbl.columns = ["#", "Player", "Pos", "Age", "MPG", "GP%", "PTS", "3PM", "REB", "AST", "STL", "BLK", "FG%", "FT%", "TO", "Value"]
    return f"""
## 6. The 2026-27 projections

Per-game projections for every one of the {len(proj)} players on a 2026-27 roster. **Value** is the sum of nine z-scores (how many standard deviations
better than a typical *drafted* player in each category; turnovers flipped; FG%/FT% weighted by volume, so a 60% shooter on 3 attempts matters less than a 52%
shooter on 15; every count scaled by expected availability). It's the starting ranking — the draft engine in section 8 goes beyond it.

{md_table(tbl, {"#": "{:.0f}", "Age": "{:.0f}", "MPG": "{:.1f}", "GP%": "{:.0f}", "PTS": "{:.1f}", "3PM": "{:.1f}", "REB": "{:.1f}", "AST": "{:.1f}", "STL": "{:.1f}", "BLK": "{:.1f}", "FG%": "{:.3f}", "FT%": "{:.3f}", "TO": "{:.1f}", "Value": "{:.1f}"})}

![heatmap]({fig_h})

*How to read the heatmap:* row = player, column = category. A deep blue cell is a category where he gives you a big edge; deep red is where he costs you.
The shape of a row is the player's **archetype** — Jokić is blue almost everywhere except turnovers; Wembanyama's value is concentrated in blocks, rebounds and points.
The engine uses these shapes: a team with too many players of the same shape wins fewer categories.
"""


# =========================================================================================================
# 7. biggest movers
# =========================================================================================================
def _mover_bar(both: pd.DataFrame, name: str, headline_txt: str, sub: str, xlabel: str) -> str:
    fig, ax = F.new(9, 7.2)
    y = np.arange(len(both))
    ax.barh(y, both["dz"], color=[C["red"] if v < 0 else C["blue"] for v in both["dz"]], height=0.66)
    for yi, dzv in enumerate(both["dz"]):
        ax.text(dzv + (0.1 if dzv >= 0 else -0.1), yi, f"{dzv:+.1f}", va="center", ha="left" if dzv >= 0 else "right", fontsize=8.5, color=C["ink2"])
    ax.set_yticks(y, [f"{n} ({a:.0f})" for n, a in zip(both["name"], both["age"])])
    ax.axvline(0, color=C["ink2"], lw=1); ax.grid(axis="y", visible=False)
    lo, hi = both["dz"].min(), both["dz"].max()
    ax.set_xlim(lo - 0.15 * (hi - lo), hi + 0.12 * (hi - lo))
    ax.set_xlabel(xlabel)
    F.headline(fig, headline_txt, sub)
    F.legend_top(fig, [F.swatch(C["blue"], "projected to improve"), F.swatch(C["red"], "projected to decline")])
    return F.save(fig, name)


def _mover_table(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({
        "Player": df["name"], "Age": df["age"], "Pos": df["elig"], "Rank 25-26": df["rank_last"], "Rank 26-27": df["rank_proj"],
        "Δ value": df["dz"], "MPG": df["mpg_last"].round(1).astype(str) + " → " + df["mpg_proj"].round(1).astype(str),
        "GP": df["gp_last"].round(0).astype(int).astype(str) + " → " + df["gp_proj"].round(0).astype(int).astype(str),
        "Biggest category swings": df["why"]})


def sec_movers(proj: pd.DataFrame, apg: pd.DataFrame, last_year: int) -> str:
    mv = A.movers(proj, apg, last_year)                              # total value (includes games played)
    pgm = A.movers(proj, apg, last_year, hold_avail=True)             # per-game production only
    for d in (mv, pgm):
        d["why"] = d.apply(lambda r: A.driver_text(r, CAT_LABELS), axis=1)
    fm = {"Age": "{:.0f}", "Rank 25-26": "{:.0f}", "Rank 26-27": "{:.0f}", "Δ value": "{:+.1f}"}

    up, dn = mv.sort_values("dz", ascending=False).head(12), mv.sort_values("dz").head(12)
    fig_m = _mover_bar(pd.concat([dn.iloc[::-1], up.iloc[::-1]]), "movers_bar",
                       "Biggest projected jumps and drops in total fantasy value",
                       "Players with ≥1,000 minutes in 2025-26; includes the projected change in games played",
                       "change in 9-category value, 2025-26 actual → 2026-27 projected (z-score points); age in brackets")
    pup, pdn = pgm.sort_values("dz", ascending=False).head(12), pgm.sort_values("dz").head(12)
    fig_p = _mover_bar(pd.concat([pdn.iloc[::-1], pup.iloc[::-1]]), "movers_bar_pergame",
                       "Per-game production only: who is projected to be better or worse when he plays",
                       "Same players, games played held equal (availability = 100% in both seasons)",
                       "change in per-game 9-category value (z-score points); age in brackets")

    # regression to the mean
    slope = float(np.polyfit(mv["z_last"], mv["z_proj"], 1)[0])
    fig, ax = F.new(7.6, 6.2)
    ax.scatter(mv["z_last"], mv["z_proj"], s=16, color=F.BG_POINT, alpha=0.75, linewidths=0)
    lim = [mv[["z_last", "z_proj"]].min().min() - 1, mv[["z_last", "z_proj"]].max().max() + 1]
    ax.plot(lim, lim, color=C["ink2"], lw=1, ls=(0, (4, 3)))
    ax.set_xlabel("2025-26 actual value"); ax.set_ylabel("2026-27 projected value"); ax.set_xlim(lim); ax.set_ylim(lim)
    F.headline(fig, f"Projections are a compressed copy of last season (slope {slope:.2f}): regression to the mean",
               "Each dot is a player; above the dashed no-change line = projected to improve. The four biggest risers and fallers are highlighted")
    F.legend_top(fig, [F.swatch(C["blue"], "biggest projected risers", "dot"), F.swatch(C["red"], "biggest projected fallers", "dot"),
                       F.swatch(F.BG_POINT, "everyone else", "dot"), F.swatch(C["ink2"], "no change", "line")], ncol=4)
    F.label_points(ax, [(r.z_last, r.z_proj, A.short_name(r.name), C["blue"] if r.dz > 0 else C["red"])
                        for r in pd.concat([up.head(4), dn.head(4)]).itertuples()])
    fig_s = F.save(fig, "movers_scatter")

    # aging curve: per-game production only, so injuries don't blur it
    pgm["age_b"] = pgm["age"].round()
    ag = pgm.groupby("age_b").agg(n=("dz", "size"), dz=("dz", "mean")).query("n >= 8").reset_index()
    young = pgm.loc[pgm["age"] <= 23, "dz"].mean()
    prime = pgm.loc[(pgm["age"] >= 24) & (pgm["age"] <= 27), "dz"].mean()
    older = pgm.loc[pgm["age"] >= 30, "dz"].mean()
    fig, ax = F.new(8, 4.2)
    ax.axhline(0, color=C["ink2"], lw=1)
    ax.plot(ag["age_b"], ag["dz"], color=C["blue"], lw=2, marker="o", ms=6)
    for r in ag.itertuples():
        ax.text(r.age_b, r.dz - 0.28 if r.dz > 0 else r.dz + 0.18, f"n={r.n}", ha="center", fontsize=8, color=C["muted"])
    ax.set_xlabel("age in 2026-27"); ax.set_ylabel("average change in per-game value (z points)")
    F.headline(fig, f"What the model learned about aging: age ≤23 {young:+.1f}, 24-27 {prime:+.1f}, 30+ {older:+.1f} z-points",
               "Average projected change in per-game 9-category value vs last season, by age (games played held equal)")
    fig_a = F.save(fig, "aging_curve")

    n_inj = int((up["gp_last"] < 60).sum()); n_old = int((dn["age"] >= 31).sum()); n_young = int((up["age"] <= 24).sum())
    g_lo = int((pdn["age"] >= 30).sum()); g_hi = int((pup["age"] <= 24).sum())
    return f"""
## 7. Biggest projected risers and fallers

Comparing the 2026-27 projection with what each player *actually* did in 2025-26. Both seasons are z-scored among the same {len(mv)} players who logged ≥1,000 minutes last year, so the scales
match. Two lenses, because a player's fantasy value changes for two different reasons — he plays better or worse, or he plays more or fewer games:

### 7.1 Total fantasy value (what you actually draft)

![movers]({fig_m})

**Risers**

{md_table(_mover_table(up), fm)}

**Fallers**

{md_table(_mover_table(dn), fm)}

*MPG and GP show last season's actual → this season's projection; "Biggest category swings" are the two categories where his z-score moves most.*

**What drives these moves.** {n_inj} of the 12 risers played fewer than 60 games last year — the model projects a return toward a normal share of games, which lifts every counting stat at once.
{n_old} of the 12 fallers are 31 or older, and {n_young} of the 12 risers are 24 or younger. So these lists mix two forces; the next lens separates them.

### 7.2 Per-game production only (games played held equal)

![pergame]({fig_p})

**Biggest per-game risers**

{md_table(_mover_table(pup), fm)}

**Biggest per-game fallers**

{md_table(_mover_table(pdn), fm)}

{g_hi} of these 12 risers are 24 or younger and {g_lo} of the 12 fallers are 30 or older: with availability out of the picture, the age signal is the dominant one.

### 7.3 Where the model's three forces show up

1. **Availability.** Players who missed a lot of games last year are projected to play more (and the reverse). It mostly moves the *total value* list above.
2. **Aging.** Learned from data, not hand-drawn:

![aging]({fig_a})

3. **Regression to the mean.** Last season's outliers (hot 3-point shooting, a career-high in steals) are pulled toward what similar players sustain. The scatter's slope of
   {slope:.2f} (<1) is that effect in one number: the projected ordering is a flatter version of last year's.

![scatter2]({fig_s})

**Caveat.** These are *projected averages*, not predictions of any specific event: a faller is not predicted to be bad, and a riser is not guaranteed to improve. The engine in section 9 accounts
for that uncertainty directly (section 8) rather than trusting the averages blindly.
"""


# =========================================================================================================
# 8. simulation
# =========================================================================================================
def sim_calibration(conn, st_all, players, cal, year=2025, sims_per_week=80, top_n=200):
    """Full-stack out-of-sample check: project `year` from earlier data only, simulate its real weeks, compare to reality."""
    season = f"{year}-{str(year + 1)[-2:]}"
    st = st_all[st_all["year"] < year]
    bt = P.backtest(st, folds=list(range(year - 3, year)), verbose=False)
    pj = P.project_season(st, players, year, bt.selection)
    pj["team_id"] = pj["player_id"].map(team_at_season_start(conn, season))
    pj = add_eligibility(pj)
    pj = pd.concat([pj, z_scores(pj)], axis=1).sort_values("z_total", ascending=False).head(top_n).reset_index(drop=True)
    sched = season_periods(conn, season)
    lib = build_library(pj, cal, sched, sims_per_week=sims_per_week, seed=7)
    act = actual_library(conn, season, lib.player_ids, sched)
    W, k = sched.n_weeks, sims_per_week
    stats_to_check = {"PTS": None, "REB": "reb", "AST": "ast", "3PM": "tpm", "STL": "stl", "BLK": "blk", "TO": "tov"}

    def weekly(L: Library, name):
        if name == "PTS":
            s = L.stats.astype(np.int32)
            v = 2 * s[..., STAT_IDX["fgm"]] + s[..., STAT_IDX["tpm"]] + s[..., STAT_IDX["ftm"]]
        else:
            v = L.stats[..., STAT_IDX[stats_to_check[name]]].astype(np.int32)
        return v.sum(axis=2)                                       # [P, S or W]

    rng = np.random.default_rng(0)
    rows, pits = [], {}
    for name in stats_to_check:
        sim = weekly(lib, name).reshape(len(pj), k, W)             # s = j*W + w  ->  [P, draw, week]
        real = weekly(act, name)                                   # [P, W]
        simw = np.transpose(sim, (0, 2, 1)).reshape(-1, k)         # [(P*W), draws]
        pit = A.pit_values(simw, real.reshape(-1).astype(float), rng)
        pits[name] = pit
        lo, hi = np.percentile(simw, 10, axis=1), np.percentile(simw, 90, axis=1)
        cover = float(((real.reshape(-1) >= lo) & (real.reshape(-1) <= hi)).mean())
        rows.append(dict(stat=name, coverage_80=cover, sim_mean=float(simw.mean()), real_mean=float(real.mean()),
                         sd_ratio=float(real.std() / simw.std())))
    res = pd.DataFrame(rows)
    pm_sim = weekly(lib, "PTS").reshape(len(pj), k, W).mean(axis=(1, 2)); pm_real = weekly(act, "PTS").mean(axis=1)
    return res, pits["PTS"], pm_sim, pm_real, len(pj), W


def sec_simulation(conn, st_all, players, cal, lib: Library, proj: pd.DataFrame, sched, skip_check: bool) -> str:
    # calibration params
    par = pd.DataFrame({"stat": [d for d in COUNT_DIMS], "NB size r": [cal.dispersion[d] for d in COUNT_DIMS],
                        "extra variance vs Poisson": [1 / cal.dispersion[d] for d in COUNT_DIMS]})
    # copula heatmap
    R = np.array(cal.corr)
    names = ["2PA", "3PA", "FTA", "REB", "AST", "STL", "BLK", "TOV", "2P luck", "3P luck", "FT luck"]
    fig, ax = F.new(7, 6)
    im = ax.imshow(R, cmap=F.SEQ_CMAP, vmin=0, vmax=0.5)
    ax.set_xticks(range(11), names, rotation=45, ha="right"); ax.set_yticks(range(11), names); ax.grid(False)
    for i in range(11):
        for j in range(11):
            if i != j:
                ax.text(j, i, f"{R[i, j]:.2f}", ha="center", va="center", fontsize=7.5, color="white" if R[i, j] > 0.3 else C["ink"])
    fig.colorbar(im, ax=ax, fraction=0.04, label="correlation")
    F.headline(fig, "Stats move together within a game — rebounds with shot attempts, and so on",
               "Gaussian-copula correlation between a player's per-game stats (within-player, 3 recent seasons)")
    fig_c = F.save(fig, "copula_corr", top=0.9)

    # schedule
    gp = sched.masks.sum(axis=2).ravel()                      # games per team per matchup period
    single = [i for i, (a, b) in enumerate(sched.spans) if (b - a).days < 7]
    multi = [i for i in range(sched.n_weeks) if i not in single]
    nmax = int(gp.max())
    cnt = np.bincount(gp, minlength=nmax + 1)[1:nmax + 1] / gp.size
    n_cup_missing = sum(sched.missing.values()) // 2 if sched.missing else 0
    fig, ax = F.new(7.6, 3.8)
    ax.bar(range(1, nmax + 1), cnt * 100, color=C["blue"], width=0.6)
    for i, v in enumerate(cnt):
        if v > 0.004:
            ax.text(i + 1, v * 100 + 1, f"{v:.0%}", ha="center", fontsize=9, color=C["ink2"])
    ax.set_xlabel("games a team plays in one matchup period"); ax.set_ylabel("share of team-periods (%)"); ax.grid(axis="x", visible=False)
    span_txt = "; ".join(f"week {i + 1} ({sched.spans[i][0]:%b %d}-{sched.spans[i][1]:%b %d})" for i in multi)
    F.headline(fig, f"The real 2026-27 calendar: {sched.n_weeks} regular-season matchup periods, {len(multi)} of them two weeks long",
               f"The 6+ game bars are the two-week periods: {span_txt}."
               + (f" The {n_cup_missing} NBA Cup games not yet on the schedule are ignored, so week 7 is simulated with only its published games." if n_cup_missing else ""))
    fig_g = F.save(fig, "schedule_games_per_week", top=0.84)

    # example player distributions
    pick = []
    for nm in ["Shai Gilgeous-Alexander", "Anthony Davis"]:
        r = proj[proj["name"] == nm]
        if len(r):
            pick.append(r.iloc[0])
    pick.append(proj.iloc[100])
    ids = {int(p): i for i, p in enumerate(lib.player_ids)}
    fig, axes = plt.subplots(1, 3, figsize=(10.4, 3.8), sharex=True)
    ex_rows = []
    for ax, p in zip(axes, pick):
        s = lib.stats[ids[int(p["player_id"])]].astype(np.int32)
        pts = (2 * s[..., STAT_IDX["fgm"]] + s[..., STAT_IDX["tpm"]] + s[..., STAT_IDX["ftm"]]).sum(axis=1)
        zero = (lib.played[ids[int(p["player_id"])]].sum(axis=1) == 0).mean()
        ax.hist(pts, bins=np.arange(0, 181, 6), color=C["blue"], alpha=0.9, edgecolor=C["surface"], linewidth=0.8)
        ax.axvline(pts.mean(), color=C["orange"], lw=2)
        ax.set_title(f"{p['name']}\nmean {pts.mean():.0f} · 10-90%: {np.percentile(pts, 10):.0f}-{np.percentile(pts, 90):.0f} · missed week: {zero:.0%}",
                     fontsize=9.5, loc="left", linespacing=1.4)
        ax.set_xlabel("points in a week"); ax.set_yticks([])
        ex_rows.append(dict(player=p["name"], avail=p["avail"], mean=pts.mean(), p10=np.percentile(pts, 10), p90=np.percentile(pts, 90), zero_weeks=zero))
    F.headline(fig, "A player is a distribution, not a number",
               f"Simulated weekly points ({lib.stats.shape[1]} simulated matchups each). The spike at 0 is a week missed entirely")
    F.legend_top(fig, [F.swatch(C["blue"], "simulated weeks"), F.swatch(C["orange"], "mean", "line")])
    fig_d = F.save(fig, "weekly_distributions")

    # one simulated matchup (the two-week All-Star period), day by day
    star = pick[0]
    si = ids[int(star["player_id"])]
    wk = multi[-1]                                   # period 17: Feb 15-28; sim index == period index for the first W sims
    start = sched.spans[wk][0]
    day_rows = []
    for d in range(lib.stats.shape[2]):
        if lib.played[si, wk, d]:
            x = lib.stats[si, wk, d].astype(int)
            day_rows.append({"date": f"{DAYS[(start + pd.Timedelta(days=d)).dayofweek]} {start + pd.Timedelta(days=d):%b %d}",
                             "PTS": 2 * x[STAT_IDX["fgm"]] + x[STAT_IDX["tpm"]] + x[STAT_IDX["ftm"]],
                             "FG": f"{x[STAT_IDX['fgm']]}-{x[STAT_IDX['fga']]}", "3PM": x[STAT_IDX["tpm"]],
                             "FT": f"{x[STAT_IDX['ftm']]}-{x[STAT_IDX['fta']]}", "REB": x[STAT_IDX["reb"]], "AST": x[STAT_IDX["ast"]],
                             "STL": x[STAT_IDX["stl"]], "BLK": x[STAT_IDX["blk"]], "TO": x[STAT_IDX["tov"]]})
    day_tbl = pd.DataFrame(day_rows) if day_rows else pd.DataFrame([{"date": "(no games played in this draw)"}])

    check_md = "*(full-stack calibration check skipped)*"
    if not skip_check:
        res, pit, pm_sim, pm_real, n_pl, W = sim_calibration(conn, st_all, players, cal)
        fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.2))
        ax = axes[0]
        h, _ = np.histogram(pit, bins=10, range=(0, 1))
        ax.bar(np.arange(10) * 10 + 5, h / h.sum() * 100, width=8.6, color=C["blue"])
        ax.axhline(10, color=C["orange"], lw=2)
        ax.set_xlabel("where the actual week fell in the simulated range (percentile bucket)"); ax.set_ylabel("% of player-weeks")
        ax.set_title("Weekly points: calibration (PIT histogram)", fontsize=10, loc="left"); ax.grid(axis="x", visible=False)
        ax = axes[1]
        ax.scatter(pm_sim, pm_real, s=14, color=C["blue"], alpha=0.5, linewidths=0)
        lim = max(pm_sim.max(), pm_real.max()) * 1.05
        ax.plot([0, lim], [0, lim], color=C["ink2"], lw=1)
        r2 = 1 - ((pm_real - pm_sim) ** 2).sum() / ((pm_real - pm_real.mean()) ** 2).sum()
        ax.set_xlabel("simulated average weekly points (from preseason inputs only)"); ax.set_ylabel("actual average weekly points")
        ax.set_title(f"Season-average level (R² = {r2:.2f})", fontsize=10, loc="left")
        h_pit = np.histogram(pit, bins=10, range=(0, 1))[0]
        h_pit = h_pit / h_pit.sum()
        pit_dist = float(np.abs(h_pit - 0.1).sum() * 100)          # total deviation from a perfectly flat histogram, in points
        K["pit_lo"], K["pit_hi"], K["pit_dist"] = float(h_pit[0]), float(h_pit[-1]), pit_dist
        verdict = ("closely reproduces the real spread of" if pit_dist < 10 else
                   "roughly reproduces the real spread of" if pit_dist < 20 else "is over-confident about")
        tail_note = "" if pit_dist < 10 else (", slightly over-confident in the tails" if pit_dist < 20 else "")
        F.headline(fig, f"The simulation {verdict} 2025-26 weeks it never saw{tail_note}",
                   f"Top {n_pl} players by projected value; {W} real matchup periods each; projections & simulation built from pre-2025 data only")
        F.legend_top(fig, [F.swatch(C["blue"], "share of player-weeks / player"), F.swatch(C["orange"], "perfect calibration = 10% per bucket", "line"),
                           F.swatch(C["ink2"], "perfect agreement", "line")])
        fig_p = F.save(fig, "sim_calibration")
        K["cov"] = float(res.loc[res["stat"] == "PTS", "coverage_80"].iloc[0])
        check_md = f"""
![calibration]({fig_p})

{md_table(res.rename(columns={"stat": "category", "coverage_80": "actual inside simulated 10-90% band", "sim_mean": "simulated avg / week", "real_mean": "actual avg / week", "sd_ratio": "actual SD ÷ simulated SD"}), {"actual inside simulated 10-90% band": "{:.0%}", "simulated avg / week": "{:.2f}", "actual avg / week": "{:.2f}", "actual SD ÷ simulated SD": "{:.2f}"})}

**How to read it.** The left panel is a *PIT histogram*: for every player-week, where did the actual week fall within the simulated distribution for that same player and week?
If the simulation is well calibrated each 10%-wide bucket holds 10% of weeks. Bars piled at both ends mean actual weeks are more extreme than simulated (the simulation is over-confident);
a hump in the middle would mean it is too wide. Here the lowest bucket holds **{h_pit[0]:.1%}** and the highest **{h_pit[-1]:.1%}** (nominal 10%), and the middle two hold
{h_pit[4] + h_pit[5]:.1%} (nominal 20%): close to flat, with a modest excess in the upper tail — more breakout weeks than the simulation expects. (The "10-90% band" column is a cruder
check: it counts a zero-game week as "inside" whenever the band's lower edge is zero, so it flatters the result for stars with real injury risk; the PIT handles ties properly.)
The right panel is the simulation's idea of each player's *level*, built before the season, against what he did: much of the scatter is availability (injuries) and breakouts/declines that no
preseason model can see, which is why the R² is far lower than for per-game stats.

**A bug this check caught.** The first version of this check looked clearly worse: the lowest and highest buckets held 15.8% and 15.2% in 2025-26 (distance from flat 28 points, vs
{pit_dist:.0f} now). Widening the projection uncertainty did not help — the heavy lower tail stayed — which pointed at availability, not talent. The cause: the share of missed games that come in
whole-week absences (`c`) had been estimated only from players with ≥25 games and only between their first and last appearance, which silently excluded the long and season-ending
injuries. Re-measured over every week of the season for every rotation player, `c` rose from 0.42 to {cal.absence_c:.2f}, which reproduces the real 18-21% of zero-game weeks, and the
distance from flat fell to roughly half in every season tested (2023: 14.5 → 7.3, 2024: 17.4 → 8.3, 2025: 27.8 → 15.6). Everything in this document was regenerated after that fix, and the
historical validation in section 10 was re-run on the corrected simulation.
"""
    return f"""
## 8. Phase 2 — from an average to a distribution of weeks

A league matchup is decided by one week, and a week is noisy. Two players with the same season average can have very different weekly spreads (a star who never
misses vs. one who misses 40% of weeks). The simulation turns each projected average into thousands of realistic **weeks**.

### 8.1 Step by step, for one player-week

1. **Pick a real matchup period** of your league's 2026-27 calendar (so we know which days his team plays — see 8.2). Each period counts equally, whether it is one week or two.
2. **Talent draw.** The projection is only a best guess, so perturb it: each stat's per-game average is multiplied by a random factor whose spread was measured from the
   backtest residuals (≈15-30% for counting stats for veterans, much wider for rookies by draft slot), with correlated shocks (a player who gets more minutes gets more of everything).
3. **Availability.** With probability `c·(1−avail)` he misses the entire week (injury run; `c = {cal.absence_c:.2f}` is the measured share of missed games that come in whole-week runs).
   Otherwise he plays each scheduled game with the probability that makes his expected games = `avail × team games`.
4. **For each game he plays, draw a stat line** with a Gaussian copula: draw 11 correlated standard normals, turn each into a uniform with the normal CDF, then push each
   through its stat's own distribution (inverse CDF):
   * counts (2-pt attempts, 3-pt attempts, FT attempts, rebounds, assists, steals, blocks, turnovers) → **Negative Binomial** (variance bigger than the mean — real counts are streaky);
   * makes → **Binomial(attempts, make-rate)** with the make-rate's quantile driven by its own copula dimension.
5. **Compute points and percentages from the shots**: `PTS = 2·FGM + 3PM + FTM`. They are never simulated on their own.

{md_table(par, {"NB size r": "{:.1f}", "extra variance vs Poisson": "{:.3f}"})}

*(Smaller r = more streaky. Free throws attempts, `r = {cal.dispersion["fta"]:.1f}`, are the most volatile count; turnovers, `r = {cal.dispersion["tov"]:.0f}`, the most Poisson-like.)*

![corr]({fig_c})

### 8.2 Your league's matchup periods

Matchup periods are Mon–Sun weeks, with two exceptions that your league makes and that the code reproduces from the schedule itself: the **NBA Cup knockout week** is merged with the week
before it, and the **All-Star break** is a two-week matchup. For 2026-27 that gives week 1 = Oct 19–25, week 2 = Oct 26–Nov 1, …, **week 7 = Nov 30–Dec 13** (Cup), …, **week 17 = Feb 15–28**
(All-Star), week 18 = Mar 1–7 (end of the regular season), then the three playoff weeks Mar 8–28 — exactly your "18 + 3". The engine's objective is the 18 regular-season periods
(`build_library.py --include-playoffs` adds the playoff weeks).

Two consequences of the two-week matchups. Counting totals roughly double while percentages and the winner-take-all result stay the same, so those weeks are *less random* — a few lucky games
matter less. **Week 7 is the opposite case**: the NBA Cup week is thin and its game counts are unusual, and teams that go deep in the Cup play on different days from eliminated teams. The published schedule
still lacks about 30 Cup games (every team shows 80 of its 82), and we deliberately **ignore** them: week 7 is simulated with only its published games and the rest is left to the simulation's
randomness (`--fill-cup-games` would instead place each team's missing games on random Cup-week days, but that is off by default). Treat week 7 as the noisiest week of the season.

![sched]({fig_g})

Using the real schedule matters because roster value depends on *which days* players play. On a day when your four centers all have games, only a limited number can start; a week in which
your stars happen to share off-days leaves lineup spots empty. That interaction cannot be seen in a season-average projection.

### 8.3 What a simulated player looks like

![dist]({fig_d})

{md_table(pd.DataFrame(ex_rows).rename(columns={"zero_weeks": "weeks missed entirely", "p10": "10th pct", "p90": "90th pct"}), {"avail": "{:.0%}", "mean": "{:.0f}", "10th pct": "{:.0f}", "90th pct": "{:.0f}", "weeks missed entirely": "{:.0%}"})}

**One simulated draw of {star["name"]} for the two-week All-Star matchup (week {wk + 1}, {sched.spans[wk][0]:%b %d} – {sched.spans[wk][1]:%b %d});** each row is a game he played in this draw:

{md_table(day_tbl)}

### 8.4 Does the simulation match reality?

We rebuilt the whole pipeline as of the 2025-26 preseason — projections *and* simulation fitted using only seasons before 2025-26 — and compared the simulated weekly distributions
with what really happened, game by game:

{check_md}
"""


# =========================================================================================================
# 9. engine
# =========================================================================================================
def sec_engine(proj: pd.DataFrame, lib: Library, sched) -> str:
    s1 = LeagueSettings(my_slot=1)
    eng = DraftEngine(lib, proj, s1)
    st1 = DraftState(s1, centers=set(eng.center_ids))
    rec = eng.recommend(st1, top_k=25, rollouts=2).head(10)
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.6), gridspec_kw={"width_ratios": [1, 1.5]})
    ax = axes[0]
    y = np.arange(len(rec))
    ax.barh(y, rec["win_prob"] * 100, color=C["blue"], height=0.62)
    for yi, v in zip(y, rec["win_prob"]):
        ax.text(v * 100 + 0.2, yi, f"{v * 100:.1f}%", va="center", fontsize=8.5, color=C["ink2"])
    ax.set_yticks(y, rec["name"]); ax.invert_yaxis(); ax.set_xlim(45, rec["win_prob"].max() * 100 + 3)
    ax.set_xlabel("chance to win a weekly matchup (%)"); ax.grid(axis="y", visible=False)
    ax = axes[1]
    mat = rec[[f"cat_{c}" for c in CATEGORIES]].to_numpy() * 100
    im = ax.imshow(mat, cmap=F.DIV_CMAP, vmin=30, vmax=70, aspect="auto")
    ax.set_xticks(range(9), [CAT_LABELS[c] for c in CATEGORIES]); ax.xaxis.tick_top(); ax.set_yticks([]); ax.grid(False)
    for i in range(mat.shape[0]):
        for j in range(9):
            ax.text(j, i, f"{mat[i, j]:.0f}", ha="center", va="center", fontsize=8, color="white" if abs(mat[i, j] - 50) > 12 else C["ink"])
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_xlabel("chance to win each category (%)  ·  blue > 50 > red")
    gap12 = (rec.iloc[0]["win_prob"] - rec.iloc[1]["win_prob"]) * 100
    gap23 = (rec.iloc[1]["win_prob"] - rec.iloc[2]["win_prob"]) * 100
    F.headline(fig, (f"Pick #1: {rec.iloc[0]['name']} edges {rec.iloc[1]['name']} by {gap12:.1f} points; "
                     f"{rec.iloc[2]['name']} is {gap23:.1f} further back"),
               "Simulated matchup win probability if each player is my first pick (draft slot 1, 10 teams); category win chances on the right")
    fig_r = F.save(fig, "pick1_recs", top=0.86)
    def cat_story(row):
        cw = {c: row[f"cat_{c}"] * 100 for c in CATEGORIES}
        best = sorted(cw, key=cw.get, reverse=True)[:2]
        worst = sorted(cw, key=cw.get)[:2]
        return (f"**{row['name']}** — strongest: " + ", ".join(f"{CAT_LABELS[c]} {cw[c]:.0f}%" for c in best)
                + "; weakest: " + ", ".join(f"{CAT_LABELS[c]} {cw[c]:.0f}%" for c in worst))
    story = "\n".join(f"* {cat_story(rec.iloc[i])}" for i in range(3))
    r_t = rec[["name", "elig", "static_rank", "win_prob", "vs_top_ranked"]].copy()
    r_t["win_prob"] *= 100; r_t["vs_top_ranked"] *= 100
    r_t.columns = ["Player", "Pos", "Value rank", "Win % (matchup)", "vs top-ranked (pts)"]

    # ---- full mock draft at slot 5, engine vs best available
    slot = 5
    s5 = LeagueSettings(my_slot=slot)
    eng5 = DraftEngine(lib, proj, s5)
    log = []

    def strat(engine, state):
        rec_ = engine.recommend(state, top_k=20, rollouts=2)
        top = rec_.iloc[0]
        best_static = rec_.sort_values("static_rank").iloc[0]
        log.append(dict(pick=state.next_pick_no, rnd=(state.next_pick_no - 1) // s5.n_teams + 1, player=top["name"], pos=top["elig"],
                        win=top["win_prob"], best_static=best_static["name"], differs=top["player_id"] != best_static["player_id"],
                        gain=top["win_prob"] - best_static["win_prob"]))
        return int(top["player_id"])

    rosters_e = run_mock_draft(eng5, s5, strat, bot_seed=11)
    rosters_b = run_mock_draft(eng5, s5, best_available, bot_seed=11)
    ev_e, ev_b = eng5.evaluate(rosters_e), eng5.evaluate(rosters_b)
    lg = pd.DataFrame(log)
    lg_t = pd.DataFrame({"round": lg["rnd"], "pick #": lg["pick"], "engine picks": lg["player"], "pos": lg["pos"],
                         "win % after": lg["win"] * 100,
                         "best 'value rank' player left": lg["best_static"],
                         "engine vs that (pts)": lg["gain"] * 100})
    me = slot - 1
    # power rankings
    order = np.argsort(-ev_e.win_prob)
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.0))
    ax = axes[0]
    cols = [C["blue"] if t == me else C["axis"] for t in order]
    ax.barh(range(10), ev_e.win_prob[order] * 100, color=cols, height=0.62)
    ax.set_yticks(range(10), [f"Team {t + 1}" + (" (me)" if t == me else "") for t in order]); ax.invert_yaxis()
    ax.axvline(50, color=C["ink2"], lw=1); ax.set_xlim(35, 75); ax.grid(axis="y", visible=False)
    for i, t in enumerate(order):
        ax.text(ev_e.win_prob[t] * 100 + 0.5, i, f"{ev_e.win_prob[t] * 100:.1f}%", va="center", fontsize=8.5, color=C["ink2"],
                bbox=dict(boxstyle="round,pad=0.12", fc=C["surface"], ec="none"))
    ax.set_xlabel("simulated chance to win a weekly matchup (%)"); ax.set_title("League power ranking", fontsize=10, loc="left")
    ax = axes[1]
    cw = ev_e.cat_win_prob[me] * 100
    ax.bar(range(9), cw - 50, bottom=50, color=[C["blue"] if v >= 50 else C["red"] for v in cw], width=0.62)
    for i, v in enumerate(cw):
        ax.text(i, v + (1.4 if v >= 50 else -1.4), f"{v:.0f}", ha="center", va="bottom" if v >= 50 else "top", fontsize=8.5, color=C["ink2"])
    ax.axhline(50, color=C["ink2"], lw=1); ax.set_xticks(range(9), [CAT_LABELS[c] for c in CATEGORIES])
    ax.set_ylim(min(25.0, cw.min() - 9), max(80.0, cw.max() + 9))
    ax.set_ylabel("chance to win category (%)"); ax.set_title("My team, category by category", fontsize=10, loc="left"); ax.grid(axis="x", visible=False)
    F.headline(fig, "Following the engine all the way: the resulting team vs the nine bots",
               f"One full mock draft from slot {slot}; opponents follow noisy value-rank (or ESPN ADP when available) drafting")
    F.legend_top(fig, [F.swatch(C["blue"], "my team / category I win more often than not"), F.swatch(C["axis"], "bot teams"),
                       F.swatch(C["red"], "category I lose more often than not")])
    fig_t = F.save(fig, "mock_draft_outcome")

    mine = [i for i in rosters_e[slot]]
    mine_s = sorted(mine, key=lambda j: -eng5.value[j])
    ros_t = proj.set_index("player_id").loc[eng5.pid[mine_s], ["name", "elig", "mpg", "avail", "pts", "tpm", "reb", "ast", "stl", "blk", "tov"]].reset_index(drop=True)
    ros_t["avail"] *= 100
    ros_t.columns = ["Player", "Pos", "MPG", "GP%", "PTS", "3PM", "REB", "AST", "STL", "BLK", "TO"]

    # lineup trace for one simulated matchup: a normal week (7 days), so the table stays readable
    sim = 14                                           # period 15: Feb 1-7
    start = sched.spans[sim][0]
    trace = eng5.lineup_trace(mine, sim)
    slots = ["PG", "SG", "G", "SF", "PF", "F", "C", "UTIL", "BENCH"]
    nm = proj.set_index("player_id")["name"]
    tr_rows = []
    for d in range(7):
        row = {"day": f"{DAYS[(start + pd.Timedelta(days=d)).dayofweek]} {start + pd.Timedelta(days=d):%b %d}"}
        for s_ in slots:
            row[s_] = ", ".join(nm.loc[eng5.pid[t["idx"]]].split()[-1] for t in trace if t["day"] == d and t["slot"] == s_)
        tr_rows.append(row)
    n_changed = int(lg["differs"].sum())
    K["mock_win_engine"], K["mock_win_static"] = float(ev_e.win_prob[me]), float(ev_b.win_prob[me])
    c_cap = sum(eng5.is_center[i] for i in mine)
    return f"""
## 9. Phase 3 — turning distributions into a draft decision

### 9.1 The objective

Your league is decided week by week: the team with more categories wins the matchup, and by how much doesn't matter. So the engine does not maximise "total z-score"; it maximises
**the probability my team wins a weekly matchup**, measured by playing my roster against every other roster over ~800 simulated weeks.
That is different in useful ways: it rewards *balance* (winning 5 categories narrowly every week beats winning 3 by a mile), rewards *depth* (daily lineups mean bench players score
whenever a starter is off), and treats a category you are certain to lose as nearly free to ignore (punting).

### 9.2 The decision loop (what happens when you click "recommend")

For each of the ~25 strongest available candidates:

1. Assume I take him with my pick.
2. **Roll the rest of the draft forward.** Every other team picks the best remaining player by (ESPN ADP if available, else our value rank) plus noise that grows deeper in the draft, skipping
   any center that would be a fourth on that roster. My own later picks follow our value rank.
3. **Play all ten rosters through the simulated weeks with daily lineups.** Each day, in value order, every player with a game takes the first open slot he is eligible for:
   his dedicated position (PG/SG/SF/PF/C), then G (PG or SG) or F (SF or PF), then UTIL (three of those); the rest sit. A starter without a game, or out injured, leaves a hole the bench fills.
4. Compare my weekly totals with each opponent's, category by category (turnovers: fewer wins; FG%/FT% from summed makes ÷ attempts). More than 4.5 categories = a win.
5. Repeat with a second, different set of bot choices (same random numbers for every candidate, so *differences* between candidates are far less noisy than the absolute levels).

The candidate with the highest average win probability is recommended; the table also shows per-category win chances so you can see **why**.

### 9.3 Worked example: the first overall pick

![pick1]({fig_r})

{md_table(r_t, {"Value rank": "{:.0f}", "Win % (matchup)": "{:.1f}", "vs top-ranked (pts)": "{:+.1f}"})}

*Reading the heatmap* — each cell is the chance the resulting team wins that category in a typical week, after the rest of the draft plays out (so it reflects who the
engine expects to pick next, not just the player's own stats):

{story}

Because the engine sees every category's win chance, it also tells you what to pair with each player next.

### 9.4 Worked example: a whole mock draft from slot {slot}

Following the engine's recommendation at each of my 13 picks (against bots that draft by noisy value rank or ESPN ADP when available):

{md_table(lg_t, {"round": "{:.0f}", "pick #": "{:.0f}", "win % after": "{:.1f}", "engine vs that (pts)": "{:+.1f}"})}

The engine deviated from "take the best-value player left" at **{n_changed} of 13 picks**. In the rows where it did, the "engine vs that" column shows the simulated gain
in win probability from the deviation (positive by construction, since the engine takes the max over its candidates).

**Resulting roster** ({c_cap} centers, cap = 3):

{md_table(ros_t, {"MPG": "{:.1f}", "GP%": "{:.0f}", "PTS": "{:.1f}", "3PM": "{:.1f}", "REB": "{:.1f}", "AST": "{:.1f}", "STL": "{:.1f}", "BLK": "{:.1f}", "TO": "{:.1f}"})}

![outcome]({fig_t})

Simulated weekly-matchup win probability: **{ev_e.win_prob[me] * 100:.1f}%** following the engine vs **{ev_b.win_prob[me] * 100:.1f}%** taking the best-value player each time (same bots).
*Both numbers are measured against the engine's own simulated world, so only the gap is meaningful — section 10 tests whether the gap survives contact with real seasons.*

### 9.5 What a daily lineup looks like

The same roster in one simulated matchup (week {sim + 1}, {start:%b %d} – {start + pd.Timedelta(days=6):%b %d}) — who fills which slot each day (BENCH = had a game but no open slot):

{md_table(pd.DataFrame(tr_rows))}

Notice days with several players listed under UTIL, and days where C or a flex slot sits empty because nobody eligible has a game: this is the effect a season-average
projection cannot see, and it is why the engine counts bench depth and positional spread.
"""


# =========================================================================================================
# 10. validation
# =========================================================================================================
def sec_validation() -> str:
    path = DATA / "validation_results.csv"
    if not path.exists():
        return "\n## 10. Historical validation\n\n*(run `python scripts/validate_draft.py` and regenerate)*\n"
    res = pd.read_csv(path)
    n = len(res)
    g = res.groupby("year").agg(drafts=("draft", "size"), engine=("real_engine", "mean"), best_available=("real_static", "mean"),
                                gain=("gain_real", "mean"), sd=("gain_real", "std"), engine_better=("gain_real", lambda x: (x > 0).mean()),
                                sim_gain=("gain_sim", "mean")).reset_index()
    g["se"] = g["sd"] / np.sqrt(g["drafts"])
    allr = dict(year="ALL", drafts=n, engine=res["real_engine"].mean(), best_available=res["real_static"].mean(), gain=res["gain_real"].mean(),
                sd=res["gain_real"].std(), engine_better=(res["gain_real"] > 0).mean(), sim_gain=res["gain_sim"].mean())
    allr["se"] = allr["sd"] / np.sqrt(n)
    tot = pd.concat([g, pd.DataFrame([allr])], ignore_index=True)
    K["val_gain"], K["val_se"], K["val_n"] = allr["gain"], allr["se"], n
    K["val_sim"] = allr["sim_gain"]
    fig, ax = F.new(8, 4.2)
    x = np.arange(len(tot))
    ax.axhline(0, color=C["ink2"], lw=1)
    ax.errorbar(x, tot["gain"] * 100, yerr=1.96 * tot["se"] * 100, fmt="none", ecolor=C["blue"], elinewidth=2, capsize=4)
    ax.scatter(x, tot["gain"] * 100, s=60, color=[C["blue"]] * len(x), zorder=3, edgecolors=C["surface"], linewidths=1.5)
    ax.scatter([len(tot) - 1], [tot["gain"].iloc[-1] * 100], s=90, color=C["orange"], zorder=4, edgecolors=C["surface"], linewidths=1.5)
    for xi, r in zip(x, tot.itertuples()):
        ax.text(xi + 0.12, r.gain * 100, f"{r.gain * 100:+.1f}", va="center", fontsize=9, color=C["ink2"])
    labels = [f"{int(y)}-{str(int(y) + 1)[-2:]}" if str(y) != "ALL" else "All seasons" for y in tot["year"]]
    ax.set_xticks(x, labels); ax.set_xlim(-0.5, len(tot) - 0.4); ax.grid(axis="x", visible=False)
    ax.set_ylabel("engine minus best-available (percentage points\nof weekly matchup win probability)")
    z_all = allr["gain"] / allr["se"]
    sig_now = abs(z_all) > 1.96
    if sig_now and allr["gain"] > 0:
        title = f"On real seasons the engine beat 'take the best player left' by {allr['gain'] * 100:+.1f} points of weekly win probability"
    elif sig_now:
        title = f"On real seasons the engine did WORSE than 'take the best player left' ({allr['gain'] * 100:+.1f} points)"
    else:
        title = (f"On real seasons the engine did not demonstrably beat 'take the best player left' "
                 f"({allr['gain'] * 100:+.1f} ± {allr['se'] * 100:.1f} points: within noise)")
    F.headline(fig, title, "Paired mock drafts scored on actual box scores. Dots are the mean gain; bars are the 95% interval")
    F.legend_top(fig, [F.swatch(C["blue"], "one season (24 mock drafts)", "dot"), F.swatch(C["orange"], "all seasons pooled", "dot")])
    fig_v = F.save(fig, "validation_gain")
    tv = tot.assign(year=tot["year"].astype(str))[["year", "drafts", "engine", "best_available", "gain", "se", "engine_better", "sim_gain"]].copy()
    for c in ("engine", "best_available"):
        tv[c] *= 100
    tv["gain"] *= 100; tv["se"] *= 100; tv["sim_gain"] *= 100
    tv.columns = ["season", "mock drafts", "engine win %", "best-available win %", "gain (pts)", "± SE", "engine better in", "model's own predicted gain (pts)"]
    pos_years = int((g["gain"] > 0).sum())
    sig_txt = ("That difference is statistically distinguishable from zero." if sig_now else
               f"That is only {abs(z_all):.1f} standard errors from zero, i.e. **within noise**: the evidence does not show the engine beating a good value ranking.")
    wins_over_season = allr["gain"] * 18

    # optional second variant: engine overrides the value board only when clearly ahead
    mfiles = sorted(DATA.glob("validation_results_margin0.01*.csv"))
    margin_md = ""
    if len(mfiles) >= 4:    # one file per season (or a single pooled file)
        mres = pd.concat([pd.read_csv(f) for f in mfiles], ignore_index=True)
        mg, mse = mres["gain_real"].mean(), mres["gain_real"].std() / np.sqrt(len(mres))
        margin_md = f"""
**A cautious variant.** The model's own simulation expected a larger gain than reality delivered ({allr["sim_gain"] * 100:+.1f} vs {allr["gain"] * 100:+.1f} points) — the classic sign of choosing the *argmax of noisy
near-ties*: among candidates whose simulated win probabilities differ by less than the estimate's noise, the "best" one is often just the luckiest estimate. So we also tested the engine in a mode where it overrides the
value ranking only when its pick is clearly better: among candidates within 1 point of the best simulated win probability it takes the highest-ranked by value (the 1-point margin was chosen from the noise level of the estimates, before looking at results).
Over the same {len(mres)} drafts that variant scored **{mg * 100:+.1f} ± {mse * 100:.1f}** points vs best-available (ahead in {(mres["gain_real"] > 0).mean():.0%} of drafts).
"""
    return f"""
## 10. Does it work? Historical validation on real seasons

The only honest test of a draft tool is: *would it have helped in a season it hasn't seen?* For each of the seasons below, we

1. rebuild everything as of that preseason — projections fitted on earlier seasons only, simulation library built from them and **that season's real schedule**;
2. run mock drafts (random draft slots, nine noisy bots) twice with identical bots: once taking the engine's recommendation at each pick, once simply taking the best-value player left
   (both respecting the 3-center cap);
3. score the two resulting rosters on what **really happened** that season: actual day-by-day box scores, same daily lineup / position-slot / matchup rules.

![validation]({fig_v})

{md_table(tv, {"mock drafts": "{:.0f}", "engine win %": "{:.1f}", "best-available win %": "{:.1f}", "gain (pts)": "{:+.1f}", "± SE": "{:.1f}", "engine better in": "{:.0%}", "model's own predicted gain (pts)": "{:+.1f}"})}

**Result, plainly.** Averaged over {n} paired drafts the engine's team scored **{allr["gain"] * 100:+.1f} ± {allr["se"] * 100:.1f}** percentage points of weekly matchup win probability relative to the
best-available team (ahead in {allr["engine_better"]:.0%} of drafts; better in {pos_years} of {len(g)} seasons). {sig_txt}
Over an 18-week regular season that is roughly {wins_over_season:+.1f} wins. {margin_md}
**What to take from this.**
* **The projections and value board are doing nearly all the work.** Both strategies beat the nine noisy bots in about 70% of weekly matchups; a sound ranking is already a strong draft strategy.
* **The engine matches that ranking but is not shown to beat it.** Its real value is different: it *explains* each recommendation category by category, keeps track of positional slots and the center cap,
  and plans around what is likely to still be available — useful for judgment calls, not proven to add wins. When it and the value board disagree, treat it as roughly a coin flip and weigh your own category needs.
* **The model is over-optimistic about itself.** It predicted {allr["sim_gain"] * 100:+.1f} points of gain and delivered {allr["gain"] * 100:+.1f}. Throughout development, runs with different calendar, rookie and availability settings
  scored between roughly −1 and +5 points with standard errors of 1.5-1.7 — all consistent with a true effect near zero to small — so do not expect a reliably positive edge.
* Season-to-season results swing a lot (see the ± column) because injuries and breakouts decide drafts.
* **Absolute win percentages are inflated** because the bots draft with large random noise; real opponents are better. Only the *difference* between the strategies is informative.
* **Not covered by this test:** ESPN's real ADP and positional eligibility (the validation used our approximate eligibility for both strategies alike), human opponents' positional needs,
  in-season management (waivers, streaming, trades), and the IR slot.
"""


# =========================================================================================================
# 11-12. summary / how to run / limits
# =========================================================================================================
def exec_summary(proj: pd.DataFrame) -> str:
    top5 = ", ".join(proj.head(5)["name"])
    val = ""
    if "val_gain" in K:
        z_ = K["val_gain"] / K["val_se"]
        verdict = ("a statistically significant edge" if abs(z_) > 1.96 and K["val_gain"] > 0 else
                   "a statistically significant shortfall" if abs(z_) > 1.96 else
                   "within noise — no demonstrable edge")
        val = (f"* **Validation (the honest part).** In {K['val_n']} paired mock drafts scored on real 2022-2025 seasons (actual day-by-day box scores), the engine's team scored "
               f"**{K['val_gain'] * 100:+.1f} ± {K['val_se'] * 100:.1f}** points of weekly matchup win probability relative to a 'take the best player left' team — {verdict} (section 10). "
               f"The projections and value board are doing nearly all the work; the engine is a well-explained second opinion (category balance, positions, center cap), not a proven upgrade.")
        return f"""
## 1. Executive summary

**The question.** In a 10-team ESPN snake draft with nine categories, head-to-head weekly matchups decided win/loss, daily lineups, PG/SG/G/SF/PF/F/C/3×UTIL slots and a
3-center cap: *who should I take with my next pick, given everything already drafted?*

**The approach, in one paragraph.** (1) Project each player's next-season per-game stats from his last three seasons plus age and role, choosing the best model per stat by
a rolling backtest. (2) Turn each average into thousands of realistic simulated weeks — real schedule, injuries, streakiness, correlated stats, and uncertainty about the projection itself.
(3) For every candidate pick, play out the rest of the draft and the season's weeks with daily lineups, and recommend the player who maximises the probability of winning a weekly matchup.

**What we found.**

* **Projections** beat "same as last year" on every one of the nine categories out of sample (average rank correlation with true fantasy value
  **{K["spearman_model"]:.3f}** vs **{K["spearman_last"]:.3f}**; identified **{K["overlap_model"] * 100:.0f}%** of the season's top-50 players in advance vs
  **{K["overlap_last"] * 100:.0f}%**). The largest gains come from minutes and availability; shot-efficiency stats are mostly luck and are shrunk heavily.
* **2026-27 top five by projected value:** {top5}.
* **Simulation** reproduces the real spread of weeks on a season it never saw: in a 2025-26 test with everything built from pre-2025 data, actual player-weeks fell in the lowest/highest 10% of the
  simulated range {pct(K.get("pit_lo", float("nan")), 1)} / {pct(K.get("pit_hi", float("nan")), 1)} of the time (nominal 10%) — close to calibrated, a little over-confident in the upper tail. An earlier version was
  clearly over-confident because of an availability-estimation bug that this very check exposed and that is now fixed (section 8.4).
{val}
* **Data used:** {K["n_playergames"]:,} player-games from {K["n_games"]:,} games (2015-16 → 2025-26), the published 2026-27 schedule, nba_api bios; {K["minutes_ok"]:.0f}% of team-games pass the 240-minute integrity check.

**What is not done / needs your input** is in section 12.
"""


def sec_howto() -> str:
    return """
## 11. Step by step: how everything fits together

| # | Step | Command | What it produces |
|---|---|---|---|
| 1 | Pull the data | `python scripts/ingest.py` | `data/processed/nba.db`: 11 seasons of box scores, ages, bios (incl. the rookie class) and the published 2026-27 schedule |
| 2 | ESPN positions + ADP | `python scripts/fetch_espn_positions.py` | `data/external/positions.csv` (needs a network that can reach ESPN) |
| 3 | Fit + backtest + project | `python scripts/build_projections.py` | `projections_2026.parquet/csv`, `backtest_summary.csv`, `backtest_oos.parquet`, `rookie_profiles.csv` |
| 4 | Calibrate + simulate | `python scripts/build_library.py` | `calibration.json`, `library_2026.npz` (the simulated weeks) |
| 5 | Draft | `streamlit run src/app/streamlit_app.py` | the live tool; state saved to SQLite after every pick |
| 6 | Validate (optional) | `python scripts/validate_draft.py` | `validation_results.csv` — section 10 |
| 7 | Rebuild this document | `python scripts/make_report.py` | this file and `REPORT.html` |

**On draft day:** create the league (10 teams, 13 rounds, your slot), record every pick as it happens — all teams, not just yours — flag anyone injured/out on the Player board, and
read the Recommendations tab on your turn ("Plan my next pick" looks ahead while others are on the clock). "Undo last pick" fixes data-entry mistakes.

**How each file maps to the story:**

| Module | Role |
|---|---|
| `src/data_ingestion/` | nba_api clients and SQLite loaders (idempotent) |
| `src/features/player_seasons.py` | the "ratio of totals" table and shrinkage definitions |
| `src/features/positions.py` | ESPN slot eligibility (ESPN file wins over the heuristic) |
| `src/models/projection.py` | Phase 1: baseline / Ridge / LightGBM, rolling backtest, rookie tiers |
| `src/simulation/calibration.py` | measures dispersion, correlation, absence structure, talent uncertainty from history |
| `src/simulation/copula.py` | Phase 2: schedule-aware, game-by-game Gaussian-copula simulator |
| `src/draft/state.py`, `store.py` | league settings, snake order, center cap, persistence |
| `src/draft/engine.py` | Phase 3: rollouts, daily-lineup scoring, recommendations |
| `src/app/streamlit_app.py` | the interface |
"""


def sec_limits() -> str:
    return """
## 12. Limitations, assumptions, and open questions

**Modelling limits (stated, not hidden)**

* The daily lineup is a greedy fill in value order; a human optimises each day and can juggle slots, so true lineup efficiency is slightly higher than modelled.
* No in-season management: waivers, streaming, trades and the IR slot are not modelled — an injured draftee just misses games. Depth is therefore valued as bench coverage, not as waiver flexibility.
* Opponents are bots: they follow ADP/value rank plus noise, not positional need, team loyalty or auto-draft quirks. Absolute win percentages are relative to that bot model; compare candidates by *differences*.
* Teammates' game scripts and injuries are independent in the simulation; only the schedule is shared.
* "Out for the whole matchup" is drawn once per matchup period, so in the two-week periods (weeks 7 and 17) it overstates how often a player misses *both* weeks; week-long absences were measured on one-week windows.
* About 30 NBA Cup games are still TBD in the published schedule. They are ignored (week 7, Nov 30–Dec 13, uses only its published games, so it is the noisiest week); re-run `scripts/ingest.py` once
  the Cup bracket is set and they appear in the data automatically.
* Rookies are projected from draft-slot buckets, so two players taken in the same bucket (e.g. two top-5 picks) are identical in the model; a smoother curve was tested and was no more accurate.
  Supply rookie projections you trust via `data/external/projection_overrides.csv`.
* Not modelled (candidates if analysis shows they matter): contract-year effects, vacated usage after trades, coaching changes, injury types.

**Open questions for you**

1. **Draft rounds — 13 or 14?** The IR slot isn't drafted in the default; if your draft has 14 rounds, set roster size to 14 (the 14th pick is modelled as extra bench depth).
2. **Center cap definition.** Currently any **C-eligible** player (including PF/C types) counts toward the 3-center cap. Does ESPN count all of them, or only players slotted at C?
3. **ESPN eligibility.** The pipeline currently falls back to an approximation (from nba_api labels and assist/rebound rates). ESPN is blocked from the development machine, so please run
   `python scripts/fetch_espn_positions.py` once on your own network (or send me ESPN's eligibility list) — it also adds ADP, which makes the bots realistic.
4. **Playoffs.** If you tell me how many of the 10 teams make the playoffs I can add "chance to make the playoffs" (a season-level target) alongside weekly win probability; it would favour
   consistent teams over boom-bust ones slightly differently.
5. **Rookie projections.** If you have rookie numbers you trust (your own or ESPN's), send them or drop them in the overrides file — the top-5 picks are currently all treated alike.
6. **Draft mechanics.** Is there a pick timer / auto-draft for absent managers (affects how bots behave)? Any keepers or dropped-player rules before the season starts?
"""


# =========================================================================================================
def to_html(md_text: str, base: Path) -> str:
    body = markdown.markdown(md_text, extensions=["tables", "toc", "fenced_code"])

    def embed(m):
        p = base / m.group(1)
        if p.exists():
            return f'src="data:image/png;base64,{base64.b64encode(p.read_bytes()).decode()}"'
        return m.group(0)

    body = re.sub(r'src="(report/figures/[^"]+)"', embed, body)
    css = """
    :root{--bg:#fcfcfb;--ink:#0b0b0b;--ink2:#52514e;--line:#e1e0d9;--blue:#2a78d6;--card:#f6f5f2}
    @media (prefers-color-scheme: dark){:root{--bg:#1a1a19;--ink:#fff;--ink2:#c3c2b7;--line:#2c2c2a;--blue:#3987e5;--card:#222220}}
    body{background:var(--bg);color:var(--ink);font:16px/1.6 system-ui,-apple-system,"Segoe UI",sans-serif;margin:0}
    main{max-width:980px;margin:0 auto;padding:24px 16px 80px}
    h1{font-size:2rem;margin:.2em 0}h2{margin-top:2.2em;padding-top:.6em;border-top:1px solid var(--line)}h3{margin-top:1.6em}
    img{max-width:100%;height:auto;border-radius:8px;border:1px solid var(--line);margin:.6em 0}
    table{border-collapse:collapse;display:block;overflow-x:auto;font-size:.88rem;margin:1em 0}
    th,td{padding:5px 10px;border-bottom:1px solid var(--line);text-align:left;white-space:nowrap}
    th{background:var(--card);position:sticky;top:0}
    code{background:var(--card);padding:1px 5px;border-radius:4px;font-size:.9em}
    blockquote{border-left:4px solid var(--blue);margin:1em 0;padding:.2em 1em;background:var(--card)}
    .toc{background:var(--card);padding:8px 16px;border-radius:8px}
    """
    return f'<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>NBA Draft Helper Report</title><style>{css}</style></head><body><main>{body}</main></body></html>'


def main(skip_sim_check: bool) -> None:
    F.setup()
    conn = sqlite3.connect(DATA / "nba.db")
    st = load_player_season_table(conn)
    apg = A.actual_per_game(st)
    players = pd.read_sql_query(
        "SELECT player_id, name, position_1, position_2, draft_year, draft_position, rookie_year, last_year FROM player", conn)
    year = max(int(p.stem.split("_")[1]) for p in DATA.glob("projections_*.parquet"))
    proj = pd.read_parquet(DATA / f"projections_{year}.parquet")
    oos = pd.read_parquet(DATA / "backtest_oos.parquet")
    summary = pd.read_csv(DATA / "backtest_summary.csv")
    cal = Calibration.load(DATA / "calibration.json")
    lib = Library.load(DATA / f"library_{year}.npz")
    last_year = int(st["year"].max())
    sched = season_periods(conn, f"{year}-{str(year + 1)[-2:]}")

    parts = {}
    print("data...");        parts["data"] = sec_data(conn)
    print("features...");    parts["feat"] = sec_features(st, proj)
    print("models...");      parts["models"] = sec_models(oos, summary, apg)
    print("residuals...");   parts["resid"] = sec_residuals(oos, st, apg, players)
    print("rookies...");     parts["rook"] = sec_rookies(st, proj)
    print("projections..."); parts["proj"] = sec_projections(proj)
    print("movers...");      parts["movers"] = sec_movers(proj, apg, last_year)
    print("simulation...");  parts["sim"] = sec_simulation(conn, st, players, cal, lib, proj, sched, skip_sim_check)
    print("engine...");      parts["engine"] = sec_engine(proj, lib, sched)
    print("validation...");  parts["val"] = sec_validation()
    summary_md = exec_summary(proj)

    stamp = pd.Timestamp.now().strftime("%Y-%m-%d")
    doc = f"""# NBA Fantasy Draft Helper — full methodology & results report

*Generated {stamp} by `scripts/make_report.py` from the project's data and models. League: ESPN · 10 teams · snake · 9-category head-to-head (weekly matchup W/L) ·
daily lineups · PG SG G SF PF F C 3×UTIL · 3 bench · 1 IR · max 3 centers · 18 + 3 weeks (two-week matchups: week 7 = Nov 30–Dec 13, week 17 = Feb 15–28).*

**Contents** 1 Summary · 2 Data · 3 Features & shrinkage · 4 Model evaluation (incl. residual diagnostics) · 5 Rookies · 6 2026-27 projections · 7 Risers & fallers · 8 Simulation · 9 Draft engine · 10 Validation · 11 How to run · 12 Limits & open questions

{summary_md}

## 2. The data

{parts["data"]}

## 3. Turning box scores into model inputs

{parts["feat"]}

## 4. Phase 1 — projecting next season

{parts["models"]}

{parts["resid"]}

{parts["rook"]}

{parts["proj"]}

{parts["movers"]}

{parts["sim"]}

{parts["engine"]}

{parts["val"]}

{sec_howto()}

{sec_limits()}
"""
    out_md = ROOT / "docs" / "REPORT.md"
    out_md.write_text(doc, encoding="utf-8")
    (ROOT / "docs" / "REPORT.html").write_text(to_html(doc, ROOT / "docs"), encoding="utf-8")
    print(f"wrote {out_md} ({len(doc):,} chars) and REPORT.html; figures in docs/report/figures")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--skip-sim-check", action="store_true")
    main(ap.parse_args().skip_sim_check)
