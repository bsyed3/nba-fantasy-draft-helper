"""Sections 1-4: introduction, data and sources, the baseline (shrinkage), the learned models, the walk-forward backtest, residual diagnostics."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from scipy.stats import norm

from scripts.final_report.common import *  # noqa: F403
from src.features.player_seasons import STAT_DEFS, league_priors
from src.models import projection as P
from src.simulation.schedule import season_periods


# ---------------------------------------------------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------------------------------------------------
def fig_pipeline(ctx) -> None:
    fig, ax = plt.subplots(figsize=(11, 5.6))
    ax.set_xlim(0, 100); ax.set_ylim(0, 56); ax.axis("off"); ax.grid(False)
    boxes = [
        (1, "DATA", C["ink2"], ["NBA box scores 2015-16 → 2025-26", "player bios, draft slots, ages", "2026-27 published schedule", "ESPN injuries 'Out' list", "Yahoo rookie projections", "expert judgments (4 players)"]),
        (26, "PHASE 1  Projections", C["blue"], ["player-season table (ratios of totals)", "shrinkage baseline (recency + prior)", "Ridge / LightGBM corrections", "availability, team context, rookies", "z-score value and the −3σ floor"]),
        (51, "PHASE 2  Simulation", C["aqua"], ["Gaussian copula per game", "negative-binomial counts, binomial makes", "talent uncertainty, absence runs", "real 21-week matchup calendar", "4,200-week table for the pick policy"]),
        (76, "PHASE 3  Draft engine", C["orange"], ["roll the draft forward (bots)", "daily lineups, slots, 3-C cap", "P(win ≥ 5 of 9 categories)", "category-aware pick policy", "top-200 candidates every pick"]),
    ]
    for x, title, col, lines in boxes:
        ax.add_patch(FancyBboxPatch((x, 22), 23, 28, boxstyle="round,pad=0.4,rounding_size=1.2", fc=C["surface"], ec=col, lw=2.2))
        ax.text(x + 11.5, 47.2, title, ha="center", va="center", fontsize=10.5, fontweight="bold", color=col)
        for i, ln in enumerate(lines):
            ax.text(x + 1.2, 42.2 - i * 3.8, "• " + ln, ha="left", va="center", fontsize=8.2, color=C["ink"])
    for x in (24.2, 49.2, 74.2):
        ax.add_patch(FancyArrowPatch((x, 36), (x + 2.2, 36), arrowstyle="-|>", mutation_scale=16, color=C["ink2"], lw=1.8))
    ax.add_patch(FancyBboxPatch((1, 3), 98, 14, boxstyle="round,pad=0.4,rounding_size=1.2", fc=C["mid"], ec=C["axis"], lw=1.2))
    ax.text(50, 14.2, "HOW EACH PHASE IS TESTED ON SEASONS IT HAS NOT SEEN", ha="center", fontsize=9.5, fontweight="bold", color=C["ink2"])
    tests = ["walk-forward backtest\n(train on earlier seasons,\npredict the next; 2021-25)", "residual and ranking\ndiagnostics", "simulation realism:\nPIT, reliability, correlations\n(2022-25 weeks)", "mock drafts scored on real\nbox scores (2021-25), plus\nheld-out simulated drafts"]
    for xc, tx in zip((13, 38, 63, 88), tests):
        ax.text(xc, 8.2, tx, ha="center", va="center", fontsize=8.1, color=C["ink"])
    F.headline(fig, "The project in one picture: data → projections → simulated weeks → a pick, each stage tested on past seasons",
               "Three phases feed one question: who should I take with my next pick?")
    F.save(fig, "fig_pipeline")


def fig_schedule(ctx) -> dict:
    s = season_periods(ctx.conn, "2026-27", n_regular=18, n_playoff=3, fill_unscheduled=False)
    g = s.masks.sum(axis=2)                                    # [periods, teams] games
    mean, lo, hi = g.mean(axis=1), g.min(axis=1), g.max(axis=1)
    spans = s.spans
    two = [i for i, (a, b) in enumerate(spans) if (b - a).days > 7]
    fig, ax = F.new(10, 4.3)
    x = np.arange(1, len(mean) + 1)
    cols = [C["orange"] if i in two else (C["aqua"] if i >= 18 else C["blue"]) for i in range(len(mean))]
    ax.bar(x, mean, color=cols, width=0.7)
    ax.errorbar(x, mean, yerr=[mean - lo, hi - mean], fmt="none", ecolor=C["ink2"], elinewidth=1, capsize=2)
    ax.set_xticks(x); ax.set_xlabel("matchup period (1-18 regular season, 19-21 fantasy playoffs)"); ax.set_ylabel("games per team (bar = mean; whiskers = min to max)")
    ax.grid(axis="x", visible=False)
    for i in two:
        ax.text(x[i], hi[i] + 0.15, f"{spans[i][0]:%b %d}–\n{spans[i][1]:%b %d}", ha="center", fontsize=7.6, color=C["ink2"])
    F.headline(fig, "The real 2026-27 calendar: 21 matchup periods, two of them two weeks long",
               "Mon-Sun weeks, except the NBA Cup stretch (week 7) and the All-Star stretch (week 17). The ~30 unscheduled Cup games are ignored, so week 7 shows fewer games.")
    F.legend_top(fig, [F.swatch(C["blue"], "regular-season week"), F.swatch(C["orange"], "two-week period"), F.swatch(C["aqua"], "fantasy playoff week")])
    F.save(fig, "fig_schedule")
    return dict(n_periods=len(mean), two=[(spans[i][0], spans[i][1]) for i in two])


def fig_folds(ctx) -> None:
    seasons = list(range(2015, 2026))
    folds = [2021, 2022, 2023, 2024, 2025]
    fig, ax = F.new(10, 3.9)
    for r, T in enumerate(folds):
        for y in seasons:
            col = C["blue"] if y < T and y >= 2016 else (C["orange"] if y == T else (C["axis"] if y > T else "#dfe6ef"))
            ax.add_patch(plt.Rectangle((y - 0.45, r - 0.36), 0.9, 0.72, color=col, lw=0))
        ax.text(2025.7, r, f"fold {T}", va="center", fontsize=8.8, color=C["ink2"])
    ax.set_xlim(2014.4, 2027); ax.set_ylim(-0.7, len(folds) - 0.3); ax.invert_yaxis()
    ax.set_xticks(seasons, [f"{y}-{str(y + 1)[-2:]}" for y in seasons], rotation=0, fontsize=7.8)
    ax.set_yticks([]); ax.grid(False)
    F.headline(fig, "Walk-forward (rolling-origin) backtest: every fold predicts a season using only the seasons before it",
               "Blue = training seasons (targets 2016-17 onward; 2015-16 is only history), orange = the season being predicted, grey = the future, which the model never sees")
    F.legend_top(fig, [F.swatch(C["blue"], "training seasons"), F.swatch(C["orange"], "season predicted (out of sample)"), F.swatch(C["axis"], "future (unseen)")])
    F.save(fig, "fig_folds")


def fig_shrinkage(ctx, train: pd.DataFrame, params: dict) -> dict:
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
        rows.append(dict(bucket=name, n=len(g), raw=np.sqrt(((g["y_p3"] - g["raw"]) ** 2).mean()), league=np.sqrt(((g["y_p3"] - g["lg"]) ** 2).mean()),
                         shrunk=np.sqrt(((g["y_p3"] - g["blend"]) ** 2).mean())))
    sh = pd.DataFrame(rows)
    fig, ax = F.new(8.4, 4.4)
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
    F.headline(fig, "Small samples lie: shrinking toward the norm removes most of the error", "Error predicting next season's 3P% (RMSE, percentage points); lower is better; training seasons only")
    F.legend_top(fig, *ax.get_legend_handles_labels())
    F.save(fig, "fig_shrinkage")
    return dict(table=sh, gain_small=(1 - sh.iloc[0]["shrunk"] / sh.iloc[0]["raw"]) * 100)


def fig_backtest_bars(ctx) -> None:
    summary = ctx.summary
    pv = summary[summary["model"] != "baseline"].pivot(index="stat", columns="model", values="vs_baseline_pct")
    order = pv.min(axis=1).sort_values().index
    fig, ax = F.new(8.6, 5.2)
    y = np.arange(len(order)); h = 0.36
    ax.barh(y - h / 2, pv.loc[order, "ridge"], h, color=C["blue"], label="Ridge on the residual")
    lg = pv.loc[order, "lgbm"]
    ax.barh(y + h / 2, lg.fillna(0), h, color=C["orange"], label="LightGBM on the residual")
    for yi, s in zip(y, order):
        ax.text(pv.loc[s, "ridge"] + (-0.4 if pv.loc[s, "ridge"] < 0 else 0.4), yi - h / 2, f"{pv.loc[s, 'ridge']:+.1f}", va="center",
                ha="right" if pv.loc[s, "ridge"] < 0 else "left", fontsize=8, color=C["ink2"])
        if not np.isnan(lg[s]):
            ax.text(lg[s] + (-0.4 if lg[s] < 0 else 0.4), yi + h / 2, f"{lg[s]:+.1f}", va="center", ha="right" if lg[s] < 0 else "left", fontsize=8, color=C["ink2"])
    ax.axvline(0, color=C["ink2"], lw=1)
    ax.set_yticks(y, [STAT_LABEL[s] for s in order]); ax.invert_yaxis()
    ax.set_xlabel("change in out-of-sample squared error vs the shrunken-average baseline (%)  (left of 0 = better)")
    ax.grid(axis="y", visible=False); ax.set_xlim(-21, 5)
    F.headline(fig, "The learned corrections help most on minutes, shot volume and availability", "Walk-forward backtest, folds 2021-2025 (percentages use Ridge only)")
    F.legend_top(fig, *ax.get_legend_handles_labels())
    F.save(fig, "fig_backtest_bars")


def fig_pred_vs_actual(ctx) -> pd.DataFrame:
    acc = A.accuracy_table(ctx.oos, ctx.apg)
    fig, axes = plt.subplots(2, 3, figsize=(10.4, 6.5))
    cur, m = ctx.apg.set_index(["player_id", "year"]), ctx.oos.set_index(["player_id", "year"])
    idx = m.index.intersection(cur.index)
    for ax, s in zip(axes.ravel(), ["pts", "reb", "ast", "tpm", "stl", "blk"]):
        yv, pv_ = cur.loc[idx, s], m.loc[idx, s]
        ax.scatter(pv_, yv, s=9, color=C["blue"], alpha=0.35, linewidths=0)
        lim = max(pv_.max(), yv.max()) * 1.03
        ax.plot([0, lim], [0, lim], color=C["ink2"], lw=1)
        ax.set_title(f"{CAT_LABELS[s]} per game", fontsize=10, loc="left")
        ax.text(0.04, 0.9, f"R² = {acc.set_index('stat').loc[s, 'r2_model']:.2f}", transform=ax.transAxes, fontsize=9, color=C["ink2"])
        ax.set_xlim(0, lim); ax.set_ylim(0, lim); ax.set_xlabel("projected"); ax.set_ylabel("actual")
    F.headline(fig, "Projections track what players actually did: points and rebounds best, steals hardest",
               f"Each dot is a player-season the model had not seen ({len(idx):,} of them, 2021-2025, ≥500 minutes)")
    F.legend_top(fig, [F.swatch(C["blue"], "player-season (never seen in training)", "dot"), F.swatch(C["ink2"], "perfect projection", "line")])
    F.save(fig, "fig_pred_vs_actual")
    return acc


def fig_ranking(ctx) -> pd.DataFrame:
    rk = A.ranking_accuracy(ctx.oos, ctx.apg)
    fig, ax = F.new(8.2, 4.3)
    x = np.arange(len(rk)); w = 0.36
    ax.bar(x - w / 2, rk["spearman_last_season"], w, color=C["orange"], label="rank by last season's production")
    ax.bar(x + w / 2, rk["spearman_model"], w, color=C["blue"], label="rank by our projection")
    for xi, a, b in zip(x, rk["spearman_last_season"], rk["spearman_model"]):
        ax.text(xi - w / 2, a + 0.008, f"{a:.2f}", ha="center", fontsize=8.5, color=C["ink2"])
        ax.text(xi + w / 2, b + 0.008, f"{b:.2f}", ha="center", fontsize=8.5, color=C["ink2"])
    ax.set_xticks(x, [SEASON(y) for y in rk["year"]]); ax.set_ylim(0, 1.0)
    ax.set_ylabel("Spearman ρ with actual value")
    F.headline(fig, "A draft needs an ordering, and our projection orders players better than last year's stats did",
               "Spearman correlation between predicted and actual 9-category value in each backtest season, same players")
    F.legend_top(fig, *ax.get_legend_handles_labels())
    F.save(fig, "fig_ranking")
    return rk


def residual_figures(ctx) -> dict:
    rf = A.residual_frame(ctx.oos, ctx.st, ctx.apg, ctx.names)
    rs = A.residual_summary(rf).set_index("stat")
    lab = {**CAT_LABELS, "mpg": "MPG"}
    # A: residual vs projected level
    fig, axes = plt.subplots(2, 3, figsize=(10.4, 6.5))
    flagged = 0
    for ax, c in zip(axes.ravel(), ["pts", "reb", "ast", "tpm", "stl", "mpg"]):
        x, y = rf[f"{c}_pred"], rf[f"{c}_res"]
        ax.scatter(x, y, s=7, color=C["blue"], alpha=0.22, linewidths=0)
        b = A.binned_mean(x, y)
        ax.fill_between(b["x"], b["lo"], b["hi"], color=C["orange"], alpha=0.25, linewidth=0)
        ax.plot(b["x"], b["mean"], color=C["orange"], lw=2.2)
        ax.axhline(0, color=C["ink2"], lw=1)
        ax.set_title(lab[c], fontsize=10, loc="left"); ax.set_xlabel("projected"); ax.set_ylabel("actual − projected")
        lim = np.percentile(np.abs(y), 99) * 1.1
        ax.set_ylim(-lim, lim)
    F.headline(fig, "Residuals against the projected level: flat through the middle, with mild regression to the mean at the extremes",
               "Each dot is a player-season (actual minus projected). The orange line is the average miss within projection deciles, with its 95% interval")
    F.legend_top(fig, [F.swatch(C["blue"], "player-season", "dot"), F.swatch(C["orange"], "average miss by projection decile", "line"), F.swatch(C["orange"], "95% interval of that average")])
    F.save(fig, "fig_resid_fitted")
    # B: distributions with a normal curve
    fig, axes = plt.subplots(3, 3, figsize=(10, 7.4))
    for ax, c in zip(axes.ravel(), A.CAT_COLS):
        z = (rf[f"{c}_res"] - rf[f"{c}_res"].mean()) / rf[f"{c}_res"].std()
        ax.hist(z.clip(-4.5, 4.5), bins=np.linspace(-4.5, 4.5, 31), density=True, color=C["blue"], edgecolor=C["surface"], linewidth=0.6)
        xs = np.linspace(-4.5, 4.5, 200)
        ax.plot(xs, norm.pdf(xs), color=C["orange"], lw=2)
        ax.set_title(f"{lab[c]}  (skew {rs.loc[c, 'skew']:+.2f})", fontsize=9.5, loc="left")
        ax.set_yticks([]); ax.grid(axis="y", visible=False)
    F.headline(fig, "Miss sizes are close to bell-shaped; the heavy tails are where injuries and role changes live",
               "Standardised residuals (actual − projected, in standard deviations) against a normal curve")
    F.legend_top(fig, [F.swatch(C["blue"], "standardised miss (player-seasons)"), F.swatch(C["orange"], "normal curve", "line")])
    F.save(fig, "fig_resid_dist")
    # C: normal Q-Q for points and minutes
    fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.8))
    for ax, c in zip(axes, ["pts", "ast", "mpg"]):
        r = np.sort(((rf[f"{c}_res"] - rf[f"{c}_res"].mean()) / rf[f"{c}_res"].std()).dropna().to_numpy())
        q = norm.ppf((np.arange(1, len(r) + 1) - 0.5) / len(r))
        ax.scatter(q, r, s=6, color=C["blue"], alpha=0.5, linewidths=0)
        ax.plot([-4, 4], [-4, 4], color=C["orange"], lw=2)
        ax.set_xlim(-4, 4); ax.set_ylim(-5, 5); ax.set_title(lab[c], fontsize=10, loc="left")
        ax.set_xlabel("theoretical normal quantile"); ax.set_ylabel("observed standardised residual")
    F.headline(fig, "Normal Q-Q plots: the middle follows the line; the upper tail (breakout seasons) is heavier than normal for points and assists",
               "If residuals were Gaussian the dots would lie on the orange line; dots above it on the right are larger positive surprises than a normal curve allows. Minutes are close to Gaussian")
    F.legend_top(fig, [F.swatch(C["blue"], "residual quantile", "dot"), F.swatch(C["orange"], "normal reference", "line")])
    F.save(fig, "fig_resid_qq")
    # D: bias by age and projected minutes
    def rel_by(col_bucket, edges, labels, cats):
        out = {}
        b = pd.cut(rf[col_bucket], edges, labels=labels, right=False)
        for c in cats:
            g = rf.assign(b=b).groupby("b", observed=True)
            m_res, n = g[f"{c}_res"].mean(), g[f"{c}_res"].size()
            se = g[f"{c}_res"].std() / np.sqrt(n)
            avg = g[f"{c}_act"].mean()
            out[c] = pd.DataFrame({"rel": m_res / avg * 100, "se": se / avg * 100, "n": n})
        return out
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.3), sharey=True)
    cols3 = {"pts": C["blue"], "reb": C["orange"], "ast": C["aqua"]}
    age_edges, age_lab = [0, 22, 24, 26, 28, 30, 32, 34, 60], ["<22", "22-23", "24-25", "26-27", "28-29", "30-31", "32-33", "34+"]
    mpg_edges, mpg_lab = [0, 15, 20, 25, 30, 35, 60], ["<15", "15-20", "20-25", "25-30", "30-35", "35+"]
    for ax, (key, edges, labels, ttl) in zip(axes, [("age", age_edges, age_lab, "by age"), ("mpg_pred", mpg_edges, mpg_lab, "by projected minutes")]):
        rb = rel_by(key, edges, labels, list(cols3))
        x = np.arange(len(labels))
        for i, (c, colr) in enumerate(cols3.items()):
            d = rb[c].reindex(labels)
            ax.errorbar(x + (i - 1) * 0.12, d["rel"], yerr=1.96 * d["se"], fmt="o-", color=colr, lw=1.6, ms=5, capsize=2, label=lab[c])
        ax.axhline(0, color=C["ink2"], lw=1)
        ax.set_xticks(x, labels); ax.set_title(f"Average miss as % of typical value, {ttl}", fontsize=10, loc="left")
        ax.set_xlabel("age" if key == "age" else "projected minutes per game")
    axes[0].set_ylabel("actual − projected, % of average (bars = 95% interval)")
    F.headline(fig, "Bias by age and by role: points above zero mean the model under-projected that group", "Look for departures that are both large and consistent across the three categories")
    F.legend_top(fig, *axes[0].get_legend_handles_labels())
    F.save(fig, "fig_resid_group")
    # E: stability by season
    rows = []
    for c in A.CAT_COLS + ["mpg"]:
        g = rf.groupby("year")
        rows.append((g[f"{c}_res"].mean() / g[f"{c}_act"].mean() * 100).rename(lab[c]))
    st_t = pd.concat(rows, axis=1).T
    fig, ax = F.new(8.2, 4.8)
    im = ax.imshow(st_t.to_numpy(), cmap=F.DIV_CMAP, vmin=-12, vmax=12, aspect="auto")
    ax.set_xticks(range(st_t.shape[1]), [SEASON(y) for y in st_t.columns]); ax.set_yticks(range(st_t.shape[0]), st_t.index); ax.grid(False)
    for i in range(st_t.shape[0]):
        for j in range(st_t.shape[1]):
            v = st_t.iloc[i, j]
            ax.text(j, i, f"{v:+.1f}%", ha="center", va="center", fontsize=8.5, color="white" if abs(v) > 8 else C["ink"])
    for sp in ax.spines.values():
        sp.set_visible(False)
    fig.colorbar(im, ax=ax, fraction=0.04, label="average miss (% of typical value)")
    F.headline(fig, "Is the bias stable season to season? Mostly, with some league-wide shifts", "Average (actual − projected) as % of the season's typical value, by category and backtest season")
    F.save(fig, "fig_resid_season")
    # F: residual correlation
    cc = rf[[f"{c}_res" for c in A.RESID_COLS]].corr()
    fig, ax = F.new(6.8, 5.8)
    im = ax.imshow(cc.to_numpy(), cmap=F.DIV_CMAP, vmin=-1, vmax=1)
    names = [lab[c] for c in A.RESID_COLS]
    ax.set_xticks(range(len(names)), names, rotation=45, ha="right"); ax.set_yticks(range(len(names)), names); ax.grid(False)
    for i in range(len(names)):
        for j in range(len(names)):
            v = cc.iloc[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7.5, color="white" if abs(v) > 0.6 else C["ink"])
    fig.colorbar(im, ax=ax, fraction=0.04, label="correlation of misses")
    F.headline(fig, "Misses are correlated: when the model under-projects minutes, it under-projects everything", "Correlation between residuals across categories (these shape the simulation's talent uncertainty)")
    F.save(fig, "fig_resid_corr")
    return dict(rf=rf, rs=rs, mpg_corr=float(cc.loc["pts_res", "mpg_res"]), lab=lab)


# ---------------------------------------------------------------------------------------------------------------------
# text
# ---------------------------------------------------------------------------------------------------------------------
def build(ctx) -> str:
    st, proj, S = ctx.st, ctx.proj, ctx.S
    conn = ctx.conn
    first, last = int(st["year"].min()), int(st["year"].max())

    # data volume + integrity
    t = pd.read_sql_query("""SELECT g.season, COUNT(DISTINCT g.game_id) AS games, COUNT(*) AS player_games, COUNT(DISTINCT bs.player_id) AS players
                             FROM box_score bs JOIN game g ON g.game_id = bs.game_id GROUP BY g.season ORDER BY g.season""", conn)
    m = pd.read_sql_query("""SELECT g.season, bs.game_id, bs.team_id, SUM(bs.min) AS m FROM box_score bs JOIN game g ON g.game_id = bs.game_id GROUP BY 1, 2, 3""", conn)
    m["ok"] = (m["m"] - (240 + 25 * ((m["m"] - 240) / 25).round())).abs() < 3
    S["n_playergames"], S["n_games"], S["minutes_ok"] = int(t["player_games"].sum()), int(t["games"].sum()), float(m["ok"].mean())
    sched = pd.read_sql_query("SELECT COUNT(*) n, MIN(date) a, MAX(date) b FROM game WHERE season = '2026-27'", conn).iloc[0]
    t_show = t.rename(columns={"season": "season", "games": "games", "player_games": "player-games", "players": "players"})

    fig_pipeline(ctx)
    sch = fig_schedule(ctx)
    fig_folds(ctx)

    # baseline parameters + shrinkage
    priors, sg = league_priors(st), st.groupby("year")["sg"].first().to_dict()
    wides = {T: P.build_wide(st, priors, T, sg) for T in range(first + 1, last + 2)}
    train = pd.concat([wides[T] for T in range(first + 1, last + 1)], ignore_index=True)
    params = P.fit_baseline_params(train)
    S["params"] = params
    shr = fig_shrinkage(ctx, train, params)
    sh = shr["table"]
    w26 = wides[last + 1]
    cand = w26[(w26["n_lags"] == 3) & (w26["fga3_1"] >= 100) & (w26["fga3_1"] <= 300)].copy()
    cand["raw1"] = cand["tpm_1"] / cand["fga3_1"]
    cand["bl"] = P.blend(cand, "p3", *params["p3"])
    row = cand.loc[(cand["raw1"] - cand["bl"]).abs().idxmax()]
    decay, k = params["p3"]
    prior = float(row["prior_p3"])
    ex, num, den = [], 0.0, 0.0
    for l in (1, 2, 3):
        att, mk = row[f"fga3_{l}"], row[f"tpm_{l}"]
        wgt = decay ** (l - 1)
        ex.append({"season": SEASON(int(row["year"]) - l), "3PA": att, "3PM": mk, "raw 3P%": mk / att if att else np.nan, "weight": wgt})
        num += wgt * (0 if np.isnan(mk) else mk); den += wgt * (0 if np.isnan(att) else att)
    est = (num + k * prior) / (den + k)
    final = float(proj.loc[proj["player_id"] == row["player_id"], "r_p3"].iloc[0])
    pname = row["name"]

    # parameter table
    ptab = pd.DataFrame([{"stat": STAT_LABEL[s], "recency decay": params[s][0], "prior strength k": params[s][1]} for s in P.ALL_STATS])

    # model competition + accuracy + ranking + residuals
    fig_backtest_bars(ctx)
    acc = fig_pred_vs_actual(ctx)
    rk = fig_ranking(ctx)
    summ = ctx.summary
    sel = summ[summ["selected"]].set_index("stat")
    base = summ[summ["model"] == "baseline"].set_index("stat")
    comp = pd.DataFrame({"stat": [STAT_LABEL[s] for s in sel.index], "winning model": sel["model"].values, "baseline RMSE": base.loc[sel.index, "rmse"].values,
                         "winning RMSE": sel["rmse"].values, "change in squared error": sel["vs_baseline_pct"].values})
    S["sel"] = sel
    acc_t = pd.DataFrame({"category": [CAT_LABELS[s] for s in acc["stat"]], "player-seasons": acc["n"], "average value": acc["mean_actual"], "MAE (ours)": acc["mae_model"],
                          "MAE (same as last year)": acc["mae_last_season"], "MAE improvement": acc["mae_gain_pct"], "R² (ours)": acc["r2_model"], "R² (last year)": acc["r2_last_season"]})
    S["spearman_model"], S["spearman_last"] = float(rk["spearman_model"].mean()), float(rk["spearman_last_season"].mean())
    S["spearman_by_season"] = (float(rk["spearman_model"].min()), float(rk["spearman_model"].max()))
    S["acc"] = acc
    res = residual_figures(ctx)
    rf, rs, lab = res["rf"], res["rs"], res["lab"]
    scale = {c: (100.0 if c in ("fg_pct", "ft_pct") else 1.0) for c in A.RESID_COLS}
    rt = pd.DataFrame({"stat": [lab[c] + (" (pts)" if c in ("fg_pct", "ft_pct") else "") for c in A.RESID_COLS], "n": rs["n"].values,
                       "bias (actual − projected)": [rs.loc[c, "bias"] * scale[c] for c in A.RESID_COLS], "± SE": [rs.loc[c, "bias_se"] * scale[c] for c in A.RESID_COLS],
                       "SD of miss": [rs.loc[c, "sd"] * scale[c] for c in A.RESID_COLS], "MAE": [rs.loc[c, "mae"] * scale[c] for c in A.RESID_COLS],
                       "skew": rs["skew"].values, "share beyond 2 SD": rs["tail2"].values, "ρ (error vs level)": rs["hetero_rho"].values})
    z_bias = (rs["bias"] / rs["bias_se"]).abs()
    biased = [lab[c] for c in A.RESID_COLS if z_bias[c] > 2.5]
    het = rs["hetero_rho"].drop("mpg")
    fold_n = ctx.oos.groupby("year").size()
    S["n_oos"] = int(fold_n.sum())

    return f"""
# 1. Introduction

## 1.1 Purpose

This project builds a **fantasy-basketball draft assistant** for one specific league: an ESPN snake draft with 10 teams, nine scoring categories (points, three-pointers made, rebounds, assists, steals, blocks, field-goal percentage, free-throw percentage and turnovers), and **head-to-head weekly matchups in which only the result matters**. A team wins its week by beating its opponent in more than 4.5 of the nine categories; the margin is irrelevant. Lineups are set daily into ten active slots (PG, SG, G, SF, PF, F, C and three UTIL), there are three bench spots, a roster may hold at most three centers, and the season is 18 regular-season weeks followed by three playoff weeks.

The purpose is deliberately narrow: *for each pick, tell me which of the top-200 players gives my team the best chance of winning*, assuming everyone keeps the roster they draft all season (no waiver pickups or trades; those are deliberately out of scope). The project is a personal one, but it is written up here like a small applied-statistics study: define the question, specify the models, test them on seasons they have not seen, and report where they fail.

## 1.2 Why this is a statistics problem

Three features of the setting make a simple "sum of z-scores" ranking inadequate, and each one motivates a part of the method:

1. **The target is noisy and skewed.** A player's next-season stat line is a random variable whose mean we must estimate from a few noisy seasons. Small samples (a 12-for-20 shooting stretch) mislead, and the most important nuisance variable, *games played*, is dominated by injuries. This motivates **shrinkage estimation**, **regularised and boosted regression of residuals**, and a careful definition of availability (sections 3-5).
2. **The objective is a probability, not a sum.** The payoff of a roster is the probability that it wins more than 4.5 of 9 categories in a random week against a random opponent. That depends on the *distribution* of weekly totals (their spread, correlation and tails), so we need a **stochastic simulation** of weeks rather than a table of averages (section 9).
3. **Decisions are sequential and interactive.** The value of a player depends on who else is on my roster and on who will still be available at my next pick, which depends on nine other drafters. This motivates **roll-outs of the rest of the draft** and a category-aware pick rule (section 10).

## 1.3 Overview of the approach

{ctx.fig("fig_pipeline", "The three phases of the project and how each is tested on seasons it has not seen.")}

Phase 1 projects each player's per-game statistics for 2026-27. Phase 2 turns those averages into thousands of simulated fantasy weeks played on the real schedule. Phase 3 plays out the draft and the season for each candidate pick and recommends the player who maximises the probability of winning a weekly matchup. Every phase is evaluated against seasons it was not fitted on (the **walk-forward backtest**, the **realism check** of the simulated weeks, and **mock drafts scored on real box scores**); sections 4, 9 and 10 explain exactly how, because the credibility of the whole project rests on those tests.

*Roadmap.* Section 2 describes the data and its sources. Sections 3-8 cover Phase 1 (baseline, learned models and their testing, availability, team context, rookies and expert judgment, and the conversion to a ranking, including the long "Giannis" episode). Section 9 covers the simulation and its validation, section 10 the draft engine and pick policy, section 11 compares the final ranking with the market, section 12 lists last season's biggest jumpers and fallers, section 13 concludes, and section 14 sets out next steps (role and usage, returning from injury, roster change, positional depth).

# 2. Data and sources

## 2.1 Where the data come from

| Source | What was taken | Use in the project |
|---|---|---|
| NBA statistics API (`nba_api`, endpoint `LeagueGameLog`) | every player's box score in every regular-season game, 2015-16 to 2025-26 | the raw facts behind every projection |
| `nba_api` `PlayerIndex` | position label, draft slot and year, rookie year, current team (including the incoming rookie class) | priors for rookies, current teams, position groups |
| `nba_api` `LeagueDashPlayerBioStats` | age by season | aging effects |
| `nba_api` `ScheduleLeagueV2` | the published 2026-27 schedule ({int(sched.n)} games, {sched.a} to {sched.b}) | the simulation plays every player on the days his team plays |
| ESPN injury report (espn.com/nba/injuries), "Out" designations, read on 14 Oct 2026 | players ruled out at the start of the season and the expected duration | the *known absences* input (section 5.3) |
| Yahoo draft-class projections (workbook supplied by the user) | per-game projections for 15 rookies | replaces the draft-slot profile for those rookies (section 7) |
| Hashtag Basketball ADP workbook (blended Yahoo, ESPN and Fantrax average draft position, 14 Oct 2026) | the market's draft order | **comparison only**; never an input to any model (section 11) |
| Expert judgment (the project owner) | four per-player inputs (Ingram, Porziņģis, Giannis, Haliburton) | documented overrides (section 7.2) |

Only regular-season games count (game ids beginning `002`). The NBA Cup final carries a different id family and is not part of the fantasy season; the Cup quarter- and semi-finals are regular-season games and are included. Exact ESPN position eligibility was not available (ESPN blocks the development machine), so positions are *approximated* from the API's position labels and each player's statistics; this affects only lineup-slot assignment, not projections.

## 2.2 What the database contains

{md_table(t_show, {"games": "{:,.0f}", "player-games": "{:,.0f}", "players": "{:.0f}"})}

In total there are **{S["n_playergames"]:,} player-games from {S["n_games"]:,} games**. The 2019-20 and 2020-21 seasons are shorter (about 71 and 72 games per team: the pandemic bubble and a compressed schedule) and the 2026-27 row is the schedule only. As an integrity check, the players' minutes in an NBA game must sum to 240 plus 25 per overtime for each team. {pct(S["minutes_ok"])} of the {len(m):,} team-games pass this test (within 3 minutes), so essentially every player who took the floor is in the data.

**Design rules that keep the numbers honest.** Points are a *generated column* (`2·FGM + 3PM + FTM`), so a points total can never disagree with its shots. Percentages are never stored or averaged: FG% is always `ΣFGM ÷ ΣFGA` over whatever window is asked about, which matters because a player who goes 1-for-1 one night and 1-for-20 the next shot 2-for-21 (9.5%), not 52.5%. This "totals first, divide last" rule is the basis of the next section.

# 3. Phase 1, step 1: the player-season table and a shrinkage baseline

## 3.1 Everything is a ratio of two totals

Almost every quantity we model is "something per something": rebounds per 36 minutes, makes per attempt, minutes per game, games played per team game. For each player-season we store the two totals and form the ratio only at the end:

| What we model | numerator ÷ denominator | scale |
|---|---|---|
{chr(10).join(f"| {STAT_LABEL[s]} | {n_} ÷ {d_} | {'×36' if sc == 36 else ''} |" for s, (n_, d_, sc) in STAT_DEFS.items())}

Points, FG% and FGM are not modelled separately; they follow arithmetically from the shot-level quantities: `FGM = 2P% × 2PA + 3P% × 3PA` and `PTS = 2·FGM + 3PM + FTM`. Modelling the *components* (attempts and make rates) is statistically preferable to modelling points directly, because the components have different noise levels: shot volume is a stable, role-driven quantity, whereas make rates are mostly luck at season scale.

## 3.2 Shrinkage: how much to trust a small sample

A player who went 12-for-20 from three last year is probably not a 60% shooter. The cure is to blend his own record with what comparable players do, in proportion to how much evidence there is. This is an **empirical-Bayes (shrinkage) estimator**, in the spirit of Efron and Morris's classic baseball-batting-average example:

> estimate = ( Σₗ wₗ · makesₗ + k · league_rate ) ÷ ( Σₗ wₗ · attemptsₗ + k ),   wₗ = decay^(l−1),  l = 1, 2, 3 seasons back

Here `league_rate` is the average rate for the player's position group (the *prior*), `k` is the number of "pseudo-attempts" of league-average play added to his record (a large `k` means a strong prior), and `decay` down-weights older seasons. Neither is guessed: for each stat they are chosen by grid search to minimise next-season weighted squared error **on training seasons only**. The fitted values (using all seasons up to 2025-26) are:

{md_table(ptab, {"recency decay": "{:.2f}", "prior strength k": "{:g}"})}

Reading the table: stats that are mostly luck (3-point and free-throw percentage) get a large `k`, so the estimate leans on the league prior; stable, role-driven quantities (minutes, shot attempts) get small `k` and fast decay, so they follow recent seasons closely. For minutes and availability the "exposure" is games, so `k` is on a games scale.

{ctx.fig("fig_shrinkage", "Error predicting next season's 3P% by last season's attempts. Shrinkage helps most for the smallest samples and converges with the raw rate for high-volume shooters.")}

For a player who attempted fewer than 50 threes, using his raw percentage is off by {sh.iloc[0]["raw"] * 100:.1f} points on average; the shrunken blend is off by {sh.iloc[0]["shrunk"] * 100:.1f} points, a **{shr["gain_small"]:.0f}% reduction**. For 300+ attempts the methods agree because there is enough evidence to trust the player.

**Worked example: {pname}** (the 2026-27 shooter, among those with 100-300 attempts, where shrinkage changes the estimate most):

{md_table(pd.DataFrame(ex), {"3PA": "{:.0f}", "3PM": "{:.0f}", "raw 3P%": "{:.3f}", "weight": "{:.2f}"})}

With a position-group prior of **{prior:.3f}** and `k = {k:g}`: numerator = {num:.1f} + {k:g}·{prior:.3f} = {num + k * prior:.1f}, denominator = {den:.1f} + {k:g} = {den + k:.1f}, so the shrunken estimate is **{est:.3f}**, compared with {row["tpm_1"] / row["fga3_1"]:.3f} from his last season alone. The final 2026-27 projection after the learned correction of section 4 is {final:.3f}.

# 4. Phase 1, step 2: learned corrections and how they were tested

## 4.1 Three predictors per statistic

The shrinkage baseline uses only a player's own history. To use *other* information (age, experience, draft slot, position group, minutes and availability in each of the last three seasons, and the blends of all other stats) the baseline is supplemented by learned models of its **residual**, `y − baseline`. Fitting the residual rather than the raw statistic keeps the models stable: with no signal, they predict "no correction". For each statistic three candidates compete:

1. **Baseline**: the shrunken, recency-weighted average of section 3.
2. **Ridge regression**: a linear model of the residual with an L2 penalty, `argmin Σ wᵢ (rᵢ − xᵢᵀβ)² + α‖β‖²`, with features standardised and `α` chosen by cross-validation over a logarithmic grid (1 to 10,000). Ridge is appropriate when features are many and correlated (the blends of nine stats are highly collinear) and the signal per feature is small.
3. **LightGBM** (gradient-boosted regression trees): an additive ensemble of 250 small trees (6 leaves, learning rate 0.03, row and column subsampling 0.8, L2 penalty 5, at least 40 samples per leaf). Trees can learn non-linear shapes such as the bend of the aging curve and interactions such as age × role. Percentages use only baseline and Ridge, because shot-efficiency changes are close to linear in age and sample size and flexible models mostly fit noise there.

All models are weighted by exposure (minutes for rates, attempts for percentages, games for minutes and availability), so a 2,500-minute season counts more than a 600-minute one.

## 4.2 How the models were tested on previous years

A model's training error says almost nothing about how it will do next season, so the central question is: *how well does each method predict a season it has never seen?* Two design choices answer it.

**(a) A time-ordered (walk-forward) split, not a random split.** Randomly assigning player-seasons to training and test sets would let the model train on a player's 2024 season while predicting his 2023, using the future to predict the past, and it would also put the same player in both sets. Instead we use a **rolling-origin backtest** (a form of time-series cross-validation). For each target season `T` in 2021-22 ... 2025-26 we train on every season *before* `T` and predict `T`:

{ctx.fig("fig_folds", "Structure of the walk-forward backtest. Each fold trains only on seasons before the one it predicts; the final fold (2025-26) trains on nine seasons of targets.")}

Fold sizes (players with at least 500 minutes in the predicted season): {", ".join(f"{SEASON(y)}: {n:,}" for y, n in fold_n.items())}, a total of {int(fold_n.sum()):,} out-of-sample player-seasons. Even the baseline's `decay` and `k` are re-fitted inside each fold from that fold's training seasons only, so there is no leakage of any kind.

**(b) A single pooled error criterion and a pre-declared selection rule.** For every statistic and fold we compute the exposure-weighted squared error `Σ wᵢ (yᵢ − ŷᵢ)² / Σ wᵢ` over players with ≥500 minutes in the target season (availability is judged separately, on established players, section 5), pool the five folds, and select the candidate with the lowest pooled error. Pooling five seasons guards against a method that wins one year by luck. The winner for each statistic and its gain over the baseline:

{ctx.fig("fig_backtest_bars", "Out-of-sample change in squared error relative to the shrunken baseline, by statistic and method. Bars to the left of zero are improvements.")}

{md_table(comp, {"baseline RMSE": "{:.3f}", "winning RMSE": "{:.3f}", "change in squared error": "{:+.1f}%"})}

**What this says.**

* **Minutes ({sel.loc["mpg", "vs_baseline_pct"]:.0f}%), shot volume (2-point attempts {sel.loc["fga2", "vs_baseline_pct"]:.0f}%), availability ({sel.loc["avail", "vs_baseline_pct"]:.0f}%) and free-throw attempts ({sel.loc["fta", "vs_baseline_pct"]:.0f}%) improve most.** These are *role* variables: young players' minutes rise, veterans' fall, and a tree model can find that structure. LightGBM wins for minutes and availability, where non-linearities matter; Ridge wins for most of the rest.
* **Rebounds, steals and free-throw percentage stay with the baseline.** Their year-to-year signal is already captured by the shrunken average, and the extra model only adds noise. That is a finding rather than a failure: the simpler model is kept wherever it wins.
* **Percentages move least** because at season scale they are mostly luck (see the shrinkage figure).

## 4.3 What the projections look like in fantasy terms

After the rate models run, the rates are multiplied back into per-game statistics. As a sanity benchmark we compare with the "naive" forecaster who assumes next year equals last year, on the same player-seasons (MAE = average absolute miss in per-game units; R² = share of the spread between players that the projection explains):

{md_table(acc_t, {"player-seasons": "{:,.0f}", "average value": "{:.2f}", "MAE (ours)": "{:.3f}", "MAE (same as last year)": "{:.3f}", "MAE improvement": "{:+.1f}%", "R² (ours)": "{:.2f}", "R² (last year)": "{:.2f}"})}

The projection beats "same as last year" in every category. The gain is largest where last season's number is mostly noise: free-throw percentage (R² rises from {acc.set_index("stat").loc["ft_pct", "r2_last_season"]:.2f} to {acc.set_index("stat").loc["ft_pct", "r2_model"]:.2f}) and steals ({acc.set_index("stat").loc["stl", "r2_last_season"]:.2f} to {acc.set_index("stat").loc["stl", "r2_model"]:.2f}). Steals remain the hardest category to predict.

{ctx.fig("fig_pred_vs_actual", "Projected versus actual per-game values for out-of-sample player-seasons (2021-2025). The grey-orange diagonal is a perfect projection.")}

## 4.4 The test that matters for a draft: ranking

A draft only needs to *order* players. For each backtest season we z-score the nine categories both for the projection and for what really happened (section 8 defines the z-score), then measure how well the projected ordering agrees with the actual ordering using **Spearman's rank correlation** ρ (the Pearson correlation of the two rank vectors). This is compared with ordering players by last season's actual production:

{ctx.fig("fig_ranking", "Spearman rank correlation between predicted and actual 9-category value in each backtest season, for our projection and for last season's production.")}

{md_table(rk.rename(columns={"year": "season start", "n": "players", "spearman_model": "ρ (ours)", "spearman_last_season": "ρ (last year's stats)", "top_overlap_model": "top-50 hit rate (ours)", "top_overlap_last_season": "top-50 hit rate (last year)", "k": "k"}), {"season start": "{:.0f}", "players": "{:.0f}", "ρ (ours)": "{:.3f}", "ρ (last year's stats)": "{:.3f}", "top-50 hit rate (ours)": "{:.0%}", "top-50 hit rate (last year)": "{:.0%}", "k": "{:.0f}"})}

The projection ranks better than last year's stats in **all {len(rk)} seasons** (average ρ = **{S["spearman_model"]:.3f}** against **{S["spearman_last"]:.3f}**), a modest but consistent edge. The edge in *picking out the top 50* is less uniform (it matched or beat last year's stats in {int((rk["top_overlap_model"] >= rk["top_overlap_last_season"]).sum())} of {len(rk)} seasons), which is why the project does not oversell the projections: ranking is hard because injuries and breakouts decide a large share of the variance in any fantasy season. One caution on the 2025-26 fold (ρ = {rk.iloc[-1]["spearman_model"]:.2f}, the lowest): it contained an unusually large number of long injuries and the earliest versions of our availability model could not see them (section 5).

## 4.5 Residual diagnostics: where is the model wrong?

A **residual** is *actual − projected* for an out-of-sample player-season. Residual plots answer three questions that a single accuracy number cannot: is the model **biased** (consistently high or low in some region), is its **error size** steady across players (homoscedasticity), and are its **misses related to each other**?

{ctx.fig("fig_resid_fitted", "Residuals versus projected level for six quantities. A trend line hugging zero means no bias at that level; a slope would mean the model mishandles stars or role players.")}

Dots scatter evenly above and below zero and the decile averages stay close to the zero line through the middle of each panel, so stars are not systematically over-projected and role players are not systematically under-projected. Two mild departures are visible at the extremes. **Players projected for few minutes (below about 20 per game) were under-projected by one to two minutes on average**; this is partly a selection effect, because the evaluation only includes players who actually logged at least 500 minutes, and a player who earns those minutes tends to have been given more of a role than the projection expected. **Players projected for the most three-pointers are slightly over-projected** (the line slopes down to about −0.2 per game at the top), the usual regression to the mean for extreme shooting volume. Neither departure is large relative to the spread of the dots.

The spread of the misses also changes with the projected level (heteroscedasticity): the Spearman correlation between a player's projected level and the absolute size of his miss is reported in the table below (largest for {lab[het.idxmax()]}, ρ = {het.max():+.2f}). That is why the simulation of section 9 sizes talent uncertainty *relative* to a player's level for counting statistics rather than using one flat number.

{md_table(rt, {"n": "{:,.0f}", "bias (actual − projected)": "{:+.3f}", "± SE": "{:.3f}", "SD of miss": "{:.3f}", "MAE": "{:.3f}", "skew": "{:+.2f}", "share beyond 2 SD": "{:.1%}", "ρ (error vs level)": "{:+.2f}"})}

* **Overall bias.** {"Categories whose average miss is more than 2.5 standard errors from zero: " + ", ".join(biased) + "." if biased else "No category's average miss is more than 2.5 standard errors from zero, so the model is, on average, unbiased."} (Bias is in per-game units; FG% and FT% are in percentage points.)
* **Shape.** A normal distribution would have {pct(2 * (1 - norm.cdf(2)))} of misses beyond two standard deviations; the table shows the observed shares for most categories, and the Q-Q plots below show that the *upper* tail of points and assists is heavier than Gaussian (breakout seasons and sudden role increases), while minutes are close to normal. The extremes are role changes and breakouts rather than smooth noise.

{ctx.fig("fig_resid_dist", "Standardised residuals for the nine categories with a normal curve overlaid.")}
{ctx.fig("fig_resid_qq", "Normal Q-Q plots for points, assists and minutes: close to the diagonal in the middle; the upper tail of points and assists is heavier than normal.")}
{ctx.fig("fig_resid_group", "Average miss by age and by projected minutes (95% intervals). Departures from zero that are large and consistent would be fixable biases, for example a too-gentle aging curve.")}
{ctx.fig("fig_resid_season", "Stability of the average miss across the five backtest seasons; the colour scale is symmetric around zero.")}
{ctx.fig("fig_resid_corr", "Correlation between the misses of different categories. The strong block shows that most of what the model gets wrong about a player is shared across categories.")}

**Misses move together.** The correlation between the points miss and the minutes miss is **{res["mpg_corr"]:.2f}**: most of what the model gets wrong about a player is his *role* (minutes), which then spreads across every counting stat. This is the structure the simulation's "talent draw" reproduces (section 9): the projection's uncertainty is not nine independent errors but largely one shared error. It also explains why availability and minutes are the first places to look for improvements, which is what sections 5 and 6 do.
"""
