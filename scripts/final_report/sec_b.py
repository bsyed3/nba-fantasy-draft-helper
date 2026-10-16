"""Sections 5-8: availability, team context, rookies and expert inputs, the value score and the Giannis dilemma."""
from __future__ import annotations

import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scripts.final_report.common import *  # noqa: F403
from src.models import projection as P


def _read(name: str) -> pd.DataFrame:
    return pd.read_parquet(EXP / f"{name}.parquet")


# ---------------------------------------------------------------------------------------------------------------------
# availability figures
# ---------------------------------------------------------------------------------------------------------------------
def fig_gp_dist(ctx) -> dict:
    d = ctx.apg[(ctx.apg.year >= 2021) & (ctx.apg.mpg >= 25) & (ctx.apg.gp >= 1)]
    mean, med = d.gp.mean(), d.gp.median()
    fig, ax = F.new(8.4, 4.3)
    ax.hist(d.gp, bins=np.arange(0, 86, 3), color=C["blue"], edgecolor=C["surface"], linewidth=0.7)
    ax.axvline(mean, color=C["orange"], lw=2.4); ax.axvline(med, color=C["aqua"], lw=2.4)
    ymax = ax.get_ylim()[1]
    ax.text(mean - 1, ymax * 0.93, f"mean {mean:.0f}", ha="right", color=C["orange"], fontsize=9.5, fontweight="bold")
    ax.text(med + 1, ymax * 0.93, f"median {med:.0f}", ha="left", color=C["aqua"], fontsize=9.5, fontweight="bold")
    ax.set_xlabel("games played in a season"); ax.set_ylabel("player-seasons"); ax.grid(axis="x", visible=False)
    F.headline(fig, "Games played are left-skewed: a few long injuries drag the mean below what a typical starter plays",
               f"Rotation players (≥25 minutes per game), seasons 2021-22 to 2025-26, n = {len(d):,}; {(d.gp >= 70).mean():.0%} played 70+ games, {(d.gp < 50).mean():.0%} played fewer than 50")
    F.save(fig, "fig_gp_dist")
    return dict(mean=float(mean), med=float(med), n=len(d), ge70=float((d.gp >= 70).mean()), lt50=float((d.gp < 50).mean()))


def fig_naive_vs_after(ctx) -> pd.DataFrame:
    st = ctx.st
    x = st[(st.year == 2025) & (st.smiss >= 20) & (st.gp >= 12) & (st["min"] / st.gp >= 20)].copy()
    x["name"] = x.player_id.map(ctx.names)
    x["naive"], x["after_first"] = x.gp / x.sg, x.gp / x.sgo
    pick = x[x["name"].isin(["Jayson Tatum", "Dejounte Murray", "Max Strus"])].sort_values("smiss", ascending=False)
    fig, ax = F.new(8.0, 4.0)
    xx = np.arange(len(pick)); w = 0.34
    ax.bar(xx - w / 2, pick.naive * 100, w, color=C["orange"], label="games ÷ team games in the whole season (naive)")
    ax.bar(xx + w / 2, pick.after_first * 100, w, color=C["blue"], label="games ÷ team games since his first appearance (used)")
    for xi, r in zip(xx, pick.itertuples()):
        ax.text(xi - w / 2, r.naive * 100 + 1.5, f"{r.naive:.0%}", ha="center", fontsize=9, color=C["ink2"])
        ax.text(xi + w / 2, r.after_first * 100 + 1.5, f"{r.after_first:.0%}", ha="center", fontsize=9, color=C["ink2"])
    ax.set_xticks(xx, [f"{r.name}\n{int(r.gp)} GP after missing {int(r.smiss)} team games" for r in pick.itertuples()], fontsize=8.5)
    ax.set_ylabel("availability (%)"); ax.set_ylim(0, 105); ax.grid(axis="x", visible=False)
    F.headline(fig, "A player who is out at tip-off is not a random injury: measuring from his first game separates the two",
               "2025-26 examples of players who missed the first two months and then played. The naive ratio calls them 15-20% available; they were 58-80% available once healthy")
    F.legend_top(fig, *ax.get_legend_handles_labels(), ncol=1)
    F.save(fig, "fig_naive_vs_after")
    return pick


def fig_avail_groups(ctx) -> pd.DataFrame:
    base, nc = _read("projfix_base_avail"), _read("projfix_noctx_avail_avail")
    def groups(e):
        return {"all established players": e, "moved teams": e[e.moved == 1], "age 33+": e[e.age >= 33], "age 30-32": e[(e.age >= 30) & (e.age < 33)],
                "stayed, under 30": e[(e.moved == 0) & (e.age < 30)]}
    rows = []
    for k, g in groups(base).items():
        gn = groups(nc)[k]
        rows.append(dict(group=k, n=len(g), model_mae=(g.act_avail - g.pred_avail).abs().mean() * 82, model_bias=(g.act_avail - g.pred_avail).mean() * 82,
                         base_mae=(g.act_avail - g.base_avail).abs().mean() * 82, base_bias=(g.act_avail - g.base_avail).mean() * 82,
                         noctx_mae=(gn.act_avail - gn.pred_avail).abs().mean() * 82, noctx_bias=(gn.act_avail - gn.pred_avail).mean() * 82))
    t = pd.DataFrame(rows)
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.4))
    x = np.arange(len(t)); w = 0.26
    for ax, (cols, ttl, yl) in zip(axes, [(("model_mae", "base_mae", "noctx_mae"), "Typical miss (mean absolute error), games per season", "games"),
                                           (("model_bias", "base_bias", "noctx_bias"), "Average miss (actual − projected), games per season", "games; below 0 = over-projected")]):
        for i, (c, colr, lab_) in enumerate(zip(cols, (C["blue"], C["orange"], C["aqua"]), ("our model", "recency baseline alone", "model without team context"))):
            ax.bar(x + (i - 1) * w, t[c], w, color=colr, label=lab_)
        ax.axhline(0, color=C["ink2"], lw=1)
        ax.set_xticks(x, [f"{g}\n(n={n})" for g, n in zip(t.group, t.n)], fontsize=7.8); ax.set_title(ttl, fontsize=10, loc="left"); ax.set_ylabel(yl)
        ax.grid(axis="x", visible=False)
    F.headline(fig, "The availability model has a similar typical miss to a recency average but a far smaller bias; team context roughly halves the bias for movers",
               "Out-of-sample games played (walk-forward folds 2021-2025), established players (≥500 minutes the year before) who appeared in a game, in games per 82")
    F.legend_top(fig, *axes[0].get_legend_handles_labels())
    F.save(fig, "fig_avail_groups")
    return t


def fig_returners(ctx) -> dict:
    r = _read("projfix_returners")
    mins = r[r.act_mpg.notna() & (r.min_t >= 500)].copy()
    mins["name"] = mins.player_id.map(ctx.names)
    stars = mins[mins.last_mpg >= 28]
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.6), gridspec_kw={"width_ratios": [1.05, 1]})
    ax = axes[0]
    ax.scatter(mins.last_mpg, mins.act_mpg, s=26, color=C["blue"], label="returner (≥500 min after the missed year)")
    ax.scatter(stars.last_mpg, stars.act_mpg, s=40, color=C["orange"], zorder=3, label="former rotation star (≥28 mpg before)")
    ax.plot([5, 40], [5, 40], color=C["ink2"], lw=1.2, ls=(0, (4, 3)))
    drops = (stars.act_mpg - stars.last_mpg).sort_values().head(4).index
    for i in drops:
        ax.annotate(f"{stars.loc[i, 'name']} ({SEASON(stars.loc[i, 'year'])})", (stars.loc[i, "last_mpg"], stars.loc[i, "act_mpg"]), xytext=(-8, -13), textcoords="offset points", ha="right", fontsize=7.6, color=C["ink2"])
    ax.set_xlim(8, 40); ax.set_ylim(8, 40); ax.set_aspect("equal")
    ax.set_xlabel("minutes per game the last time he played"); ax.set_ylabel("minutes per game on his return"); ax.set_title("Returners come back with fewer minutes", fontsize=10, loc="left")
    ax = axes[1]
    labs, cols = ["recency baseline", "our model (as shipped)", "blend 25% baseline", "blend 50% baseline"], [C["orange"], C["blue"], C["aqua"], C["yellow"]]
    def bias(g):
        return [(g.act_mpg - g.base_mpg).mean(), (g.act_mpg - g.model_mpg).mean(), (g.act_mpg - (.25 * g.base_mpg + .75 * g.model_mpg)).mean(), (g.act_mpg - (.5 * g.base_mpg + .5 * g.model_mpg)).mean()]
    allb, starb = bias(mins), bias(stars)
    x = np.arange(2); w = 0.2
    for i, (lab_, colr) in enumerate(zip(labs, cols)):
        ax.bar(x + (i - 1.5) * w, [allb[i], starb[i]], w, color=colr, label=lab_)
        for xi, v in zip(x, [allb[i], starb[i]]):
            ax.text(xi + (i - 1.5) * w, v + (0.15 if v >= 0 else -0.15), f"{v:+.1f}", ha="center", va="bottom" if v >= 0 else "top", fontsize=7.6, color=C["ink2"])
    ax.axhline(0, color=C["ink2"], lw=1)
    ax.set_xticks(x, [f"all returners\n(n={len(mins)})", f"former stars\n(n={len(stars)})"]); ax.set_ylim(-4.6, 2.6)
    ax.set_ylabel("actual − projected minutes per game"); ax.set_title("Bias of each minutes projection", fontsize=10, loc="left"); ax.grid(axis="x", visible=False)
    F.headline(fig, "Players returning from a missed season come back with fewer minutes; the model's cut is about right, a recency average is far too high",
               "Players with no season the year before who then played ≥500 minutes, folds 2021-2025")
    F.legend_top(fig, *axes[1].get_legend_handles_labels(), ncol=4)
    F.save(fig, "fig_returners")
    g = r[r.act_avail.notna() & (r.w_avail > 0)]
    gs = g[g.last_mpg >= 28]
    return dict(n_min=len(mins), n_star=len(stars), last=mins.last_mpg.mean(), act=mins.act_mpg.mean(), last_s=stars.last_mpg.mean(), act_s=stars.act_mpg.mean(),
                bias=allb, bias_s=starb, n_g=len(g), n_gs=len(gs),
                g_model=(g.act_avail - g.model_avail).abs().mean(), g_base=(g.act_avail - g.base_avail).abs().mean(),
                g_model_b=(g.act_avail - g.model_avail).mean(), g_base_b=(g.act_avail - g.base_avail).mean(), gs_model_b=(gs.act_avail - gs.model_avail).mean())


def fig_context_ablation(ctx) -> dict:
    base = _read("projfix_base_summary"); nc = _read("projfix_noctx_all_summary")
    sel = lambda d: d[d.selected].set_index("stat")["wmse"]
    pct_ = (sel(nc) / sel(base) - 1) * 100
    order = pct_.sort_values(ascending=False).index
    fig, ax = F.new(8.2, 4.8)
    y = np.arange(len(order))
    ax.barh(y, pct_[order], color=[C["red"] if v > 0 else C["blue"] for v in pct_[order]], height=0.64)
    for yi, v in zip(y, pct_[order]):
        ax.text(max(v, 0) + 0.15, yi, f"{v:+.1f}%", va="center", ha="left", fontsize=8.5, color=C["ink2"])      # labels always to the right of the axis line
    ax.axvline(0, color=C["ink2"], lw=1)
    ax.set_yticks(y, [STAT_LABEL[s] for s in order]); ax.invert_yaxis(); ax.grid(axis="y", visible=False)
    ax.set_xlabel("change in out-of-sample squared error when team-context features are removed (red = worse without)")
    F.headline(fig, "Removing team-context features makes shot volume, turnover, assist and minutes projections worse",
               "Ablation in the walk-forward backtest (folds 2021-2025): same models, same folds, context columns dropped for every statistic")
    F.save(fig, "fig_context_ablation")
    return dict(mean=float(pct_.mean()), pct=pct_)


def fig_third_year(ctx) -> pd.DataFrame:
    apg = ctx.apg
    rows = []
    for v, lab_ in (("base", "as shipped"), ("expflags", "with season-of-career flags")):
        o = _read(f"projfix_{v}_oos").merge(apg[["player_id", "year", "pts", "mpg"]].rename(columns={"pts": "act_pts", "mpg": "act_mpg"}), on=["player_id", "year"], how="left")
        for g_lab, g in (("2nd year", o[o.exp == 1]), ("3rd year", o[o.exp == 2]), ("4th year", o[o.exp == 3]), ("5+ years", o[o.exp >= 4])):
            rows.append(dict(variant=lab_, group=g_lab, n=len(g), mpg=(g.act_mpg - g.mpg).mean(), ppg=(g.act_pts - g.pts).mean()))
    t = pd.DataFrame(rows)
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.0), sharey=False)
    groups = ["2nd year", "3rd year", "4th year", "5+ years"]
    x = np.arange(4); w = 0.34
    for ax, (col, ttl) in zip(axes, [("mpg", "Minutes per game"), ("ppg", "Points per game")]):
        for i, (v, colr) in enumerate(zip(("as shipped", "with season-of-career flags"), (C["blue"], C["aqua"]))):
            vals = [t[(t.variant == v) & (t.group == g)][col].iloc[0] for g in groups]
            ax.bar(x + (i - 0.5) * w, vals, w, color=colr, label=v)
            for xi, val in zip(x, vals):
                ax.text(xi + (i - 0.5) * w, val + (0.02 if val >= 0 else -0.02), f"{val:+.2f}", ha="center", va="bottom" if val >= 0 else "top", fontsize=7.8, color=C["ink2"])
        ax.axhline(0, color=C["ink2"], lw=1)
        ax.set_xticks(x, [f"{g}\n(n={int(t[(t.group == g)].n.iloc[0])})" for g in groups], fontsize=8.5); ax.set_title(ttl + " (actual − projected)", fontsize=10, loc="left"); ax.grid(axis="x", visible=False)
    F.headline(fig, "Third- and fourth-season players are slightly under-projected; season-of-career flags trim but do not remove it",
               "Average miss by years of experience in the walk-forward backtest (folds 2021-2025); positive = the model was too low")
    F.legend_top(fig, *axes[0].get_legend_handles_labels())
    F.save(fig, "fig_third_year")
    return t


# ---------------------------------------------------------------------------------------------------------------------
# rookies, value, Giannis figures
# ---------------------------------------------------------------------------------------------------------------------
def fig_rookie_tiers(ctx) -> pd.DataFrame:
    means, cvs = P.rookie_profiles(ctx.st)
    pg = P.to_per_game(means[P.ALL_STATS].reset_index(drop=True))
    tab = pd.DataFrame({"draft slot": means.index, "rookies": means["n"].astype(int).values, "mpg": pg["mpg"].values, "availability": pg["avail"].values,
                        "PTS": pg["pts"].values, "REB": pg["reb"].values, "AST": pg["ast"].values})
    fig, axes = plt.subplots(1, 3, figsize=(10.4, 3.9))
    for ax, (col, ttl) in zip(axes, [("mpg", "Minutes per game"), ("PTS", "Points per game"), ("availability", "Share of games played")]):
        ax.bar(tab["draft slot"], tab[col], color=C["blue"], width=0.62)
        for xi, v in enumerate(tab[col]):
            ax.text(xi, v * 1.01, f"{v:.1f}" if col != "availability" else f"{v:.0%}", ha="center", fontsize=8.5, color=C["ink2"])
        ax.set_title(ttl, fontsize=10, loc="left"); ax.tick_params(axis="x", labelsize=8); ax.grid(axis="x", visible=False)
        if col == "availability":
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    F.headline(fig, "A rookie's draft slot says a lot about his minutes, and about how uncertain his season is", "Average first-NBA-season profile by draft slot, 2015-16 to 2025-26 (≥200 minutes)")
    F.save(fig, "fig_rookie_tiers")
    return tab


def fig_top25(ctx) -> None:
    top = ctx.proj.head(25)
    zc = [f"z_{c}" for c in CATEGORIES]
    fig, ax = F.new(9, 8.2)
    mat = top[zc].to_numpy()
    im = ax.imshow(mat, cmap=F.DIV_CMAP, vmin=-4, vmax=4, aspect="auto")
    ax.set_xticks(range(9), [CAT_LABELS[c] for c in CATEGORIES]); ax.xaxis.tick_top()
    ax.set_yticks(range(len(top)), [f"{r}. {n}" for r, n in zip(top["rank"], top["name"])]); ax.grid(False)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = mat[i, j]
            ax.text(j, i, f"{v:+.1f}", ha="center", va="center", fontsize=8, color="white" if abs(v) > 2.6 else C["ink"])
    cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02); cb.set_label("z-score (standard deviations above a typical drafted player)")
    for s in ax.spines.values():
        s.set_visible(False)
    gz_ft = float(top.loc[top["name"].str.contains("Antetokounmpo"), "z_ft_pct"].iloc[0])
    raw = f"{gz_ft:.1f}".replace("-", "−")
    F.headline(fig, "Where each top-25 player's value comes from",
               f"Projected 2026-27 category z-scores (blue helps, red hurts; turnovers flipped so blue is fewer). Giannis's FT% shows the raw {raw} even though the total floors it at −3")
    F.save(fig, "fig_top25")


GID = 203507


def giannis_data(ctx) -> dict:
    cols = ["player_id", "avail", "mpg", "pts", "tpm", "reb", "ast", "stl", "blk", "tov", "fg_pct", "ft_pct", "fgm", "fga", "ftm", "fta"]
    rows = []
    for y in range(2021, 2026):
        df = ctx.apg[(ctx.apg.year == y) & (ctx.apg["min"] >= 1000)][cols].reset_index(drop=True)
        i = df.index[df.player_id == GID][0]
        z0, z1 = z_scores(df, floor=None), z_scores(df)
        rows.append(dict(season=SEASON(y), gp=int(ctx.apg[(ctx.apg.year == y) & (ctx.apg.player_id == GID)].gp.iloc[0]),
                         plain=int(z0.z_total.rank(ascending=False)[i]), floored=int(z1.z_total.rank(ascending=False)[i]), ft_z=float(z0.loc[i, "z_ft_pct"]),
                         pts=float(df.loc[i, "pts"]), mpg=float(df.loc[i, "mpg"])))
    return dict(seasons=pd.DataFrame(rows))


def fig_giannis_z(ctx) -> dict:
    p = ctx.proj
    g = p[p.player_id == GID].iloc[0]
    raw = np.array([g[f"z_{c}"] for c in CATEGORIES])
    fl = np.clip(raw, -3, None)
    pool = p.head(200)
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.4), gridspec_kw={"width_ratios": [1.2, 1]})
    ax = axes[0]
    x = np.arange(9); w = 0.38
    ax.bar(x - w / 2, raw, w, color=C["orange"], label="raw z-score")
    ax.bar(x + w / 2, fl, w, color=C["blue"], label="counted in the total (floored at −3)")
    for xi, v in zip(x, raw):
        ax.text(xi - w / 2, v + (0.15 if v >= 0 else -0.15), f"{v:+.1f}", ha="center", va="bottom" if v >= 0 else "top", fontsize=8, color=C["ink2"])
    ax.axhline(0, color=C["ink2"], lw=1); ax.axhline(-3, color=C["red"], lw=1, ls=(0, (4, 3)))
    ax.set_xticks(x, [CAT_LABELS[c] for c in CATEGORIES]); ax.set_ylabel("category z-score"); ax.grid(axis="x", visible=False)
    ax.set_title("Giannis's projected value by category", fontsize=10, loc="left")
    ax = axes[1]
    ax.hist(pool["z_ft_pct"], bins=np.arange(-7, 5.5, 0.5), color=C["blue"], edgecolor=C["surface"], linewidth=0.7)
    ax.axvline(g["z_ft_pct"], color=C["orange"], lw=2.4)
    ax.text(g["z_ft_pct"] + 0.15, ax.get_ylim()[1] * 0.85, f"Giannis\n{g['z_ft_pct']:+.1f}", color=C["orange"], fontsize=9, fontweight="bold")
    ax.set_xlabel("FT% impact z-score"); ax.set_ylabel("players (board top 200)"); ax.grid(axis="x", visible=False)
    ax.set_title("He is a statistical outlier at the line", fontsize=10, loc="left")
    F.headline(fig, "One enormous weakness: Giannis's free-throw impact is more than 5 standard deviations below a drafted player",
               "A plain sum of nine z-scores lets that one category cancel several of his strengths; weekly head-to-head only counts it as one lost category")
    F.legend_top(fig, *axes[0].get_legend_handles_labels())
    F.save(fig, "fig_giannis_z")
    return dict(ft_z=float(g["z_ft_pct"]), floor_add=float(g["z_floor_adj"]), z_total=float(g["z_total"]), rank=int(g["rank"]))


def fig_giannis_seasons(ctx, gd: dict) -> None:
    t = gd["seasons"]
    fig, ax = F.new(8.4, 4.4)
    x = np.arange(len(t))
    for xi, r in zip(x, t.itertuples()):
        ax.plot([xi, xi], [r.plain, r.floored], color=C["axis"], lw=2)
    ax.scatter(x, t.plain, s=70, color=C["orange"], zorder=3, label="plain sum of nine z-scores")
    ax.scatter(x, t.floored, s=70, color=C["blue"], zorder=3, label="with the −3 floor")
    for xi, r in zip(x, t.itertuples()):
        ax.text(xi + 0.08, r.plain, f"{r.plain}", color=C["orange"], fontsize=9, va="center")
        ax.text(xi + 0.08, r.floored, f"{r.floored}", color=C["blue"], fontsize=9, va="center")
    ax.set_xticks(x, [f"{r.season}\n{r.gp} GP" for r in t.itertuples()]); ax.invert_yaxis(); ax.set_ylabel("his rank that season (lower is better)")
    ax.grid(axis="x", visible=False)
    F.headline(fig, "In the three full seasons before his rest year our own scoring rule ranked him 133rd, 28th and 59th, which cannot be right",
               "Giannis's rank by season using only what he really did (lower is better); the 2025-26 point is the 36-game rest season")
    F.legend_top(fig, *ax.get_legend_handles_labels())
    F.save(fig, "fig_giannis_seasons")


def fig_giannis_timeline(ctx) -> None:
    # board ranks recorded at each stage of the investigation (project log, docs/HANDOFF.md)
    stages = [("first look\n(age 32, 36-game\n2025-26 anchors him)", 93), ("after the team-context\nmodel was added\n(he moved to Miami)", 103), ("2025-26 treated\nas a rest year\n(ignored)", 59),
              ("team-context\npenalty switched\noff for him", 29), ("−3σ floor on each\ncategory in\nthe total", 6)]
    fig, ax = F.new(9.6, 4.6)
    x = np.arange(len(stages)); y = [s[1] for s in stages]
    ax.plot(x, y, color=C["blue"], lw=2.6, marker="o", ms=9, zorder=3)
    for xi, yi in zip(x, y):
        ax.text(xi, yi - 6, f"#{yi}", ha="center", fontsize=10.5, fontweight="bold", color=C["ink"])
    ax.axhline(6, color=C["orange"], lw=1.4, ls=(0, (4, 3))); ax.text(-0.3, 10, "market order: 6th", ha="left", color=C["orange"], fontsize=9, fontweight="bold")
    ax.set_xticks(x, [s[0] for s in stages], fontsize=8.4); ax.invert_yaxis(); ax.set_ylim(112, -6)
    ax.set_ylabel("Giannis's board rank (lower is better)"); ax.grid(axis="x", visible=False)
    F.headline(fig, "How Giannis went from 93rd to 6th: three separate modelling decisions, each with its own justification",
               "Board rank after each change (documented in the project log); the market's blended order places him 6th")
    F.save(fig, "fig_giannis_timeline")


def fig_giannis_counterfactual(ctx) -> None:
    # counterfactual ranks computed while he sat at rank 103 (project log): what would have to change to put him where the market had him?
    items = [("as projected (board rank 103)", 103), ("80% of games instead of 56 GP\n(82% of team games)", 57), ("... and his 2023-24 minutes (34.2)", 25),
             ("... and his actual 2023-24 line\ninstead of the model's per-minute rates", 6), ("... and neutral free-throw %", 3)]
    fig, ax = F.new(8.6, 4.3)
    y = np.arange(len(items))
    ax.barh(y, [i[1] for i in items], color=[C["orange"]] + [C["blue"]] * (len(items) - 1), height=0.62)
    for yi, (_, v) in zip(y, items):
        ax.text(v + 1.5, yi, f"#{v}", va="center", fontsize=9.5, color=C["ink2"])
    ax.set_yticks(y, [i[0] for i in items], fontsize=8.6); ax.invert_yaxis(); ax.grid(axis="y", visible=False)
    ax.set_xlabel("his board rank under each cumulative counterfactual (shorter bar = better)")
    F.headline(fig, "Counterfactuals pinpointed the causes: games and minutes anchored on a rest year, then a single category",
               "Cumulative what-ifs run when he stood at rank 103; each row keeps the earlier changes")
    F.save(fig, "fig_giannis_counterfactual")


def fig_punt(ctx) -> None:
    # punt experiment (project log): win probability of Giannis when later picks follow the board vs a punt build, draft slots 5, 6, 8
    slots = ["slot 5", "slot 6", "slot 8"]
    balanced, best = [0.532, 0.531, 0.527], [0.541, 0.542, 0.532]
    punt1 = [0.517, 0.515, 0.517]
    fig, ax = F.new(8.4, 4.2)
    x = np.arange(3); w = 0.2
    ax.bar(x - 1.5 * w, np.array(balanced) * 100, w, color=C["blue"], label="Giannis, later picks follow the board (balanced)")
    ax.bar(x - 0.5 * w, np.array(punt1) * 100, w, color=C["orange"], label="Giannis, punt free throws (later picks ignore FT%)")
    ax.bar(x + 0.5 * w, np.array([0.415] * 3) * 100, w, color=C["red"], label="punt FT% + 3PM + AST + STL (reported range 39-44%)")
    ax.bar(x + 1.5 * w, np.array(best) * 100, w, color=C["aqua"], label="best alternative first pick at that slot")
    for xi, vals in zip(x, zip(balanced, punt1, [0.415] * 3, best)):
        for k, v in enumerate(vals):
            ax.text(xi + (k - 1.5) * w, v * 100 + 0.5, f"{v * 100:.1f}" if k != 2 else "39-44", ha="center", fontsize=7.8, color=C["ink2"])
    ax.set_xticks(x, slots); ax.set_ylim(30, 60); ax.set_ylabel("simulated weekly matchup win probability (%)"); ax.grid(axis="x", visible=False)
    F.headline(fig, "Building around his weakness did not help: a balanced roster that includes good free-throw shooters wins more weeks",
               "Punt experiment from the project log (6 roll-outs per bar): the simulation already treats a lost category as lost, so extra punting only loses more")
    F.legend_top(fig, *ax.get_legend_handles_labels(), ncol=2)
    F.save(fig, "fig_punt")


# ---------------------------------------------------------------------------------------------------------------------
# text
# ---------------------------------------------------------------------------------------------------------------------
def build(ctx) -> str:
    S, proj, st = ctx.S, ctx.proj, ctx.st
    gp = fig_gp_dist(ctx)
    naive = fig_naive_vs_after(ctx)
    grp = fig_avail_groups(ctx)
    ret = fig_returners(ctx)
    abl = fig_context_ablation(ctx)
    ty = fig_third_year(ctx)
    tiers = fig_rookie_tiers(ctx)
    fig_top25(ctx)
    gd = giannis_data(ctx)
    gz = fig_giannis_z(ctx)
    fig_giannis_seasons(ctx, gd)
    fig_giannis_timeline(ctx)
    fig_giannis_counterfactual(ctx)
    fig_punt(ctx)
    S["giannis"] = gz
    S["abl"] = abl
    S["returners"] = ret
    S["gp"] = gp

    gq = json.load(open(DATA / "gp_quantiles.json"))
    coefs = {float(q): (v["intercept"], v["slope"]) for q, v in gq.items()}
    gmap = P.apply_gp_quantiles(np.array([0.5, 0.6, 0.7, 0.8, 0.9]), coefs).round(0)
    gmap.insert(0, "projected mean availability", ["50%", "60%", "70%", "80%", "90%"])
    gmap.columns = ["projected mean availability", "P25 games", "median games", "P75 games"]
    ka = pd.read_csv(ROOT / "data" / "external" / "known_absences.csv")
    km = pd.read_csv(ROOT / "data" / "external" / "known_absences_manual.csv")
    kt = pd.concat([ka[["name", "team", "games_out"]].assign(source="ESPN 'Out' list"), km[["name", "games_out"]].assign(team="", source="manual (owner)")], ignore_index=True).sort_values("games_out", ascending=False)
    kt.columns = ["player", "team", "games out", "source"]
    ntab = naive[["name", "gp", "sg", "sgo", "smiss", "naive", "after_first"]].copy()
    ntab.columns = ["player", "games played", "team games", "team games since his first game", "team games before it", "naive availability", "availability once playing"]
    gt = grp.rename(columns={"group": "group", "n": "n", "model_mae": "model MAE", "model_bias": "model bias", "base_mae": "baseline MAE", "base_bias": "baseline bias", "noctx_mae": "no-context MAE", "noctx_bias": "no-context bias"})
    yr = pd.read_csv(ROOT / "data" / "external" / "rookie_projections_yahoo.csv")
    yr_t = yr[["pick", "name", "team", "pts", "reb", "ast", "mpg"]].rename(columns={"pick": "pick", "pts": "PPG", "reb": "RPG", "ast": "APG", "mpg": "MPG (blank = model's)"})
    cmp_ = A.rookie_model_comparison(st)
    cmp_t = cmp_.rename(columns={"mpg": "MPG", "pts": "PTS", "reb": "REB", "ast": "AST", "stl": "STL", "blk": "BLK"}).reset_index().rename(columns={"index": "method"})
    cmp_t["method"] = cmp_t["method"].map({"tier": "draft-slot bucket means (used)", "curve": "smooth curve in log(pick)"})
    tyt = ty.copy()
    tyt["variant"] = tyt["variant"].str.replace("as shipped", "shipped")
    gs = gd["seasons"].rename(columns={"season": "season", "gp": "GP", "plain": "rank, plain sum", "floored": "rank, floored", "ft_z": "FT% z-score", "pts": "PPG", "mpg": "MPG"})
    ib, pz = proj[proj.name == "Brandon Ingram"].iloc[0], proj[proj.name == "Kristaps Porziņģis"].iloc[0]
    ha = proj[proj.name == "Tyrese Haliburton"].iloc[0]
    ab = {s: ret["bias"][i] for i, s in enumerate(["baseline", "model", "b25", "b50"])}
    abs_ = {s: ret["bias_s"][i] for i, s in enumerate(["baseline", "model", "b25", "b50"])}
    mv_row = grp.set_index("group").loc["moved teams"]

    return f"""
# 5. Phase 1, step 3: availability, the biggest source of error

## 5.1 The problem: games played is a different kind of variable

Counting statistics scale with games played, so a projection of 22 points per game is worth very different amounts to a fantasy team depending on whether the player appears in 80 games or 40. Availability turned out to be both the largest source of error and the place where the first version of the model was conceptually wrong. Three observations drove the redesign.

**(1) Games played are left-skewed, so the mean and the median disagree.** For rotation players (at least 25 minutes per game) over 2021-22 to 2025-26, the mean season was **{gp["mean"]:.0f} games** but the median was **{gp["med"]:.0f}**: {gp["ge70"]:.0%} played 70 or more games, while a long tail of season-ending injuries (**{gp["lt50"]:.0%} played fewer than 50**) pulls the mean down.

{ctx.fig("fig_gp_dist", "Distribution of games played for rotation players, with the mean and median marked.")}

Fantasy production is a *sum* over games, so the expected contribution of a player is governed by his **mean** games, and the ranking and the simulation use the mean. But "how many games should I expect a healthy player to play?" has a different answer, the **median**, which is what an owner intuitively means. We therefore also report a typical (median) games figure with a P25-P75 range, obtained by quantile regression of actual games on projected mean availability (display only; it never changes the ranking):

{md_table(gmap)}

Reading the table: a player projected at 70% mean availability would typically play about {int(gmap.iloc[2]["median games"])} games, with a middle half-range of {int(gmap.iloc[2]["P25 games"])}-{int(gmap.iloc[2]["P75 games"])}. (In development we also built a board ranked on the median instead of the mean; its ordering was almost identical, Spearman ρ = 0.996, so nothing was lost by keeping the mean.)

**(2) A player who is out at tip-off is not a random injury.** The first version defined the training target as *games played ÷ 82*. A player who misses the first two months with a known injury and then plays every game looks "50% available", exactly like a player who is hurt randomly all year. Yet the first is almost perfectly healthy once back and, more importantly, **was known to be out before the season started**. At the start of 2025-26 Lillard, Kyrie Irving, Haliburton and Tatum were all publicly known to be injured; the model, unaware, expected normal seasons from them and was then "wrong" in training, which also taught it that availability is lower than it really is. The fix has three parts:

* The target becomes **availability since the first appearance**, `avail = games played ÷ team games from his first game on` (`sgo`), and players who never played in a season are not training targets. This measures how often a player is available *when he is in the league that season*.
* **Known absences become an input**, not something to be predicted (section 5.3): the final availability is `avail × (82 − games_out) ÷ 82`.
* A player's first appearance is detected from the box scores (the first game with minutes), and `smiss` records how many team games came before it.

{ctx.fig("fig_naive_vs_after", "Naive availability versus availability since the first appearance, for three players who missed their team's first two months in 2025-26.")}

{md_table(ntab, {"games played": "{:.0f}", "team games": "{:.0f}", "team games since his first game": "{:.0f}", "team games before it": "{:.0f}", "naive availability": "{:.0%}", "availability once playing": "{:.0%}"})}

**(3) A first diagnosis that turned out to be a selection artifact.** Early on a backtest seemed to show that the model under-projected games for *everyone* by two to three games. That was an artifact of the evaluation set (only players with at least 500 minutes can appear, which silently removes the injury-shortened seasons the model had correctly predicted). A selection-free evaluation, on all established players who appeared, showed that the real misses were confined to **returners** (players with no season the year before) and to players who changed teams, which is where sections 5.4 and 6 look.

## 5.2 How availability was tested

Availability is evaluated on *established* players (at least 500 minutes the previous season) who appeared in at least one game that season, in the same walk-forward folds as every other statistic. Errors are expressed in games per 82. Against a pure recency average, the model's typical miss is slightly smaller in four of the five groups and tied for ages 30-32, but the real difference is in the **bias**: the recency average over-projects games by about 7 for players who changed teams, 5 for players aged 33 and over and 4.5 for ages 30-32, while the model's bias is 2-3 games in those groups.

{ctx.fig("fig_avail_groups", "Out-of-sample games-played error and bias by group: the model, the recency baseline alone, and the model without team-context features.")}

{md_table(gt, {"n": "{:,.0f}", "model MAE": "{:.1f}", "model bias": "{:+.1f}", "baseline MAE": "{:.1f}", "baseline bias": "{:+.1f}", "no-context MAE": "{:.1f}", "no-context bias": "{:+.1f}"})}

The typical miss is about {grp.iloc[0]["model_mae"]:.0f} games for the whole group, which is large; that is the irreducible randomness of injuries, and it is why the simulation of section 9 treats availability as a random variable rather than a constant. The model's bias is small and **negative** (it over-projects games by about {abs(grp.iloc[0]["model_bias"]):.1f} on average and by {abs(mv_row["model_bias"]):.1f} for players who moved teams); without the team-context features the bias for movers grows to {abs(mv_row["noctx_bias"]):.1f} games.

## 5.3 Known absences: using information the model cannot learn

Some information is simply not in the box scores: the ESPN injury report lists who is ruled out and for how long. We read the "Out" designations on 14 October 2026 and convert each expected return date into games missed (using each team's real schedule). Two entries are hand-set by the project owner and take precedence over ESPN (section 7.2). The result is {len(ka)} ESPN entries plus {len(km)} manual ones:

{md_table(kt, {"games out": "{:.0f}"})}

"Day-to-day" listings (mostly pre-season rest) are ignored by design. A player with `games_out` games to miss has his availability scaled by `(82 − games_out) ÷ 82` on top of whatever the model gives him for the rest of the season, which is why Porziņģis (82 games out) is ranked last among the 614 projected players and Ingram (60 out, board rank {int(ib["rank"])}) near the bottom.

## 5.4 Returners: a missed season is information

A *returner* is a player with no season in the previous year but earlier history (Kyrie Irving and Tyrese Haliburton in 2026-27). Two choices follow from the data.

**Availability for returners is a 50/50 blend of the model and the recency baseline.** On the {ret["n_g"]} returner-seasons in which the player appeared in a game, the shipped blend has a mean absolute error of {ret["g_model"] * 82:.1f} games against {ret["g_base"] * 82:.1f} for the baseline, and its bias is {ret["g_model_b"] * 82:+.1f} games (baseline {ret["g_base_b"] * 82:+.1f}); for former rotation stars the bias is {ret["gs_model_b"] * 82:+.1f} games (n = {ret["n_gs"]}).

**Minutes for returners fall, and the model's cut is about the right size.** Among {ret["n_min"]} returners who then played at least 500 minutes, the average last-known minutes were {ret["last"]:.1f} and the return minutes {ret["act"]:.1f}. For the {ret["n_star"]} former rotation stars (28+ minutes before) the drop was larger: **{ret["last_s"]:.1f} → {ret["act_s"]:.1f}**.

{ctx.fig("fig_returners", "Left: minutes before and after a missed season for returners (dashed line: no change). Right: the average error of each minutes projection for all returners and for former stars.")}

A recency average would over-project minutes by {abs(ab["baseline"]):.1f} for all returners and {abs(abs_["baseline"]):.1f} for former stars; the shipped model is within {abs(ab["model"]):.1f} and {abs(abs_["model"]):.1f} minutes of the truth, and blending the minutes back toward the baseline *increases* the error, so no such blend is used for minutes. The model is, if anything, slightly *low* ({abs_["model"]:+.1f} minutes for stars, a 1.4-minute under-projection in a sample of {ret["n_star"]}), a point revisited in section 7.2 for Haliburton.

**Why does the model cut so much?** For Haliburton it projects 26.3 minutes against a recency baseline of 31.1. The cut does not come from his age, a "returner flag" or his team context: when we filled in his missing 2025-26 season with a copy of his 2024-25 season the model returned 33.4 minutes. The reduction comes entirely from the *blank* last-season slot, which the models have learned from past players (in the training seasons, 16 former rotation stars who missed a year went from about 33.0 to 27.1 minutes) to treat as a sign of a smaller role on return.

## 5.5 What we did not change

Because the first suspicion was that team-context features might be corrupting the games-played model (Powell's and Murray's projected games dropped by five to six when context was added), we ran an explicit ablation. Removing the context features from the *availability* model made it worse (squared error +1.3%) and raised the bias for movers, so the features were kept (section 6). Likewise, season-of-career indicators (second- to fourth-year players) reduced the under-projection of third-season players only slightly and did not improve overall accuracy:

{ctx.fig("fig_third_year", "Average miss in minutes and points by years of experience, for the shipped model and a variant with season-of-career flags.")}

{md_table(tyt, {"n": "{:.0f}", "mpg": "{:+.2f}", "ppg": "{:+.2f}"})}

The third-year bias (+0.6 minutes, +0.4 points per game for the shipped model) is real but small relative to the typical miss of two points, and even a full correction would move a player such as Stephon Castle only about nine places on the board, so the simpler model was kept.

# 6. Phase 1, step 4: team context

## 6.1 Roster moves change opportunity

A player's stat line depends on who else is on the floor. When a team loses 150 minutes of rotation play to departures, the minutes and possessions are redistributed; when it adds a star, they shrink. For each player-season we therefore compute, from the player-level box scores:

* **vacated minutes and possessions**: the minutes and possessions used last season by players who are no longer on the team;
* **incoming minutes and possessions**: those brought by players who joined;
* the same quantities **within the player's position group** (guards, forwards or centers), because a guard's role depends more on departing guards;
* a **moved** indicator, and interactions with the player's own usage (a star absorbs vacated shots; a role player does not).

His own contribution is removed so that a player does not "vacate" himself. For past seasons the "opening roster" is the set of players who appeared in the team's first ten games; for 2026-27 it is each team's current roster minus known absentees (a player who will miss many games opens minutes for others). These features simply join the other inputs of the Ridge and LightGBM corrections of section 4; the baseline itself is unchanged.

## 6.2 Evidence from the walk-forward backtest

Held fixed everything else and dropped the context columns for every statistic (the "no context" model of section 4's folds):

{ctx.fig("fig_context_ablation", "Out-of-sample error change when team-context features are removed, by statistic.")}

On average the squared error is **{abl["mean"]:+.1f}%** higher without context, with the largest losses for 2-point attempts, turnovers, free-throw attempts and assists, exactly the quantities that depend on usage and role. In the original development test (movers and stayers separated, which the pooled folds cannot do), context reduced the points error for players who changed teams by about 7%, with similar gains for attempts and turnovers, and left stayers essentially unchanged; we report that earlier result as a design check, not a rerun.

**One caution that led to a player-specific decision (section 8.3).** For the 20 star movers in that test the context features reduced the *points* error by 13% but increased the *minutes* error by 11% (the model under-projected their minutes by about 0.8 per game). The context adjustment is therefore directionally right but can be heavy for a star changing teams, which is exactly what happened to Giannis Antetokounmpo (Milwaukee to Miami).

# 7. Rookies and expert inputs

## 7.1 Rookies: a prior from the draft slot

A rookie has no box scores, so the only prior is his draft slot. All first-season players since 2015-16 are grouped into five buckets (picks 1-5, 6-14, 15-30, 31-60, undrafted) and each bucket's average rates are used as the projection, with each bucket's coefficient of variation stored for the simulation's uncertainty.

{ctx.fig("fig_rookie_tiers", "Average first-season minutes, points and games played by draft-slot bucket.")}

**Why buckets and not a smooth curve?** We compared the bucket means with a regression on log(draft pick) by leaving out each rookie class in turn and predicting it from the others (mean absolute error per game):

{md_table(cmp_t, {c: "{:.3f}" for c in cmp_t.columns if c != "method"})}

The smooth curve is no more accurate (the differences are in the third decimal place), so the simpler buckets are used. The consequence is that every rookie in a bucket gets the same projection, so individual standouts cannot be anticipated from draft slot alone. For the 2026 class, 15 rookies have an outside projection from Yahoo, which **replaces** the draft-slot profile for points, rebounds, assists, steals, blocks, field-goal and three-point percentage and (where given) minutes, with shot volume solved so that the projected points and percentages are reproduced exactly; free-throw percentage, turnovers and availability keep the draft-slot profile:

{md_table(yr_t, {"PPG": "{:.1f}", "RPG": "{:.1f}", "APG": "{:.1f}", "MPG (blank = model's)": "{:.0f}"})}

## 7.2 Four documented judgments

A few players are cases where information outside the data matters more than anything the model can learn. Each is a recorded override or input, not a silent edit:

| Player | Input | Reason | Effect on the 2026-27 board |
|---|---|---|---|
| Brandon Ingram | 60 games out | partially torn Achilles; ESPN lists a short absence, the owner judged at least 3 months and likely much longer | board rank {int(ib["rank"])} |
| Kristaps Porziņģis | 82 games out (whole season) | owner instruction; listed day-to-day but away from the team | board rank {int(pz["rank"])} (last) |
| Giannis Antetokounmpo | 2025-26 season ignored; no team-context adjustment | his 36 games were rest while seeking a trade, not injury; the context penalty for the Miami move was heavy | board rank 6 (section 8.3) |
| Tyrese Haliburton | expected minutes 29.8 | model gave 26.3 after the missed season; owner judged it low | board rank 62 → {int(ha["rank"])} |

For a minutes override, the player's other per-game totals are rescaled at the same time (per-minute rates unchanged); otherwise the board and the simulation, which read per-game totals, would not change. All overrides live in `data/external/` and are reapplied by every rebuild.

# 8. Phase 1, step 5: from projections to a ranking

## 8.1 The nine-category value score

Each projected line is converted to nine **z-scores**, one per category, measured against a reference pool of the 156 highest-valued players (selected iteratively, four passes, so the benchmark is "a drafted player" rather than the league including end-of-bench filler). Counting statistics are multiplied by expected availability; percentages use the **impact** form: the quantity that is z-scored is `(FGM − pool FG% × FGA) × availability`, so a player's shooting percentage matters in proportion to his volume (a 60% shooter on 3 attempts matters far less than a 52% shooter on 15). Turnovers are sign-flipped. The **value** is the sum of the nine z-scores, with one modification, described next.

{ctx.fig("fig_top25", "The top 25 players of the 2026-27 board by category z-score. Rows show each player's archetype.")}

*Reading the heat map.* Jokić is blue almost everywhere except turnovers; Wembanyama's value is concentrated in blocks, rebounds and points; a player with the same total but a different pattern of blue and red is a different kind of asset, which the draft engine of section 10 exploits.

## 8.2 The "z-floor": why a sum of z-scores is wrong at the extremes

In weekly head-to-head each category is **won or lost**, so being six standard deviations bad at one category costs the same *one* category as being three standard deviations bad, but a plain sum charges the six-sigma player twice as much. We therefore floor each category's contribution at −3 when forming the total (only the negative tail; strengths count in full). The raw z-scores are retained and `z_floor_adj` records what the floor added back. For 2026-27 the floor changes {int((proj.z_floor_adj > 0).sum())} of 614 projected players, but **only one inside the board's top 150: Giannis Antetokounmpo**, whose total rises by {gz["floor_add"]:.1f}. The other ten are players projected to miss most or all of the season, whose total is already hopeless.

## 8.3 The Giannis dilemma

This episode is worth telling in full because it shows the process of reconsidering a model: a result that disagreed with every other projection and with the owner's judgment was *not* dismissed as the market being wrong, and *not* patched, but investigated until each piece had a specific, testable cause.

**The symptom.** Our board ranked Giannis Antetokounmpo 93rd (later 103rd), far below every public ranking and the owner's own instinct, who is a consensus top-10 player.

**Step 1: is it a bug? No, and the counterfactuals show why.** At that time his projection was 24.8 points, 29.0 minutes and 65% availability at age 32, with a free-throw impact z-score of −4.7 and three-pointers at −1.7, against a field-goal-percentage z-score of +3.6, giving a total of −1.4. The per-36 rates were normal; the drag was **minutes and games, both anchored on his injury-shortened 2025-26 (36 games, 28.9 minutes)** through the 0.4 recency decay. Cumulative counterfactuals, run when he stood at 103rd, located the causes:

{ctx.fig("fig_giannis_counterfactual", "Cumulative what-if ranks for Giannis: the lower the bar the better. Each bar keeps the earlier changes.")}

* with 82% of games instead of 56, he moves to 57th;
* adding his 2023-24 minutes (34.2) moves him to 25th;
* replacing the model's per-minute rates by his actual 2023-24 line moves him to 6th;
* a neutral free-throw percentage alone moves him to 3rd; free throws by themselves were worth roughly 22 places at full role.

So two separate things were involved: a **data** problem (a depressed 2025-26 anchoring his minutes and games) and a **scoring-rule** problem (free throws). (At the very first look, with him at 93rd, the same exercise gave 66th if nobody were discounted for games, 84th with half the discount, and 11th if he were given his 2023-24 role of 34.2 minutes and 82% of games.)

**Step 2: the 2025-26 season was not an injury.** The owner pointed out that Giannis's 36 games were *rest* while he pushed for a trade, not a health problem. The model cannot know that. We added a mechanism to **drop a season from a player's history** (`ignore_seasons`): earlier seasons slide forward one lag and age is adjusted, as if 2025-26 had never happened. Result: his projection rose to 32.1 minutes, 26.6 points and 62 games, and his rank improved from **103rd to 59th**.

**Step 3: the team-context penalty was heavy.** The move to Miami cost him 1.7 points and 0.9 minutes through the context features (without them he would have ranked 33rd). We tested that choice directly on star movers (the check described in section 6.2): the context features reduced the points error by 13% but made the minutes error 11% worse, i.e. directionally right but too heavy for stars. At the owner's request his context adjustment was switched off (`no_team_context`): **59th to 29th**.

**Step 4: the rule itself was the last problem.** Even then he stood 29th, and "even with availability 100% for everyone he is only 79th per game". The cause is the free-throw category: his FT% impact is a six-sigma outlier (z of −5.2, −5.5 and −6.3 in his last three full seasons) while the standard deviation of free-throw impact across the pool is only about 0.3 makes per game. By a plain sum of z-scores, **our own method ranked his actual 2022-23, 2023-24 and 2024-25 seasons 133rd, 28th and 59th**, a result nobody believes:

{ctx.fig("fig_giannis_z", "Giannis's category z-scores for 2026-27 (raw and as counted in the total) and where his free-throw impact lies in the distribution of the top 200 players.")}

{ctx.fig("fig_giannis_seasons", "Giannis's rank in each of the last five seasons, computed from what he actually did, with and without the −3 floor.")}

{md_table(gs, {"GP": "{:.0f}", "rank, plain sum": "{:.0f}", "rank, floored": "{:.0f}", "FT% z-score": "{:+.1f}", "PPG": "{:.1f}", "MPG": "{:.1f}"})}

The floor (section 8.2) is the principled fix: it encodes what a weekly category format does with a lost category. With the floor, his actual seasons rank 73rd, 10th and 12th, and his 2026-27 projection moves from **29th to 6th**, which matches his market order of 6.

**Step 5: did we try to *exploit* his weakness (punting free throws)?** Since only 5 of 9 categories are needed, a team that gives up free throws can build around a player like Giannis. Our first test of that idea forced the later picks to ignore FT%. It made the team *worse*: a balanced roster that includes good free-throw shooters won more weeks, and extra punting (free throws plus three-pointers, assists and steals) was far worse. 

{ctx.fig("fig_punt", "Punt experiment: simulated weekly win probability with Giannis as the pick when the later picks follow the board, punt free throws, or punt four categories (bars labelled with the percentage; the four-category result was recorded as a 39-44% range).")}

{ctx.fig("fig_giannis_timeline", "Giannis's board rank after each modelling decision.")}

**What the episode shows.** (i) A single ranking number can hide three different problems: a data anchor (rest year), a model choice (team context) and a scoring rule (sum of z-scores). (ii) Counterfactual analysis (change one input, re-rank) is a fast way to separate them. (iii) The adjustments were each tested against data or justified by how the league scores weeks, and the floor changes *no one else* in the top 150, so it did not buy Giannis a better rank at anyone's expense. (iv) What did **not** work (punting) was tested and reported, not assumed.
"""
