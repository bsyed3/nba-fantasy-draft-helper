"""Sections 9-10: the simulation (Phase 2) and its validation; the draft engine and pick policy (Phase 3), tuning and real-season validation."""
from __future__ import annotations

import glob

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm, t as tdist

from scripts.final_report.common import *  # noqa: F403
from src.draft.engine import DraftEngine
from src.draft.mock import best_available, run_mock_draft
from src.draft.policy import AdaptivePolicy, load_weeks
from src.draft.state import DraftState, LeagueSettings
from src.simulation.calibration import COUNT_DIMS, Calibration
from src.simulation.library import STAT_IDX, Library

KW = dict(n_teams=10, roster_size=13, active_slots=10, max_centers=3)


# ---------------------------------------------------------------------------------------------------------------------
# simulation figures
# ---------------------------------------------------------------------------------------------------------------------
def fig_copula(cal) -> None:
    R = np.array(cal.corr)
    names = ["2PA", "3PA", "FTA", "REB", "AST", "STL", "BLK", "TOV", "2P luck", "3P luck", "FT luck"]
    fig, ax = F.new(7.2, 6.2)
    im = ax.imshow(R, cmap=F.SEQ_CMAP, vmin=0, vmax=0.5)
    ax.set_xticks(range(11), names, rotation=45, ha="right"); ax.set_yticks(range(11), names); ax.grid(False)
    for i in range(11):
        for j in range(11):
            if i != j:
                ax.text(j, i, f"{R[i, j]:.2f}", ha="center", va="center", fontsize=7.5, color="white" if R[i, j] > 0.3 else C["ink"])
    fig.colorbar(im, ax=ax, fraction=0.04, label="correlation")
    F.headline(fig, "Statistics move together within a game: rebounds with shot attempts, assists with turnovers",
               "Gaussian-copula correlation between a player's per-game statistics (within player, estimated from three recent seasons)")
    F.save(fig, "fig_copula")


def fig_weekly_dist(ctx, lib: Library) -> pd.DataFrame:
    proj = ctx.proj
    pick = [proj[proj["name"] == nm].iloc[0] for nm in ["Shai Gilgeous-Alexander", "Anthony Davis", "Kyrie Irving"]]
    ids = {int(p): i for i, p in enumerate(lib.player_ids)}
    fig, axes = plt.subplots(1, 3, figsize=(10.4, 3.9), sharex=True)
    rows = []
    for ax, p in zip(axes, pick):
        s = lib.stats[ids[int(p["player_id"])]].astype(np.int32)
        pts = (2 * s[..., STAT_IDX["fgm"]] + s[..., STAT_IDX["tpm"]] + s[..., STAT_IDX["ftm"]]).sum(axis=1)
        zero = (lib.played[ids[int(p["player_id"])]].sum(axis=1) == 0).mean()
        ax.hist(pts, bins=np.arange(0, 181, 6), color=C["blue"], edgecolor=C["surface"], linewidth=0.8)
        ax.axvline(pts.mean(), color=C["orange"], lw=2)
        ax.set_title(f"{p['name']}\nmean {pts.mean():.0f}; 10-90%: {np.percentile(pts, 10):.0f}-{np.percentile(pts, 90):.0f}; week missed: {zero:.0%}", fontsize=9.3, loc="left", linespacing=1.4)
        ax.set_xlabel("points in a matchup period"); ax.set_yticks([])
        rows.append(dict(player=p["name"], availability=p["avail"], mean=pts.mean(), p10=np.percentile(pts, 10), p90=np.percentile(pts, 90), zero_weeks=zero))
    F.headline(fig, "A player is a distribution, not a number: the spike at zero is a week missed entirely",
               f"Simulated weekly points for three players with different availability ({lib.stats.shape[1]} simulated matchup periods each)")
    F.legend_top(fig, [F.swatch(C["blue"], "simulated periods"), F.swatch(C["orange"], "mean", "line")])
    F.save(fig, "fig_weekly_dist")
    return pd.DataFrame(rows)


def library_ratios(ctx, lib: Library) -> dict:
    proj = ctx.proj.set_index("player_id").loc[lib.player_ids]
    w = lib.stats.astype(np.float64).sum(axis=2)                                   # [P, S, 10] weekly totals
    g = lib.played.sum(axis=2).sum(axis=1).astype(float)                           # games per player over all sims
    tot = w.sum(axis=1)
    ok = (g > 200) & (proj["avail"].to_numpy() > 0.3)
    f = lambda k: tot[ok, STAT_IDX[k]]
    pts = 2 * f("fgm") + f("tpm") + f("ftm")
    gg = g[ok]
    pj = proj[ok]
    r = lambda lib_pg, proj_pg: float(np.mean(lib_pg) / np.mean(proj_pg))               # ratio of means, as in library_diagnostics.py
    return {"PTS": r(pts / gg, pj["pts"]), "3PM": r(f("tpm") / gg, pj["tpm"]), "REB": r(f("reb") / gg, pj["reb"]), "AST": r(f("ast") / gg, pj["ast"]),
            "STL": r(f("stl") / gg, pj["stl"]), "BLK": r(f("blk") / gg, pj["blk"]), "FG%": r(f("fgm") / f("fga"), pj["fg_pct"]), "FT%": r(f("ftm") / f("fta"), pj["ft_pct"])}


def fig_mean_match(ctx, lib: Library) -> dict:
    after = library_ratios(ctx, lib)
    # before the correction (library_diagnostics.py run on the first rebuilt library, project log): simulated / projected, per-game while playing
    before = {"PTS": 1.018, "3PM": 1.042, "REB": 1.001, "AST": 1.001, "STL": 1.000, "BLK": 1.004, "FG%": 1.020, "FT%": 1.015}
    cats = list(before)
    fig, ax = F.new(8.6, 4.2)
    x = np.arange(len(cats)); w = 0.36
    ax.bar(x - w / 2, [(before[c] - 1) * 100 for c in cats], w, color=C["orange"], label="before the correction")
    ax.bar(x + w / 2, [(after[c] - 1) * 100 for c in cats], w, color=C["blue"], label="after (shipped library)")
    for xi, c in zip(x, cats):
        for dx, ratio in ((-w / 2, before[c]), (w / 2, after[c])):
            v = (ratio - 1) * 100
            ax.text(xi + dx, v + (0.06 if v >= 0 else -0.06), f"{v:+.1f}", ha="center", va="bottom" if v >= 0 else "top", fontsize=8, color=C["ink2"])
    ax.axhline(0, color=C["ink2"], lw=1); ax.set_xticks(x, cats); ax.set_ylabel("simulated minus projected per-game value (%)"); ax.grid(axis="x", visible=False)
    F.headline(fig, "The simulation made too many shots: points ran 1.8% and threes 4.2% above the projections until the make rates were mean-matched",
               "Library mean divided by projection (players with a real chance to play), before and after the correction")
    F.legend_top(fig, *ax.get_legend_handles_labels())
    F.save(fig, "fig_mean_match")
    return dict(before=before, after=after)


def realism_arrays() -> dict:
    parts = [np.load(f) for f in sorted(glob.glob(str(EXP / "realism_20*.npz")))]
    return dict(parts=parts, years=[int(p["year"]) for p in parts], n=int(parts[0]["n_draws"]),
                pit=np.concatenate([p["pit"] for p in parts]), sim_pit=np.concatenate([p["sim_pit"] for p in parts]),
                ny=np.concatenate([np.full(len(p["pit"]), int(p["year"])) for p in parts]),
                pm=np.concatenate([p["p"] for p in parts]), po=np.concatenate([p["o"] for p in parts]),
                py=np.concatenate([np.full(len(p["p"]), int(p["year"])) for p in parts]))


def fig_realism(ctx, rz: dict) -> dict:
    n = rz["n"]
    clip = lambda u: np.clip(u, 0.5 / n, 1 - 0.5 / n)
    pit, z = rz["pit"], norm.ppf(clip(rz["pit"]))
    zs = norm.ppf(clip(rz["sim_pit"]))
    # PIT histograms
    fig, axes = plt.subplots(3, 3, figsize=(10, 7.4))
    for ax, (k, c) in zip(axes.ravel(), enumerate(CATEGORIES)):
        h, _ = np.histogram(pit[:, k], bins=10, range=(0, 1))
        ax.bar(np.arange(10) * 10 + 5, h / h.sum() * 100, width=8.6, color=C["blue"])
        ax.axhline(10, color=C["orange"], lw=2)
        ax.set_title(f"{CAT_LABELS[c]}  (z variance {z[:, k].var():.2f})", fontsize=9.5, loc="left"); ax.set_yticks([0, 5, 10, 15]); ax.set_ylim(0, 17)
        ax.set_xticks([0, 50, 100]); ax.grid(axis="x", visible=False)
    F.headline(fig, "Where did each real team-week fall inside its simulated distribution? Flat means calibrated; U-shaped means the simulation is too narrow",
               f"PIT histograms, {len(pit):,} team-weeks from 2022-23 to 2025-26 (percentile bucket of the real value among {n} simulated draws, % of team-weeks)")
    F.legend_top(fig, [F.swatch(C["blue"], "share of team-weeks"), F.swatch(C["orange"], "perfect calibration: 10% per bucket", "line")])
    F.save(fig, "fig_pit_hist")
    # z variance by category and by season
    cat_var = z.var(axis=0)
    seasons = sorted(set(rz["ny"]))
    sea_var = [z[rz["ny"] == y].var() for y in seasons]
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.0))
    ax = axes[0]
    ax.bar(range(9), cat_var, color=C["blue"], width=0.62)
    ax.axhline(zs.var(), color=C["orange"], lw=2)
    for xi, v in enumerate(cat_var):
        ax.text(xi, max(v, zs.var()) + 0.02, f"{v:.2f}", ha="center", fontsize=8, color=C["ink2"])
    ax.set_xticks(range(9), [CAT_LABELS[c] for c in CATEGORIES]); ax.set_ylim(0.8, 1.32); ax.set_title("By category", fontsize=10, loc="left"); ax.grid(axis="x", visible=False)
    ax.set_ylabel("variance of normal-scored PIT")
    ax = axes[1]
    ax.bar(range(len(seasons)), sea_var, color=C["blue"], width=0.55)
    ax.axhline(zs.var(), color=C["orange"], lw=2)
    for xi, v in enumerate(sea_var):
        ax.text(xi, max(v, zs.var()) + 0.02, f"{v:.2f}", ha="center", fontsize=8.5, color=C["ink2"])
    ax.set_xticks(range(len(seasons)), [SEASON(y) for y in seasons]); ax.set_ylim(0.8, 1.32); ax.set_title("By season", fontsize=10, loc="left"); ax.grid(axis="x", visible=False)
    F.headline(fig, "Real weeks are about 5% wider than simulated overall; the 2022-23 and 2023-24 seasons match, 2024-25 and 2025-26 are noisier",
               "Variance of the PIT converted to normal scores (above 1 = the simulation is too narrow); the orange line is the same statistic for the simulation placed among itself")
    F.legend_top(fig, [F.swatch(C["blue"], "real team-weeks"), F.swatch(C["orange"], "simulation benchmark", "line")])
    F.save(fig, "fig_pit_variance")
    # correlation heatmaps
    cm, cs_ = np.corrcoef(z.T), np.corrcoef(zs.T)
    names = [CAT_LABELS[c] for c in CATEGORIES]
    fig, axes = plt.subplots(1, 3, figsize=(11.4, 5.3))
    for ax, (mat, ttl, vmin, vmax, cmap) in zip(axes, [(cm, "Real weeks", -0.2, 0.8, F.SEQ_CMAP), (cs_, "Simulation benchmark", -0.2, 0.8, F.SEQ_CMAP), (cm - cs_, "Real minus simulated", -0.3, 0.3, F.DIV_CMAP)]):
        im = ax.imshow(mat, cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_xticks(range(9), names, rotation=45, ha="right", fontsize=8); ax.set_yticks(range(9), names, fontsize=8); ax.grid(False)
        for i in range(9):
            for j in range(9):
                if i != j:
                    ax.text(j, i, f"{mat[i, j]:.2f}".replace("0.", ".").replace("-.", "−."), ha="center", va="center", fontsize=6, color="white" if (cmap is F.SEQ_CMAP and mat[i, j] > 0.45) else C["ink"])
        ax.set_title(ttl, fontsize=10, loc="left")
    F.headline(fig, "Categories move together in real weeks about as much as they do in the simulation",
               f"Correlation of the normal-scored PITs across categories: mean {cm[~np.eye(9, dtype=bool)].mean():+.2f} real vs {cs_[~np.eye(9, dtype=bool)].mean():+.2f} simulated")
    F.save(fig, "fig_corr_real_sim")
    # reliability
    pm, po = rz["pm"], rz["po"]
    bins = np.linspace(0, 1, 11)
    idx = np.clip(np.digitize(pm, bins) - 1, 0, 9)
    rel = pd.DataFrame([dict(lo=bins[i], hi=bins[i + 1], n=int((idx == i).sum()), p=pm[idx == i].mean(), o=po[idx == i].mean()) for i in range(10) if (idx == i).sum() > 0])
    from sklearn.linear_model import LogisticRegression
    xl = np.log(np.clip(pm, 0.01, 0.99) / (1 - np.clip(pm, 0.01, 0.99)))[:, None]
    lr = LogisticRegression(C=1e6, max_iter=1000).fit(np.vstack([xl, xl]), np.r_[np.ones(len(xl)), np.zeros(len(xl))], sample_weight=np.r_[po, 1 - po])
    brier = float(np.mean((pm - po) ** 2))
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.6), gridspec_kw={"width_ratios": [1.15, 1]})
    ax = axes[0]
    ax.plot([0, 1], [0, 1], color=C["ink2"], lw=1.2, ls=(0, (4, 3)))
    ax.scatter(rel.p, rel.o, s=20 + rel.n / 40, color=C["blue"], zorder=3, edgecolor="white", linewidth=1)
    ax.plot(rel.p, rel.o, color=C["blue"], lw=1.6)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect("equal")
    ax.set_xlabel("simulated probability that team A beats team B in a week"); ax.set_ylabel("how often team A actually won")
    ax.set_title(f"Reliability: slope {lr.coef_[0][0]:.2f}, Brier {brier:.3f} (50/50 guess: {np.mean((0.5 - po) ** 2):.3f})", fontsize=9.6, loc="left")
    ax = axes[1]
    ax.hist(pm, bins=np.linspace(0, 1, 21), color=C["blue"], edgecolor=C["surface"], linewidth=0.7)
    ax.set_xlabel("simulated probability"); ax.set_ylabel("team-pair-weeks"); ax.set_title(f"{len(pm):,} team-pair-weeks", fontsize=9.6, loc="left"); ax.grid(axis="x", visible=False)
    F.headline(fig, "Simulated matchup probabilities are close to the observed win rates, slightly too extreme at the top",
               "Each dot is a decile of predicted probability (dot area = number of team-pair-weeks); the dashed line is perfect calibration")
    F.save(fig, "fig_reliability")
    # star premium by draft-slot gap
    T, DR = 10, 6
    pairs = []
    for part in rz["parts"]:
        W = int(part["n_weeks"])
        p, o = part["p"].reshape(DR, -1, W), part["o"].reshape(DR, -1, W)
        k = 0
        for a in range(T):
            for b in range(a + 1, T):
                pairs.append((b - a, p[:, k].mean(), o[:, k].mean()))
                k += 1
    pr = pd.DataFrame(pairs, columns=["gap", "p", "o"]).groupby("gap").mean()
    fig, ax = F.new(8.4, 4.0)
    x = np.arange(len(pr)); w = 0.36
    ax.bar(x - w / 2, pr.p * 100, w, color=C["orange"], label="simulation's probability")
    ax.bar(x + w / 2, pr.o * 100, w, color=C["blue"], label="actual win rate")
    ax.axhline(50, color=C["ink2"], lw=1); ax.set_ylim(48, 64)
    ax.set_xticks(x, [f"{g}" for g in pr.index]); ax.set_xlabel("draft-slot gap between the two teams (earlier pick vs later pick)"); ax.set_ylabel("win % of the earlier-slot team"); ax.grid(axis="x", visible=False)
    F.headline(fig, "Top-heavy early-slot rosters beat the simulation's expectation by a few points, increasingly so as the slot gap grows",
               "Mean simulated probability and mean actual win rate of the earlier-slot team in mock leagues, 2022-23 to 2025-26 (small samples at large gaps)")
    F.legend_top(fig, *ax.get_legend_handles_labels())
    F.save(fig, "fig_star_premium")
    return dict(cat_var=cat_var, ft_mean_z=float(z[:, CATEGORIES.index("ft_pct")].mean()), ft_mean_pit=float(pit[:, CATEGORIES.index("ft_pct")].mean()),
                sea_var=dict(zip(seasons, sea_var)), z_var=float(z.var()), bench=float(zs.var()), corr_real=float(cm[~np.eye(9, dtype=bool)].mean()),
                corr_sim=float(cs_[~np.eye(9, dtype=bool)].mean()), slope=float(lr.coef_[0][0]), brier=brier, brier0=float(np.mean((0.5 - po) ** 2)),
                fav=float(np.mean(po[pm > 0.55] > 0.5)), nfav=int((pm > 0.55).sum()), n_tw=len(pit), n_pair=len(pm), rel=rel, pr=pr,
                cover=[float(((pit > a) & (pit < 1 - a)).mean()) for a in (0.25, 0.1, 0.05)], mean_pit_season={y: float(pit[rz["ny"] == y].mean()) for y in seasons})


# ---------------------------------------------------------------------------------------------------------------------
# Phase 3 figures
# ---------------------------------------------------------------------------------------------------------------------
def _gain(df: pd.DataFrame, arm: str = "out_arm", board: str = "out_board") -> tuple[float, float, int]:
    d = df[arm] - df[board]
    return float(d.mean()), float(d.std(ddof=1) / np.sqrt(len(d))), len(d)


def fig_tuning(ctx) -> dict:
    spec = [("more_A_s40_c6_m010", "40 draws\n6 cand.\nguard 0.01"), ("more_B_s200_c6_m010", "200 draws\n6 cand.\nguard 0.01"), ("more_C_s200_c12_m010", "200\n12\n0.01"),
            ("more_D_s200_c24_m010", "200\n24\n0.01"), ("more_E_s200_c6_m005", "200 draws\n6 cand.\nguard 0.005"), ("more_F_s200_c12_m005", "200\n12\n0.005"), ("more_G_s200_c24_m005", "200\n24\n0.005")]
    rows = []
    for name, lab_ in spec:
        d = read_exp(name)
        if d is None:
            continue
        ho, se, n = _gain(d, "out_adaptive", "out_board")
        ins, se_i, _ = _gain(d, "in_adaptive", "in_board")
        rows.append(dict(setting=lab_, held=ho, se=se, ins=ins, n=n))
    t = pd.DataFrame(rows)
    fig, ax = F.new(10.0, 4.6)
    x = np.arange(len(t)); w = 0.36
    ax.bar(x - w / 2, t.ins * 100, w, color=C["orange"], label="in-sample (the library the policy searched)")
    ax.bar(x + w / 2, t.held * 100, w, color=C["blue"], label="held-out (a different simulation of the same season)")
    ax.errorbar(x + w / 2, t.held * 100, yerr=1.96 * t.se * 100, fmt="none", ecolor=C["ink2"], capsize=3, elinewidth=1.2)
    for xi, r in zip(x, t.itertuples()):
        ax.text(xi + w / 2, (r.held + 1.96 * r.se) * 100 + 0.12, f"{r.held * 100:.1f}", ha="center", fontsize=8.3, color=C["ink2"]); ax.text(xi - w / 2, r.ins * 100 + 0.12, f"{r.ins * 100:.1f}", ha="center", fontsize=8.3, color=C["ink2"])
    ax.set_xticks(x, t.setting, fontsize=7.9); ax.set_ylabel("gain over drafting by the board\n(points of weekly win probability)"); ax.grid(axis="x", visible=False)
    F.headline(fig, "More simulated draws and a smaller guard help; searching more than about a dozen candidates does not",
               f"Paired mock drafts, {int(t.n.iloc[0])} per setting (10 draft slots x 12 bot seeds), scored on a held-out library; bars are 95% intervals. In-sample bars show how much of a gain is noise-fitting")
    F.legend_top(fig, *ax.get_legend_handles_labels())
    F.save(fig, "fig_tuning")
    return dict(table=t)


def fig_fresh(ctx) -> dict:
    a, e = read_exp("confirm_A_fresh"), read_exp("confirm_E_fresh")
    ia, ie = a.set_index(["slot", "seed"]), e.set_index(["slot", "seed"])
    ga, ge = ia.out_adaptive - ia.out_board, ie.out_adaptive - ie.out_board
    d = ge - ga
    n = len(d)
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.2), gridspec_kw={"width_ratios": [1, 1.5]})
    ax = axes[0]
    vals = [ga.mean() * 100, ge.mean() * 100]; ses = [ga.std(ddof=1) / np.sqrt(n) * 100, ge.std(ddof=1) / np.sqrt(n) * 100]
    ax.bar([0, 1], vals, color=[C["orange"], C["blue"]], width=0.55)
    ax.errorbar([0, 1], vals, yerr=1.96 * np.array(ses), fmt="none", ecolor=C["ink2"], capsize=4, elinewidth=1.3)
    for xi, v in zip([0, 1], vals):
        ax.text(xi, v + 0.45, f"{v:+.2f}", ha="center", fontsize=9, color=C["ink2"])
    ax.set_xticks([0, 1], ["old setting\n(40 draws, guard 0.01)", "new setting\n(200 draws, guard 0.005)"], fontsize=8.8); ax.set_ylabel("gain over drafting by the board (points)"); ax.grid(axis="x", visible=False)
    ax.set_title(f"{n} fresh drafts (never used for tuning)", fontsize=10, loc="left")
    ax = axes[1]
    bys = d.groupby(level=0).mean() * 100
    ax.bar(bys.index, bys.values, color=[C["blue"] if v >= 0 else C["red"] for v in bys.values], width=0.62)
    ax.axhline(0, color=C["ink2"], lw=1); ax.axhline(d.mean() * 100, color=C["orange"], lw=1.6, ls=(0, (4, 3)))
    ax.set_xticks(range(1, 11)); ax.set_xlabel("draft slot"); ax.set_ylabel("new minus old (points)"); ax.grid(axis="x", visible=False)
    ax.set_title(f"Paired difference by slot (average {d.mean() * 100:+.2f}, se {d.std(ddof=1) / np.sqrt(n) * 100:.2f})", fontsize=10, loc="left")
    F.headline(fig, "The improvement survives fresh drafts, at about half the size it showed on the drafts it was tuned on",
               "Held-out gain over board order on 150 paired drafts with new bot seeds; right panel: how the new setting compares with the old one in each draft slot")
    F.save(fig, "fig_fresh")
    return dict(old=float(ga.mean()), old_se=float(ga.std(ddof=1) / np.sqrt(n)), new=float(ge.mean()), new_se=float(ge.std(ddof=1) / np.sqrt(n)),
                diff=float(d.mean()), diff_se=float(d.std(ddof=1) / np.sqrt(n)), better=float((d > 0).mean()), n=n, pos_slots=int((bys > 0).sum()))


def fig_candidate_pool(ctx) -> dict:
    spec = [("cand_k20_m01", "engine, short list\nof 20, guard 0.01", C["blue"]), ("cand_r200_m01", "engine, all 200,\nguard 0.01", C["aqua"]), ("cand_r200_m00", "engine, all 200,\nno guard", C["orange"]),
            ("cand_r200_m02", "engine, all 200,\nguard 0.02", C["yellow"]), ("cand_adaptive", "category-aware\npolicy (40 draws)", C["violet"])]
    rows = []
    for name, lab_, col in spec:
        d = read_exp(name)
        if d is None:
            continue
        ho, se, n = _gain(d)
        rows.append(dict(label=lab_, held=ho, se=se, n=n, col=col, changed=d["changed"].mean()))
    t = pd.DataFrame(rows)
    fig, ax = F.new(9.4, 4.4)
    x = np.arange(len(t))
    ax.bar(x, t.held * 100, color=t.col, width=0.6)
    ax.errorbar(x, t.held * 100, yerr=1.96 * t.se * 100, fmt="none", ecolor=C["ink2"], capsize=4, elinewidth=1.3)
    for xi, r in zip(x, t.itertuples()):
        ax.text(xi, (r.held + 1.96 * r.se) * 100 + 0.1, f"{r.held * 100:+.2f}", ha="center", fontsize=9, color=C["ink2"])
    ax.set_xticks(x, [f"{l}\n[{c:.1f}/13 players changed]" for l, c in zip(t.label, t.changed)], fontsize=7.8); ax.set_ylabel("held-out gain over board order (points)"); ax.grid(axis="x", visible=False)
    F.headline(fig, "Looking at every player up to rank 200 does not find better picks, and without a guard it chases noise",
               f"Paired mock drafts, {int(t.n.iloc[0])} per bar (10 slots x 6 seeds), held-out library; bars are 95% intervals. The engine's 'guard' only departs from the board when ahead by more than the margin")
    F.save(fig, "fig_candidate_pool")
    return dict(table=t)


def _val_files() -> dict:
    out = {}
    for y in range(2021, 2026):
        row = {}
        f_old = DATA / f"validation_results_adaptive_{y}.csv"
        f_eng = DATA / f"validation_results_margin0.01_{y}.csv" if y != 2021 else DATA / "validation_results_margin0.01_2021.csv"
        f_new = DATA / f"validation_results_more_{y}.csv"
        if y == 2021:
            d = pd.read_csv(f_eng)
            row["engine"], row["policy_old"] = d["real_engine"] - d["real_static"], d["real_adaptive"] - d["real_static"]
        else:
            if f_old.exists():
                d = pd.read_csv(f_old); row["policy_old"] = d["real_adaptive"] - d["real_static"]
            fe = sorted(DATA.glob(f"validation_results_margin0.01_{y}.csv"))
            if fe:
                d = pd.read_csv(fe[0]); row["engine"] = d["real_engine"] - d["real_static"]
        if f_new.exists():
            d = pd.read_csv(f_new); row["policy_new"] = d["real_adaptive"] - d["real_static"]
        f0 = DATA / "validation_results_margin0_final.csv"
        if f0.exists() and y >= 2022:
            d = pd.read_csv(f0); d = d[d["year"] == y]; row["engine_noguard"] = d["real_engine"] - d["real_static"]
        out[y] = row
    return out


def fig_real_seasons(ctx) -> dict:
    v = _val_files()
    rules = [("engine_noguard", "engine, no guard", C["orange"]), ("engine", "engine, guard 0.01", C["aqua"]), ("policy_old", "policy, old setting", C["yellow"]), ("policy_new", "policy, new setting", C["blue"])]
    years = list(range(2021, 2026))
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.6), gridspec_kw={"width_ratios": [2.4, 1]})
    ax = axes[0]
    w = 0.2
    stats = {}
    for i, (key, lab_, col) in enumerate(rules):
        means = [v[y][key].mean() * 100 if key in v[y] else np.nan for y in years]
        ses = [v[y][key].std(ddof=1) / np.sqrt(len(v[y][key])) * 100 if key in v[y] else np.nan for y in years]
        xs = np.arange(len(years)) + (i - 1.5) * w
        ax.bar(xs, means, w, color=col, label=lab_)
        ax.errorbar(xs, means, yerr=1.96 * np.array(ses), fmt="none", ecolor=C["ink2"], capsize=2, elinewidth=0.9)
        yy = [m for m in means if not np.isnan(m)]
        if len(yy) >= 4:
            m, s = float(np.mean(yy)), float(np.std(yy, ddof=1) / np.sqrt(len(yy)))
            stats[key] = dict(mean=m, se=s, n=len(yy), t=m / s, p=float(2 * (1 - tdist.cdf(abs(m / s), len(yy) - 1))), pos=int(sum(v_ > 0 for v_ in yy)), by_year=dict(zip(years, means)))
    ax.axhline(0, color=C["ink2"], lw=1)
    ax.set_xticks(np.arange(len(years)), [SEASON(y) for y in years]); ax.set_ylabel("gain over best-available (points)"); ax.grid(axis="x", visible=False)
    ax.set_title("By season (bars: 95% interval over mock drafts)", fontsize=10, loc="left")
    ax = axes[1]
    ks = [k for k, _, _ in rules if k in stats]
    ax.bar(range(len(ks)), [stats[k]["mean"] for k in ks], color=[c for k, _, c in rules if k in stats], width=0.6)
    ax.errorbar(range(len(ks)), [stats[k]["mean"] for k in ks], yerr=[1.96 * stats[k]["se"] for k in ks], fmt="none", ecolor=C["ink2"], capsize=4, elinewidth=1.3)
    for xi, k in enumerate(ks):
        ax.text(xi, stats[k]["mean"] + 1.96 * stats[k]["se"] + 0.5, f"{stats[k]['mean']:+.1f}", ha="center", fontsize=9, color=C["ink2"])
    ax.axhline(0, color=C["ink2"], lw=1); ax.set_xticks(range(len(ks)), [lab for k, lab, _ in rules if k in stats], fontsize=7.4, rotation=20, ha="right")
    ax.set_ylim(-5, 10); ax.set_title("Mean over seasons (95% t interval)", fontsize=10, loc="left"); ax.grid(axis="x", visible=False)
    F.headline(fig, "On real box scores the new policy setting has a positive average gain in every season and the highest mean over seasons",
               "Mock drafts scored on actual day-by-day results; the engine without a guard covers 2022-2025 only; the right panel averages the seasons")
    F.legend_top(fig, *axes[0].get_legend_handles_labels(), ncol=4)
    F.save(fig, "fig_real_seasons")
    return stats


def pick1_engine(ctx, lib: Library):
    proj = ctx.proj
    s1 = LeagueSettings(my_slot=1, **KW)
    eng = DraftEngine(lib, proj, s1)
    rec = eng.recommend(DraftState(s1, centers=set(eng.center_ids)), rollouts=2, max_rank=200)
    rec["engine_rank"] = rec.win_prob.rank(ascending=False, method="first").astype(int)
    top = rec.head(10)
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.7), gridspec_kw={"width_ratios": [1, 1.5]})
    ax = axes[0]
    y = np.arange(len(top))
    ax.barh(y, top["win_prob"] * 100, color=C["blue"], height=0.62)
    for yi, v in zip(y, top["win_prob"]):
        ax.text(v * 100 + 0.15, yi, f"{v * 100:.1f}%", va="center", fontsize=8.5, color=C["ink2"])
    ax.set_yticks(y, top["name"]); ax.invert_yaxis(); ax.set_xlim(45, top["win_prob"].max() * 100 + 3)
    ax.set_xlabel("chance to win a weekly matchup (%)"); ax.grid(axis="y", visible=False)
    ax = axes[1]
    mat = top[[f"cat_{c}" for c in CATEGORIES]].to_numpy() * 100
    im = ax.imshow(mat, cmap=F.DIV_CMAP, vmin=30, vmax=70, aspect="auto")
    ax.set_xticks(range(9), [CAT_LABELS[c] for c in CATEGORIES]); ax.xaxis.tick_top(); ax.set_yticks([]); ax.grid(False)
    for i in range(mat.shape[0]):
        for j in range(9):
            ax.text(j, i, f"{mat[i, j]:.0f}", ha="center", va="center", fontsize=8, color="white" if abs(mat[i, j] - 50) > 12 else C["ink"])
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_xlabel("chance to win each category (%)   blue > 50 > red")
    F.headline(fig, f"Pick 1: {top.iloc[0]['name']} leads, {top.iloc[1]['name']} is {(top.iloc[0]['win_prob'] - top.iloc[1]['win_prob']) * 100:.1f} points behind and {top.iloc[2]['name']} {(top.iloc[0]['win_prob'] - top.iloc[2]['win_prob']) * 100:.1f}",
               "Simulated matchup win probability if each player is my first pick (draft slot 1, 10 teams, 2 roll-outs); category win chances on the right")
    F.save(fig, "fig_pick1")
    return eng, rec


def mock_comparison(ctx, lib: Library, weeks: np.ndarray) -> dict:
    """One mock draft from slot 4 (the owner's slot): drafting by the board vs by the policy, with the same bots."""
    proj = ctx.proj
    s = LeagueSettings(my_slot=4, **KW)
    eng = DraftEngine(lib, proj, s)
    pol = AdaptivePolicy(eng, weeks=weeks)
    from src.draft.policy import adaptive_pick
    ros_b = run_mock_draft(eng, s, best_available, bot_seed=9003)
    ros_p = run_mock_draft(eng, s, adaptive_pick(pol), bot_seed=9003)
    evb, evp = eng.evaluate(ros_b), eng.evaluate(ros_p)
    me = s.my_slot - 1
    pn = proj.set_index("player_id")
    def roster(ros):
        idx = sorted(ros[s.my_slot], key=lambda j: -eng.value[j])
        r = pn.loc[eng.pid[idx], ["name", "elig", "rank", "mpg", "avail", "pts", "reb", "ast"]].reset_index(drop=True)
        return r
    rb, rp = roster(ros_b), roster(ros_p)
    same = set(rb["name"]) & set(rp["name"])
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.0), sharey=True)
    for ax, ev, ttl, col in zip(axes, (evb, evp), ("Drafting by the board", "Drafting with the category-aware policy"), (C["axis"], C["blue"])):
        cw = ev.cat_win_prob[me] * 100
        ax.bar(range(9), cw - 50, bottom=50, color=[col if v >= 50 else C["red"] for v in cw], width=0.62)
        for i, v in enumerate(cw):
            ax.text(i, v + (1.2 if v >= 50 else -1.2), f"{v:.0f}", ha="center", va="bottom" if v >= 50 else "top", fontsize=8.5, color=C["ink2"])
        ax.axhline(50, color=C["ink2"], lw=1); ax.set_xticks(range(9), [CAT_LABELS[c] for c in CATEGORIES]); ax.set_ylim(5, 100); ax.grid(axis="x", visible=False)
        ax.set_title(f"{ttl}: weekly win probability {ev.win_prob[me] * 100:.1f}%", fontsize=9.6, loc="left", pad=8)
    axes[0].set_ylabel("chance to win each category (%)")
    F.headline(fig, "A category-aware build: the policy trades some categories for others rather than spreading value evenly",
               "One mock draft from slot 4 against the same nine bots (seed 9003); simulated win chance by category (illustration on the engine library; the policy searched a separate 4,200-week table)")
    F.save(fig, "fig_mock_compare")
    return dict(rb=rb, rp=rp, wb=float(evb.win_prob[me]), wp=float(evp.win_prob[me]), n_same=len(same))


# ---------------------------------------------------------------------------------------------------------------------
# text
# ---------------------------------------------------------------------------------------------------------------------
def build(ctx) -> str:
    S, proj = ctx.S, ctx.proj
    cal = Calibration.load(DATA / "calibration.json")
    lib = Library.load(DATA / "library_2026.npz")
    proj_l = proj[proj["player_id"].isin(lib.player_ids)].reset_index(drop=True)
    ctx_proj = ctx.proj
    ctx.proj = proj_l                                                       # engine helpers use the library's players
    weeks = load_weeks(DATA / "policy_weeks_2026.npz", DraftEngine(lib, proj_l, LeagueSettings(my_slot=1, **KW)).pid)

    fig_copula(cal)
    wd = fig_weekly_dist(ctx, lib)
    mm = fig_mean_match(ctx, lib)
    rz = realism_arrays()
    rl = fig_realism(ctx, rz)
    tune = fig_tuning(ctx)
    fresh = fig_fresh(ctx)
    pool = fig_candidate_pool(ctx)
    rs = fig_real_seasons(ctx)
    eng, rec = pick1_engine(ctx, lib)
    mc = mock_comparison(ctx, lib, weeks)
    ctx.proj = ctx_proj
    S.update(realism=rl, fresh=fresh, real=rs, tune=tune["table"], pool=pool["table"], mm=mm)

    par = pd.DataFrame({"statistic": list(COUNT_DIMS), "NB size r": [cal.dispersion[d] for d in COUNT_DIMS], "extra variance vs Poisson (1/r)": [1 / cal.dispersion[d] for d in COUNT_DIMS]})
    ex = wd.rename(columns={"zero_weeks": "weeks missed entirely", "p10": "10th percentile", "p90": "90th percentile"})
    cover = rl["cover"]
    sea = rl["sea_var"]
    rel = rl["rel"].copy()
    rel_t = pd.DataFrame({"predicted probability": [f"{a:.1f}-{b:.1f}" for a, b in zip(rel.lo, rel.hi)], "team-pair-weeks": rel.n, "mean predicted": rel.p, "actual win rate": rel.o})
    # engine vs board vs policy ordering at pick 1
    rec_i = rec.set_index("player_id")
    from scipy.stats import spearmanr
    rho_be = spearmanr(rec["static_rank"], rec["engine_rank"])[0]
    top_t = rec.sort_values("engine_rank").head(12)[["name", "static_rank", "engine_rank", "win_prob"]].copy()
    top_t["win_prob"] *= 100
    top_t.columns = ["player", "board rank", "engine rank", "win % if my first pick"]
    tun = tune["table"].set_index("setting")
    ft_rows = [
        ("old policy setting (40 draws, 6 candidates, guard 0.01)", fresh["old"], fresh["old_se"]),
        ("new policy setting (200 draws, 6 candidates, guard 0.005)", fresh["new"], fresh["new_se"]),
    ]
    rtab = pd.DataFrame([{"rule": {"engine_noguard": "engine, no guard", "engine": "engine, guard 0.01", "policy_old": "policy, old setting", "policy_new": "policy, new setting"}[k],
                          "seasons": v["n"], "mean gain (points)": v["mean"], "se across seasons": v["se"], "t": v["t"], "p (two-sided)": v["p"], "seasons positive": f'{v["pos"]} of {v["n"]}'} for k, v in rs.items()])
    per_year = pd.DataFrame({{"engine_noguard": "engine, no guard", "engine": "engine, guard 0.01", "policy_old": "policy, old setting", "policy_new": "policy, new setting"}[k]: pd.Series(v["by_year"]) for k, v in rs.items()})
    per_year.index = [SEASON(y) for y in per_year.index]
    per_year = per_year.reset_index().rename(columns={"index": "season"})
    rb, rp = mc["rb"].copy(), mc["rp"].copy()
    for r_ in (rb, rp):
        r_["avail"] = (r_["avail"] * 82).round(0)
    rb.columns = rp.columns = ["player", "pos", "board rank", "MPG", "GP", "PPG", "RPG", "APG"]
    cmp_roster = pd.concat([rb.add_prefix("board: "), rp.add_prefix("policy: ")], axis=1)[["board: player", "board: board rank", "policy: player", "policy: board rank"]]
    new_stat, old_stat = rs.get("policy_new"), rs.get("policy_old")

    return f"""
# 9. Phase 2: from an average to a distribution of weeks

## 9.1 Why simulate?

A league matchup is decided by *one week*, and a week is noisy. Two players with the same season average can have very different weekly spreads (a star who never misses versus one who misses 40% of weeks), and the *joint* behaviour of a roster's nine categories matters, because the objective is the probability of winning more than 4.5 of them. A table of season averages cannot answer "how often does this roster win five categories?", but a stochastic model of weeks can. The simulation turns every projected average into thousands of realistic simulated **matchup periods**, played on the real 2026-27 calendar.

## 9.2 The model, step by step, for one player in one period

1. **Pick a real matchup period** of the league calendar (21 of them, including the three playoff weeks); the player's team plays on its real days.
2. **Talent draw.** The projection is only a best guess, so perturb it: each statistic's per-game mean is multiplied by a lognormal factor `exp(σ·ε − σ²/2)` (mean one) and shooting percentages are shifted additively; the shocks `ε` are *correlated across statistics* (estimated from the backtest residuals, section 4.5) and `σ` depends on experience (and on draft-slot tier for rookies). This represents "how wrong could the projection be?".
3. **Availability.** With probability `c·(1 − avail)` the player misses the whole period (an injury run). Otherwise he plays each scheduled game with the probability that makes his expected games equal `avail × team games`. The constant `c` = **{cal.absence_c:.2f}** is the share of missed games that come in whole-period absences.
4. **For each game he plays, draw a stat line with a Gaussian copula.** Draw 11 correlated standard normals `Z ~ N(0, R)`, turn each into a uniform `U = Φ(Z)`, and push each through its own marginal inverse CDF: counts (2-point, 3-point and free-throw attempts, rebounds, assists, steals, blocks, turnovers) are **negative binomial** (variance exceeds the mean, because real counts are streaky), and makes are **binomial**(attempts, make rate), with the make rate's quantile driven by its own copula dimension (a Beta jitter adds streakiness where the data show it).
5. **Compute points and percentages from the shots**: `PTS = 2·FGM + 3PM + FTM`. They are never simulated directly.

*Why these distributions.* Counting statistics are non-negative integers with variance above their mean (over-dispersion), which the negative binomial captures with one extra parameter `r` (variance = μ + μ²/r; a small `r` means very streaky). Makes given attempts are naturally binomial. The Gaussian copula separates the *marginal* distribution of each stat from the *dependence* between stats, so the correlation structure can be estimated once from game logs and reused with any marginals.

{md_table(par, {"NB size r": "{:.1f}", "extra variance vs Poisson (1/r)": "{:.3f}"})}

Free-throw attempts (`r` = {cal.dispersion["fta"]:.1f}) are the most volatile count; turnovers (`r` = {cal.dispersion["tov"]:.0f}) are the most Poisson-like.

{ctx.fig("fig_copula", "Correlation matrix of the Gaussian copula: the dependence between a player's statistics within a game.")}

## 9.3 The calendar

Matchup periods are Monday-to-Sunday weeks with two exceptions that the code reproduces from the published schedule: the **NBA Cup knockout stretch** is merged with the week before it (week 7, 30 November to 13 December) and the **All-Star break** is a two-week matchup (week 17, 15 to 28 February). Week 18 ends the regular season on 7 March, and the three playoff weeks run 8 to 28 March. About 30 Cup games are not yet on the published schedule; by the owner's instruction they are ignored, so week 7 is simulated with only its published games and is the noisiest week.

{ctx.fig("fig_schedule", "Games per team in each of the 21 matchup periods (bar: mean; whiskers: minimum and maximum across teams).")}

Using the real schedule matters because value depends on *which days* players play: if four centers all have games on the same night only a limited number can start; a week in which a roster's stars share off-days leaves lineup slots empty. That interaction is invisible in a season-average projection.

## 9.4 What a simulated player looks like

{ctx.fig("fig_weekly_dist", "Distribution of simulated weekly points for three players; the spike at zero is a period missed entirely.")}

{md_table(ex, {"availability": "{:.0%}", "mean": "{:.0f}", "10th percentile": "{:.0f}", "90th percentile": "{:.0f}", "weeks missed entirely": "{:.0%}"})}

## 9.5 Validation, part 1: does the library reproduce the projections? (and a bug it caught)

Simulating is only useful if the simulated *means* equal the projection. For every player the library's games per period should equal `avail × team games`, and its per-game line while playing should equal the projection (the talent and count noise are mean-preserving by construction). We check this for all 614 players (`library_diagnostics.py`). Games and every non-shooting statistic matched to within 0.4%, but the shooting statistics did not: **points ran 1.8% high, threes made 4.2% high, FG% 2.0% and FT% 1.5% high.**

**Diagnosis.** The projection's make rates are *ratios of totals*, which already contain how shot attempts and make rate move together in real data. The copula and the talent draw then *re-created* that correlation on top, so players who shoot more also shoot better in the simulation, which inflates makes. We confirmed the cause by switching the correlation structure off: with the copula correlation set to the identity matrix the discrepancy in made field goals fell from +0.9% to +0.1% (3PM: +2.6% to +0.5%); with both the copula and the talent correlation switched off every ratio was 1.000 (made FG 0.999, 3PM 0.999, FT 1.000).

**Fix.** A short pilot simulation estimates a multiplier for each of the 2-point, 3-point and free-throw make rates, rescaled (volume-weighted) until the simulated makes equal the projected makes, and the full library is then built with those multipliers (`estimate_make_scale`). After the correction the shipped library (recomputed for this report) is within {max(abs(mm["after"][k] - 1) for k in ("PTS", "3PM", "REB", "AST", "STL", "BLK")) * 100:.1f}% of the projection for every counting statistic and within {max(abs(mm["after"][k] - 1) for k in ("FG%", "FT%")) * 100:.1f}% for the two percentages.

{ctx.fig("fig_mean_match", "Simulated minus projected per-game values, before and after the make-rate correction.")}

## 9.6 Validation, part 2: is the *spread* realistic? A test on four past seasons

Matching the mean is necessary but not sufficient: the pick policy depends on how much weekly totals vary, how often categories are won, and how strongly categories move together. To test the spread on seasons the simulation did not see, we replay 2022-23 to 2025-26:

1. **Rebuild the season from pre-season information only.** Projections (section 3-4 pipeline, including the backtest that picks the models) use only earlier seasons; the simulation library is built from those projections with that season's *real schedule*, 100 simulated draws per matchup period.
2. **Draft realistic rosters.** For each season we mock-draft six 10-team leagues (60 rosters of 13 players) by taking the best available player by the board, with noisy bots.
3. **Score every roster twice with the same daily-lineup rules**: on the simulated periods and on the periods that really happened (the actual box scores).
4. **Compare, week by week.** For each roster, week and category we find where the *real* total falls inside the 100 simulated totals for that same week. That position is the **probability integral transform (PIT)**, a number in [0, 1] that is uniform if the simulated distribution is calibrated. In a PIT histogram a flat profile means calibration, a U-shape means the real values land in the tails too often (the simulation is too narrow), and a hump means it is too wide. Converting PITs to normal scores `Φ⁻¹(PIT)` gives a variable with variance 1 under calibration.

*A benchmark we had to add.* To read the numbers we need the value a *perfectly calibrated* simulation would produce with this machinery. We obtain it by treating one simulated draw as if it were reality and placing it among the other 99 (leave-one-out PIT): its normal-score variance is **{rl["bench"]:.3f}** and its mean PIT 0.503, close to the ideal 1 and 0.5, which confirms the test itself is unbiased. (A first version of this test compared the correlation between categories with zero; that was wrong, because categories legitimately move together with games played, and the correct benchmark is the simulation's own correlation.)

{ctx.fig("fig_pit_hist", "PIT histograms by category for the real 2022-23 to 2025-26 team-weeks.")}

{ctx.fig("fig_pit_variance", "Variance of the normal-scored PIT by category and by season, against the simulation's own benchmark (orange line).")}

**Spread.** Pooled over {rl["n_tw"]:,} team-weeks the normal-score variance is **{rl["z_var"]:.2f}** against the benchmark {rl["bench"]:.2f}, so real weekly totals are about {(np.sqrt(rl["z_var"] / rl["bench"]) - 1) * 100:.0f}% wider than simulated; the central 50%, 80% and 90% simulated intervals contain {cover[0]:.0%}, {cover[1]:.0%} and {cover[2]:.0%} of real weeks (ideal 50/80/90). The shortfall is concentrated in points ({rl["cat_var"][0]:.2f}), threes ({rl["cat_var"][1]:.2f}) and blocks ({rl["cat_var"][5]:.2f}); free-throw percentage has the right spread ({rl["cat_var"][7]:.2f}) but a **bias**: its histogram tilts to the right (mean PIT {rl["ft_mean_pit"]:.2f}, mean normal score {rl["ft_mean_z"]:+.2f}), meaning that real rosters shoot free throws about {rl["ft_mean_z"]:.1f} standard deviations better than the simulation expects (the pooled free-throw projections are slightly conservative). It also changes with the season: the 2022-23 and 2023-24 seasons are calibrated ({sea[2022]:.2f}, {sea[2023]:.2f}) while 2024-25 and 2025-26 are noisier ({sea[2024]:.2f}, {sea[2025]:.2f}); the calibration file is fit on all seasons, and more weekly variability in recent seasons (more rest and load management) is a plausible but untested explanation. The mean PIT also shifts from season to season ({", ".join(f"{SEASON(y)}: {m:.2f}" for y, m in rl["mean_pit_season"].items())}): league-wide scoring levels differ from the projections' level by season, which cancels in head-to-head matchups because both teams shift together.

{ctx.fig("fig_corr_real_sim", "Cross-category correlation of the normal-scored PITs: real weeks, the simulation benchmark, and the difference.")}

**Dependence.** The average correlation between categories is **{rl["corr_real"]:+.2f}** in real weeks against **{rl["corr_sim"]:+.2f}** in the simulation, and the largest differences (blocks with other counting categories) are about 0.1. The copula and the shared talent shocks therefore reproduce how categories rise and fall together.

**Matchups.** The quantity the draft engine ultimately uses is the probability that team A beats team B in a week. For every pair of teams, in every week, we compare the simulated probability (from all 100 x 100 pairings of simulated draws) with who actually won:

{ctx.fig("fig_reliability", "Reliability diagram of simulated matchup probabilities against actual outcomes (left) and the distribution of the probabilities (right).")}

{md_table(rel_t, {"team-pair-weeks": "{:,.0f}", "mean predicted": "{:.3f}", "actual win rate": "{:.3f}"})}

A logistic regression of the outcome on the log-odds of the simulated probability has slope **{rl["slope"]:.2f}** (1 = perfectly calibrated; below 1 means too extreme). The Brier score is {rl["brier"]:.3f}, against {rl["brier0"]:.3f} for always predicting 50%, and teams the simulation favoured by more than 5 points won {rl["fav"]:.1%} of {rl["nfav"]:,} pairings. The probabilities are therefore informative and only slightly over-confident (about 5%); they are best read as a ranking of options rather than as exact probabilities. The extreme top bin (0.9-1.0, 70 team-pair-weeks) is the most over-confident.

{ctx.fig("fig_star_premium", "Actual and simulated win rate of the earlier-draft-slot team, by the slot gap between the two teams.")}

**One systematic pattern.** Early-slot (star-heavy) rosters beat the simulation's expectation by about 1 point at a gap of one slot and 3 to 6 points at larger gaps, so the real advantage of top-heavy rosters is somewhat larger than simulated (the samples at large gaps are small). We did not adjust for this; it suggests the simulation, if anything, undervalues stars slightly.

*An earlier lesson from the same test.* The first version of this check looked worse, with real weeks piling in the lowest and highest 10% buckets (15.8% and 15.2% in 2025-26). Widening the projection uncertainty did not help, which pointed at *availability* rather than talent: the whole-week absence share `c` had been estimated only from players with at least 25 games and only between their first and last appearance, silently excluding the long and season-ending injuries. Re-measured over every week of the season for every rotation player, `c` rose from 0.42 to 0.68, which reproduces the real 18-21% of zero-game weeks and roughly halved the distance from a flat histogram in every season.

# 10. Phase 3: from distributions to a draft decision

## 10.1 The objective

Because a weekly matchup is won by taking more than 4.5 of 9 categories, the engine maximises the **probability that my team wins a weekly matchup** against a random opponent, averaged over all simulated periods (the 18 regular-season and 3 playoff weeks equally weighted; there is no model of who qualifies for the playoffs). That differs usefully from "maximise total z-score": it rewards balance (five narrow category wins every week beat three huge ones), rewards depth (daily lineups mean bench players score when a starter is off), and treats a category that is certain to be lost as nearly free to ignore.

## 10.2 The decision loop

For each candidate pick the engine:

1. assumes I take him;
2. **rolls the rest of the draft forward**: every other team picks the best remaining player by a noisy preference order (ESPN's average draft position where available, otherwise our value rank, with noise that grows deeper in the draft), skipping any center that would be a fourth on that roster; my own later picks follow the board;
3. **plays all ten rosters through the simulated periods with daily lineups**: each day, in value order, every player with a game takes the first open slot he is eligible for (his dedicated position, then G or F, then UTIL); the rest sit, so bench players score whenever a starter has no game or is injured;
4. compares my weekly category totals with each opponent's (more than 4.5 categories wins; turnovers: fewer wins; FG% and FT% from summed makes and attempts);
5. repeats over two different sets of bot choices, using the same random numbers for every candidate so that *differences* between candidates are far less noisy than their absolute levels.

Every available player within board rank 200 is evaluated this way. At the first pick of slot 1:

{ctx.fig("fig_pick1", "Simulated matchup win probability of each top candidate for the first pick (left) and the chance of winning each category (right).")}

The engine's ordering agrees closely with the board (Spearman ρ = {rho_be:.2f} over the {len(rec)} candidates) but reorders it: it likes players whose categories are *complementary to the likely rest of the roster* and discounts those whose value is concentrated where many alternatives exist.

{md_table(top_t, {"board rank": "{:.0f}", "engine rank": "{:.0f}", "win % if my first pick": "{:.1f}"})}

## 10.3 The first attempt: did the engine beat "take the best player left"?

The only honest test of a draft tool is whether it would have helped in a season it had not seen. For each past season we rebuild everything as of that preseason, run mock drafts twice with identical bots (taking the engine's pick at each turn versus simply taking the best-ranked player left, both respecting the three-center cap), and score the two rosters on the season's real box scores. The first complete validation (2022-23 to 2025-26, 96 drafts) was disappointing: the engine's *own* simulation predicted a gain of +3.6 points of weekly win probability, but on real box scores the gain was **−0.8 ± 1.7** points, indistinguishable from zero. Two lessons followed.

* **The argmax of noisy estimates is partly luck.** When two candidates' simulated win probabilities differ by less than the noise in the estimate, the "best" one is often just the luckiest. The engine gained a **guard**: it departs from the board only when its pick is clearly better, taking the best-ranked player among candidates within 1 point of the best simulated probability. With the guard the same drafts scored **+2.5 ± 1.3** points (figure below, orange versus aqua).
* **Simulation-world gains over-state real gains.** Evaluation inside the simulation that chose the pick is optimistic (a form of the "winner's curse"), so every test from here on scores picks on a *held-out* simulation (a different random draw of the same season) and, ultimately, on real box scores.

## 10.4 A category-aware pick policy

The engine needs seconds per pick and compares candidates on noisy simulated weeks. A cheaper idea is to ask directly: *given the roster I would end up with, which candidate gives me the best chance of winning at least five categories?* The **category-aware policy** does this at every pick:

1. Take the board's top six available players (more candidates did worse, see below).
2. For each, finish the draft with noise-free bots and my later picks in board order.
3. Total every team's weekly production from a stored table of **4,200 simulated weeks** (200 draws of each of the 21 periods; one weekly-total table is all the policy needs), weighting each roster slot by how much of its production the daily lineup really uses (1.00 for the ten best players and 0.97 for the bench: a 13-man roster has only about five players with a game on a given day, so almost every game finds a slot).
4. Score my team against the nine opponents week by week, `P(win > 4.5 of 9 categories)`, and take the best candidate **only if it beats the board's top choice by more than 0.5 points** (the guard).

Because the policy re-decides at every pick from the current roster, it behaves like a *punting* strategy where that is worth it, without fixing a punt in advance. In one mock draft from slot 4 (the owner's slot), with identical bots:

{ctx.fig("fig_mock_compare", "Category win chances of the team drafted by the board and by the policy in one mock draft (slot 4, bot seed 9003).")}

{md_table(cmp_roster)}

The two teams share {mc["n_same"]} of 13 players; the simulated weekly win probability is {mc["wb"] * 100:.1f}% by the board and {mc["wp"] * 100:.1f}% with the policy (a single draw, scored on the engine's own 840-week library, so illustrative only).

## 10.5 Choosing the policy's settings without fooling ourselves

Three settings matter: the number of simulated draws the policy searches, how many candidates it considers, and the size of the guard. All were tuned on **development** mock drafts (bot seeds 5000-5011 in each of the ten slots, 120 drafts per setting) and scored on a **held-out** library, then confirmed on **fresh** seeds that were never used to choose anything.

{ctx.fig("fig_tuning", "Gain over drafting by the board for seven settings of the policy; blue bars are scored on a held-out library, orange on the library the policy searched.")}

* **More simulated draws help.** Going from 40 to 200 draws per week raises the held-out gain from {tun.iloc[0].held * 100:.1f} to {tun.iloc[1].held * 100:.1f} points with the same candidates and guard, and shrinks the in-sample-versus-held-out gap (the amount of noise-fitting) from {(tun.iloc[0].ins - tun.iloc[0].held) * 100:.1f} to {(tun.iloc[1].ins - tun.iloc[1].held) * 100:.1f} points.
* **With 200 draws a smaller guard helps** (0.005 beats 0.01 by about 0.7 points for 6 candidates).
* **More than about a dozen candidates hurts.** With 24 candidates the gain falls back toward the old level. Each additional candidate is another chance for a noisy estimate to beat the board by luck; this is the multiple-comparisons problem in miniature.

The best of seven settings on shared drafts is biased upward (selecting a winner from seven on the same data), so the confirmation on fresh seeds matters:

{ctx.fig("fig_fresh", "Confirmation on 150 fresh paired drafts: old versus new setting, and their difference by draft slot.")}

On the fresh drafts the new setting gains **{fresh["new"] * 100:+.2f} ± {fresh["new_se"] * 100:.2f}** points against **{fresh["old"] * 100:+.2f} ± {fresh["old_se"] * 100:.2f}** for the old setting, a paired difference of **{fresh["diff"] * 100:+.2f} ± {fresh["diff_se"] * 100:.2f}** points (better in {fresh["better"]:.0%} of drafts and in {fresh["pos_slots"]} of 10 draft slots). That is about half of the +1.2 points the same comparison showed on the development drafts: the other half was selection luck. We report the fresh number.

## 10.6 A negative result: considering every player up to rank 200 does not find better picks

The owner's preference was that every player up to rank 200 be considered at each pick. We tested whether evaluating all of them improves the *engine's* choices over a short list of 20:

{ctx.fig("fig_candidate_pool", "Held-out gain over board order for the engine with a short candidate list, with all players up to rank 200 (three guard sizes), and for the 40-draw policy.")}

With the 1-point guard the 200-wide search scores the same as the short list ({pool["table"].iloc[1].held * 100:+.2f} versus {pool["table"].iloc[0].held * 100:+.2f} points), and without the guard it changes about twice as many players and does *worse* ({pool["table"].iloc[2].held * 100:+.2f}). So the application shows every player up to rank 200 in its table, which costs nothing, but the recommended pick comes from a guarded rule. A caution about our own process: an interim reading of this experiment at 31 to 43 of 60 drafts (draft slots 1 to 8 only) had the policy at about twice the engine's gain; that lead vanished when slots 9 and 10 completed, and the final 60-draft result has the guarded engine and the 40-draw policy equal within noise. Interim results from a subset of conditions should not be trusted.

## 10.7 Validation on real seasons (never used for tuning)

The decisive test is again real box scores. For each of 2021-22 to 2025-26 we draft 40 mock leagues for each rule (policy rows use the same bots as the board team) and score the rosters on what really happened:

{ctx.fig("fig_real_seasons", "Gain over 'take the best player left' on actual box scores, by season (left) and averaged over seasons (right).")}

{md_table(per_year, {c: "{:+.1f}" for c in per_year.columns if c != "season"})}

{md_table(rtab, {"seasons": "{:.0f}", "mean gain (points)": "{:+.1f}", "se across seasons": "{:.1f}", "t": "{:.1f}", "p (two-sided)": "{:.3f}"})}

How much evidence is this? The 160-plus mock drafts per rule are *not* independent: all drafts in a season face the same real outcomes (an injury to a star hurts every roster that holds him), so the honest unit of evidence is the **season**, and there are five. Treating each season's mean gain as one observation, the new policy setting averages **{new_stat["mean"]:+.1f} points (standard error {new_stat["se"]:.1f}, t = {new_stat["t"]:.1f} on {new_stat["n"] - 1} degrees of freedom, p = {new_stat["p"]:.3f})** and is positive in {new_stat["pos"]} of {new_stat["n"]} seasons; the old setting averages {old_stat["mean"]:+.1f}. The engine's guard result is similar ({rs["engine"]["mean"]:+.1f}), and removing the guard turns the engine's gain into roughly zero. These settings were chosen on *simulated* drafts of 2026-27, not on these real seasons, so the real-season results are out of sample for the settings (the engine's 1-point guard is the exception: it was chosen after viewing the 2022-25 results, so only 2021-22 is clean for it). Five seasons are enough to rule out a large harm, not to measure a small gain precisely, and the calibration of the simulation (section 9) was fitted on all seasons, so the replay is not strictly out of sample.

**Which rule is used?** The application's "Best pick" is the category-aware policy (a sidebar switch selects the engine's guarded pick instead); both rules are statistically equivalent in our tests, the policy being about 350 times faster. The table of win probabilities for every player up to rank 200 is shown next to it.
"""
