"""Section 14: next steps and areas to explore (role and usage, returning from injury, roster change, positional depth)."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scripts.final_report import roles as R
from scripts.final_report.common import *  # noqa: F403

GROUPS = ["healthy (control)", "partial season", "missed a season"]
GCOL = {"healthy (control)": C["axis"], "partial season": C["blue"], "missed a season": C["orange"]}
GLAB = {"healthy (control)": "healthy\ncontrol", "partial season": "partial\nseason", "missed a season": "missed a\nseason"}


def pm(x: float, se: float, d: int = 2) -> str:
    """'+0.33 (SE 0.09)' style text."""
    return f"{x:+.{d}f} (SE {se:.{d}f})"


# ---------------------------------------------------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------------------------------------------------
def fig_returners(o: dict) -> None:
    rpm, ramp = o["rp_means"], o["ramp"]
    fig, axes = plt.subplots(1, 3, figsize=(11.6, 4.5), gridspec_kw={"width_ratios": [1, 1, 1.2]})
    for ax, metric, ttl, fmt in ((axes[0], "mpg", "Change in minutes per game", "{:+.1f}"), (axes[1], "usg", "Change in usage (USG% points)", "{:+.1f}")):
        for i, g in enumerate(GROUPS):
            r = rpm[(rpm.group == g) & (rpm.metric == metric)].iloc[0]
            ax.bar(i, r["mean"], color=GCOL[g], width=0.62)
            ax.errorbar(i, r["mean"], yerr=1.96 * r["se"], fmt="none", ecolor=C["ink2"], capsize=4, elinewidth=1.3)
            up = r["mean"] >= 0
            ax.text(i, r["mean"] + (1 if up else -1) * (1.96 * r["se"] + 0.12), fmt.format(r["mean"]), ha="center", va="bottom" if up else "top", fontsize=9, color=C["ink2"])
        ns = [int(rpm[(rpm.group == g) & (rpm.metric == metric)]["n"].iloc[0]) for g in GROUPS]
        ax.set_xticks(range(3), [f"{GLAB[g]}\n(n={n})" for g, n in zip(GROUPS, ns)], fontsize=8.6)
        ax.axhline(0, color=C["ink2"], lw=1); ax.set_title(ttl, fontsize=10, loc="left"); ax.grid(axis="x", visible=False)
        lo, hi = ax.get_ylim(); ax.set_ylim(lo - 0.6, hi + 0.4)
    ax = axes[2]
    mins = ramp[ramp.kind == "minutes"].reset_index(drop=True); pos = ramp[ramp.kind == "possessions per 36"].reset_index(drop=True)
    x = np.arange(len(mins))
    ax.bar(x, mins["mean"] * 100, color=C["blue"], width=0.55)
    ax.errorbar(x, mins["mean"] * 100, yerr=1.96 * mins["se"] * 100, fmt="none", ecolor=C["ink2"], capsize=4, elinewidth=1.3)
    for xi, r in zip(x, mins.itertuples()):
        ax.text(xi, 42, f"{r.mean * 100:.0f}%", ha="center", color="white", fontsize=11, fontweight="bold")
    ax.errorbar(x + 0.0, pos["mean"] * 100, yerr=1.96 * pos["se"] * 100, fmt="o", color=C["orange"], ecolor=C["orange"], capsize=3, ms=7, mec="white", mew=1.2, zorder=5)
    ax.axhline(100, color=C["ink2"], lw=1, ls=(0, (4, 3)))
    ax.set_xticks(x, [w.replace("games ", "games\n") for w in mins["window"]]); ax.set_ylim(0, 118); ax.set_ylabel("% of his own steady level (games 21-40)")
    ax.set_title(f"Ramp-up after a layoff of 15+ games (n = {int(mins['n'].iloc[0])} returns)", fontsize=10, loc="left"); ax.grid(axis="x", visible=False)
    F.headline(fig, "After a layoff the minutes come back in stages; the per-minute role mostly survives, except after a part-season",
               "Rotation players (24+ minutes) in the 2015-16 to 2025-26 box scores; bars show means with 95% intervals. Left: change from the last known season. Right: blue bars = minutes, orange dots = possessions used per 36 minutes")
    F.legend_top(fig, [F.swatch(GCOL[g], g) for g in GROUPS] + [F.swatch(C["orange"], "possessions per 36 (ramp-up panel)", "dot")], ncol=4)
    F.save(fig, "fig_next_returners")


def fig_roster(o: dict) -> dict:
    d = o["stayers"].copy()
    d["dep"] = np.select([(d["vall"] == 0), (d["n_hi"] == 0), (d["n_hi"] == 1)], ["no rotation-level\ndeparture", "only lower-usage\ndepartures", "one high-usage\ndeparture"], "two or more\nhigh-usage departures")
    order = ["no rotation-level\ndeparture", "only lower-usage\ndepartures", "one high-usage\ndeparture", "two or more\nhigh-usage departures"]
    g = d.groupby("dep")["d_usg"].agg(["mean", "std", "size"]).reindex(order)
    g["se"] = g["std"] / np.sqrt(g["size"])
    tier = o["tier_effect"]
    allr = R.regress(d, "d_usg", ["lost_hi"]).loc["lost_hi"]
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.4), gridspec_kw={"width_ratios": [1.15, 1]})
    ax = axes[0]
    cols = [C["axis"], C["axis"], C["blue"], C["blue"]]
    ax.bar(range(4), g["mean"], color=cols, width=0.62)
    ax.errorbar(range(4), g["mean"], yerr=1.96 * g["se"], fmt="none", ecolor=C["ink2"], capsize=4, elinewidth=1.3)
    for i, (m, se) in enumerate(zip(g["mean"], g["se"])):
        ax.text(i, m + (1 if m >= 0 else -1) * (1.96 * se + 0.06), f"{m:+.2f}", ha="center", va="bottom" if m >= 0 else "top", fontsize=9, color=C["ink2"])
    ax.set_xticks(range(4), [f"{k}\n(n={int(n)})" for k, n in zip(order, g["size"])], fontsize=8.2)
    ax.axhline(0, color=C["ink2"], lw=1); ax.set_ylabel("average change in his usage, USG% points"); ax.grid(axis="x", visible=False)
    ax.set_ylim(-1.0, 1.35)
    ax.set_title("What left the team, and what happened to the players who stayed", fontsize=10, loc="left", pad=10)
    ax = axes[1]
    rows = [("all stayers", allr["coef"], allr["se"], int(o["stayers"].shape[0]))] + [(r.tier, r.coef, r.se, r.n) for r in tier.itertuples()]
    y = np.arange(len(rows))[::-1]
    for yi, (lab_, c_, s_, n_) in zip(y, rows):
        ax.errorbar(c_, yi, xerr=1.96 * s_, fmt="o", color=C["blue"], ecolor=C["blue"], capsize=4, ms=8, mec="white", mew=1.2)
        ax.text(c_, yi + 0.28, f"{c_:+.2f}", ha="center", fontsize=8.6, color=C["ink2"])
    ax.axvline(0, color=C["ink2"], lw=1)
    ax.set_yticks(y, [f"{r[0]} (n={r[3]})" for r in rows], fontsize=8.4); ax.set_xlabel("extra usage points when a high-usage player left\n(adjusted, 95% interval)")
    ax.set_ylim(-0.6, len(rows) - 0.3)
    ax.set_title("By the player's own usage rank on the team", fontsize=10, loc="left", pad=10); ax.grid(axis="y", visible=False)
    F.headline(fig, "Players who stay pick up usage when a high-usage teammate leaves, but not when only role players do",
               f"{len(o['stayers']):,} rotation players who stayed with their team (2016-17 to 2025-26); a high-usage player has USG% of {R.HI_USG:.0f} or more. Right panel adjusts for age, last year's minutes and usage; intervals clustered by team-season")
    F.save(fig, "fig_next_roster")
    return dict(groups=g, all=allr)


def fig_starout(o: dict) -> None:
    s = o["so_sum"]
    buckets = ["next four by minutes", "rotation (5th-8th)", "deep bench (9th and below)", "all teammates"]
    fig, ax = F.new(9.0, 4.4)
    x = np.arange(len(buckets)); w = 0.36
    for dx, kind, col in ((-w / 2, "minutes", C["blue"]), (w / 2, "possessions", C["orange"])):
        r = s[s.kind == kind].set_index("bucket").loc[buckets]
        ax.bar(x + dx, r["mean"] * 100, w, color=col, label=f"share of his {kind}")
        ax.errorbar(x + dx, r["mean"] * 100, yerr=1.96 * r["se"] * 100, fmt="none", ecolor=C["ink2"], capsize=3, elinewidth=1.2)
        for xi, m, se in zip(x + dx, r["mean"] * 100, r["se"] * 100):
            ax.text(xi, m + (1 if m >= 0 else -1) * (1.96 * se + 2), f"{m:.0f}%", ha="center", va="bottom" if m >= 0 else "top", fontsize=9, color=C["ink2"])
    ax.axhline(0, color=C["ink2"], lw=1); ax.set_xticks(x, [b.replace(" (", "\n(") if "(" in b else b for b in buckets], fontsize=9)
    ax.set_ylabel("% of the star's minutes or possessions picked up"); ax.grid(axis="x", visible=False); ax.set_ylim(-35, 100)
    F.headline(fig, "When a team's top-usage player is out, the bench absorbs most of his minutes and possessions",
               f"{o['so_seasons']} team-seasons (median {o['so_games_out']:.0f} games missed); changes in games he missed versus games he played, for teammates who stayed all season, counting a missed game as zero")
    F.legend_top(fig, *ax.get_legend_handles_labels())
    F.save(fig, "fig_next_starout")


def fig_position(o: dict) -> None:
    pv = o["pos_vacated"]
    pv = pv[pv.outcome == "d_mpg"]
    m = o["crowd_frame"]
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.5), gridspec_kw={"width_ratios": [1, 1.1]})
    ax = axes[0]
    labs = []
    y = 0
    for grp, nm in (("C", "Centers"), ("F", "Forwards"), ("G", "Guards")):
        for term, col in (("vacated, same position", C["violet"]), ("vacated, other positions", C["muted"])):
            r = pv[(pv.grp == grp) & (pv.term == term)].iloc[0]
            ax.errorbar(r.coef, y, xerr=1.96 * r.se, fmt="o", color=col, ecolor=col, capsize=4, ms=8, mec="white", mew=1.2)
            ax.text(r.coef + 1.96 * r.se + 0.05, y, f"{r.coef:+.2f}", ha="left", va="center", fontsize=8.5, color=C["ink2"])
            labs.append((y, f"{nm}: {term.replace('vacated, ', '')} (n={int(r.n)})"))
            y += 1
        y += 0.6
    ax.axvline(0, color=C["ink2"], lw=1); ax.set_yticks([a for a, _ in labs], [b for _, b in labs], fontsize=8.2); ax.invert_yaxis()
    ax.set_xlim(-0.8, 2.6)
    ax.set_xlabel("change in minutes per game per 10 points of\nteam possessions vacated (adjusted, 95% interval)"); ax.grid(axis="y", visible=False)
    ax.set_title("Stayers: who left matters most for centers", fontsize=10, loc="left", pad=10)
    ax = axes[1]
    for grp, nm, col, nb in (("G", "Guards", C["blue"], 4), ("F", "Forwards", C["aqua"], 4), ("C", "Centers", C["orange"], 3)):
        s = m[m.grp == grp]
        b = A.binned_mean(s["m_stay_same36"], s["mpg_res"], bins=nb)
        ax.errorbar(b["x"], b["mean"], yerr=1.96 * b["se"], color=col, marker="o", ms=6, lw=2, capsize=3, label=f"{nm} (n={len(s)})")
    ax.axhline(0, color=C["ink2"], lw=1)
    ax.set_xlabel("same-position teammates who stayed: last year's role minutes\n(in full-time-player equivalents of 36 minutes)")
    ax.set_ylabel("actual minus projected minutes per game")
    ax.set_title("Out of sample: more competition, fewer minutes than projected", fontsize=10, loc="left", pad=10); ax.grid(axis="x", visible=False)
    F.headline(fig, "Positional crowding takes minutes, especially from guards and centers, beyond what the projection already accounts for",
               "Left: stayers' change in minutes against possessions vacated by same-position (violet) and other-position (grey) departures. Right (legend): walk-forward minutes residuals (2021-22 to 2025-26) by same-position depth, in quantile bins with 95% intervals")
    F.legend_top(fig, *ax.get_legend_handles_labels())
    F.save(fig, "fig_next_position")


# ---------------------------------------------------------------------------------------------------------------------
# text
# ---------------------------------------------------------------------------------------------------------------------
def build(ctx) -> str:
    S = ctx.S
    resid = A.residual_frame(ctx.oos, ctx.st, ctx.apg, ctx.names)
    o = R.run_all(ctx, resid)
    fig_returners(o)
    rs = fig_roster(o)
    fig_starout(o)
    fig_position(o)

    # ---- A: returners ----
    rpm, adj, rp = o["rp_means"], o["rp_adj"].set_index("metric"), o["rp"]
    n = rp.groupby("group").size()
    get = lambda g, m: rpm[(rpm.group == g) & (rpm.metric == m)].iloc[0]
    tab = pd.DataFrame([{"group": g, "players": int(n[g]), "age": get(g, "mpg")["age"], "last known MPG": get(g, "mpg")["last_mpg"],
                         "Δ minutes": pm(get(g, "mpg")["mean"], get(g, "mpg")["se"], 1), "Δ usage (points)": pm(get(g, "usg")["mean"], get(g, "usg")["se"], 1),
                         "Δ points per 36": pm(get(g, "pts36")["mean"], get(g, "pts36")["se"], 1)} for g in GROUPS])
    lab = {"mpg": "minutes per game", "usg": "usage (USG% points)", "pts36": "points per 36", "ast36": "assists per 36", "tov36": "turnovers per 36", "ts": "true shooting"}
    atab = pd.DataFrame([{"change relative to the healthy control": lab[m], "partial season": pm(adj.loc[m, "partial"], adj.loc[m, "partial_se"], 3 if m == "ts" else 2),
                          "missed a season": pm(adj.loc[m, "missed"], adj.loc[m, "missed_se"], 3 if m == "ts" else 2)} for m in lab])
    ramp = o["ramp"]; ru = o["ru"]
    rm = ramp[ramp.kind == "minutes"].set_index("window")["mean"]
    ru_u = ramp[ramp.kind == "possessions per 36"].set_index("window")["mean"]
    sev = ru.assign(sev=pd.cut(ru["gap"], [14, 20, 30, 100], labels=["15-20", "21-30", "31+"])).groupby("sev", observed=True)["m_1_3"].agg(["mean", "size"])
    pre = o["ramp_pre"]
    ss = ru.groupby("season_start")["m_1_3"].mean()

    # ---- B: roster change ----
    ra, rh = o["reg_all"], o["reg_hi"]
    tier = o["tier_effect"].set_index("tier")
    gr = rs["groups"]
    # ---- E: star out ----
    so = o["so_sum"].set_index(["bucket", "kind"])
    sg = lambda b, k: so.loc[(b, k), "mean"] * 100
    # ---- C: position ----
    cr = o["crowd"].set_index(["grp", "outcome"])
    pv = o["pos_vacated"]
    pvc = pv[(pv.grp == "C") & (pv.outcome == "d_mpg") & (pv.term == "vacated, same position")].iloc[0]
    pvg = pv[(pv.grp == "G") & (pv.outcome == "d_mpg") & (pv.term == "vacated, same position")].iloc[0]
    pvf = pv[(pv.grp == "F") & (pv.outcome == "d_mpg") & (pv.term == "vacated, same position")].iloc[0]
    crt = o["crowd"][o["crowd"].grp != "all"].copy()
    crt = crt[crt.outcome.isin(["mpg_res", "pts_res", "reb_res", "ast_res", "blk_res"])]
    names = {"mpg_res": "minutes", "pts_res": "points", "reb_res": "rebounds", "ast_res": "assists", "blk_res": "blocks"}
    ctab = crt.pivot(index="grp", columns="outcome", values="same").reindex(["G", "F", "C"])
    ctab_se = crt.pivot(index="grp", columns="outcome", values="same_se").reindex(["G", "F", "C"])
    ctab_n = crt.groupby("grp")["n"].first().reindex(["G", "F", "C"])
    ct = pd.DataFrame({"position group": ["Guards", "Forwards", "Centers"], "n": ctab_n.to_numpy()})
    for c in ("mpg_res", "pts_res", "reb_res", "ast_res", "blk_res"):
        ct[names[c]] = [pm(ctab.loc[g, c], ctab_se.loc[g, c], 2) for g in ("G", "F", "C")]
    c_all = cr.loc[("all", "mpg_res")]
    xsd = cr.loc[("all", "mpg_res"), "x_sd"]

    gbin = A.binned_mean(o["crowd_frame"].query("grp == 'G'")["m_stay_same36"], o["crowd_frame"].query("grp == 'G'")["mpg_res"], bins=4)
    g_gap = float(gbin["mean"].iloc[0] - gbin["mean"].iloc[-1])
    rows_i = [gr.iloc[i]["mean"] for i in range(4)]                       # none / only lower-usage / one high-usage / two or more
    sp_lo, sp_hi = S["spearman_by_season"]

    return f"""
# 14. Next steps: areas to explore

This report treats a player's *role* (how many minutes he plays and how many possessions he uses) as something to be inferred from his past minutes, his age, and a handful of team-context features. That choice is where most of the remaining error sits: in section 4.5 the misses in points, rebounds, assists, threes and turnovers were all strongly correlated with the miss in minutes (0.6 to 0.8), and the largest disagreements with the market in section 11 were about playing time. The most promising next steps therefore concern **role**. This section takes the three questions the project owner raised, gives a first look at what the data already say about each, and then lists further recommendations, a protocol for testing them, and some questions.

**How to read the first-look results.** They are exploratory descriptive analyses of the same 2015-16 to 2025-26 box scores, run for this section; none of them changes the 2026-27 board. *Usage* is the standard box-score estimate USG% = 100 × (FGA + 0.44 FTA + TOV) × (team minutes ÷ 5) ÷ (player minutes × team possessions used), where the team's possessions used are FGA + 0.44 FTA + TOV. Position groups are the three buckets (G, F, C) of the API's position label, so every positional result is approximate. Standard errors (SE) are clustered by team-season (or by player where noted), and the intervals shown in the figures are 95%. Many things were looked at, none was corrected for multiple comparisons, and the observational design cannot separate cause from selection: treat each result as a hypothesis worth a proper test (section 14.5), not as a finding.

## 14.1 Question 1: role and usage after a layoff

**What the model does now.** For a player with no games in the previous season, availability is a 50/50 blend of the model and the recency baseline, and minutes are cut by what the model has learned from blank last-season slots (for former rotation stars, 33.1 to 26.5 minutes; section 5.4). Per-minute rates are kept, so counting statistics scale with minutes. For Haliburton the owner's 29.8-minute override followed exactly this logic (section 7.2).

**Questions.** When a rotation player comes back, does his *role* change (fewer minutes, fewer possessions per minute) or only his *exposure*? How long does the ramp-up last? Does the answer depend on how long he was out?

**First look.** Three groups of rotation players (at least 24 minutes per game in the last season in which they played) who then played at least 500 minutes: players who missed a whole season, players who played less than half of the team's games (at least 8) the season before, and healthy controls (at least 80% of games). Each is compared with his own last known season; the controls show the ordinary regression of a high-minutes group toward the average.

{md_table(tab, {"age": "{:.1f}", "last known MPG": "{:.1f}"})}

Adjusting for age, age squared and last known minutes, and measuring each returner's change *relative to the healthy control*:

{md_table(atab)}

{ctx.fig("fig_next_returners", "Left and centre: change in minutes and usage from the last known season for rotation players who missed a season, played part of one, or stayed healthy. Right: minutes (bars) and possessions per 36 (dots) in the first games after a layoff of 15 or more team games, as a share of the player's own steady level in games 21-40.")}

* **After a missed season the role shrinks through minutes, not through usage.** The {int(n["missed a season"])} rotation players who missed a whole season lost {abs(get("missed a season", "mpg")["mean"]):.1f} minutes per game ({abs(adj.loc["mpg", "missed"]):.1f} more than controls after adjustment, SE {adj.loc["mpg", "missed_se"]:.1f}), consistent with the cut already built into the model (section 5.4). Their usage changed by {adj.loc["usg", "missed"]:+.1f} points relative to controls (SE {adj.loc["usg", "missed_se"]:.1f}) and their points per 36 by {adj.loc["pts36", "missed"]:+.1f} (SE {adj.loc["pts36", "missed_se"]:.1f}), neither distinguishable from zero, which supports projecting a returner at his old per-minute rates with fewer minutes. The sample is small ({int(n["missed a season"])}), so this is a consistency check, not a measurement.
* **After part of a season the usage rate also falls.** The {int(n["partial season"])} players who played part of a season lost {abs(adj.loc["mpg", "partial"]):.1f} more minutes than controls (SE {adj.loc["mpg", "partial_se"]:.1f}) and {abs(adj.loc["usg", "partial"]):.1f} more points of usage (SE {adj.loc["usg", "partial_se"]:.2f}); per-36 turnovers also fall ({pm(adj.loc["tov36", "partial"], adj.loc["tov36", "partial_se"])}). Plausible reasons (teammates took on their shots while they were out; a more cautious return; or an injury that was already affecting them before they left) cannot be told apart with box scores.
* **The ramp-up lasts about ten games and is the same whatever the layoff.** Across {int(ramp["n"].iloc[0])} returns after at least 15 missed team games, minutes are {rm["games 1-3"] * 100:.0f}% of the player's steady level in his first three games, {rm["games 4-10"] * 100:.0f}% in games 4-10 and {rm["games 11-20"] * 100:.0f}% in games 11-20, while possessions per 36 minutes stay at {ru_u["games 1-3"] * 100:.0f}%, {ru_u["games 4-10"] * 100:.0f}% and {ru_u["games 11-20"] * 100:.0f}%. The first-three-game figure is {sev.loc["15-20", "mean"] * 100:.0f}% after layoffs of 15-20 games, {sev.loc["21-30", "mean"] * 100:.0f}% after 21-30 and {sev.loc["31+", "mean"] * 100:.0f}% after 31 or more, and {ss.get(1, np.nan) * 100:.0f}% for returns at the start of a season against {ss.get(0, np.nan) * 100:.0f}% mid-season. Altogether a return costs about **{o["ramp_lost_games"]:.1f} games' worth of minutes**. His steady level equals his pre-layoff minutes (median ratio {pre.median():.2f}, n = {len(pre)}), so the loss is temporary for those who get back.

**What it suggests.** (i) Keep per-minute rates for returners and cut minutes (as now); (ii) count about two fewer games of minutes for any player projected to start the season out or to return mid-season, a small adjustment (about 3% of a 65-game season) that is not modelled explicitly today and is only implicit in the average minutes of past players who missed games; (iii) check whether the minutes uncertainty for returners in the simulation is wide enough, since the talent draw depends on experience only (section 9.2); (iv) look for role signals beyond minutes for players who return after a partial season.

**Proposed study.** Add return-specific features (days since last game, number of games missed, minutes in the first games back when available) and test them in the walk-forward backtest on returners only. The binding constraint is sample size: {int(n["missed a season"])} full-season returners and {int(n["partial season"])} part-season returners since 2016-17. Downloading the five seasons before 2015-16 would add about half again (this needs the project owner's yes). The most valuable missing information is the *type* of injury and the stated timeline (the database's `player_absence` table is empty); the ESPN injury report gives only the current designation.

## 14.2 Question 2: how roles evolve as the roster changes

**What the model does now.** The team-context features (section 6) give each player the minutes and possessions *vacated* by departed teammates and *incoming* from arrivals, overall and for his own position group. The backtest showed that they help most for players who changed teams (points error -7%) and leave stayers unchanged (within 0.7%).

**Questions.** Does a stayer's usage and minutes respond to what left? Is it the *amount* of vacated role or the *type* of player who left? And inside a season, when the team's top-usage player is out, who gets his minutes and possessions?

**First look (offseason changes).** {len(o["stayers"]):,} rotation players (15+ minutes, 30+ games) who stayed with their team, {o["stayers"]["tyear"].nunique()} team-seasons. Outcome: change in usage and minutes from last season. Total vacated possessions (as a share of last year's team possessions) do not predict usage ({pm(ra["d_usg"].loc["vac10", "coef"], ra["d_usg"].loc["vac10", "se"], 3)} points per 10 points of vacated share) and the minutes coefficient has the opposite of the expected sign ({pm(ra["d_mpg"].loc["vac10", "coef"], ra["d_mpg"].loc["vac10", "se"], 3)}). Splitting the departures by *type* changes the picture: possessions vacated by high-usage players ({R.HI_USG:.0f}+ USG%, about {o["hi_share_of_rotation"] * 100:.0f}% of rotation players) raise a stayer's usage by {pm(rh["d_usg"].loc["vhi10", "coef"], rh["d_usg"].loc["vhi10", "se"])} points per 10 points of team possessions, and his points per 36 by {pm(rh["d_pts36"].loc["vhi10", "coef"], rh["d_pts36"].loc["vhi10", "se"])}, while possessions vacated by other rotation-level players have no positive effect ({pm(rh["d_usg"].loc["voth10", "coef"], rh["d_usg"].loc["voth10", "se"])}). The usage of players whose teams lost a high-usage player ({o["share_lost_hi"] * 100:.0f}% of stayers) rose on average by {rows_i[2]:.2f} points (one such departure) and {rows_i[3]:.2f} points (two or more), against {rows_i[0]:+.2f} when no rotation-level player left and {rows_i[1]:+.2f} when only lower-usage players left. Minutes barely move ({pm(rh["d_mpg"].loc["vhi10", "coef"], rh["d_mpg"].loc["vhi10", "se"])} per 10 points): the extra shots come from a larger share of the same minutes.

{ctx.fig("fig_next_roster", "Left: average change in a stayer's usage by what left his team. Right: the adjusted extra usage when a high-usage player left, by the stayer's own usage rank.")}

The effect is not concentrated in the team's top option: the adjusted gain is {pm(tier.loc["top usage on the team", "coef"], tier.loc["top usage on the team", "se"])} points for the highest-usage player, {pm(tier.loc["2nd-3rd", "coef"], tier.loc["2nd-3rd", "se"])} for the second and third, {pm(tier.loc["4th-6th", "coef"], tier.loc["4th-6th", "se"])} for the fourth to sixth and {pm(tier.loc["7th and below", "coef"], tier.loc["7th and below", "se"])} below that.

**First look (inside a season).** For each of {o["so_seasons"]} team-seasons we take the team's top-usage player (who played for that team all year) and compare his teammates' minutes and possessions in the games he missed (median {o["so_games_out"]:.0f} games) with the games he played, counting a missed game as zero so that the changes add up. Teammates pick up {sg("all teammates", "minutes"):.0f}% of his minutes and {sg("all teammates", "possessions"):.0f}% of his possessions (the rest goes to players who arrived or left, or to players absent in the same games). The next four players by minutes pick up {sg("next four by minutes", "possessions"):.0f}% of the possessions but lose {abs(sg("next four by minutes", "minutes")):.0f}% of the star's minutes in total because the other starters are also more likely to be out in those games; the rotation players (5th to 8th) pick up {sg("rotation (5th-8th)", "possessions"):.0f}% of the possessions and the deep bench {sg("deep bench (9th and below)", "possessions"):.0f}%.

{ctx.fig("fig_next_starout", "Share of the star's minutes and possessions picked up by teammate group when the team's top-usage player is out.")}

**What it suggests.** Role is not a fixed quantity that a roster change merely rescales: *who* leaves matters more than *how much* leaves. The current context features add up vacated and incoming role without regard to usage; a usage-weighted version (for example possessions vacated by players above a usage threshold, and the incoming usage of arrivals) is a cheap candidate that uses data already in the database. Inside a season, absences appear to cluster (the other starters lost minutes in the games the star missed), so the independent absences in the simulation (section 9.2) may understate how often several starters miss the same week, and the redistribution of a star's possessions to the bench is something a pure season-average projection cannot show.

**Proposed study.** (1) Add usage-weighted vacated and incoming features and compare walk-forward error for stayers and movers separately; (2) model usage × minutes × efficiency as separate pieces instead of per-36 rates alone; (3) add team-level random effects (a hierarchical model) so that each team's rotation habits are partially pooled; (4) the database has empty `coach` and `team_era` tables: coach changes are a likely driver of rotation changes and would need a data source; (5) test a team-level absence correlation in the simulation (section 14.4).

## 14.3 Question 3: how positional depth affects output

**What the model does now.** The context features include vacated and incoming role *at the player's own position group*, and a lineup rule with a three-center cap in the draft simulation. They do not include how crowded the position is with teammates who stayed.

**Questions.** Do the departure or arrival of players at one's own position matter more than elsewhere? Beyond what the projection already knows, do players with more same-position competition play less or produce less?

**First look.** (a) *Stayers.* For the same {len(o["stayers"]):,} stayers, the minutes change per 10 points of team possessions vacated by same-position departures is {pm(pvc["coef"], pvc["se"])} for centers (n = {int(pvc["n"])}), against {pm(pvg["coef"], pvg["se"])} for guards and {pm(pvf["coef"], pvf["se"])} for forwards; possessions vacated at *other* positions do not help centers ({pm(pv[(pv.grp == "C") & (pv.outcome == "d_mpg") & (pv.term == "vacated, other positions")].iloc[0]["coef"], pv[(pv.grp == "C") & (pv.outcome == "d_mpg") & (pv.term == "vacated, other positions")].iloc[0]["se"])}). (b) *Residuals.* For the {int(c_all["n"])} out-of-sample player-seasons of the backtest, we regressed what the minutes projection missed (actual minus projected) on the role minutes last season of same-position teammates who stayed, in full-time-player equivalents of 36 minutes (standard deviation {xsd:.1f}). Each additional same-position teammate of this kind is associated with {pm(c_all["same"], c_all["same_se"])} minutes per game (R² {c_all["r2"] * 100:.0f}%); for other-position teammates the association is smaller ({pm(c_all["other"], c_all["other_se"])}).

{ctx.fig("fig_next_position", "Left: change in a stayer's minutes per 10 points of team possessions vacated by same-position versus other-position departures, by position group. Right: what the walk-forward minutes projection missed, by the same-position depth of the team.")}

By position group, the change in each quantity per additional full-time same-position teammate (each cell is the coefficient and its standard error; minutes was the outcome looked at first, the others are descriptive):

{md_table(ct)}

* **Guards** projected on a deep team lose about half a minute per extra guard-sized role and about {abs(ctab.loc["G", "ast_res"]):.2f} assists per game; the gap between the least and the most crowded quarter of guards is {g_gap:.1f} minutes per game (right panel).
* **Centers** have the largest but also the noisiest effect (n = {int(ctab_n["C"])}): {abs(ctab.loc["C", "mpg_res"]):.1f} minutes, {abs(ctab.loc["C", "pts_res"]):.1f} points and {abs(ctab.loc["C", "reb_res"]):.1f} rebounds per game per additional full-time center, and the same pattern in the stayers' regression. A team seldom has two full-time centers, so the sample of crowded situations is small.
* **Forwards** show no clear relationship ({pm(ctab.loc["F", "mpg_res"], ctab_se.loc["F", "mpg_res"])} minutes). The forward label is the broadest of the three buckets and the least informative.

**What it suggests.** There is a modest, interpretable signal that the projection misses: competition at one's own position lowers minutes and the counting statistics that follow from them, and vacated positional minutes lift centers. The effect is small against the typical minutes miss of {c_all["sd"]:.1f}, so it would improve rankings only slightly, but it is cheap to add and bears directly on a league with a three-center cap. All of it rests on three coarse position buckets; with exact ESPN eligibility (blocked from the development machine) a sharper definition (PG, SG, SF, PF, C, and multi-position players) could be tested.

**Proposed study.** Add a positional-crowding feature set (same-position and other-position role minutes of stayers and of arrivals) to the minutes and attempts models, run the walk-forward backtest with and without it, and use the ranking correlation and the real-season draft validation of section 10.7 as the final tests. Because positions are fluid, a data-driven alternative is to describe each player by his own profile (rebounding, assists, rim protection per 36) and measure crowding by *similarity of profile* instead of position label.

## 14.4 Other recommendations

| Priority | Idea | Why | Data needed | First test |
|---|---|---|---|---|
| 1 | Usage-weighted vacated and incoming features | Who leaves matters more than how much (14.2) | In the database | Walk-forward error for stayers and movers; ranking correlation |
| 1 | Returner ramp-up in the availability model | About two game-equivalents of minutes lost per return (14.1) | In the database | Effective-availability error for returners; simulation check |
| 2 | Positional crowding features | Minutes and counting stats fall with same-position competition (14.3) | In the database (coarse positions); ESPN eligibility would sharpen it | Minutes and attempts models, with and without |
| 2 | Role uncertainty in the simulation: wider minutes spread for returners, movers and rookies, and correlated team-level absences | Absences appear to cluster (14.2); the talent spread depends only on experience (section 9.2); the realism test found the spread about 5% too narrow and top-heavy rosters under-credited (section 9.6) | In the database | Probability-integral-transform histograms by group; weekly spread |
| 2 | Starter versus bench flag and role-change events | A move from the bench to the starting five is a bigger change than the minutes alone show | Starting lineups (an additional download; needs a yes) | Feature in the minutes model; subgroup error |
| 3 | Injury type, severity and return timeline for returners | The main thing the box scores cannot tell us | External source (the `player_absence` table is empty) | Returner error; needs more seasons to be testable |
| 3 | Historical average draft position | To learn whether the market or the model is closer in past seasons, and whether blending them helps | External archive | Rank correlation with realised value, by year |
| 3 | Earlier seasons (2010-11 to 2014-15) | More returners, more test folds, more seasons for the real-season validation | Download (needs a yes) | Re-run the backtest and the validation |
| 3 | Preseason signals (preseason minutes, coaching changes, contract status) | Roles are announced before they show up in regular-season box scores | External sources | Feature in the minutes model, for the owner's draft date |

## 14.5 A protocol for testing any of these ideas

Each idea should pass the same sequence that the existing model passed: (1) state the primary metric before looking (for role features: the out-of-sample squared error of minutes and of attempts, and the Spearman ranking correlation with realised value); (2) add the feature to the walk-forward backtest (folds 2021-22 to 2025-26, refitting everything inside each fold) and compare with the current model on identical player-seasons; (3) report subgroup results where the idea should help (returners, movers, centers) *and* where it should not, to catch accidental gains; (4) correct for the number of ideas tried, because the best of several is biased upward (the winner's curse of section 10.5); (5) only then rebuild the board and, for changes that move many players, rerun the real-season mock-draft validation of section 10.7. A feature that improves the ranking by much less than its season-to-season variability (the model's Spearman correlation ranged from {sp_lo:.2f} to {sp_hi:.2f} across the five test seasons) should be adopted only if it is simple and interpretable.

## 14.6 Questions for the project owner

1. **Priority.** Do you want the first round to be the two data-in-hand items (usage-weighted context, returner ramp-up), which could be tested within a day, or a larger rebuild around role (usage × minutes × efficiency, with team-level effects)?
2. **Downloads.** May I download (a) starting-lineup flags and (b) the 2010-11 to 2014-15 seasons? Both enlarge the sample for role questions but add data to the repository's database; each needs your yes.
3. **Injury information.** Can you obtain injury types or return timelines for the returners you care about (for example from the team reports you read before drafts)? Even a short manual list would let us test whether return minutes depend on injury type.
4. **Risk preference.** The pick policy maximises the chance of winning a week. Do you want a more volatile roster (high-minutes-uncertainty players such as returners) or a steadier one? The role-uncertainty work in 14.4 would let us offer that choice explicitly.
5. **Scope.** Is the project still draft-only? Several findings in 14.2 (the bench picking up a star's possessions) matter most for in-season pickups, which are out of scope by your instruction; I would keep them in this section as findings and not build on them unless you widen the scope.
6. **Positions.** Can you paste or export exact ESPN position eligibility from a machine that is not blocked? It would make the positional work in 14.3 much sharper.
"""
