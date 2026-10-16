"""Sections 11-13: results against the market, case studies, last season's jumpers and fallers, and the conclusion."""
from __future__ import annotations

import shutil

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scripts.final_report.common import *  # noqa: F403

COUNT_COLS = ("pts", "fgm", "fga", "tpm", "tpa", "ftm", "fta", "reb", "ast", "stl", "blk", "tov")
AVAIL_GAP = 8          # a top-30 disagreement is "games and minutes" when the per-game rank is at least this many places better than the board rank


def ordn(n: float) -> str:
    n = int(round(n))
    return f"{n}{'th' if 10 <= n % 100 <= 20 else {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th')}"


def board_rank_after(proj: pd.DataFrame, name: str, mpg: float | None = None, avail: float | None = None) -> int:
    """Board rank of one player if only his minutes (counting stats rescaled) and/or availability change."""
    d = proj.copy()
    i = d.index[d.name == name][0]
    if mpg is not None:
        f = mpg / d.loc[i, "mpg"]
        for c in COUNT_COLS:
            d.loc[i, c] = d.loc[i, c] * f
        d.loc[i, "mpg"] = mpg
    if avail is not None:
        d.loc[i, "avail"] = avail
    return int(z_scores(d)["z_total"].rank(ascending=False, method="first")[i])


def per_game_rank(proj: pd.DataFrame) -> pd.Series:
    """Board rank if every player were available for every game (isolates per-game production from games played)."""
    return z_scores(proj.assign(avail=1.0))["z_total"].rank(ascending=False, method="first").astype(int)


def cat_text(row: pd.Series, k: int = 2, strongest: bool = True) -> str:
    z = {c: float(row[f"z_{c}"]) for c in CATEGORIES}
    keys = sorted(z, key=lambda c: -z[c] if strongest else z[c])[:k]
    return ", ".join(f"{CAT_LABELS[c]} {z[c]:+.1f}" for c in keys)


# ---------------------------------------------------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------------------------------------------------
def market_frame(ctx) -> pd.DataFrame:
    m = ctx.market.copy()
    m = m[~m["placeholder"]].copy()
    pj = ctx.proj.set_index("player_id")
    pgr = per_game_rank(ctx.proj)
    pgr.index = ctx.proj["player_id"].to_numpy()
    m["age"] = m.player_id.map(pj["age_t"]); m["mpg"] = m.player_id.map(pj["mpg"]); m["ppg"] = m.player_id.map(pj["pts"])
    m["gp"] = m.player_id.map(pj["avail"]) * 82
    m["pg_rank"] = m.player_id.map(pgr)
    return m


def placeholder_counts() -> tuple[int | None, int | None]:
    """How many workbook players carry the 'rarely drafted' placeholder ADP (139-141); read from the workbook itself."""
    try:
        from scripts.plot_rank_vs_adp import read_adp
        a = read_adp(str(ADP_XLSX))
        return int(a.blend_adp.between(139.0, 141.0).sum()), len(a)
    except Exception:
        return None, None


def season_line(ctx, pid: int, year: int) -> tuple | None:
    h = ctx.apg[(ctx.apg.player_id == pid) & (ctx.apg.year == year)]
    return None if h.empty else (float(h.gp.iloc[0]), float(h.mpg.iloc[0]), float(h.pts.iloc[0]), float(h.ft_pct.iloc[0]))


def fs(s: tuple | None) -> str:
    return "no games" if s is None else f"{s[0]:.0f} games at {s[1]:.1f} MPG"


# ---------------------------------------------------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------------------------------------------------
def fig_market_tiers(ctx, m: pd.DataFrame) -> dict:
    d = m[m.blend_order <= 130].copy()
    tiers = [(1, 10), (11, 30), (31, 60), (61, 100), (101, 130)]
    rows = []
    for a, b in tiers:
        s = d[(d.blend_order >= a) & (d.blend_order <= b)]
        rows.append(dict(tier=f"{a}-{b}", n=len(s), med=float(s.gap.abs().median())))
    t = pd.DataFrame(rows)
    n_all = len(ctx.proj)
    ks = np.arange(10, 131, 10)
    ov = [len(set(m[m.blend_order <= k].player_id) & set(m[m.model_rank <= k].player_id)) / k for k in ks]
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.2))
    ax = axes[0]
    ax.bar(range(len(t)), t.med, color=C["blue"], width=0.62)
    for xi, r in zip(range(len(t)), t.itertuples()):
        ax.text(xi, r.med + 0.6, f"{r.med:.0f}", ha="center", fontsize=9, color=C["ink2"])
    ax.set_xticks(range(len(t)), [f"{a}\n(n={n})" for a, n in zip(t.tier, t.n)]); ax.set_xlabel("tier of the market order"); ax.set_ylabel("median |our rank − market order| (places)")
    ax.set_title("Disagreement grows with draft position", fontsize=10, loc="left"); ax.grid(axis="x", visible=False)
    ax = axes[1]
    ax.plot(ks, np.array(ov) * 100, color=C["blue"], lw=2.4, marker="o", ms=5)
    ax.plot(ks, ks / n_all * 100, color=C["axis"], lw=1.6, ls=(0, (4, 3)))
    ax.set_xlabel("k"); ax.set_ylabel("share of the market's top k that is in our top k (%)"); ax.set_ylim(0, 100)
    ax.set_title("Overlap of the top-k sets", fontsize=10, loc="left"); ax.grid(axis="x", visible=False)
    F.headline(fig, "We agree with the market most about the stars, and the disagreement grows steadily down the draft",
               "Left: median absolute rank gap by tier of the market order. Right: overlap between our top k and the market's top k")
    F.legend_top(fig, [F.swatch(C["blue"], "observed overlap", "line"), F.swatch(C["axis"], f"expected by chance among {n_all} players", "line")])
    F.save(fig, "fig_market_tiers")
    return dict(tiers=t, overlap=dict(zip(ks.tolist(), ov)))


def fig_market_gaps(ctx, m: pd.DataFrame) -> dict:
    d = m[m.blend_order <= 130]
    low = d.nlargest(14, "gap")                 # we rank him lower (later) than the market
    high = d.nsmallest(14, "gap")               # we rank him higher
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 5.4))
    for ax, df, ttl, col in ((axes[0], low, "We rank him LOWER than the market", C["orange"]), (axes[1], high, "We rank him HIGHER than the market", C["blue"])):
        df = df.iloc[::-1]
        y = np.arange(len(df))
        ax.barh(y, df.gap.abs(), color=col, height=0.66)
        for yi, r in zip(y, df.itertuples()):
            note = f" ({int(r.games_out)} games out)" if r.games_out >= 20 else ""
            ax.text(abs(r.gap) + 3, yi, f"{r.blend_order:.0f} → {int(r.model_rank)}{note}", va="center", fontsize=7.8, color=C["ink2"])
        ax.set_yticks(y, df.player, fontsize=8.2); ax.set_title(ttl, fontsize=10, loc="left"); ax.grid(axis="y", visible=False)
        ax.set_xlim(0, df.gap.abs().max() * 1.55)
        ax.set_xlabel("gap in places (label: market order → our rank)")
    F.headline(fig, "The largest disagreements: assumed absences and low projected minutes on one side, steal-heavy role players on the other",
               "Players the market drafts in its first 130 with the biggest rank gaps; 'games out' = a known absence assumed by the model")
    F.save(fig, "fig_market_gaps")
    return dict(low=low, high=high)


def fig_cats(ctx, names: list[str], fname: str, head: str, sub: str, note: str | None = None) -> None:
    """Small multiples: one player per panel, nine category z-scores (before the -3 floor)."""
    p = ctx.proj
    mk = ctx.market.set_index("player_id")["blend_order"]
    n = len(names)
    cols = 4
    rows = -(-(n + (1 if note else 0)) // cols)
    fig, axes = plt.subplots(rows, cols, figsize=(11.2, 3.0 * rows))
    axes = np.atleast_1d(axes).ravel()
    for ax, nm in zip(axes, names):
        r = p[p.name == nm].iloc[0]
        z = np.array([r[f"z_{c}"] for c in CATEGORIES])
        ax.bar(range(9), z, color=[C["blue"] if v >= 0 else C["red"] for v in z], width=0.7)
        ax.axhline(0, color=C["ink2"], lw=1)
        ax.set_xticks(range(9), [CAT_LABELS[c] for c in CATEGORIES], fontsize=6.6, rotation=60); ax.set_ylim(-3.2, 3.2)
        ax.set_title(f"{nm}\nboard {int(r['rank'])} · market {mk.get(int(r.player_id), float('nan')):.0f}", fontsize=8.6, loc="left", linespacing=1.3)
    for ax in axes[n:]:
        ax.axis("off")
    if note:
        axes[n].text(0.0, 0.8, note, fontsize=8.6, color=C["ink2"], va="top")
    F.headline(fig, head, sub)
    F.save(fig, fname)


def fig_haliburton_whatif(ctx) -> pd.DataFrame:
    p = ctx.proj
    grid = np.arange(24, 35.1, 1.0)
    ranks = [board_rank_after(p, "Tyrese Haliburton", mpg=float(g)) for g in grid]
    fig, ax = F.new(8.2, 4.2)
    ax.plot(grid, ranks, color=C["blue"], lw=2.4, marker="o", ms=5)
    marks = [(26.3, "model: 26.3", C["orange"]), (29.8, "owner's override: 29.8", C["aqua"]), (33.6, "2024-25: 33.6", C["violet"])]
    for xv, lab_, col in marks:
        rr = board_rank_after(p, "Tyrese Haliburton", mpg=xv)
        ax.scatter([xv], [rr], s=90, color=col, zorder=4, edgecolor="white", linewidth=1.4)
        ax.annotate(f"{lab_}\nboard rank {rr}", (xv, rr), xytext=(12, -30), textcoords="offset points", fontsize=8.3, color=col, fontweight="bold", ha="left", va="top")
    ax.set_xlim(23.5, 38.5); ax.invert_yaxis(); ax.set_xlabel("Haliburton's projected minutes per game (per-minute rates and games played unchanged)"); ax.set_ylabel("his 2026-27 board rank")
    ax.grid(axis="x", visible=False)
    F.headline(fig, "A returner's minutes are the whole story: each extra minute per game is worth several places on the board",
               "Haliburton's board rank under alternative minutes assumptions (all other players unchanged)")
    F.save(fig, "fig_haliburton_whatif")
    return pd.DataFrame({"mpg": grid, "rank": ranks})


def fig_risers_fallers(ctx):
    mv = A.movers(ctx.proj, ctx.apg, 2025)
    mv["why"] = mv.apply(lambda r: A.driver_text(r, CAT_LABELS), axis=1)
    pg = A.movers(ctx.proj, ctx.apg, 2025, hold_avail=True)
    pg["why"] = pg.apply(lambda r: A.driver_text(r, CAT_LABELS), axis=1)

    def bar(both, name, head, sub, xl):
        fig, ax = F.new(9.2, 7.4)
        y = np.arange(len(both))
        ax.barh(y, both["dz"], color=[C["red"] if v < 0 else C["blue"] for v in both["dz"]], height=0.66)
        for yi, v in enumerate(both["dz"]):
            ax.text(v + (0.1 if v >= 0 else -0.1), yi, f"{v:+.1f}", va="center", ha="left" if v >= 0 else "right", fontsize=8.4, color=C["ink2"])
        ax.set_yticks(y, [f"{n} ({a:.0f})" for n, a in zip(both["name"], both["age"])], fontsize=8.4)
        ax.axvline(0, color=C["ink2"], lw=1); ax.grid(axis="y", visible=False)
        lo, hi = both["dz"].min(), both["dz"].max()
        ax.set_xlim(lo - 0.15 * (hi - lo), hi + 0.12 * (hi - lo)); ax.set_xlabel(xl)
        F.headline(fig, head, sub)
        F.legend_top(fig, [F.swatch(C["blue"], "projected to improve"), F.swatch(C["red"], "projected to decline")])
        F.save(fig, name)

    up, dn = mv.sort_values("dz", ascending=False).head(12), mv.sort_values("dz").head(12)
    bar(pd.concat([dn.iloc[::-1], up.iloc[::-1]]), "fig_movers_total", "Biggest projected jumps and falls in total fantasy value, 2025-26 actual to 2026-27 projected",
        "Players with at least 1,000 minutes in 2025-26; includes the change in projected games played; ages in brackets", "change in 9-category value (z-score points)")
    pup, pdn = pg.sort_values("dz", ascending=False).head(12), pg.sort_values("dz").head(12)
    bar(pd.concat([pdn.iloc[::-1], pup.iloc[::-1]]), "fig_movers_pergame", "Per-game production only: who is projected to be better or worse when he plays",
        "Same players with availability set to 100% in both seasons", "change in per-game 9-category value (z-score points)")
    slope = float(np.polyfit(mv["z_last"], mv["z_proj"], 1)[0])
    fig, ax = F.new(7.8, 6.2)
    ax.scatter(mv["z_last"], mv["z_proj"], s=16, color=F.BG_POINT, alpha=0.75, linewidths=0)
    lim = [mv[["z_last", "z_proj"]].min().min() - 1, mv[["z_last", "z_proj"]].max().max() + 1]
    ax.plot(lim, lim, color=C["ink2"], lw=1, ls=(0, (4, 3)))
    ax.set_xlabel("2025-26 actual value"); ax.set_ylabel("2026-27 projected value"); ax.set_xlim(lim); ax.set_ylim(lim)
    F.headline(fig, f"Projections are a flatter copy of last season (slope {slope:.2f}): regression to the mean, plus a few assumed absences", "Each dot is a player; above the dashed no-change line = projected to improve. The four biggest risers and fallers are highlighted")
    F.legend_top(fig, [F.swatch(C["blue"], "biggest projected risers", "dot"), F.swatch(C["red"], "biggest projected fallers", "dot"), F.swatch(F.BG_POINT, "everyone else", "dot"), F.swatch(C["ink2"], "no change", "line")], ncol=4)
    F.label_points(ax, [(r.z_last, r.z_proj, A.short_name(r.name), C["blue"] if r.dz > 0 else C["red"]) for r in pd.concat([up.head(4), dn.head(4)]).itertuples()])
    F.save(fig, "fig_movers_scatter")
    pg["age_b"] = pg["age"].round()
    ag = pg.groupby("age_b").agg(n=("dz", "size"), dz=("dz", "mean")).query("n >= 8").reset_index()
    fig, ax = F.new(8.2, 4.2)
    ax.axhline(0, color=C["ink2"], lw=1)
    ax.plot(ag["age_b"], ag["dz"], color=C["blue"], lw=2, marker="o", ms=6)
    ax.set_xticks(ag["age_b"], [f"{a:.0f}\n(n={n})" for a, n in zip(ag["age_b"], ag["n"])], fontsize=8.5)
    ax.set_xlabel("age in 2026-27 (number of players)"); ax.set_ylabel("average change in per-game value (z points)")
    F.headline(fig, "What the model learned about aging: players in their early twenties improve and players over 30 decline",
               "Average projected change in per-game 9-category value versus last season, by age (games played held equal)")
    F.save(fig, "fig_aging")
    return mv, pg, up, dn, pup, pdn, slope


# ---------------------------------------------------------------------------------------------------------------------
# text
# ---------------------------------------------------------------------------------------------------------------------
def build(ctx) -> str:
    S, p = ctx.S, ctx.proj
    shutil.copy(EXP / "rank_vs_adp.png", FIGS / "fig_market_scatter.png")
    m = market_frame(ctx)
    d130 = m[m.blend_order <= 130]
    rho = float(m[["model_rank", "blend_order"]].corr(method="spearman").iloc[0, 1])
    rho130 = float(d130[["model_rank", "blend_order"]].corr(method="spearman").iloc[0, 1])
    w10, w20, w40 = (d130.gap.abs() <= 10).mean(), (d130.gap.abs() <= 20).mean(), (d130.gap.abs() <= 40).mean()
    med_gap = float(d130.gap.abs().median())
    tiers = fig_market_tiers(ctx, m)
    gaps = fig_market_gaps(ctx, m)
    n_ph, n_adp = placeholder_counts()
    S["rho"], S["rho130"], S["med_gap"] = rho, rho130, med_gap

    # ---- the market's top 30 ----
    t30 = m[m.blend_order <= 30].sort_values("blend_order")
    top30 = t30[["blend_order", "player", "model_rank", "gap", "pg_rank", "age", "mpg", "gp"]].copy()
    top30.columns = ["market order", "player", "our rank", "our rank − market", "per-game rank*", "age", "proj MPG", "proj GP"]
    n_in30, n_in40 = int((t30.model_rank <= 30).sum()), int((t30.model_rank <= 40).sum())
    same4 = set(m[m.blend_order <= 4].player) == set(m[m.model_rank <= 4].player)
    top4 = ", ".join(m.sort_values("model_rank").head(4).player)
    sga = m[m.player.str.contains("Gilgeous")].iloc[0]
    low30 = t30[t30.gap >= 15].sort_values("gap", ascending=False)
    high30 = t30[t30.gap <= -8].sort_values("gap")
    avail_drv = low30[(low30.model_rank - low30.pg_rank) >= AVAIL_GAP]
    fit_drv = low30[(low30.model_rank - low30.pg_rank) < AVAIL_GAP].sort_values("gap", ascending=False)
    pg_t = p.drop_duplicates("name").set_index("name")
    dtab = pd.DataFrame({"player": low30.player, "market": low30.blend_order, "our rank": low30.model_rank, "per-game rank*": low30.pg_rank, "proj MPG": low30.mpg, "proj GP": low30.gp,
                         "strongest categories (z)": [cat_text(pg_t.loc[n]) for n in low30.player], "weakest categories (z)": [cat_text(pg_t.loc[n], strongest=False) for n in low30.player]})
    fig_cats(ctx, list(low30.player), "fig_top30_gaps", "The market's top-30 players that the board ranks 15 or more places lower, by category",
             "Projected 2026-27 category z-scores (blue helps, red hurts); the title line gives our board rank and the market's blended order")
    tat = p[p.name == "Jayson Tatum"].iloc[0]
    lam = p[p.name == "LaMelo Ball"].iloc[0]
    tat25 = season_line(ctx, int(tat.player_id), 2025)

    # ---- profile of the large disagreements ----
    lowg, highg = d130[d130.gap >= 30], d130[d130.gap <= -30]
    rest = d130[d130.gap.abs() < 30]

    def prof(name, g):
        return {"group": name, "players": len(g), "mean age": g.age.mean(), "share age 31+": (g.age >= 31).mean(), "share age ≤ 24": (g.age <= 24).mean(), "mean proj MPG": g.mpg.mean(), "mean proj GP": g.gp.mean(),
                "share with an assumed absence ≥ 20 games": (g.games_out >= 20).mean()}
    grp = pd.DataFrame([prof("we rank him 30+ places LOWER", lowg), prof("we rank him 30+ places HIGHER", highg), prof("everyone else in the drafted range", rest)])
    low, high = gaps["low"].copy(), gaps["high"].copy()
    abs_names = list(lowg[lowg.games_out >= 20].player)

    def why(r):
        bits = []
        if r.games_out >= 20:
            bits.append(f"assumed out {int(r.games_out)} games")
        if not np.isnan(r.age) and (r.age >= 31 or r.age <= 24):
            bits.append(f"age {r.age:.0f}")
        bits.append(f"{r.mpg:.0f} MPG, {r.gp:.0f} GP")
        return "; ".join(bits)
    gt_low = pd.DataFrame({"player": low.player, "market order": low.blend_order, "our rank": low.model_rank, "gap (places)": low.gap, "profile": [why(r) for r in low.itertuples()]})
    gt_high = pd.DataFrame({"player": high.player, "market order": high.blend_order, "our rank": high.model_rank, "gap (places)": high.gap, "profile": [why(r) for r in high.itertuples()]})
    zz = p.set_index("player_id")
    hi_t = high.head(10)[["player", "blend_order", "model_rank", "age", "mpg", "ppg"]].copy()
    hi_t["strongest categories (z)"] = [cat_text(zz.loc[pid]) for pid in high.head(10).player_id]
    hi_t.columns = ["player", "market order", "our rank", "age", "proj MPG", "proj PPG", "strongest categories (z)"]
    n_low_mpg = int((low.mpg <= 26.5).sum())
    n_high_stl = int(sum(1 for pid in high.head(10).player_id if "STL" in cat_text(zz.loc[pid])))

    # ---- case studies ----
    pgr = per_game_rank(p)
    case_names = ["Kyrie Irving", "Tyrese Haliburton", "Stephon Castle", "Walker Kessler", "Pascal Siakam", "Dejounte Murray", "Norman Powell"]
    mkt = ctx.market.set_index("player_id").blend_order
    info = {}
    for nm in case_names:
        i = p.index[p.name == nm][0]
        r = p.loc[i]
        pid = int(r.player_id)
        info[nm] = dict(board=int(r["rank"]), pg=int(pgr[i]), market=float(mkt.get(pid, np.nan)), age=float(r.age_t), mpg=float(r.mpg), ppg=float(r.pts), gp=float(r.avail * 82),
                        h23=season_line(ctx, pid, 2023), h24=season_line(ctx, pid, 2024), h25=season_line(ctx, pid, 2025),
                        at82=board_rank_after(p, nm, avail=0.82), w65=board_rank_after(p, nm, avail=65 / 82), w30=board_rank_after(p, nm, mpg=30.0),
                        w30_65=board_rank_after(p, nm, mpg=30.0, avail=65 / 82), w34_70=board_rank_after(p, nm, mpg=34.0, avail=70 / 82),
                        strong=cat_text(r, 3), weak=cat_text(r, 3, strongest=False))
    fig_cats(ctx, case_names, "fig_case_studies", "Seven players the board ranks far below the market, by category", "Projected 2026-27 category z-scores; the title line gives our board rank and the market's blended order",
             note="How to read: each bar is the player's\nz-score in one category (blue helps,\nred hurts). The board's value is the\nsum of these nine bars.")
    cst = pd.DataFrame([{"player": nm, "age": v["age"], "proj MPG": v["mpg"], "proj PPG": v["ppg"], "proj GP": v["gp"], "2025-26": ("did not play" if v["h25"] is None else f"{v['h25'][0]:.0f} GP, {v['h25'][1]:.1f} MPG"),
                         "board rank": v["board"], "per-game rank*": v["pg"], "market order": v["market"], "board rank at 82% of games": v["at82"]} for nm, v in info.items()])
    hw = fig_haliburton_whatif(ctx)
    h26 = board_rank_after(p, "Tyrese Haliburton", mpg=26.3)
    h30, h34 = (int(hw[hw.mpg == v]["rank"].iloc[0]) for v in (30.0, 34.0))
    kyr, hal, cas, kes, sia, mur, pow_ = (info[n] for n in case_names)

    # ---- movers ----
    mv, pg, up, dn, pup, pdn, slope = fig_risers_fallers(ctx)
    pool_n = len(mv)
    n_ret_up = int((up["gp_last"] < 60).sum()); n_old_dn = int((dn["age"] >= 31).sum())
    ko = p.drop_duplicates("name").set_index("name")["games_out"]
    out_dn = [n for n, g in zip(dn["name"], dn["gp_proj"]) if g < 30]
    out_known = [n for n in out_dn if ko.get(n, 0) >= 20]

    def mtab(df):
        return pd.DataFrame({"player": df["name"], "age": df["age"], "rank 2025-26": df["rank_last"], "rank 2026-27": df["rank_proj"], "Δ value": df["dz"], "MPG": df["mpg_last"].round(1).astype(str) + " → " + df["mpg_proj"].round(1).astype(str),
                             "GP": df["gp_last"].round(0).astype(int).astype(str) + " → " + df["gp_proj"].round(0).astype(int).astype(str), "biggest category swings": df["why"]})
    fm = {"age": "{:.0f}", "rank 2025-26": "{:.0f}", "rank 2026-27": "{:.0f}", "Δ value": "{:+.1f}"}
    rise_r = mv.sort_values("drank", ascending=False).head(8)[["name", "age", "rank_last", "rank_proj", "drank", "gp_last", "gp_proj"]]
    fall_r = mv.sort_values("drank").head(8)[["name", "age", "rank_last", "rank_proj", "drank", "gp_last", "gp_proj"]]
    rise_s = ", ".join(f"{r.name} ({int(r.rank_last)}→{int(r.rank_proj)})" for r in rise_r.head(6).itertuples())
    fall_s = ", ".join(f"{r.name} ({int(r.rank_last)}→{int(r.rank_proj)})" for r in fall_r.head(6).itertuples())
    S["movers"] = dict(rise=rise_r, fall=fall_r)
    g_hi = int((pup["age"] <= 24).sum()); g_lo = int((pdn["age"] >= 30).sum())

    rl, fresh, rs = S["realism"], S["fresh"], S["real"]
    mm_after = max(abs(S["mm"]["after"][k] - 1) for k in ("PTS", "3PM")) * 100
    sel, spm, spl = S["sel"], S["spearman_model"], S["spearman_last"]
    ks = tiers["overlap"]
    pnew, pold, eng, engng = rs["policy_new"], rs["policy_old"], rs["engine"], rs["engine_noguard"]
    ph_text = (f"{n_ph} of the {n_adp} players in the workbook carry an ADP between 139 and 141, which the source uses as a placeholder for rarely drafted players, so they are excluded (the note on the figure counts the 94 of them that match a player in our pool)" if n_ph else
               "Players with the source's placeholder ADP (139 to 141, used for rarely drafted players) are excluded")
    top4_text = (f"Our top four ({top4}) are exactly the market's top four, in a different order" if same4 else f"Our top four are {top4}")

    return f"""
# 11. Results: the 2026-27 board against the market

## 11.1 The reference

The *market* is the blended average draft position (ADP) of three sites (Yahoo, ESPN and Fantrax) on 14 October 2026, converted to an *order* (1 = first player off the board). {ph_text}. The market is **not ground truth**: it is the average opinion of other drafters, with their biases and herding, so a disagreement is a question, not an error. It was never an input to any model here. What the comparison can tell us is where our independent projection differs from consensus and whether the differences are explainable. The comparison covers the {len(m)} players who are within the market's or our top 150.

{ctx.fig("fig_market_scatter", "Our board rank against the market's blended order. Above the dashed line the model ranks a player lower than the market; below it, higher. Players off the chart are listed at the right.")}

**Overall agreement.** The rank correlation between our board and the market is **ρ = {rho:.2f}** over the {len(m)} players ({rho130:.2f} for the market's first 130, the players a 10-team, 13-round draft takes). The median absolute gap in that drafted range is **{med_gap:.0f} places**; {w10:.0%} of players are within 10 places of the market, {w20:.0%} within 20 and {w40:.0%} within 40. Of the market's top 30, we have **{n_in30} in our top 30 and {n_in40} in our top 40**. A correlation of {rho:.2f} says the two are measuring the same thing but differ substantially, which is the expected pattern: nearly everyone agrees about the stars, and the middle of a draft is mostly opinion about role, health and age.

{ctx.fig("fig_market_tiers", "Median disagreement by tier of the market order (left) and overlap of the top-k sets (right).")}

Agreement is strongest at the top (median gap {tiers["tiers"].iloc[0].med:.0f} places for the market's top 10) and the gap grows tier by tier ({tiers["tiers"].iloc[2].med:.0f} for market order 31-60, {tiers["tiers"].iloc[3].med:.0f} for 61-100, {tiers["tiers"].iloc[4].med:.0f} for 101-130): the later a player goes, the more his draft position rests on role and opinion. The top-10 sets share {ks[10] * 10:.0f} players and the top-30 sets {ks[30] * 30:.0f}; the overlap rises with k only because larger sets are harder to miss, and it stays far above what chance would give.

## 11.2 The market's top 30 side by side

{md_table(top30, {"market order": "{:.0f}", "our rank": "{:.0f}", "our rank − market": "{:+.0f}", "per-game rank*": "{:.0f}", "age": "{:.0f}", "proj MPG": "{:.1f}", "proj GP": "{:.0f}"})}

\\*Per-game rank = the player's board rank if every player in the pool were available for every game. Comparing it with the board rank separates a disagreement about how well a player plays from one about how often he plays.

{top4_text}; the model has {sga.player} first where the market has him {ordn(sga.blend_order)}. The **largest disagreements inside the market's top 30** are the {len(low30)} players that we rank 15 or more places lower than the market ({", ".join(f"{r.player} {r.blend_order:.0f}→{int(r.model_rank)}" for r in low30.itertuples())}). The per-game rank separates two kinds of disagreement:

* **Games and minutes** ({", ".join(avail_drv.player)}). Their per-game ranks are at least {AVAIL_GAP} places better than their board ranks, so the gap is mainly projected availability and minutes. Jayson Tatum had {fs(tat25)} in 2025-26 and is projected for {tat.avail * 82:.0f} games (at 65 games he would be {ordn(board_rank_after(p, "Jayson Tatum", avail=65 / 82))}); LaMelo Ball is projected for {lam.mpg:.1f} minutes and {lam.avail * 82:.0f} games; Haliburton is the returner discussed in section 11.4.
* **Category fit** ({", ".join(fit_drv.player)}). Their per-game ranks are at most {int((fit_drv.model_rank - fit_drv.pg_rank).max())} places better than their board ranks, so availability is not the issue: in a nine-category format their production is worth less than their reputation. The table and figure below show what offsets their points and assists. Trae Young and Josh Giddey are elite assist men whose turnovers cost them a z-score of −2.1 each; LaMelo Ball and Jayson Tatum give back value in field-goal percentage; Domantas Sabonis is a rebounding and field-goal-percentage specialist with negative threes and turnovers; and Jalen Brunson and Austin Reaves are well-rounded scorers without a single category above +1.7 and +1.3 respectively, so they add up to a moderate total.

{md_table(dtab, {"market": "{:.0f}", "our rank": "{:.0f}", "per-game rank*": "{:.0f}", "proj MPG": "{:.1f}", "proj GP": "{:.0f}"})}

{ctx.fig("fig_top30_gaps", "Category z-scores of the market's top-30 players that the board ranks 15 or more places lower.")}

On the other side, the model is higher than the market on {", ".join(f"{r.player} (market {r.blend_order:.0f}, ours {int(r.model_rank)}, per-game rank {int(r.pg_rank)}, age {r.age:.0f})" for r in high30.itertuples())}. Their per-game ranks are all inside the top {int(high30.pg_rank.max())}, so the model values them highly when they play; {int((high30.age >= 36).sum())} of the {len(high30)} are 36 or older, and the market appears to discount age and availability more heavily than the model does.

## 11.3 Where we disagree beyond the top 30, and the pattern behind it

{ctx.fig("fig_market_gaps", "The 14 largest disagreements in each direction, among players the market drafts in the first 130.")}

The profile of the disagreements of 30 places or more shows what separates the groups:

{md_table(grp, {"players": "{:.0f}", "mean age": "{:.1f}", "share age 31+": "{:.0%}", "share age ≤ 24": "{:.0%}", "mean proj MPG": "{:.1f}", "mean proj GP": "{:.0f}", "share with an assumed absence ≥ 20 games": "{:.0%}"})}

The {len(lowg)} players we rank much lower have the *fewest projected games* ({lowg.gp.mean():.0f}, against {rest.gp.mean():.0f} for the rest of the drafted range) and lower projected minutes ({lowg.mpg.mean():.1f} against {rest.mpg.mean():.1f}); the {len(highg)} players we rank much higher have the most games ({highg.gp.mean():.0f}) and are young more often ({(highg.age <= 24).mean():.0%} are 24 or under, against {(rest.age <= 24).mean():.0%}). Only {len(abs_names)} of the {len(lowg)} lower-ranked players carry a known absence ({", ".join(abs_names)}), but they account for the two largest gaps in the table below; {n_low_mpg} of the 14 players in the table are projected for 26.5 minutes or fewer. The main disagreement with the market is therefore **how much playing time to expect**, which is exactly where a projection built from past minutes and games (with shrinkage toward typical roles) and a market that rewards perceived upside should differ.

{md_table(gt_low, {"market order": "{:.0f}", "our rank": "{:.0f}", "gap (places)": "{:+.0f}"})}

**Players we rank higher** have steady full-time roles ({high.mpg.min():.0f} to {high.mpg.max():.0f} projected minutes and {high.gp.min():.0f} to {high.gp.max():.0f} games, across the 14 largest gaps) and a particular strength: {n_high_stl} of the ten players tabulated below have steals among their two best categories.

{md_table(gt_high, {"market order": "{:.0f}", "our rank": "{:.0f}", "gap (places)": "{:+.0f}"})}

{md_table(hi_t, {"market order": "{:.0f}", "our rank": "{:.0f}", "age": "{:.0f}", "proj MPG": "{:.1f}", "proj PPG": "{:.1f}"})}

These "model likes" cases have not been stress-tested the way the "model dislikes" cases were, so they are hypotheses about where the market may be underpricing role and category fit, not established edges. The realism test also found that the simulation, if anything, under-credits top-heavy rosters (section 9.6), which is a reason for caution about trading stars for role players on this basis.

## 11.4 Case studies: seven players the board ranks far below the market

For each player we ask what drives the gap and what would close it. Rankings are on the board of {len(p)} projected players. Each *what-if* changes only that player's minutes and/or games, with everyone else unchanged.

{ctx.fig("fig_case_studies", "Category z-scores for seven players ranked far below the market.")}

{md_table(cst, {"age": "{:.0f}", "proj MPG": "{:.1f}", "proj PPG": "{:.1f}", "proj GP": "{:.0f}", "board rank": "{:.0f}", "per-game rank*": "{:.0f}", "market order": "{:.0f}", "board rank at 82% of games": "{:.0f}"})}

* **Kyrie Irving (age {kyr["age"]:.0f}; board {kyr["board"]}, market {kyr["market"]:.0f}).** A returner: no games in 2025-26 and {fs(kyr["h24"])} in 2024-25. The model projects {kyr["gp"]:.0f} games at {kyr["mpg"]:.1f} MPG. Per game he would rank {ordn(kyr["pg"])} if everyone played every game, so the gap to the market is mostly games and minutes: at 82% of games he would be {ordn(kyr["at82"])} and at 65 games {ordn(kyr["w65"])}. Strongest categories {kyr["strong"]}; weakest {kyr["weak"]}. The backtest supports the *direction* of both adjustments (section 5.4); it cannot tell us how Kyrie specifically has recovered.
* **Tyrese Haliburton (age {hal["age"]:.0f}; board {hal["board"]} after the owner's override, {h26} before; market {hal["market"]:.0f}).** Also a returner: no games in 2025-26 and {fs(hal["h24"])} in 2024-25. The model's own minutes (26.3, against a recency baseline of 31.1) came entirely from the blank last-season slot (section 5.4); the owner's 29.8 is described in section 7.2. The sensitivity is large (figure below): at 26.3, 30 and 34 minutes his board rank would be {h26}, {h30} and {h34}. Strongest categories {hal["strong"]}.
* **Stephon Castle (age {cas["age"]:.0f}; board {cas["board"]}, market {cas["market"]:.0f}).** Not an availability story: {fs(cas["h25"])} in 2025-26 and {cas["gp"]:.0f} games projected. His per-game rank is {ordn(cas["pg"])}, below his board rank, because assists are his one strong category (strongest {cas["strong"]}) while the weakest ({cas["weak"]}) cost almost as much. Even at 30 minutes he would be {ordn(cas["w30"])}. As a third-season player he is in the group the model slightly under-projects (section 5.5), but a full correction would move him only about nine places, so the gap is mainly a bet on growth that the model can only partly credit.
* **Walker Kessler (age {kes["age"]:.0f}; board {kes["board"]}, market {kes["market"]:.0f}).** {fs(kes["h24"])} in 2024-25, but only {fs(kes["h25"])} in 2025-26. The model projects {kes["mpg"]:.1f} MPG and {kes["gp"]:.0f} games. His value depends heavily on minutes: strongest {kes["strong"]}; weakest {kes["weak"]}. At 30 minutes he would be {ordn(kes["w30"])}, and at 30 minutes and 65 games {ordn(kes["w30_65"])}. This is the largest single gap of the seven and is mainly a bet on his role.
* **Pascal Siakam (age {sia["age"]:.0f}; board {sia["board"]}, market {sia["market"]:.0f}).** {fs(sia["h25"])} in 2025-26 ({sia["h25"][2]:.1f} PPG, {sia["h25"][3]:.1%} from the line); projected {sia["mpg"]:.1f} MPG, {sia["ppg"]:.1f} PPG and {sia["gp"]:.0f} games. The free-throw category costs him the most (weakest {sia["weak"]}). With a larger role (34 minutes, 70 games) he would be {ordn(sia["w34_70"])}. The projection mostly repeats his own recent record, and we found no model error.
* **Dejounte Murray (age {mur["age"]:.0f}; board {mur["board"]}, market {mur["market"]:.0f}).** {fs(mur["h23"])} in 2023-24, {fs(mur["h24"])} in 2024-25 and {fs(mur["h25"])} in 2025-26. Projected {mur["gp"]:.0f} games at {mur["mpg"]:.1f} MPG. His per-game rank is {ordn(mur["pg"])}, so the games discount is what pushes him out of the drafted range; at 65 games he would be {ordn(mur["w65"])}. The team-context features reduced his projected games by five to six (section 5.5).
* **Norman Powell (age {pow_["age"]:.0f}; board {pow_["board"]}, market {pow_["market"]:.0f}).** {fs(pow_["h25"])} in 2025-26 ({pow_["h25"][2]:.1f} PPG); projected {pow_["mpg"]:.1f} MPG, {pow_["ppg"]:.1f} PPG and {pow_["gp"]:.0f} games. His per-game rank is only {ordn(pow_["pg"])} (weakest {pow_["weak"]}) and age and games reduce it further; the context features also trimmed his games by five to six. The ablation of section 5.5 showed that the features improve games-played projections overall, so the change was kept. At 30 minutes and 65 games he would be {ordn(pow_["w30_65"])}.

{ctx.fig("fig_haliburton_whatif", "Haliburton's board rank under alternative minutes assumptions.")}

# 12. Last season's biggest jumpers and fallers

We compare each player's *actual* 2025-26 production with his 2026-27 projection, both z-scored within the same pool (the {pool_n} players with at least 1,000 minutes last season, so ranks in this section are within that pool and differ from the board ranks above). Two lenses are shown because a player's fantasy value changes for two reasons: he plays better or worse per game, or he plays more or fewer games.

## 12.1 Total fantasy value (what you actually draft)

{ctx.fig("fig_movers_total", "Biggest projected changes in total value, 2025-26 actual to 2026-27 projected.")}

**Biggest risers**

{md_table(mtab(up), fm)}

**Biggest fallers**

{md_table(mtab(dn), fm)}

*MPG and GP show last season's actual and this season's projection.* **What drives the moves.** {n_ret_up} of the 12 biggest risers played fewer than 60 games last year, and the projection returns them toward a normal share of games, which lifts every counting statistic at once; the largest riser is Giannis Antetokounmpo (36 games last year, 221st in the pool, projected 6th; section 8.3). Of the 12 biggest fallers, {len(out_dn)} are projected for fewer than 30 games ({", ".join(out_dn)}), {len(out_known)} of them because of a known absence, and {n_old_dn} are aged 31 or older.

**By places gained or lost in the pool:** biggest risers {rise_s}; biggest fallers {fall_s}.

## 12.2 Per-game production only (games held equal)

{ctx.fig("fig_movers_pergame", "Biggest changes in per-game value with availability set to 100% in both seasons.")}

{g_hi} of the 12 per-game risers are 24 or younger and {g_lo} of the 12 per-game fallers are 30 or older: once availability is out of the picture, the age signal dominates.

{ctx.fig("fig_aging", "Average projected change in per-game value by age.")}

{ctx.fig("fig_movers_scatter", "Projected versus last season's value: the slope below one is regression to the mean.")}

Three forces are visible: availability (returning players are projected to play more, which mostly moves the total-value list), aging (learned from the data, not imposed: young players gain and players past 30 lose), and regression to the mean (last season's outliers are pulled toward what comparable players sustain; the fitted slope of {slope:.2f} summarises it, although a few players far below the line, such as the labelled fallers, are there because of assumed absences rather than regression). These are projected *averages*, not predictions of any specific event: a faller is not predicted to be bad, and a riser is not guaranteed to improve.

# 13. Conclusion

## 13.1 What was built

A three-phase system that, for a specific 10-team, 9-category, head-to-head league, recommends at every pick which of the top-200 players gives the best chance of winning weekly matchups: **(1)** projections of every player's 2026-27 per-game statistics built from a shrinkage baseline plus learned corrections and a careful treatment of availability, roster context, rookies and documented expert inputs; **(2)** a simulation that turns each projection into a distribution of weeks on the real calendar, including the three playoff weeks; and **(3)** a draft engine and a category-aware pick policy that play out the rest of the draft and choose the player who maximises the probability of taking at least five of nine categories. Each phase was tested on seasons it had not seen.

## 13.2 Summary of the findings

**Projections (Phase 1).**

* On {S["n_oos"]:,} out-of-sample player-seasons (walk-forward folds 2021-22 to 2025-26; {int(S["acc"]["n"].iloc[0]):,} of them have a previous season to compare with) the projections reduced mean absolute error against "same as last year" in all nine categories (by {S["acc"]["mae_gain_pct"].min():.0f}% to {S["acc"]["mae_gain_pct"].max():.0f}%); the average rank correlation with true fantasy value was **{spm:.3f}** against **{spl:.3f}** for last year's production, and the projection ranked better in all five seasons.
* The learned corrections reduced squared error against the shrinkage baseline most for minutes ({sel.loc["mpg", "vs_baseline_pct"]:.0f}%), 2-point attempts ({sel.loc["fga2", "vs_baseline_pct"]:.0f}%), availability ({sel.loc["avail", "vs_baseline_pct"]:.0f}%) and free-throw attempts ({sel.loc["fta", "vs_baseline_pct"]:.0f}%); rebounds, steals and free-throw percentage stay with the simple shrinkage baseline.
* Residuals were close to zero on average, with mild regression to the mean at the extremes and a heavy upper tail (breakouts and sudden role increases); the extremes are role changes rather than smooth noise. The misses are strongly correlated across categories (0.6 to 0.8 between the minutes miss and the misses in points, rebounds, assists, threes and turnovers), so most of what the model gets wrong about a player is his role.
* **Availability was the largest source of error and the biggest conceptual correction**: counting games only after a player's first appearance, treating known absences as inputs, and using the mean for ranking but the median for display.
* Team context improved projections overall (out-of-sample squared error {S["abl"]["mean"]:+.1f}% higher on average, and {S["abl"]["pct"]["avail"]:+.1f}% for games played, when it is removed), so it was kept; third-year corrections were small and not adopted.

**Simulation (Phase 2).**

* The library reproduces projected means once the make rates are mean-matched (a check found points 1.8% and threes 4.2% too high before the fix; after it both are within {mm_after:.1f}%).
* Tested on four past seasons ({rl["n_tw"]:,} team-weeks), simulated weekly totals are about 5% too narrow overall (the 2024-25 and 2025-26 seasons noticeably so), the dependence between categories is reproduced (mean correlation {rl["corr_sim"]:+.2f} simulated against {rl["corr_real"]:+.2f} real), and simulated matchup probabilities are close to observed win rates (calibration slope {rl["slope"]:.2f}, Brier {rl["brier"]:.3f} against {rl["brier0"]:.3f} for always saying 50%). The simulation slightly understates how much top-heavy rosters win and how well real rosters shoot free throws.

**Draft decisions (Phase 3).**

* On the first complete validation (2022-23 to 2025-26, 96 drafts) the engine looked better inside its own simulation (+3.6 points of weekly win probability) than on real box scores (−0.8 ± 1.7); a guard against noisy near-ties repaired most of that (+2.5 ± 1.3 on the same drafts, {eng["mean"]:+.1f} over all five seasons).
* The category-aware policy (4,200 simulated weeks, six candidates per pick, 0.5-point guard) averages **{pnew["mean"]:+.1f} points (standard error {pnew["se"]:.1f}, t = {pnew["t"]:.1f}, positive in {pnew["pos"]} of {pnew["n"]} seasons)** above "take the best player left" on real 2021-22 to 2025-26 box scores; the previous setting averaged {pold["mean"]:+.1f}, the guarded engine {eng["mean"]:+.1f} and the unguarded engine {engng["mean"]:+.1f} (four seasons). On fresh simulated drafts the final setting beat the previous one by {fresh["diff"] * 100:+.2f} ± {fresh["diff_se"] * 100:.2f} points. Over an 18-week regular season a gain of about 5 points of weekly win probability is roughly +0.9 wins.
* Considering every player up to rank 200 does not improve picks, and doing so with an unguarded rule makes them worse.

**The board against the market.** Rank correlation {rho:.2f}; median gap {med_gap:.0f} places; agreement is best at the top ({"the market's top four are our top four" if same4 else "the top four overlap strongly"}). Inside the market's top 30 the board is lower on {", ".join(low30.player)}, partly because of projected games and minutes and partly because of category fit. Beyond the top 30 it is lower on players it projects for few minutes or games (Porziņģis and Ingram because of known absences; Kessler, Edey, Harper, Jerome, Dëmin and others because of projected minutes) and higher on role players with steady minutes and steals (Wallace, Suggs, Camara, Sheppard, Pritchard, Edgecombe). Giannis Antetokounmpo, who started the investigation at 93rd, ends at 6th, level with the market, after three separately justified adjustments.

## 13.3 Where we reconsidered the model

| Issue | What we saw | What we did | Result |
|---|---|---|---|
| Availability definition | Players known to be out at tip-off looked like random injuries; games seemed under-projected | Availability since first appearance; known absences as an input; returner blend | Mean bias of −1.0 games (−2.7 for players who changed teams, −2.3 for age 33+); stars returning handled explicitly |
| Mean versus median games | Games are left-skewed | Mean for ranking, median and P25-P75 for display | Same ranking quality (Spearman ρ = 0.996 between the two boards) |
| Team context | Context moved Powell's and Murray's projected games down by 5-6 | Ablation in the backtest | Context helps (error +1.3% for games, +1.4% on average without it); kept |
| Third-year players | Under-projected by about 0.4 points and 0.6 minutes | Season-of-career flags | Small trim, no accuracy gain; not adopted |
| Giannis | Ranked 93rd, then 103rd | Treated 2025-26 as a rest year; switched off his context penalty; floored each category's z-score at −3 | 103rd to 59th, then 29th, then 6th; the floor changes no one else in the top 150 |
| Punting | Hypothesis that his weakness could be exploited | Experiment | A balanced roster won more weeks; not adopted |
| Simulation shooting bias | Points +1.8%, threes +4.2% above projections | Mean-matching of the make rates | Within {mm_after:.1f}% after the correction |
| Absence share in the simulation | Real weeks piled in the lowest and highest 10% buckets (15.8% and 15.2% in 2025-26) | Re-measured the whole-week absence share over every week for every rotation player (0.42 to 0.68) | Reproduces the real 18-21% zero-game weeks and roughly halves the distance from a flat histogram |
| First engine | +3.6 in simulation, −0.8 on real seasons | Guard against noisy near-ties | +2.5 on the same real drafts |
| Policy tuning | More candidates seemed better | Held-out and fresh-seed tests; winner's-curse awareness | Six candidates, 200 draws, guard 0.005 |
| Interim result | An early lead for the policy over the engine | Waited for the full run | The lead vanished: the two are equal within noise |
| Haliburton | Model minutes 26.3 after a missed season | Owner's 29.8-minute override with per-minute rates fixed | Board rank {h26} to {hal["board"]} |

## 13.4 Limitations

* **Five real seasons of validation.** Within a season, mock drafts share the same real outcomes, so the effective sample is five seasons; that is enough to rule out a large harm but too few to measure a small gain precisely. The calibration of the simulation was fitted on all seasons, so the replay is not strictly out of sample for it.
* **Simulated comparisons share one model.** Held-out simulations guard against fitting noise, but they cannot reveal errors in the projections themselves.
* **The market is not truth.** No historical ADP was available, so we cannot say whether the market or the model would have been closer in past seasons.
* **Known structural gaps.** The simulation's weekly spread is about 5% narrow (more in recent seasons) and it slightly understates top-heavy rosters and free-throw shooting; projected probabilities are about 5% too extreme. Position eligibility is approximate; there is no model of injuries by type, of depth charts, or of in-season management (waivers, trades, injured-reserve use), by design.
* **Small samples in subgroup tests.** The returner results rest on 24 players with at least 500 minutes after a missed season (12 former rotation stars) and 94 returner-seasons with at least one game; they support the model's direction, not its precision for any one player.

## 13.5 Future work

The role-focused agenda (returning from injury, roster change, positional depth) is developed, with a first look at the data, in section 14. Beyond it: include historical ADP to compare the model's accuracy with the market's; add playoff qualification to the objective; widen the weekly spread, perhaps by season-level variability in absences; investigate the free-throw and star-premium biases; give returners an injury-type feature if reliable data can be sourced; and download seasons before 2015-16 so that the walk-forward backtest can cover more test years.
"""
