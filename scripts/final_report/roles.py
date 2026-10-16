"""
First-look analyses behind the "next steps" section: role and usage, returning from injury, roster change, positional depth.

These are exploratory descriptive statistics on the 2015-16 to 2025-26 box scores (no model is changed by them). Usage is the standard
box-score estimate  USG% = 100 * (FGA + 0.44 FTA + TOV) * (team minutes / 5) / (player minutes * team (FGA + 0.44 FTA + TOV)).
Positions are the three buckets (G / F / C) of the API's position label, so every positional result is approximate.
"""
from __future__ import annotations

import sqlite3

import numpy as np
import pandas as pd

from src.features.player_seasons import pos_group
from src.features.team_context import team_context

POSS_FT = 0.44
HI_USG = 24.0                 # a "high-usage" player: at least this USG% (roughly the top quarter of rotation players)
K_OPEN = 10                   # opening roster = played in the team's first K games
METRICS = ["mpg", "usg", "pts36", "ast36", "reb36", "tov36", "ts"]
WINDOWS = [(1, 3), (4, 10), (11, 20), (21, 40)]


# ---------------------------------------------------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------------------------------------------------
def load_box(conn: sqlite3.Connection) -> pd.DataFrame:
    """Regular-season rows with minutes > 0, plus each team's own game number and possessions used."""
    b = pd.read_sql_query(
        """SELECT g.game_id, g.season, g.date, bs.team_id, bs.player_id, bs.min, bs.fga, bs.fta, bs.tov, bs.ast, bs.points AS pts
           FROM box_score bs JOIN game g ON g.game_id = bs.game_id
           WHERE g.game_type = 'regular' AND bs.min > 0""", conn)
    b["year"] = b["season"].str[:4].astype(int)
    b["poss"] = b["fga"].fillna(0) + POSS_FT * b["fta"].fillna(0) + b["tov"].fillna(0)
    d = b[["year", "team_id", "date"]].drop_duplicates().sort_values(["year", "team_id", "date"])
    d["tgn"] = d.groupby(["year", "team_id"]).cumcount() + 1
    return b.merge(d, on=["year", "team_id", "date"])


def team_usage(box: pd.DataFrame) -> pd.DataFrame:
    """Usage per (year, team, player) stint, plus the minutes and games behind it."""
    tt = box.groupby(["year", "team_id"]).agg(tmin=("min", "sum"), tposs=("poss", "sum")).reset_index()
    pt = box.groupby(["year", "team_id", "player_id"]).agg(min=("min", "sum"), poss=("poss", "sum"), gp=("min", "size")).reset_index().merge(tt, on=["year", "team_id"])
    pt["usg"] = 100 * pt["poss"] * (pt["tmin"] / 5) / (pt["min"] * pt["tposs"])
    pt["mpg"] = pt["min"] / pt["gp"]
    return pt


def season_panel(box: pd.DataFrame, st: pd.DataFrame, apg: pd.DataFrame) -> pd.DataFrame:
    """One row per (player_id, year): minutes, usage (minute-weighted across teams), per-36 rates, true shooting, availability, age."""
    pt = team_usage(box)
    ps = pt.assign(w=pt["usg"] * pt["min"]).groupby(["year", "player_id"]).agg(w=("w", "sum"), m=("min", "sum")).reset_index()
    ps["usg"] = ps["w"] / ps["m"]
    p = apg.merge(ps[["year", "player_id", "usg"]], on=["year", "player_id"], how="left").merge(
        st[["year", "player_id", "age", "sg", "pos", "name"]], on=["year", "player_id"], how="left")
    p["pts36"] = p["pts"] / p["mpg"] * 36
    for c in ("ast", "reb", "tov"):
        p[f"{c}36"] = p[c] / p["mpg"] * 36
    p["ts"] = p["pts"] / (2 * (p["fga"] + POSS_FT * p["fta"]))
    p["availt"] = p["gp"] / p["sg"]
    return p


def ols(y: np.ndarray, X: np.ndarray, cluster: np.ndarray | None = None):
    """OLS with an intercept; returns coefficients, (cluster-robust if clusters are given, else HC1) standard errors and R^2."""
    X1 = np.column_stack([np.ones(len(y)), X])
    XtXi = np.linalg.pinv(X1.T @ X1)
    b = XtXi @ X1.T @ y
    e = y - X1 @ b
    n, k = X1.shape
    if cluster is None:
        meat = (X1 * e[:, None]).T @ (X1 * e[:, None]) * n / (n - k)
    else:
        meat = np.zeros((k, k))
        cl = pd.factorize(cluster)[0]
        g = cl.max() + 1
        for c in range(g):
            s = (X1[cl == c] * e[cl == c, None]).sum(axis=0)
            meat += np.outer(s, s)
        meat *= g / (g - 1) * (n - 1) / (n - k)
    cov = XtXi @ meat @ XtXi
    return b, np.sqrt(np.diag(cov)), float(1 - (e ** 2).sum() / ((y - y.mean()) ** 2).sum())


# ---------------------------------------------------------------------------------------------------------------------
# A. returners: does the role change, or only the minutes?
# ---------------------------------------------------------------------------------------------------------------------
def returners_panel(p: pd.DataFrame, rot_mpg: float = 24.0, min_now: float = 500.0) -> pd.DataFrame:
    """
    Rotation players (>= rot_mpg minutes per game) who then played >= min_now minutes in year T, in three groups:
      'missed a season'     no games in T-1 (last known role = T-2, at least 20 games),
      'partial season'      played < 50% of team games in T-1 (at least 8),
      'healthy (control)'   played >= 80% of team games in T-1.
    Change = year T minus the last known season.
    """
    cols = ["player_id", "year"] + METRICS + ["gp", "availt", "age", "min"]
    cur = p[p["min"] >= min_now][cols]
    pr = p[["player_id", "year"] + METRICS + ["gp", "availt"]].copy(); pr["year"] += 1
    pr.columns = ["player_id", "year"] + [f"{c}_p" for c in METRICS] + ["gp_p", "availt_p"]
    pp = p[["player_id", "year"] + METRICS + ["gp"]].copy(); pp["year"] += 2
    pp.columns = ["player_id", "year"] + [f"{c}_pp" for c in METRICS] + ["gp_pp"]
    d = cur.merge(pr, on=["player_id", "year"], how="left").merge(pp, on=["player_id", "year"], how="left")
    out = []
    for _, r in d.iterrows():
        if pd.isna(r["gp_p"]):
            if pd.notna(r["gp_pp"]) and r["mpg_pp"] >= rot_mpg and r["gp_pp"] >= 20:
                g, suf = "missed a season", "_pp"
            else:
                continue
        elif r["mpg_p"] >= rot_mpg and r["availt_p"] < 0.5 and r["gp_p"] >= 8:
            g, suf = "partial season", "_p"
        elif r["mpg_p"] >= rot_mpg and r["availt_p"] >= 0.8:
            g, suf = "healthy (control)", "_p"
        else:
            continue
        rec = {"player_id": int(r["player_id"]), "year": int(r["year"]), "group": g, "age": r["age"]}
        for m in METRICS:
            rec[f"d_{m}"] = r[m] - r[f"{m}{suf}"]
            rec[f"last_{m}"] = r[f"{m}{suf}"]
        out.append(rec)
    return pd.DataFrame(out)


def returners_adjusted(rp: pd.DataFrame) -> pd.DataFrame:
    """Group difference from the healthy control after adjusting for age (linear and quadratic) and last known minutes; clustered by player."""
    rows = []
    g = rp["group"].map({"healthy (control)": 0, "partial season": 1, "missed a season": 2})
    for m in METRICS:
        y = rp[f"d_{m}"].to_numpy(float)
        X = np.column_stack([(g == 1).astype(float), (g == 2).astype(float), rp["age"] - 28, (rp["age"] - 28) ** 2, rp["last_mpg"] - 30])
        ok = np.isfinite(y) & np.isfinite(X).all(axis=1)
        b, se, _ = ols(y[ok], X[ok], cluster=rp["player_id"].to_numpy()[ok])
        rows.append(dict(metric=m, partial=b[1], partial_se=se[1], missed=b[2], missed_se=se[2], control_mean=rp.loc[g == 0, f"d_{m}"].mean()))
    return pd.DataFrame(rows)


def group_means(rp: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for gname, d in rp.groupby("group"):
        for m in METRICS:
            x = d[f"d_{m}"].dropna()
            rows.append(dict(group=gname, metric=m, mean=x.mean(), se=x.std(ddof=1) / np.sqrt(len(x)), n=len(x), age=d["age"].mean(), last_mpg=d["last_mpg"].mean()))
    return pd.DataFrame(rows)


def ramp_up(box: pd.DataFrame, min_gap: int = 15, ref=(21, 40), min_ref_games: int = 15) -> pd.DataFrame:
    """
    A 'return' is a player's first appearance after missing >= min_gap consecutive team games (including the start of a season),
    for players who stayed on one team that season. For each return, minutes and possessions per 36 in windows of games played
    after the return, relative to the same player's reference window (games 21-40 after the return).
    """
    one_team = box.groupby(["year", "player_id"])["team_id"].nunique()
    keep = one_team[one_team == 1].index
    b = box.set_index(["year", "player_id"]).loc[keep].reset_index().sort_values(["year", "player_id", "tgn"])
    b["gap"] = b.groupby(["year", "player_id"])["tgn"].diff().fillna(b["tgn"]) - 1          # team games missed before this appearance
    rows = []
    for (y, pid), g in b.groupby(["year", "player_id"]):
        idx = np.flatnonzero(g["gap"].to_numpy() >= min_gap)
        if len(idx) == 0:
            continue
        mins = g["min"].to_numpy(); poss = g["poss"].to_numpy(); gaps = g["gap"].to_numpy()
        for j in idx:
            nxt = idx[idx > j]
            end = nxt[0] if len(nxt) else len(g)                                              # the next layoff ends this stretch
            seg_m, seg_p = mins[j:end], poss[j:end]
            lo, hi = ref
            if len(seg_m) < lo - 1 + min_ref_games:
                continue
            refm = seg_m[lo - 1:hi]; refp = seg_p[lo - 1:hi]
            if refm.mean() < 12:
                continue
            rec = dict(year=y, player_id=pid, gap=gaps[j], ref_mpg=refm.mean(), ref_u36=refp.sum() / refm.sum() * 36, pre_mpg=np.nan,
                       season_start=int(g["tgn"].iloc[j] - gaps[j] == 1))
            pre = g.iloc[max(0, j - 20):j]
            if len(pre) >= 10 and j > 0:
                rec["pre_mpg"] = pre["min"].mean()
            for a, c in WINDOWS:
                s = seg_m[a - 1:c]
                if len(s):
                    rec[f"m_{a}_{c}"] = s.mean() / refm.mean()
                    rec[f"u_{a}_{c}"] = (seg_p[a - 1:c].sum() / s.sum() * 36) / rec["ref_u36"]
            rows.append(rec)
    return pd.DataFrame(rows)


def ramp_summary(ru: pd.DataFrame) -> pd.DataFrame:
    """Mean minutes and usage per 36 (as a share of the player's own steady level) by window of games since the return."""
    rows = []
    for a, c in WINDOWS[:-1]:
        for kind, col in (("minutes", f"m_{a}_{c}"), ("possessions per 36", f"u_{a}_{c}")):
            x = ru[col].dropna()
            rows.append(dict(window=f"games {a}-{c}", games=c - a + 1, kind=kind, mean=x.mean(), se=x.std(ddof=1) / np.sqrt(len(x)), n=len(x)))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------------------------------------------------
# B / C. roster change and positional depth
# ---------------------------------------------------------------------------------------------------------------------
def roster_table(box: pd.DataFrame, conn: sqlite3.Connection, k: int = K_OPEN) -> pd.DataFrame:
    """
    Opening roster (appeared in the first k games of the team) for every team-year, with each player's role last season
    (per-game minutes and possessions at any team, tapered below 40 games), whether he stayed with the team, his position group
    and the role-minutes of his teammates split by same / other position group and stayers / arrivals.
    """
    first = box[box["tgn"] <= k][["year", "team_id", "player_id"]].drop_duplicates()
    tot = box.groupby(["year", "player_id"]).agg(min=("min", "sum"), poss=("poss", "sum"), gp=("min", "size")).reset_index()
    f = (tot["gp"] / 40).clip(upper=1.0)
    tot["role_min"] = tot["min"] / tot["gp"] * f
    tot["role_poss"] = tot["poss"] / tot["gp"] * f
    prev = tot[["year", "player_id", "role_min", "role_poss"]].copy(); prev["year"] += 1
    ro = first.merge(prev, on=["year", "player_id"], how="left")
    stay = box.groupby(["year", "team_id", "player_id"]).size().reset_index()[["year", "team_id", "player_id"]]
    stay["year"] += 1; stay["stay"] = 1.0
    ro = ro.merge(stay, on=["year", "team_id", "player_id"], how="left")
    ro["stay"] = ro["stay"].fillna(0.0)
    ro[["role_min", "role_poss"]] = ro[["role_min", "role_poss"]].fillna(0.0)
    pos = pd.read_sql_query("SELECT player_id, position_1 FROM player", conn)
    ro["grp"] = ro["player_id"].map(dict(zip(pos["player_id"], pos["position_1"].map(pos_group)))).fillna("F")
    ro["m_stay"] = ro["role_min"] * ro["stay"]
    ro["m_arr"] = ro["role_min"] * (1 - ro["stay"])
    key = ["year", "team_id"]
    tot_t = ro.groupby(key)[["m_stay", "m_arr"]].sum().rename(columns=lambda c: f"T_{c}")
    tot_g = ro.groupby(key + ["grp"])[["m_stay", "m_arr"]].sum().rename(columns=lambda c: f"G_{c}").reset_index()
    ro = ro.merge(tot_t.reset_index(), on=key).merge(tot_g, on=key + ["grp"])
    for c in ("m_stay", "m_arr"):
        ro[f"{c}_same"] = ro[f"G_{c}"] - ro[c]
        ro[f"{c}_other"] = ro[f"T_{c}"] - ro[f"G_{c}"]
    return ro.drop(columns=["T_m_stay", "T_m_arr", "G_m_stay", "G_m_arr"]).drop_duplicates(["year", "player_id"], keep="first")


def hi_usage_departures(box: pd.DataFrame, thr: float = HI_USG, min_gp: int = 20, min_mpg: float = 20.0) -> pd.DataFrame:
    """Per (year T, team): possessions per game vacated by players who played for the team in T-1 (>= min_gp games, >= min_mpg, USG% >= thr) and are not on its opening roster in T."""
    pt = team_usage(box)
    ros = box[box["tgn"] <= K_OPEN][["year", "team_id", "player_id"]].drop_duplicates().assign(on=1)
    dep = pt.assign(year=pt["year"] + 1).merge(ros, on=["year", "team_id", "player_id"], how="left")
    dep = dep[dep["on"].isna() & (dep["gp"] >= min_gp) & (dep["mpg"] >= min_mpg)]
    dep["hi"] = (dep["usg"] >= thr).astype(float)
    dep["vposs"] = dep["poss"] / dep["gp"]
    dep["vposs_hi"] = dep["vposs"] * dep["hi"]
    return dep.groupby(["year", "team_id"]).agg(vhi=("vposs_hi", "sum"), vall=("vposs", "sum"), n_hi=("hi", "sum")).reset_index()


def stayers_frame(P: pd.DataFrame, cdf: pd.DataFrame, box: pd.DataFrame, ro: pd.DataFrame,
                  min_prev_mpg: float = 15.0, min_prev_gp: int = 30, min_now: float = 500.0) -> pd.DataFrame:
    """Players who stayed with their team and were rotation players last year: how their usage and minutes changed, and what happened around them."""
    cur = P[P["min"] >= min_now][["player_id", "year", "mpg", "usg", "pts36", "ast36", "tov36", "ts", "age", "pos", "name"]]
    pr = P[(P["mpg"] >= min_prev_mpg) & (P["gp"] >= min_prev_gp)][["player_id", "year", "mpg", "usg", "ts", "pts36", "ast36", "tov36"]].copy()
    pr["year"] += 1
    pr.columns = ["player_id", "year"] + [f"{c}_p" for c in ["mpg", "usg", "ts", "pts36", "ast36", "tov36"]]
    d = cur.merge(pr, on=["player_id", "year"]).merge(cdf, on=["player_id", "year"]).merge(
        ro[["year", "player_id", "team_id", "grp"]], on=["year", "player_id"])
    d = d[d["ctx_moved"] == 0].copy()
    tt = box.groupby(["year", "team_id"]).agg(tposs=("poss", "sum"), ng=("tgn", "max")).reset_index()
    tt["year"] += 1                                         # last year's team totals, attached to this year
    tt["tposs_pg"] = tt["tposs"] / tt["ng"]
    d = d.merge(tt[["year", "team_id", "tposs_pg"]], on=["year", "team_id"], how="left")
    d = d.merge(hi_usage_departures(box), on=["year", "team_id"], how="left").fillna({"vhi": 0.0, "vall": 0.0, "n_hi": 0.0})
    d["vac"] = d["ctx_vac_poss"] / d["tposs_pg"]; d["inc"] = d["ctx_inc_poss"] / d["tposs_pg"]
    d["vac_same"] = d["ctx_vac_poss_pos"] / d["tposs_pg"]; d["vac_other"] = d["vac"] - d["vac_same"]
    d["inc_same"] = d["ctx_inc_poss_pos"] / d["tposs_pg"]; d["inc_other"] = d["inc"] - d["inc_same"]
    d["vhi_share"] = d["vhi"] / d["tposs_pg"]
    d["voth_share"] = (d["vall"] - d["vhi"]) / d["tposs_pg"]            # other rotation-level departures (< HI_USG usage)
    d["lost_hi"] = (d["n_hi"] > 0).astype(float)
    for m in ("mpg", "usg", "ts", "pts36", "ast36", "tov36"):
        d[f"d_{m}"] = d[m] - d[f"{m}_p"]
    d["tyear"] = d["team_id"].astype(str) + "_" + d["year"].astype(str)
    d["usg_rank"] = d.groupby("tyear")["usg_p"].rank(ascending=False, method="first")
    return d.dropna(subset=["vac", "inc", "usg_p"])


def regress(d: pd.DataFrame, y: str, xs: list[str], cluster: str = "tyear") -> pd.DataFrame:
    """OLS of y on xs plus age, age squared, last year's minutes and usage; standard errors clustered by team-season."""
    d = d.assign(age28=d["age"] - 28, age28sq=(d["age"] - 28) ** 2, mpg_p25=d["mpg_p"] - 25, usg_p18=d["usg_p"] - 18)
    cols = xs + ["age28", "age28sq", "mpg_p25", "usg_p18"]
    ok = d[[y] + cols].notna().all(axis=1)
    b, se, r2 = ols(d.loc[ok, y].to_numpy(float), d.loc[ok, cols].to_numpy(float), cluster=d.loc[ok, cluster].to_numpy())
    return pd.DataFrame({"term": ["const"] + cols, "coef": b, "se": se}).assign(n=int(ok.sum()), r2=r2).set_index("term")


def crowding_models(oos_resid: pd.DataFrame, ro: pd.DataFrame, outcomes: list[str]) -> pd.DataFrame:
    """
    Does positional crowding explain what the projection missed? Regress each residual (actual - projected, per game) on the role-minutes
    of same-position and other-position teammates who stayed or arrived (in full-time-player equivalents of 36 minutes), by position group.
    """
    m = oos_resid.merge(ro[["year", "player_id", "team_id", "grp", "m_stay_same", "m_stay_other", "m_arr_same", "m_arr_other"]], on=["year", "player_id"])
    m["tyear"] = m["team_id"].astype(str) + "_" + m["year"].astype(str)
    xs = []
    for c in ("m_stay_same", "m_stay_other", "m_arr_same", "m_arr_other"):
        m[c + "36"] = m[c] / 36
        xs.append(c + "36")
    rows = []
    for grp in ("G", "F", "C", "all"):
        s = m if grp == "all" else m[m["grp"] == grp]
        for y in outcomes:
            ok = s[[y] + xs].notna().all(axis=1)
            b, se, r2 = ols(s.loc[ok, y].to_numpy(float), s.loc[ok, xs].to_numpy(float), cluster=s.loc[ok, "tyear"].to_numpy())
            rows.append(dict(grp=grp, outcome=y, n=int(ok.sum()), same=b[1], same_se=se[1], other=b[2], other_se=se[2], sd=float(s.loc[ok, y].std()),
                             x_sd=float(s.loc[ok, xs[0]].std()), r2=r2))
    return pd.DataFrame(rows), m


# ---------------------------------------------------------------------------------------------------------------------
# E. when the team's top-usage player is out, where do his minutes and possessions go?
# ---------------------------------------------------------------------------------------------------------------------
def star_out(box: pd.DataFrame, min_out: int = 8, min_in: int = 20) -> pd.DataFrame:
    """
    Per team-season: the star is the player with the most possessions per team game among those who played at least half the games
    for that team only. Teammates are those who also stayed with the team all season. For every teammate we compare his average minutes and
    possessions over ALL team games (0 when he did not play) in games the star missed against games the star played, so that the changes
    add up to what the star's absence frees. Teammates are ranked by season minutes (1 = most).
    """
    one = box.groupby(["year", "player_id"])["team_id"].nunique()
    rows = []
    for (y, t), g in box.groupby(["year", "team_id"]):
        ng = int(g["tgn"].max())
        s = g.groupby("player_id").agg(gp=("min", "size"), poss=("poss", "sum"), mn=("min", "sum"))
        cand = s[(s["gp"] >= 0.5 * ng)]
        cand = cand[[one.get((y, p), 2) == 1 for p in cand.index]]
        if cand.empty:
            continue
        star = (cand["poss"] / ng).idxmax()
        out = sorted(set(range(1, ng + 1)) - set(g.loc[g["player_id"] == star, "tgn"]))
        if len(out) < min_out or ng - len(out) < min_in:
            continue
        mates = [p for p in s.index if p != star and one.get((y, p), 2) == 1]
        gm = g[g["player_id"].isin(mates)]
        flag = np.array([k in set(out) for k in range(1, ng + 1)])
        mn = gm.pivot_table(index="tgn", columns="player_id", values="min", aggfunc="sum").reindex(range(1, ng + 1)).fillna(0)
        ps = gm.pivot_table(index="tgn", columns="player_id", values="poss", aggfunc="sum").reindex(range(1, ng + 1)).fillna(0)
        dm = mn[flag].mean() - mn[~flag].mean(); dp = ps[flag].mean() - ps[~flag].mean()
        rank = s.loc[mates, "mn"].rank(ascending=False, method="first")
        rows.append(pd.DataFrame({"year": y, "team_id": t, "pid": mates, "rank_min": rank.to_numpy(), "d_min": dm.reindex(mates).to_numpy(), "d_poss": dp.reindex(mates).to_numpy(),
                                  "star": star, "star_mpg": s.loc[star, "mn"] / s.loc[star, "gp"], "star_ppg": s.loc[star, "poss"] / s.loc[star, "gp"], "n_out": len(out)}))
    return pd.concat(rows, ignore_index=True)


def star_summary(so: pd.DataFrame) -> pd.DataFrame:
    """Share of the star's minutes and possessions picked up, by teammate group (mean over team-seasons, with standard errors)."""
    so = so.assign(bucket=pd.cut(so["rank_min"], [0, 4, 8, 30], labels=["next four by minutes", "rotation (5th-8th)", "deep bench (9th and below)"]))
    g = so.groupby(["year", "team_id", "bucket"], observed=True).agg(dm=("d_min", "sum"), dp=("d_poss", "sum"), sm=("star_mpg", "first"), sp=("star_ppg", "first")).reset_index()
    g["minutes"] = g["dm"] / g["sm"]; g["possessions"] = g["dp"] / g["sp"]
    rows = []
    for b, d in g.groupby("bucket", observed=True):
        for kind in ("minutes", "possessions"):
            rows.append(dict(bucket=b, kind=kind, mean=d[kind].mean(), se=d[kind].std(ddof=1) / np.sqrt(len(d)), n=len(d)))
    tot = g.groupby(["year", "team_id"]).agg(dm=("dm", "sum"), sm=("sm", "first"), dp=("dp", "sum"), sp=("sp", "first"))
    rows.append(dict(bucket="all teammates", kind="minutes", mean=(tot["dm"] / tot["sm"]).mean(), se=(tot["dm"] / tot["sm"]).std(ddof=1) / np.sqrt(len(tot)), n=len(tot)))
    rows.append(dict(bucket="all teammates", kind="possessions", mean=(tot["dp"] / tot["sp"]).mean(), se=(tot["dp"] / tot["sp"]).std(ddof=1) / np.sqrt(len(tot)), n=len(tot)))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------------------------------------------------
# everything the section needs
# ---------------------------------------------------------------------------------------------------------------------
def run_all(ctx, resid: pd.DataFrame) -> dict:
    box = load_box(ctx.conn)
    P = season_panel(box, ctx.st, ctx.apg)
    out: dict = {"box_rows": len(box)}
    # A
    rp = returners_panel(P)
    out["rp"], out["rp_means"], out["rp_adj"] = rp, group_means(rp), returners_adjusted(rp)
    ru = ramp_up(box)
    out["ru"], out["ramp"] = ru, ramp_summary(ru)
    ru = ru.assign(ratio_pre=ru["ref_mpg"] / ru["pre_mpg"])
    out["ramp_pre"] = ru["ratio_pre"].dropna()
    out["ramp_lost_games"] = float(sum(r.games * (1 - r["mean"]) for _, r in out["ramp"][out["ramp"]["kind"] == "minutes"].iterrows()))
    # B / C
    cdf = team_context(ctx.conn, years=list(range(2016, 2026)))
    ro = roster_table(box, ctx.conn)
    d = stayers_frame(P, cdf, box, ro)
    d["vac10"] = d["vac"] * 10; d["vhi10"] = d["vhi_share"] * 10; d["voth10"] = d["voth_share"] * 10; d["inc10"] = d["inc"] * 10
    out["stayers"] = d
    out["reg_all"] = {y: regress(d, y, ["vac10", "inc10"]) for y in ("d_usg", "d_mpg")}
    out["reg_hi"] = {y: regress(d, y, ["vhi10", "voth10"]) for y in ("d_usg", "d_mpg", "d_pts36")}
    tiers = pd.cut(d["usg_rank"], [0, 1, 3, 6, 30], labels=["top usage on the team", "2nd-3rd", "4th-6th", "7th and below"])
    rows = []
    for tier, g in d.groupby(tiers, observed=True):
        r = regress(g, "d_usg", ["lost_hi"]).loc["lost_hi"]
        rows.append(dict(tier=tier, n=len(g), coef=r["coef"], se=r["se"]))
    out["tier_effect"] = pd.DataFrame(rows)
    rot = team_usage(box); rot = rot[(rot["gp"] >= 20) & (rot["mpg"] >= 20)]
    out["hi_share_of_rotation"] = float((rot["usg"] >= HI_USG).mean())
    out["share_lost_hi"] = float(d["lost_hi"].mean())
    pos_rows = []
    for grp in ("G", "F", "C"):
        sub = d[d["grp"] == grp].assign(vs=lambda x: x["vac_same"] * 10, vo=lambda x: x["vac_other"] * 10, is_=lambda x: x["inc_same"] * 10, io=lambda x: x["inc_other"] * 10)
        for y in ("d_mpg", "d_usg"):
            r = regress(sub, y, ["vs", "vo", "is_", "io"])
            for term, lab in (("vs", "vacated, same position"), ("vo", "vacated, other positions")):
                pos_rows.append(dict(grp=grp, outcome=y, term=lab, coef=r.loc[term, "coef"], se=r.loc[term, "se"], n=int(r["n"].iloc[0])))
    out["pos_vacated"] = pd.DataFrame(pos_rows)
    cm, m = crowding_models(resid, ro, ["mpg_res", "pts_res", "reb_res", "ast_res", "blk_res", "tov_res"])
    out["crowd"], out["crowd_frame"] = cm, m
    # E
    so = star_out(box)
    out["so"], out["so_sum"] = so, star_summary(so)
    out["so_seasons"] = int(so.groupby(["year", "team_id"]).ngroups)
    out["so_games_out"] = float(so.groupby(["year", "team_id"])["n_out"].first().median())
    return out
