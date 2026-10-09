"""Computations behind the report (kept separate from the narrative so they can be tested/reused)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from src.draft.values import CATEGORIES, z_scores

CAT_COLS = ["pts", "tpm", "reb", "ast", "stl", "blk", "tov", "fg_pct", "ft_pct"]


def actual_per_game(st: pd.DataFrame) -> pd.DataFrame:
    """Per-game box-score values (same names as the projection frame) for every player-season in `st`."""
    gp = st["gp"]
    out = pd.DataFrame({"player_id": st["player_id"], "year": st["year"], "gp": gp, "min": st["min"]})
    out["fgm"] = (st["fgm2"] + st["tpm"]) / gp
    out["fga"] = (st["fga2"] + st["fga3"]) / gp
    out["tpm"] = st["tpm"] / gp
    out["ftm"] = st["ftm"] / gp
    out["fta"] = st["fta"] / gp
    out["pts"] = 2 * out["fgm"] + out["tpm"] + out["ftm"]
    for c in ("reb", "ast", "stl", "blk", "tov"):
        out[c] = st[c] / gp
    out["fg_pct"] = out["fgm"] / out["fga"]
    out["ft_pct"] = out["ftm"] / out["fta"].replace(0, np.nan)
    out["mpg"] = st["min"] / gp
    out["avail"] = gp / st["sg"]
    return out


def with_z(df: pd.DataFrame, n_pool: int = 156) -> pd.DataFrame:
    """Append z_<cat> and z_total (and a 1-based rank) computed within this set of players."""
    z = z_scores(df.reset_index(drop=True), n_pool=n_pool)
    out = pd.concat([df.reset_index(drop=True), z], axis=1)
    out["rank"] = out["z_total"].rank(ascending=False, method="first").astype(int)
    return out


def accuracy_table(oos: pd.DataFrame, apg: pd.DataFrame) -> pd.DataFrame:
    """
    Model vs 'same as last season' on the nine category inputs, per-game, out of sample.
    Rows: players with >= 500 minutes in the target season AND a previous season (so both
    predictors exist for the same player-seasons).
    """
    cur = apg.set_index(["player_id", "year"])
    prev = apg.assign(year=apg["year"] + 1).set_index(["player_id", "year"])
    m = oos.set_index(["player_id", "year"])
    idx = m.index.intersection(cur.index).intersection(prev.index)
    rows = []
    for c in CAT_COLS:
        y, p, n = cur.loc[idx, c], m.loc[idx, c], prev.loc[idx, c]
        ok = y.notna() & p.notna() & n.notna()
        y, p, n = y[ok], p[ok], n[ok]
        mae_m, mae_n = (y - p).abs().mean(), (y - n).abs().mean()
        r2_m = 1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum()
        r2_n = 1 - ((y - n) ** 2).sum() / ((y - y.mean()) ** 2).sum()
        rows.append(dict(stat=c, n=int(ok.sum()), mean_actual=y.mean(), mae_model=mae_m, mae_last_season=mae_n,
                         mae_gain_pct=(1 - mae_m / mae_n) * 100, r2_model=r2_m, r2_last_season=r2_n))
    return pd.DataFrame(rows)


def ranking_accuracy(oos: pd.DataFrame, apg: pd.DataFrame, top: int = 50) -> pd.DataFrame:
    """
    Per backtest season: how well does the projected fantasy value (z_total) rank players vs what they
    actually produced? Compared with ranking by last season's actual production.
    """
    rows = []
    cur_all = apg.set_index(["player_id", "year"])
    for year, grp in oos.groupby("year"):
        ids = grp["player_id"].to_numpy()
        prev = apg[apg["year"] == year - 1].set_index("player_id")
        cur = apg[apg["year"] == year].set_index("player_id")
        keep = [i for i in ids if i in prev.index and i in cur.index]
        if len(keep) < 80:
            continue
        pred = with_z(grp.set_index("player_id").loc[keep].reset_index()[["player_id", "avail", "mpg"] + CAT_COLS + ["fgm", "fga", "ftm", "fta"]])
        act = with_z(cur.loc[keep].reset_index())
        naive = with_z(prev.loc[keep].reset_index())
        a = act.set_index("player_id")["z_total"]
        p = pred.set_index("player_id")["z_total"].loc[a.index]
        n = naive.set_index("player_id")["z_total"].loc[a.index]
        k = min(top, len(a) // 4)
        top_a = set(a.nlargest(k).index)
        rows.append(dict(year=int(year), n=len(a),
                         spearman_model=spearmanr(p, a)[0], spearman_last_season=spearmanr(n, a)[0],
                         top_overlap_model=len(top_a & set(p.nlargest(k).index)) / k,
                         top_overlap_last_season=len(top_a & set(n.nlargest(k).index)) / k, k=k))
    return pd.DataFrame(rows)


def movers(proj: pd.DataFrame, apg: pd.DataFrame, last_year: int, min_minutes: float = 1000,
           hold_avail: bool = False) -> pd.DataFrame:
    """
    Projected 2026-27 value vs actual last-season value for the same players, both z-scored within that
    same player set so they are directly comparable. hold_avail=True sets availability to 1 in BOTH frames,
    isolating the change in per-game production from the change in games played.
    """
    last = apg[(apg["year"] == last_year) & (apg["min"] >= min_minutes)].copy()
    both = proj[proj["player_id"].isin(last["player_id"])].copy()
    if hold_avail:
        last["avail"], both["avail"] = 1.0, 1.0
    cols = ["player_id", "avail", "mpg"] + CAT_COLS + ["fgm", "fga", "ftm", "fta"]
    a = with_z(last[cols + ["gp"]].reset_index(drop=True)).set_index("player_id")
    p = with_z(both[cols].reset_index(drop=True)).set_index("player_id")
    ids = p.index.intersection(a.index)
    out = pd.DataFrame(index=ids)
    out["name"] = proj.set_index("player_id").loc[ids, "name"]
    out["age"] = proj.set_index("player_id").loc[ids, "age_t"]
    out["elig"] = proj.set_index("player_id").loc[ids, "elig"]
    out["z_last"], out["z_proj"] = a.loc[ids, "z_total"], p.loc[ids, "z_total"]
    out["rank_last"], out["rank_proj"] = a.loc[ids, "rank"], p.loc[ids, "rank"]
    out["dz"] = out["z_proj"] - out["z_last"]
    out["drank"] = out["rank_last"] - out["rank_proj"]            # positive = moves up
    out["mpg_last"], out["mpg_proj"] = a.loc[ids, "mpg"], p.loc[ids, "mpg"]
    out["gp_last"] = a.loc[ids, "gp"]
    out["gp_proj"] = proj.set_index("player_id").loc[ids, "avail"] * 82
    for c in CATEGORIES:
        out[f"dz_{c}"] = p.loc[ids, f"z_{c}"] - a.loc[ids, f"z_{c}"]
    return out.reset_index()


def driver_text(row: pd.Series, labels: dict[str, str], n: int = 2) -> str:
    """The categories whose z-score changed most (signed), e.g. '+AST, -BLK'."""
    d = {c: row[f"dz_{c}"] for c in CATEGORIES}
    top = sorted(d, key=lambda c: -abs(d[c]))[:n]
    return ", ".join(f"{'+' if d[c] > 0 else '-'}{labels[c]}" for c in top)


def pit_values(sim_week_vals: np.ndarray, actual: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """
    Randomised probability-integral transform of `actual` against its predictive draws.
    sim_week_vals [N, draws], actual [N]. Uniform(0,1) if the predictive distributions are calibrated.
    """
    lo = (sim_week_vals < actual[:, None]).mean(axis=1)
    hi = (sim_week_vals <= actual[:, None]).mean(axis=1)
    return lo + rng.random(len(actual)) * (hi - lo)


def short_name(name: str) -> str:
    """Last name, keeping generational suffixes: 'Gary Payton II' -> 'Payton II'."""
    parts = str(name).split()
    return " ".join(parts[-2:]) if len(parts) > 2 and parts[-1].strip(".") in {"II", "III", "IV", "Jr", "Sr"} else parts[-1]


def rookie_model_comparison(st: pd.DataFrame, min_minutes: float = 200) -> pd.DataFrame:
    """
    Leave-one-rookie-class-out comparison of two ways to project a rookie from his draft slot:
    bucket means ("tier", what the pipeline uses) vs a smooth weighted regression on log(pick) ("curve").
    Returns mean absolute error per game for several stats, averaged over held-out classes.
    """
    from src.features.player_seasons import STAT_DEFS
    from src.models import projection as P

    r = st[(st["year"] == st["rookie_year"]) & (st["min"] >= min_minutes)].copy()
    r["x"] = np.log(r["draft_pick"].clip(1, 61))
    act = pd.DataFrame({"mpg": r["min"] / r["gp"], "pts": (2 * (r["fgm2"] + r["tpm"]) + r["tpm"] + r["ftm"]) / r["gp"],
                        "reb": r["reb"] / r["gp"], "ast": r["ast"] / r["gp"], "stl": r["stl"] / r["gp"], "blk": r["blk"] / r["gp"]})

    def fit(df):
        out = {}
        for stat in P.ALL_STATS:
            num, den, sc = STAT_DEFS[stat]
            y, w = df[num] / df[den] * sc, df[den]
            ok = np.isfinite(y) & (w > 0)
            X = np.column_stack([np.ones(ok.sum()), df.loc[ok, "x"]])
            sw = np.sqrt(w[ok].to_numpy())
            out[stat] = np.linalg.lstsq(X * sw[:, None], y[ok].to_numpy() * sw, rcond=None)[0]
        return out

    rows = {"tier": [], "curve": []}
    for y in sorted(r["year"].unique()):
        test, train = r[r["year"] == y], r[r["year"] != y]
        if len(test) < 10:
            continue
        coef = fit(train)
        rates = pd.DataFrame({s_: coef[s_][0] + coef[s_][1] * test["x"].to_numpy() for s_ in P.ALL_STATS}, index=test.index)
        rates[["p2", "p3", "ft", "avail"]] = rates[["p2", "p3", "ft", "avail"]].clip(0.02, 0.98)
        rates[P.RATE_STATS + ["mpg"]] = rates[P.RATE_STATS + ["mpg"]].clip(lower=0.0)
        means, _ = P.rookie_profiles(train)
        tr = means.loc[test["draft_pick"].map(P.draft_tier), P.ALL_STATS].reset_index(drop=True)
        tr.index = test.index
        for name, pr in (("curve", P.to_per_game(rates)), ("tier", P.to_per_game(tr))):
            rows[name].append({c: (act.loc[test.index, c] - pr[c]).abs().mean() for c in act.columns})
    return pd.DataFrame({k: pd.DataFrame(v).mean() for k, v in rows.items()}).T


RESID_COLS = CAT_COLS + ["mpg"]


def residual_frame(oos: pd.DataFrame, st: pd.DataFrame, apg: pd.DataFrame, names: dict | None = None) -> pd.DataFrame:
    """
    One row per out-of-sample player-season with `<c>_pred`, `<c>_act`, `<c>_res` (= actual - predicted;
    positive = the model under-projected him) for the nine categories and minutes per game, plus age,
    actual minutes, experience and position group.
    """
    pred = oos[["player_id", "year", "exp", "pos"] + RESID_COLS].rename(columns={c: f"{c}_pred" for c in RESID_COLS})
    act = apg[["player_id", "year", "min"] + RESID_COLS].rename(columns={c: f"{c}_act" for c in RESID_COLS})
    out = pred.merge(act, on=["player_id", "year"]).merge(st[["player_id", "year", "age"]], on=["player_id", "year"], how="left")
    for c in RESID_COLS:
        out[f"{c}_res"] = out[f"{c}_act"] - out[f"{c}_pred"]
    out["name"] = out["player_id"].map(names or {}).fillna(out["player_id"].astype(str))
    return out


def binned_mean(x: pd.Series, y: pd.Series, bins: int = 10) -> pd.DataFrame:
    """Mean of y within quantile bins of x, with a 95% interval on the mean."""
    d = pd.DataFrame({"x": x, "y": y}).dropna()
    d["bin"] = pd.qcut(d["x"], bins, duplicates="drop")
    g = d.groupby("bin", observed=True).agg(x=("x", "mean"), mean=("y", "mean"), sd=("y", "std"), n=("y", "size"))
    g["se"] = g["sd"] / np.sqrt(g["n"])
    g["lo"], g["hi"] = g["mean"] - 1.96 * g["se"], g["mean"] + 1.96 * g["se"]
    return g.reset_index(drop=True)


def residual_summary(rf: pd.DataFrame) -> pd.DataFrame:
    """Per category: bias, spread, shape, tail share and whether the error grows with the projected level."""
    from scipy.stats import kurtosis, skew, spearmanr

    rows = []
    for c in RESID_COLS:
        r = rf[f"{c}_res"].dropna()
        sd = r.std()
        pred = rf.loc[r.index, f"{c}_pred"]
        rows.append(dict(stat=c, n=len(r), bias=r.mean(), bias_se=sd / np.sqrt(len(r)), sd=sd, mae=r.abs().mean(),
                         skew=skew(r), ex_kurt=kurtosis(r), tail2=(r.abs() > 2 * sd).mean(),
                         hetero_rho=spearmanr(pred, r.abs())[0], mean_actual=rf.loc[r.index, f"{c}_act"].mean()))
    return pd.DataFrame(rows)
