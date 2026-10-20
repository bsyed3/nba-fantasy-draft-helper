"""
Phase 1 -- next-season per-game projections.

What gets modelled (see src/features/player_seasons.py for the definitions):
  * per-36 rates for FGA2, FGA3, FTA, REB, AST, STL, BLK, TOV
  * 2P%, 3P%, FT% (makes / attempts)
  * minutes per game and availability (games / team games)
Per-game FGM, 3PM, FTM, FG%, PTS are *derived* from those, never modelled
on their own.

For each target stat three candidate predictors compete in a rolling-origin
backtest (train on seasons < T, test on season T, for the last several T):

  baseline  recency-weighted, shrunken (empirical-Bayes style) average of
            the player's last three seasons. The decay and the shrinkage
            strength k are grid-searched on the training seasons only.
  ridge     baseline + Ridge on the residual, using age/experience/draft
            slot/minutes features. (The percentage stats use Ridge only,
            never trees.)
  lgbm      baseline + LightGBM on the residual (count/volume stats only).

The winner per stat -- lowest weighted out-of-sample MSE -- is used for the
real projection, and the backtest table is saved so the choice is auditable.
Out-of-sample residuals are also returned: Phase 2 uses them to size the
"talent uncertainty" around each projection.

Rookies / players with no NBA history use draft-tier buckets: the mean
profile and the coefficient of variation of historical first seasons by
draft slot.
"""
from __future__ import annotations

import itertools
import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.linear_model import QuantileRegressor, RidgeCV

from src.features.player_seasons import (
    ALL_STATS, PCT_STATS, RATE_STATS, STAT_DEFS, TOTAL_COLS, league_priors,
)

LAGS = 3
DECAY_GRID = [0.4, 0.6, 0.8, 1.0]
K_GRID = [0, 50, 100, 250, 500, 1000, 2000, 4000]
PCT_K_GRID = [0, 25, 50, 100, 200, 400, 800]
MIN_EVAL_MINUTES = 500          # target-season minutes required to count in backtest metrics
MIN_TRAIN_MINUTES = 100
# Players with no season last year (a whole missed season, so lag 1 is empty): the Ridge/LightGBM availability models
# learn from fringe players who vanish and come back and project returning rotation players too low (0.48 vs 0.60
# actual, backtest 2021-25, n=17); the shrunken baseline overshoots (0.68). Their average matched (MSE 0.056 vs 0.070 raw).
RETURNER_BASELINE_WEIGHT = 0.5
DRAFT_TIERS = [(1, 5, "1-5"), (6, 14, "6-14"), (15, 30, "15-30"), (31, 60, "31-60"), (61, 999, "undrafted")]

warnings.filterwarnings("ignore", message=".*X does not have valid feature names.*")


def draft_tier(pick: float) -> str:
    for lo, hi, name in DRAFT_TIERS:
        if lo <= pick <= hi:
            return name
    return "undrafted"


# ---------------------------------------------------------------------------
# Pooling / shrinkage
# ---------------------------------------------------------------------------

def pooled_ratio(tot: np.ndarray, den: np.ndarray, prior: np.ndarray, decay: float, k: float) -> np.ndarray:
    """
    Recency-weighted ratio of totals across lags, shrunk toward `prior`.
    tot/den: [n, LAGS] (NaN = season missing), lag 1 = most recent.
    result = (sum_l w_l*tot_l + k*prior) / (sum_l w_l*den_l + k)
    """
    w = decay ** np.arange(tot.shape[1])
    num = (np.nan_to_num(tot) * w).sum(1) + k * prior
    dn = (np.nan_to_num(den) * w).sum(1) + k
    with np.errstate(invalid="ignore", divide="ignore"):
        out = np.where(dn > 0, num / dn, prior)
    return out


def _lag_matrix(wide: pd.DataFrame, col: str) -> np.ndarray:
    return np.column_stack([wide[f"{col}_{l}"].to_numpy(float) for l in range(1, LAGS + 1)])


def stat_arrays(wide: pd.DataFrame, stat: str) -> tuple[np.ndarray, np.ndarray, float]:
    num, den, scale = STAT_DEFS[stat]
    return _lag_matrix(wide, num), _lag_matrix(wide, den), scale


def blend(wide: pd.DataFrame, stat: str, decay: float, k: float) -> np.ndarray:
    tot, den, scale = stat_arrays(wide, stat)
    prior = wide[f"prior_{stat}"].to_numpy(float)
    return pooled_ratio(tot, den, prior / scale, decay, k) * scale


# ---------------------------------------------------------------------------
# Wide (one row per player x target year) feature table
# ---------------------------------------------------------------------------

def build_wide(st: pd.DataFrame, priors: pd.DataFrame, target_year: int, season_games: dict[int, int],
               context: pd.DataFrame | None = None) -> pd.DataFrame:
    """
    Rows: every player with at least one season in [T-3, T-1].
    Columns: lagged totals (`<col>_<lag>`), `prior_<stat>` (league ratio for
    the player's position group in season T-1), age/experience/draft
    features, and -- if season T exists in `st` -- targets `y_<stat>` with
    weights `w_<stat>`. Availability is games played / team games after his
    first appearance: games missed at the very start of a season (known
    at tip-off) and seasons never played are not targets -- they come from
    the known-absences input, not from the model.
    """
    T = target_year
    lag_frames = []
    for l in range(1, LAGS + 1):
        sub = st[st["year"] == T - l][["player_id", "age"] + TOTAL_COLS]
        lag_frames.append(sub.rename(columns={c: f"{c}_{l}" for c in ["age"] + TOTAL_COLS}))
    ids = pd.concat([f[["player_id"]] for f in lag_frames]).drop_duplicates()
    wide = ids
    for f in lag_frames:
        wide = wide.merge(f, on="player_id", how="left")

    bio = (
        st.sort_values("year").groupby("player_id").tail(1)
        [["player_id", "name", "pos", "draft_pick", "rookie_year", "last_year"]]
    )
    wide = wide.merge(bio, on="player_id", how="left")

    wide["year"] = T
    wide["exp"] = (T - wide["rookie_year"]).clip(lower=0)
    # age at the start of T, from the most recent lag with an age
    age_t = pd.Series(np.nan, index=wide.index)
    for l in range(LAGS, 0, -1):
        a = wide[f"age_{l}"] + l
        age_t = a.where(a.notna(), age_t)
    wide["age_t"] = age_t
    wide["n_lags"] = sum(wide[f"gp_{l}"].notna().astype(int) for l in range(1, LAGS + 1))
    wide["gap"] = np.select(
        [wide["gp_1"].notna(), wide["gp_2"].notna()], [0, 1], default=2
    )
    wide["sg_t"] = season_games.get(T, 82)

    if context is not None:      # team context (src/features/team_context.py): vacated / incoming minutes and possessions, moved
        wide = wide.merge(context[context["year"] == T].drop(columns="year"), on="player_id", how="left")

    for stat in ALL_STATS:
        pri = priors.loc[T - 1] if (T - 1) in priors.index.get_level_values(0) else priors.groupby(level=1).last()
        wide[f"prior_{stat}"] = wide["pos"].map(pri[stat])

    cur = st[st["year"] == T].set_index("player_id")
    if len(cur):
        for stat, (num, den, scale) in STAT_DEFS.items():
            n = wide["player_id"].map(cur[num]) if num != "gp" else wide["player_id"].map(cur["gp"])
            d = wide["player_id"].map(cur[den]) if den != "sg" else wide["sg_t"]
            with np.errstate(invalid="ignore", divide="ignore"):
                wide[f"y_{stat}"] = n / d * scale
            wide[f"w_{stat}"] = d.fillna(0.0)
        wide["min_t"] = wide["player_id"].map(cur["min"]).fillna(0.0)
        wide["gp_t"] = wide["player_id"].map(cur["gp"]).fillna(0.0)
        wide["smiss_t"] = wide["player_id"].map(cur["smiss"])        # games missed before his first appearance (NaN = never played)
        wide["sgo_t"] = wide["player_id"].map(cur["sgo"])
    return wide


def feature_matrix(wide: pd.DataFrame, params: dict[str, tuple[float, float]], exp_flags: bool = False) -> pd.DataFrame:
    """Shared ML features: blends of every stat, per-lag usage, age/experience/draft. `exp_flags` adds season-of-career indicators."""
    X = pd.DataFrame(index=wide.index)
    for stat in ALL_STATS:
        X[f"bl_{stat}"] = blend(wide, stat, *params[stat])
    for l in range(1, LAGS + 1):
        X[f"mpg_{l}"] = wide[f"min_{l}"] / wide[f"gp_{l}"]
        X[f"avail_{l}"] = wide[f"gp_{l}"] / wide[f"sgo_{l}"]                 # in-season availability (after his first appearance)
        X[f"open_{l}"] = wide[f"smiss_{l}"] / wide[f"sg_{l}"]                # share of that season missed before his first game
        X[f"min_{l}"] = wide[f"min_{l}"]
    X["age"] = wide["age_t"]
    X["age2"] = wide["age_t"] ** 2
    X["exp"] = wide["exp"]
    X["draft_pick"] = wide["draft_pick"]
    X["is_big"] = (wide["pos"] == "C").astype(float)
    X["is_guard"] = (wide["pos"] == "G").astype(float)
    X["n_lags"] = wide["n_lags"]
    X["gap"] = wide["gap"]
    if exp_flags:    # lets the models treat 2nd/3rd/4th-year players (and where they were drafted) differently from veterans
        dp = wide["draft_pick"].fillna(61.0).clip(upper=61.0)
        for e in (1, 2, 3):
            X[f"is_exp{e}"] = (wide["exp"] == e).astype(float)
            X[f"pick_x_exp{e}"] = X[f"is_exp{e}"] * dp
    if "ctx_vac_min" in wide.columns:        # team context, when supplied
        for c in ("ctx_moved", "ctx_vac_min", "ctx_inc_min", "ctx_vac_poss", "ctx_inc_poss",
                  "ctx_vac_min_pos", "ctx_inc_min_pos", "ctx_vac_poss_pos", "ctx_inc_poss_pos"):
            X[c] = wide[c]
        X["ctx_net_min"] = wide["ctx_vac_min"] - wide["ctx_inc_min"]
        X["ctx_net_poss"] = wide["ctx_vac_poss"] - wide["ctx_inc_poss"]
        X["ctx_net_min_pos"] = wide["ctx_vac_min_pos"] - wide["ctx_inc_min_pos"]
        X["ctx_net_min_x_mpg"] = X["ctx_net_min_pos"] * X["mpg_1"].fillna(X["mpg_2"]).fillna(X["mpg_3"]) / 36.0   # a vacancy matters for rotation players
        # stars absorb vacated shots, role players don't: scale the team's net vacated possessions by the player's own usage
        gp = _latest(wide, "gp")
        poss = (_latest(wide, "fga2") + _latest(wide, "fga3") + 0.44 * _latest(wide, "fta") + _latest(wide, "tov")) / gp   # possessions used per game, latest season
        u = (poss / 15.0).fillna(0.0)
        moved = wide["ctx_moved"].fillna(0.0)
        X["ctx_net_poss_x_usage"] = X["ctx_net_poss"] * u
        X["ctx_net_poss_pos_x_usage"] = (wide["ctx_vac_poss_pos"] - wide["ctx_inc_poss_pos"]) * u
        X["ctx_moved_x_usage"] = moved * u
        X["ctx_moved_x_netposs"] = moved * X["ctx_net_poss"]
        X["usage_pg"] = poss
    return X


def _latest(wide: pd.DataFrame, base: str) -> pd.Series:
    """Most recent available lag of a total (`<base>_1`, else `_2`, else `_3`)."""
    out = pd.Series(np.nan, index=wide.index)
    for l in range(LAGS, 0, -1):
        out = wide[f"{base}_{l}"].where(wide[f"{base}_{l}"].notna(), out)
    return out


# ---------------------------------------------------------------------------
# Baseline parameter search + model candidates
# ---------------------------------------------------------------------------

def fit_baseline_params(wide_train: pd.DataFrame) -> dict[str, tuple[float, float]]:
    """Grid-search (decay, k) per stat on training rows only."""
    params = {}
    for stat in ALL_STATS:
        y = wide_train[f"y_{stat}"].to_numpy(float)
        w = wide_train[f"w_{stat}"].to_numpy(float)
        ok = np.isfinite(y) & (w > 0)
        k_grid = PCT_K_GRID if stat in PCT_STATS else K_GRID
        if stat in ("mpg", "avail"):
            k_grid = [0, 2, 5, 10, 20, 40]  # exposure here is games, not minutes
        best, best_err = (1.0, 0.0), np.inf
        for decay, k in itertools.product(DECAY_GRID, k_grid):
            p = blend(wide_train, stat, decay, k)
            err = np.nansum(w[ok] * (y[ok] - p[ok]) ** 2) / w[ok].sum()
            if err < best_err:
                best, best_err = (decay, k), err
        params[stat] = best
    return params


def _train_mask(wide: pd.DataFrame) -> np.ndarray:
    """Rows usable for training: player was in the league at T (played, or avail=0 absent)."""
    return ((wide["min_t"] >= MIN_TRAIN_MINUTES) | (wide["w_avail"] > 0)).to_numpy()


def _clip(stat: str, p: np.ndarray) -> np.ndarray:
    if stat in PCT_STATS or stat == "avail":
        return np.clip(p, 0.0, 1.0)
    if stat == "mpg":
        return np.clip(p, 0.0, 42.0)
    return np.clip(p, 0.0, None)


def _fit_predict(model: str, stat: str, X_tr, resid_tr, w_tr, X_te) -> np.ndarray:
    """Fit a residual model; returns predicted residual for X_te."""
    if model == "ridge":
        med = X_tr.median()
        mu = X_tr.fillna(med).mean()
        sd = X_tr.fillna(med).std().replace(0, 1).fillna(1)
        f = lambda X: ((X.fillna(med) - mu) / sd).to_numpy()
        m = RidgeCV(alphas=np.logspace(0, 4, 13)).fit(f(X_tr), resid_tr, sample_weight=w_tr)
        return m.predict(f(X_te))
    m = LGBMRegressor(
        n_estimators=250, learning_rate=0.03, num_leaves=6, min_child_samples=40,
        subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=5.0,
        random_state=0, verbose=-1,
    )
    m.fit(X_tr, resid_tr, sample_weight=w_tr)
    return m.predict(X_te)


def candidates_for(stat: str) -> list[str]:
    return ["baseline", "ridge"] if stat in PCT_STATS else ["baseline", "ridge", "lgbm"]


def predict_all(wide_train: pd.DataFrame, wide_test: pd.DataFrame, only_baseline: bool = False, drop: dict[str, tuple[str, ...]] | None = None,
                exp_flags: bool = False) -> dict[str, dict[str, np.ndarray]]:
    """
    {stat: {candidate: predictions for wide_test}} trained on wide_train.
    drop: {stat: feature-name prefixes} to leave out of that stat's ML models (e.g. {"avail": ("ctx_",)}), for experiments.
    exp_flags: add the season-of-career indicators of `feature_matrix`.
    """
    params = fit_baseline_params(wide_train)
    mask = _train_mask(wide_train)
    tr = wide_train[mask]
    X_tr_all = feature_matrix(wide_train, params, exp_flags)[mask]
    X_te = feature_matrix(wide_test, params, exp_flags)
    out: dict[str, dict[str, np.ndarray]] = {}
    for stat in ALL_STATS:
        base_te = X_te[f"bl_{stat}"].to_numpy()
        res = {"baseline": _clip(stat, base_te)}
        if not only_baseline:
            y = tr[f"y_{stat}"].to_numpy(float)
            w = tr[f"w_{stat}"].to_numpy(float)
            ok = np.isfinite(y) & (w > 0)
            resid = (y - X_tr_all[f"bl_{stat}"].to_numpy())[ok]
            wt = w[ok] / w[ok].mean()
            prefixes = tuple((drop or {}).get(stat, ()))
            cols = [c for c in X_tr_all.columns if not (prefixes and c.startswith(prefixes))]
            for model in candidates_for(stat)[1:]:
                r = _fit_predict(model, stat, X_tr_all[cols][ok], resid, wt, X_te[cols])
                res[model] = _clip(stat, base_te + r)
            if stat == "avail":
                returner = (wide_test["gap"] > 0).to_numpy()
                for model in res:
                    if model != "baseline":
                        res[model] = np.where(returner, RETURNER_BASELINE_WEIGHT * res["baseline"]
                                              + (1 - RETURNER_BASELINE_WEIGHT) * res[model], res[model])
        out[stat] = res
    return out


# ---------------------------------------------------------------------------
# Per-game assembly
# ---------------------------------------------------------------------------

def to_per_game(rates: pd.DataFrame) -> pd.DataFrame:
    """
    rates: columns for every stat in ALL_STATS (per-36 rates, pct, mpg, avail).
    Returns per-game counting stats with FGM/3PM/FTM/PTS/FG% derived.
    """
    mpg = rates["mpg"]
    pg = pd.DataFrame(index=rates.index)
    for s in RATE_STATS:
        pg[s] = rates[s] / 36.0 * mpg  # fga2, fga3, fta, reb, ast, stl, blk, tov (per game)
    pg["fga"] = pg["fga2"] + pg["fga3"]
    pg["tpa"] = pg["fga3"]
    pg["tpm"] = pg["fga3"] * rates["p3"]
    pg["fgm"] = pg["fga2"] * rates["p2"] + pg["tpm"]
    pg["ftm"] = pg["fta"] * rates["ft"]
    pg["pts"] = 2 * pg["fgm"] + pg["tpm"] + pg["ftm"]
    pg["fg_pct"] = pg["fgm"] / pg["fga"].replace(0, np.nan)
    pg["ft_pct"] = rates["ft"]
    pg["mpg"] = mpg
    pg["avail"] = rates["avail"]
    return pg


# ---------------------------------------------------------------------------
# Rolling-origin backtest
# ---------------------------------------------------------------------------

@dataclass
class BacktestResult:
    metrics: pd.DataFrame      # fold x stat x model weighted errors
    summary: pd.DataFrame      # stat x model pooled errors, winner flagged
    selection: dict[str, str]  # stat -> winning model
    oos: pd.DataFrame          # selected-model out-of-sample per-game predictions + actuals
    avail_eval: pd.DataFrame | None = None   # selection-free availability predictions vs actuals (absentees = 0), established players


GP_QUANTILES = (0.25, 0.5, 0.75)


def fit_gp_quantiles(avail_eval: pd.DataFrame, qs=GP_QUANTILES, min_mpg: float = 20.0) -> dict[float, tuple[float, float]]:
    """
    The projected availability is a MEAN, and games played are left-skewed (season-ending injuries), so a typical
    player plays more than the mean says. Linear quantile fits of actual availability on projected availability
    (rotation players, backtest folds, absentees counted as 0) turn the mean into a P25 / median / P75 range for display.
    The draft engine keeps using the mean: the weekly simulation generates the injury runs itself.
    """
    d = avail_eval[(avail_eval["pred_mpg"] >= min_mpg) & avail_eval["act_avail"].notna()]
    coefs = {}
    for q in qs:
        m = QuantileRegressor(quantile=q, alpha=0.0, solver="highs").fit(d[["pred_avail"]], d["act_avail"])
        coefs[float(q)] = (float(m.intercept_), float(m.coef_[0]))
    return coefs


def apply_gp_quantiles(avail: np.ndarray | pd.Series, coefs: dict[float, tuple[float, float]], games: int = 82) -> pd.DataFrame:
    """Games-played quantiles (out of `games`) for projected mean availability; ordered p25 <= median <= p75 and never below zero."""
    a = np.asarray(avail, float)
    cols = {}
    prev = np.zeros_like(a)
    for q in sorted(coefs):
        b0, b1 = coefs[q]
        prev = np.maximum(np.clip(b0 + b1 * a, 0.0, 1.0), prev)
        cols[q] = prev * games
    return pd.DataFrame({"gp_p25": cols[0.25], "gp_median": cols[0.5], "gp_p75": cols[0.75]},
                        index=avail.index if isinstance(avail, pd.Series) else None)


def drop_seasons(st: pd.DataFrame, ignore: pd.DataFrame | None) -> pd.DataFrame:
    """
    Treat chosen (player_id, year) seasons as if they never happened -- e.g. a season of rest while angling for a trade, which says
    nothing about the player's health or role. The row is removed and the same player's EARLIER seasons slide forward one year (and
    one year older), so the model reads his last real seasons as lags 1, 2, 3 instead of seeing a missed year (which it treats as an
    injury). Use it for the history of the player being projected only, never for training targets.
    """
    if ignore is None or len(ignore) == 0:
        return st
    st = st.copy()
    for pid, year in zip(ignore["player_id"], ignore["year"]):
        drop = (st["player_id"] == pid) & (st["year"] == year)
        if not drop.any():
            continue
        st = st[~drop].copy()
        earlier = (st["player_id"] == pid) & (st["year"] < year)
        st.loc[earlier, "year"] += 1
        if "age" in st.columns:
            st.loc[earlier, "age"] += 1
    return st


def sync_rates_from_per_game(proj: pd.DataFrame, idx) -> pd.DataFrame:
    """
    After hand-setting per-game values (mpg, fga, fgm, tpa, tpm, fta, ftm, reb, ast, stl, blk, tov) for the rows `idx`, bring the columns the
    simulation reads (fga2, fga3, r_p2, r_p3, r_ft and the per-36 `r_*` rates) in step, so what is displayed is what is simulated.
    """
    for i in idx:
        mpg, fga, tpa = proj.at[i, "mpg"], proj.at[i, "fga"], proj.at[i, "tpa"]
        fga3, fga2 = tpa, max(fga - tpa, 0.0)
        proj.at[i, "fga3"], proj.at[i, "fga2"] = fga3, fga2
        if fga3 > 0:
            proj.at[i, "r_p3"] = min(max(proj.at[i, "tpm"] / fga3, 0.0), 1.0)
        if fga2 > 0:
            proj.at[i, "r_p2"] = min(max((proj.at[i, "fgm"] - proj.at[i, "tpm"]) / fga2, 0.0), 1.0)
        if proj.at[i, "fta"] > 0:
            proj.at[i, "r_ft"] = min(max(proj.at[i, "ftm"] / proj.at[i, "fta"], 0.0), 1.0)
        if mpg > 0:
            for s in RATE_STATS:
                proj.at[i, f"r_{s}"] = proj.at[i, s] / mpg * 36.0
        proj.at[i, "r_mpg"], proj.at[i, "r_avail"] = mpg, proj.at[i, "avail"]
    return proj


def apply_external_rookies(proj: pd.DataFrame, ext: pd.DataFrame) -> pd.DataFrame:
    """
    Replace the draft-slot profile of rookies we have an outside projection for (columns name, pts, reb, ast, stl, blk, fg_pct, tp_pct, mpg;
    anything missing keeps the model's value). Points, FG% and 3P% are matched exactly by solving for shot volume with the profile's own
    shot mix (3PA share, FTA per FGA, FT%); free throws, turnovers and availability stay the draft-slot profile's. Only rows still on the
    draft-tier path are touched, and every stat the simulation reads is rewritten consistently.
    """
    from src.features.names import normalize_name

    proj = proj.copy()
    keys = proj["name"].map(normalize_name)
    n = 0
    for _, r in ext.iterrows():
        idx = proj.index[(keys == normalize_name(r["name"])) & (proj["source"] == "rookie_tier")]
        if len(idx) != 1:
            print(f"  external rookie projection skipped ({len(idx)} draft-tier rows match): {r['name']}")
            continue
        i = idx[0]
        mpg = float(r["mpg"]) if pd.notna(r.get("mpg")) else float(proj.at[i, "mpg"])
        fga0 = float(proj.at[i, "fga"])
        s3, ratio, ft = proj.at[i, "fga3"] / fga0, proj.at[i, "fta"] / fga0, float(proj.at[i, "r_ft"])
        fg = float(r["fg_pct"]) if pd.notna(r.get("fg_pct")) else float(proj.at[i, "fg_pct"])
        p3 = float(r["tp_pct"]) if pd.notna(r.get("tp_pct")) else float(proj.at[i, "r_p3"])
        fga = float(r["pts"]) / (2 * fg + s3 * p3 + ratio * ft)                      # PTS = FGA * (2 FG% + 3PA-share * 3P% + FTA/FGA * FT%)
        fga3, fta = s3 * fga, ratio * fga
        fga2 = fga - fga3
        p2 = float(np.clip((fg * fga - p3 * fga3) / fga2, 0.30, 0.75))
        rates = pd.DataFrame([{
            "fga2": fga2 / mpg * 36, "fga3": fga3 / mpg * 36, "fta": fta / mpg * 36,
            **{s: float(r[s]) / mpg * 36 for s in ("reb", "ast", "stl", "blk")}, "tov": float(proj.at[i, "r_tov"]),
            "p2": p2, "p3": p3, "ft": ft, "mpg": mpg, "avail": float(proj.at[i, "avail"])}])
        pg = to_per_game(rates).iloc[0]
        for s in ALL_STATS:
            proj.at[i, f"r_{s}"] = rates.iloc[0][s]
        for c in ("fga2", "fga3", "fta", "reb", "ast", "stl", "blk", "tov", "fga", "tpa", "tpm", "fgm", "ftm", "pts", "fg_pct", "ft_pct", "mpg", "avail"):
            proj.at[i, c] = pg[c]
        proj.at[i, "source"] = "external_rookie"
        n += 1
    print(f"  applied {n} external rookie projections")
    return proj


def apply_known_absences(proj: pd.DataFrame, path, season_games: int = 82) -> pd.DataFrame:
    """
    Games a player is already known to miss at the START of the season (an injury or suspension known at draft time).
    The model projects availability for a player who is active at tip-off (`avail_active`: games played / team games after his
    first appearance); a known absence scales it: avail = avail_active * (82 - games_out) / 82. games_out = 82 means out for the
    season. `path` is a CSV (or a list of CSVs, applied in order -- a later file wins for the same player, so a hand-edited file
    can override the one generated from ESPN) with `player_id` or `name` and `games_out`; without any, everyone is assumed active.
    """
    from pathlib import Path

    proj = proj.copy()
    proj["avail_active"] = proj["avail"]
    proj["games_out"] = 0.0
    n = 0
    for p in ([] if path is None else [path] if isinstance(path, (str, Path)) else list(path)):
        if not Path(p).exists():
            continue
        ka = pd.read_csv(p)
        key = "player_id" if "player_id" in ka.columns else "name"
        lookup = proj[key].str.lower() if key == "name" else proj[key]
        for _, row in ka.iterrows():
            k = str(row[key]).lower() if key == "name" else row[key]
            idx = proj.index[lookup == k]
            if len(idx) != 1:
                print(f"  known absence skipped (matched {len(idx)} players): {row[key]}")
                continue
            proj.loc[idx, "games_out"] = float(min(max(row["games_out"], 0), season_games))
            n += 1
    proj["avail"] = proj["avail_active"] * (season_games - proj["games_out"]) / season_games
    print(f"  applied {n} known absences")
    return proj


def _pooled_summary(metrics: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, str]]:
    g = metrics.groupby(["stat", "model"]).agg(sse=("sse", "sum"), sw=("sw", "sum"), n=("n", "sum")).reset_index()
    g["wmse"] = g["sse"] / g["sw"]
    g["rmse"] = np.sqrt(g["wmse"])
    base = g[g["model"] == "baseline"].set_index("stat")["wmse"]
    g["vs_baseline_pct"] = (g["wmse"] / g["stat"].map(base) - 1) * 100
    selection = g.loc[g.groupby("stat")["wmse"].idxmin()].set_index("stat")["model"].to_dict()
    g["selected"] = [selection[s] == m for s, m in zip(g["stat"], g["model"])]
    return g.drop(columns=["sse", "sw"]), selection


def backtest(st: pd.DataFrame, folds: list[int], verbose: bool = True, context: pd.DataFrame | None = None,
             drop: dict[str, tuple[str, ...]] | None = None, exp_flags: bool = False) -> BacktestResult:
    """
    Train on seasons < T, predict season T, for each T in `folds`. `context` = team-context features (optional);
    `drop` / `exp_flags` switch features off/on for experiments (see `predict_all`).
    """
    priors = league_priors(st)
    season_games = st.groupby("year")["sg"].first().to_dict()
    first_year = int(st["year"].min())

    wides = {T: build_wide(st, priors, T, season_games, context) for T in range(first_year + 1, max(folds) + 1)}
    metrics_rows = []
    fold_preds: dict[int, dict[str, dict[str, np.ndarray]]] = {}
    for T in folds:
        train = pd.concat([wides[t] for t in range(first_year + 1, T)], ignore_index=True)
        test = wides[T]
        preds = predict_all(train, test, drop=drop, exp_flags=exp_flags)
        fold_preds[T] = preds
        for stat in ALL_STATS:
            y = test[f"y_{stat}"].to_numpy(float)
            w = test[f"w_{stat}"].to_numpy(float)
            # availability is judged on established players (>=500 min last year) who played that season (games after their first appearance);
            # everything else on players who logged >= 500 minutes in the target season
            relevant = (test["min_1"].fillna(0) >= MIN_EVAL_MINUTES) if stat == "avail" else (test["min_t"] >= MIN_EVAL_MINUTES)
            ok = np.isfinite(y) & (w > 0) & relevant.to_numpy()
            for model, p in preds[stat].items():
                err = (y[ok] - p[ok]) ** 2
                metrics_rows.append(dict(fold=T, stat=stat, model=model, sse=float((w[ok] * err).sum()),
                                         sw=float(w[ok].sum()), n=int(ok.sum())))
        if verbose:
            print(f"  fold {T}: train rows {len(train)}, test rows {len(test)}")
    metrics = pd.DataFrame(metrics_rows)
    summary, selection = _pooled_summary(metrics)

    # Out-of-sample per-game predictions from the *selected* model per stat,
    # next to what actually happened (per game), for Phase 2 calibration.
    act = st[["player_id", "year", "gp", "min"]].copy()
    for c in RATE_STATS:
        act[f"a_{c}"] = st[c] / st["gp"] if c in st else np.nan
    act["a_fga2"] = st["fga2"] / st["gp"]
    act["a_fga3"] = st["fga3"] / st["gp"]
    act["a_p2"] = st["fgm2"] / st["fga2"].replace(0, np.nan)
    act["a_p3"] = st["tpm"] / st["fga3"].replace(0, np.nan)
    act["a_ft"] = st["ftm"] / st["fta"].replace(0, np.nan)
    act["a_mpg"] = st["min"] / st["gp"]
    act["att2"], act["att3"], act["attft"] = st["fga2"], st["fga3"], st["fta"]
    oos_parts = []
    for T in folds:
        test = wides[T]
        rates = pd.DataFrame({s_: fold_preds[T][s_][selection[s_]] for s_ in ALL_STATS}, index=test.index)
        pg = to_per_game(rates)
        pg = pd.concat([rates.add_prefix("r_"), pg], axis=1)
        pg.insert(0, "player_id", test["player_id"].values)
        pg.insert(1, "year", T)
        for c in ["exp", "pos", "draft_pick"]:
            pg[c] = test[c].values
        oos_parts.append(pg)
    oos = pd.concat(oos_parts, ignore_index=True).merge(act, on=["player_id", "year"], how="inner")
    oos = oos[oos["min"] >= MIN_EVAL_MINUTES].reset_index(drop=True)

    # availability predictions vs actuals for established players, WITHOUT conditioning on the target season (absentees = 0)
    ev = []
    for T in folds:
        test = wides[T]
        ok = (test["min_1"].fillna(0) >= MIN_EVAL_MINUTES) & test["y_avail"].notna() & (test["w_avail"] > 0)
        ev.append(pd.DataFrame({"year": T, "player_id": test.loc[ok, "player_id"].to_numpy(),
                                "pred_avail": fold_preds[T]["avail"][selection["avail"]][ok.to_numpy()],
                                "pred_mpg": fold_preds[T]["mpg"][selection["mpg"]][ok.to_numpy()],
                                "act_avail": test.loc[ok, "y_avail"].to_numpy(),
                                "act_gp": test.loc[ok, "gp_t"].to_numpy(), "act_smiss": test.loc[ok, "smiss_t"].to_numpy(),
                                # for subgroup diagnostics (returners, movers, age): the recency baseline next to the model, and who the player is
                                "base_avail": fold_preds[T]["avail"]["baseline"][ok.to_numpy()], "base_mpg": fold_preds[T]["mpg"]["baseline"][ok.to_numpy()],
                                "act_mpg": test.loc[ok, "y_mpg"].to_numpy(), "gap": test.loc[ok, "gap"].to_numpy(), "exp": test.loc[ok, "exp"].to_numpy(),
                                "age": test.loc[ok, "age_t"].to_numpy(),
                                "moved": test.loc[ok, "ctx_moved"].fillna(0).to_numpy() if "ctx_moved" in test.columns else 0.0}))
    return BacktestResult(metrics, summary, selection, oos, pd.concat(ev, ignore_index=True))


# ---------------------------------------------------------------------------
# Rookie / no-history profiles
# ---------------------------------------------------------------------------

def rookie_profiles(st: pd.DataFrame, min_minutes: float = 200) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Mean rate profile and per-game coefficient of variation of historical
    first NBA seasons, bucketed by draft slot. Only seasons in the data
    window where year == rookie_year count (veterans already in the league
    in the first ingested season are not rookies).
    """
    r = st[(st["year"] == st["rookie_year"]) & (st["min"] >= min_minutes)].copy()
    r["tier"] = r["draft_pick"].map(draft_tier)
    means, cvs = [], []
    for tier, grp in r.groupby("tier"):
        row = {"tier": tier, "n": len(grp)}
        for stat, (num, den, scale) in STAT_DEFS.items():
            row[stat] = grp[num].sum() / grp[den].sum() * scale
        means.append(row)
        pg = pd.DataFrame({
            "fga2": grp["fga2"] / grp["gp"], "fga3": grp["fga3"] / grp["gp"], "fta": grp["fta"] / grp["gp"],
            "reb": grp["reb"] / grp["gp"], "ast": grp["ast"] / grp["gp"], "stl": grp["stl"] / grp["gp"],
            "blk": grp["blk"] / grp["gp"], "tov": grp["tov"] / grp["gp"],
            "p2": grp["fgm2"] / grp["fga2"].replace(0, np.nan), "p3": grp["tpm"] / grp["fga3"].replace(0, np.nan),
            "ft": grp["ftm"] / grp["fta"].replace(0, np.nan),
        })
        crow = {"tier": tier}
        for s in RATE_STATS:
            crow[s] = float(pg[s].std() / pg[s].mean()) if pg[s].mean() > 0 else np.nan
        for s in PCT_STATS:
            crow[s] = float(pg[s].std())  # absolute sd for percentages
        cvs.append(crow)
    order = [name for _, _, name in DRAFT_TIERS]
    means = pd.DataFrame(means).set_index("tier").reindex(order)
    cvs = pd.DataFrame(cvs).set_index("tier").reindex(order)
    return means, cvs


# ---------------------------------------------------------------------------
# Final projection
# ---------------------------------------------------------------------------

RECENT_MIN_MINUTES = 300.0     # a player with this many minutes last season belongs in the draft pool even if no roster lists him today


def add_recent_players(players: pd.DataFrame, st: pd.DataFrame, target_year: int,
                       min_minutes: float = RECENT_MIN_MINUTES) -> tuple[pd.DataFrame, set[int]]:
    """
    The projection pool is "on a current roster" (nba_api PlayerIndex, last_year >= target_year). That index lags: a free agent, or a
    player who was signed or traded after the last data refresh, still shows last_year = target_year - 1 and would silently be missing
    from the draft. Treat anyone with at least `min_minutes` in the last season as in the pool. Returns the players table with those
    rows' last_year raised, and their ids (flagged in the projections as not currently rostered; the team shown is the last one known).
    """
    played = st[(st["year"] == target_year - 1) & (st["min"] >= min_minutes)]["player_id"]
    stale = players["last_year"].eq(target_year - 1) & players["player_id"].isin(set(played))
    out = players.copy()
    out.loc[stale, "last_year"] = target_year
    return out, set(int(p) for p in players.loc[stale, "player_id"])


def project_season(st: pd.DataFrame, players: pd.DataFrame, target_year: int, selection: dict[str, str],
                   context: pd.DataFrame | None = None, ignore_seasons: pd.DataFrame | None = None,
                   no_context_ids: set[int] | None = None) -> pd.DataFrame:
    """
    Per-game projections for `target_year` for every player with
    last_year >= target_year. Model path: players with NBA history in the
    last three seasons. Rookie path: everyone else, by draft tier.
    `players` needs [player_id, name, position_1, draft_year, draft_position,
    rookie_year, last_year]. `context` = team-context features (must match
    what the selection was backtested with). `ignore_seasons` = (player_id, year) seasons to treat as if they never happened for the
    PROJECTED players' histories (see drop_seasons); training data is left alone. `no_context_ids` = players whose projection should NOT
    use team context (a judgment that the team change shouldn't matter for him): their rows come from a context-free run of the same models.
    """
    from src.features.player_seasons import pos_group

    priors = league_priors(st)
    season_games = st.groupby("year")["sg"].first().to_dict()
    first_year = int(st["year"].min())
    train = pd.concat(
        [build_wide(st, priors, t, season_games, context) for t in range(first_year + 1, target_year)], ignore_index=True
    )
    test = build_wide(drop_seasons(st, ignore_seasons), priors, target_year, season_games, context)
    test = test[test["player_id"].isin(players.loc[players["last_year"] >= target_year, "player_id"])].copy()
    preds = predict_all(train, test)
    rates = pd.DataFrame({s_: preds[s_][selection.get(s_, "baseline")] for s_ in ALL_STATS}, index=test.index)
    model_rows = pd.concat(
        [test[["player_id", "exp", "age_t", "draft_pick", "pos"]].reset_index(drop=True),
         rates.add_prefix("r_").reset_index(drop=True), to_per_game(rates).reset_index(drop=True)], axis=1)
    model_rows["source"] = "model"

    # rookies / no recent NBA history: draft-tier profile
    means, _ = rookie_profiles(st)
    have = set(test["player_id"])
    rook = players[(players["last_year"] >= target_year) & (~players["player_id"].isin(have))].copy()
    rook["pos"] = rook["position_1"].map(pos_group)
    rook["draft_pick"] = rook["draft_position"].astype(float).fillna(61.0)
    rook["exp"] = (target_year - rook["rookie_year"]).clip(lower=0)
    # A draft slot only says something about a player in his first season. Someone drafted years ago who has no NBA
    # history in the window (an overseas returnee, a long-time G-League player) gets the undrafted-tier profile.
    tier_pick = np.where(rook["exp"] == 0, rook["draft_pick"], 61.0)
    r_rates = means.loc[pd.Series(tier_pick).map(draft_tier).to_numpy(), ALL_STATS].reset_index(drop=True)
    rookie_rows = pd.concat(
        [rook[["player_id", "exp", "draft_pick", "pos"]].reset_index(drop=True),
         r_rates.add_prefix("r_"), to_per_game(r_rates)], axis=1)
    rookie_rows["age_t"] = np.nan
    rookie_rows["source"] = "rookie_tier"

    out = pd.concat([model_rows, rookie_rows], ignore_index=True)
    out = out.merge(players[["player_id", "name", "position_1", "position_2"]], on="player_id", how="left")
    out["tier"] = out["draft_pick"].map(draft_tier)
    out["year"] = target_year
    if no_context_ids and context is not None:
        alt = project_season(st, players, target_year, selection, context=None, ignore_seasons=ignore_seasons).set_index("player_id")
        swap = out["player_id"].isin(no_context_ids) & (out["source"] == "model") & out["player_id"].isin(alt.index)
        cols = [c for c in out.columns if c.startswith("r_")] + ["fga2", "fga3", "fta", "reb", "ast", "stl", "blk", "tov", "fga", "tpa", "tpm", "fgm",
                                                              "ftm", "pts", "fg_pct", "ft_pct", "mpg", "avail"]
        out.loc[swap, cols] = alt.loc[out.loc[swap, "player_id"].to_numpy(), cols].to_numpy()
    return out
