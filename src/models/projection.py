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
from sklearn.linear_model import RidgeCV

from src.features.player_seasons import (
    ALL_STATS, PCT_STATS, RATE_STATS, STAT_DEFS, TOTAL_COLS, league_priors,
)

LAGS = 3
DECAY_GRID = [0.4, 0.6, 0.8, 1.0]
K_GRID = [0, 50, 100, 250, 500, 1000, 2000, 4000]
PCT_K_GRID = [0, 25, 50, 100, 200, 400, 800]
MIN_EVAL_MINUTES = 500          # target-season minutes required to count in backtest metrics
MIN_TRAIN_MINUTES = 100
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

def build_wide(st: pd.DataFrame, priors: pd.DataFrame, target_year: int, season_games: dict[int, int]) -> pd.DataFrame:
    """
    Rows: every player with at least one season in [T-3, T-1].
    Columns: lagged totals (`<col>_<lag>`), `prior_<stat>` (league ratio for
    the player's position group in season T-1), age/experience/draft
    features, and -- if season T exists in `st` -- targets `y_<stat>` with
    weights `w_<stat>`. Players who were in the league at T (last_year >= T)
    but logged no minutes get avail=0 (season-ending injuries must count).
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

    for stat in ALL_STATS:
        pri = priors.loc[T - 1] if (T - 1) in priors.index.get_level_values(0) else priors.groupby(level=1).last()
        wide[f"prior_{stat}"] = wide["pos"].map(pri[stat])

    cur = st[st["year"] == T].set_index("player_id")
    if len(cur):
        for stat, (num, den, scale) in STAT_DEFS.items():
            n = wide["player_id"].map(cur[num]) if num != "gp" else wide["player_id"].map(cur["gp"])
            d = wide["player_id"].map(cur[den]) if den != "sg" else wide["sg_t"]
            if stat == "avail":
                n = n.fillna(0.0)  # in the league (last_year >= T) but never played => 0 games
                present = wide["player_id"].isin(cur.index) | (wide["last_year"] >= T)
                wide[f"y_{stat}"] = np.where(present, n / d, np.nan)
                wide[f"w_{stat}"] = np.where(present, d, 0.0)
            else:
                with np.errstate(invalid="ignore", divide="ignore"):
                    wide[f"y_{stat}"] = n / d * scale
                wide[f"w_{stat}"] = d.fillna(0.0)
        wide["min_t"] = wide["player_id"].map(cur["min"]).fillna(0.0)
        wide["gp_t"] = wide["player_id"].map(cur["gp"]).fillna(0.0)
    return wide


def feature_matrix(wide: pd.DataFrame, params: dict[str, tuple[float, float]]) -> pd.DataFrame:
    """Shared ML features: blends of every stat, per-lag usage, age/experience/draft."""
    X = pd.DataFrame(index=wide.index)
    for stat in ALL_STATS:
        X[f"bl_{stat}"] = blend(wide, stat, *params[stat])
    for l in range(1, LAGS + 1):
        X[f"mpg_{l}"] = wide[f"min_{l}"] / wide[f"gp_{l}"]
        X[f"avail_{l}"] = wide[f"gp_{l}"] / wide[f"sg_{l}"]
        X[f"min_{l}"] = wide[f"min_{l}"]
    X["age"] = wide["age_t"]
    X["age2"] = wide["age_t"] ** 2
    X["exp"] = wide["exp"]
    X["draft_pick"] = wide["draft_pick"]
    X["is_big"] = (wide["pos"] == "C").astype(float)
    X["is_guard"] = (wide["pos"] == "G").astype(float)
    X["n_lags"] = wide["n_lags"]
    X["gap"] = wide["gap"]
    return X


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


def predict_all(wide_train: pd.DataFrame, wide_test: pd.DataFrame, only_baseline: bool = False) -> dict[str, dict[str, np.ndarray]]:
    """{stat: {candidate: predictions for wide_test}} trained on wide_train."""
    params = fit_baseline_params(wide_train)
    mask = _train_mask(wide_train)
    tr = wide_train[mask]
    X_tr_all = feature_matrix(wide_train, params)[mask]
    X_te = feature_matrix(wide_test, params)
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
            for model in candidates_for(stat)[1:]:
                r = _fit_predict(model, stat, X_tr_all[ok], resid, wt, X_te)
                res[model] = _clip(stat, base_te + r)
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


def _pooled_summary(metrics: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, str]]:
    g = metrics.groupby(["stat", "model"]).agg(sse=("sse", "sum"), sw=("sw", "sum"), n=("n", "sum")).reset_index()
    g["wmse"] = g["sse"] / g["sw"]
    g["rmse"] = np.sqrt(g["wmse"])
    base = g[g["model"] == "baseline"].set_index("stat")["wmse"]
    g["vs_baseline_pct"] = (g["wmse"] / g["stat"].map(base) - 1) * 100
    selection = g.loc[g.groupby("stat")["wmse"].idxmin()].set_index("stat")["model"].to_dict()
    g["selected"] = [selection[s] == m for s, m in zip(g["stat"], g["model"])]
    return g.drop(columns=["sse", "sw"]), selection


def backtest(st: pd.DataFrame, folds: list[int], verbose: bool = True) -> BacktestResult:
    """Train on seasons < T, predict season T, for each T in `folds`."""
    priors = league_priors(st)
    season_games = st.groupby("year")["sg"].first().to_dict()
    first_year = int(st["year"].min())

    wides = {T: build_wide(st, priors, T, season_games) for T in range(first_year + 1, max(folds) + 1)}
    metrics_rows = []
    fold_preds: dict[int, dict[str, dict[str, np.ndarray]]] = {}
    for T in folds:
        train = pd.concat([wides[t] for t in range(first_year + 1, T)], ignore_index=True)
        test = wides[T]
        preds = predict_all(train, test)
        fold_preds[T] = preds
        for stat in ALL_STATS:
            y = test[f"y_{stat}"].to_numpy(float)
            w = test[f"w_{stat}"].to_numpy(float)
            # availability is judged on established players (>=500 min last year, absences count as 0);
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
    return BacktestResult(metrics, summary, selection, oos)


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

def project_season(st: pd.DataFrame, players: pd.DataFrame, target_year: int, selection: dict[str, str]) -> pd.DataFrame:
    """
    Per-game projections for `target_year` for every player with
    last_year >= target_year. Model path: players with NBA history in the
    last three seasons. Rookie path: everyone else, by draft tier.
    `players` needs [player_id, name, position_1, draft_year, draft_position,
    rookie_year, last_year].
    """
    from src.features.player_seasons import pos_group

    priors = league_priors(st)
    season_games = st.groupby("year")["sg"].first().to_dict()
    first_year = int(st["year"].min())
    train = pd.concat(
        [build_wide(st, priors, t, season_games) for t in range(first_year + 1, target_year)], ignore_index=True
    )
    test = build_wide(st, priors, target_year, season_games)
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
    return out
