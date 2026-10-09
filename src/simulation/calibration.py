"""
Phase 2 calibration -- everything the weekly simulation needs that can be
learned from history, estimated from the database / backtest residuals.

  dispersion   Negative Binomial size `r` per count stat (var = m + m^2/r),
               ratio-of-sums estimator over player-seasons' game-to-game
               variance. Includes minutes variability, which is the point.
  rho          Beta-Binomial intra-game overdispersion for 2P%/3P%/FT%
               (how much "hot shooting" clusters beyond plain binomial).
  corr         Gaussian-copula correlation matrix across the 11 per-game
               dimensions (8 counts + 3 shooting-luck residuals), from
               pooled normal scores of within-player-season standardized
               game logs.
  games_pmf    distribution of a team's games in a Mon-Sun fantasy week.
  absence_c    share of missed games that come from whole-week absences
               (injury runs) rather than scattered single-game misses.
  talent       relative sigma of the projection error (counts) / absolute
               sigma (percentages) by experience bucket, with the sampling
               noise of the observed season subtracted, and the correlation
               of those errors across stats.
"""
from __future__ import annotations

import json
import sqlite3
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

COUNT_DIMS = ["fga2", "fga3", "fta", "reb", "ast", "stl", "blk", "tov"]
PCT_DIMS = ["p2", "p3", "ft"]
DIMS = COUNT_DIMS + PCT_DIMS
EXP_BUCKETS = [(0, 1, "0-1"), (2, 4, "2-4"), (5, 99, "5+")]


def exp_bucket(exp: float) -> str:
    for lo, hi, name in EXP_BUCKETS:
        if lo <= exp <= hi:
            return name
    return "5+"


@dataclass
class Calibration:
    dispersion: dict[str, float]
    rho: dict[str, float]                      # per-game beta-binomial rho for p2, p3, ft
    corr: list[list[float]]                    # 11x11 copula correlation, order = DIMS
    games_pmf: dict[str, float]                # {"2": .., "3": .., ...}
    absence_c: float
    talent_sigma: dict[str, dict[str, float]]  # bucket -> dim -> sigma (rel for counts, abs for pct)
    talent_corr: list[list[float]]             # 11x11
    rookie_sigma: dict[str, dict[str, float]]  # draft tier -> dim -> sigma
    notes: dict = field(default_factory=dict)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(asdict(self), indent=2))

    @staticmethod
    def load(path: str | Path) -> "Calibration":
        return Calibration(**json.loads(Path(path).read_text()))


# ---------------------------------------------------------------------------

def nearest_correlation(m: np.ndarray, floor: float = 1e-4) -> np.ndarray:
    """Symmetrize, clip eigenvalues to make it PSD, re-normalize the diagonal to 1."""
    m = np.nan_to_num((m + m.T) / 2, nan=0.0)
    np.fill_diagonal(m, 1.0)
    vals, vecs = np.linalg.eigh(m)
    m = (vecs * np.clip(vals, floor, None)) @ vecs.T
    d = np.sqrt(np.diag(m))
    m = m / np.outer(d, d)
    return (m + m.T) / 2


def load_game_logs(conn: sqlite3.Connection, seasons: list[str]) -> pd.DataFrame:
    q = f"""
        SELECT g.season, bs.player_id, bs.min,
               bs.fga - bs.tpa AS fga2, bs.tpa AS fga3, bs.fta,
               bs.oreb + bs.dreb AS reb, bs.ast, bs.stl, bs.blk, bs.tov,
               bs.fgm - bs.tpm AS m2, bs.tpm AS m3, bs.ftm AS mf
        FROM box_score bs JOIN game g ON g.game_id = bs.game_id
        WHERE g.game_type = 'regular' AND bs.min > 0
          AND g.season IN ({",".join("?" * len(seasons))})
    """
    return pd.read_sql_query(q, conn, params=seasons)


def estimate_dispersion(logs: pd.DataFrame, min_games: int = 30) -> dict[str, float]:
    out = {}
    cnt = logs.groupby(["season", "player_id"])
    n = cnt["min"].transform("size")
    d = logs[n >= min_games]
    g = d.groupby(["season", "player_id"])
    for s in COUNT_DIMS:
        m, v = g[s].mean(), g[s].var(ddof=1)
        keep = m > 0.15
        excess = (v[keep] - m[keep]).sum()
        inv_r = excess / (m[keep] ** 2).sum()
        out[s] = float(np.clip(1.0 / inv_r, 0.5, 500.0)) if inv_r > 0 else 500.0
    return out


def estimate_rho(logs: pd.DataFrame, min_games: int = 30) -> dict[str, float]:
    """Method-of-moments Beta-Binomial rho per shooting type, attempts-weighted."""
    pairs = {"p2": ("m2", "fga2"), "p3": ("m3", "fga3"), "ft": ("mf", "fta")}
    n = logs.groupby(["season", "player_id"])["min"].transform("size")
    d = logs[n >= min_games]
    out = {}
    for name, (m, a) in pairs.items():
        g = d.groupby(["season", "player_id"])
        M, A = g[m].transform("sum"), g[a].transform("sum")
        q = (M / A.replace(0, np.nan)).clip(0.02, 0.98)
        att = d[a]
        resid2 = (d[m] - att * q) ** 2
        binvar = att * q * (1 - q)
        t = pd.DataFrame({"season": d["season"], "pid": d["player_id"], "resid2": resid2, "binvar": binvar,
                          "a": att, "a2": att ** 2}).dropna()
        s = t.groupby(["season", "pid"]).sum()
        phi = s["resid2"] / s["binvar"]
        a_eff = s["a2"] / s["a"]
        ok = (a_eff > 1.5) & (s["a"] > 150)
        rho = ((phi[ok] - 1) / (a_eff[ok] - 1))
        w = s["a"][ok]
        out[name] = float(np.clip((rho * w).sum() / w.sum(), 0.0, 0.25))
    return out


def estimate_copula_corr(logs: pd.DataFrame, min_games: int = 30, max_rows: int = 200_000, seed: int = 0) -> np.ndarray:
    """
    Gaussian-copula correlation across DIMS. Each game-log value is
    standardized within its player-season (removing between-player level
    differences), shooting dims become binomial-standardized make
    residuals, then pooled values are mapped to normal scores.
    """
    n = logs.groupby(["season", "player_id"])["min"].transform("size")
    d = logs[n >= min_games].copy()
    g = d.groupby(["season", "player_id"])
    for name, (m, a) in {"p2": ("m2", "fga2"), "p3": ("m3", "fga3"), "ft": ("mf", "fta")}.items():
        q = (g[m].transform("sum") / g[a].transform("sum").replace(0, np.nan)).clip(0.02, 0.98)
        e = (d[m] - d[a] * q) / np.sqrt(d[a] * q * (1 - q))
        d[name] = e.where(d[a] >= 2)
    g = d.groupby(["season", "player_id"])
    z = pd.DataFrame({s: (d[s] - g[s].transform("mean")) / g[s].transform("std").replace(0, np.nan) for s in DIMS})
    if len(z) > max_rows:
        z = z.sample(max_rows, random_state=seed)
    ns = pd.DataFrame(index=z.index)
    for s in DIMS:
        col = z[s]
        k = col.notna().sum()
        ns[s] = norm.ppf((col.rank(method="average") - 0.5) / k)
    return nearest_correlation(ns.corr(min_periods=500).to_numpy())


def weekly_game_structure(conn: sqlite3.Connection, seasons: list[str]) -> tuple[dict[str, float], float]:
    """(pmf of team games per Mon-Sun week, share of missed games lost to whole-week absences)."""
    ph = ",".join("?" * len(seasons))
    games = pd.read_sql_query(
        f"SELECT game_id, date, season, home_team_id, away_team_id FROM game WHERE game_type='regular' AND season IN ({ph})",
        conn, params=seasons)
    games["date"] = pd.to_datetime(games["date"])
    games["week"] = games["date"].dt.to_period("W-SUN").dt.start_time
    tg = pd.concat([games[["season", "week", "home_team_id"]].rename(columns={"home_team_id": "team_id"}),
                    games[["season", "week", "away_team_id"]].rename(columns={"away_team_id": "team_id"})])
    team_week = tg.groupby(["season", "team_id", "week"]).size().rename("G").reset_index()
    # first/last NBA weeks are partial (season starts midweek); ESPN folds them into neighbours
    bounds = team_week.groupby("season")["week"].agg(["min", "max"])
    team_week = team_week.join(bounds, on="season")
    team_week = team_week[(team_week["week"] > team_week["min"]) & (team_week["week"] < team_week["max"])]
    team_week = team_week.drop(columns=["min", "max"])
    pmf = team_week[team_week["G"] >= 2]["G"].value_counts(normalize=True).sort_index()
    games_pmf = {str(int(k)): float(v) for k, v in pmf.items()}

    # Share of missed games that come as WHOLE-WEEK absences (injury runs, season-ending injuries) rather than
    # scattered single-game misses. Measured over every week of the season for every rotation player -- anyone
    # who plays >= 20 minutes when he plays and appeared in >= 8 games, however many games he then missed. (An earlier
    # version required >= 25 games and counted only weeks between a player's first and last game, which silently
    # dropped the long and season-ending injuries and put this share at ~0.42 instead of ~0.70; the simulation then
    # produced too few zero-game weeks, visible as a heavy lower tail in the calibration check.)
    box = pd.read_sql_query(
        f"""SELECT g.season, g.date, bs.player_id, bs.team_id, bs.min FROM box_score bs
            JOIN game g ON g.game_id = bs.game_id
            WHERE g.game_type='regular' AND bs.min > 0 AND g.season IN ({ph}) ORDER BY g.date""",
        conn, params=seasons)
    box["date"] = pd.to_datetime(box["date"])
    box["week"] = box["date"].dt.to_period("W-SUN").dt.start_time
    ps = box.groupby(["season", "player_id"]).agg(gp=("min", "size"), mpg=("min", "mean"), team_id=("team_id", "last")).reset_index()
    rot = ps[(ps["mpg"] >= 20) & (ps["gp"] >= 8)]
    played = box.merge(rot[["season", "player_id", "team_id"]], on=["season", "player_id", "team_id"])   # games with his final team
    pw = played.groupby(["season", "player_id", "week"]).size().rename("g").reset_index()
    full = rot.merge(team_week[team_week["G"] >= 2], on=["season", "team_id"])                            # every week of that team's season
    full = full.merge(pw, on=["season", "player_id", "week"], how="left")
    full["g"] = full["g"].fillna(0)
    missed = (full["G"] - full["g"]).clip(lower=0)
    whole = full.loc[full["g"] == 0, "G"].sum()
    c = float(whole / missed.sum()) if missed.sum() > 0 else 0.5
    return games_pmf, float(np.clip(c, 0.0, 0.95))


def talent_uncertainty(oos: pd.DataFrame, dispersion: dict[str, float], rho: dict[str, float]):
    """
    Sigma of next-season projection error, with the observed season's own
    game-to-game / shooting sampling noise removed (that part is simulated
    separately, week by week).
    """
    pairs = {"p2": ("att2",), "p3": ("att3",), "ft": ("attft",)}
    sig: dict[str, dict[str, float]] = {}
    res = {}
    oos = oos.copy()
    oos["bucket"] = oos["exp"].map(exp_bucket)
    for bucket, grp in oos.groupby("bucket"):
        sig[bucket] = {}
        w = grp["min"].to_numpy(float)
        for s in COUNT_DIMS:
            p, a, gp = grp[s].to_numpy(float), grp[f"a_{s}"].to_numpy(float), grp["gp"].to_numpy(float)
            samp = (p + p ** 2 / dispersion[s]) / gp
            excess = (a - p) ** 2 - samp
            ok = np.isfinite(excess) & (p > 0)
            val = (w[ok] * excess[ok]).sum() / (w[ok] * p[ok] ** 2).sum()
            sig[bucket][s] = float(np.sqrt(max(val, 0.02 ** 2)))
        for s in PCT_DIMS:
            att = grp[pairs[s][0]].to_numpy(float)
            p, a = grp[s].to_numpy(float) if s in grp else grp[f"r_{s}"].to_numpy(float), grp[f"a_{s}"].to_numpy(float)
            samp = p * (1 - p) / np.maximum(att, 1)
            excess = (a - p) ** 2 - samp
            ok = np.isfinite(excess) & (att >= 30)
            wt = att[ok]
            val = (wt * excess[ok]).sum() / wt.sum()
            sig[bucket][s] = float(np.sqrt(max(val, 0.005 ** 2)))
    # error correlation across dims (standardized by bucket-specific sigma)
    errs = pd.DataFrame(index=oos.index)
    for s in COUNT_DIMS:
        errs[s] = (oos[f"a_{s}"] - oos[s]) / oos[s].where(oos[s] > 0)
    for s in PCT_DIMS:
        errs[s] = (oos[f"a_{s}"] - oos[f"r_{s}"])
    errs = errs.replace([np.inf, -np.inf], np.nan)
    errs = errs.clip(lower=errs.quantile(0.01), upper=errs.quantile(0.99), axis=1)
    corr = nearest_correlation(errs.corr(min_periods=100).to_numpy())
    return sig, corr


def rookie_sigma(rookie_profiles_csv: str | Path) -> dict[str, dict[str, float]]:
    rp = pd.read_csv(rookie_profiles_csv, index_col=0)
    out = {}
    for tier, row in rp.iterrows():
        out[str(tier)] = {s: float(row[f"cv_{s}"]) for s in DIMS if pd.notna(row.get(f"cv_{s}"))}
    return out


def calibrate(conn: sqlite3.Connection, oos: pd.DataFrame, rookie_csv: str | Path,
              seasons: list[str] | None = None) -> Calibration:
    if seasons is None:
        all_seasons = [r[0] for r in conn.execute(
            "SELECT DISTINCT g.season FROM game g JOIN box_score b ON b.game_id = g.game_id ORDER BY g.season")]
        seasons = all_seasons[-4:]
    logs = load_game_logs(conn, seasons)
    disp = estimate_dispersion(logs)
    rho = estimate_rho(logs)
    corr = estimate_copula_corr(logs)
    pmf, c = weekly_game_structure(conn, seasons)
    sig, tcorr = talent_uncertainty(oos, disp, rho)
    return Calibration(
        dispersion=disp, rho=rho, corr=corr.tolist(), games_pmf=pmf, absence_c=c,
        talent_sigma=sig, talent_corr=tcorr.tolist(), rookie_sigma=rookie_sigma(rookie_csv),
        notes={"seasons": seasons, "n_game_rows": int(len(logs))},
    )
