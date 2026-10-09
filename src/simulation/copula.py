"""
Phase 2 -- Gaussian-copula simulation of fantasy weeks, game by game.

Every simulated matchup is one of the real matchup periods of the (published)
NBA schedule -- a Mon-Sun week, or a 14-day period for the opening and All-Star
stretches (see schedule.py) -- so each player's team plays on its actual days.
For every player and sim:

  1. Talent draw: the projection's per-game means are perturbed by a
     correlated shock (lognormal for counts, additive for shooting %),
     sized from the Phase-1 backtest residuals (draft-tier CVs for
     rookies). This is "how wrong could the projection be".
  2. Availability: with probability c*(1-avail) the player misses the whole
     week (an injury run); otherwise he plays each scheduled game with the
     probability that makes his expected games equal avail * team games.
  3. Stat line, for each game he plays: correlated standard normals
     Z ~ N(0, R) -> uniforms U = Phi(Z) -> marginal inverse CDFs. Counts
     (2PA, 3PA, FTA, REB, AST, STL, BLK, TOV) are Negative Binomial; makes
     are Binomial(attempts, q), the make-rate quantile driven by its own
     copula dimension (Beta-jittered when the data shows shooting
     over-dispersion).

Points, FG% and FT% are never simulated directly: they are computed from
the simulated makes/attempts (PTS = 2*FGM + 3PM + FTM), exactly as in the
box score. Output is a day-resolution Library (see library.py) so the
draft engine can set daily lineups against position slots.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import beta, binom, nbinom, norm

from src.models.projection import draft_tier
from src.simulation.calibration import COUNT_DIMS, DIMS, PCT_DIMS, Calibration, exp_bucket
from src.simulation.library import STAT_IDX, STAT_ORDER, Library  # noqa: F401  (re-exported)
from src.simulation.schedule import Schedule

MAX_SIGMA = 0.8


def _sigma_matrix(proj: pd.DataFrame, cal: Calibration) -> np.ndarray:
    """[P, 11] talent sigma per player per dim (relative for counts, absolute for pct)."""
    out = np.zeros((len(proj), len(DIMS)))
    for i, (exp, pick) in enumerate(zip(proj["exp"].to_numpy(), proj["draft_pick"].to_numpy())):
        base = cal.talent_sigma[exp_bucket(exp)]
        row = cal.rookie_sigma.get(draft_tier(pick), {}) if exp == 0 else {}
        for j, d in enumerate(DIMS):
            out[i, j] = row.get(d, base[d])
    out[:, : len(COUNT_DIMS)] = np.clip(out[:, : len(COUNT_DIMS)], 0.0, MAX_SIGMA)
    return out


def build_library(proj: pd.DataFrame, cal: Calibration, schedule: Schedule, sims_per_week: int = 40,
                  seed: int = 0, chunk: int = 40) -> Library:
    """
    proj: one row per player with per-game means `fga2 fga3 fta reb ast stl blk tov`,
    shooting rates `r_p2 r_p3 r_ft`, `avail`, `exp`, `draft_pick`, `team_id` (current NBA team), `player_id`.
    Sim s plays matchup period (s % n_periods).
    """
    rng = np.random.default_rng(seed)
    P, W = len(proj), schedule.n_weeks
    S = W * sims_per_week
    L_copula = np.linalg.cholesky(np.array(cal.corr))
    L_talent = np.linalg.cholesky(np.array(cal.talent_corr))
    sigma = _sigma_matrix(proj, cal)
    mu = np.column_stack([proj[d].to_numpy(float) for d in COUNT_DIMS])                    # [P, 8]
    q0 = np.column_stack([proj["r_p2"], proj["r_p3"], proj["r_ft"]]).astype(float)         # [P, 3]
    avail = np.clip(proj["avail"].to_numpy(float), 0.03, 0.995)
    r_nb = np.array([cal.dispersion[d] for d in COUNT_DIMS])
    rho = np.array([cal.rho[d] for d in PCT_DIMS])
    c = cal.absence_c
    tmap = schedule.team_index()
    team_col = np.array([tmap.get(int(t), -1) if pd.notna(t) else -1 for t in proj["team_id"]])
    # one game-day mask per (simulated matchup, team): the published schedule, plus -- in the NBA Cup week of a future
    # season -- each team's not-yet-scheduled games on random days, shared by every player on that team
    team_masks = schedule.team_day_masks(S, rng).transpose(1, 0, 2)          # [T, S, D]

    D = schedule.n_days
    stats = np.zeros((P, S, D, len(STAT_ORDER)), dtype=np.uint8)
    played = np.zeros((P, S, D), dtype=bool)

    for lo in range(0, P, chunk):
        hi = min(lo + chunk, P)
        n_c = hi - lo
        # schedule: which weekdays this player's team plays in sim s's week (free agents: never)
        tc = team_col[lo:hi]
        sched = team_masks[np.maximum(tc, 0)] & (tc >= 0)[:, None, None]                             # [n, S, D]


        # availability
        a = avail[lo:hi, None]
        p_out = c * (1 - a)
        p_play = np.clip(a / (1 - p_out), 0, 1)
        out = rng.random((n_c, S)) < p_out
        pl = sched & ~out[..., None] & (rng.random((n_c, S, D)) < p_play[..., None])
        played[lo:hi] = pl
        pi, si, di = np.nonzero(pl)
        N = len(pi)
        if N == 0:
            continue

        # talent draw per (player, sim), shared by that week's games
        eps = (rng.standard_normal((n_c, S, len(DIMS))) @ L_talent.T)[pi, si]                 # [N, 11]
        sg = sigma[lo:hi][pi]
        mu_g = mu[lo:hi][pi] * np.exp(sg[:, :8] * eps[:, :8] - 0.5 * sg[:, :8] ** 2)          # [N, 8]
        q = np.clip(q0[lo:hi][pi] + sg[:, 8:] * eps[:, 8:], 0.02, 0.98)                       # [N, 3]

        # per-game correlated stat line
        u = np.clip(norm.cdf(rng.standard_normal((N, len(DIMS))) @ L_copula.T), 1e-6, 1 - 1e-6)
        counts = nbinom.ppf(u[:, :8], r_nb[None, :], r_nb[None, :] / (r_nb[None, :] + mu_g))   # [N, 8]
        a2, a3, af, reb, ast, stl, blk, tov = [counts[:, i] for i in range(8)]
        makes = []
        for j, att in enumerate((a2, a3, af)):
            qj = q[:, j]
            if rho[j] > 1e-4:  # shooting streakiness beyond binomial noise
                kappa = 1.0 / rho[j] - 1.0
                qj = beta.ppf(rng.random(N), qj * kappa, (1 - qj) * kappa)
            makes.append(binom.ppf(u[:, 8 + j], att, np.clip(qj, 0.01, 0.99)))
        m2, m3, mf = makes
        line = np.stack([m2 + m3, a2 + a3, mf, af, m3, reb, ast, stl, blk, tov], axis=-1)
        stats[lo + pi, si, di] = np.clip(line, 0, 255).astype(np.uint8)

    return Library(proj["player_id"].to_numpy(), stats, played)


def weekly_means(lib: Library) -> pd.DataFrame:
    """Mean weekly totals per player (diagnostics / tests)."""
    m = lib.stats.sum(axis=2).mean(axis=1)
    df = pd.DataFrame(m, columns=STAT_ORDER)
    df.insert(0, "player_id", lib.player_ids)
    df["games"] = lib.played.sum(axis=2).mean(axis=1)
    return df
