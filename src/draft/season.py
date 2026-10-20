"""
In-season tools on top of the draft engine: the projected league table, free-agent swaps and trades.

Everything is scored the way the draft engine scores a roster. Each team plays the library's simulated weeks with daily lineups
(`DraftEngine.team_totals`); a matchup is won by taking more than 4.5 of the 9 categories in the SAME simulated week, so a team's win
probability is its share of simulated weeks won against the other nine teams. Because every roster is scored on the same simulated
weeks, the difference between two versions of a roster (before and after a swap or trade) is far less noisy than either number.

The library's simulated week `s` plays matchup period `s % n_periods` (18 regular-season weeks, then the 3 playoff weeks), so regular-season
and playoff-week win probabilities are separable.
"""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.draft.engine import DraftEngine, category_matrix
from src.draft.values import CATEGORIES, LOWER_IS_BETTER

LOWER = np.array([c in LOWER_IS_BETTER for c in CATEGORIES])                 # categories where the smaller number wins (TO)
RATIO = np.array([c in ("fg_pct", "ft_pct") for c in CATEGORIES])            # categories that are shares, not weekly counts
NOISE = 0.5                                                                  # win-probability points: a smaller difference is simulation noise


def team_cats(engine: DraftEngine, roster: list[int], cache: bool = True) -> np.ndarray:
    """[S, 9] weekly category lines of a roster (engine player indices) over the simulated weeks."""
    return category_matrix(engine._cached_totals(roster) if cache else engine.team_totals(roster))


def score_against(mine: np.ndarray, others: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """`mine` [S, 9] against `others` [U, S, 9] in the same simulated weeks: (matchup win chance [S], category win chance [S, 9])."""
    a = mine[None]
    cat = np.where(LOWER, a < others, a > others) + 0.5 * (a == others)       # [U, S, 9]
    wins = cat.sum(-1)
    match = (wins > 4.5) + 0.5 * (wins == 4.5)
    return match.mean(0), cat.mean(0)


def period_weeks(conn: sqlite3.Connection, season: str, n_regular: int, n_playoff: int) -> np.ndarray | None:
    """Length of each matchup period in weeks (the opening and All-Star periods are two weeks long); None if the schedule is not loaded."""
    from src.simulation.schedule import season_periods

    try:
        sched = season_periods(conn, season, n_regular=n_regular, n_playoff=n_playoff)
    except Exception:
        return None
    return np.array([max(1, round(((e - s).days + 1) / 7)) for s, e in sched.spans], dtype=float)


# ---------------------------------------------------------------------------
# the league table
# ---------------------------------------------------------------------------
@dataclass
class LeagueTable:
    win: np.ndarray        # [T] chance to win a weekly matchup against the other teams (all matchup periods)
    win_reg: np.ndarray    # [T] regular-season weeks only
    win_po: np.ndarray     # [T] playoff weeks only
    cat_win: np.ndarray    # [T, 9] chance to win each category in a weekly matchup
    totals: np.ndarray     # [T, 9] projected weekly total per 7 days (FG%/FT% are shares)
    cat_rank: np.ndarray   # [T, 9] rank among the teams in each category, 1 = best
    h2h: np.ndarray        # [T, T] row team's chance to beat the column team (NaN on the diagonal)
    by_week: np.ndarray    # [T, n_periods] chance to win in each matchup period
    cats: np.ndarray       # [T, S, 9] weekly category lines, kept so a roster can be scored against the field

    @property
    def power_rank(self) -> np.ndarray:
        """[T] 1 = highest win probability."""
        return (-self.win).argsort().argsort() + 1


def league_table(engine: DraftEngine, rosters: dict[int, list[int]], n_periods: int, n_regular: int,
                 weeks: np.ndarray | None = None) -> LeagueTable:
    """Score every team (`rosters`: slot 1..T -> engine player indices) against all the others on the simulated weeks."""
    T = len(rosters)
    cats = np.stack([team_cats(engine, rosters[t]) for t in range(1, T + 1)])                  # [T, S, 9]
    S = cats.shape[1]
    per = np.arange(S) % n_periods
    a, b = cats[:, None], cats[None, :]
    cat = np.where(LOWER, a < b, a > b) + 0.5 * (a == b)                                       # [T, T, S, 9] team t vs team u
    wins = cat.sum(-1)
    match = (wins > 4.5) + 0.5 * (wins == 4.5)                                                 # [T, T, S]
    off = ~np.eye(T, dtype=bool)
    vs = np.stack([match[t][off[t]] for t in range(T)])                                        # [T, T-1, S] against the other teams
    cat_vs = np.stack([cat[t][off[t]] for t in range(T)])                                      # [T, T-1, S, 9]
    reg = per < n_regular
    h2h = match.mean(-1)
    h2h[~off] = np.nan
    scale = np.ones(S) if weeks is None else np.asarray(weeks, float)[per]                     # two-week periods count double: show totals per 7 days
    totals = (cats / np.where(RATIO, 1.0, scale[:, None])).mean(1)
    sign = np.where(LOWER, 1.0, -1.0)
    return LeagueTable(
        win=vs.mean((1, 2)), win_reg=vs[:, :, reg].mean((1, 2)), win_po=vs[:, :, ~reg].mean((1, 2)),
        cat_win=cat_vs.mean((1, 2)), totals=totals, cat_rank=(totals * sign).argsort(0).argsort(0) + 1, h2h=h2h,
        by_week=np.stack([vs[:, :, per == w].mean((1, 2)) for w in range(n_periods)], axis=1), cats=cats)


def lineup_usage(engine: DraftEngine, roster: list[int]) -> dict[int, float]:
    """Share of the days a player has a game on which he fills a lineup slot (the rest of those days he sits on the bench)."""
    usage: dict = {}
    engine.team_totals(roster, usage)
    return {i: (got / played if played else 0.0) for i, (got, played) in usage.items()}


# ---------------------------------------------------------------------------
# free-agent swaps
# ---------------------------------------------------------------------------
def n_centers(engine: DraftEngine, roster: list[int]) -> int:
    return int(sum(engine.is_center[i] for i in roster))


@dataclass
class SwapGrid:
    adds: list[int]                  # free agents tried
    drops: list                      # my players tried (None = add without dropping anyone)
    win: np.ndarray                  # [A, D] my win probability after adding adds[a] and dropping drops[d]; NaN if that breaks the center cap
    cat: np.ndarray                  # [A, D, 9] my category win probabilities after the swap
    base_win: float
    base_cat: np.ndarray             # [9]

    def gain(self) -> np.ndarray:
        """[A, D] change in win probability, in points."""
        return (self.win - self.base_win) * 100

    def best(self) -> pd.DataFrame:
        """Per free agent: his best drop, the gain in win probability (points) and the change in each category win probability (points)."""
        g = np.where(np.isnan(self.win), -np.inf, self.gain())
        d = g.argmax(1)
        rows = np.arange(len(self.adds))
        ok = np.isfinite(g[rows, d])
        cat = (self.cat[rows, d] - self.base_cat) * 100
        out = pd.DataFrame({"add": self.adds, "drop": pd.Series([self.drops[j] for j in d], dtype=object),      # object: a None (no drop) must stay None
                            "gain": np.where(ok, g[rows, d], np.nan), "win": np.where(ok, self.win[rows, d] * 100, np.nan)})
        for k, c in enumerate(CATEGORIES):
            out[f"d_{c}"] = np.where(ok, cat[:, k], np.nan)
        return out


def swap_grid(engine: DraftEngine, roster: list[int], opp_cats: np.ndarray, adds: list[int], drops: list | None = None,
              max_centers: int | None = None, progress=None) -> SwapGrid:
    """
    Add each of `adds` and drop each of `drops` (default: each player on `roster`; None means "just add") and score my team against
    `opp_cats` [U, S, 9], the other teams' weekly lines. A swap is skipped when it would leave more centers than the cap allows
    (unless it does not increase the count).
    """
    drops = list(roster) if drops is None else list(drops)
    base_m, base_c = score_against(team_cats(engine, roster), opp_cats)
    before = n_centers(engine, roster)
    win = np.full((len(adds), len(drops)), np.nan)
    cat = np.full((len(adds), len(drops), len(CATEGORIES)), np.nan)
    for ai, f in enumerate(adds):
        for di, d in enumerate(drops):
            new = [x for x in roster if x != d] + [f]
            after = n_centers(engine, new)
            if max_centers is not None and after > max(max_centers, before):
                continue
            m, c = score_against(team_cats(engine, new, cache=False), opp_cats)
            win[ai, di], cat[ai, di] = m.mean(), c.mean(0)
        if progress is not None:
            progress((ai + 1) / len(adds))
    return SwapGrid(list(adds), drops, win, cat, float(base_m.mean()), base_c.mean(0))


def removal_cost(engine: DraftEngine, roster: list[int], opp_cats: np.ndarray) -> dict[int, float]:
    """Win-probability points a team loses if it plays without each player (no replacement): how much each one is adding."""
    base = score_against(team_cats(engine, roster), opp_cats)[0].mean()
    return {i: float((base - score_against(team_cats(engine, [x for x in roster if x != i], cache=False), opp_cats)[0].mean()) * 100)
            for i in roster}


def field_cats(table: LeagueTable, slot: int) -> np.ndarray:
    """[T-1, S, 9] the weekly lines of every team except `slot` (1-based)."""
    return np.delete(table.cats, slot - 1, axis=0)


# ---------------------------------------------------------------------------
# trades
# ---------------------------------------------------------------------------
def apply_trade(rosters: dict[int, list[int]], a: int, b: int, give_a: list[int], give_b: list[int],
                drop_a: list[int] = (), drop_b: list[int] = ()) -> dict[int, list[int]]:
    """Rosters after team `a` sends `give_a` to team `b` for `give_b`; `drop_a`/`drop_b` are released to make room."""
    new = {t: list(v) for t, v in rosters.items()}
    new[a] = [x for x in rosters[a] if x not in give_a and x not in drop_a] + list(give_b)
    new[b] = [x for x in rosters[b] if x not in give_b and x not in drop_b] + list(give_a)
    return new


def verdict(d_me: float, d_them: float) -> tuple[str, str]:
    """(level, message): how a trade reads from the change in each side's win probability, in points. level: good | ok | warn | bad."""
    if d_me >= NOISE and d_them >= NOISE:
        return "good", "Win-win: both teams improve, so the other side has a reason to say yes."
    if d_me >= NOISE and d_them >= -0.7:
        return "ok", "Fair: you gain and the other team gives up very little."
    if d_me >= NOISE:
        return "warn", f"Lopsided in your favour: the other team loses {-d_them:.1f} points of win probability, so they are unlikely to accept."
    if d_me > -NOISE and d_them >= 0.7:
        return "warn", f"Lopsided against you: it helps the other team by {d_them:.1f} points and does nothing for you."
    if d_me > -NOISE:
        return "ok", "Roughly neutral: neither team's win probability moves by more than the simulation noise."
    if d_them >= NOISE:
        return "bad", "Bad for you: the other team gains and you lose."
    return "bad", "Bad for you: it lowers your win probability and does not help the other team either."
