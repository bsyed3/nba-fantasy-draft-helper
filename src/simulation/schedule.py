"""
Matchup periods and the schedule inside them, for day-level simulation.

An ESPN head-to-head matchup period is a Mon-Sun week, with two exceptions that this module reproduces from
the game table (so it works for the published future schedule and for past seasons alike):

  * the **NBA Cup knockout week**, which is thin (many teams play only a game or two while the bracket is
    decided), is merged with the week BEFORE it;
  * the **All-Star break** week is merged with the week AFTER it.

For 2026-27 this gives: week 1 = Oct 19-25, week 2 = Oct 26-Nov 1, ..., **week 7 = Nov 30-Dec 13** (Cup),
..., **week 17 = Feb 15-28** (All-Star), week 18 = Mar 1-7 (last regular-season week), then the three
playoff weeks Mar 8-28: 18 regular-season weeks + 3 playoff weeks.

Future schedules are incomplete: the Cup's filler and knockout games are not scheduled until the bracket is
known, so every team shows 80 of its 82 games. By default these unscheduled games are IGNORED -- the Cup week
simply uses the published games and the week's real uncertainty is left to the simulation's randomness. Passing
`fill_unscheduled=True` instead gives each team its missing games on random days of the Cup week (weekday pattern
learned from past Cup weeks), re-drawn for every simulated matchup (`Schedule.team_day_masks`).

The same periods of REAL per-player results are provided for validation.
"""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.simulation.library import STAT_ORDER, Library

SEASON_GAMES = 82


@dataclass
class Schedule:
    masks: np.ndarray                  # [W, T, D] bool: team plays on day `offset` of matchup period w (published games)
    team_ids: list[int]                # column order of `masks`
    weeks: list                        # period start dates (Mondays)
    spans: list = field(default_factory=list)       # [(start, end)] per period, inclusive
    missing: dict = field(default_factory=dict)     # team_id -> games not yet scheduled (future seasons only)
    flex_period: int | None = None                  # period (index) in which the missing games will be played (the Cup week)
    flex_weights: list = field(default_factory=lambda: [0, 1, 1, 1, 1, 1, 0])   # weekday weights (Mon..Sun) for those games

    @property
    def n_weeks(self) -> int:
        return self.masks.shape[0]

    @property
    def n_days(self) -> int:
        return self.masks.shape[2]

    def team_index(self) -> dict[int, int]:
        return {t: i for i, t in enumerate(self.team_ids)}

    def expected_games(self) -> np.ndarray:
        """[W, T] expected games per team per period, including the not-yet-scheduled Cup games."""
        g = self.masks.sum(axis=2).astype(float)
        if self.flex_period is not None:
            g[self.flex_period] += np.array([self.missing.get(t, 0) for t in self.team_ids], dtype=float)
        return g

    def team_day_masks(self, n_sims: int, rng: np.random.Generator) -> np.ndarray:
        """
        [S, T, D] game-day masks, one per simulated matchup (sim s plays period s % W). Identical to the
        published schedule except in the flex (Cup) period, where each team's missing games are placed on
        random free days of the period's final week, weighted by the weekday pattern of past Cup weeks.
        Teammates share the draw because it is made per team, not per player.
        """
        W, T, D = self.masks.shape
        wk = np.arange(n_sims) % W
        out = self.masks[wk].copy()
        if self.flex_period is None or not self.missing:
            return out
        sims = np.flatnonzero(wk == self.flex_period)
        if len(sims) == 0:
            return out
        miss = np.array([self.missing.get(t, 0) for t in self.team_ids])
        w = np.zeros(D)
        w[D - 7:] = np.asarray(self.flex_weights, dtype=float)
        u = rng.random((len(sims), T, D))
        keys = np.where(w > 0, u ** (1.0 / np.maximum(w, 1e-9)), -1.0)      # weighted sampling without replacement
        keys = np.where(out[sims], -1.0, keys)                               # a team can't play twice in a day
        rank = (-keys).argsort(axis=-1).argsort(axis=-1)                     # 0 = largest key
        out[sims] |= (rank < miss[None, :, None]) & (keys > 0)
        return out


def _week_groups(g: pd.DataFrame) -> tuple[list[list[pd.Timestamp]], int | None]:
    """
    Group the season's Mon-Sun weeks into matchup periods. Returns (groups, index of the Cup group or None).
    """
    per_week = g.groupby("week").agg(n=("date", "size")).sort_index()
    weeks = list(per_week.index)
    n = len(weeks)
    med = per_week["n"].median()
    merge_next: set[int] = set()
    cup_week: int | None = None
    # NBA Cup knockout week: the thinnest week in the early/mid-season window folds into the week before it
    lo, hi = max(1, int(0.08 * n)), int(0.40 * n)
    early = per_week["n"].iloc[lo:hi]
    if len(early) and early.min() < 0.6 * med:
        cup_week = lo + int(np.argmin(early.to_numpy()))
        merge_next.add(cup_week - 1)
    # All-Star break: the thinnest week in the back half folds into the week after it
    lo2, hi2 = int(0.35 * n), int(0.85 * n)
    late = per_week["n"].iloc[lo2:hi2]
    if len(late) and late.min() < 0.8 * med:
        merge_next.add(lo2 + int(np.argmin(late.to_numpy())))
    groups: list[list[pd.Timestamp]] = []
    group_of: dict[int, int] = {}
    i = 0
    while i < n:
        members = [i]
        while i in merge_next and i + 1 < n:
            i += 1
            members.append(i)
        for j in members:
            group_of[j] = len(groups)
        groups.append([weeks[j] for j in members])
        i += 1
    return groups, (group_of.get(cup_week) if cup_week is not None else None)


def _flex_day_weights(conn: sqlite3.Connection, season: str) -> list[float]:
    """Weekday pattern (Mon..Sun) of games in the Cup week of earlier, completed seasons; Tue-Sat uniform if none."""
    acc = np.zeros(7)
    for (other,) in conn.execute(
            "SELECT DISTINCT g.season FROM game g JOIN box_score b ON b.game_id = g.game_id WHERE g.season != ?", (season,)).fetchall():
        if int(other[:4]) < 2023:        # no NBA Cup before 2023-24
            continue
        g = pd.read_sql_query("SELECT date, home_team_id h, away_team_id a FROM game WHERE season = ? AND game_type = 'regular'",
                              conn, params=[other])
        g["date"] = pd.to_datetime(g["date"])
        g["week"] = g["date"].dt.to_period("W-SUN").dt.start_time
        groups, cup = _week_groups(g)
        if cup is None:
            continue
        days = g.loc[g["week"] == groups[cup][-1], "date"].dt.dayofweek.value_counts()
        for d, c in days.items():
            acc[int(d)] += c
    if acc.sum() == 0:
        return [0, 1, 1, 1, 1, 1, 0]
    return (acc / acc.sum()).tolist()


def season_periods(conn: sqlite3.Connection, season: str, n_regular: int = 18, n_playoff: int = 0,
                   fill_unscheduled: bool = False) -> Schedule:
    """
    The first `n_regular` (+ `n_playoff`) matchup periods of `season`, as a Schedule.
    Default is the 18 regular-season weeks your league plays; pass n_playoff=3 to include the playoff weeks.
    fill_unscheduled=True (future seasons only) adds each team's not-yet-scheduled games to the Cup week at random.
    """
    g = pd.read_sql_query(
        "SELECT date, home_team_id, away_team_id FROM game WHERE season = ? AND game_type = 'regular'", conn, params=[season])
    if g.empty:
        raise ValueError(f"no games for season {season}; run scripts/ingest.py (it loads the published schedule)")
    g["date"] = pd.to_datetime(g["date"])
    g["week"] = g["date"].dt.to_period("W-SUN").dt.start_time
    g["dow"] = g["date"].dt.dayofweek
    all_groups, cup_group = _week_groups(g)
    groups = all_groups[: n_regular + n_playoff]
    spans = [(grp[0], grp[-1] + pd.Timedelta(days=6)) for grp in groups]
    D = max((e - s).days + 1 for s, e in spans)
    teams = sorted(set(g["home_team_id"]) | set(g["away_team_id"]))
    tidx = {t: i for i, t in enumerate(teams)}
    masks = np.zeros((len(spans), len(teams), D), dtype=bool)
    for p, (s, e) in enumerate(spans):
        sub = g[(g["date"] >= s) & (g["date"] <= e)]
        off = (sub["date"] - s).dt.days.to_numpy()
        for col in ("home_team_id", "away_team_id"):
            masks[p, sub[col].map(tidx).to_numpy(), off] = True

    # A season that hasn't been played yet has an incomplete published schedule (Cup filler/knockout games unscheduled).
    # Seasons already played are left exactly as they happened -- their light Cup weeks are real history.
    played_already = conn.execute(
        "SELECT 1 FROM box_score b JOIN game g ON g.game_id = b.game_id WHERE g.season = ? LIMIT 1", (season,)).fetchone() is not None
    sched = Schedule(masks, [int(t) for t in teams], [s for s, _ in spans], spans)
    if not played_already:
        per_team = pd.concat([g["home_team_id"], g["away_team_id"]]).value_counts()
        missing = {int(t): int(max(0, SEASON_GAMES - per_team.get(t, 0))) for t in teams}
        if any(missing.values()):
            sched.missing = missing                                   # recorded either way, for reporting
            if fill_unscheduled:
                sched.flex_period = cup_group if (cup_group is not None and cup_group < len(spans)) else int(masks.sum(axis=(1, 2)).argmin())
                sched.flex_weights = _flex_day_weights(conn, season)
    return sched


season_weeks = season_periods  # backwards-compatible name


def actual_library(conn: sqlite3.Connection, season: str, player_ids: np.ndarray, schedule: Schedule) -> Library:
    """What really happened, day by day, in the same shape as a simulated Library (one 'sim' per real matchup period)."""
    box = pd.read_sql_query(
        """
        SELECT g.date, bs.player_id, bs.fgm, bs.fga, bs.ftm, bs.fta, bs.tpm,
               bs.oreb + bs.dreb AS reb, bs.ast, bs.stl, bs.blk, bs.tov
        FROM box_score bs JOIN game g ON g.game_id = bs.game_id
        WHERE g.game_type = 'regular' AND g.season = ? AND bs.min > 0
        """, conn, params=[season])
    box["date"] = pd.to_datetime(box["date"])
    pidx = {int(p): i for i, p in enumerate(player_ids)}
    box = box[box["player_id"].isin(pidx)]
    P, W, D = len(player_ids), schedule.n_weeks, schedule.n_days
    stats = np.zeros((P, W, D, len(STAT_ORDER)), dtype=np.uint8)
    played = np.zeros((P, W, D), dtype=bool)
    for w, (s, e) in enumerate(schedule.spans):
        sub = box[(box["date"] >= s) & (box["date"] <= e)]
        if sub.empty:
            continue
        pi = sub["player_id"].map(pidx).to_numpy()
        di = (sub["date"] - s).dt.days.to_numpy()
        stats[pi, w, di] = np.clip(sub[STAT_ORDER].to_numpy(), 0, 255).astype(np.uint8)
        played[pi, w, di] = True
    return Library(np.asarray(player_ids), stats, played)
