"""
Category-aware ("adaptive punt") drafting policy -- an optional Phase 3 strategy.

The board orders players by a fixed sum of category z-scores, and in the engine's roll-outs my own later
picks follow that board. A weekly H2H matchup is won by taking MORE THAN 4.5 of the 9 categories, so the value
of a player depends on the team around him: once a roster is strong in AST and 3PM, a third assist guard adds
little, while a big who locks up REB/BLK/FG% adds a lot -- and a category that is already lost is worth nothing.

This policy picks, among the best few players left on the board, the one that maximises a cheap proxy of
P(win >= 5 of 9 categories) for the roster it leads to:

  1. for each candidate, finish the draft with the bots' (noise-free) order and my later picks in board order,
  2. total every team's simulated weeks from the library WITHOUT the daily-lineup rules: each player's weekly
     production is weighted by his roster rank (a starter plays most of his games, bench players cover the
     rest -- weights calibrated against the real daily lineup by `calibrate_rank_weights`),
  3. score my weekly category totals against every opponent's, week by week, and average.

It deliberately avoids the full daily-lineup evaluation (~20x slower) so it can be run at every pick. A
candidate only overrides the board's top choice when it is ahead by `margin` (win-probability units), which
guards against chasing noise.
"""
from __future__ import annotations

import copy

import numpy as np
import pandas as pd

from src.draft.engine import DraftEngine, category_matrix
from src.draft.state import DraftState
from src.draft.values import CATEGORIES

# Share of a roster slot's simulated weekly production that the real daily lineup actually uses, by rank within
# the roster (rank 1 = highest board value). Measured by `calibrate_rank_weights` on board-drafted mock rosters
# (2026-27 library): 1.00 for the ten best, 0.99 / 0.97 / 0.95 for the three bench players -- a 13-man roster has
# about six players with a game on a given day, so almost every game finds an open slot.
DEFAULT_RANK_WEIGHT_ACTIVE = 1.0
DEFAULT_RANK_WEIGHT_BENCH = 0.97
DEFAULT_MARGIN = 0.01          # tuned on development seeds, confirmed on fresh ones (scripts/analysis/adaptive_policy_test.py)
DEFAULT_CANDIDATES = 6         # 6 beat 12 and 24 on both development and fresh seeds (more candidates = more noise to chase)


class AdaptivePolicy:
    def __init__(self, engine: DraftEngine, n_candidates: int = DEFAULT_CANDIDATES, margin: float = DEFAULT_MARGIN,
                 rank_weights: np.ndarray | None = None, sim_stride: int = 1):
        self.engine = engine
        self.n_candidates, self.margin = n_candidates, margin
        lib = engine.lib
        sims = np.arange(0, lib.played.shape[1], max(sim_stride, 1))
        P = lib.played.shape[0]
        # W[p, s, k]: player p's whole-week total of stat k in simulated week s (no lineup rules)
        self.W = np.empty((P, len(sims), lib.stats.shape[-1]), dtype=np.float32)
        for p in range(P):
            self.W[p] = np.einsum("sd,sdk->sk", lib.played[p][sims].astype(np.float32), lib.stats[p][sims].astype(np.float32))
        st = engine.settings
        if rank_weights is None:
            rank_weights = np.where(np.arange(st.roster_size) < st.active_slots, DEFAULT_RANK_WEIGHT_ACTIVE, DEFAULT_RANK_WEIGHT_BENCH)
        self.rank_weights = np.asarray(rank_weights, dtype=np.float32)
        self.lower = np.array([c == "tov" for c in CATEGORIES])
        self._plain: np.ndarray | None = None

    def for_engine(self, engine: DraftEngine) -> "AdaptivePolicy":
        """The same policy (and its weekly-totals table) bound to another engine, e.g. one with a different draft slot."""
        other = copy.copy(self)
        other.engine, other._plain = engine, None
        return other

    # ------------------------------------------------------------------
    def team_cats(self, roster: list[int]) -> np.ndarray:
        """[S, 9] weekly category values of a roster under the rank-weighted proxy."""
        order = sorted(roster, key=lambda j: -self.engine.value[j])
        w = self.rank_weights[: len(order)]
        return category_matrix(np.tensordot(w, self.W[order][: len(w)], axes=(0, 0)))

    def win_prob(self, mine: np.ndarray, opp: np.ndarray) -> float:
        """P(my team takes more than 4.5 of the 9 categories) vs each opponent, per simulated week (ties split)."""
        a, b = mine[None], opp
        cat = np.where(self.lower, a < b, a > b) + 0.5 * (a == b)          # [O, S, 9]
        wins = cat.sum(-1)
        return float(((wins > 4.5) + 0.5 * (wins == 4.5)).mean())

    def plain_orders(self) -> np.ndarray:
        """Noise-free preference orders: bots follow their base order, my later picks the board."""
        if self._plain is None:
            e = self.engine
            T = e.settings.n_teams
            bots = np.argsort(e.bot_base, kind="stable").astype(np.int32)
            mine = np.argsort(e.rank, kind="stable").astype(np.int32)
            self._plain = np.stack([mine if t + 1 == e.settings.my_slot else bots for t in range(T)])
        return self._plain

    def candidates(self, taken: np.ndarray, cc_me: int) -> list[int]:
        e, cap = self.engine, self.engine.settings.max_centers
        out: list[int] = []
        for j in e.static_order:
            if taken[j] or (e.is_center[j] and cc_me >= cap):
                continue
            out.append(int(j))
            if len(out) == self.n_candidates:
                break
        return out

    def score_candidates(self, taken, rosters, ptr, orders, cc, pick_no: int) -> list[tuple[int, float]]:
        """[(player index, proxy win probability)] for the board's top `n_candidates` available players."""
        e, me = self.engine, self.engine.settings.my_slot
        res = []
        for c in self.candidates(taken, cc[me - 1]):
            tk, ros = taken.copy(), {t: list(v) for t, v in rosters.items()}
            pt, cc2 = list(ptr), list(cc)
            tk[c] = True
            ros[me].append(c)
            cc2[me - 1] += int(e.is_center[c])
            e._advance(tk, ros, pt, orders, cc2, pick_no + 1, e.settings.total_picks, plain=True)
            opp = np.stack([self.team_cats(r) for t, r in ros.items() if t != me])
            res.append((c, self.win_prob(self.team_cats(ros[me]), opp)))
        return res

    def pick(self, taken, rosters, ptr, orders, cc, pick_no: int) -> int:
        """Library index of the player I take at overall pick `pick_no`."""
        scored = self.score_candidates(taken, rosters, ptr, orders, cc, pick_no)
        board_choice, board_p = scored[0]
        best, best_p = max(scored, key=lambda x: x[1])
        return best if best_p > board_p + self.margin else board_choice

    def plan(self, state: DraftState) -> tuple[int | None, pd.DataFrame]:
        """
        The policy's pick for MY next pick given the live draft state, and the candidates it weighed. If it is not my turn
        yet, the bots' picks up to mine are played out first (in their noise-free order). Returns (player_id or None,
        table with player_id / board_rank / proxy_win_prob / chosen, sorted by proxy win probability).
        """
        e = self.engine
        my_pick = state.my_next_pick_no()
        if my_pick is None or state.done:
            return None, pd.DataFrame(columns=["player_id", "board_rank", "proxy_win_prob", "chosen"])
        taken, rosters = e._initial(state)
        cc, ptr, orders = e._center_counts(rosters), [0] * e.settings.n_teams, self.plain_orders()
        e._advance(taken, rosters, ptr, orders, cc, state.next_pick_no, my_pick - 1, plain=True)
        scored = self.score_candidates(taken, rosters, ptr, orders, cc, my_pick)
        board_choice, board_p = scored[0]
        best, best_p = max(scored, key=lambda x: x[1])
        choice = best if best_p > board_p + self.margin else board_choice
        table = pd.DataFrame({"player_id": [int(e.pid[c]) for c, _ in scored], "board_rank": [int(e.rank[c]) + 1 for c, _ in scored],
                              "proxy_win_prob": [p for _, p in scored], "chosen": [c == choice for c, _ in scored]})
        return int(e.pid[choice]), table.sort_values("proxy_win_prob", ascending=False).reset_index(drop=True)


def calibrate_rank_weights(engine: DraftEngine, rosters: list[list[int]]) -> np.ndarray:
    """
    Fraction of each roster rank's played days that the real daily lineup puts in a slot, averaged over `rosters`
    (library indices). Rank 1 is the highest board value on the roster.
    """
    used = np.zeros(engine.settings.roster_size)
    played = np.zeros(engine.settings.roster_size)
    for r in rosters:
        usage: dict[int, tuple[float, float]] = {}
        engine.team_totals(r, usage=usage)
        for k, i in enumerate(sorted(r, key=lambda j: -engine.value[j])):
            u, p = usage.get(i, (0.0, 0.0))
            used[k] += u
            played[k] += p
    return used / np.maximum(played, 1)


def adaptive_pick(policy: AdaptivePolicy | None = None, **kw):
    """Mock-draft strategy (see src.draft.mock): take the category-aware policy's choice for my pick."""
    cache: dict[int, AdaptivePolicy] = {}

    def strategy(engine: DraftEngine, state) -> int:
        pol = policy or cache.setdefault(id(engine), AdaptivePolicy(engine, **kw))
        taken, rosters = engine._initial(state)
        cc = engine._center_counts(rosters)
        j = pol.pick(taken, rosters, [0] * engine.settings.n_teams, pol.plain_orders(), cc, state.next_pick_no)
        return int(engine.pid[j])
    return strategy
