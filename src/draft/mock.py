"""
Mock drafts: bots + one "me" strategy, for validating the recommender.

Bots draft from a noisy version of the static ranking (noise grows with
rank, a different noise draw per team), which is a stand-in for the
spread of real ADP-driven drafters. The noise here is deliberately larger
than the engine's internal opponent model so validation is not graded
against the engine's own assumptions.
"""
from __future__ import annotations

from typing import Callable

import numpy as np

from src.draft.engine import DraftEngine
from src.draft.state import DraftState, LeagueSettings

Strategy = Callable[[DraftEngine, DraftState], int]  # returns a player_id


def best_available(engine: DraftEngine, state: DraftState) -> int:
    """Highest static value left that my roster may legally take (center cap)."""
    taken = engine._initial(state)[0]
    me = state.settings.my_slot
    full_c = state.n_centers(me) >= state.settings.max_centers
    return int(engine.pid[next(i for i in engine.static_order if not taken[i] and not (full_c and engine.is_center[i]))])


def engine_pick(top_k: int = 20, rollouts: int = 2, margin: float = 0.0) -> Strategy:
    """
    Take the engine's best candidate. With margin > 0 the engine only overrides the value ranking when it
    is ahead by more than `margin` (win-probability units): among candidates within `margin` of the best,
    take the one with the best static value rank. That guards against picking the argmax of noisy near-ties.
    """
    def pick(engine: DraftEngine, state: DraftState) -> int:
        rec = engine.recommend(state, top_k=top_k, rollouts=rollouts)
        if margin > 0:
            close = rec[rec["win_prob"] >= rec["win_prob"].iloc[0] - margin]
            return int(close.sort_values("static_rank").iloc[0]["player_id"])
        return int(rec.iloc[0]["player_id"])
    return pick


def run_mock_draft(engine: DraftEngine, settings: LeagueSettings, strategy: Strategy, bot_seed: int,
                   bot_noise_base: float = 6.0, bot_noise_slope: float = 0.2) -> dict[int, list[int]]:
    """Returns {team_slot: [lib index, ...]} suitable for DraftEngine.evaluate."""
    rng = np.random.default_rng(bot_seed)
    P = len(engine.pid)
    sd = bot_noise_base + bot_noise_slope * engine.bot_base
    orders = [np.argsort(engine.bot_base + rng.standard_normal(P) * sd, kind="stable") for _ in range(settings.n_teams)]
    ptr = [0] * settings.n_teams
    state = DraftState(settings, centers=set(engine.center_ids))
    taken = np.zeros(P, dtype=bool)
    rosters: dict[int, list[int]] = {t: [] for t in range(1, settings.n_teams + 1)}
    while not state.done:
        t = state.team_on_clock
        if t == settings.my_slot:
            j = engine.idx_of[strategy(engine, state)]
        else:
            o, k = orders[t - 1], ptr[t - 1]
            while taken[o[k]] or not state.can_draft(int(engine.pid[o[k]]), t):
                k += 1
            ptr[t - 1] = k + 1
            j = int(o[k])
        taken[j] = True
        rosters[t].append(j)
        state.make_pick(int(engine.pid[j]))
    return rosters
