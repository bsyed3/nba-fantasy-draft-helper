"""
Phase 3 -- live draft recommendation engine.

Objective: maximise the probability that my team wins a weekly H2H
matchup (more categories than the opponent; the week is one W/L, margin
and category count don't matter beyond that). That is deliberately not a
sum of z-scores: it values balance, punting, depth and variance correctly
because it is computed on simulated weeks.

For every candidate pick it:
  1. fixes the candidate as my pick,
  2. rolls the *rest of the draft* forward: every other team (and my own
     later picks) takes the best player left by a noisy static ranking --
     a mock-draft model -- respecting the center cap,
  3. plays each team's roster through the simulated weeks with DAILY
     lineups: each day, in order of static value, every player who has a
     game takes the first open slot he is eligible for (dedicated PG/SG/
     SF/PF/C, then G or F, then UTIL); the rest of the day he sits.
     Bench players therefore contribute whenever a starter has no game or
     is out, and positional scarcity falls out of the slot rules,
  4. scores my team against every other team over all simulated weeks.
Averaging over a few noisy rollouts (common random numbers across
candidates, so differences between candidates are low-variance) gives
each candidate's win probability and per-category win probabilities.

Known simplifications (stated, not hidden): the daily lineup is a greedy
fill in static-value order (a manager would optimise per day, and move
players between slots with the matching-problem a greedy fill can miss);
bots follow a ranking, not positional need or personal bias; position
eligibility is derived from coarse labels unless overridden (see
src/features/positions.py).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.draft.state import DraftState, LeagueSettings, team_for_pick
from src.draft.values import CATEGORIES, CAT_LABELS, z_scores
from src.simulation.library import STAT_IDX, Library

DEDICATED = ["C", "PG", "SG", "SF", "PF"]
N_DEDICATED_AND_FLEX = 7  # PG SG SF PF C + G + F; the remaining active slots are UTIL


def category_matrix(totals: np.ndarray) -> np.ndarray:
    """totals [..., 10] (STAT_ORDER) -> [..., 9] in CATEGORIES order."""
    g = lambda k: totals[..., STAT_IDX[k]]
    with np.errstate(invalid="ignore", divide="ignore"):
        fg = np.where(g("fga") > 0, g("fgm") / g("fga"), 0.0)
        ft = np.where(g("fta") > 0, g("ftm") / g("fta"), 0.0)
    pts = 2 * g("fgm") + g("tpm") + g("ftm")
    return np.stack([pts, g("tpm"), g("reb"), g("ast"), g("stl"), g("blk"), fg, ft, g("tov")], axis=-1)


@dataclass
class TeamEval:
    win_prob: np.ndarray       # [T] each team's matchup win prob vs a random other team
    cat_win_prob: np.ndarray   # [T, 9] each team's per-category win prob vs a random other team


class DraftEngine:
    def __init__(self, lib: Library, proj: pd.DataFrame, settings: LeagueSettings, seed: int = 0,
                 noise_base: float = 3.0, noise_slope: float = 0.12):
        """
        proj: projections with player_id, z_total (static value), name, position_1 and the eligibility
        columns from features.positions.add_eligibility; must contain every player in `lib`.
        """
        self.lib, self.settings, self.seed = lib, settings, seed
        self.noise_base, self.noise_slope = noise_base, noise_slope
        self.proj = proj.set_index("player_id").loc[lib.player_ids]
        self.value = self.proj["z_total"].to_numpy(float)
        self.pid = lib.player_ids
        self.idx_of = {int(p): i for i, p in enumerate(self.pid)}
        order = np.argsort(-self.value, kind="stable")
        self.rank = np.empty(len(order), dtype=int)
        self.rank[order] = np.arange(len(order))
        self.static_order = order
        # A second ordering that ignores availability (pure per-game value). The board above discounts every counting stat by
        # projected games played, which can bury a star coming off an injury-shortened year; this lets such players still be
        # *evaluated* by the simulation (which models injuries directly) via recommend(..., pergame_k=N).
        pg = self.proj.copy()
        pg["avail"] = 1.0
        zpg = z_scores(pg, n_pool=min(156, len(pg)))["z_total"].to_numpy()
        self.pergame_order = np.argsort(-zpg, kind="stable")
        self.pergame_rank = np.empty(len(zpg), dtype=int)
        self.pergame_rank[self.pergame_order] = np.arange(len(zpg))
        # What bots draft by: ESPN's average draft position where known (auto-draft and most managers follow it),
        # otherwise our own static rank, placed after the ADP-ranked players.
        adp = self.proj["espn_adp"].to_numpy(float) if "espn_adp" in self.proj else np.full(len(self.pid), np.nan)
        self.bot_base = np.where(np.isfinite(adp), adp - 1, np.nanmax(np.append(adp, 0)) + self.rank * 1.0)
        self.is_center = self.proj["is_center"].to_numpy(bool)
        self.center_ids = {int(p) for p in self.pid[self.is_center]}
        self.n_util = max(settings.active_slots - N_DEDICATED_AND_FLEX, 0)
        elig = {s: self.proj[f"e_{s}"].to_numpy(bool) for s in DEDICATED}
        self.slot_order: list[list[str]] = []
        for i in range(len(self.pid)):
            ded = [s for s in DEDICATED if elig[s][i]]
            flex = (["G"] if elig["PG"][i] or elig["SG"][i] else []) + (["F"] if elig["SF"][i] or elig["PF"][i] else [])
            self.slot_order.append(ded + flex)

    # ------------------------------------------------------------------
    # draft roll-out
    # ------------------------------------------------------------------
    def _bot_orders(self, rollout: int, state: DraftState) -> np.ndarray:
        """[T, P] noisy preference order per team (my team uses the noise-free order)."""
        T, P = self.settings.n_teams, len(self.pid)
        rng = np.random.default_rng([self.seed, rollout, len(state.picks)])
        sd = self.noise_base + self.noise_slope * self.bot_base
        orders = np.empty((T, P), dtype=np.int32)
        for t in range(T):
            if t + 1 == self.settings.my_slot:  # my later picks: noise-free static ranking
                orders[t] = np.argsort(self.rank, kind="stable")
            else:
                orders[t] = np.argsort(self.bot_base + rng.standard_normal(P) * sd, kind="stable")
        return orders

    def _initial(self, state: DraftState):
        taken = np.zeros(len(self.pid), dtype=bool)
        rosters = {t: [] for t in range(1, self.settings.n_teams + 1)}
        for _, t, p in state.picks:
            i = self.idx_of.get(int(p))
            if i is not None:
                taken[i] = True
                rosters[t].append(i)
        for p in state.excluded:
            i = self.idx_of.get(int(p))
            if i is not None:
                taken[i] = True  # nobody drafts a flagged-out player in the roll-out
        return taken, rosters

    def _center_counts(self, rosters) -> list[int]:
        return [sum(self.is_center[i] for i in rosters[t]) for t in range(1, self.settings.n_teams + 1)]

    def _advance(self, taken, rosters, ptr, orders, cc, start_pick: int, end_pick: int) -> None:
        """Auto-draft picks [start_pick, end_pick] in place (respects the center cap)."""
        n, cap = self.settings.n_teams, self.settings.max_centers
        for pn in range(start_pick, end_pick + 1):
            t = team_for_pick(pn, n)
            o, k = orders[t - 1], ptr[t - 1]
            while taken[o[k]] or (self.is_center[o[k]] and cc[t - 1] >= cap):
                k += 1
            ptr[t - 1] = k + 1
            j = int(o[k])
            taken[j] = True
            rosters[t].append(j)
            cc[t - 1] += int(self.is_center[j])

    # ------------------------------------------------------------------
    # evaluation
    # ------------------------------------------------------------------
    def team_totals(self, roster: list[int]) -> np.ndarray:
        """[S, 10] weekly category totals for a roster under daily lineups."""
        stats, played = self.lib.stats, self.lib.played
        S, D = played.shape[1:]
        free = {s: np.ones((S, D), dtype=bool) for s in DEDICATED + ["G", "F"]}
        util_left = np.full((S, D), self.n_util, dtype=np.int8)
        total = np.zeros((S, stats.shape[-1]), dtype=np.float32)
        for i in sorted(roster, key=lambda j: -self.value[j]):
            m = played[i]
            if not m.any():
                continue
            got = np.zeros((S, D), dtype=bool)
            for slot in self.slot_order[i]:
                take = m & ~got & free[slot]
                free[slot] &= ~take
                got |= take
            take = m & ~got & (util_left > 0)
            util_left -= take
            got |= take
            total += np.einsum("sd,sdk->sk", got.astype(np.float32), stats[i].astype(np.float32))
        return total

    def lineup_trace(self, roster: list[int], sim: int) -> list[dict]:
        """
        Who fills which slot on each day of simulated week `sim` (same rules as team_totals, for
        one sim, with the assignment recorded). Returns [{day, slot, idx} ...] for explanation/debugging.
        """
        played = self.lib.played
        D = played.shape[2]
        out = []
        for d in range(D):
            free = {s: 1 for s in DEDICATED + ["G", "F"]}
            util_left = self.n_util
            for i in sorted(roster, key=lambda j: -self.value[j]):
                if not played[i, sim, d]:
                    continue
                slot = next((s for s in self.slot_order[i] if free[s]), None)
                if slot is not None:
                    free[slot] = 0
                elif util_left > 0:
                    slot, util_left = "UTIL", util_left - 1
                out.append({"day": d, "slot": slot or "BENCH", "idx": i})
        return out

    def evaluate(self, rosters: dict[int, list[int]]) -> TeamEval:
        T = self.settings.n_teams
        totals = np.stack([self.team_totals(rosters[t]) for t in range(1, T + 1)])    # [T, S, 10]
        cats = category_matrix(totals)                                                # [T, S, 9]
        lower = np.array([c == "tov" for c in CATEGORIES])
        a, b = cats[:, None], cats[None, :]                                            # team t vs team u
        better = np.where(lower, a < b, a > b)
        cat_score = better + 0.5 * (a == b)                                            # [T, T, S, 9]
        wins = cat_score.sum(-1)                                                       # [T, T, S]
        match = (wins > 4.5) + 0.5 * (wins == 4.5)
        off = ~np.eye(T, dtype=bool)
        win_prob = np.array([match[t][off[t]].mean() for t in range(T)])
        cat_wp = np.array([cat_score[t][off[t]].mean(axis=(0, 1)) for t in range(T)])
        return TeamEval(win_prob, cat_wp)

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------
    def available(self, state: DraftState) -> np.ndarray:
        taken, _ = self._initial(state)
        return np.flatnonzero(~taken)

    def recommend(self, state: DraftState, top_k: int = 25, rollouts: int = 2,
                  extra_candidates: list[int] | None = None, pergame_k: int = 0) -> pd.DataFrame:
        """
        Rank candidates for my next pick by simulated matchup win probability.
        If it is not my turn yet, the bots' picks up to my next pick are
        simulated first and candidates come from who is projected to be left.
        """
        my_pick = state.my_next_pick_no()
        if my_pick is None or state.done:
            return pd.DataFrame()
        me, S, cap = self.settings.my_slot, self.settings, self.settings.max_centers
        results: dict[int, list[tuple[float, np.ndarray]]] = {}
        cand_list: list[int] | None = None

        for r in range(rollouts):
            orders = self._bot_orders(r, state)
            taken, rosters = self._initial(state)
            ptr, cc = [0] * S.n_teams, self._center_counts(rosters)
            self._advance(taken, rosters, ptr, orders, cc, state.next_pick_no, my_pick - 1)
            if cand_list is None:  # rollout 0 defines the candidate pool
                ok = lambda i: not taken[i] and not (self.is_center[i] and cc[me - 1] >= cap)
                cand_list = [i for i in self.static_order if ok(i)][:top_k]
                for i in [j for j in self.pergame_order if ok(j)][:pergame_k]:   # stars the availability-discounted board buries
                    if i not in cand_list:
                        cand_list.append(i)
                for p in extra_candidates or []:
                    i = self.idx_of.get(int(p))
                    if i is not None and ok(i) and i not in cand_list:
                        cand_list.append(i)
            for c in cand_list:
                if taken[c] or (self.is_center[c] and cc[me - 1] >= cap):
                    continue  # a bot took him before my pick in this roll-out
                tk, ros = taken.copy(), {t: list(v) for t, v in rosters.items()}
                pt, cc2 = list(ptr), list(cc)
                tk[c] = True
                ros[me].append(c)
                cc2[me - 1] += int(self.is_center[c])
                self._advance(tk, ros, pt, orders, cc2, my_pick + 1, S.total_picks)
                ev = self.evaluate(ros)
                results.setdefault(c, []).append((ev.win_prob[me - 1], ev.cat_win_prob[me - 1]))

        rows = []
        for c in cand_list or []:
            res = results.get(c)
            if not res:
                continue
            cw = np.mean([x[1] for x in res], axis=0)
            row = {"player_id": int(self.pid[c]), "static_rank": int(self.rank[c]) + 1, "pergame_rank": int(self.pergame_rank[c]) + 1,
                   "win_prob": float(np.mean([x[0] for x in res])), "n_rollouts": len(res)}
            row.update({f"cat_{k}": float(v) for k, v in zip(CATEGORIES, cw)})
            rows.append(row)
        df = pd.DataFrame(rows)
        if df.empty:
            return df
        df["exp_wins"] = df["win_prob"] * S.reg_season_weeks
        df["vs_top_ranked"] = df["win_prob"] - df.sort_values("static_rank").iloc[0]["win_prob"]
        meta = self.proj[["name", "position_1", "elig"]].reset_index()
        df = df.merge(meta, on="player_id", how="left")
        return df.sort_values("win_prob", ascending=False).reset_index(drop=True)

    def projected_outcome(self, state: DraftState, rollout: int = 0) -> tuple[TeamEval, dict[int, list[int]]]:
        """Finish the draft with the bot model and score every team (power rankings)."""
        orders = self._bot_orders(rollout, state)
        taken, rosters = self._initial(state)
        ptr, cc = [0] * self.settings.n_teams, self._center_counts(rosters)
        self._advance(taken, rosters, ptr, orders, cc, state.next_pick_no, self.settings.total_picks)
        return self.evaluate(rosters), rosters


def cat_table(cat_win_prob: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame({"category": [CAT_LABELS[c] for c in CATEGORIES], "win_prob": cat_win_prob})
