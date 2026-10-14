import numpy as np
import pandas as pd

from src.draft.engine import DraftEngine, category_matrix
from src.draft.policy import AdaptivePolicy, adaptive_pick, calibrate_rank_weights
from src.draft.state import DraftState, LeagueSettings
from src.draft.values import CATEGORIES
from src.simulation.copula import build_library
from src.simulation.library import STAT_IDX, STAT_ORDER, Library
from tests.conftest import make_projection


def hand_engine(lines: dict[int, dict[str, int]], z_total: list[float], n_teams=2, roster=2, my_slot=1):
    """One simulated week; every player plays day 0 only, with the given stat line. Player ids 1..n."""
    n = len(z_total)
    proj = make_projection(n)
    proj["elig"] = ["SG/SF/PF"] * n
    for s in ("PG", "SG", "SF", "PF", "C"):
        proj[f"e_{s}"] = s != "PG" and s != "C"
    proj["is_center"] = False
    proj["z_total"] = z_total
    stats = np.zeros((n, 1, 7, len(STAT_ORDER)), dtype=np.uint8)
    played = np.zeros((n, 1, 7), dtype=bool)
    for pid, line in lines.items():
        played[pid - 1, 0, 0] = True
        for k, v in line.items():
            stats[pid - 1, 0, 0, STAT_IDX[k]] = v
    lib = Library(proj["player_id"].to_numpy(), stats, played)
    s = LeagueSettings(n_teams=n_teams, roster_size=roster, active_slots=roster, my_slot=my_slot)
    return DraftEngine(lib, proj, s), s


def small_engine(cal, schedule, n_teams=4, roster=5, active=4, my_slot=2):
    proj = make_projection(40)
    lib = build_library(proj, cal, schedule, sims_per_week=50, seed=0)
    s = LeagueSettings(n_teams=n_teams, roster_size=roster, active_slots=active, my_slot=my_slot, max_centers=3)
    return DraftEngine(lib, proj, s), s


# ---------------------------------------------------------------------------
# the proxy
# ---------------------------------------------------------------------------
def test_weekly_table_and_roster_cats_match_the_library(cal, schedule):
    eng, s = small_engine(cal, schedule)
    pol = AdaptivePolicy(eng, rank_weights=np.ones(s.roster_size))
    assert np.allclose(pol.W[3], eng.lib.stats[3].sum(axis=1))                 # whole-week totals, no lineup rules
    ros = [3, 9, 14]
    expect = category_matrix(sum(eng.lib.stats[i].sum(axis=1).astype(np.float64) for i in ros))
    assert np.allclose(pol.team_cats(ros), expect, rtol=1e-4)


def test_win_prob_counts_categories_and_treats_turnovers_as_lower_is_better(cal, schedule):
    eng, s = small_engine(cal, schedule)
    pol = AdaptivePolicy(eng)
    opp = np.ones((3, 5, len(CATEGORIES)))
    tov = CATEGORIES.index("tov")
    better = np.full((5, len(CATEGORIES)), 2.0)
    better[:, tov] = 0.0                                                         # fewer turnovers = better
    assert pol.win_prob(better, opp) == 1.0
    assert pol.win_prob(np.where(np.arange(len(CATEGORIES)) == tov, 2.0, 0.0) * np.ones((5, 1)), opp) == 0.0
    four = np.full((5, len(CATEGORIES)), 0.0)
    four[:, :4] = 2.0
    four[:, tov] = 5.0                                                           # 4 of 9 is not enough...
    assert pol.win_prob(four, opp) == 0.0
    five = four.copy()
    five[:, 4] = 2.0                                                             # ...5 of 9 is
    assert pol.win_prob(five, opp) == 1.0


def test_drafts_the_player_that_flips_the_matchup_even_when_he_is_not_the_board_choice():
    # I hold A (3PM). The opponent has X + Y. The board prefers C1 (more 3PM, a category I already win), but C2 gives me
    # REB/AST/STL and the matchup. 5 of 9 categories wins; ties in FG%, FT%, TOV split.
    lines = {1: dict(tpm=20), 2: dict(tpm=10, reb=10), 3: dict(ast=10, stl=10, blk=10),
             4: dict(tpm=30), 5: dict(reb=11, ast=11, stl=11), 6: {}}
    eng, s = hand_engine(lines, z_total=[9, 5, 4, 8, 7, 1])
    state = DraftState(s, [(1, 1, 1), (2, 2, 2), (3, 2, 3)])
    pol = AdaptivePolicy(eng, n_candidates=4, margin=0.0, rank_weights=np.ones(2))
    taken, rosters = eng._initial(state)
    scored = dict(pol.score_candidates(taken, rosters, [0, 0], pol.plain_orders(), eng._center_counts(rosters), state.next_pick_no))
    assert scored[eng.idx_of[4]] < 0.5 < scored[eng.idx_of[5]]                   # C1 loses the week, C2 wins it
    assert adaptive_pick(pol)(eng, state) == 5
    assert eng.pid[int(eng.static_order[[not taken[j] for j in eng.static_order].index(True)])] == 4   # the board would have taken C1


def test_margin_keeps_the_board_choice_unless_the_gain_is_large():
    lines = {1: dict(tpm=20), 2: dict(tpm=10, reb=10), 3: dict(ast=10, stl=10, blk=10),
             4: dict(tpm=30), 5: dict(reb=11, ast=11, stl=11), 6: {}}
    eng, s = hand_engine(lines, z_total=[9, 5, 4, 8, 7, 1])
    state = DraftState(s, [(1, 1, 1), (2, 2, 2), (3, 2, 3)])
    # C2 is ahead by exactly 1.0 (a certain win against a certain loss): a margin of 1.0 is not exceeded
    assert adaptive_pick(AdaptivePolicy(eng, n_candidates=4, margin=1.0, rank_weights=np.ones(2)))(eng, state) == 4
    assert adaptive_pick(AdaptivePolicy(eng, n_candidates=4, margin=0.9, rank_weights=np.ones(2)))(eng, state) == 5


def test_policy_respects_the_center_cap_and_never_picks_a_drafted_player(cal, schedule):
    eng, s = small_engine(cal, schedule, my_slot=1)
    pol = AdaptivePolicy(eng, n_candidates=6)
    st = DraftState(s, centers=set(eng.center_ids))
    c_ids = sorted(eng.center_ids)
    for pid in c_ids[:3]:                                                        # my roster would hold 3 centers: the cap
        st.picks.append((len(st.picks) + 1, 1, pid))
    taken, rosters = eng._initial(st)
    cands = pol.candidates(taken, eng._center_counts(rosters)[0])
    assert cands and not any(eng.is_center[j] or taken[j] for j in cands)
    pick = adaptive_pick(pol)(eng, st)
    assert pick not in st.drafted and pick not in eng.center_ids


# ---------------------------------------------------------------------------
# engine integration
# ---------------------------------------------------------------------------
def test_policy_is_off_by_default_and_the_recommendation_is_unchanged_by_attaching_none(cal, schedule):
    eng, s = small_engine(cal, schedule, my_slot=1)
    assert eng.policy is None
    a = eng.recommend(DraftState(s), top_k=5, rollouts=1)
    eng.policy = None
    pd.testing.assert_frame_equal(a, eng.recommend(DraftState(s), top_k=5, rollouts=1))


def test_rollouts_with_the_policy_still_build_legal_complete_rosters_and_are_deterministic(cal, schedule):
    eng, s = small_engine(cal, schedule, my_slot=2)
    eng.policy = AdaptivePolicy(eng, n_candidates=4)
    ev, rosters = eng.projected_outcome(DraftState(s, centers=set(eng.center_ids)))
    assert all(len(r) == s.roster_size and len(set(r)) == len(r) for r in rosters.values())
    assert len({i for r in rosters.values() for i in r}) == s.n_teams * s.roster_size            # nobody drafted twice
    assert all(sum(eng.is_center[i] for i in r) <= s.max_centers for r in rosters.values())
    a = eng.recommend(DraftState(s, centers=set(eng.center_ids)), top_k=4, rollouts=1)
    b = eng.recommend(DraftState(s, centers=set(eng.center_ids)), top_k=4, rollouts=1)
    pd.testing.assert_frame_equal(a, b)
    assert not a.empty and a["win_prob"].between(0, 1).all()


def test_lineup_usage_reporting_does_not_change_totals_and_top_players_always_play(cal, schedule):
    eng, s = small_engine(cal, schedule, roster=8, active=5)
    ros = list(range(8))
    usage: dict = {}
    assert np.array_equal(eng.team_totals(ros), eng.team_totals(ros, usage=usage))
    assert all(0 <= u <= p for u, p in usage.values())
    w = calibrate_rank_weights(eng, [ros, list(range(10, 18))])
    assert w.shape == (8,) and w[0] == 1.0 and (w >= 0).all() and (w <= 1).all()
