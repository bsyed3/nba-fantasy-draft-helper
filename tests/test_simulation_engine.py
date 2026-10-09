import numpy as np
import pandas as pd

from src.draft.engine import DraftEngine, category_matrix
from src.draft.state import DraftState, LeagueSettings
from src.draft.values import CATEGORIES
from src.simulation.copula import build_library, weekly_means
from src.simulation.library import STAT_IDX, STAT_ORDER, Library
from tests.conftest import make_projection


def test_library_weekly_means_match_projection_on_the_schedule(cal, schedule):
    proj = make_projection(30)
    lib = build_library(proj, cal, schedule, sims_per_week=600, seed=1)
    wm = weekly_means(lib)
    team_games = schedule.masks.sum(axis=2).mean(axis=0)  # avg games/week per schedule team
    exp_games = proj["avail"] * proj["team_id"].map(dict(zip(schedule.team_ids, team_games)))
    assert abs((wm["games"] / exp_games).mean() - 1) < 0.03
    for col in ("reb", "ast", "tov", "stl"):
        ratio = wm[col] / (proj[col] * exp_games)
        assert abs(ratio.mean() - 1) < 0.05, col
    pts = 2 * wm["fgm"] + wm["tpm"] + wm["ftm"]
    assert abs((pts / (proj["pts"] * exp_games)).mean() - 1) < 0.05
    s = lib.stats
    assert (s[..., STAT_IDX["fgm"]] <= s[..., STAT_IDX["fga"]]).all()
    assert (s[..., STAT_IDX["ftm"]] <= s[..., STAT_IDX["fta"]]).all()
    assert (s[..., STAT_IDX["tpm"]] <= s[..., STAT_IDX["fgm"]]).all()
    assert (s[~lib.played] == 0).all()  # no stats on days without a game


def test_players_only_play_on_their_teams_schedule_days(cal, schedule):
    proj = make_projection(12)
    lib = build_library(proj, cal, schedule, sims_per_week=50, seed=2)
    wk = np.arange(lib.n_sims) % schedule.n_weeks
    tmap = schedule.team_index()
    for i, t in enumerate(proj["team_id"]):
        allowed = schedule.masks[wk, tmap[int(t)]]          # [S, 7]
        assert not (lib.played[i] & ~allowed).any()


def test_library_has_cross_stat_correlation_and_is_reproducible(cal, schedule):
    proj = make_projection(10)
    a = build_library(proj, cal, schedule, sims_per_week=300, seed=3)
    b = build_library(proj, cal, schedule, sims_per_week=300, seed=3)
    assert np.array_equal(a.stats, b.stats)
    wk = a.stats[0].sum(axis=1)  # weekly totals for player 0
    r = np.corrcoef(wk[:, STAT_IDX["reb"]], wk[:, STAT_IDX["ast"]])[0, 1]
    assert r > 0.3


def test_library_save_load(tmp_path, cal, schedule):
    lib = build_library(make_projection(8), cal, schedule, sims_per_week=5, seed=0)
    lib.save(tmp_path / "l.npz")
    back = Library.load(tmp_path / "l.npz")
    assert np.array_equal(back.stats, lib.stats) and np.array_equal(back.played, lib.played)


def test_category_matrix_uses_ratios_not_averages():
    totals = np.zeros((2, 10))
    totals[0, [STAT_IDX["fgm"], STAT_IDX["fga"]]] = [5, 10]
    totals[1, [STAT_IDX["fgm"], STAT_IDX["fga"]]] = [1, 10]
    cats = category_matrix(totals)
    assert cats[0, CATEGORIES.index("fg_pct")] == 0.5 and cats[1, CATEGORIES.index("fg_pct")] == 0.1
    assert cats[0, CATEGORIES.index("pts")] == 10  # 2*5 + 0 + 0


# ---------------------------------------------------------------------------
# daily lineup slot rules (hand-built library, no randomness)
# ---------------------------------------------------------------------------
def slot_engine(elig_by_player: list[str], plays_day0: list[bool], n_teams=2, roster=None):
    n = len(elig_by_player)
    proj = make_projection(n)
    proj["elig"] = elig_by_player
    for s in ("PG", "SG", "SF", "PF", "C"):
        proj[f"e_{s}"] = [s in e.split("/") for e in elig_by_player]
    proj["is_center"] = proj["e_C"]
    proj["z_total"] = np.linspace(10, 1, n)  # player 0 is the highest priority
    stats = np.zeros((n, 1, 7, len(STAT_ORDER)), dtype=np.uint8)
    played = np.zeros((n, 1, 7), dtype=bool)
    for i, p in enumerate(plays_day0):
        played[i, 0, 0] = p
        stats[i, 0, 0, STAT_IDX["reb"]] = 1 if p else 0
    lib = Library(proj["player_id"].to_numpy(), stats, played)
    s = LeagueSettings(n_teams=n_teams, roster_size=n, active_slots=10)
    return DraftEngine(lib, proj, s), list(range(n))


def day_reb(eng, roster):
    return eng.team_totals(roster)[0, STAT_IDX["reb"]]


def test_five_centers_on_one_day_fill_only_C_and_util():
    eng, ros = slot_engine(["C"] * 5, [True] * 5)
    assert day_reb(eng, ros) == 4  # 1 C slot + 3 UTIL; G/F slots are closed to centers


def test_ten_pg_only_players_fill_PG_G_and_util():
    eng, ros = slot_engine(["PG"] * 10, [True] * 10)
    assert day_reb(eng, ros) == 5  # PG + G + 3 UTIL


def test_balanced_roster_fills_all_ten_active_slots_and_bench_sits():
    elig = ["PG", "SG", "SG", "SF", "PF", "SF/PF", "C", "PG/SG", "C", "SF", "PF", "SG", "C"]
    eng, ros = slot_engine(elig, [True] * 13)
    assert day_reb(eng, ros) == 10  # 13 play, only 10 slots


def test_bench_player_covers_when_a_starter_has_no_game():
    elig = ["C", "C", "C", "C"]
    eng, ros = slot_engine(elig, [True, False, True, True])
    assert day_reb(eng, ros) == 3  # three centers play, all fit in C + 2 UTIL


def test_center_cap_in_state_and_rollouts(cal, schedule):
    proj = make_projection(60)
    lib = build_library(proj, cal, schedule, sims_per_week=10, seed=0)
    s = LeagueSettings(n_teams=4, roster_size=6, active_slots=5, my_slot=1, max_centers=1)
    eng = DraftEngine(lib, proj, s)
    st = DraftState(s, centers=set(eng.center_ids))
    c_ids = sorted(eng.center_ids)
    st.make_pick(c_ids[0])
    assert not st.can_draft(c_ids[1], slot=1) and st.can_draft(c_ids[1], slot=2)
    st.make_pick(next(p for p in proj["player_id"] if p not in eng.center_ids))  # team 2, non-center
    ev, rosters = eng.projected_outcome(st)
    for t, r in rosters.items():
        assert sum(eng.is_center[i] for i in r) <= 1, f"team {t} exceeded the center cap"
    rec = eng.recommend(DraftState(s, [(1, 1, c_ids[0]), (2, 2, 7), (3, 3, 8), (4, 4, 9), (5, 4, 10), (6, 3, 11),
                                       (7, 2, 12), (8, 1, 13)], centers=set(eng.center_ids)), top_k=15)
    assert not rec["player_id"].isin(c_ids).any()  # my roster already has its one center


# ---------------------------------------------------------------------------
# engine invariants
# ---------------------------------------------------------------------------
def small_engine(cal, schedule, n_teams=4, roster=5, active=4, my_slot=2, seed=0):
    proj = make_projection(40)
    lib = build_library(proj, cal, schedule, sims_per_week=50, seed=seed)
    s = LeagueSettings(n_teams=n_teams, roster_size=roster, active_slots=active, my_slot=my_slot, max_centers=3)
    return DraftEngine(lib, proj, s), s, proj


def test_win_probabilities_are_zero_sum(cal, schedule):
    eng, s, _ = small_engine(cal, schedule)
    ev, rosters = eng.projected_outcome(DraftState(s))
    assert all(len(r) == s.roster_size for r in rosters.values())
    assert abs(ev.win_prob.mean() - 0.5) < 1e-9  # every matchup has exactly one winner (ties split)
    assert (ev.cat_win_prob >= 0).all() and (ev.cat_win_prob <= 1).all()


def test_recommend_prefers_strong_players_and_skips_taken_and_flagged(cal, schedule):
    eng, s, _ = small_engine(cal, schedule, my_slot=1)
    st = DraftState(s, centers=set(eng.center_ids))
    rec = eng.recommend(st, top_k=8, rollouts=2, extra_candidates=[35])
    assert rec.iloc[0]["player_id"] in (1, 2, 3, 4)  # the top of a strict talent gradient
    wp = rec.set_index("player_id")["win_prob"]
    assert wp[rec.iloc[0]["player_id"]] > wp[35] + 0.05  # a far weaker player is clearly worse
    st.make_pick(1)
    st.make_pick(2)
    st.excluded.add(3)
    rec = eng.recommend(st, top_k=8, rollouts=2)  # not my turn: plans for the next pick
    assert not set(rec["player_id"]) & {1, 2, 3}
    assert rec["win_prob"].is_monotonic_decreasing


def test_recommendation_is_deterministic(cal, schedule):
    eng, s, _ = small_engine(cal, schedule)
    a = eng.recommend(DraftState(s), top_k=6, rollouts=2)
    b = eng.recommend(DraftState(s), top_k=6, rollouts=2)
    pd.testing.assert_frame_equal(a, b)


def test_no_recommendation_when_roster_full(cal, schedule):
    eng, s, _ = small_engine(cal, schedule, n_teams=2, roster=2, active=2, my_slot=1)
    st = DraftState(s)
    for pid in (1, 2, 3, 4):
        st.make_pick(pid)
    assert eng.recommend(st).empty


def test_bots_follow_espn_adp_when_given(cal, schedule):
    proj = make_projection(40)
    proj["espn_adp"] = np.nan
    proj.loc[proj["player_id"] == 40, "espn_adp"] = 1.0   # ESPN's #1 overall is our worst player
    lib = build_library(proj, cal, schedule, sims_per_week=5, seed=0)
    s = LeagueSettings(n_teams=4, roster_size=5, active_slots=4, my_slot=4)
    eng = DraftEngine(lib, proj, s, noise_base=0.0, noise_slope=0.0)
    ev, rosters = eng.projected_outcome(DraftState(s))
    assert eng.idx_of[40] in rosters[1]  # team 1's first pick is ESPN's #1


def test_lineup_trace_matches_team_totals(cal, schedule):
    eng, s, _ = small_engine(cal, schedule, roster=8)
    roster = list(range(10, 18))
    for sim in (0, 7, 31):
        trace = eng.lineup_trace(roster, sim)
        from_trace = np.zeros(len(STAT_ORDER))
        for t in trace:
            if t["slot"] != "BENCH":
                from_trace += eng.lib.stats[t["idx"], sim, t["day"]]
        assert np.allclose(from_trace, eng.team_totals(roster)[sim])
        assert max(sum(1 for t in trace if t["day"] == d and t["slot"] != "BENCH") for d in range(7)) <= s.active_slots
