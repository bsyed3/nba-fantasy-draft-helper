import numpy as np
import pandas as pd

from src.features.player_seasons import STAT_DEFS
from src.models import projection as P


def synthetic_seasons(n_players=150, years=range(2015, 2024), seed=0) -> pd.DataFrame:
    """Players with persistent talent + noise, minutes that grow with experience."""
    rng = np.random.default_rng(seed)
    rows = []
    for pid in range(1, n_players + 1):
        rookie = int(rng.integers(2012, 2022))
        talent = rng.normal(0, 0.25)
        base_mpg = rng.uniform(15, 34)
        pos = rng.choice(list("GFC"))
        pick = float(rng.integers(1, 62))
        for y in years:
            if y < rookie or y > rookie + 12:
                continue
            gp = int(np.clip(rng.normal(66, 12), 5, 82))
            mpg = base_mpg * (1 + 0.04 * min(y - rookie, 4)) * rng.uniform(0.9, 1.1)
            mins = gp * mpg

            def r(base):
                return max(base * np.exp(talent + rng.normal(0, 0.08)), 0.05) * mins / 36

            fga2, fga3, fta = r(5), r(3), r(3)
            rows.append(dict(
                player_id=pid, year=y, season=f"{y}-{str(y + 1)[-2:]}", gp=gp, min=mins, sg=82,
                fga2=fga2, fga3=fga3, fgm2=fga2 * np.clip(rng.normal(0.52, 0.03), 0.3, 0.7),
                tpm=fga3 * np.clip(rng.normal(0.36, 0.04), 0.1, 0.5), fta=fta,
                ftm=fta * np.clip(rng.normal(0.77, 0.05), 0.3, 0.95),
                reb=r(6), ast=r(4), stl=r(1.1), blk=r(0.7), tov=r(2), age=20 + (y - rookie), name=f"P{pid}",
                pos=pos, draft_pick=pick, rookie_year=rookie, last_year=2024,
            ))
    return pd.DataFrame(rows)


def test_pooled_ratio_shrinks_toward_prior_and_weights_recent():
    tot, den, prior = np.array([[10.0, 0.0, 0.0]]), np.array([[10.0, 0.0, 0.0]]), np.array([0.5])
    assert P.pooled_ratio(tot, den, prior, 1.0, 0)[0] == 1.0                         # no shrinkage -> observed ratio
    assert abs(P.pooled_ratio(tot, den, prior, 1.0, 10)[0] - 0.75) < 1e-12           # equal weight prior/data
    assert P.pooled_ratio(np.zeros((1, 3)), np.zeros((1, 3)), prior, 1.0, 0)[0] == 0.5  # no data -> prior
    t2, d2 = np.array([[2.0, 8.0, 0.0]]), np.array([[10.0, 10.0, 0.0]])
    assert P.pooled_ratio(t2, d2, prior, 0.5, 0)[0] < P.pooled_ratio(t2, d2, prior, 1.0, 0)[0]


def test_percentages_are_aggregated_from_totals_not_averaged():
    # 1/100 then 1/1 -> 2/101, not the average of 1% and 100%
    est = P.pooled_ratio(np.array([[1.0, 1.0, 0.0]]), np.array([[100.0, 1.0, 0.0]]), np.array([0.5]), 1.0, 0)[0]
    assert abs(est - 2 / 101) < 1e-12


def test_per_game_assembly_derives_points_from_makes():
    rates = pd.DataFrame([{**{s: 4.0 for s in P.RATE_STATS}, "p2": 0.5, "p3": 0.4, "ft": 0.8, "mpg": 36.0, "avail": 0.8}])
    pg = P.to_per_game(rates)
    assert abs(pg["fgm"][0] - (4 * 0.5 + 4 * 0.4)) < 1e-9
    assert abs(pg["pts"][0] - (2 * pg["fgm"][0] + pg["tpm"][0] + pg["ftm"][0])) < 1e-9
    assert pg["tpm"][0] <= pg["fgm"][0]


def test_backtest_and_projection_run_on_synthetic_history():
    st = synthetic_seasons()
    bt = P.backtest(st, folds=[2021, 2022, 2023], verbose=False)
    assert set(bt.selection) == set(STAT_DEFS)
    assert (bt.summary.groupby("stat")["selected"].sum() == 1).all()
    assert len(bt.oos) > 100 and bt.oos["pts"].notna().all()

    players = st.groupby("player_id").agg(name=("name", "first"), draft_position=("draft_pick", "first"),
                                          rookie_year=("rookie_year", "first")).reset_index()
    players["position_1"], players["position_2"] = "G", None
    players["draft_year"], players["last_year"] = players["rookie_year"], 2024
    extra = pd.DataFrame([{"player_id": 9999, "name": "Rookie", "position_1": "F", "position_2": None, "draft_year": 2024,
                           "draft_position": 3, "rookie_year": 2024, "last_year": 2024}])
    players = pd.concat([players, extra], ignore_index=True)
    proj = P.project_season(st, players, 2024, bt.selection)
    assert proj["player_id"].is_unique
    assert proj.loc[proj["player_id"] == 9999, "source"].iloc[0] == "rookie_tier"
    assert proj["pts"].between(0, 60).all() and proj["avail"].between(0, 1).all()


def test_returning_veteran_without_history_does_not_get_a_top_pick_rookie_profile():
    st = synthetic_seasons()
    bt = P.backtest(st, folds=[2022, 2023], verbose=False)
    players = st.groupby("player_id").agg(name=("name", "first"), draft_position=("draft_pick", "first"),
                                          rookie_year=("rookie_year", "first")).reset_index()
    players["position_1"], players["position_2"] = "G", None
    players["draft_year"], players["last_year"] = players["rookie_year"], 2024
    extra = pd.DataFrame([
        {"player_id": 9001, "name": "True #3 pick", "position_1": "G", "position_2": None, "draft_year": 2024,
         "draft_position": 3, "rookie_year": 2024, "last_year": 2024},
        {"player_id": 9002, "name": "Returned from Europe", "position_1": "G", "position_2": None, "draft_year": 2015,
         "draft_position": 3, "rookie_year": 2015, "last_year": 2024},   # drafted #3 nine years ago, no NBA seasons in the window
        {"player_id": 9003, "name": "Undrafted rookie", "position_1": "G", "position_2": None, "draft_year": None,
         "draft_position": None, "rookie_year": 2024, "last_year": 2024},
    ])
    proj = P.project_season(st, pd.concat([players, extra], ignore_index=True), 2024, bt.selection).set_index("player_id")
    cols = ["mpg", "pts", "reb", "avail"]
    # the veteran is projected exactly like an undrafted rookie, not like the #3 pick he once was
    assert np.allclose(proj.loc[9002, cols].astype(float), proj.loc[9003, cols].astype(float))
    assert proj.loc[9001, "source"] == proj.loc[9002, "source"] == "rookie_tier"
