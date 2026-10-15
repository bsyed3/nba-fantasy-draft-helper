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
            smiss = int(rng.integers(10, 50)) if rng.random() < 0.12 else 0            # a known injury at tip-off
            gp = int(np.clip(rng.normal(66, 12), 5, 82 - smiss))
            mpg = base_mpg * (1 + 0.04 * min(y - rookie, 4)) * rng.uniform(0.9, 1.1)
            mins = gp * mpg

            def r(base):
                return max(base * np.exp(talent + rng.normal(0, 0.08)), 0.05) * mins / 36

            fga2, fga3, fta = r(5), r(3), r(3)
            rows.append(dict(
                player_id=pid, year=y, season=f"{y}-{str(y + 1)[-2:]}", gp=gp, min=mins, sg=82, smiss=float(smiss), sgo=float(82 - smiss),
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


def test_availability_for_players_who_missed_last_season_is_pulled_toward_the_baseline(monkeypatch):
    from src.features.player_seasons import league_priors

    st = synthetic_seasons()
    have = st.groupby("player_id")["year"].apply(set)
    skippers = [p for p, ys in have.items() if {2020, 2021, 2023} <= ys][:10]
    assert skippers
    st = st[~((st["player_id"].isin(skippers)) & (st["year"] == 2022))]          # a whole missed season
    priors, sg = league_priors(st), st.groupby("year")["sg"].first().to_dict()
    train = pd.concat([P.build_wide(st, priors, t, sg) for t in range(2016, 2023)], ignore_index=True)
    test = P.build_wide(st, priors, 2023, sg)
    returner = (test["gap"] > 0).to_numpy()
    assert returner.sum() >= len(skippers) and (~returner).sum() > 20

    def avail_preds(weight):
        monkeypatch.setattr(P, "RETURNER_BASELINE_WEIGHT", weight)
        return P.predict_all(train, test)["avail"]

    pure, full = avail_preds(0.0), avail_preds(1.0)
    for model in ("ridge", "lgbm"):
        assert np.allclose(full[model][returner], full["baseline"][returner])            # weight 1 -> the baseline
        assert np.allclose(full[model][~returner], pure[model][~returner])               # everyone else untouched
    assert not np.allclose(pure["lgbm"][returner], pure["baseline"][returner])           # weight 0 -> the raw model, which differs


def test_games_played_quantiles_put_the_median_above_a_left_skewed_mean_and_stay_ordered():
    rng = np.random.default_rng(1)
    pred = rng.uniform(0.55, 0.85, 4000)
    lost = rng.random(4000) < 0.15                                              # season-ending injuries: availability collapses
    act = np.where(lost, rng.uniform(0.0, 0.4, 4000), np.clip(pred + 0.12 + rng.normal(0, 0.06, 4000), 0, 1))
    ev = pd.DataFrame({"pred_avail": pred, "pred_mpg": 30.0, "act_avail": act})
    coefs = P.fit_gp_quantiles(ev)
    q = P.apply_gp_quantiles(pd.Series([0.6, 0.75, 0.85]), coefs)
    assert (q["gp_p25"] <= q["gp_median"]).all() and (q["gp_median"] <= q["gp_p75"]).all()
    assert (q["gp_median"] > np.array([0.6, 0.75, 0.85]) * 82 - 1e-9).all()    # skew: typical > mean-based games
    assert q["gp_median"].is_monotonic_increasing and (q.to_numpy() >= 0).all() and (q.to_numpy() <= 82).all()


def test_backtest_returns_selection_free_availability_evaluation():
    st = synthetic_seasons()
    bt = P.backtest(st, folds=[2021, 2022, 2023], verbose=False)
    ev = bt.avail_eval
    assert {"year", "player_id", "pred_avail", "pred_mpg", "act_avail"} <= set(ev.columns)
    assert ev["pred_avail"].between(0, 1).all() and ev["act_avail"].between(0, 1).all() and len(ev) > 100


def test_team_context_features_are_used_only_when_supplied_and_survive_a_backtest():
    from src.features.player_seasons import league_priors
    from src.features.team_context import CTX_COLS

    st = synthetic_seasons()
    rng = np.random.default_rng(3)
    ids = st["player_id"].unique()
    ctx = pd.DataFrame([dict(player_id=int(p), year=int(y), **{c: float(rng.uniform(0, 80)) for c in CTX_COLS}) for p in ids for y in range(2016, 2025)])
    ctx["ctx_moved"] = (ctx["ctx_moved"] > 60).astype(float)
    priors, sg = league_priors(st), st.groupby("year")["sg"].first().to_dict()
    plain = P.build_wide(st, priors, 2022, sg)
    withc = P.build_wide(st, priors, 2022, sg, ctx)
    params = P.fit_baseline_params(plain)
    assert not any(c.startswith("ctx_") for c in P.feature_matrix(plain, params).columns)               # no context -> exactly the old features
    X = P.feature_matrix(withc, params)
    assert {"ctx_vac_min", "ctx_net_poss_x_usage", "ctx_moved_x_usage", "usage_pg"} <= set(X.columns)
    assert len(plain) == len(withc)
    bt = P.backtest(st, folds=[2021, 2022], verbose=False, context=ctx)
    assert set(bt.selection) == set(STAT_DEFS) and len(bt.oos) > 50


def test_dropped_season_slides_earlier_seasons_forward_and_leaves_everyone_else_alone():
    st = synthetic_seasons()
    pid = int(st[st["year"] == 2022]["player_id"].iloc[0])
    mine = st[st["player_id"] == pid].set_index("year")
    assert {2021, 2022} <= set(mine.index)
    out = P.drop_seasons(st, pd.DataFrame({"player_id": [pid], "year": [2022]}))
    new = out[out["player_id"] == pid].set_index("year")
    assert abs(new.loc[2022, "gp"] - mine.loc[2021, "gp"]) < 1e-9 and abs(new.loc[2022, "age"] - (mine.loc[2021, "age"] + 1)) < 1e-9   # 2021 now sits where 2022 was
    if 2020 in mine.index:
        assert abs(new.loc[2021, "gp"] - mine.loc[2020, "gp"]) < 1e-9                                                                      # ... and 2020 where 2021 was
    if 2023 in mine.index:
        assert new.loc[2023, "gp"] == mine.loc[2023, "gp"] and new.loc[2023, "age"] == mine.loc[2023, "age"]                            # later seasons are untouched
    assert len(new) == len(mine) - 1
    other = st[st["player_id"] != pid]
    assert out[out["player_id"] != pid].reset_index(drop=True).equals(other.reset_index(drop=True))
    assert P.drop_seasons(st, None) is st
    assert len(P.drop_seasons(st, pd.DataFrame({"player_id": [pid], "year": [1999]}))) == len(st)   # a season he never played: nothing to drop


def _synthetic_projection():
    st = synthetic_seasons()
    bt = P.backtest(st, folds=[2022, 2023], verbose=False)
    players = st.groupby("player_id").agg(name=("name", "first"), draft_position=("draft_pick", "first"), rookie_year=("rookie_year", "first")).reset_index()
    players["position_1"], players["position_2"] = "G", None
    players["draft_year"], players["last_year"] = players["rookie_year"], 2024
    extra = pd.DataFrame([{"player_id": 9999, "name": "Rookie Star", "position_1": "F", "position_2": None, "draft_year": 2024,
                           "draft_position": 3, "rookie_year": 2024, "last_year": 2024}])
    return P.project_season(st, pd.concat([players, extra], ignore_index=True), 2024, bt.selection)


def test_external_rookie_projection_matches_the_source_and_rewrites_every_simulated_stat():
    proj = _synthetic_projection()
    ext = pd.DataFrame([dict(name="Rookie Star", pts=18.4, reb=4.2, ast=3.1, stl=1.2, blk=0.4, fg_pct=0.437, tp_pct=0.368, mpg=np.nan),
                        dict(name="Not In The League", pts=10.0, reb=3.0, ast=2.0, stl=1.0, blk=0.5, fg_pct=0.45, tp_pct=0.35, mpg=25.0)])
    before = proj[proj["name"] == "Rookie Star"].iloc[0]
    out = P.apply_external_rookies(proj, ext)
    r = out[out["name"] == "Rookie Star"].iloc[0]
    assert r["source"] == "external_rookie"
    assert abs(r["pts"] - 18.4) < 1e-6 and abs(r["fg_pct"] - 0.437) < 1e-6 and abs(r["r_p3"] - 0.368) < 1e-9
    assert abs(r["reb"] - 4.2) < 1e-6 and abs(r["ast"] - 3.1) < 1e-6 and abs(r["stl"] - 1.2) < 1e-6 and abs(r["blk"] - 0.4) < 1e-6
    assert abs(r["mpg"] - before["mpg"]) < 1e-9 and abs(r["avail"] - before["avail"]) < 1e-9                  # no minutes given: keep the model's
    assert abs(r["fga2"] + r["fga3"] - r["fga"]) < 1e-9 and abs(r["fga3"] - r["tpa"]) < 1e-9
    assert abs(r["pts"] - (2 * r["fgm"] + r["tpm"] + r["ftm"])) < 1e-9                                       # internally consistent line
    assert abs(r["fgm"] - (r["fga2"] * r["r_p2"] + r["fga3"] * r["r_p3"])) < 1e-9                             # ... in the simulation's own terms too
    assert abs(r["r_reb"] - 4.2 / r["mpg"] * 36) < 1e-9
    others = out[out["name"] != "Rookie Star"].reset_index(drop=True)
    assert others.equals(proj[proj["name"] != "Rookie Star"].reset_index(drop=True))                          # nobody else touched


def test_no_context_ids_take_their_projection_from_a_context_free_run():
    from src.features.team_context import CTX_COLS

    st = synthetic_seasons()
    rng = np.random.default_rng(5)
    ids = st["player_id"].unique()
    ctx = pd.DataFrame([dict(player_id=int(p), year=int(y), **{c: float(rng.uniform(0, 80)) for c in CTX_COLS}) for p in ids for y in range(2016, 2025)])
    ctx["ctx_moved"] = (ctx["ctx_moved"] > 40).astype(float)
    bt = P.backtest(st, folds=[2022, 2023], verbose=False, context=ctx)
    players = st.groupby("player_id").agg(name=("name", "first"), draft_position=("draft_pick", "first"), rookie_year=("rookie_year", "first")).reset_index()
    players["position_1"], players["position_2"] = "G", None
    players["draft_year"], players["last_year"] = players["rookie_year"], 2024
    with_ctx = P.project_season(st, players, 2024, bt.selection, context=ctx).set_index("player_id")
    without = P.project_season(st, players, 2024, bt.selection, context=None).set_index("player_id")
    pid = int(with_ctx[with_ctx["source"] == "model"].index[0])
    swapped = P.project_season(st, players, 2024, bt.selection, context=ctx, no_context_ids={pid}).set_index("player_id")
    cols = ["pts", "mpg", "avail", "r_fga2", "r_p3", "reb"]
    assert np.allclose(swapped.loc[pid, cols].astype(float), without.loc[pid, cols].astype(float))              # his row = the context-free projection
    assert not np.allclose(with_ctx.loc[pid, cols].astype(float), without.loc[pid, cols].astype(float))        # ... which differs from the context one
    others = with_ctx.index != pid
    assert np.allclose(swapped.loc[others, cols].astype(float), with_ctx.loc[others, cols].astype(float))      # everyone else keeps the context projection


def test_sync_rates_from_per_game_brings_the_simulated_columns_in_step():
    proj = _synthetic_projection()
    i = proj.index[proj["name"] == "Rookie Star"][0]
    proj.loc[i, ["mpg", "fga", "tpa", "fgm", "tpm", "fta", "ftm", "reb"]] = [30.0, 15.0, 6.0, 7.0, 2.0, 4.0, 3.0, 5.0]
    P.sync_rates_from_per_game(proj, [i])
    r = proj.loc[i]
    assert (r["fga3"], r["fga2"]) == (6.0, 9.0) and abs(r["r_p3"] - 2 / 6) < 1e-9 and abs(r["r_p2"] - 5 / 9) < 1e-9 and abs(r["r_ft"] - 0.75) < 1e-9
    assert abs(r["r_reb"] - 5.0 / 30.0 * 36.0) < 1e-9 and abs(r["r_mpg"] - 30.0) < 1e-9


def test_known_absences_scale_the_active_availability(tmp_path):
    proj = pd.DataFrame({"player_id": [1, 2, 3], "name": ["Out For Year", "Back Late", "Healthy"], "avail": [0.8, 0.8, 0.8]})
    f = tmp_path / "ka.csv"
    f.write_text("name,games_out\nout for year,82\nBack Late,41\nNobody Here,10\n")
    out = P.apply_known_absences(proj, f)
    assert list(out["games_out"]) == [82.0, 41.0, 0.0]
    assert np.allclose(out["avail"], [0.0, 0.4, 0.8]) and np.allclose(out["avail_active"], 0.8)
    assert np.allclose(P.apply_known_absences(proj, None)["avail"], 0.8)           # no file: everyone active at tip-off
    manual = tmp_path / "manual.csv"
    manual.write_text("name,games_out\nBack Late,10\nHealthy,20\n")
    both = P.apply_known_absences(proj, [f, manual, tmp_path / "missing.csv"])    # later file wins; a missing file is skipped
    assert list(both["games_out"]) == [82.0, 10.0, 20.0]


def test_opening_absences_count_team_games_before_a_players_first_appearance():
    import sqlite3
    from src.features.player_seasons import opening_absences

    conn = sqlite3.connect(":memory:")
    conn.executescript("""
        CREATE TABLE game (game_id TEXT, date TEXT, home_team_id INT, away_team_id INT, season TEXT, game_type TEXT);
        CREATE TABLE box_score (game_id TEXT, player_id INT, team_id INT, min REAL);
    """)
    for i in range(10):                                                              # team 1 hosts team 2 ten times, one game a day
        conn.execute("INSERT INTO game VALUES (?,?,?,?,?,?)", (f"g{i}", f"2024-10-{20 + i}", 1, 2, "2024-25", "regular"))
    for i in range(10):
        conn.execute("INSERT INTO box_score VALUES (?,?,?,?)", (f"g{i}", 7, 1, 30.0))     # player 7 plays every game
        if i >= 4:
            conn.execute("INSERT INTO box_score VALUES (?,?,?,?)", (f"g{i}", 8, 1, 25.0))   # player 8 first appears in game 5
        if i == 9:
            conn.execute("INSERT INTO box_score VALUES (?,?,?,?)", (f"g{i}", 9, 2, 0.0))    # player 9: 0 minutes = did not play
    out = opening_absences(conn).set_index("player_id")
    assert out.loc[7, "smiss"] == 0 and out.loc[8, "smiss"] == 4 and out.loc[8, "tg"] == 10
    assert 9 not in out.index


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


def test_experiment_switches_change_features_without_changing_the_defaults():
    from src.features.player_seasons import league_priors
    from src.features.team_context import CTX_COLS

    st = synthetic_seasons()
    rng = np.random.default_rng(5)
    ctx = pd.DataFrame([dict(player_id=int(p), year=int(y), **{c: float(rng.uniform(0, 80)) for c in CTX_COLS})
                        for p in st["player_id"].unique() for y in range(2016, 2025)])
    ctx["ctx_moved"] = (ctx["ctx_moved"] > 60).astype(float)
    priors, sg = league_priors(st), st.groupby("year")["sg"].first().to_dict()
    wide = P.build_wide(st, priors, 2022, sg, ctx)
    params = P.fit_baseline_params(wide)
    plain, flagged = P.feature_matrix(wide, params), P.feature_matrix(wide, params, exp_flags=True)
    assert not any(c.startswith("is_exp") for c in plain.columns)                                   # off by default: shipped features unchanged
    assert {"is_exp1", "is_exp2", "is_exp3", "pick_x_exp2"} <= set(flagged.columns)
    assert flagged["is_exp2"].isin([0.0, 1.0]).all() and (flagged["is_exp2"] == (wide["exp"] == 2)).all()
    # dropping the context features from one stat's models changes that stat's predictions and leaves the others alone
    train = pd.concat([P.build_wide(st, priors, t, sg, ctx) for t in range(2016, 2022)], ignore_index=True)
    test = P.build_wide(st, priors, 2022, sg, ctx)
    base = P.predict_all(train, test)
    dropped = P.predict_all(train, test, drop={"avail": ("ctx_", "usage_pg")})
    assert set(base) == set(dropped)
    assert not np.allclose(base["avail"]["lgbm"], dropped["avail"]["lgbm"])
    assert np.allclose(base["reb"]["ridge"], dropped["reb"]["ridge"])
    assert set(P.predict_all(train, test, exp_flags=True)) == set(base)


def test_backtest_availability_evaluation_carries_subgroup_columns():
    bt = P.backtest(synthetic_seasons(), folds=[2021, 2022], verbose=False)
    assert {"base_avail", "base_mpg", "act_mpg", "gap", "exp", "age", "moved"} <= set(bt.avail_eval.columns)


def test_minutes_override_rescales_per_game_totals_explicit_values_win_and_reapplying_changes_nothing(tmp_path):
    from scripts.build_projections import apply_overrides

    proj = _synthetic_projection()
    i = proj.index[proj["name"] == "Rookie Star"][0]
    before = proj.loc[i].copy()
    others = proj.drop(index=i).copy()
    csv = tmp_path / "overrides.csv"
    pd.DataFrame([dict(name="Rookie Star", mpg=before["mpg"] * 1.25, reb=9.0)]).to_csv(csv, index=False)
    out = apply_overrides(proj, csv)
    r = out.loc[i]
    assert abs(r["mpg"] - before["mpg"] * 1.25) < 1e-9 and r["reb"] == 9.0                       # minutes set; the explicit rebound value wins
    for c in ("fga", "fgm", "tpa", "tpm", "fta", "ftm", "ast", "stl", "blk", "tov"):             # everything else scales with minutes
        assert abs(r[c] - before[c] * 1.25) < 1e-9, c
    assert abs(r["fg_pct"] - before["fgm"] / before["fga"]) < 1e-9                                # per-minute rates, so percentages do not move
    assert abs(r["fga2"] - (r["fga"] - r["tpa"])) < 1e-9 and r["source"] == "override"            # the simulated columns follow
    pd.testing.assert_frame_equal(out.drop(index=i)[others.columns], others)                      # nobody else changed
    again = apply_overrides(out, csv)
    assert np.allclose(again.loc[i, ["mpg", "fga", "ast", "reb"]].astype(float), r[["mpg", "fga", "ast", "reb"]].astype(float))   # applying twice = once
