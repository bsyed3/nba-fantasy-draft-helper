"""
The Draft page: enter every pick as it happens (all teams, not just yours). On your turn -- or any time via "Plan my next pick" -- the
engine ranks the available players by the simulated probability that adding them gets your team a weekly H2H matchup win, given
everything drafted so far.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from src.app.common import (COL_LABELS, PER_GAME_COLS, Ctx, all_players, cached_outlook, cached_recommend, cat_columns, get_policy, label)
from src.draft import store
from src.draft.engine import DEFAULT_PICK_MARGIN, recommended_pick
from src.draft.values import CAT_LABELS, CATEGORIES
from src.features.names import search_key


def render(ctx: Ctx) -> None:
    proj, proj_by_id, state, S, engine, conn = ctx.proj, ctx.proj_by_id, ctx.state, ctx.S, ctx.engine, ctx.conn
    tname, player_name = ctx.tname, ctx.player_name

    max_rank = st.sidebar.slider("Evaluate every player up to board rank", 30, 300, 200, 10,
                                 help="Every player still available whose board rank is within this cap is simulated as my pick "
                                      "(about 3-4 seconds at 200).")
    rollouts = st.sidebar.slider("Mock-draft rollouts per candidate", 1, 6, 2)
    pick_rule = st.sidebar.radio(
        "Best-pick rule", ["Category-aware policy", "Engine simulation"],
        help="Category-aware policy (default): of the board's top few available players, take the one whose finished roster is most "
             "likely to win at least 5 of 9 categories; on real 2022-25 seasons it beat drafting by the board by about +3 points of "
             "weekly win probability. Engine simulation: the highest simulated win % in the table below (near-ties go to the better-ranked player). "
             "In 60 simulated drafts the two rules were within noise of each other; the policy is far quicker.")
    _label2id = {f"{r['name']} (board #{int(r['rank'])})": int(r["player_id"]) for _, r in proj.sort_values("rank").iterrows()}
    _picked = st.sidebar.multiselect(
        "Always evaluate these players", list(_label2id),
        help="The recommender simulates the top board/per-game players. A player the availability-discounted board ranks low "
             "(e.g. a star coming off an injury-shortened year) is skipped unless you add him here.")
    watch = [_label2id[x] for x in _picked]

    c1, c2 = st.sidebar.columns(2)
    if c1.button("↩ Undo last pick", disabled=not state.picks, width="stretch"):
        store.undo_pick(conn, state)
        st.rerun()
    with st.sidebar.expander("Danger zone"):
        if st.checkbox("I want to reset this draft") and st.button("Reset draft", type="primary"):
            store.reset_draft(conn, state)
            st.rerun()

    st.title("Draft")
    # --- header ---------------------------------------------------------------
    if state.done:
        st.success("Draft complete.")
    else:
        rnd = (state.next_pick_no - 1) // S.n_teams + 1
        h = st.columns(4)
        h[0].metric("Pick", f"#{state.next_pick_no} of {S.total_picks}")
        h[1].metric("Round", rnd)
        h[2].metric("On the clock", tname(state.team_on_clock) + (" (YOU)" if state.my_turn else ""))
        until = state.picks_until_my_turn()
        h[3].metric("Picks until your turn", "NOW" if until == 0 else (until if until is not None else "-"))
        if state.my_turn:
            st.success("🟢 It's your pick.")

    tab_draft, tab_team, tab_league, tab_board = st.tabs(["Draft", "My team", "League", "Player board"])

    drafted = state.drafted
    avail = proj[~proj["player_id"].isin(drafted | state.excluded)].sort_values("rank")

    # --- draft tab ------------------------------------------------------------
    with tab_draft:
        left, right = st.columns([1, 2.4])
        with left:
            st.subheader("Record a pick")
            if state.done:
                st.write("No picks left.")
            else:
                ok = avail["player_id"].map(state.can_draft)
                legal, blocked = avail[ok], avail[~ok]
                options = {int(r["player_id"]): label(r) for _, r in legal.iterrows()}               # everyone still available, not just the top of the board
                pid = st.selectbox(f"{tname(state.team_on_clock)} selects", list(options), format_func=options.get,
                                   key=f"pick_{state.next_pick_no}",
                                   help="Type to search. Plain spelling works: 'dayron' finds Day'Ron Sharpe, 'doncic' finds Dončić.")
                if len(blocked):                                                                      # not a missing player: the league's center cap hides him
                    st.caption(f"{len(blocked)} available players are hidden because {tname(state.team_on_clock)} already has "
                               f"{S.max_centers} centers (the league maximum): " + ", ".join(blocked["name"].head(6)) + (" ..." if len(blocked) > 6 else ""))
                if st.button(f"Record pick #{state.next_pick_no}", type="primary", width="stretch"):
                    store.record_pick(conn, state, pid)
                    st.rerun()
                with st.expander("Player not on the board?"):
                    oq = st.text_input("Search every NBA player in the database", key=f"other_q_{state.next_pick_no}")
                    if oq:
                        ap = all_players()
                        hits = ap[ap["search"].str.contains(search_key(oq), regex=False) & ~ap["player_id"].isin(set(proj["player_id"]) | drafted)]
                        hits = hits.sort_values("last_year", ascending=False).head(25)
                        if hits.empty:
                            st.caption("No match outside the board (a player already on the board is in the box above).")
                        else:
                            other = {int(r["player_id"]): f"{r['name']} (last played {int(r['last_year'])}-{str(int(r['last_year']) + 1)[-2:]})" for _, r in hits.iterrows()}
                            opid = st.selectbox("Player", list(other), format_func=other.get, key=f"other_{state.next_pick_no}")
                            if st.button("Record this pick (no projection)", key=f"other_rec_{state.next_pick_no}"):
                                store.record_pick(conn, state, opid)
                                st.rerun()
                            st.caption("He takes the team's pick and roster slot but adds nothing to its projected totals, since he has no projection.")
            st.divider()
            st.caption("Recent picks")
            recent = pd.DataFrame(
                [{"#": p[0], "Team": tname(p[1]), "Player": player_name(p[2])} for p in state.picks[-8:][::-1]])
            st.dataframe(recent, hide_index=True, width="stretch")

        with right:
            st.subheader("Recommendations")
            if state.done or state.my_next_pick_no() is None:
                st.write("Your roster is complete.")
            else:
                plan = state.my_turn or st.toggle("Plan my next pick (simulate bots up to it)", value=False)
                if plan:
                    policy = get_policy(engine, S)
                    pol_pick, pol_table = policy.plan(state)
                    extra = set(watch) | ({pol_pick} if pol_pick is not None else set())     # the policy's pick must be in the table
                    rec = cached_recommend(engine, S, tuple(state.picks), tuple(sorted(state.excluded)), max_rank, rollouts, tuple(sorted(extra)), tuple(state.moves))
                    if rec.empty:
                        st.write("Nothing to evaluate.")
                    else:
                        eng_best = recommended_pick(rec)
                        use_policy = pick_rule.startswith("Category") and pol_pick is not None and pol_pick in set(rec["player_id"])
                        best = rec[rec["player_id"] == pol_pick].iloc[0] if use_policy else eng_best
                        st.markdown(f"**Best pick: {best['name']}** — {best['win_prob']:.1%} chance to win a weekly matchup "
                                    f"(≈ {best['exp_wins']:.1f} wins over {S.reg_season_weeks} weeks). "
                                    f"Board rank #{int(best['static_rank'])}, per-game rank #{int(best['pergame_rank'])}. "
                                    f"{len(rec)} players simulated.")
                        if use_policy:
                            same = "the engine's simulation agrees." if eng_best["player_id"] == best["player_id"] else \
                                f"the engine's own top choice is {eng_best['name']} ({eng_best['win_prob']:.1%})."
                            st.caption(f"Chosen by the category-aware policy: {same}")
                            with st.expander("Why this pick? Category-aware candidates"):
                                why = pol_table.assign(Player=proj_by_id.loc[pol_table["player_id"], "name"].to_numpy())
                                why["vs board's top"] = (why["proxy_win_prob"] - why.sort_values("board_rank").iloc[0]["proxy_win_prob"]) * 100
                                why["proxy_win_prob"] *= 100
                                why["Pick"] = np.where(why["chosen"], "★", "")
                                st.dataframe(why[["Pick", "Player", "board_rank", "proxy_win_prob", "vs board's top"]].rename(
                                    columns={"board_rank": "Board rank", "proxy_win_prob": "P(win ≥5 of 9) %"}), hide_index=True, width="stretch",
                                    column_config={"P(win ≥5 of 9) %": st.column_config.NumberColumn(format="%.1f"),
                                                   "vs board's top": st.column_config.NumberColumn(format="%+.1f")})
                                st.caption(f"Each of the board's top {len(why)} available players is added to my roster, the draft is finished with "
                                           f"the bots' usual order, and the whole roster is scored on {policy.W.shape[1]:,} simulated weeks. A player is taken "
                                           f"over the board's top choice only if he is ahead by more than {policy.margin * 100:.1f} points (smaller gaps are noise).")
                        else:
                            top = rec.iloc[0]
                            if top["player_id"] != best["player_id"]:
                                st.caption(f"Highest simulated number: {top['name']} ({top['win_prob']:.1%}). It is within "
                                           f"{DEFAULT_PICK_MARGIN:.0%} of {best['name']}, so the board-ranked player is recommended: "
                                           f"with this many candidates, differences that small are mostly simulation noise.")
                        stat_cols = ["gp82", "gp_med", "gp_rng", "mpg", "pts", "tpm", "reb", "ast", "stl", "blk", "fg_pct", "ft_pct", "tov"]
                        info = proj_by_id.loc[rec["player_id"], stat_cols].reset_index(drop=True)
                        info[["fg_pct", "ft_pct"]] *= 100
                        cat_cols_ = [f"cat_{c}" for c in CATEGORIES]
                        show = pd.concat([rec[["name", "elig", "static_rank", "pergame_rank", "win_prob", "vs_top_ranked"]].reset_index(drop=True),
                                          info, rec[cat_cols_].reset_index(drop=True)], axis=1)
                        pct_cols = ["win_prob", "vs_top_ranked"] + cat_cols_
                        show[pct_cols] = show[pct_cols] * 100
                        show.insert(0, "Pick", ["★" if p == best["player_id"] else "" for p in rec["player_id"]])
                        show = show.rename(columns={**COL_LABELS, "elig": "Pos", "static_rank": "Board rank", "pergame_rank": "Per-game rank",
                                                    "win_prob": "Win %", "vs_top_ranked": "vs top-ranked"})
                        one = st.column_config.NumberColumn
                        st.dataframe(
                            show, hide_index=True, width="stretch", height=620,
                            column_config={"Win %": one(format="%.1f%%"), "vs top-ranked": one(format="%+.1f%%"),
                                           "Proj GP (mean)": one(format="%.0f"), "Typical GP (median)": one(format="%.0f"), "MPG": one(format="%.1f"), "PPG": one(format="%.1f"),
                                           "3PM": one(format="%.1f"), "REB": one(format="%.1f"), "AST": one(format="%.1f"),
                                           "STL": one(format="%.1f"), "BLK": one(format="%.1f"), "TO": one(format="%.1f"),
                                           "FG%": one(format="%.1f%%"), "FT%": one(format="%.1f%%"), **cat_columns()},
                        )
                        st.caption("Win % = chance my team beats a random opponent in a weekly matchup (averaged over all "
                                   f"{S.reg_season_weeks + S.playoff_weeks} weeks: {S.reg_season_weeks} regular-season + {S.playoff_weeks} playoff) "
                                   "if I take this player and "
                                   "the rest of the draft plays out like the mock-draft model; category columns = chance to win each "
                                   "category. Compare candidates by difference, not absolute level. Proj GP (mean) = expected games "
                                   "played (of 82) for a player who is healthy at tip-off, including the chance of an in-season lost stretch; "
                                   "players already known to be out at the start are set in data/external/known_absences.csv; Typical GP (median) and the P25–P75 range "
                                   "show what a player like him usually plays (the median is higher because season-ending injuries skew "
                                   "the average down; display only). MPG/PPG/etc. are per-game averages when he plays. Board rank "
                                   "discounts counting stats by projected (mean) games; per-game rank ignores availability. The pool includes the top per-game players "
                                   "even if the availability-discounted board ranks them low.")
                else:
                    st.info("Not your turn yet. Enter picks as they happen, or flip the toggle to plan ahead.")

    # --- my team --------------------------------------------------------------
    with tab_team:
        mine = state.roster(S.my_slot)
        st.subheader(f"My roster ({len(mine)}/{S.roster_size}) · centers {state.n_centers(S.my_slot)}/{S.max_centers}")
        st.dataframe(ctx.roster_frame(mine), width="stretch")
        if state.ir(S.my_slot):
            st.caption("Injured reserve: " + ", ".join(player_name(i) for i in state.ir(S.my_slot)))
        if mine and not state.done:
            ev, _ = cached_outlook(engine, S, tuple(state.picks), tuple(sorted(state.excluded)), tuple(state.moves))
            me = S.my_slot - 1
            st.subheader("Projected outlook")
            st.caption("Rest of the draft filled in by the mock-draft model.")
            m = st.columns(2)
            m[0].metric("Weekly matchup win %", f"{ev.win_prob[me]:.1%}")
            m[1].metric(f"Expected wins / {S.reg_season_weeks}", f"{ev.win_prob[me] * S.reg_season_weeks:.1f}")
            chart = pd.DataFrame({"Category": [CAT_LABELS[c] for c in CATEGORIES], "Win %": ev.cat_win_prob[me] * 100}).set_index("Category")
            st.bar_chart(chart)

    # --- league ---------------------------------------------------------------
    with tab_league:
        if state.done:
            st.write("Draft complete -- final rosters below.")
        ev, rosters = cached_outlook(engine, S, tuple(state.picks), tuple(sorted(state.excluded)), tuple(state.moves))
        rows = []
        for t in range(1, S.n_teams + 1):
            drafted_now = state.roster(t)
            rows.append({"Team": tname(t, " (me)"), "Players": len(drafted_now),
                         "Projected win %": ev.win_prob[t - 1] * 100,
                         "Exp. wins": ev.win_prob[t - 1] * S.reg_season_weeks,
                         "Roster": ", ".join(player_name(i) for i in drafted_now) + ("".join(f"  [IR: {player_name(i)}]" for i in state.ir(t)))})
        st.dataframe(pd.DataFrame(rows).sort_values("Projected win %", ascending=False), hide_index=True, width="stretch",
                     column_config={"Projected win %": st.column_config.NumberColumn(format="%.1f%%"),
                                    "Exp. wins": st.column_config.NumberColumn(format="%.1f")})
        st.caption("Projected win % = each team's chance to win a weekly matchup vs a random opponent once the draft is "
                   "completed by the mock-draft model. Rosters include any waiver pickups and drops since the draft; players on injured reserve do not count.")
        if state.moves:
            with st.expander(f"Roster moves since the draft ({len(state.moves)})"):
                st.dataframe(pd.DataFrame([{"#": i + 1, "Team": tname(t), "Move": {"add": "added", "drop": "dropped", "ir": "to IR", "activate": "off IR"}[a], "Player": player_name(p)}
                                           for i, (t, p, a) in enumerate(state.moves)]), hide_index=True, width="stretch")

    # --- board ----------------------------------------------------------------
    with tab_board:
        f1, f2, f3 = st.columns([2, 1, 1])
        q = f1.text_input("Search player")
        pos = f2.multiselect("Position", ["G", "F", "C"])
        show_flagged = f3.checkbox("Show flagged-out players", value=False)
        b = proj[~proj["player_id"].isin(drafted)].copy()
        if not show_flagged:
            b = b[~b["player_id"].isin(state.excluded)]
        if q:
            b = b[b["search"].str.contains(search_key(q), regex=False, na=False)]            # plain spelling: "dayron", "doncic", "pj washington"
        if pos:
            b = b[b["position_1"].isin(pos)]
        cols = ["rank", "pg_rank", "name", "pos", "team", "age_t", "espn_adp", "gp82", "gp_med", "gp_rng"] + PER_GAME_COLS + ["z_total"]
        bb = b[cols].copy()
        bb[["fg_pct", "ft_pct"]] *= 100
        st.dataframe(bb.round(1).rename(columns=COL_LABELS), hide_index=True, width="stretch", height=560)
        st.divider()
        st.caption("Flag a player who is injured / out for the start of the season: the recommender will skip him and "
                   "assume nobody drafts him in the mock draft.")
        names = {int(r["player_id"]): f"{r['name']} ({r['team']})" for _, r in proj.sort_values("rank").iterrows()
                 if int(r["player_id"]) not in drafted}
        flagged = [p for p in state.excluded if p in names]
        new_flags = st.multiselect("Flagged out", list(names), default=flagged, format_func=names.get)
        if set(new_flags) != set(flagged):
            for p in set(new_flags) - set(flagged):
                store.set_excluded(conn, state, p, True)
            for p in set(flagged) - set(new_flags):
                store.set_excluded(conn, state, p, False)
            st.rerun()
