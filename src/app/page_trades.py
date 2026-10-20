"""
The Trade evaluator page: pick the players each side sends, and see what the trade does to both teams' win probability, category by category.
"""
from __future__ import annotations

import pandas as pd
import streamlit as st

from src.app.common import CAT_COLS, Ctx, shade_multi
from src.draft import season, store

LEVEL = {"good": st.success, "ok": st.info, "warn": st.warning, "bad": st.error}
VALUE_GAP = 1.5          # board z-score difference above which a trade is flagged as uneven in raw player value


def _player_label(ctx: Ctx, pid: int) -> str:
    r = ctx.proj_by_id.loc[pid]
    return f"{r['name']} ({r['pos']}, {r['team']}) · board #{int(r['rank'])}"


def _drops_needed(ctx: Ctx, slot: int, sends: list[int], receives: list[int], key: str) -> list[int]:
    """The players a team must release when it takes back more players than it sends and would go over the roster limit."""
    S = ctx.S
    keep = [p for p in ctx.state.roster(slot) if p not in sends]
    over = len(keep) + len(receives) - S.roster_size
    if over <= 0:
        return []
    known = [p for p in keep if p in ctx.proj_by_id.index]
    default = sorted(known, key=lambda p: ctx.proj_by_id.loc[p, "z_total"])[:over]
    chosen = st.multiselect(f"{ctx.tname(slot)} is over the {S.roster_size}-player limit: drop {over} to make room", known, default=default, key=key,
                            format_func=lambda p: _player_label(ctx, p))
    if len(chosen) != over:
        st.warning(f"Choose exactly {over} player{'s' if over != 1 else ''} for {ctx.tname(slot)} to drop.")
    return chosen


def render(ctx: Ctx) -> None:
    S, state, engine = ctx.S, ctx.state, ctx.engine
    me = S.my_slot
    slots = list(range(1, S.n_teams + 1))
    st.title("Trade evaluator")
    if "trade_flash" in st.session_state:
        st.success(st.session_state.pop("trade_flash"))
    st.caption("Pick who each team sends. Both rosters are replayed through the simulated weeks with the trade made, and the change in each team's weekly "
               "win probability, category by category, shows who the trade helps. A fair trade helps both sides or costs the other side very little.")

    rosters = ctx.rosters_idx()
    before = ctx.league_table(rosters)

    c1, c2 = st.columns(2)
    a = c1.selectbox("Team", slots, index=me - 1, format_func=lambda t: ctx.tname(t, " (me)"), key="trade_a")
    others = [t for t in slots if t != a]
    b = c2.selectbox("Trades with", others, format_func=lambda t: ctx.tname(t, " (me)"), key=f"trade_b_{a}")
    known = lambda t: [p for p in state.roster(t) if p in ctx.proj_by_id.index and p in engine.idx_of]
    pa, pb = known(a), known(b)
    c1, c2 = st.columns(2)
    give_a = c1.multiselect(f"{ctx.tname(a)} sends", sorted(pa, key=lambda p: ctx.proj_by_id.loc[p, "rank"]), format_func=lambda p: _player_label(ctx, p), key=f"give_{a}_{b}")
    give_b = c2.multiselect(f"{ctx.tname(b)} sends", sorted(pb, key=lambda p: ctx.proj_by_id.loc[p, "rank"]), format_func=lambda p: _player_label(ctx, p), key=f"give_{b}_{a}")
    if not give_a or not give_b:
        st.info("Choose at least one player for each side to evaluate a trade. Only players with a projection can be traded here.")
        return
    if len(give_a) > 4 or len(give_b) > 4:
        st.warning("Trades of more than four players a side are not supported.")
        return
    drop_a = _drops_needed(ctx, a, give_a, give_b, f"drop_{a}_{b}")
    drop_b = _drops_needed(ctx, b, give_b, give_a, f"drop_{b}_{a}")

    idx = ctx.idx
    new = season.apply_trade(rosters, a, b, idx(give_a), idx(give_b), idx(drop_a), idx(drop_b))
    after = ctx.league_table(new)
    ia, ib = a - 1, b - 1
    d_a, d_b = (after.win[ia] - before.win[ia]) * 100, (after.win[ib] - before.win[ib]) * 100

    st.divider()
    if a == me:
        level, msg = season.verdict(d_a, d_b)
        LEVEL[level](f"**{msg}**")
    else:
        st.info("The verdict is shown for trades that involve your team; the numbers below cover any two teams.")

    # --- win probability, both teams ------------------------------------------------
    cols = st.columns(2)
    for col, t, i, d in ((cols[0], a, ia, d_a), (cols[1], b, ib, d_b)):
        with col:
            st.markdown(f"**{ctx.tname(t, ' (me)')}**")
            m = st.columns(3)
            m[0].metric("Weekly win %", f"{after.win[i] * 100:.1f}%", f"{d:+.1f} pts")
            m[1].metric("Regular season", f"{after.win_reg[i] * 100:.1f}%", f"{(after.win_reg[i] - before.win_reg[i]) * 100:+.1f}")
            m[2].metric("Playoff weeks", f"{after.win_po[i] * 100:.1f}%", f"{(after.win_po[i] - before.win_po[i]) * 100:+.1f}")
            st.caption(f"Power rank {before.power_rank[i]} → {after.power_rank[i]} · expected wins {before.win_reg[i] * S.reg_season_weeks:.1f} → "
                       f"{after.win_reg[i] * S.reg_season_weeks:.1f} of {S.reg_season_weeks}")
            rows = []
            for k, lab in enumerate(CAT_COLS):
                tot0, tot1 = before.totals[i, k] * (100 if season.RATIO[k] else 1), after.totals[i, k] * (100 if season.RATIO[k] else 1)
                rows.append({"Category": lab, "Weekly total": tot0, "Total after": tot1, "Win %": before.cat_win[i, k] * 100, "Win % after": after.cat_win[i, k] * 100,
                             "Change": (after.cat_win[i, k] - before.cat_win[i, k]) * 100, "Rank": f"{before.cat_rank[i, k]} → {after.cat_rank[i, k]}"})
            st.dataframe(shade_multi(pd.DataFrame(rows), [(["Change"], -10, 10)]).format(
                {"Weekly total": "{:.1f}", "Total after": "{:.1f}", "Win %": "{:.0f}%", "Win % after": "{:.0f}%", "Change": "{:+.0f}"}), hide_index=True, width="stretch")
    st.caption("Change = points of chance to win that category in a weekly matchup. Weekly totals are per 7 days; FG% and FT% in percent.")

    # --- value and fit ------------------------------------------------------------------
    st.subheader("Value and fit")
    rows = [{"Side": f"{ctx.tname(t)} sends", "Player": ctx.proj_by_id.loc[p, "name"], "Pos": ctx.proj_by_id.loc[p, "pos"], "Team": ctx.proj_by_id.loc[p, "team"],
             "Board rank": int(ctx.proj_by_id.loc[p, "rank"]), "Value": ctx.proj_by_id.loc[p, "z_total"], "Per-game rank": int(ctx.proj_by_id.loc[p, "pg_rank"]),
             "Proj GP": ctx.proj_by_id.loc[p, "gp82"]} for t, ps in ((a, give_a), (b, give_b)) for p in ps]
    st.dataframe(pd.DataFrame(rows).style.format({"Value": "{:.1f}", "Per-game rank": "{:.0f}", "Proj GP": "{:.0f}", "Board rank": "{:.0f}"}, na_rep="–"), hide_index=True, width="stretch")
    zv = lambda ps: float(sum(ctx.proj_by_id.loc[p, "z_total"] for p in ps))
    gap = zv(give_b) - zv(give_a)
    if abs(gap) >= VALUE_GAP:
        who = ctx.tname(a) if gap > 0 else ctx.tname(b)
        st.warning(f"On the draft board's value (z-score total), {who} receives {abs(gap):.1f} more than it sends: the trade is uneven on paper.")
    else:
        st.caption(f"Board value sent vs received differs by {abs(gap):.1f} (z-score total), roughly even.")
    for t, ps_after in ((a, new[a]), (b, new[b])):
        n_after, n_before = season.n_centers(engine, ps_after), season.n_centers(engine, rosters[t])
        if n_after > S.max_centers and n_after > n_before:
            st.warning(f"{ctx.tname(t)} would have {n_after} centers after the trade; the league allows {S.max_centers}.")
        elif n_after != n_before:
            st.caption(f"{ctx.tname(t)}: centers {n_before} → {n_after} (limit {S.max_centers}).")
    st.caption("Positions are approximate unless ESPN's exact eligibility was loaded, so the center count can be off by one.")

    # --- every team ---------------------------------------------------------------------------
    with st.expander("Effect on every team"):
        eff = pd.DataFrame({"Team": [ctx.tname(t, " (me)") for t in slots], "Win % before": before.win * 100, "Win % after": after.win * 100,
                            "Change": (after.win - before.win) * 100})
        st.dataframe(shade_multi(eff.sort_values("Change", ascending=False).reset_index(drop=True), [(["Change"], -3, 3)]).format(
            {"Win % before": "{:.1f}%", "Win % after": "{:.1f}%", "Change": "{:+.1f}"}), hide_index=True, width="stretch")
        st.caption("A trade that strengthens one team slightly lowers every other team's chances, because they face it.")

    # --- record --------------------------------------------------------------------------------
    with st.expander("Record this trade"):
        st.caption("If the trade happens, record it so every rosters-based page uses the new rosters. It can be undone from the Free agents page (Transactions).")
        ok = st.checkbox("The trade went through", key=f"trade_ok_{a}_{b}")
        if st.button("Record the trade", disabled=not ok, type="primary"):
            bad = (len(drop_a) != max(0, len([p for p in state.roster(a) if p not in give_a]) + len(give_b) - S.roster_size)
                   or len(drop_b) != max(0, len([p for p in state.roster(b) if p not in give_b]) + len(give_a) - S.roster_size))
            if bad:
                st.error("Choose the players to drop first.")
            else:
                try:
                    store.record_trade(ctx.conn, state, a, b, give_a, give_b, drop_a, drop_b, f"trade {ctx.tname(a)}-{ctx.tname(b)}")
                except ValueError as e:
                    st.error(f"Not recorded: {e}.")
                else:
                    st.session_state["trade_flash"] = f"Recorded the trade between {ctx.tname(a)} and {ctx.tname(b)}."
                    st.rerun()
