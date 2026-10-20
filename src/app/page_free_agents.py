"""
The Free agents page: who is available, and what each one would do for my team if I added him and dropped one of my players.
Also where roster transactions (pickups, drops, injured reserve) are recorded, for any team.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from src.app.common import CAT_COLS, COL_LABELS, LEAGUE, PER_GAME_COLS, Ctx, rosters_key, shade_multi
from src.draft import season, store
from src.draft.state import LeagueSettings
from src.draft.values import CATEGORIES
from src.features.names import search_key

POSITIONS = ["PG", "SG", "SF", "PF", "C"]
ACTIONS = {"add": "added", "drop": "dropped", "ir": "to IR", "activate": "off IR"}


@st.cache_data(show_spinner="Scoring each free agent against each of my players...")
def _grid(_engine, settings: LeagueSettings, rosters: tuple, me: int, adds: tuple, with_open: bool, _opp) -> season.SwapGrid:
    """Swap grid for my team; `rosters` (every team's roster) only keys the cache, since it fixes the opponents too."""
    roster = list(dict(rosters)[me])
    return season.swap_grid(_engine, roster, _opp, list(adds), roster + ([None] if with_open else []), settings.max_centers)


def _flash(msg: str | None = None) -> None:
    """Show (or, given a message, queue for after the next rerun) a one-line confirmation."""
    if msg is not None:
        st.session_state["fa_flash"] = msg
    elif "fa_flash" in st.session_state:
        st.success(st.session_state.pop("fa_flash"))


def _fa_label(ctx: Ctx, pid: int) -> str:
    r = ctx.proj_by_id.loc[pid]
    return f"{int(r['rank'])}. {r['name']} ({r['pos']}, {r['team']})"


def render(ctx: Ctx) -> None:
    S, state, engine = ctx.S, ctx.state, ctx.engine
    me = S.my_slot
    st.title("Free agents")
    _flash()

    rosters = ctx.rosters_idx()
    key = rosters_key(rosters)
    tbl = ctx.league_table(rosters)
    opp = season.field_cats(tbl, me)
    my_roster = rosters[me]
    base_win = tbl.win[me - 1] * 100

    fa_ids = ctx.free_agent_ids()
    pool = ctx.proj[ctx.proj["player_id"].isin(fa_ids)].sort_values("rank")
    open_slot = len(state.roster(me)) < S.roster_size
    m = st.columns(4)
    m[0].metric("Free agents with a projection", len(pool))
    m[1].metric("My weekly win %", f"{base_win:.1f}%")
    m[2].metric("My roster", f"{len(state.roster(me))}/{S.roster_size}" + (" (open spot)" if open_slot else ""))
    m[3].metric("My centers", f"{season.n_centers(engine, my_roster)}/{S.max_centers}")
    st.caption("A free agent is anyone on no team's roster (injured reserve counts as rostered). The pool and every number below move as roster moves are recorded "
               "at the bottom of this page. Gain = the change in my weekly win probability if I add him and drop the player in the Drop column "
               "(his best partner), measured against the other nine teams' current rosters on the same simulated weeks. Differences under about "
               f"{season.NOISE:.1f} points are within the simulation's noise.")

    # --- ranked table -----------------------------------------------------------
    st.subheader("Best pickups for my team")
    c1, c2, c3 = st.columns([2, 1.4, 1.2])
    q = c1.text_input("Search the table", placeholder="player name")
    pos = c2.multiselect("Position", POSITIONS)
    n_top = max(len(pool), 20)
    n_eval = c3.slider("Score the top N free agents", 10, n_top, min(100, n_top), 10,
                       help="Each one is tried against each of my players. About 15 seconds the first time for 100, then it is remembered until a roster changes.")
    adds_idx = [engine.idx_of[int(p)] for p in pool["player_id"].head(n_eval)]
    grid = _grid(engine, S, key, me, tuple(adds_idx), open_slot, opp)
    best = grid.best()
    info = ctx.proj_by_id.loc[[ctx.pid(i) for i in best["add"]]]
    df = pd.DataFrame({
        "Gain": best["gain"].to_numpy(),
        "Drop": [("(open roster spot)" if d is None else ctx.name_of(d)) if np.isfinite(g) else "(center cap)" for d, g in zip(best["drop"], best["gain"])],
        "Win % after": best["win"].to_numpy(), "Board rank": info["rank"].to_numpy(), "Player": info["name"].to_numpy(), "Pos": info["pos"].to_numpy(),
        "Team": info["team"].to_numpy(), "Proj GP": info["gp82"].to_numpy(), "Value": info["z_total"].to_numpy(), "Per-game rank": info["pg_rank"].to_numpy()})
    for c, lab in zip(CATEGORIES, CAT_COLS):
        df[f"Δ {lab}"] = best[f"d_{c}"].to_numpy()
    keep = np.ones(len(df), dtype=bool)
    if q:
        keep &= info["search"].str.contains(search_key(q), regex=False, na=False).to_numpy()
    for p in pos:
        keep &= info[f"e_{p}"].to_numpy(bool)
    df = df[keep].sort_values("Gain", ascending=False, na_position="last")
    if st.toggle("Show per-game stats", value=False):
        stats = info.loc[info["name"].isin(df["Player"]), ["name"] + PER_GAME_COLS].copy()
        stats[["fg_pct", "ft_pct"]] *= 100
        stats = stats.rename(columns={**COL_LABELS, "name": "Player"}).drop_duplicates("Player")
        df = df.merge(stats, on="Player", how="left").sort_values("Gain", ascending=False, na_position="last")
    d_cols = [f"Δ {lab}" for lab in CAT_COLS]
    fmt = {"Gain": "{:+.1f}", "Win % after": "{:.1f}%", "Value": "{:.1f}", "Per-game rank": "{:.0f}", "Proj GP": "{:.0f}", "Board rank": "{:.0f}", **{c: "{:+.0f}" for c in d_cols}}
    fmt.update({c: "{:.1f}" for c in ["MPG", "PPG", "3PM", "REB", "AST", "STL", "BLK", "TO", "FG%", "FT%"] if c in df})
    st.dataframe(shade_multi(df.reset_index(drop=True), [(["Gain"], -3, 3), (d_cols, -10, 10)]).format(fmt, na_rep="–"), hide_index=True, width="stretch", height=480)
    st.caption("Δ columns = the change, in points, in my chance to win each category in a weekly matchup if I make that swap. Value is the draft board's z-score "
               "total (higher is better), which discounts counting stats by projected games; per-game rank ignores availability.")

    # --- my players: who is adding the least ------------------------------------
    st.subheader("My roster: who is adding the least")
    cost = season.removal_cost(engine, my_roster, opp)
    usage = season.lineup_usage(engine, my_roster)
    gains = np.where(np.isnan(grid.win), -np.inf, grid.gain())
    rows = []
    for di, d in enumerate(grid.drops):
        if d is None:
            continue
        r = ctx.proj_by_id.loc[ctx.pid(d)]
        j = int(gains[:, di].argmax())
        rows.append({"Player": r["name"], "Pos": r["pos"], "Board rank": int(r["rank"]), "Value": r["z_total"], "Starts %": usage.get(d, 0) * 100,
                     "Win % lost without him": cost[d], "Best replacement": ctx.name_of(grid.adds[j]) if np.isfinite(gains[j, di]) else "–",
                     "Gain from replacing": gains[j, di] if np.isfinite(gains[j, di]) else np.nan})
    mine = pd.DataFrame(rows).sort_values("Win % lost without him")
    st.dataframe(mine.style.format({"Value": "{:.1f}", "Starts %": "{:.0f}%", "Win % lost without him": "{:.1f}", "Gain from replacing": "{:+.1f}"}, na_rep="–"),
                 hide_index=True, width="stretch")
    st.caption("Win % lost without him = how far my win probability falls if I play without him and add nobody; the lowest are the most expendable. "
               "Best replacement = the free agent (among those scored above) who helps most if he takes his spot.")

    # --- compare one free agent with my players ---------------------------------------
    st.subheader("Compare a free agent with my players")
    pick = st.selectbox("Free agent", [int(p) for p in pool["player_id"]], format_func=lambda p: _fa_label(ctx, p),
                        help="Any free agent, not only the ones scored above. Type to search.")
    one = season.swap_grid(engine, my_roster, opp, [engine.idx_of[pick]], my_roster + ([None] if open_slot else []), S.max_centers)
    fa = ctx.proj_by_id.loc[pick]
    g1 = one.gain()[0]
    rows = []
    for di, d in enumerate(one.drops):
        if np.isnan(one.win[0, di]):
            continue
        r = None if d is None else ctx.proj_by_id.loc[ctx.pid(d)]
        row = {"If I drop": "(nobody: open spot)" if d is None else r["name"], "Win % change": g1[di],
               "Board rank": "–" if d is None else f"{int(fa['rank'])} vs {int(r['rank'])}", "Value": fa["z_total"] - (0 if d is None else r["z_total"])}
        row.update({f"Δ {lab}": (one.cat[0, di, k] - one.base_cat[k]) * 100 for k, lab in enumerate(CAT_COLS)})
        rows.append(row)
    cmp_df = pd.DataFrame(rows).sort_values("Win % change", ascending=False)
    if cmp_df.empty:
        st.info("Adding him would put my team over the center limit whoever I drop.")
    else:
        st.dataframe(shade_multi(cmp_df.reset_index(drop=True), [(["Win % change"], -3, 3), (d_cols, -10, 10)]).format(
            {"Win % change": "{:+.1f}", "Value": "{:+.1f}", **{c: "{:+.0f}" for c in d_cols}}), hide_index=True, width="stretch")
        st.caption(f"Each row: add {fa['name']} and drop that player. Value = his z-score total minus the dropped player's.")
        names = [r for r in cmp_df["If I drop"] if not r.startswith("(")]
        if names:
            vs = st.selectbox("Side-by-side with", names, index=0, key="fa_vs")
            r = ctx.proj_by_id[ctx.proj_by_id["name"] == vs].iloc[0]
            stat_rows = {"Board rank": ("rank", "{:.0f}"), "Proj GP": ("gp82", "{:.0f}"), "MPG": ("mpg", "{:.1f}"), "PPG": ("pts", "{:.1f}"), "3PM": ("tpm", "{:.1f}"), "REB": ("reb", "{:.1f}"),
                         "AST": ("ast", "{:.1f}"), "STL": ("stl", "{:.1f}"), "BLK": ("blk", "{:.1f}"), "FG%": ("fg_pct", "{:.1f}"), "FT%": ("ft_pct", "{:.1f}"), "TO": ("tov", "{:.1f}")}
            a, b = st.columns(2)
            per = pd.DataFrame({"": list(stat_rows), fa["name"]: [fa[k] * (100 if k.endswith("pct") else 1) for k, _ in stat_rows.values()],
                                vs: [r[k] * (100 if k.endswith("pct") else 1) for k, _ in stat_rows.values()]})
            per["Difference"] = per[fa["name"]] - per[vs]
            a.markdown("**Per game**")
            a.dataframe(per.style.format({fa["name"]: "{:.1f}", vs: "{:.1f}", "Difference": "{:+.1f}"}), hide_index=True, width="stretch")
            z = pd.DataFrame({"": CAT_COLS, fa["name"]: [fa[f"z_{c}"] for c in CATEGORIES], vs: [r[f"z_{c}"] for c in CATEGORIES]})
            z["Difference"] = z[fa["name"]] - z[vs]
            b.markdown("**Category value (z-score; turnovers already flipped, so higher is better)**")
            b.dataframe(shade_multi(z, [(["Difference"], -2, 2)]).format({fa["name"]: "{:+.1f}", vs: "{:+.1f}", "Difference": "{:+.1f}"}), hide_index=True, width="stretch")
            st.caption("The z-scores measure a player's strength in each category against a drafted player, scaled by the games he is expected to play.")

    _transactions(ctx, pool)


def _transactions(ctx: Ctx, pool: pd.DataFrame) -> None:
    S, state = ctx.S, ctx.state
    st.divider()
    st.subheader("Transactions")
    st.caption("Record pickups, drops and injured-reserve moves for any team, so the free-agent pool and every projection stay current. "
               "A pickup and its drop go in together as one transaction.")
    team = st.selectbox("Team", list(range(1, S.n_teams + 1)), index=S.my_slot - 1, format_func=lambda t: ctx.tname(t, " (me)"), key="txn_team")
    active, ir = state.roster(team), state.ir(team)
    fa_ids = [int(p) for p in pool["player_id"]]
    none = "—"
    with st.form("txn_form", clear_on_submit=True):
        c1, c2 = st.columns(2)
        add = c1.selectbox("Add a free agent", [None] + fa_ids, format_func=lambda p: none if p is None else _fa_label(ctx, p))
        drop = c2.selectbox("Drop", [None] + active + ir, format_func=lambda p: none if p is None else ctx.player_name(p))
        c3, c4 = st.columns(2)
        to_ir = c3.selectbox("Move to injured reserve", [None] + active, format_func=lambda p: none if p is None else ctx.player_name(p))
        off_ir = c4.selectbox("Activate from injured reserve", [None] + ir, format_func=lambda p: none if p is None else ctx.player_name(p))
        note = st.text_input("Note (optional)")
        go = st.form_submit_button("Record transaction", type="primary")
    if go:
        moves = [(team, p, a) for p, a in ((drop, "drop"), (to_ir, "ir"), (off_ir, "activate"), (add, "add")) if p is not None]
        if not moves:
            st.warning("Choose at least one move.")
        else:
            try:
                store.record_transaction(ctx.conn, state, moves, note.strip() or "transaction")
            except ValueError as e:
                st.error(f"Not recorded: {e}.")
            else:
                size = len(store.load_state(ctx.conn, LEAGUE).roster(team))
                _flash(f"Recorded {len(moves)} move{'s' if len(moves) != 1 else ''} for {ctx.tname(team)}."
                       + (f" Their roster now has {size} players (the limit is {S.roster_size})." if size > S.roster_size else ""))
                st.rerun()

    log = store.moves_log(ctx.conn, LEAGUE)
    if log:
        rows = [{"#": r["move_no"], "Team": ctx.tname(r["team_slot"]), "Move": ACTIONS[r["action"]], "Player": ctx.player_name(r["player_id"]),
                 "Note": store.TXN_NOTE.sub("", r["note"] or "")} for r in log]
        st.dataframe(pd.DataFrame(rows[::-1]), hide_index=True, width="stretch")
        if st.button("↩ Undo the last transaction"):
            n = store.undo_move(ctx.conn, state)
            _flash(f"Took back {n} move{'s' if n != 1 else ''}.")
            st.rerun()
    else:
        st.caption("No roster moves recorded yet.")
