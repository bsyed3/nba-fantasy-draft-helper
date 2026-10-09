"""
Live draft assistant (Streamlit).

    streamlit run src/app/streamlit_app.py

Enter every pick as it happens (all teams, not just yours). On your turn
-- or any time via "Plan my next pick" -- the engine ranks the available
players by the simulated probability that adding them gets your team a
weekly H2H matchup win, given everything drafted so far.
"""
from __future__ import annotations

import os
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.draft import store
from src.draft.engine import DraftEngine
from src.draft.state import DraftState, LeagueSettings
from src.draft.values import CAT_LABELS, CATEGORIES
from src.simulation.copula import Library

DATA = ROOT / "data" / "processed"
DB_PATH = Path(os.environ.get("NBA_DB", DATA / "nba.db"))  # override to keep test drafts out of the real DB
PER_GAME_COLS = ["mpg", "pts", "tpm", "reb", "ast", "stl", "blk", "fg_pct", "ft_pct", "tov"]

st.set_page_config(page_title="NBA Fantasy Draft Helper", page_icon="🏀", layout="wide")


# ---------------------------------------------------------------------------
# cached resources
# ---------------------------------------------------------------------------
@st.cache_resource
def get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    store.ensure_schema(conn)
    return conn


@st.cache_resource
def load_data():
    years = sorted(int(p.stem.split("_")[1]) for p in DATA.glob("library_*.npz"))
    if not years:
        return None
    year = years[-1]
    proj = pd.read_parquet(DATA / f"projections_{year}.parquet")
    lib = Library.load(DATA / f"library_{year}.npz")
    from nba_api.stats.static import teams as static_teams
    abbr = {t["id"]: t["abbreviation"] for t in static_teams.get_teams()}
    proj["team"] = proj["team_id"].map(abbr).fillna("FA")
    proj["pos"] = proj["elig"]
    proj = proj[proj["player_id"].isin(lib.player_ids)].reset_index(drop=True)
    return proj, lib, year


@st.cache_resource
def get_engine(_proj, _lib, settings: LeagueSettings) -> DraftEngine:
    return DraftEngine(_lib, _proj, settings)


@st.cache_data(show_spinner="Simulating the rest of the draft...")
def cached_recommend(_engine: DraftEngine, settings: LeagueSettings, picks: tuple, excluded: tuple,
                     top_k: int, rollouts: int) -> pd.DataFrame:
    state = DraftState(settings, list(picks), set(excluded), set(_engine.center_ids))
    return _engine.recommend(state, top_k=top_k, rollouts=rollouts)


@st.cache_data(show_spinner="Projecting every team...")
def cached_outlook(_engine: DraftEngine, settings: LeagueSettings, picks: tuple, excluded: tuple):
    state = DraftState(settings, list(picks), set(excluded), set(_engine.center_ids))
    ev, rosters = _engine.projected_outcome(state)
    return ev, rosters


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def label(row) -> str:
    return f"{int(row['rank'])}. {row['name']} ({row['pos']}, {row['team']})"


def roster_frame(proj_by_id: pd.DataFrame, ids: list[int]) -> pd.DataFrame:
    if not ids:
        return pd.DataFrame(columns=["name", "pos", "team"] + PER_GAME_COLS)
    return proj_by_id.loc[ids, ["name", "pos", "team"] + PER_GAME_COLS].round(2)


def cat_columns() -> dict:
    return {f"cat_{c}": st.column_config.ProgressColumn(CAT_LABELS[c], min_value=0, max_value=100, format="%.0f%%") for c in CATEGORIES}


# ---------------------------------------------------------------------------
# page
# ---------------------------------------------------------------------------
data = load_data()
if data is None:
    st.error("No simulation library found. Run `python scripts/ingest.py`, `python scripts/build_projections.py` "
             "and `python scripts/build_library.py` first.")
    st.stop()
proj, lib, year = data
proj_by_id = proj.set_index("player_id")
conn = get_conn()

st.sidebar.title("🏀 Draft Helper")
st.sidebar.caption(f"Projections for the {year}-{str(year + 1)[-2:]} season · {len(proj)} players")
if (proj["elig_source"] != "espn").any():
    st.sidebar.warning("Position eligibility is **approximated** from nba_api labels + stats. For ESPN's exact values run "
                       "`python scripts/fetch_espn_positions.py` (needs ESPN access), then rebuild projections + library.")

leagues = store.list_leagues(conn)
choice = st.sidebar.selectbox("League", leagues + ["➕ New league"], index=0 if leagues else 0)
if choice == "➕ New league":
    with st.sidebar.form("new_league"):
        name = st.text_input("Name", "ESPN 10-team")
        n_teams = st.number_input("Teams", 4, 20, 10)
        roster = st.number_input("Roster size", 5, 25, 13)
        active = st.number_input("Active (counting) slots", 1, 20, 10)
        slot = st.number_input("My draft slot (round 1)", 1, 20, 1)
        max_c = st.number_input("Max centers per roster", 1, 13, 3)
        if st.form_submit_button("Create league"):
            if active > roster or slot > n_teams:
                st.sidebar.error("Active slots must be <= roster size and slot <= teams.")
            else:
                store.save_league(conn, LeagueSettings(name=name, n_teams=int(n_teams), roster_size=int(roster),
                                                       active_slots=int(active), my_slot=int(slot), max_centers=int(max_c), season=f"{year}-{str(year + 1)[-2:]}"))
                st.rerun()
    st.info("Create a league in the sidebar to start drafting.")
    st.stop()

state = store.load_state(conn, choice)
S = state.settings
engine = get_engine(proj, lib, S)
state.centers = set(engine.center_ids)  # C-eligible players count toward the roster cap

with st.sidebar.expander("League settings", expanded=False):
    new_slot = st.number_input("My draft slot", 1, S.n_teams, S.my_slot, key="slot_edit")
    if new_slot != S.my_slot and st.button("Update slot"):
        store.save_league(conn, LeagueSettings(**{**S.__dict__, "my_slot": int(new_slot)}))
        st.rerun()
    st.write(f"{S.n_teams} teams · {S.roster_size} drafted · {S.active_slots} active (PG SG G SF PF F C + 3 UTIL, daily) · "
             f"max {S.max_centers} centers · snake")
    st.write(f"{S.reg_season_weeks} regular-season weeks + {S.playoff_weeks} playoff weeks")

top_k = st.sidebar.slider("Candidates to evaluate", 10, 60, 30, 5)
rollouts = st.sidebar.slider("Mock-draft rollouts per candidate", 1, 6, 2)

c1, c2 = st.sidebar.columns(2)
if c1.button("↩ Undo last pick", disabled=not state.picks, width="stretch"):
    store.undo_pick(conn, state)
    st.rerun()
with st.sidebar.expander("Danger zone"):
    if st.checkbox("I want to reset this draft") and st.button("Reset draft", type="primary"):
        store.reset_draft(conn, state)
        st.rerun()

# --- header ---------------------------------------------------------------
if state.done:
    st.success("Draft complete.")
else:
    rnd = (state.next_pick_no - 1) // S.n_teams + 1
    h = st.columns(4)
    h[0].metric("Pick", f"#{state.next_pick_no} of {S.total_picks}")
    h[1].metric("Round", rnd)
    h[2].metric("On the clock", f"Team {state.team_on_clock}" + (" (YOU)" if state.my_turn else ""))
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
            legal = avail[avail["player_id"].map(state.can_draft)]
            options = {int(r["player_id"]): label(r) for _, r in legal.head(400).iterrows()}
            pid = st.selectbox(f"Team {state.team_on_clock} selects", list(options), format_func=options.get,
                               key=f"pick_{state.next_pick_no}")
            if st.button(f"Record pick #{state.next_pick_no}", type="primary", width="stretch"):
                store.record_pick(conn, state, pid)
                st.rerun()
        st.divider()
        st.caption("Recent picks")
        recent = pd.DataFrame(
            [{"#": p[0], "Team": p[1], "Player": proj_by_id.loc[p[2], "name"] if p[2] in proj_by_id.index else p[2]}
             for p in state.picks[-8:][::-1]])
        st.dataframe(recent, hide_index=True, width="stretch")

    with right:
        st.subheader("Recommendations")
        if state.done or state.my_next_pick_no() is None:
            st.write("Your roster is complete.")
        else:
            plan = state.my_turn or st.toggle("Plan my next pick (simulate bots up to it)", value=False)
            if plan:
                rec = cached_recommend(engine, S, tuple(state.picks), tuple(sorted(state.excluded)), top_k, rollouts)
                if rec.empty:
                    st.write("Nothing to evaluate.")
                else:
                    best = rec.iloc[0]
                    st.markdown(f"**Best pick: {best['name']}** — {best['win_prob']:.1%} chance to win a weekly matchup "
                                f"(≈ {best['exp_wins']:.1f} wins over {S.reg_season_weeks} weeks). "
                                f"Static rank #{int(best['static_rank'])}.")
                    show = rec[["name", "elig", "static_rank", "win_prob", "vs_top_ranked"]
                               + [f"cat_{c}" for c in CATEGORIES]].copy()
                    pct_cols = ["win_prob", "vs_top_ranked"] + [f"cat_{c}" for c in CATEGORIES]
                    show[pct_cols] = show[pct_cols] * 100
                    show = show.rename(
                        columns={"name": "Player", "elig": "Pos", "static_rank": "Rank",
                                 "win_prob": "Win %", "vs_top_ranked": "vs top-ranked"})
                    st.dataframe(
                        show, hide_index=True, width="stretch", height=620,
                        column_config={"Win %": st.column_config.NumberColumn(format="%.1f%%"),
                                       "vs top-ranked": st.column_config.NumberColumn(format="%+.1f%%"),
                                       **cat_columns()},
                    )
                    st.caption("Win % = chance my team beats a random opponent in a weekly matchup if I take this player and "
                               "the rest of the draft plays out like the mock-draft model. Category columns = chance to win each "
                               "category. Compare candidates by difference, not absolute level.")
            else:
                st.info("Not your turn yet. Enter picks as they happen, or flip the toggle to plan ahead.")

# --- my team --------------------------------------------------------------
with tab_team:
    mine = state.roster(S.my_slot)
    st.subheader(f"My roster ({len(mine)}/{S.roster_size}) · centers {state.n_centers(S.my_slot)}/{S.max_centers}")
    st.dataframe(roster_frame(proj_by_id, mine), width="stretch")
    if mine and not state.done:
        ev, _ = cached_outlook(engine, S, tuple(state.picks), tuple(sorted(state.excluded)))
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
    ev, rosters = cached_outlook(engine, S, tuple(state.picks), tuple(sorted(state.excluded)))
    rows = []
    for t in range(1, S.n_teams + 1):
        drafted_now = state.roster(t)
        rows.append({"Team": f"{t}" + (" (me)" if t == S.my_slot else ""), "Picks": len(drafted_now),
                     "Projected win %": ev.win_prob[t - 1] * 100,
                     "Exp. wins": ev.win_prob[t - 1] * S.reg_season_weeks,
                     "Drafted so far": ", ".join(proj_by_id.loc[drafted_now, "name"]) if drafted_now else ""})
    st.dataframe(pd.DataFrame(rows).sort_values("Projected win %", ascending=False), hide_index=True, width="stretch",
                 column_config={"Projected win %": st.column_config.NumberColumn(format="%.1f%%"),
                                "Exp. wins": st.column_config.NumberColumn(format="%.1f")})
    st.caption("Projected win % = each team's chance to win a weekly matchup vs a random opponent once the draft is "
               "completed by the mock-draft model.")

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
        b = b[b["name"].str.contains(q, case=False, na=False)]
    if pos:
        b = b[b["position_1"].isin(pos)]
    cols = ["rank", "name", "pos", "team", "age_t", "espn_adp"] + PER_GAME_COLS + ["avail", "z_total"]
    st.dataframe(b[cols].round(2), hide_index=True, width="stretch", height=560)
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
