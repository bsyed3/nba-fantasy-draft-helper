"""
The Teams page: every team's projected weekly stats, category ranks and win probability, from the simulated weeks.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from src.app.common import CAT_COLS, PER_GAME_COLS, Ctx, shade
from src.draft import season
from src.draft.values import CAT_LABELS, CATEGORIES

STAT_FORMAT = {"PTS": "{:.0f}", "3PM": "{:.1f}", "REB": "{:.1f}", "AST": "{:.1f}", "STL": "{:.1f}", "BLK": "{:.1f}", "FG%": "{:.1f}%", "FT%": "{:.1f}%", "TO": "{:.1f}"}


def _totals_frame(tbl: season.LeagueTable) -> pd.DataFrame:
    """[T, 9] projected weekly totals, FG%/FT% shown as percentages."""
    t = tbl.totals.copy()
    t[:, season.RATIO] *= 100
    return pd.DataFrame(t, columns=CAT_COLS)


def render(ctx: Ctx) -> None:
    S, state = ctx.S, ctx.state
    st.title("Teams")
    rosters = ctx.rosters_idx()
    tbl = ctx.league_table(rosters)
    T = S.n_teams
    label = [ctx.tname(t, " (me)") for t in range(1, T + 1)]
    ranks = tbl.power_rank
    n_moves = len(state.moves)
    st.caption(f"Each team's active roster ({n_moves} roster move{'s' if n_moves != 1 else ''} since the draft applied, injured reserve not counted) played through "
               f"{ctx.lib.n_sims} simulated weeks, {S.reg_season_weeks} regular-season weeks and {S.playoff_weeks} playoff weeks, with daily lineups. "
               "Win % is the share of those weeks a team beats the other nine in the same simulated week (more than 4.5 of 9 categories).")

    # --- power rankings -------------------------------------------------------
    order = np.argsort(ranks)
    rows = []
    for t in order:
        cw = tbl.cat_win[t]
        best, worst = np.argsort(-cw)[:3], np.argsort(cw)[:3]
        rows.append({"Rank": int(ranks[t]), "Team": label[t], "Win %": tbl.win[t] * 100, "Regular season %": tbl.win_reg[t] * 100, "Playoff weeks %": tbl.win_po[t] * 100,
                     f"Exp. wins / {S.reg_season_weeks}": tbl.win_reg[t] * S.reg_season_weeks,
                     "Top-3 cats": int((tbl.cat_rank[t] <= 3).sum()),
                     "Strongest": ", ".join(CAT_LABELS[CATEGORIES[k]] for k in best), "Weakest": ", ".join(CAT_LABELS[CATEGORIES[k]] for k in worst)})
    power = pd.DataFrame(rows)
    st.subheader("Power rankings")
    st.dataframe(power.style.apply(lambda r: ["font-weight: 700" if r["Team"].endswith("(me)") else "" for _ in r], axis=1),
                 hide_index=True, width="stretch",
                 column_config={"Win %": st.column_config.ProgressColumn("Win %", min_value=30, max_value=70, format="%.1f%%"),
                                "Regular season %": st.column_config.NumberColumn(format="%.1f%%"), "Playoff weeks %": st.column_config.NumberColumn(format="%.1f%%"),
                                f"Exp. wins / {S.reg_season_weeks}": st.column_config.NumberColumn(format="%.1f"),
                                "Top-3 cats": st.column_config.NumberColumn(help="Categories in which this team's projected weekly total ranks in the top 3 of the league.")})
    st.caption("Playoff weeks use the real weeks 19-21 schedule, so a team with more games there is rated higher. Strongest / weakest = the categories this team is "
               "most / least likely to win in a weekly matchup.")

    # --- category view --------------------------------------------------------
    st.subheader("Category strength")
    view = st.radio("Show", ["Rank in the league (1 = best)", "Projected weekly total", "Chance to win the category"], horizontal=True, label_visibility="collapsed")
    totals, cat_rank = _totals_frame(tbl), pd.DataFrame(tbl.cat_rank, columns=CAT_COLS)
    cat_win = pd.DataFrame(tbl.cat_win * 100, columns=CAT_COLS)
    base = {"Rank in the league (1 = best)": cat_rank, "Projected weekly total": totals, "Chance to win the category": cat_win}[view].copy()
    base.insert(0, "Team", label)
    base = base.iloc[order].reset_index(drop=True)
    if view.startswith("Rank"):
        base["Average rank"] = base[CAT_COLS].mean(axis=1)
        styled = shade(base, CAT_COLS, 1, T, reverse=True).format({**{c: "{:.0f}" for c in CAT_COLS}, "Average rank": "{:.1f}"})
        note = "Rank of each team's projected weekly total among the ten teams; turnovers rank fewest first."
    elif view.startswith("Projected"):
        rk = cat_rank.iloc[order].reset_index(drop=True)
        styled = shade(base, CAT_COLS, 1, T, reverse=True, basis=rk).format(STAT_FORMAT)
        note = ("Average per 7 days over the simulated matchup periods (the two-week opening and All-Star periods are scaled to a week). "
                "Cells are coloured by the team's rank in that category.")
    else:
        styled = shade(base, CAT_COLS, 25, 75).format({c: "{:.0f}%" for c in CAT_COLS})
        note = "Share of simulated weeks in which the team beats a random other team in that category (50% = league average)."
    st.dataframe(styled, hide_index=True, width="stretch")
    st.caption(note)

    # --- head to head ---------------------------------------------------------
    with st.expander("Head-to-head matchup matrix"):
        h = pd.DataFrame(tbl.h2h * 100, index=label, columns=label).iloc[order, order]
        h.insert(0, "Team", h.index)
        st.dataframe(shade(h.reset_index(drop=True), list(h.columns[1:]), 30, 70).format({c: "{:.0f}%" for c in h.columns[1:]}, na_rep="–"),
                     hide_index=True, width="stretch")
        st.caption("Row team's chance to win a weekly matchup against the column team, over the simulated weeks.")

    # --- one team ---------------------------------------------------------------
    st.subheader("Team detail")
    slot = st.selectbox("Team", list(range(1, T + 1)), index=S.my_slot - 1, format_func=lambda t: f"{ranks[t - 1]}. {label[t - 1]}")
    t0 = slot - 1
    m = st.columns(5)
    m[0].metric("Power rank", f"{ranks[t0]} of {T}")
    m[1].metric("Win %", f"{tbl.win[t0] * 100:.1f}%")
    m[2].metric("Regular season", f"{tbl.win_reg[t0] * 100:.1f}%")
    m[3].metric("Playoff weeks", f"{tbl.win_po[t0] * 100:.1f}%")
    m[4].metric(f"Exp. wins / {S.reg_season_weeks}", f"{tbl.win_reg[t0] * S.reg_season_weeks:.1f}")

    left, right = st.columns([1.7, 1])
    with left:
        ids = state.roster(slot)
        usage = season.lineup_usage(ctx.engine, rosters[slot])
        rows = []
        for pid in ids:
            if pid not in ctx.proj_by_id.index:
                rows.append({"Player": ctx.player_name(pid)})
                continue
            r = ctx.proj_by_id.loc[pid]
            row = {"Board rank": int(r["rank"]), "Player": r["name"], "Pos": r["pos"], "Team": r["team"], "Proj GP": r["gp82"], "Starts %": usage.get(ctx.engine.idx_of[int(pid)], 0) * 100}
            row.update({c: (r[k] * 100 if k in ("fg_pct", "ft_pct") else r[k]) for k, c in zip(PER_GAME_COLS, ["MPG"] + CAT_COLS)})
            rows.append(row)
        roster = pd.DataFrame(rows).sort_values("Board rank")
        n_c = state.n_centers(slot)
        st.markdown(f"**{label[t0]} roster** ({len(ids)}/{S.roster_size}) · centers {n_c}/{S.max_centers}")
        st.dataframe(roster.round(1), hide_index=True, width="stretch",
                     column_config={"Starts %": st.column_config.NumberColumn(format="%.0f%%", help="Share of the days he has a game that he fills a lineup slot (otherwise he sits on the bench).")})
        if state.ir(slot):
            st.caption("Injured reserve (not counted): " + ", ".join(ctx.player_name(i) for i in state.ir(slot)))
        st.caption("Per-game stats are for a game he plays; Proj GP is expected games out of 82. Board rank is the draft board's value rank.")
    with right:
        league_avg = totals.mean()
        detail = pd.DataFrame({"Category": CAT_COLS, "Weekly total": totals.iloc[t0].to_numpy(), "League avg": league_avg.to_numpy(),
                               "Rank": tbl.cat_rank[t0], "Win %": tbl.cat_win[t0] * 100})
        st.markdown("**Projected weekly stats**")
        st.dataframe(shade(detail, ["Rank"], 1, T, reverse=True).format({"Weekly total": "{:.1f}", "League avg": "{:.1f}", "Rank": "{:.0f}", "Win %": "{:.0f}%"}),
                     hide_index=True, width="stretch")
        st.bar_chart(pd.DataFrame({"Win %": tbl.cat_win[t0] * 100}, index=CAT_COLS), height=200, sort=False)

    weekly = pd.DataFrame({"Win %": tbl.by_week[t0] * 100}, index=[f"W{w + 1}" + ("*" if w >= S.reg_season_weeks else "") for w in range(tbl.by_week.shape[1])])
    st.markdown("**Win % by matchup week** (* = playoff weeks)")
    st.bar_chart(weekly, height=220, sort=False)
    st.caption("Weeks differ because of the schedule: how many games each team's players have, and the two-week opening and All-Star periods.")
