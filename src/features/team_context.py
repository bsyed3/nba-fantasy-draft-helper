"""
Team context for the projection model: what happened to a player's team since last season.

For a player P on team X in season T the model is told how many minutes (and possessions used) were VACATED
(players who played for X in T-1 and are not on X's opening roster in T) and how many are INCOMING (players on X's
opening roster in T who did not play for X in T-1, at what they did elsewhere in T-1) -- overall and among players at
P's own position group (G / F / C) -- plus whether P himself moved. P's own contribution is always removed, so the
numbers describe his teammates.

"Opening roster" = players who played for the team in one of its first K games of the season. Historically that is
what the box scores show; for the season being projected it is the current roster (PlayerIndex) minus anyone already
known to be out for the opening stretch (data/external/known_absences*.csv), whose minutes therefore count as vacated.
Rookies / players with no previous-season minutes bring no incoming minutes (we have no prior), which is the same in
training and in the projection.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

from src.features.names import normalize_name
from src.features.player_seasons import pos_group

K_OPENING = 10                      # a player is on the opening roster if he played for the team in one of its first K games
ROLE_GP_FULL = 40                   # role size = per-game minutes while playing, tapered below this many games (an injured star still leaves a big hole)
POSS_FT = 0.44                      # possessions used = FGA + 0.44*FTA + TOV
CTX_COLS = ["ctx_moved", "ctx_vac_min", "ctx_inc_min", "ctx_vac_poss", "ctx_inc_poss",
            "ctx_vac_min_pos", "ctx_inc_min_pos", "ctx_vac_poss_pos", "ctx_inc_poss_pos"]


def _box_by_team(conn: sqlite3.Connection) -> pd.DataFrame:
    df = pd.read_sql_query(
        """SELECT g.season, g.date, bs.team_id, bs.player_id, bs.min, bs.fga, bs.fta, bs.tov
           FROM box_score bs JOIN game g ON g.game_id = bs.game_id
           WHERE g.game_type = 'regular' AND bs.min > 0""", conn)
    df["year"] = df["season"].str[:4].astype(int)
    df["poss"] = df["fga"].fillna(0) + POSS_FT * df["fta"].fillna(0) + df["tov"].fillna(0)
    return df


def known_absent_ids(paths: list, players: pd.DataFrame, min_games: int = K_OPENING) -> set[int]:
    """Player ids listed in the known-absence CSVs as missing at least `min_games` games at the start (so off the opening roster)."""
    by_name = {}
    for pid, nm in zip(players["player_id"], players["name"]):
        by_name.setdefault(normalize_name(nm), []).append(int(pid))
    games_out: dict[int, float] = {}
    for p in paths:                                       # later files win, as in projection.apply_known_absences
        if p is None or not Path(p).exists():
            continue
        ka = pd.read_csv(p)
        for _, r in ka.iterrows():
            if "player_id" in ka.columns and pd.notna(r["player_id"]):
                ids = [int(r["player_id"])]
            else:
                ids = by_name.get(normalize_name(r["name"]), [])
            if len(ids) == 1:
                games_out[ids[0]] = float(r["games_out"])
    return {pid for pid, g in games_out.items() if g >= min_games}


def team_context(conn: sqlite3.Connection, years: list[int], current_team: pd.Series | None = None,
                 absent_ids: set[int] | None = None, k: int = K_OPENING, role_based: bool = True) -> pd.DataFrame:
    """
    One row per (player_id, year) with the CTX_COLS. Years present in the box scores use them; any other year (the one being
    projected) uses `current_team` (player_id -> team_id, current roster) minus `absent_ids` for the opening roster.
    `role_based` measures what a departed / arriving player's role was when he played (per-game minutes and possessions, tapered
    for under ROLE_GP_FULL games) rather than minutes per TEAM game, which would understate the hole left by someone who was hurt all year.
    """
    box = _box_by_team(conn)
    players = pd.read_sql_query("SELECT player_id, position_1 FROM player", conn)
    pos = dict(zip(players["player_id"], players["position_1"].map(pos_group)))
    absent_ids = absent_ids or set()
    last_box_year = int(box["year"].max())

    # per (year, team, player) totals
    pt = box.groupby(["year", "team_id", "player_id"]).agg(min=("min", "sum"), poss=("poss", "sum"), gp=("min", "size")).reset_index()
    tg = box.groupby(["year", "team_id"])["date"].nunique().rename("tg").reset_index()
    sgy = tg.groupby("year")["tg"].max().to_dict()                        # season length (82, or 72 / 75 in the COVID years)
    tot = pt.groupby(["year", "player_id"]).agg(tmin=("min", "sum"), tposs=("poss", "sum"), tgp=("gp", "sum")).reset_index()

    # opening rosters and each player's team at the start of the season (historical years)
    d = box[["year", "team_id", "date"]].drop_duplicates().sort_values("date")
    d["n"] = d.groupby(["year", "team_id"]).cumcount() + 1
    ros = box.merge(d[d["n"] <= k][["year", "team_id", "date"]], on=["year", "team_id", "date"])[["year", "team_id", "player_id"]].drop_duplicates()
    first = box.sort_values("date").drop_duplicates(["year", "player_id"], keep="first")[["year", "player_id", "team_id"]]

    rows = []
    for T in years:
        prev_year = T - 1
        if T <= last_box_year:
            ros_T = ros[ros["year"] == T][["team_id", "player_id"]]
            team_of = first[first["year"] == T].set_index("player_id")["team_id"]
        else:
            if current_team is None:
                continue
            cur = current_team.dropna()
            cur = cur[cur > 0]
            ros_T = pd.DataFrame({"team_id": cur.to_numpy(), "player_id": cur.index.to_numpy()})
            ros_T = ros_T[~ros_T["player_id"].isin(absent_ids)]
            team_of = cur
        prev = pt[pt["year"] == prev_year][["team_id", "player_id", "min", "poss", "gp"]]
        if prev.empty:
            continue
        tg_prev = tg[tg["year"] == prev_year].set_index("team_id")["tg"]
        prev = prev.assign(g=prev["team_id"].map(tg_prev))
        if role_based:
            f = (prev["gp"] / ROLE_GP_FULL).clip(upper=1.0)
            prev["vmin"] = prev["min"] / prev["gp"] * f; prev["vposs"] = prev["poss"] / prev["gp"] * f
        else:
            prev["vmin"] = prev["min"] / prev["g"]; prev["vposs"] = prev["poss"] / prev["g"]
        ptot = tot[tot["year"] == prev_year].set_index("player_id")
        sg_prev = sgy.get(prev_year, 82)
        roster_pairs = set(zip(ros_T["team_id"], ros_T["player_id"]))
        prev_pairs = set(zip(prev["team_id"], prev["player_id"]))

        # departed: played for X last year, not on X's opening roster now
        dep = prev[[(t, p) not in roster_pairs for t, p in zip(prev["team_id"], prev["player_id"])]].copy()
        dep["pos"] = dep["player_id"].map(pos).fillna("F")
        # arrivals: on X's opening roster now, did not play for X last year; credited with last year's per-game minutes elsewhere
        arr = ros_T[[(t, p) not in prev_pairs for t, p in zip(ros_T["team_id"], ros_T["player_id"])]].copy()
        arr = arr[arr["player_id"].isin(ptot.index)]
        if role_based:
            tgp = arr["player_id"].map(ptot["tgp"])
            f = (tgp / ROLE_GP_FULL).clip(upper=1.0)
            arr["imin"] = arr["player_id"].map(ptot["tmin"]) / tgp * f
            arr["iposs"] = arr["player_id"].map(ptot["tposs"]) / tgp * f
        else:
            arr["imin"] = arr["player_id"].map(ptot["tmin"]) / sg_prev
            arr["iposs"] = arr["player_id"].map(ptot["tposs"]) / sg_prev
        arr["pos"] = arr["player_id"].map(pos).fillna("F")

        V = dep.groupby("team_id")[["vmin", "vposs"]].sum()
        Vg = dep.groupby(["team_id", "pos"])[["vmin", "vposs"]].sum()
        I = arr.groupby("team_id")[["imin", "iposs"]].sum()
        Ig = arr.groupby(["team_id", "pos"])[["imin", "iposs"]].sum()
        dep_own = dep.set_index(["team_id", "player_id"])[["vmin", "vposs"]]
        arr_own = arr.set_index(["team_id", "player_id"])[["imin", "iposs"]]
        had_prev_any = set(ptot.index)

        for pid, tid in team_of.items():
            g = pos.get(pid, "F")
            v = V.loc[tid] if tid in V.index else pd.Series({"vmin": 0.0, "vposs": 0.0})
            i = I.loc[tid] if tid in I.index else pd.Series({"imin": 0.0, "iposs": 0.0})
            vg = Vg.loc[(tid, g)] if (tid, g) in Vg.index else pd.Series({"vmin": 0.0, "vposs": 0.0})
            ig = Ig.loc[(tid, g)] if (tid, g) in Ig.index else pd.Series({"imin": 0.0, "iposs": 0.0})
            own_v = dep_own.loc[(tid, pid)] if (tid, pid) in dep_own.index else None
            own_i = arr_own.loc[(tid, pid)] if (tid, pid) in arr_own.index else None
            vmin, vposs, vminp, vpossp = v["vmin"], v["vposs"], vg["vmin"], vg["vposs"]
            imin, iposs, iminp, ipossp = i["imin"], i["iposs"], ig["imin"], ig["iposs"]
            if own_v is not None:
                vmin -= own_v["vmin"]; vposs -= own_v["vposs"]; vminp -= own_v["vmin"]; vpossp -= own_v["vposs"]
            if own_i is not None:
                imin -= own_i["imin"]; iposs -= own_i["iposs"]; iminp -= own_i["imin"]; ipossp -= own_i["iposs"]
            if (tid, pid) in prev_pairs:
                moved = 0.0
            elif pid in had_prev_any:
                moved = 1.0
            else:
                moved = np.nan
            rows.append((pid, T, moved, vmin, imin, vposs, iposs, vminp, iminp, vpossp, ipossp))
    return pd.DataFrame(rows, columns=["player_id", "year"] + CTX_COLS)
