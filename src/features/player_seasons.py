"""
Per-player-season table built from box_score, plus the ratio definitions
shared by the projection model and the simulation calibration.

Everything the models consume is a *ratio of two totals* (per-36 rate =
total / minutes * 36, FG-type % = makes / attempts, mpg = minutes / games,
availability = games / team games). Keeping totals and a (numerator,
denominator, scale) definition for each stat means shrinkage, pooling
across seasons, and weighting are all the same one-liner for every stat,
and percentages are always aggregated as sum(makes)/sum(attempts), never
as an average of percentages.

Two-point and three-point shooting are modelled separately (fga2/fgm2 vs
tpa/tpm) so that FGM = 2PM + 3PM can never be smaller than 3PM, and
FG% / PTS are derived rather than modelled.
"""
from __future__ import annotations

import sqlite3

import numpy as np
import pandas as pd

# stat name -> (numerator total, denominator total, scale)
STAT_DEFS: dict[str, tuple[str, str, float]] = {
    "fga2": ("fga2", "min", 36.0),
    "fga3": ("fga3", "min", 36.0),
    "fta": ("fta", "min", 36.0),
    "reb": ("reb", "min", 36.0),
    "ast": ("ast", "min", 36.0),
    "stl": ("stl", "min", 36.0),
    "blk": ("blk", "min", 36.0),
    "tov": ("tov", "min", 36.0),
    "p2": ("fgm2", "fga2", 1.0),
    "p3": ("tpm", "fga3", 1.0),
    "ft": ("ftm", "fta", 1.0),
    "mpg": ("min", "gp", 1.0),
    "avail": ("gp", "sg", 1.0),
}
RATE_STATS = ["fga2", "fga3", "fta", "reb", "ast", "stl", "blk", "tov"]  # per-36
PCT_STATS = ["p2", "p3", "ft"]
ALL_STATS = RATE_STATS + PCT_STATS + ["mpg", "avail"]
TOTAL_COLS = ["gp", "min", "sg", "fga2", "fga3", "fgm2", "tpm", "fta", "ftm",
              "reb", "ast", "stl", "blk", "tov"]


def season_start_year(season: str) -> int:
    return int(season[:4])


def pos_group(position_1: str | None) -> str:
    """G / F / C bucket used for shrinkage priors."""
    if not position_1:
        return "F"
    c = str(position_1)[0].upper()
    return c if c in "GFC" else "F"


def load_player_season_table(conn: sqlite3.Connection) -> pd.DataFrame:
    """One row per (player_id, year) of regular-season totals + bio fields."""
    totals = pd.read_sql_query(
        """
        SELECT g.season, bs.player_id,
               COUNT(*) AS gp, SUM(bs.min) AS min,
               SUM(bs.fga - bs.tpa) AS fga2, SUM(bs.tpa) AS fga3,
               SUM(bs.fgm - bs.tpm) AS fgm2, SUM(bs.tpm) AS tpm,
               SUM(bs.fta) AS fta, SUM(bs.ftm) AS ftm,
               SUM(bs.oreb + bs.dreb) AS reb, SUM(bs.ast) AS ast,
               SUM(bs.stl) AS stl, SUM(bs.blk) AS blk, SUM(bs.tov) AS tov
        FROM box_score bs JOIN game g ON g.game_id = bs.game_id
        WHERE g.game_type = 'regular' AND bs.min > 0
        GROUP BY g.season, bs.player_id
        """,
        conn,
    )
    team_games = pd.read_sql_query(
        """
        SELECT season, MAX(n) AS sg FROM (
            SELECT season, team_id, COUNT(*) AS n FROM (
                SELECT season, home_team_id AS team_id FROM game WHERE game_type = 'regular'
                UNION ALL
                SELECT season, away_team_id FROM game WHERE game_type = 'regular'
            ) GROUP BY season, team_id
        ) GROUP BY season
        """,
        conn,
    )
    players = pd.read_sql_query(
        "SELECT player_id, name, position_1, draft_year, draft_position, rookie_year, last_year FROM player",
        conn,
    )
    ages = pd.read_sql_query("SELECT player_id, season, age FROM player_season", conn)

    st = (
        totals.merge(team_games, on="season")
        .merge(players, on="player_id", how="left")
        .merge(ages, on=["player_id", "season"], how="left")
    )
    st["year"] = st["season"].map(season_start_year)
    st["pos"] = st["position_1"].map(pos_group)
    st["draft_pick"] = st["draft_position"].astype(float).fillna(61.0)
    return st.sort_values(["year", "player_id"]).reset_index(drop=True)


def league_priors(st: pd.DataFrame) -> pd.DataFrame:
    """
    League-average ratio for every stat by (year, pos group), weighted by
    denominator exposure. Used as the shrinkage target. Indexed by
    (year, pos); one column per stat.
    """
    rows = []
    for (year, pos), grp in st.groupby(["year", "pos"]):
        row = {"year": year, "pos": pos}
        for stat, (num, den, scale) in STAT_DEFS.items():
            d = grp[den].sum()
            row[stat] = grp[num].sum() / d * scale if d > 0 else np.nan
        rows.append(row)
    return pd.DataFrame(rows).set_index(["year", "pos"])
