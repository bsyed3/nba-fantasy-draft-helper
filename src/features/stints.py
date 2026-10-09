"""
Detect player stint boundaries (team changes) from the ingested box score
+ game tables. Stints are detected, never hand-entered — see
docs/data_model.md "How stint/era boundaries get populated".
"""
from __future__ import annotations

import pandas as pd


def compute_player_stints(box_scores_with_dates: pd.DataFrame) -> pd.DataFrame:
    """
    box_scores_with_dates: needs at least [player_id, team_id, date]
    (typically box_score joined to game on game_id). Order doesn't matter
    going in; this sorts internally.

    Returns one row per (player_id, team_id, start_date, end_date) stint.
    A trade shows up as the previous stint's end_date landing on the last
    game before team_id changes, and a new stint opening at the next
    game's date. Vectorized: a stint boundary is wherever player_id or
    team_id differs from the previous row after sorting by (player, date).
    """
    cols = ["player_id", "team_id", "start_date", "end_date"]
    if box_scores_with_dates.empty:
        return pd.DataFrame(columns=cols)

    df = (
        box_scores_with_dates[["player_id", "team_id", "date"]]
        .drop_duplicates()
        .sort_values(["player_id", "date"])
        .reset_index(drop=True)
    )
    new_stint = (df["player_id"] != df["player_id"].shift()) | (df["team_id"] != df["team_id"].shift())
    df["stint_id"] = new_stint.cumsum()
    out = (
        df.groupby("stint_id", sort=True)
        .agg(player_id=("player_id", "first"), team_id=("team_id", "first"),
             start_date=("date", "min"), end_date=("date", "max"))
        .reset_index(drop=True)
    )
    return out[cols]
