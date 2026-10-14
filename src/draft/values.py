"""
Static (state-independent) player value: 9-category z-scores.

Used for (a) the baseline ranking / ADP-like ordering that the draft
rollouts use for opponents' picks and (b) lineup priority inside the
weekly simulation. The live recommendation itself does NOT rank by this
-- it ranks by simulated probability of winning the weekly matchup
(src/draft/engine.py).

Percentages use the impact formulation: a player's FG% matters in
proportion to his volume, so the stat that gets z-scored is
(FGM - pool_FG% * FGA), not FG% itself. All counting stats are scaled by
availability so an injury-prone player is credited with the games he is
expected to play.
"""
from __future__ import annotations

import pandas as pd

CATEGORIES = ["pts", "tpm", "reb", "ast", "stl", "blk", "fg_pct", "ft_pct", "tov"]
CAT_LABELS = {"pts": "PTS", "tpm": "3PM", "reb": "REB", "ast": "AST", "stl": "STL",
              "blk": "BLK", "fg_pct": "FG%", "ft_pct": "FT%", "tov": "TO"}
LOWER_IS_BETTER = {"tov"}


def _inputs(df: pd.DataFrame, fg_pool: float, ft_pool: float) -> pd.DataFrame:
    a = df["avail"]
    return pd.DataFrame({
        "pts": df["pts"] * a, "tpm": df["tpm"] * a, "reb": df["reb"] * a,
        "ast": df["ast"] * a, "stl": df["stl"] * a, "blk": df["blk"] * a,
        "fg_pct": (df["fgm"] - fg_pool * df["fga"]) * a,
        "ft_pct": (df["ftm"] - ft_pool * df["fta"]) * a,
        "tov": df["tov"] * a,
    })


Z_FLOOR = -3.0


def z_scores(df: pd.DataFrame, n_pool: int = 156, iters: int = 4, floor: float | None = Z_FLOOR) -> pd.DataFrame:
    """
    Returns a frame indexed like `df` with z_<cat> columns (raw), z_total and z_floor_adj.
    The mean/std reference pool is the top `n_pool` players by z_total,
    found iteratively (so the benchmark is "a drafted player", not the
    whole league including end-of-bench filler).

    `floor`: when the nine z-scores are summed, no single category may subtract more than |floor| (default -3). In weekly head-to-head
    a category is won or lost, so being 6 standard deviations bad at one (Giannis at free throws) costs the same one category as being
    3 bad: a plain sum would bury such a player well below where the simulation, which counts category wins, values him. Only the
    negative tail is floored; huge strengths still count in full. `z_floor_adj` is what the floor added back (0 for almost everyone);
    pass floor=None for the plain sum.
    """
    d = df.copy()
    pool = d.sort_values("mpg", ascending=False).head(n_pool).index
    for _ in range(iters):
        p = d.loc[pool]
        fg_pool = p["fgm"].mul(p["avail"]).sum() / p["fga"].mul(p["avail"]).sum()
        ft_pool = p["ftm"].mul(p["avail"]).sum() / p["fta"].mul(p["avail"]).sum()
        x = _inputs(d, fg_pool, ft_pool)
        mu, sd = x.loc[pool].mean(), x.loc[pool].std().replace(0, 1)
        z = (x - mu) / sd
        z["tov"] = -z["tov"]
        z_total = (z if floor is None else z.clip(lower=floor)).sum(axis=1)
        pool = z_total.sort_values(ascending=False).head(n_pool).index
    out = z.add_prefix("z_")
    out["z_total"] = z_total
    out["z_floor_adj"] = z_total - z.sum(axis=1)
    return out
