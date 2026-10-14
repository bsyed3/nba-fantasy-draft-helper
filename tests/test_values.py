import numpy as np
import pandas as pd

from src.draft.values import CATEGORIES, Z_FLOOR, z_scores


def _league(n=60, seed=0):
    rng = np.random.default_rng(seed)
    fga = rng.uniform(5, 18, n); fta = rng.uniform(1, 7, n)
    df = pd.DataFrame({"mpg": rng.uniform(15, 36, n), "avail": 0.8, "fga": fga, "fta": fta, "tpm": rng.uniform(0, 3, n), "tpa": rng.uniform(1, 7, n),
                       "reb": rng.uniform(2, 10, n), "ast": rng.uniform(1, 8, n), "stl": rng.uniform(0.4, 1.8, n), "blk": rng.uniform(0.1, 1.5, n), "tov": rng.uniform(0.8, 3.5, n)})
    df["fgm"] = fga * rng.uniform(0.40, 0.55, n)
    df["ftm"] = fta * rng.uniform(0.72, 0.90, n)
    df["pts"] = 2 * df["fgm"] + df["tpm"] + df["ftm"]
    return df


def test_floor_limits_how_much_one_category_can_subtract_and_leaves_everyone_else_alone():
    df = _league()
    df.loc[0, ["fta", "ftm"]] = [12.0, 6.0]                       # a 50% free-throw shooter at 12 attempts a game: an extreme outlier
    raw = z_scores(df, n_pool=40, floor=None)
    capped = z_scores(df, n_pool=40)
    assert raw.loc[0, "z_ft_pct"] < Z_FLOOR - 1.0                  # the raw penalty is far beyond the floor
    assert np.allclose(raw["z_total"], raw[[f"z_{c}" for c in CATEGORIES]].sum(axis=1)) and (raw["z_floor_adj"] == 0).all()
    assert capped.loc[0, "z_floor_adj"] > 0.5 and capped.loc[0, "z_total"] > raw.loc[0, "z_total"]
    # the per-category z columns stay raw; the total just doesn't subtract more than |floor| per category
    expected = sum(max(capped.loc[0, f"z_{c}"], Z_FLOOR) for c in CATEGORIES)
    assert abs(capped.loc[0, "z_total"] - expected) < 1e-9
    nobody_else = capped.index != 0
    deep = (capped[[f"z_{c}" for c in CATEGORIES]] < Z_FLOOR).any(axis=1)
    assert np.allclose(capped.loc[nobody_else & ~deep.to_numpy(), "z_floor_adj"], 0.0)                     # only players beyond the floor are touched
    assert (capped["z_floor_adj"] >= -1e-12).all()                                                          # the floor can only add back, never subtract
