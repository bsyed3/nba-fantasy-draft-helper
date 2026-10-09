"""Shared synthetic fixtures (no network, no real data needed)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.features.positions import add_eligibility
from src.simulation.calibration import COUNT_DIMS, DIMS, Calibration
from src.simulation.schedule import Schedule


@pytest.fixture
def cal() -> Calibration:
    n = len(DIMS)
    corr = np.full((n, n), 0.15)
    np.fill_diagonal(corr, 1.0)
    sig = {b: {**{d: 0.2 for d in COUNT_DIMS}, "p2": 0.03, "p3": 0.03, "ft": 0.04} for b in ("0-1", "2-4", "5+")}
    rook = {t: {**{d: 0.5 for d in COUNT_DIMS}, "p2": 0.08, "p3": 0.09, "ft": 0.1}
            for t in ("1-5", "6-14", "15-30", "31-60", "undrafted")}
    return Calibration(
        dispersion={d: 10.0 for d in COUNT_DIMS}, rho={"p2": 0.0, "p3": 0.0, "ft": 0.004},
        corr=corr.tolist(), games_pmf={"2": 0.15, "3": 0.4, "4": 0.45}, absence_c=0.4,
        talent_sigma=sig, talent_corr=corr.tolist(), rookie_sigma=rook,
    )


@pytest.fixture
def schedule() -> Schedule:
    """4 weeks, 3 NBA teams: 4-game, 3-game and 5-game weeks every week."""
    masks = np.zeros((4, 3, 7), dtype=bool)
    masks[:, 0, [0, 2, 4, 6]] = True
    masks[:, 1, [1, 3, 5]] = True
    masks[:, 2, [0, 1, 2, 3, 4]] = True
    return Schedule(masks, [101, 102, 103], list(range(4)))


def make_projection(n: int = 60, seed: int = 0) -> pd.DataFrame:
    """Synthetic projections with a clear talent gradient (player 1 best in every category)."""
    rng = np.random.default_rng(seed)
    skill = np.linspace(1.6, 0.4, n)
    df = pd.DataFrame({
        "player_id": np.arange(1, n + 1), "name": [f"P{i}" for i in range(n)],
        "position_1": rng.choice(list("GFC"), n), "position_2": None,
        "team_id": rng.choice([101, 102, 103], n),
        "exp": 5, "draft_pick": 20.0, "avail": np.clip(rng.normal(0.8, 0.08, n), 0.4, 0.98),
        "fga2": 6 * skill, "fga3": 3 * skill, "fta": 3.5 * skill, "reb": 6 * skill, "ast": 4 * skill,
        "stl": 1.1 * skill, "blk": 0.8 * skill, "tov": 2.0 * skill,
        "r_p2": 0.53, "r_p3": 0.36, "r_ft": 0.78, "mpg": 30 * skill.clip(0.7, 1.2),
    })
    df["r_ast"] = df["ast"] / df["mpg"] * 36
    df["r_reb"] = df["reb"] / df["mpg"] * 36
    df["tpm"] = df["fga3"] * df["r_p3"]
    df["fgm"] = df["fga2"] * df["r_p2"] + df["tpm"]
    df["fga"] = df["fga2"] + df["fga3"]
    df["ftm"] = df["fta"] * df["r_ft"]
    df["pts"] = 2 * df["fgm"] + df["tpm"] + df["ftm"]
    df["fg_pct"] = df["fgm"] / df["fga"]
    df["ft_pct"] = df["r_ft"]
    from src.draft.values import z_scores
    df = pd.concat([df, z_scores(df, n_pool=30)], axis=1)
    return add_eligibility(df)
