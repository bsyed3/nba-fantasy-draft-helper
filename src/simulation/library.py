"""The simulation library container (kept separate so schedule/copula can both import it)."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

STAT_ORDER = ["fgm", "fga", "ftm", "fta", "tpm", "reb", "ast", "stl", "blk", "tov"]
STAT_IDX = {s: i for i, s in enumerate(STAT_ORDER)}


@dataclass
class Library:
    """
    Simulated (or actual) fantasy matchup periods at day resolution.
    stats:  [players, periods, D days from period start (D=14 so two-week periods fit), 10] uint8 box-score line
            for that day (all counts are small integers)
    played: [players, periods, D] bool, True if the player appeared in a game that day
    """
    player_ids: np.ndarray
    stats: np.ndarray
    played: np.ndarray

    @property
    def n_sims(self) -> int:
        return self.stats.shape[1]

    def save(self, path: str | Path) -> None:
        np.savez_compressed(path, player_ids=self.player_ids, stats=self.stats, played=self.played)

    @staticmethod
    def load(path: str | Path) -> "Library":
        z = np.load(path)
        return Library(z["player_ids"], z["stats"], z["played"])
