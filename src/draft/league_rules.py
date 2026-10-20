"""League rules that live outside the draft state."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

BANNED_PATH = Path(__file__).resolve().parents[2] / "data" / "external" / "banned_players.csv"


def banned_player_ids(path: Path | str | None = None) -> set[int]:
    """
    Players the league does not allow anyone to draft or roster (data/external/banned_players.csv: player_id, name, reason).
    They are dropped from the projection pool when the projections are built, so the board, the simulation library, the draft
    recommendations and every analysis never see them; the app also keeps them out of its "any player" search.
    """
    p = Path(path) if path else BANNED_PATH
    if not p.exists():
        return set()
    return {int(x) for x in pd.read_csv(p)["player_id"].dropna()}
