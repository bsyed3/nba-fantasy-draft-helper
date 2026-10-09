"""
Fantasy position eligibility (ESPN slots PG, SG, SF, PF, C).

nba_api only exposes coarse positions (G, F, C and pairs like G-F, F-C).
ESPN's real eligibility is finer and usually broader (a player is eligible
wherever he played enough games), so this is an approximation that errs
narrow: per-36 assists/rebounds split the coarse labels into the five
slots. Anything here can be replaced with exact ESPN eligibility via
`data/external/positions.csv` (columns: `player_id` or `name`, and
`eligibility` like "PG,SG"), which always wins.

Flex slots are derived, not stored: G accepts PG/SG, F accepts SF/PF,
UTIL accepts anyone. "Center" for the roster cap = C-eligible.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

SLOTS = ["PG", "SG", "SF", "PF", "C"]
PG_AST36 = 6.0      # at/above: point guard
SG_AST36 = 3.5      # below: shooting guard only
BIG_REB36 = 9.0     # forwards at/above: power forward only
WING_REB36 = 5.5    # forwards below: small forward only


def default_eligibility(pos1: str | None, pos2: str | None, ast36: float, reb36: float) -> set[str]:
    p = {x for x in (pos1, pos2) if isinstance(x, str) and x}  # NaN/None are not positions
    if p == {"C"}:
        e = {"C"}
    elif p == {"C", "F"}:
        e = {"PF", "C"}
    elif p == {"F", "G"}:
        e = {"SG", "SF"}
    elif p == {"G"}:
        e = {"PG"} if ast36 >= PG_AST36 else {"SG"} if ast36 < SG_AST36 else {"PG", "SG"}
    elif p == {"F"}:
        e = {"PF"} if reb36 >= BIG_REB36 else {"SF"} if reb36 < WING_REB36 else {"SF", "PF"}
    elif p == {"C", "G"}:
        e = {"PG", "C"}
    else:
        e = {"SF", "PF"}
    if ast36 >= 7.5 and "PG" not in e and "C" not in e:
        e = e | {"PG"}  # playmaking forwards/wings run the point in fantasy
    return e


def parse_eligibility(text: str) -> set[str]:
    return {t.strip().upper() for t in str(text).replace("/", ",").split(",") if t.strip().upper() in SLOTS}


def add_eligibility(proj: pd.DataFrame, overrides_path: str | Path | None = None) -> pd.DataFrame:
    """
    Adds `elig` ("PG/SG"), `elig_source` ("espn" | "heuristic"), `is_center`, one bool column per
    slot (`e_PG`...) and `espn_adp` (NaN unless the file supplies it). Needs r_ast, r_reb.
    positions.csv (from scripts/fetch_espn_positions.py) wins over the heuristic.
    """
    out = proj.copy()
    sets = [default_eligibility(a, b, ast, reb) for a, b, ast, reb in
            zip(out["position_1"], out["position_2"], out["r_ast"], out["r_reb"])]
    src = ["heuristic"] * len(out)
    adp = pd.Series(np.nan, index=out.index, dtype=float)
    if overrides_path and Path(overrides_path).exists():
        ov = pd.read_csv(overrides_path)
        elig_col = "eligibility" if "eligibility" in ov else None
        keyed = {}
        for key in ("player_id", "name"):
            if key in ov:
                for _, r in ov.iterrows():
                    k = r[key] if key == "player_id" else str(r[key]).lower()
                    keyed.setdefault((key, k), r)
        for i, (pid, name) in enumerate(zip(out["player_id"], out["name"])):
            r = keyed.get(("player_id", pid))
            if r is None:
                r = keyed.get(("name", str(name).lower()))
            if r is None:
                continue
            if elig_col and parse_eligibility(r[elig_col]):
                sets[i], src[i] = parse_eligibility(r[elig_col]), "espn"
            if "espn_adp" in ov and pd.notna(r.get("espn_adp")):
                adp.iloc[i] = float(r["espn_adp"])
    out["elig"] = ["/".join(s for s in SLOTS if s in e) for e in sets]
    out["elig_source"] = src
    out["espn_adp"] = adp.to_numpy()
    for s in SLOTS:
        out[f"e_{s}"] = [s in e for e in sets]
    out["is_center"] = out["e_C"]
    return out
