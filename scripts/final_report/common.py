"""
Shared pieces for the final project report: paths, the loaded data, figure numbering, and markdown helpers.

Every number in the report is computed from the project's data and saved experiment outputs; the few figures that come
from earlier one-off runs (documented in docs/HANDOFF.md) are marked as such where they are used.
"""
from __future__ import annotations

import sqlite3
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.draft.values import CAT_LABELS, CATEGORIES, z_scores  # noqa: E402
from src.features.player_seasons import load_player_season_table  # noqa: E402
from src.reporting import analysis as A  # noqa: E402
from src.reporting import figs as F  # noqa: E402
from src.reporting.figs import C, md_table  # noqa: E402

DATA = ROOT / "data" / "processed"
EXP = ROOT / "exports"
REPORT_DATA = EXP / "report_data"
ADP_XLSX = Path(r"C:\Users\SyedB\Downloads\fantasy_basketball_adp_2026-27.xlsx")
OUT = Path(r"C:\Users\SyedB\Personal\NBA Fantasy\NBA Draft Model Report")
FIGS = OUT / "figures"

SEASON = lambda y: f"{int(y)}-{str(int(y) + 1)[-2:]}"          # 2025 -> "2025-26"
STAT_LABEL = {"fga2": "2-pt attempts /36", "fga3": "3-pt attempts /36", "fta": "FT attempts /36", "reb": "Rebounds /36",
              "ast": "Assists /36", "stl": "Steals /36", "blk": "Blocks /36", "tov": "Turnovers /36",
              "p2": "2-pt %", "p3": "3-pt %", "ft": "FT %", "mpg": "Minutes per game", "avail": "Availability (GP / team games)"}


@dataclass
class Ctx:
    conn: sqlite3.Connection
    st: pd.DataFrame                  # per-player-season totals
    apg: pd.DataFrame                 # actual per-game values per player-season
    proj: pd.DataFrame                # 2026-27 projections (current, with overrides)
    oos: pd.DataFrame                 # walk-forward out-of-sample per-game predictions
    summary: pd.DataFrame             # backtest summary (stat x model)
    names: dict
    market: pd.DataFrame | None       # our board vs the market (derived from the ADP workbook; comparison only)
    S: dict = field(default_factory=dict)      # numbers collected while writing, reused later (abstract / conclusion)
    nfig: int = 0

    def fig(self, name: str, caption: str, width: str = "100%") -> str:
        """Markdown for a numbered figure (the PNG must already exist in FIGS)."""
        self.nfig += 1
        return f"\n![Figure {self.nfig}](figures/{name}.png)\n\n*Figure {self.nfig}. {caption}*\n"


def load() -> Ctx:
    conn = sqlite3.connect(DATA / "nba.db")
    st = load_player_season_table(conn)
    players = pd.read_sql_query("SELECT player_id, name FROM player", conn)
    market = pd.read_csv(EXP / "rank_vs_adp.csv") if (EXP / "rank_vs_adp.csv").exists() else None
    F.setup()
    FIGS.mkdir(parents=True, exist_ok=True)
    F.FIG_DIR = FIGS
    return Ctx(conn=conn, st=st, apg=A.actual_per_game(st), proj=pd.read_parquet(DATA / "projections_2026.parquet"),
               oos=pd.read_parquet(DATA / "backtest_oos.parquet"), summary=pd.read_csv(DATA / "backtest_summary.csv"),
               names=dict(zip(players["player_id"], players["name"])), market=market)


def pct(x: float, d: int = 1) -> str:
    return f"{x * 100:.{d}f}%"


def read_exp(name: str) -> pd.DataFrame | None:
    p = REPORT_DATA / f"{name}.csv"
    return pd.read_csv(p) if p.exists() else None
