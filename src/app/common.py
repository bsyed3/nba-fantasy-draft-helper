"""
Shared pieces of the Streamlit app: the cached data and engine, the league (there is only one), and the helpers every page uses.
"""
from __future__ import annotations

import os
import sqlite3
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.draft import season, store
from src.draft.engine import DraftEngine
from src.draft.league_rules import banned_player_ids
from src.draft.policy import AdaptivePolicy, load_weeks
from src.draft.state import DraftState, LeagueSettings
from src.draft.values import CAT_LABELS, CATEGORIES
from src.features.names import search_key
from src.simulation.copula import Library

DATA = ROOT / "data" / "processed"
DB_PATH = Path(os.environ.get("NBA_DB", DATA / "nba.db"))  # override to keep test drafts out of the real DB
LEAGUE = "ESPN 10-team"                                    # the one league this app manages
PER_GAME_COLS = ["mpg", "pts", "tpm", "reb", "ast", "stl", "blk", "fg_pct", "ft_pct", "tov"]
COL_LABELS = {"name": "Player", "pos": "Pos", "team": "Team", "age_t": "Age", "espn_adp": "ESPN ADP", "gp82": "Proj GP (mean)", "gp_med": "Typical GP (median)", "gp_rng": "GP range (P25–P75)",
              "rank": "Board rank", "pg_rank": "Per-game rank", "mpg": "MPG", "pts": "PPG", "tpm": "3PM", "reb": "REB",
              "ast": "AST", "stl": "STL", "blk": "BLK", "fg_pct": "FG%", "ft_pct": "FT%", "tov": "TO", "z_total": "Value"}
CAT_COLS = [CAT_LABELS[c] for c in CATEGORIES]


# ---------------------------------------------------------------------------
# cached resources
# ---------------------------------------------------------------------------
@st.cache_resource
def get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    store.ensure_schema(conn)
    return conn


@st.cache_resource
def load_data():
    years = sorted(int(p.stem.split("_")[1]) for p in DATA.glob("library_*.npz"))
    if not years:
        return None
    year = years[-1]
    proj = pd.read_parquet(DATA / f"projections_{year}.parquet")
    lib = Library.load(DATA / f"library_{year}.npz")
    from nba_api.stats.static import teams as static_teams
    abbr = {t["id"]: t["abbreviation"] for t in static_teams.get_teams()}
    proj["team"] = proj["team_id"].map(abbr).fillna("FA")
    if "not_rostered" in proj:                                                     # last known team, on no current roster as of the data refresh
        nr = proj["not_rostered"].fillna(False).astype(bool)
        proj["team"] = proj["team"].where(~nr, proj["team"] + "*")
    proj["pos"] = proj["elig"]
    proj["search"] = proj["name"].map(search_key)                                  # plain-letter spelling: "dayron sharpe" finds Day'Ron Sharpe
    proj = proj[proj["player_id"].isin(lib.player_ids)].reset_index(drop=True)
    from src.draft.values import z_scores
    proj["gp82"] = (proj["avail"] * 82).round(0)                                   # projected (mean) games played over an 82-game season
    if "gp_median" in proj:                                                        # typical (median) games and the P25-P75 range: display only
        proj["gp_med"] = proj["gp_median"].round(0)
        proj["gp_rng"] = proj["gp_p25"].round(0).astype(int).astype(str) + "–" + proj["gp_p75"].round(0).astype(int).astype(str)
    else:
        proj["gp_med"], proj["gp_rng"] = float("nan"), ""
    proj["pg_rank"] = (z_scores(proj.assign(avail=1.0))["z_total"].rank(ascending=False, method="first")
                       .astype(int).to_numpy())                                     # value rank ignoring availability
    return proj, lib, year


@st.cache_resource
def get_engine(_proj, _lib, settings: LeagueSettings) -> DraftEngine:
    return DraftEngine(_lib, _proj, settings)


@st.cache_resource
def get_policy(_engine: DraftEngine, settings: LeagueSettings) -> AdaptivePolicy:
    """The category-aware policy, searching the 200-draws-per-week table built by scripts/build_library.py when it exists
    (it was tested to pick better than the engine library's 40 draws), else the engine's own library."""
    years = sorted(int(p.stem.split("_")[1]) for p in DATA.glob("library_*.npz"))
    weeks = load_weeks(DATA / f"policy_weeks_{years[-1]}.npz", _engine.pid) if years else None
    return AdaptivePolicy(_engine, weeks=weeks)


@st.cache_data(show_spinner="Simulating the rest of the draft...")
def cached_recommend(_engine: DraftEngine, settings: LeagueSettings, picks: tuple, excluded: tuple,
                     max_rank: int, rollouts: int, extra: tuple = (), moves: tuple = ()) -> pd.DataFrame:
    state = DraftState(settings, list(picks), set(excluded), set(_engine.center_ids), moves=list(moves))
    return _engine.recommend(state, rollouts=rollouts, pergame_k=10, extra_candidates=list(extra), max_rank=max_rank)


@st.cache_data(show_spinner="Projecting every team...")
def cached_outlook(_engine: DraftEngine, settings: LeagueSettings, picks: tuple, excluded: tuple, moves: tuple = ()):
    state = DraftState(settings, list(picks), set(excluded), set(_engine.center_ids), moves=list(moves))
    ev, rosters = _engine.projected_outcome(state)
    return ev, rosters


@st.cache_data
def all_players() -> pd.DataFrame:
    """Every player the database knows (not only the 600-odd on the board), for picks of someone who has no projection."""
    df = pd.read_sql_query("SELECT player_id, name, last_year FROM player WHERE name IS NOT NULL", get_conn())
    df = df[~df["player_id"].isin(banned_player_ids())]                                   # the league's banned players are never offered
    df["search"] = df["name"].map(search_key)
    return df


@st.cache_data(show_spinner=False)
def cached_period_weeks(_conn, season_name: str, n_regular: int, n_playoff: int):
    return season.period_weeks(_conn, season_name, n_regular, n_playoff)


@st.cache_data(show_spinner="Scoring every team on the simulated weeks...")
def cached_league_table(_engine: DraftEngine, settings: LeagueSettings, rosters: tuple, weeks: tuple | None = None):
    """The projected league table for the rosters given as ((slot, (player index, ...)), ...)."""
    return season.league_table(_engine, {t: list(r) for t, r in rosters}, settings.reg_season_weeks + settings.playoff_weeks,
                               settings.reg_season_weeks, None if weeks is None else np.array(weeks))


def rosters_key(rosters: dict[int, list[int]]) -> tuple:
    return tuple((t, tuple(sorted(rosters[t]))) for t in sorted(rosters))


# ---------------------------------------------------------------------------
# the page context
# ---------------------------------------------------------------------------
@dataclass
class Ctx:
    proj: pd.DataFrame
    proj_by_id: pd.DataFrame
    lib: Library
    year: int
    conn: sqlite3.Connection
    state: DraftState
    S: LeagueSettings
    engine: DraftEngine
    team_names: list[str]

    @property
    def me(self) -> int:
        return self.S.my_slot

    def tname(self, slot: int, me_tag: str = "") -> str:
        """A team's name by draft slot ('Team 3' until named); `me_tag` is appended for my own team."""
        return self.team_names[slot - 1] + (me_tag if slot == self.me else "")

    def player_name(self, pid: int) -> str:
        if pid in self.proj_by_id.index:
            return self.proj_by_id.loc[pid, "name"]
        a = all_players()
        m = a.loc[a["player_id"] == pid, "name"]
        return f"{m.iloc[0]} (not on the board)" if len(m) else str(pid)

    def roster_frame(self, ids: list[int]) -> pd.DataFrame:
        cols = ["name", "pos", "team", "gp82", "gp_med"] + PER_GAME_COLS
        known = [i for i in ids if i in self.proj_by_id.index]
        out = self.proj_by_id.loc[known, cols].copy() if known else pd.DataFrame(columns=cols)
        out[["fg_pct", "ft_pct"]] *= 100
        out = out.round(1).reset_index(drop=True)
        extra = [{"name": self.player_name(i)} for i in ids if i not in self.proj_by_id.index]   # a pick with no projection: shown by name only
        if extra:
            out = pd.concat([out, pd.DataFrame(extra)], ignore_index=True)
        return out.rename(columns=COL_LABELS)

    # --- rosters on the engine's player indices -------------------------------
    def idx(self, ids: list[int]) -> list[int]:
        """Engine player indices of the player ids that have a projection."""
        return [self.engine.idx_of[int(p)] for p in ids if int(p) in self.engine.idx_of]

    def rosters_idx(self) -> dict[int, list[int]]:
        """Every team's active roster (bench included, injured reserve not) as engine indices, with the post-draft moves applied."""
        return {t: self.idx(r) for t, r in self.state.rosters().items()}

    def pid(self, i: int) -> int:
        return int(self.engine.pid[i])

    def name_of(self, i: int) -> str:
        return str(self.proj_by_id.loc[self.pid(i), "name"])

    def free_agent_ids(self) -> list[int]:
        """Players with a projection who are on no roster (injured reserve included) and are not banned."""
        taken = self.state.drafted | banned_player_ids()
        return [int(p) for p in self.proj["player_id"] if int(p) not in taken]

    def league_table(self, rosters: dict[int, list[int]] | None = None) -> season.LeagueTable:
        rosters = self.rosters_idx() if rosters is None else rosters
        w = cached_period_weeks(self.conn, self.S.season, self.S.reg_season_weeks, self.S.playoff_weeks)
        return cached_league_table(self.engine, self.S, rosters_key(rosters), None if w is None else tuple(float(x) for x in w))


def cat_columns() -> dict:
    return {f"cat_{c}": st.column_config.ProgressColumn(CAT_LABELS[c], min_value=0, max_value=100, format="%.0f%%") for c in CATEGORIES}


def label(row) -> str:
    """Pick-box label. When the plain-letter spelling differs from the name (accents, apostrophes, periods, hyphens) it is appended, so the
    box's own type-to-search finds "Dayron" as well as "Day'Ron"."""
    base = f"{int(row['rank'])}. {row['name']} ({row['pos']}, {row['team']})"
    return base if row["search"] == str(row["name"]).lower() else f"{base} · {row['search']}"


def _colour(v: float, low: float, high: float, reverse: bool) -> str:
    if pd.isna(v):
        return ""
    t = min(max((v - low) / (high - low), 0.0), 1.0)
    t = 1 - t if reverse else t
    r, g = (255, int(180 * 2 * t)) if t < 0.5 else (int(255 * 2 * (1 - t)), 180)
    return f"background-color: rgba({r}, {g}, 80, 0.35)"


def shade_multi(df: pd.DataFrame, specs: list[tuple]):
    """A Styler that colours cells from red (low) to green (high). Each spec is (cols, low, high[, reverse[, basis]]): `reverse` flips the
    scale (for ranks, where 1 is best); `basis` (same shape as df[cols]) supplies the numbers to colour by, e.g. the team's rank in a
    weekly total. The colours are translucent, so the table reads in both the light and the dark theme."""
    css = pd.DataFrame("", index=df.index, columns=df.columns)
    for cols, low, high, *rest in specs:
        reverse = rest[0] if rest else False
        basis = rest[1] if len(rest) > 1 else df[cols]
        css[cols] = basis.map(lambda v: _colour(v, low, high, reverse))
    return df.style.apply(lambda _: css, axis=None)


def shade(df: pd.DataFrame, cols: list[str], low: float, high: float, reverse: bool = False, basis: pd.DataFrame | None = None):
    """`shade_multi` for one set of columns."""
    return shade_multi(df, [(cols, low, high, reverse, df[cols] if basis is None else basis)])


def sidebar(ctx: Ctx) -> None:
    """The sidebar every page shares: data notes, team names and the league's rules."""
    st.sidebar.caption(f"Projections for the {ctx.year}-{str(ctx.year + 1)[-2:]} season · {len(ctx.proj)} players")
    if ctx.proj["team"].str.endswith("*").any():
        st.sidebar.caption("* = on no current NBA roster as of the last data refresh (a free agent, or signed or traded since): his last known team is shown.")
    if (ctx.proj["elig_source"] != "espn").any():
        st.sidebar.warning("Position eligibility is **approximated** from nba_api labels + stats. For ESPN's exact values run "
                           "`python scripts/fetch_espn_positions.py` (needs ESPN access), then rebuild projections + library.")
    S = ctx.S
    with st.sidebar.expander("Team names", expanded=False):
        st.caption("Draft-slot order (slot 1 picks first in round 1).")
        typed = [st.text_input(f"Slot {i + 1}" + (" (me)" if i + 1 == S.my_slot else ""), value=ctx.team_names[i], key=f"tn_{i}") for i in range(S.n_teams)]
        if typed != ctx.team_names and st.button("Save team names"):
            store.set_team_names(ctx.conn, LEAGUE, typed)
            st.rerun()
    with st.sidebar.expander("League settings", expanded=False):
        new_slot = st.number_input("My draft slot", 1, S.n_teams, S.my_slot, key="slot_edit")
        if new_slot != S.my_slot and st.button("Update slot"):
            store.save_league(ctx.conn, LeagueSettings(**{**S.__dict__, "my_slot": int(new_slot)}))
            st.rerun()
        st.write(f"{S.n_teams} teams · {S.roster_size} drafted · {S.active_slots} active (PG SG G SF PF F C + 3 UTIL, daily) · "
                 f"max {S.max_centers} centers · snake")
        st.write(f"{S.reg_season_weeks} regular-season weeks + {S.playoff_weeks} playoff weeks")


def load_context() -> Ctx:
    """Load everything a page needs, or stop the app with a message that says what to run."""
    data = load_data()
    if data is None:
        st.error("No simulation library found. Run `python scripts/ingest.py`, `python scripts/build_projections.py` "
                 "and `python scripts/build_library.py` first.")
        st.stop()
    proj, lib, year = data
    if set(proj["player_id"]) & banned_player_ids():
        st.error("The projections still include a player the league has banned (data/external/banned_players.csv). Rebuild them: "
                 "`python scripts/build_projections.py --reuse-backtest`, then `python scripts/build_library.py`.")
        st.stop()
    conn = get_conn()
    store.ensure_league(conn, LeagueSettings(name=LEAGUE, season=f"{year}-{str(year + 1)[-2:]}"))
    state = store.load_state(conn, LEAGUE)
    S = state.settings
    engine = get_engine(proj, lib, S)
    state.centers = set(engine.center_ids)  # C-eligible players count toward the roster cap
    return Ctx(proj, proj.set_index("player_id"), lib, year, conn, state, S, engine, store.get_team_names(conn, LEAGUE, S.n_teams))
