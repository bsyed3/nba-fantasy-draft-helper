"""
Thin wrappers around nba_api endpoints, returning pandas DataFrames shaped
to match schema.sql / docs/data_model.md. Nothing here writes to the
database — see load_to_db.py for that.

IMPORTANT: nba_api calls stats.nba.com directly over the network. Some
sandboxed/restricted environments (proxies that only allowlist package
registries) will block this with a connection or proxy error — that's a
network policy issue, not a bug in this code. Run ingestion from a normal
dev machine. get_teams() is the exception: it reads a static list bundled
in the package and needs no network call at all.

Column names below are taken from nba_api's LeagueGameLog and
CommonPlayerInfo endpoints as of when this was written — if nba_api
changes its response shape, the rename dicts here are the first place to
check.
"""
from __future__ import annotations

import time

import pandas as pd
from nba_api.stats.endpoints import (
    commonplayerinfo,
    leaguedashplayerbiostats,
    leaguegamelog,
    playerindex,
    scheduleleaguev2,
)
from nba_api.stats.static import teams as static_teams

REQUEST_DELAY_SECONDS = 0.6  # be polite to stats.nba.com; raise if rate-limited
MAX_RETRIES = 4


def _with_retry(fn, *args, **kwargs):
    """stats.nba.com times out intermittently; retry with backoff."""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return fn(*args, **kwargs)
        except Exception:
            if attempt == MAX_RETRIES:
                raise
            time.sleep(2 ** attempt)


def get_teams() -> pd.DataFrame:
    """Static team list (team_id, name). No network call."""
    rows = static_teams.get_teams()
    return (
        pd.DataFrame(rows)[["id", "full_name"]]
        .rename(columns={"id": "team_id", "full_name": "name"})
    )


def get_season_game_logs(season: str, season_type: str = "Regular Season") -> pd.DataFrame:
    """
    Every player-game box score for one season, in a single call.

    season: e.g. "2023-24"

    Returns box_score's columns (game_id, player_id, team_id, min, fga,
    fgm, tpa, tpm, fta, ftm, oreb, dreb, ast, stl, blk, tov) plus `name`,
    `game_date`, `matchup` kept alongside for building the player/game
    tables — drop them before loading into box_score itself (see
    load_to_db.load_box_scores, which already does this).
    """
    df = _with_retry(
        lambda: leaguegamelog.LeagueGameLog(
            season=season,
            season_type_all_star=season_type,
            player_or_team_abbreviation="P",
            timeout=60,
        ).get_data_frames()[0]
    )

    df = df.rename(columns={
        "GAME_ID": "game_id",
        "PLAYER_ID": "player_id",
        "PLAYER_NAME": "name",
        "TEAM_ID": "team_id",
        "MIN": "min",
        "FGA": "fga",
        "FGM": "fgm",
        "FG3A": "tpa",
        "FG3M": "tpm",
        "FTA": "fta",
        "FTM": "ftm",
        "OREB": "oreb",
        "DREB": "dreb",
        "AST": "ast",
        "STL": "stl",
        "BLK": "blk",
        "TOV": "tov",
        "GAME_DATE": "game_date",
        "MATCHUP": "matchup",
    })

    cols = [
        "game_id", "player_id", "team_id", "name", "min", "fga", "fgm",
        "tpa", "tpm", "fta", "ftm", "oreb", "dreb", "ast", "stl", "blk",
        "tov", "game_date", "matchup",
    ]
    return df[cols]


def get_season_games(season: str, season_type: str = "Regular Season") -> pd.DataFrame:
    """
    Build the `game` table (one row per game_id with home/away team_id)
    from the TEAM-level league game log — each game_id has exactly two
    team rows, distinguished by MATCHUP ("vs." = home, "@" = away).

    Games without exactly two team rows are dropped rather than guessed at — check the printed count in
    scripts/ingest.py and inspect manually if it's more than a handful.
    """
    df = _with_retry(
        lambda: leaguegamelog.LeagueGameLog(
            season=season,
            season_type_all_star=season_type,
            player_or_team_abbreviation="T",
            timeout=60,
        ).get_data_frames()[0]
    )
    game_type = "regular" if season_type == "Regular Season" else "playoff"

    games = []
    for game_id, grp in df.groupby("GAME_ID"):
        # '002' = regular season. The NBA Cup final is '006...' and does not
        # count toward regular-season stats (or the fantasy season).
        if len(grp) != 2 or not str(game_id).startswith("002"):
            continue
        home = grp[grp["MATCHUP"].str.contains("vs.", regex=False)]
        away = grp[grp["MATCHUP"].str.contains("@", regex=False)]
        if home.empty or away.empty:
            # Neutral-site games (international games, NBA Cup final) list
            # both teams as "@". Home/away is not used downstream, so keep
            # the game and assign sides arbitrarily rather than drop it.
            home, away = grp.iloc[[0]], grp.iloc[[1]]
        games.append({
            "game_id": game_id,
            "date": grp["GAME_DATE"].iloc[0],
            "home_team_id": int(home["TEAM_ID"].iloc[0]),
            "away_team_id": int(away["TEAM_ID"].iloc[0]),
            "season": season,
            "game_type": game_type,
        })
    return pd.DataFrame(games)


def _safe_int(value) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def get_player_bio(player_id: int) -> dict:
    """
    One player's birthdate/position/draft info. This is ONE network call
    per player — fine for enriching a few hundred players with the delay
    above, but don't call it in a tight loop without REQUEST_DELAY_SECONDS,
    and expect it to take a while for the full league.
    """
    info = commonplayerinfo.CommonPlayerInfo(player_id=player_id, timeout=30)
    row = info.get_data_frames()[0].iloc[0]
    time.sleep(REQUEST_DELAY_SECONDS)

    position = row["POSITION"] or ""
    parts = position.split("-")
    return {
        "player_id": player_id,
        "name": row["DISPLAY_FIRST_LAST"],
        "birthdate": row["BIRTHDATE"],
        "position_1": parts[0] if parts and parts[0] else None,
        "position_2": parts[1] if len(parts) > 1 else None,
        "draft_year": _safe_int(row["DRAFT_YEAR"]),
        "draft_position": _safe_int(row["DRAFT_NUMBER"]),
    }


def get_player_index() -> pd.DataFrame:
    """
    Every player who has ever appeared in the NBA, in ONE call: position,
    draft slot, rookie year and last year. This replaces ~600 per-player
    get_player_bio() calls for everything except birthdate.

    PlayerIndex FROM_YEAR / TO_YEAR use the *start-year* convention
    (2016 = the 2016-17 season). Players drafted in the current year show
    up here before they have any box score, which is how the rookie
    class reaches the draft pool.
    """
    df = _with_retry(
        lambda: playerindex.PlayerIndex(historical_nullable=1, timeout=60).get_data_frames()[0]
    )
    pos = df["POSITION"].fillna("").astype(str).str.split("-", n=1, expand=True)
    if 1 not in pos.columns:
        pos[1] = None
    out = pd.DataFrame({
        "player_id": df["PERSON_ID"].astype(int),
        "name": (df["PLAYER_FIRST_NAME"].fillna("") + " " + df["PLAYER_LAST_NAME"].fillna("")).str.strip(),
        "position_1": pos[0].where(pos[0] != ""),
        "position_2": pos[1].where(pos[1].fillna("") != ""),
        "draft_year": pd.to_numeric(df["DRAFT_YEAR"], errors="coerce"),
        "draft_position": pd.to_numeric(df["DRAFT_NUMBER"], errors="coerce"),
        "rookie_year": pd.to_numeric(df["FROM_YEAR"], errors="coerce"),
        "last_year": pd.to_numeric(df["TO_YEAR"], errors="coerce"),
        "team_id": pd.to_numeric(df["TEAM_ID"], errors="coerce").fillna(0).astype(int),
    })
    for col in ["draft_year", "draft_position", "rookie_year", "last_year"]:
        out[col] = out[col].astype("Int64")
    return out


def get_season_player_ages(season: str) -> pd.DataFrame:
    """player_id, season, age, team_id for everyone who played in `season` (one call)."""
    df = _with_retry(
        lambda: leaguedashplayerbiostats.LeagueDashPlayerBioStats(
            season=season, timeout=60
        ).get_data_frames()[0]
    )
    return pd.DataFrame({
        "player_id": df["PLAYER_ID"].astype(int),
        "season": season,
        "age": df["AGE"].astype(float),
        "team_id": df["TEAM_ID"].astype(int),
    })


def get_season_schedule(season: str) -> pd.DataFrame:
    """
    The published regular-season schedule as `game` rows (no box scores).
    Only '002...' (regular season) games with both teams known: preseason
    ('001') games, and the NBA Cup final ('006', not a fantasy game), are
    dropped, as are knockout games whose teams aren't decided yet.
    """
    df = _with_retry(lambda: scheduleleaguev2.ScheduleLeagueV2(season=season, timeout=60).get_data_frames()[0])
    df = df[df["gameId"].astype(str).str.startswith("002")]
    df = df[(df["homeTeam_teamId"] > 0) & (df["awayTeam_teamId"] > 0)]
    return pd.DataFrame({
        "game_id": df["gameId"].astype(str),
        "date": pd.to_datetime(df["gameDate"], format="%m/%d/%Y %H:%M:%S").dt.strftime("%Y-%m-%d"),
        "home_team_id": df["homeTeam_teamId"].astype(int),
        "away_team_id": df["awayTeam_teamId"].astype(int),
        "season": season,
        "game_type": "regular",
    }).reset_index(drop=True)
