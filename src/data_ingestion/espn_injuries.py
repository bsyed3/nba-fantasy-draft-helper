"""
ESPN's NBA injuries page (https://www.espn.com/nba/injuries, data by Rotowire) -> games each player is expected to miss at the
START of the season, for data/external/known_absences.csv.

The page lists, per team, every injured player with a status (Out / Day-To-Day) and an estimated return date written only as
month and day ("Oct 23", "Jan 2"). We keep the players with the statuses asked for (default: Out only), work out the full date
(the comment on each row is dated, so the return is the first such month/day on or after the comment), and count the team's
regular-season games before it in the schedule table -- that is `games_out`, scaled to an 82-game season.
"""
from __future__ import annotations

import datetime as dt
import re
import sqlite3

import numpy as np
import pandas as pd

from src.features.names import normalize_name

URL = "https://www.espn.com/nba/injuries"
COLUMNS = ["team", "name", "pos", "est_return", "status", "comment"]
_MONTHS = {m: i for i, m in enumerate(["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"], 1)}


def fetch_html(url: str = URL, timeout: float = 30.0) -> str:
    """Download the page. Certificate verification stays on; run this from a network that can reach ESPN."""
    import requests

    r = requests.get(url, headers={"User-Agent": "Mozilla/5.0 (compatible; nba-draft-helper)"}, timeout=timeout)
    r.raise_for_status()
    return r.text


def parse_html(html: str) -> pd.DataFrame:
    """One row per injury listing: team, name, pos, est_return ('Oct 23'), status ('Out'), comment."""
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html, "lxml")
    rows = []
    for box in soup.select(".Table__league-injuries"):
        team_el = box.select_one(".injuries__teamName")
        team = team_el.get_text(strip=True) if team_el else ""
        for tr in box.select("tbody tr"):
            td = tr.find_all("td")
            if len(td) < 5:
                continue
            rows.append(dict(team=team, name=td[0].get_text(strip=True), pos=td[1].get_text(strip=True),
                             est_return=td[2].get_text(strip=True), status=td[3].get_text(strip=True), comment=td[4].get_text(strip=True)))
    return pd.DataFrame(rows, columns=COLUMNS)


def _month_day(text: str) -> tuple[int, int] | None:
    m = re.match(r"\s*([A-Za-z]{3})[a-z]*\.?\s+(\d{1,2})\b", str(text))
    if not m or m.group(1).lower() not in _MONTHS:
        return None
    return _MONTHS[m.group(1).lower()], int(m.group(2))


def _team_key(name: str) -> str:
    """ESPN says 'LA Clippers', the NBA's team table says 'Los Angeles Clippers'."""
    key = normalize_name(name)
    return "los angeles " + key[3:] if key.startswith("la ") else key


def _safe_date(year: int, month: int, day: int) -> dt.date | None:
    try:
        return dt.date(year, month, day)
    except ValueError:
        return None


def resolve_return_date(est_return: str, comment: str, today: dt.date) -> dt.date | None:
    """
    'Oct 23' + a comment dated 'Oct 9: ...' -> 2026-10-23. The comment date is the latest occurrence of its month/day on or before
    `today`; the return date is the first occurrence of the estimated month/day on or after the comment date (so 'Jan 1' written
    in a February comment means the following January). None when ESPN gives no usable date.
    """
    ret = _month_day(est_return)
    if ret is None:
        return None
    cd = _month_day(comment)
    c_date = today
    if cd is not None:
        c = _safe_date(today.year, *cd)
        if c is not None and c > today:
            c = _safe_date(today.year - 1, *cd)
        c_date = c or today
    r = _safe_date(c_date.year, *ret)
    if r is not None and r < c_date:
        r = _safe_date(c_date.year + 1, *ret)
    return r


def team_games_before(conn: sqlite3.Connection, team_id: int, season: str, return_date: dt.date) -> tuple[int, int]:
    """(regular-season games the team plays before `return_date`, games the team has in the schedule table)."""
    dates = [r[0] for r in conn.execute(
        "SELECT date FROM game WHERE season = ? AND game_type = 'regular' AND (home_team_id = ? OR away_team_id = ?)",
        (season, team_id, team_id))]
    return sum(d < return_date.isoformat() for d in dates), len(dates)


def known_absences(listings: pd.DataFrame, conn: sqlite3.Connection, season: str = "2026-27", today: dt.date | None = None,
                   statuses: tuple[str, ...] = ("Out",), season_games: int = 82) -> tuple[pd.DataFrame, list[str]]:
    """
    Listings with a status in `statuses` -> player_id, name, games_out (of `season_games`), team, status, est_return, return_date.
    Players are matched to our player table by normalised name (accents, punctuation, suffixes ignored), using the team to break ties.
    Returns (table, warnings) -- anything that could not be matched or dated is reported, not guessed.
    """
    today = today or dt.date.today()
    season_year = int(season[:4])
    teams = pd.read_sql_query("SELECT team_id, name FROM team", conn)
    team_id = {_team_key(n): int(t) for t, n in zip(teams["team_id"], teams["name"])}
    players = pd.read_sql_query("SELECT player_id, name, team_id FROM player WHERE last_year >= ?", conn, params=(season_year,))
    players["key"] = players["name"].map(normalize_name)
    by_key = {k: g for k, g in players.groupby("key")}

    out, warnings = [], []
    for _, r in listings[listings["status"].isin(statuses)].iterrows():
        tid = team_id.get(_team_key(r["team"]))
        ret = resolve_return_date(r["est_return"], r["comment"], today)
        if tid is None:
            warnings.append(f"{r['name']}: unknown team '{r['team']}'")
            continue
        if ret is None:
            warnings.append(f"{r['name']}: no usable return date ('{r['est_return']}')")
            continue
        cand = by_key.get(normalize_name(r["name"]))
        if cand is not None and len(cand) > 1:
            cand = cand[cand["team_id"] == tid]
        if cand is None or len(cand) != 1:
            warnings.append(f"{r['name']} ({r['team']}): not matched to exactly one player")
            continue
        before, total = team_games_before(conn, tid, season, ret)
        if total == 0:
            warnings.append(f"{r['name']}: no {season} schedule for {r['team']}")
            continue
        out.append(dict(player_id=int(cand["player_id"].iloc[0]), name=cand["name"].iloc[0], games_out=int(round(season_games * before / total)),
                        team=r["team"], status=r["status"], est_return=r["est_return"], return_date=ret.isoformat()))
    cols = ["player_id", "name", "games_out", "team", "status", "est_return", "return_date"]
    return pd.DataFrame(out, columns=cols), warnings
