"""
Project the rest of this week's matchups from ESPN's current totals.

    python scripts/project_week.py                       # every matchup, from the last ESPN download (data/espn_cache/latest)
    python scripts/project_week.py --refresh             # check the network and download a fresh copy first (same checks as sync_espn.py)
    python scripts/project_week.py --next                # next week, in full (nothing banked yet); --week 7 for a later one
    python scripts/project_week.py --team Bilal          # only the matchup that involves this team
    python scripts/project_week.py --from 2026-11-20     # treat games from this day on as still to be played
    python scripts/project_week.py --sims 50000 --seed 3
(scripts/project_season.py runs the same engine over every remaining week and adds expected records.)

For every matchup it prints the current category score, the projected final totals, each team's chance to win each category, the chance of
every possible final score (e.g. 5-4, 6-3), the chance to win the week, and each fantasy team's NBA games (in its lineup, on its roster,
and per player: 0 = out or no game).

How it works: the current week's category totals come from ESPN (they only include finished days). The days still to play come from the NBA
schedule in nba.db; the Cup-week games that are not on the published schedule are placed at random, as in the draft library. Each day every
team plays its best 10 healthy players who have a game (position slots are ignored). Players on IR or tagged OUT do not play unless
known_absences.csv / known_absences_manual.csv has them back by that day. Beyond the known injuries, every player can also suffer a random
injury spell (about 7% of days missed, spells of about 8 days, so a hurt starter is replaced by the bench) and a day-to-day tag settles
within two days. Each player's per-game line is his ESPN season-to-date line blended with the preseason projection (worth 10 games).
Every game is simulated (negative-binomial counts for rebounds, assists, steals, blocks, turnovers and shot attempts, binomial makes), with
+-10% uncertainty in each player's true rate shared across his games, and the totals are added to what each team has already banked.

Assumptions to remember: ESPN's totals count games through the day before the "from" date (by default today's scoring period), so run it
before tonight's games or pass --from tomorrow; rosters stay as they are on ESPN (no waiver pickups or trades); real managers' lineups and
rest days will differ from "best 10 each day".
"""
from __future__ import annotations

import argparse
import datetime as dt
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding="utf-8")

from scripts.sync_espn import CACHE, CATS, Context, check_network, download, load_cache, matchup_table, read_env
from src.features.names import normalize_name
from src.simulation.schedule import season_periods

CAT_NAMES = [c for c, _ in CATS]                                   # PTS 3PM REB AST STL BLK FG% FT% TO
SEASON = "2026-27"
SID = dict(pts="0", tpm="17", tpa="18", reb="6", ast="3", stl="2", blk="1", tov="11", fgm="13", fga="14", ftm="15", fta="16")
NB_R = dict(reb=10.5, ast=12.6, stl=9.7, blk=6.5, tov=33.5, fga=15.6, tpa=23.4, fta=3.1)      # negative-binomial dispersion, from the model's calibration
BLEND_GAMES = 10                                                   # the preseason projection counts as this many games
P_HEALTHY, P_DAY_TO_DAY, TALENT_SD = 0.93, 0.70, 0.10
SPELL_DAYS = 8.0                                                   # mean length of a random injury spell; 7% of days are missed on average
LINEUP_SIZE = 10


def load_absences() -> tuple[dict, pd.DataFrame]:
    """player_id -> first day back (ISO). ESPN's generated list first, then known_absences_manual.csv, which wins (its return_date, or 82 games = all season)."""
    ab: dict = {}
    espn = ROOT / "data/external/known_absences.csv"
    if espn.exists():
        ab.update(pd.read_csv(espn).set_index("player_id").return_date.dropna().to_dict())
    manual, ov = ROOT / "data/external/known_absences_manual.csv", pd.DataFrame()
    if manual.exists():
        ov = pd.read_csv(manual)
        for r in ov.itertuples():
            back = getattr(r, "return_date", None)
            if back is not None and pd.notna(back):
                ab[int(r.player_id)] = str(back)
            elif r.games_out >= 82:
                ab[int(r.player_id)] = "2099-12-31"
    return ab, ov


def load_players(cx: Context, proj: pd.DataFrame, last_team: dict, absences: dict) -> pd.DataFrame:
    """Everyone on an ESPN roster (IR players only if a return date is known): blended per-game line, NBA team, injury tag, return date."""
    rows = []
    slot_by_name = {t["name"]: cx.slot_of[t["id"]] for t in cx.teams}
    for t in cx.data["league"]["teams"]:
        for e in t["roster"]["entries"]:
            p = e["playerPoolEntry"]["player"]
            pid = cx.by_norm.get(normalize_name(p["fullName"]))
            if e["lineupSlotId"] == 13 and pid not in absences:          # on IR with no known return date: assume he does not come back
                continue
            s = next((x for x in p.get("stats", []) if x["id"] == "002027"), None)
            avg, tot = ((s.get("averageStats") or {}), (s.get("stats") or {})) if s else ({}, {})
            gp = float(tot.get("42") or 0)
            row = dict(slot=slot_by_name[t["name"]], name=p["fullName"], pid=pid, gp=gp, inj=p.get("injuryStatus", "ACTIVE"), ret=absences.get(pid))
            for k, sid in SID.items():
                espn_v = float(avg.get(sid) or 0.0)
                prior = float(proj.loc[pid, k]) if (pid in proj.index and k in proj.columns) else espn_v
                row[k] = (gp * espn_v + BLEND_GAMES * prior) / (gp + BLEND_GAMES)
            team = last_team.get(pid) if pid in last_team else (proj.loc[pid, "team_id"] if pid in proj.index else None)
            row["team"] = int(team) if team is not None and not pd.isna(team) else None
            rows.append(row)
    P = pd.DataFrame(rows)
    missing = P[P.team.isna()].name.tolist()
    if missing:
        print(f"note: no NBA team found for {missing}; they are left out of the projection.")
    P = P[P.team.notna()].reset_index(drop=True)
    fg, ft = P.fgm.sum() / P.fga.sum(), P.ftm.sum() / P.fta.sum()
    P["fgi"], P["fti"] = P.fgm - fg * P.fga, P.ftm - ft * P.fta
    z = lambda c, sign=1: sign * (P[c] - P[c].mean()) / P[c].std()
    P["value"] = z("pts") + z("tpm") + z("reb") + z("ast") + z("stl") + z("blk") + z("fgi") + z("fti") + z("tov", -1)    # picks each day's best 10
    return P


def base_avail(r, day: str, today: dt.date) -> float:
    """Chance he can play if he has a game, before random injuries: 0 when known to be out, 0.97 on his first day back, 0.75 for a fresh day-to-day tag."""
    ret = None if r.ret is None or pd.isna(r.ret) else str(r.ret)
    if r.inj == "OUT":
        return 0.9 / P_HEALTHY if (ret is not None and ret <= day) else 0.0
    if ret is not None and ret > day:
        return 0.0
    if r.inj == "DAY_TO_DAY" and (dt.date.fromisoformat(day) - today).days <= 2:
        return P_DAY_TO_DAY / P_HEALTHY
    return 1.0


def make_plays_fn(S, week_idx: int, sims: int, rng):
    """(NBA team id, ISO day) -> None (no game) | True (published game) | bool array over sims (a Cup-week game placed at random, per simulation)."""
    tix, start, pub = S.team_index(), S.spans[week_idx][0].date(), S.masks[week_idx]
    flex = None
    if S.flex_period == week_idx and S.missing:
        T, D = pub.shape
        miss = np.array([S.missing.get(t, 0) for t in S.team_ids])
        w = np.zeros(D)
        w[D - 7:] = np.asarray(S.flex_weights, dtype=float)
        u = rng.random((sims, T, D))
        keys = np.where(w > 0, u ** (1.0 / np.maximum(w, 1e-9)), -1.0)       # weighted sampling without replacement, as in Schedule.team_day_masks
        keys = np.where(pub[None], -1.0, keys)                               # a team cannot play twice in a day
        rank = (-keys).argsort(axis=-1).argsort(axis=-1)
        flex = (rank < miss[None, :, None]) & (keys > 0)

    def fn(team, day):
        j = (dt.date.fromisoformat(day) - start).days
        if team not in tix or j < 0 or j >= pub.shape[1]:
            return None
        i = tix[team]
        if pub[i, j]:
            return True
        return flex[:, i, j] if (flex is not None and flex[:, i, j].any()) else None
    return fn


def simulate_period(P: pd.DataFrame, plays_fn, days: list[str], sims: int, rng, today: dt.date, talent: dict | None = None, health: dict | None = None):
    """
    Per team: simulated sums over `days` of [pts, 3pm, reb, ast, stl, blk, fgm, fga, ftm, fta, tov], and the number of lineup games used.
    Pass the same `talent` and `health` dicts for consecutive periods to carry each player's true-rate draw and any ongoing injury forward.
    """
    def nb(mean, r):
        mean = np.maximum(mean, 1e-6)
        return rng.negative_binomial(r, r / (r + mean), sims)

    if talent is None:
        talent = {i: rng.lognormal(0, TALENT_SD, sims) for i in P.index}
    if health is None:
        health = {i: np.zeros(sims, dtype=bool) for i in P.index}                 # True = in a random injury spell
    q, rec = (1 - P_HEALTHY) / P_HEALTHY / SPELL_DAYS, 1.0 / SPELL_DAYS
    teams = sorted(P.slot.unique())
    roster = {s: list(P[P.slot == s].sort_values("value", ascending=False).iterrows()) for s in teams}
    rem = {s: np.zeros((sims, 11)) for s in teams}
    used = {s: np.zeros(sims) for s in teams}
    for day in days:
        for i in P.index:                                                         # injury spells start and end every day
            u = rng.random(sims)
            health[i] = np.where(health[i], u > rec, u < q)
        for s in teams:
            cand = []
            for i, r in roster[s]:
                g = plays_fn(r.team, day)
                p = base_avail(r, day, today) if g is not None else 0.0
                if p > 0:
                    cand.append((i, r, p, g))
            if not cand:
                continue
            avail = np.stack([(~health[i]) & (rng.random(sims) < p) & (g if isinstance(g, np.ndarray) else True) for i, r, p, g in cand])
            sel = avail & (np.cumsum(avail, axis=0) <= LINEUP_SIZE)               # the best ten available that day; a hurt starter is replaced
            used[s] += sel.sum(0)
            for (i, r, p, g), on in zip(cand, sel):
                if not on.any():
                    continue
                m = talent[i]
                fga = nb(r.fga * m, NB_R["fga"])
                tpa = np.minimum(nb(r.tpa * m, NB_R["tpa"]), fga + 3)
                fga = np.maximum(fga, tpa)
                fg_pct = np.clip(r.fgm / max(r.fga, 1e-6), 0.2, 0.75)
                tp_pct = np.clip(r.tpm / max(r.tpa, 1e-6), 0.1, 0.6)
                ft_pct = np.clip(r.ftm / max(r.fta, 1e-6), 0.4, 0.95)
                fgm = rng.binomial(fga, fg_pct)
                tpm = np.minimum(rng.binomial(tpa, tp_pct), fgm)
                fta = nb(r.fta * m, NB_R["fta"])
                ftm = rng.binomial(fta, ft_pct)
                line = np.stack([2 * fgm + tpm + ftm, tpm, nb(r.reb * m, NB_R["reb"]), nb(r.ast * m, NB_R["ast"]), nb(r.stl * m, NB_R["stl"]),
                                 nb(r.blk * m, NB_R["blk"]), fgm, fga, ftm, fta, nb(r.tov * m, NB_R["tov"])], axis=1)
                rem[s] += line * on[:, None]
    return rem, used


def final_totals(now: pd.Series, R: np.ndarray) -> np.ndarray:
    """Current ESPN totals + simulated remainder -> sims x 9 (PTS 3PM REB AST STL BLK FG% FT% TO)."""
    fgm, fga, ftm, fta = now.fgm + R[:, 6], now.fga + R[:, 7], now.ftm + R[:, 8], now.fta + R[:, 9]
    return np.stack([now.PTS + R[:, 0], now["3PM"] + R[:, 1], now.REB + R[:, 2], now.AST + R[:, 3], now.STL + R[:, 4], now.BLK + R[:, 5],
                     fgm / fga, ftm / fta, now.TO + R[:, 10]], axis=1)


def cats_won(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    w = (A > B).astype(int)
    w[:, 8] = (A[:, 8] < B[:, 8]).astype(int)                      # fewer turnovers wins
    return w


SUFFIXES = {"jr.", "jr", "sr.", "sr", "ii", "iii", "iv"}


def short_name(name: str) -> str:
    parts = name.split()
    last = parts[-2] if len(parts) > 2 and parts[-1].lower() in SUFFIXES else parts[-1]
    return f"{parts[0][0]}. {last}" if len(parts) > 1 else last                # initial + last name, so the two Thompsons stay apart


def roster_games(P: pd.DataFrame, plays_fn, days: list[str], today: dt.date) -> dict:
    """Per fantasy team: [(value, short name, NBA games he can play in the projected days)], best player first. 0 = out or no game."""
    out: dict = {}
    for _, r in P.iterrows():
        n = 0.0
        for d in days:
            g = plays_fn(r.team, d)
            if g is not None and base_avail(r, d, today) > 0:
                n += 1.0 if g is True else float(g.mean())                     # a random Cup game counts as its chance
        out.setdefault(r.slot, []).append((r.value, short_name(r["name"]), round(n, 1), r["name"]))
    result = {}
    for s, v in out.items():
        dup = {sn for _, sn, _, _ in v if sum(sn == x[1] for x in v) > 1}                  # same initial + surname on one team: spell the first name out
        result[s] = [(val, full if sn in dup else sn, n) for val, sn, n, full in sorted(v, key=lambda x: -x[0])]
    return result


def report(a: int, b: int, now: pd.DataFrame, fin: dict, left: dict, names: list[str], rg: dict, future: bool = False) -> None:
    A, B = fin[a], fin[b]
    wa, wb = cats_won(A, B), cats_won(B, A)
    ca, cb = wa.sum(1), wb.sum(1)
    p_win, p_tie = (ca > cb).mean(), (ca == cb).mean()
    na, nb_ = names[a - 1], names[b - 1]
    banked = "" if future else f"now {int(now.loc[a].cat_w)}-{int(now.loc[a].cat_l)}-{int(now.loc[a].cat_t)}   "
    print(f"\n=== {na} vs {nb_} ===   {banked}projected {ca.mean():.1f}-{cb.mean():.1f}")
    for s, nm in ((a, na), (b, nb_)):                                # NBA games each fantasy team has, and who plays them
        n = lambda v: f"{v:g}"
        print(f"games {'in the week' if future else 'left'} - {nm}: {left[s]:.0f} in the lineup of {sum(x for _, _, x in rg[s]):.0f} on the roster | "
              + ", ".join(f"{p} {n(x)}" for _, p, x in rg[s]))
    print(f"chance to win the week:  {na} {100 * p_win:.0f}%   {nb_} {100 * (cb > ca).mean():.0f}%" + (f"   tie {100 * p_tie:.0f}%" if p_tie >= 0.005 else ""))
    rows = []
    for k, cat in enumerate(CAT_NAMES):
        pct = cat in ("FG%", "FT%")
        f = (lambda v: f"{v:.3f}") if pct else (lambda v: f"{v:.0f}")
        row = {"cat": cat}
        if not future:
            row.update({f"{na} now": f(now.loc[a, cat]), f"{nb_} now": f(now.loc[b, cat])})
        row.update({f"{na} {'proj' if future else 'final'}": f(A[:, k].mean()), f"{nb_} {'proj' if future else 'final'}": f(B[:, k].mean()),
                    f"{na} wins": f"{100 * wa[:, k].mean():.0f}%", f"{nb_} wins": f"{100 * wb[:, k].mean():.0f}%"})
        rows.append(row)
    print(pd.DataFrame(rows).to_string(index=False))
    print(f"chance of each final score ({na}-{nb_}): " + finishes(ca, cb))


def finishes(ca: np.ndarray, cb: np.ndarray, floor: float = 0.005) -> str:
    fin_str = pd.Series([f"{x}-{y}" + (f"-{9 - x - y}" if x + y < 9 else "") for x, y in zip(ca, cb)]).value_counts(normalize=True)
    fin_str = fin_str[fin_str >= floor].sort_index(key=lambda idx: [(-int(s.split("-")[0]), int(s.split("-")[1])) for s in idx])
    return "  ".join(f"{s} {100 * p:.0f}%" for s, p in fin_str.items())


def setup(args):
    """Shared by project_week.py and project_season.py: the ESPN download, context, schedule, players and the first day still to play."""
    if args.refresh:
        env = read_env(ROOT / ".env")
        try:
            check_network()
        except SystemExit as e:                                      # same checks as sync_espn.py; this script's "offline" is simply leaving out --refresh
            sys.exit(str(e).replace("Use --offline to work from the last download.", "Run without --refresh to use the last download."))
        print("network ok; downloading your league ...")
        folder = download(env, args.year)
    else:
        folder = CACHE / "latest"
    data = load_cache(folder)
    if "league" not in data or "matchups" not in data:
        sys.exit("The download has no league/matchup data; run scripts/sync_espn.py (or add --refresh) first.")
    fetched = dt.datetime.fromtimestamp((folder / "league.json").stat().st_mtime)
    cx = Context(data, Path(args.db), force=False, verbose=False)
    mt = matchup_table(cx)
    conn = sqlite3.connect(f"file:{args.db}?mode=ro", uri=True)
    S = season_periods(conn, SEASON, n_regular=18, n_playoff=3, fill_unscheduled=True)
    season_start = dt.date.fromisoformat(conn.execute("SELECT MIN(date) FROM game WHERE season = ?", (SEASON,)).fetchone()[0])
    today = season_start + dt.timedelta(days=int(data["league"]["scoringPeriodId"]) - 1)         # ESPN scoring period 1 = the first game day
    live_db = sqlite3.connect(f"file:{ROOT / 'data/processed/live_2026.db'}?mode=ro", uri=True)
    last_team = pd.read_sql_query("SELECT b.player_id, b.team_id FROM live_box_score b JOIN live_game g USING(game_id) ORDER BY g.date", live_db).groupby("player_id").team_id.last().to_dict()
    proj = pd.read_parquet(ROOT / "data/processed/projections_2026.parquet").set_index("player_id")
    absences, overrides = load_absences()
    P = load_players(cx, proj, last_team, absences)
    return dict(data=data, cx=cx, mt=mt, S=S, today=today, fetched=fetched, P=P, overrides=overrides)


def week_days(S, week: int, first: dt.date) -> list[str]:
    last_day = S.spans[week - 1][1].date()
    return [(first + dt.timedelta(days=i)).isoformat() for i in range((last_day - first).days + 1)]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1], formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--refresh", action="store_true", help="check the network and download a fresh ESPN copy first")
    ap.add_argument("--from", dest="start", default=None, help="first day still to be played, YYYY-MM-DD (default: ESPN's current scoring-period day)")
    ap.add_argument("--week", type=int, default=None, help="project this matchup week instead of the current one (a future week is projected in full)")
    ap.add_argument("--next", action="store_true", help="shortcut for next week")
    ap.add_argument("--team", default=None, help="only matchups involving this team (e.g. Bilal)")
    ap.add_argument("--sims", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--year", type=int, default=2027)
    ap.add_argument("--db", default=str(ROOT / "data/processed/nba.db"))
    args = ap.parse_args()
    E = setup(args)
    cx, mt, S, today, P, fetched = E["cx"], E["mt"], E["S"], E["today"], E["P"], E["fetched"]
    cur_week = int(E["data"]["league"]["status"]["currentMatchupPeriod"])
    week = cur_week + 1 if args.next else (args.week or cur_week)
    if week < cur_week:
        sys.exit(f"Week {week} is already over; the current week is {cur_week}.")
    future = week > cur_week
    live = mt[mt.week == week]
    if live.empty:
        sys.exit(f"ESPN has no matchups for week {week} in this download (playoffs or off-week?).")
    live = live.set_index("slot").copy()
    for c in CAT_NAMES + ["fgm", "fga", "ftm", "fta", "cat_w", "cat_l", "cat_t"]:
        live[c] = live[c].fillna(0) if c in live else 0                # a week that has not started has nothing banked
    start_end = S.spans[week - 1]
    first = dt.date.fromisoformat(args.start) if args.start else (start_end[0].date() if future else today)
    days = week_days(S, week, first)
    if not days:
        sys.exit(f"Week {week} ended on {start_end[1].date()}; nothing left to project.")
    rng = np.random.default_rng(args.seed)
    plays_fn = make_plays_fn(S, week - 1, args.sims, rng)
    j0, j1 = (first - start_end[0].date()).days, (start_end[1].date() - start_end[0].date()).days
    published = int(S.masks[week - 1][:, max(j0, 0):j1 + 1].sum() // 2)
    extra = (sum(S.missing.values()) // 2) if (S.flex_period == week - 1 and S.missing) else 0
    if future:
        print(f"Week {week} ({start_end[0]:%b %d} - {start_end[1]:%b %d}): full-week projection, nothing banked yet. Rosters, injuries and return dates are as of the ESPN download from {fetched:%b %d %H:%M}.")
    else:
        print(f"Week {week} ({start_end[0]:%b %d} - {start_end[1]:%b %d}). ESPN download from {fetched:%b %d %H:%M}; its totals are taken to include games through {(first - dt.timedelta(days=1)).isoformat()}.")
    print(f"Projecting {days[0]} .. {days[-1]}: {len(days)} day(s), {published} published NBA games" + (f" + {extra} Cup games not yet scheduled, placed at random" if extra else "") + f". {args.sims:,} simulations.")
    for r in E["overrides"].itertuples():                            # hand-set absences win over ESPN: say so, and flag when ESPN's tag disagrees
        back = getattr(r, "return_date", None)
        row = P[P.pid == r.player_id]
        if back is None or pd.isna(back) or str(back) <= days[0] or row.empty:
            continue
        tag = row.iloc[0].inj
        print(f"manual absence: {r.name} out until {back}" + ("" if tag == "OUT" else f" (ESPN now lists him {tag}: if that is an update, tell me)"))
    rem, used = simulate_period(P, plays_fn, days, args.sims, rng, today)
    left = {s: used[s].mean() for s in used}
    rg = roster_games(P, plays_fn, days, today)
    fin = {s: final_totals(live.loc[s], rem[s]) for s in live.index}
    pairs = [(int(s), int(r.opp_slot)) for s, r in live.iterrows() if r.side == "HOME"]
    if args.team:
        want = next((i + 1 for i, n in enumerate(cx.names) if n.lower() == args.team.lower()), None)
        if want is None:
            sys.exit(f"Unknown team {args.team!r}; choose from {cx.names}.")
        pairs = [p for p in pairs if want in p]
    pairs = [(b, a) if b == 1 else (a, b) for a, b in pairs]         # you are listed first in your own matchup
    pairs.sort(key=lambda p: (p[0] != 1, p))                         # and your matchup comes first
    for a, b in pairs:
        report(a, b, live, fin, left, cx.names, rg, future)


if __name__ == "__main__":
    main()
