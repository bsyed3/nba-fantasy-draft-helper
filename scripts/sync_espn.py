"""
One command that pulls everything from your ESPN fantasy league and brings the tool up to date.

    python scripts/sync_espn.py                # check the network, download, then update rosters, positions, matchups, standings and players
    python scripts/sync_espn.py --dry-run      # same, but write nothing to nba.db or data/external (the raw download is still cached)
    python scripts/sync_espn.py --offline      # no network: rebuild everything from the last download in data/espn_cache/latest

What it does, in order:
  1. Network check (stops with a plain-language reason if this machine can't reach ESPN safely), then downloads the league, matchups,
     activity feed (adds, drops, trades) and player pool into data/espn_cache/ (gitignored: it holds other managers' names and ids).
  2. Rosters: compares every ESPN roster with nba.db and records the differences as roster moves (undoable; --backup copies nba.db first).
  3. Positions: ESPN's eligibility and primary (default) position -> data/external/positions.csv (overwritten each run; ESPN is the source).
  4. Writes data/external/espn_matchups.csv, matchup_results.csv, espn_standings.csv, espn_transactions.csv and espn_players.csv, and
     prints the standings, this week's matchups, your next opponents and the free agents on the rise.

Credentials come from the gitignored .env in the project folder (never printed):
    leagueID = "123456789"
    SWID = "{XXXXXXXX-XXXX-XXXX-XXXX-XXXXXXXXXXXX}"
    espn_s2 = "...long cookie..."      (browser F12 > Application > Cookies > espn.com, while logged in)

Network: fantasy.espn.com must be reachable with a normal certificate. Some managed networks (a work network) re-sign that host with a local
CA that Python rejects; the script detects this up front and stops. It never turns certificate checking off. Use a phone hotspot or home
network for the download, then switch back: --offline works anywhere.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import shutil
import socket
import sqlite3
import ssl
import sys
import threading
import time
from pathlib import Path

import certifi
import numpy as np
import pandas as pd
import requests
import urllib3.util.connection as _u3
from scipy.optimize import linear_sum_assignment

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding="utf-8")

from scripts.fetch_espn_positions import SLOT_NAMES
from src.draft import store
from src.features.names import normalize_name

# Phone hotspots often hand out NAT64 IPv6 addresses (64:ff9b::...) first; they time out (8s each) before the working IPv4 address is tried,
# which made every request look hung. This script talks to ESPN over IPv4 only.
_u3.allowed_gai_family = lambda: socket.AF_INET

HOST = "lm-api-reads.fantasy.espn.com"
CACHE = ROOT / "data" / "espn_cache"
EXT = ROOT / "data" / "external"
UA = {"User-Agent": "Mozilla/5.0"}
LEAGUE = "ESPN 10-team"
IR_SLOT = 13                                                                  # lineup slot id of the injured-reserve spot
POSITION_BY_DEFAULT_ID = {1: "PG", 2: "SG", 3: "SF", 4: "PF", 5: "C"}         # ESPN defaultPositionId = the primary position (the center cap counts this)
CATS = [("PTS", "0"), ("3PM", "17"), ("REB", "6"), ("AST", "3"), ("STL", "2"), ("BLK", "1"), ("FG%", "19"), ("FT%", "20"), ("TO", "11")]
LOWER_IS_BETTER = {"TO"}
ME = 1                                                                        # your draft slot
ACTIVITY_TYPES = {178: "FA_ADD", 180: "WAIVER_ADD", 179: "DROP", 181: "DROP", 239: "DROP", 244: "TRADE"}


# ----------------------------------------------------------------------------------------------------------------------------- credentials
def read_env(path: Path) -> dict[str, str]:
    if not path.exists():
        sys.exit(f"No {path.name} found in {path.parent}. Create it with leagueID, SWID and espn_s2 (see this script's docstring).")
    env: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        m = re.match(r"\s*(?:export\s+)?([A-Za-z0-9_]+)\s*=\s*(.*?)\s*$", line)
        if m:
            env[m.group(1).lower()] = m.group(2).strip().strip('"').strip("'")
    missing = [k for k in ("leagueid", "swid", "espn_s2") if not env.get(k)]
    if missing:
        sys.exit(f".env is missing: {missing}")
    if not (env["swid"].startswith("{") and env["swid"].endswith("}") and env["swid"].count("-") == 4):
        sys.exit("The SWID in .env is not shaped like {XXXXXXXX-XXXX-XXXX-XXXX-XXXXXXXXXXXX}; re-copy it from the browser cookie, braces included.")
    if not env["leagueid"].isdigit():
        sys.exit("leagueID in .env must be digits only.")
    return env


# ----------------------------------------------------------------------------------------------------------------------------- network check
def _issuer_of_unverified_cert() -> str:
    """Best effort: who signed the certificate this network hands out for ESPN (for the error message only; nothing is trusted)."""
    try:
        ctx = ssl.create_default_context()
        ctx.check_hostname, ctx.verify_mode = False, ssl.CERT_NONE
        addr = socket.getaddrinfo(HOST, 443, socket.AF_INET, socket.SOCK_STREAM)[0][4]
        with socket.create_connection(addr, timeout=8) as s, ctx.wrap_socket(s, server_hostname=HOST) as t:
            der = t.getpeercert(binary_form=True)
        from cryptography import x509                                                       # present when pip pulled it in; optional
        return x509.load_der_x509_certificate(der).issuer.rfc4514_string()
    except Exception:
        return "unknown"


def check_network(deadline: float = 20.0) -> None:
    """Exit with a plain-language reason unless this machine can open a verified TLS connection to ESPN's fantasy host (same CA list `requests` uses)."""
    out: dict = {}

    def probe() -> None:
        try:
            ctx = ssl.create_default_context(cafile=certifi.where())
            addr = socket.getaddrinfo(HOST, 443, socket.AF_INET, socket.SOCK_STREAM)[0][4]          # IPv4 only, see the note at the top of the file
            with socket.create_connection(addr, timeout=8) as s, ctx.wrap_socket(s, server_hostname=HOST):
                out["ok"] = True
        except ssl.SSLCertVerificationError as e:
            out["ssl"] = str(e)
        except Exception as e:                                                              # DNS failure, timeout, no route, ...
            out["err"] = f"{type(e).__name__}: {e}"

    t = threading.Thread(target=probe, daemon=True)
    t.start()
    t.join(deadline)                                                                        # DNS lookups can ignore socket timeouts, so bound the whole check
    if out.get("ok"):
        return
    if "ssl" in out:
        sys.exit(f"STOP: this network re-signs {HOST} (certificate signed by: {_issuer_of_unverified_cert()}), so the connection cannot be trusted.\n"
                 "Switch to a phone hotspot or home network and run again (certificate checking is never turned off). "
                 "Use --offline to work from the last download.")
    sys.exit(f"STOP: cannot reach {HOST} ({out.get('err', f'no answer within {deadline:.0f}s')}). Check the internet connection, "
             "or use --offline to work from the last download.")


# ----------------------------------------------------------------------------------------------------------------------------- download
def api(env: dict, year: int, params: dict, headers: dict | None = None, path: str = "") -> dict:
    url = f"https://{HOST}/apis/v3/games/fba/seasons/{year}/segments/0/leagues/{env['leagueid']}{path}"
    r = requests.get(url, params=params, cookies={"espn_s2": env["espn_s2"], "SWID": env["swid"]}, headers={**UA, **(headers or {})}, timeout=(10, 90))
    if r.status_code in (401, 403):
        sys.exit(f"ESPN refused the login (HTTP {r.status_code}): the espn_s2 / SWID cookies in .env have expired or are wrong. Log in to ESPN and re-copy them.")
    r.raise_for_status()
    return r.json()


def download_activity(env: dict, year: int, page: int = 100, max_pages: int = 30) -> dict:
    """Every add / drop / trade message in the league, newest first, paged until ESPN runs out."""
    topics: dict[str, dict] = {}
    for i in range(max_pages):
        filt = {"topics": {"filterType": {"value": ["ACTIVITY_TRANSACTIONS"]}, "limit": page, "limitPerMessageSet": {"value": page}, "offset": i * page,
                           "sortMessageDate": {"sortPriority": 1, "sortAsc": False}, "sortFor": {"sortPriority": 2, "sortAsc": False},
                           "filterIncludeMessageTypeIds": {"value": list(ACTIVITY_TYPES)}}}
        got = api(env, year, {"view": "kona_league_communication"}, {"x-fantasy-filter": json.dumps(filt)}, path="/communication/").get("topics", [])
        for t in got:
            topics[t["id"]] = t
        if len(got) < page:
            break
    return {"topics": list(topics.values())}


def download_players_by_id(env: dict, year: int, ids: list[int]) -> dict:
    out = []
    for i in range(0, len(ids), 100):
        chunk = ids[i:i + 100]
        filt = {"players": {"filterIds": {"value": chunk}, "limit": len(chunk)}}
        out += api(env, year, {"view": "kona_player_info"}, {"x-fantasy-filter": json.dumps(filt)}).get("players", [])
    return {"players": out}


def espn_names(data: dict) -> dict[int, str]:
    """ESPN player id -> name from every payload that carries players."""
    out: dict[int, str] = {}
    for t in data.get("league", {}).get("teams", []):
        for e in t["roster"]["entries"]:
            p = e["playerPoolEntry"]["player"]
            out[p["id"]] = p["fullName"]
    for key in ("players", "players_extra"):
        for item in data.get(key, {}).get("players", []):
            out[item["player"]["id"]] = item["player"]["fullName"]
    aliases = EXT / "espn_player_aliases.csv"                                              # ids ESPN will not look up (outside its pool): espn_id,name
    if aliases.exists():
        for r in pd.read_csv(aliases).itertuples():
            out.setdefault(int(r.espn_id), r.name)
    return out


def download(env: dict, year: int) -> Path:
    """Fetch every view into data/espn_cache/latest (replacing the last download only once this one is complete). The league view is required; the rest warn."""
    folder = CACHE / "latest.tmp"
    shutil.rmtree(folder, ignore_errors=True)
    folder.mkdir(parents=True, exist_ok=True)
    player_filter = {"players": {"filterStatus": {"value": ["FREEAGENT", "WAIVERS", "ONTEAM"]}, "limit": 700,
                                 "sortDraftRanks": {"sortPriority": 100, "sortAsc": True, "value": "STANDARD"}}}
    fetchers = [                                                                            # (file, callable, required)
        ("league", lambda: api(env, year, {"view": ["mTeam", "mRoster", "mSettings", "mStandings"]}), True),
        ("matchups", lambda: api(env, year, {"view": ["mMatchup", "mMatchupScore"]}), False),
        ("players", lambda: api(env, year, {"view": "kona_player_info"}, {"x-fantasy-filter": json.dumps(player_filter)}), False),
        ("activity", lambda: download_activity(env, year), False),
    ]
    data: dict = {}
    for name, fn, required in fetchers:
        t0 = time.time()
        try:
            data[name] = fn()
        except Exception as e:
            if required:
                sys.exit(f"Download of the required '{name}' view failed: {type(e).__name__}: {str(e)[:200]}")
            print(f"  warning: optional '{name}' view failed ({type(e).__name__}: {str(e)[:120]}); continuing without it")
            continue
        (folder / f"{name}.json").write_text(json.dumps(data[name]), encoding="utf-8")
        print(f"  {name:13s} {(folder / f'{name}.json').stat().st_size / 1024:8.0f} KB  {time.time() - t0:4.1f}s")
    known = espn_names(data)                                                                # players in the activity feed that the pool no longer lists
    unknown = sorted({m["targetId"] for t in data.get("activity", {}).get("topics", []) for m in t["messages"]} - set(known))
    if unknown:
        try:
            data["players_extra"] = download_players_by_id(env, year, unknown)
            (folder / "players_extra.json").write_text(json.dumps(data["players_extra"]), encoding="utf-8")
            print(f"  players_extra  looked up {len(data['players_extra']['players'])} of {len(unknown)} players that only appear in the activity feed")
        except Exception as e:
            print(f"  warning: could not look up {len(unknown)} players from the activity feed ({type(e).__name__}); they stay as ids")
    latest = CACHE / "latest"
    shutil.rmtree(latest, ignore_errors=True)
    folder.rename(latest)
    return latest


def load_cache(folder: Path) -> dict:
    if not folder.exists():
        sys.exit(f"No download found at {folder}; run once on a network that can reach ESPN (without --offline).")
    return {p.stem: json.loads(p.read_text(encoding="utf-8")) for p in folder.glob("*.json")}


# ----------------------------------------------------------------------------------------------------------------------------- derive (offline)
class Context:
    """nba.db rosters + names, the ESPN <-> nba.db player match, and the ESPN team id -> draft slot match."""

    def __init__(self, data: dict, db: Path, force: bool, verbose: bool = True):
        self.data = data
        conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
        self.state = store.load_state(conn, LEAGUE)
        self.names = store.get_team_names(conn, LEAGUE, 10)
        self.pname = dict(conn.execute("SELECT player_id, name FROM player").fetchall())
        last = dict(conn.execute("SELECT player_id, COALESCE(last_year, 0) FROM player").fetchall())
        conn.close()
        self.by_norm: dict[str, int] = {}
        for pid, nm in self.pname.items():
            k = normalize_name(nm)
            if k not in self.by_norm or last[pid] > last[self.by_norm[k]]:
                self.by_norm[k] = pid                                                         # same spelling twice: keep the more recent player
        self.active, self.ir = self.state._by_team()
        self.ours = {s: set(self.active[s]) | set(self.ir[s]) for s in range(1, 11)}
        self.ename = espn_names(data)
        banned_file = EXT / "banned_players.csv"
        self.banned = set(pd.read_csv(banned_file).player_id) if banned_file.exists() else set()      # banned in this league: never shown as a target
        self.unmatched: set[str] = set()
        self.teams = data["league"]["teams"]
        for t in self.teams:
            entries = t["roster"]["entries"]
            t["a_ids"] = {self.pid(e["playerPoolEntry"]["player"]["id"]) for e in entries if e["lineupSlotId"] != IR_SLOT} - {None}
            t["i_ids"] = {self.pid(e["playerPoolEntry"]["player"]["id"]) for e in entries if e["lineupSlotId"] == IR_SLOT} - {None}
        overlap = np.array([[len((t["a_ids"] | t["i_ids"]) & self.ours[s]) for s in range(1, 11)] for t in self.teams])
        rows, cols = linear_sum_assignment(-overlap)                                          # best one-to-one match of ESPN teams to draft slots
        self.slot_of = {self.teams[r]["id"]: int(c) + 1 for r, c in zip(rows, cols)}
        self.team_by_id = {t["id"]: t for t in self.teams}
        if verbose:
            print("ESPN team -> draft slot (matched by roster overlap):")
            for r, c in zip(rows, cols):
                t = self.teams[r]
                flag = "" if overlap[r, c] >= 8 else "   <-- LOW OVERLAP"
                print(f"  '{t['name']}' -> {c + 1} {self.names[c]}  (shared {overlap[r, c]} of {len(t['a_ids'] | t['i_ids'])}){flag}")
        self.low_overlap = bool((overlap[rows, cols] < 8).any())
        if self.low_overlap and not force:
            sys.exit("Team matching looks wrong (a team shares fewer than 8 players with its draft slot). Nothing was changed; re-run with --force only if you are sure.")

    def pid(self, espn_id: int) -> int | None:
        nm = self.ename.get(espn_id)
        pid = self.by_norm.get(normalize_name(nm)) if nm else None
        if pid is None:
            self.unmatched.add(nm or f"id {espn_id}")
        return pid

    def slot_name(self, espn_team_id: int) -> str:
        s = self.slot_of.get(espn_team_id)
        return self.names[s - 1] if s else f"ESPN team {espn_team_id}"


def write_csv(df: pd.DataFrame, path: Path, args, what: str) -> None:
    if args.dry_run:
        print(f"  (dry run) would write {len(df)} rows -> {path.relative_to(ROOT)}  [{what}]")
        return
    df.to_csv(path, index=False)
    print(f"  wrote {len(df)} rows -> {path.relative_to(ROOT)}  [{what}]")


# ---- rosters
def sync_rosters(cx: Context, args) -> None:
    print("\n== ROSTERS: ESPN vs nba.db ==")
    main_moves, ir_moves, lines = [], [], []
    for t in cx.teams:
        s = cx.slot_of[t["id"]]
        esp_all, esp_ir = t["a_ids"] | t["i_ids"], t["i_ids"]
        adds, drops = esp_all - cx.ours[s], cx.ours[s] - esp_all
        to_ir = (esp_ir & set(cx.active[s])) | (esp_ir & adds)                               # a player added AND on IR needs a second step
        from_ir = (t["a_ids"] & set(cx.ir[s])) - adds
        main_moves += [(s, p, "drop") for p in drops] + [(s, p, "add") for p in adds]
        ir_moves += [(s, p, "ir") for p in to_ir] + [(s, p, "activate") for p in from_ir]
        for p in sorted(adds): lines.append(f"  {cx.names[s - 1]}: add {cx.pname[p]}")
        for p in sorted(drops): lines.append(f"  {cx.names[s - 1]}: drop {cx.pname[p]}")
        for p in sorted(to_ir): lines.append(f"  {cx.names[s - 1]}: {cx.pname[p]} to IR")
        for p in sorted(from_ir): lines.append(f"  {cx.names[s - 1]}: {cx.pname[p]} off IR")
    if cx.unmatched:
        print(f"  could not match these ESPN names to nba.db (ignored): {sorted(cx.unmatched)}")
    if not lines:
        print("  all ten rosters already match ESPN.")
        return
    print("\n".join(lines))
    if args.dry_run:
        print(f"  (dry run) {len(lines)} change(s) not recorded.")
        return
    if len(lines) > 30 and not args.force:
        sys.exit(f"{len(lines)} roster changes is more than expected; nothing recorded. Look at the list above and re-run with --force if it is right.")
    db = Path(args.db)
    if args.backup:                                                                          # off by default: moves are append-only and store.undo_move takes a sync back
        shutil.copy2(db, db.parent.parent.parent.parent / "nba_before_espn_sync.db")        # one file, overwritten each time
    w = sqlite3.connect(db)
    label = f"ESPN sync {dt.date.today():%Y-%m-%d}"
    try:
        if main_moves:
            store.record_transaction(w, store.load_state(w, LEAGUE), main_moves, label)
        if ir_moves:
            store.record_transaction(w, store.load_state(w, LEAGUE), ir_moves, label + " (IR)")        # IR moves go second: record_transaction orders adds last
    except ValueError as e:
        sys.exit(f"nba.db refused a move ({e}); nothing further was written.")
    finally:
        w.close()
    print(f"  recorded {len(main_moves) + len(ir_moves)} move(s) (undo: store.undo_move, or the Free agents page).")


# ---- positions
def sync_positions(cx: Context, args) -> None:
    print("\n== POSITIONS: ESPN eligibility and primary position ==")
    players = [e["playerPoolEntry"]["player"] for t in cx.teams for e in t["roster"]["entries"]]
    for key in ("players", "players_extra"):
        players += [i["player"] for i in cx.data.get(key, {}).get("players", [])]
    rows = {}
    for p in players:
        slots = [SLOT_NAMES[s] for s in p.get("eligibleSlots", []) if s in SLOT_NAMES]
        primary = POSITION_BY_DEFAULT_ID.get(p.get("defaultPositionId"))
        pid = cx.by_norm.get(normalize_name(p["fullName"]))
        if pid and slots:
            rows[pid] = dict(player_id=pid, name=cx.pname[pid], eligibility=",".join(slots), primary=primary or slots[0])
    new = pd.DataFrame(list(rows.values()))
    path = EXT / "positions.csv"
    old = pd.read_csv(path) if path.exists() else pd.DataFrame(columns=["player_id", "name", "eligibility"])
    first_listed = old.eligibility.str.split(",").str[0]                                     # the old file had no primary column; ESPN sorts the list by slot id
    old["primary"] = old["primary"].fillna(first_listed) if "primary" in old else first_listed
    merged = pd.concat([new, old[~old.player_id.isin(new.player_id)]], ignore_index=True)
    cmp = old.merge(new, on="player_id", suffixes=("_old", "_new"))
    elig_changed = cmp[cmp.eligibility_old != cmp.eligibility_new]
    prim_changed = cmp[(cmp.eligibility_old == cmp.eligibility_new) & (cmp.primary_old != cmp.primary_new)]
    changed = pd.concat([elig_changed, prim_changed])
    print(f"  {len(new)} players from ESPN ({len(merged) - len(new)} older rows kept for players ESPN no longer lists)")
    print(f"  {len(elig_changed)} eligibility changes, {len(prim_changed)} primary-position-only changes vs the old file")
    mine = cx.ours[ME]
    tag = lambda r: "  <-- yours" if r.player_id in mine else ""
    for r in elig_changed.sort_values("player_id").head(30).itertuples():
        print(f"    eligibility  {r.name_new:24s} {r.eligibility_old:9s} -> {r.eligibility_new:9s} (primary {r.primary_new}){tag(r)}")
    to_c, from_c = prim_changed[prim_changed.primary_new == "C"], prim_changed[prim_changed.primary_old == "C"]
    print(f"    now counted as centers (ESPN default position is C): {', '.join(to_c.name_new) or 'none'}")
    print(f"    no longer counted as centers: {', '.join(from_c.name_new) or 'none'}")
    cnt = {s: sum(rows.get(p, {}).get("primary") == "C" for p in cx.ours[s] - set(cx.ir[s])) for s in range(1, 11)}
    print("  centers on each active roster (primary = C; the cap is 4): " + ", ".join(f"{cx.names[s - 1]} {n}" for s, n in cnt.items()))
    write_csv(merged[["player_id", "name", "eligibility", "primary"]].sort_values("player_id"), path, args, "eligibility + primary position")
    if len(changed):
        print("  note: projections carry eligibility from the last build; run  python scripts/build_projections.py --reuse-backtest && python scripts/build_library.py  to apply it.")


# ---- matchups, standings
def team_label(cx: Context, espn_id: int) -> tuple[int, str]:
    return cx.slot_of[espn_id], cx.slot_name(espn_id)


def matchup_table(cx: Context) -> pd.DataFrame:
    cur = cx.data["league"]["status"]["currentMatchupPeriod"]
    rows = []
    for m in cx.data["matchups"]["schedule"]:
        wk = m["matchupPeriodId"]
        for side, other in (("home", "away"), ("away", "home")):
            if side not in m or other not in m or "teamId" not in m[side] or "teamId" not in m[other]:
                continue
            t, o = m[side], m[other]
            slot, team = team_label(cx, t["teamId"])
            oslot, opp = team_label(cx, o["teamId"])
            status = "final" if wk < cur else "live" if wk == cur else "upcoming"
            row = dict(week=wk, status=status, slot=slot, team=team, opp_slot=oslot, opp=opp, winner=m.get("winner"), side=side.upper())
            cs = t.get("cumulativeScore") or {}
            if status != "upcoming" and cs.get("scoreByStat"):
                row.update(cat_w=cs.get("wins"), cat_l=cs.get("losses"), cat_t=cs.get("ties"))
                for c, sid in CATS:
                    sc = cs["scoreByStat"].get(sid, {})
                    row[c], row[c + "_res"] = sc.get("score"), sc.get("result")
                for k, sid in (("fgm", "13"), ("fga", "14"), ("ftm", "15"), ("fta", "16")):
                    row[k] = cs["scoreByStat"].get(sid, {}).get("score")
            rows.append(row)
    return pd.DataFrame(rows).sort_values(["week", "slot"]).reset_index(drop=True)


def standings_table(cx: Context, mt: pd.DataFrame) -> pd.DataFrame:
    fin = mt[mt.status == "final"]
    rows = []
    for s in range(1, 11):
        f = fin[fin.slot == s]
        won = ((f.winner == "HOME") & (f.side == "HOME")) | ((f.winner == "AWAY") & (f.side == "AWAY"))
        decided = f.winner.isin(["HOME", "AWAY"])
        r = dict(slot=s, team=cx.names[s - 1], espn_team=next((t["name"] for t in cx.teams if cx.slot_of[t["id"]] == s), ""),
                 weeks=len(f), W=int(won.sum()), L=int((decided & ~won).sum()), T=int((~decided).sum()),
                 cat_W=int(f.cat_w.sum()), cat_L=int(f.cat_l.sum()), cat_T=int(f.cat_t.sum()))
        n = max(len(f), 1)
        for c, _ in CATS:
            if c not in ("FG%", "FT%"):
                r[c] = f[c].sum() / n                                                          # per-week average
        r["FG%"] = f.fgm.sum() / f.fga.sum() if f.fga.sum() else np.nan
        r["FT%"] = f.ftm.sum() / f.fta.sum() if f.fta.sum() else np.nan
        rows.append(r)
    st = pd.DataFrame(rows)
    for c, _ in CATS:                                                                           # 1 = best in the league
        st["rk_" + c] = st[c].rank(ascending=c in LOWER_IS_BETTER, method="min").astype("Int64")
    st["cat_win%"] = st.cat_W / (st.cat_W + st.cat_L + st.cat_T).replace(0, np.nan)
    return st.sort_values(["W", "cat_win%"], ascending=False).reset_index(drop=True)


def report_matchups(cx: Context, mt: pd.DataFrame, st: pd.DataFrame, args) -> None:
    print("\n== MATCHUPS AND STANDINGS ==")
    cur = cx.data["league"]["status"]["currentMatchupPeriod"]
    write_csv(mt, EXT / "espn_matchups.csv", args, "every matchup period: opponent + category results")
    fin = mt[(mt.status == "final") & (mt.side == "HOME") & mt.winner.isin(["HOME", "AWAY"])]
    res = pd.DataFrame([dict(week=r.week, winner_slot=r.slot if r.winner == "HOME" else r.opp_slot, winner=r.team if r.winner == "HOME" else r.opp,
                             loser_slot=r.opp_slot if r.winner == "HOME" else r.slot, loser=r.opp if r.winner == "HOME" else r.team) for r in fin.itertuples()])
    write_csv(res, EXT / "matchup_results.csv", args, "decided matchups (winner / loser)")
    write_csv(st, EXT / "espn_standings.csv", args, "record, category record, per-week averages and category ranks")
    show = st[["team", "W", "L", "T", "cat_W", "cat_L", "cat_T"] + [f"rk_{c}" for c, _ in CATS]].copy()
    show.columns = ["team", "W", "L", "T", "catW", "catL", "catT"] + [c for c, _ in CATS]
    print(f"\nStandings after {int(st.weeks.max())} completed weeks (category columns = rank 1-10 on per-week averages, 1 = best):")
    print(show.to_string(index=False))
    print(f"\nWeek {cur} so far (live):")
    live = mt[(mt.status == "live") & (mt.side == "HOME")]
    for r in live.itertuples():
        print(f"  {r.team:8s} {int(r.cat_w)}-{int(r.cat_l)}-{int(r.cat_t)} {r.opp}")
    mine = mt[(mt.status == "live") & (mt.slot == ME)]
    if len(mine):
        r = mine.iloc[0]
        o = mt[(mt.status == "live") & (mt.slot == r.opp_slot)].iloc[0]
        print(f"\nYour week-{cur} matchup vs {r.opp}: {int(r.cat_w)}-{int(r.cat_l)}-{int(r.cat_t)}")
        for c, _ in CATS:
            fmt = (lambda v: f"{v:.3f}") if c in ("FG%", "FT%") else (lambda v: f"{v:.0f}")
            print(f"  {c:4s} {fmt(r[c]):>7s}  vs {fmt(o[c]):>7s}   {r[c + '_res'] or '-'}")
    nxt = mt[(mt.status == "upcoming") & (mt.slot == ME)].sort_values("week").head(4)
    print("\nYour next opponents (their category ranks, 1 = best):")
    for r in nxt.itertuples():
        t = st[st.slot == r.opp_slot].iloc[0]
        strong = [c for c, _ in CATS if t["rk_" + c] <= 3]
        weak = [c for c, _ in CATS if t["rk_" + c] >= 8]
        print(f"  week {r.week:2d}: {r.opp:8s} record {t.W}-{t.L}  strong in {strong or '-'}  weak in {weak or '-'}")


# ---- transactions
def report_transactions(cx: Context, args) -> None:
    print("\n== TRANSACTIONS (ESPN activity feed) ==")
    topics = cx.data.get("activity", {}).get("topics", [])
    if not topics:
        print("  no activity feed in this download.")
        return
    rows = []
    for t in topics:
        for m in t["messages"]:
            kind = ACTIVITY_TYPES.get(m["messageTypeId"])
            if not kind:
                continue
            tid = m["from"] if kind == "TRADE" else m["for"] if m["messageTypeId"] == 239 else m["to"]
            nm = cx.ename.get(m["targetId"], f"id {m['targetId']}")
            rows.append(dict(date=dt.datetime.fromtimestamp(m["date"] / 1000).strftime("%Y-%m-%d %H:%M"), action=kind, slot=cx.slot_of.get(tid),
                             team=cx.slot_name(tid), to_team=cx.slot_name(m["to"]) if kind == "TRADE" else "", player=nm, player_id=cx.by_norm.get(normalize_name(nm))))
    tx = pd.DataFrame(rows).sort_values("date", ascending=False).reset_index(drop=True)
    write_csv(tx, EXT / "espn_transactions.csv", args, "every add / drop / trade since the season began")
    print(f"  {len(tx)} events from {tx.date.min()} to {tx.date.max()}: " + ", ".join(f"{k} {v}" for k, v in tx.action.value_counts().items()))
    by_team = tx.groupby("team").action.apply(lambda a: (a.isin(["FA_ADD", "WAIVER_ADD"])).sum()).sort_values(ascending=False)
    print("  pickups per team: " + ", ".join(f"{k} {v}" for k, v in by_team.items()))
    week_ago = (dt.datetime.now() - dt.timedelta(days=7)).strftime("%Y-%m-%d")
    print("  last 7 days:")
    for r in tx[tx.date >= week_ago].itertuples():
        print(f"    {r.date}  {r.team:8s} {r.action.lower():10s} {r.player}" + (f"  (to {r.to_team})" if r.to_team else ""))


# ---- players
def stat_line(p: dict, key: str) -> tuple[dict, dict]:
    for s in p.get("stats", []):
        if s["id"] == key:
            return s.get("averageStats") or {}, s.get("stats") or {}
    return {}, {}


def report_players(cx: Context, args) -> None:
    print("\n== PLAYERS (ESPN pool: ownership trends, eligibility, season and last-15 averages) ==")
    owner = {e["playerPoolEntry"]["player"]["id"]: cx.slot_of[t["id"]] for t in cx.teams for e in t["roster"]["entries"]}
    items = {i["player"]["id"]: i for i in cx.data.get("players", {}).get("players", [])}
    for t in cx.teams:                                                                          # rostered players outside the pool's top 700
        for e in t["roster"]["entries"]:
            items.setdefault(e["playerPoolEntry"]["player"]["id"], {"player": e["playerPoolEntry"]["player"], "status": "ONTEAM"})
    rows = []
    for eid, it in items.items():
        p = it["player"]
        sa, st_ = stat_line(p, "002027")
        la, _ = stat_line(p, "022027")
        own = p.get("ownership") or {}
        slots = ",".join(SLOT_NAMES[s] for s in p.get("eligibleSlots", []) if s in SLOT_NAMES)
        r = dict(espn_id=eid, player_id=cx.by_norm.get(normalize_name(p["fullName"])), name=p["fullName"], owner_slot=owner.get(eid, 0),
                 owner=cx.names[owner[eid] - 1] if eid in owner else "FREE", status=it.get("status"), pro_team_id=p.get("proTeamId"),
                 eligibility=slots, primary=POSITION_BY_DEFAULT_ID.get(p.get("defaultPositionId")), injury=p.get("injuryStatus"),
                 pct_owned=own.get("percentOwned"), pct_change=own.get("percentChange"), pct_started=own.get("percentStarted"),
                 espn_adp=own.get("averageDraftPosition"), gp=st_.get("42"), mpg=sa.get("28"))
        for c, sid in CATS:
            r[c] = sa.get(sid)
            r["l15_" + c] = la.get(sid)
        r["l15_mpg"] = la.get("28")
        rows.append(r)
    df = pd.DataFrame(rows).sort_values("pct_owned", ascending=False).reset_index(drop=True)
    df["banned"] = df.player_id.isin(cx.banned)
    write_csv(df, EXT / "espn_players.csv", args, "player pool with owner, eligibility, ownership trend, season and last-15 averages")
    fa = df[(df.owner == "FREE") & ~df.banned]                                                  # banned players are never offered as free agents
    cols = ["name", "eligibility", "pct_owned", "pct_change", "gp", "mpg", "PTS", "REB", "AST", "STL", "BLK", "3PM", "TO"]
    fmt = lambda d: d[cols].round(1).to_string(index=False)
    print("\nFree agents gaining ownership fastest (these get picked up next):")
    print(fmt(fa[fa.gp.fillna(0) >= 3].sort_values("pct_change", ascending=False).head(10)))
    print("\nMost-owned free agents:")
    print(fmt(fa.head(8)))
    hurt = df[(df.owner_slot == ME) & df.injury.notna() & (df.injury != "ACTIVE")]
    if len(hurt):
        print("\nYour players with an injury tag on ESPN:  " + ", ".join(f"{r.name} ({r.injury})" for r in hurt.itertuples()))


def derive(data: dict, args) -> None:
    if "league" not in data:
        sys.exit("The cached download has no league data.")
    if "matchups" not in data:
        print("note: no matchup data in this download; matchup and standings sections are skipped.")
    cx = Context(data, Path(args.db), args.force)
    skip = set(args.skip or [])
    if "rosters" not in skip:
        sync_rosters(cx, args)
    if "positions" not in skip:
        sync_positions(cx, args)
    if "matchups" in data and "matchups" not in skip:
        mt = matchup_table(cx)
        st = standings_table(cx, mt)
        report_matchups(cx, mt, st, args)
    if "transactions" not in skip:
        report_transactions(cx, args)
    if "players" not in skip:
        report_players(cx, args)
    print("\nDone." + (" (dry run: nba.db and data/external were not changed)" if args.dry_run else ""))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1], formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--year", type=int, default=2027, help="ESPN season id = the year the season ends (2027 = 2026-27)")
    ap.add_argument("--offline", action="store_true", help="skip the network; use the last download")
    ap.add_argument("--dry-run", action="store_true", help="show what would change; write nothing to nba.db or data/external")
    ap.add_argument("--skip", nargs="*", default=[], choices=["rosters", "positions", "matchups", "transactions", "players"], help="leave these steps out")
    ap.add_argument("--backup", action="store_true", help="copy nba.db to NBA Fantasy/nba_before_espn_sync.db before writing (off by default)")
    ap.add_argument("--force", action="store_true", help="apply roster changes even if there are many or the team matching looks off")
    ap.add_argument("--db", default=str(ROOT / "data/processed/nba.db"), help="database to update (default: the real one)")
    args = ap.parse_args()
    if args.offline:
        folder = CACHE / "latest"
        print(f"offline: using the download at {folder}")
    else:
        env = read_env(ROOT / ".env")
        check_network()
        print("network ok; downloading your league ...")
        folder = download(env, args.year)
    derive(load_cache(folder), args)


if __name__ == "__main__":
    main()
