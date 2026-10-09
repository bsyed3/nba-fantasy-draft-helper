-- SQLite schema matching docs/data_model.md.
--
-- Raw fact tables (never edited after ingestion): game, box_score,
-- player_absence.
-- Reference tables (mostly static): team, coach, player.
-- Boundary tables (identity + date range ONLY — stats always come from
-- the views at the bottom, never stored here): team_era, player_stint.
--
-- Requires SQLite 3.31+ for generated columns (bundled with Python's
-- sqlite3 on any reasonably recent install — check with
-- `python3 -c "import sqlite3; print(sqlite3.sqlite_version)"`).

CREATE TABLE IF NOT EXISTS team (
    team_id INTEGER PRIMARY KEY,
    name TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS coach (
    coach_id INTEGER PRIMARY KEY,
    name TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS player (
    player_id INTEGER PRIMARY KEY,
    name TEXT,
    birthdate DATE,
    position_1 TEXT,
    position_2 TEXT,
    draft_year INTEGER,
    draft_position INTEGER,
    team_id INTEGER,       -- CURRENT NBA team (PlayerIndex); 0/NULL = free agent. Drives the schedule a player is simulated on
    rookie_year INTEGER,   -- start year of first NBA season (2016 = 2016-17), from PlayerIndex FROM_YEAR
    last_year INTEGER      -- start year of most recent season/roster (2026 = 2026-27), from PlayerIndex TO_YEAR
);

-- Age during a given season (nba_api LeagueDashPlayerBioStats). One row
-- per player per season; birthdate isn't exposed by any bulk endpoint.
CREATE TABLE IF NOT EXISTS player_season (
    player_id INTEGER NOT NULL REFERENCES player(player_id),
    season TEXT NOT NULL,
    age REAL,
    team_id INTEGER,
    PRIMARY KEY (player_id, season)
);

CREATE TABLE IF NOT EXISTS game (
    game_id TEXT PRIMARY KEY,
    date DATE NOT NULL,
    home_team_id INTEGER NOT NULL REFERENCES team(team_id),
    away_team_id INTEGER NOT NULL REFERENCES team(team_id),
    season TEXT NOT NULL,
    game_type TEXT NOT NULL DEFAULT 'regular'
        CHECK (game_type IN ('regular', 'play-in', 'playoff'))
);

-- One row per player per game. `points` is NOT something you insert —
-- it's a generated column computed from the primitives (see the proposal
-- review + data model docs on why Points must never be stored
-- independently of FGM/3PM/FTM), so it can never drift out of sync.
CREATE TABLE IF NOT EXISTS box_score (
    game_id TEXT NOT NULL REFERENCES game(game_id),
    player_id INTEGER NOT NULL REFERENCES player(player_id),
    team_id INTEGER NOT NULL REFERENCES team(team_id),
    min REAL,
    fga INTEGER,
    fgm INTEGER,
    tpa INTEGER,  -- 3PA
    tpm INTEGER,  -- 3PM
    fta INTEGER,
    ftm INTEGER,
    oreb INTEGER,
    dreb INTEGER,
    ast INTEGER,
    stl INTEGER,
    blk INTEGER,
    tov INTEGER,
    points INTEGER GENERATED ALWAYS AS (2 * fgm + tpm + ftm) VIRTUAL,
    PRIMARY KEY (game_id, player_id)
);

-- Boundary only — no stat columns. See player_stint_stats view below.
CREATE TABLE IF NOT EXISTS player_stint (
    player_stint_id INTEGER PRIMARY KEY AUTOINCREMENT,
    player_id INTEGER NOT NULL REFERENCES player(player_id),
    team_id INTEGER NOT NULL REFERENCES team(team_id),
    start_date DATE NOT NULL,
    end_date DATE
);

-- Boundary only — no stat columns. See team_era_stats view below.
CREATE TABLE IF NOT EXISTS team_era (
    team_era_id INTEGER PRIMARY KEY AUTOINCREMENT,
    team_id INTEGER NOT NULL REFERENCES team(team_id),
    coach_id INTEGER REFERENCES coach(coach_id),
    start_date DATE NOT NULL,
    end_date DATE
);

-- Raw fact, sourced externally (nba_api has no historical injury archive)
-- — see docs/data_model.md "Injury / availability modeling".
CREATE TABLE IF NOT EXISTS player_absence (
    absence_id INTEGER PRIMARY KEY AUTOINCREMENT,
    player_id INTEGER NOT NULL REFERENCES player(player_id),
    team_id INTEGER NOT NULL REFERENCES team(team_id),
    start_date DATE NOT NULL,
    end_date DATE,
    reason_category TEXT
        CHECK (reason_category IN ('injury','rest','personal','suspension','other')),
    injury_type TEXT,
    source TEXT
);

-- ---------------------------------------------------------------------
-- Phase 3: league settings + live draft state. Draft state is a plain
-- append-only pick log (undo = delete the last row); everything else
-- (rosters, remaining pool, who is on the clock) is derived from it.
-- ---------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS league (
    league_id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL UNIQUE,
    n_teams INTEGER NOT NULL,
    roster_size INTEGER NOT NULL,
    active_slots INTEGER NOT NULL,   -- players that count in a weekly matchup
    my_slot INTEGER NOT NULL,        -- 1-based first-round draft position
    max_centers INTEGER NOT NULL DEFAULT 3,  -- league rule: max centers on a roster
    season TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS draft_pick (
    league_id INTEGER NOT NULL REFERENCES league(league_id),
    pick_no INTEGER NOT NULL,        -- 1-based overall pick
    team_slot INTEGER NOT NULL,      -- 1-based team who made the pick
    player_id INTEGER NOT NULL REFERENCES player(player_id),
    PRIMARY KEY (league_id, pick_no),
    UNIQUE (league_id, player_id)
);

-- Players the user has flagged as unavailable (injured / out) on draft
-- day; excluded from recommendations but not counted as drafted.
CREATE TABLE IF NOT EXISTS draft_exclusion (
    league_id INTEGER NOT NULL REFERENCES league(league_id),
    player_id INTEGER NOT NULL REFERENCES player(player_id),
    PRIMARY KEY (league_id, player_id)
);

CREATE INDEX IF NOT EXISTS idx_box_score_player ON box_score(player_id);
CREATE INDEX IF NOT EXISTS idx_box_score_game ON box_score(game_id);
CREATE INDEX IF NOT EXISTS idx_game_date ON game(date);
CREATE INDEX IF NOT EXISTS idx_player_stint_player ON player_stint(player_id);

-- ---------------------------------------------------------------------
-- Derived views. Never write to these directly — they're recomputed from
-- the raw fact tables every time they're queried, which is exactly the
-- point (see docs/data_model.md).
-- ---------------------------------------------------------------------

CREATE VIEW IF NOT EXISTS player_stint_stats AS
SELECT
    ps.player_stint_id,
    ps.player_id,
    ps.team_id,
    COUNT(*) AS games,
    AVG(bs.min) AS mpg,
    AVG(bs.points) AS ppg,
    AVG(bs.oreb + bs.dreb) AS rpg,
    AVG(bs.ast) AS apg,
    AVG(bs.stl) AS spg,
    AVG(bs.blk) AS bpg,
    AVG(bs.tov) AS topg,
    AVG(bs.tpm) AS tpmpg,
    SUM(bs.fgm) * 1.0 / NULLIF(SUM(bs.fga), 0) AS fg_pct,
    SUM(bs.ftm) * 1.0 / NULLIF(SUM(bs.fta), 0) AS ft_pct
FROM player_stint ps
JOIN box_score bs ON bs.player_id = ps.player_id AND bs.team_id = ps.team_id
JOIN game g ON g.game_id = bs.game_id
WHERE g.date >= ps.start_date AND g.date <= COALESCE(ps.end_date, g.date)
GROUP BY ps.player_stint_id;

CREATE VIEW IF NOT EXISTS team_era_stats AS
SELECT
    te.team_era_id,
    te.team_id,
    COUNT(DISTINCT bs.game_id) AS games,
    AVG(bs.points) AS ppg,
    SUM(bs.fga) * 1.0 / COUNT(DISTINCT bs.game_id) AS fga_per_game -- rough pace proxy; refine per docs/data_model.md
FROM team_era te
JOIN game g ON (g.home_team_id = te.team_id OR g.away_team_id = te.team_id)
JOIN box_score bs ON bs.game_id = g.game_id AND bs.team_id = te.team_id
WHERE g.date >= te.start_date AND g.date <= COALESCE(te.end_date, g.date)
GROUP BY te.team_era_id;
