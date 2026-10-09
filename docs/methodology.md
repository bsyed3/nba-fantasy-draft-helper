# Methodology

How each phase works, why, and what it deliberately leaves out. Numbers below come from the real 2015-16 →
2025-26 data; re-run the scripts to refresh them.

## League model

ESPN, 10 teams, snake, 13 drafted players. 9 categories: PTS, 3PM, REB, AST, STL, BLK, FG%, FT%, TO (lower
wins). Each week is one matchup: win if you take more than 4.5 categories (ties split). Lineups are set
daily with slots PG SG G SF PF F C UTIL×3 (10 active), 3 bench, 1 IR; at most 3 centers per roster. The
season is 18 regular-season weeks + 3 playoff weeks.

## Data (SQLite, `schema.sql`)

Raw facts are box scores and games; everything else is derived. `points` is a generated column
(`2·FGM + 3PM + FTM`), percentages are always `ΣMakes/ΣAttempts`. Regular-season games only (`002…` ids); the NBA
Cup final isn't a regular-season game and isn't in the data (Cup quarter/semifinals are, and they count).
Ingest sanity check: ≈99% of team-games sum to 240 (+5/OT) minutes. See `docs/data_model.md`.

## Phase 1 — projections (`src/models/projection.py`)

**Targets** (per player-season): per-36 rates for 2PA, 3PA, FTA, REB, AST, STL, BLK, TOV; 2P%, 3P%, FT%;
minutes per game; availability (games ÷ team games). Per-game FGM/3PM/FTM/PTS/FG% are *derived*, never
modelled. Two- and three-point shooting are separate so FGM ≥ 3PM by construction.

**Every stat is a ratio of totals**, so one function (`pooled_ratio`) handles shrinkage and multi-season
pooling for all of them: `(Σ wₗ·numₗ + k·prior) / (Σ wₗ·denₗ + k)` over the last three seasons with recency
weights `wₗ = decay^(l-1)`, shrunk toward the league mean for the player's position group. `decay` and `k` are
grid-searched per stat on training seasons only. Shrinkage is a *feature-engineering step feeding the ML
models*, not a competing model.

**Candidates per stat:** the shrunken average ("baseline"), Ridge on the baseline's residual, LightGBM on the
residual (count/volume stats only — percentages use Ridge only). Features: blends of every stat, per-lag
minutes/availability, age, age², experience, draft slot, position group, seasons of history.

**Selection is by rolling-origin backtest** (train on seasons < T, predict T, for T = 2021…2025) with weighted MSE
on players with ≥ 500 minutes, so the choice is audited rather than assumed. Change in out-of-sample MSE vs the
baseline (negative = better): minutes −16% (LightGBM), FGA₂ −10%, availability −7%, FTA −7%, AST −6%, FGA₃ −4%,
TOV −3%, BLK −3%, 2P% −1%; REB, STL and FT% stay with the baseline (the learned models don't help there).
Players with no NBA history use **draft-tier profiles** (picks 1-5, 6-14, 15-30, 31-60, undrafted): mean rates
and a coefficient of variation from historical first seasons. Hand-curated overrides can be supplied
(`data/external/projection_overrides.csv`).

**Not modelled** (the original handoff listed them; none of these is needed to run): contract-year effects,
vacated usage, coaching eras, injury-type history. The investigation plan (`docs/investigation_plan.md`) says to
build these only if analysis shows they matter; that analysis hasn't been run.

## Phase 2 — simulation (`src/simulation/`)

Learned from history (`calibration.py`): Negative-Binomial dispersion per count stat (e.g. REB r≈10.5, FTA r≈3.1),
Beta-Binomial shooting over-dispersion (≈0 for 2P%/3P%, 0.0035 for FT%), an 11×11 Gaussian-copula correlation
matrix (8 count dims + 3 make-rate residual dims) from pooled normal scores of within-player-season standardized
game logs, and the share of missed games that come from whole-week absences (≈0.72, measured over every week of the season for every rotation player so season-ending injuries count; an earlier estimate of 0.42 that excluded them made the simulation over-confident). **Projection uncertainty**
("talent") comes from the backtest residuals with the observed season's own sampling noise subtracted: relative
σ ≈ 15-30% for counts, 2-4 points for percentages, larger for rookies by draft tier, and correlated across stats
(mostly via minutes).

**Simulation** (`copula.py`): each simulated matchup is one of the season's real **matchup periods** (`schedule.py`): Mon–Sun
weeks, except that the NBA Cup knockout week folds into the week before it and the All-Star break folds into the week after — so for 2026-27
weeks 1–6 are single weeks from Oct 19, **week 7 = Nov 30–Dec 13**, weeks 8–16 are single weeks, **week 17 = Feb 15–28**, week 18 ends the regular season Mar 7
and weeks 19–21 are the playoffs (Mar 8–28). The engine's objective uses the 18 regular-season periods (`--include-playoffs` adds the rest).
About 30 NBA Cup games aren't on the published schedule yet (every team shows 80 of 82); they are ignored by default, so week 7 is simulated with its
published games only (`--fill-cup-games` places them on random Cup-week days instead). Players are played game by game on
each team's real days. Per player-week: talent draw → availability (whole-week absence with prob
`c·(1−avail)`, else per-game play prob) → for each game played, copula normals → NB counts and Binomial makes.
Output is a day-resolution library `[players, periods, 14 days, 10 box-score stats]`. Checked in tests: simulated
means match projection × schedule × availability, makes ≤ attempts, no stats on days without games.

## Phase 3 — draft engine (`src/draft/`)

See the README for the loop. Choices worth knowing:

- **Objective = P(win the weekly matchup)**, not Σ z-scores. It emerges that balance, punting, depth and
  positional scarcity matter exactly as much as the simulated weeks say.
- **Daily lineups with slot rules** (`engine.team_totals`): per day, players are placed in static-value order into the
  first open eligible slot (dedicated → G/F → UTIL). Bench players therefore count when someone has no game.
- **Rollouts** model the rest of the draft: opponents pick by ESPN ADP (when `positions.csv` has it; else our rank)
  with noise growing with rank, respecting the center cap; my later picks follow our static rank. Candidates are
  compared with common random numbers.
- **Static value** (`values.py`) is a standard z-score board with FG%/FT% as volume-weighted impact and
  availability-scaled counting stats. It's only used for bot ordering and lineup priority, never as the answer.
- **State** is an append-only pick log in SQLite; undo = delete the last row.

## Position eligibility

`data/external/positions.csv` (from `scripts/fetch_espn_positions.py`) holds ESPN's eligible slots and ADP.
Without it, eligibility is approximated from nba_api's G/F/C labels plus assist/rebound rates; that approximation
is narrower than ESPN's and is flagged in the app.

## Known limits

- The daily lineup is greedy, not an optimal per-day assignment.
- No in-season management: no waivers, trades, streaming, or IR use (a drafted injured player simply misses games).
- Team-level correlation is limited to shared schedules; teammates' injuries and game scripts are independent.
- The schedule still has ≈30 TBD NBA Cup games absent, which slightly under-counts games in early December weeks.
- Absolute win probabilities are relative to the bot model; compare candidates by *differences*.
