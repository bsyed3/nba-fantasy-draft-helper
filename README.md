# NBA Fantasy Draft Helper

A live-draft assistant for a **10-team ESPN snake draft, 9-category head-to-head** league where each
week is a single matchup win/loss (category wins only matter for deciding the week; the season is
18 regular-season weeks + 3 playoff weeks).

League rules it models: lineups set **daily**; slots PG, SG, G, SF, PF, F, C, 3 UTIL (10 active), 3 bench,
1 IR; **max 3 centers** per roster. The IR slot isn't drafted — the app assumes 13 draft rounds (see
"Open questions" below).

Three separate problems, solved separately:

| Phase | Question | Method | Code |
|---|---|---|---|
| 1. Projection | What will each player average next season? | Shrunken multi-season averages vs Ridge vs LightGBM per stat, chosen by rolling-origin backtest; draft-tier profiles for rookies | `src/models/projection.py` |
| 2. Volatility | How could a *week* actually go? | Gaussian-copula simulation: Negative Binomial counts, Binomial makes, correlated across stats; talent uncertainty from backtest residuals; injury/games-played structure | `src/simulation/` |
| 3. Draft value | Who maximises my chance to win weekly matchups, given what's already been taken? | Mock-draft roll-outs per candidate, scored on simulated weeks; live draft-state tracker + Streamlit UI | `src/draft/`, `src/app/` |

Points, FG% and FT% are **never** modelled or simulated directly: `PTS = 2·FGM + 3PM + FTM` (a generated
column in the database), `FG% = ΣFGM/ΣFGA`. Two- and three-point shooting are modelled separately.

## Quick start

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows  (source .venv/bin/activate on macOS/Linux)
pip install -r requirements.txt

python scripts/ingest.py                 # 1. box scores 2015-16..latest + the published 2026-27 schedule -> data/processed/nba.db (~2 min)
python scripts/fetch_espn_positions.py   # 2. ESPN's exact position eligibility + ADP -> data/external/positions.csv
python scripts/build_projections.py      # 3. backtest + 2026-27 projections
python scripts/build_library.py          # 4. calibrate + simulate the season's real weeks, game by game (~40 s)
streamlit run src/app/streamlit_app.py   # 5. draft
```

Step 2 needs a network that can reach `lm-api-reads.fantasy.espn.com`. Without it the pipeline still runs, using
an approximate eligibility derived from nba_api labels and assist/rebound rates (the app shows a warning until
`positions.csv` exists). Re-run steps 3–4 after step 2.

In the app: create a league in the sidebar (defaults are your ESPN league: 10 teams, 13 roster spots,
10 active), set your draft slot, then **record every pick as it happens**. On your turn the Draft tab
ranks the best available players by *simulated probability of winning a weekly matchup* if you take
them. Flip "Plan my next pick" to look ahead while others are on the clock. Flag injured/out players on
the Player board tab and they're excluded. Draft state is saved in SQLite after every pick, so
refreshing or closing the app loses nothing; "Undo last pick" fixes entry mistakes.

Re-running any script is safe (ingest skips seasons already loaded; `--refresh` re-pulls one).
Before the draft, re-run ingest + the two build scripts to pick up late roster/rookie changes.

## How the recommendation works

For each candidate pick the engine (`src/draft/engine.py`):

1. takes the candidate for my pick;
2. plays the rest of the draft forward — every other team, and my own later picks, take the best
   remaining player by a *noisy* static ranking (a z-score board) — so it knows what the player pool will
   probably look like when I pick again;
3. plays every roster through the simulated weeks with **daily lineups**: each day, in order of static value,
   every player with a game takes the first open slot he's eligible for (his dedicated position, then G or F,
   then UTIL) and the rest sit — so bench players earn their keep when a starter is off or out, and the slot
   rules (plus the 3-center cap in every roll-out) create positional scarcity;
4. scores my team against each opponent over ~700 simulated matchups (40 draws of each of the season's 18 real matchup periods —
   **week 7 = Nov 30–Dec 13** (NBA Cup week) and **week 17 = Feb 15–28** (All-Star) are two-week matchups): more categories won = matchup won.
   The ~30 NBA Cup games still missing from the published schedule are ignored (week 7 uses only its published games).

Because the objective is P(win the week), not a sum of z-scores, it values category balance, punting and
variance automatically. Candidates are compared with common random numbers, so the *differences* between
them are far more reliable than the absolute win percentages (which are relative to the bot model).

**Stated simplifications:** the daily lineup is a greedy fill in static-value order, not a per-day optimal
assignment; bots draft by ESPN ADP (or our rank if ADP is missing) plus noise, not positional need or personal
bias; IR is not modelled (a drafted player who is hurt just misses games); the schedule's still-TBD NBA Cup games
are absent; only regular-season weeks are modelled (no playoff-seeding model).

## Data

| Source | What | Notes |
|---|---|---|
| `nba_api` `LeagueGameLog` | every player-game box score + team games | regular season only (`002…` game ids). The NBA Cup *final* isn't regular-season data and is excluded; Cup quarter/semifinals are regular-season games and included |
| `nba_api` `PlayerIndex` | position, draft slot, rookie year, last year — **includes the incoming rookie class** | one call |
| `nba_api` `LeagueDashPlayerBioStats` | age by season | one call per season |
| `nba_api` `ScheduleLeagueV2` | the published regular-season schedule — drives which days every team plays | `ingest.py` loads it for the season after the latest played |
| ESPN fantasy API (`fetch_espn_positions.py`) | exact position eligibility + average draft position | optional but recommended; run where ESPN is reachable |

Everything lives in one SQLite file (`data/processed/nba.db`, committed so a fresh clone works without re-ingesting; re-run `scripts/ingest.py` to refresh); schema in `schema.sql`, design
notes in `docs/data_model.md`. Box-score completeness was checked on ingest: ≈99% of team-games sum to
240 (+5/OT) minutes.

Hand-curated overrides: put `data/external/projection_overrides.csv` (`player_id` or `name`, plus any of
`mpg, avail, fga, fgm, tpa, tpm, fta, ftm, reb, ast, stl, blk, tov` per game) next to the DB and
`build_projections.py` will apply them — e.g. to anchor a rookie to a source you trust.

## Validation and the full report

`scripts/validate_draft.py` replays past seasons end-to-end using only information available before
each season: project → simulate → mock-draft (engine vs "best available") → score the final rosters on
that season's **actual day-by-day box scores**. `scripts/build_projections.py` also prints the projection backtest
(out-of-sample error per stat vs the shrunken-average baseline) and saves it to `data/processed/backtest_summary.csv`.

**[`docs/REPORT.md`](docs/REPORT.md)** (also `docs/REPORT.html`, self-contained with embedded figures) is the complete
written explanation: data, shrinkage, the model competition and backtest, residual diagnostics, rookies, the 2026-27
projections, biggest risers/fallers, the simulation and its calibration check, worked draft examples, and the historical
validation — every number and figure is generated by `python scripts/make_report.py`, so it stays in sync with the code.

## Repo layout

```
schema.sql                  SQLite DDL (box scores, players, league/draft state)
scripts/
  ingest.py                 nba_api -> nba.db (multi-season, idempotent)
  build_projections.py      Phase 1: backtest + projections (+ rookie profiles, residuals)
  fetch_espn_positions.py   ESPN position eligibility + ADP -> data/external/positions.csv
  build_library.py          Phase 2: calibrate + simulate the season's real weeks
  validate_draft.py         end-to-end out-of-sample validation on past seasons
  make_report.py            regenerates docs/REPORT.md + REPORT.html with all figures
src/
  data_ingestion/           nba_api client + DB loaders
  features/                 per-player-season table, shrinkage definitions, stints
  models/projection.py      Phase 1
  simulation/               calibration.py (learn from history), copula.py (simulate)
  draft/                    state.py, store.py (persistence), values.py (z-board), engine.py, mock.py
  app/streamlit_app.py      live draft UI
tests/                      pytest; synthetic data, no network needed
docs/                       data model, methodology, validation, original investigation plan
```

Run the tests with `python -m pytest -q`.

## Open questions

- **Draft rounds:** 13 (excluding IR) or 14? Change `roster_size` when creating the league if your draft has 14 rounds
  (the extra pick is simply modelled as more bench depth).
- **Center cap:** the app counts any **C-eligible** player (including PF/C types such as Wembanyama) toward the 3-center
  cap. If ESPN only counts players slotted/listed at C, say so and it's a one-line change (`is_center` in `features/positions.py`).
