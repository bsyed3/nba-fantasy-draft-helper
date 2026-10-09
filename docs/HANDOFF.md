# Session handoff — NBA Fantasy Draft Helper

Written for a fresh Claude Code session (or any new contributor). Read this first, then `README.md` (setup/usage) and
`docs/REPORT.md` (the full methodology + results report, regenerated from code). This file is the "where are we, what's
shaky, what next" document. Last updated 2026-10-09, git `testing` @ `d781946` (plus this file, uncommitted unless noted).

## 1. What this is

A live-draft assistant for **one specific league**, plus the whole pipeline behind it:

- **League:** ESPN · snake draft · **10 teams** · 9-category head-to-head where each **weekly matchup is a single W/L**
  (more than 4.5 categories wins; margin and category count don't matter beyond that). 18 regular-season weeks + 3 playoff weeks.
- **Lineups are set daily.** Slots: PG, SG, G, SF, PF, F, C, 3 UTIL (10 active) + 3 bench + 1 IR. **Max 3 centers** per roster.
  Default draft = 13 rounds (IR not drafted) — *unconfirmed*.
- **Calendar (2026-27):** matchup periods are Mon–Sun weeks except **week 7 = Nov 30–Dec 13** (NBA Cup week, two weeks) and
  **week 17 = Feb 15–28** (All-Star, two weeks). Week 18 = Mar 1–7 ends the regular season; playoff weeks 19–21 = Mar 8–28.
  The NBA Cup *final* is not part of the fantasy season. The ~30 NBA Cup games missing from the published schedule are
  **deliberately ignored** (user's instruction): week 7 simulates only its published games. `--fill-cup-games` is an opt-in.
- **Projections are generated in-house** (no ESPN projections imported).

Objective of the recommender: maximise **P(win the weekly matchup)** over simulated weeks — *not* a z-score sum.

## 2. Working agreements (from the user — follow them)

- **Branches:** `testing` = test environment, `main` = production. Commit/push **only when the user asks**, and push to `testing`.
  Never touch `main` without an explicit instruction. Remote: `github.com/bsyed3/nba-fantasy-draft-helper` (user is `bsyed3`).
- The user wants **honest reporting**: say plainly when a result is within noise or a check fails. Several earlier claims in this
  project had to be walked back; don't repeat that.
- Figures follow the dataviz guidance: palette in `src/reporting/figs.py`, takeaway headlines, **legends in a band under the title
  (never over data)**, highlighted + leader-labelled points in scatters, no dual axes, bars start at zero.
- The user said to "regroup" after the last validation round and has **not yet chosen** between the directions in §7.

## 3. Environment gotchas (Windows)

- Python 3.14, project venv: `.venv\Scripts\python` (run from the repo root `nba-draft-bot`). `pip install -r requirements.txt`.
- Set `PYTHONIOENCODING=utf-8` when printing player names (accents). Scripts already call `sys.stdout.reconfigure`.
- **The Bash tool mangles multi-line heredocs containing quotes** (hangs or "unexpected EOF"). Create scratch scripts with the
  Write tool, then run them. A stray empty `python - <<EOF` launches an interactive REPL that hangs forever.
- `stats.nba.com` (nba_api) **works**. `fantasy.espn.com` / `lm-api-reads.fantasy.espn.com` are **blocked** by the dev machine's web
  filter and Python's TLS verification fails there. **Do not disable certificate verification or work around the filter.**
- Long jobs: `scripts/validate_draft.py` takes ~17 min sequentially on a quiet machine. Seasons can run in parallel:
  `--years 2022 --tag _2022` etc. (18 logical cores). Run in the background and don't launch heavy jobs alongside it.
- Headless UI test: `streamlit.testing.v1.AppTest`; point `NBA_DB` at a *copy* of the DB so test leagues don't pollute the real one.

## 4. How to run everything

```bash
python scripts/ingest.py                    # box scores 2015-16..latest + ages + PlayerIndex (incl. rookies) + published schedule  (~2 min)
python scripts/fetch_espn_positions.py      # ESPN eligibility + ADP -> data/external/positions.csv  (USER must run; ESPN blocked here)
python scripts/build_projections.py         # rolling backtest + 2026-27 projections (+ rookie profiles, OOS residuals)         (~15 s)
python scripts/build_library.py             # calibrate from history + simulate the season's real matchup periods               (~40 s)
streamlit run src/app/streamlit_app.py      # the live draft tool (state persists in SQLite after every pick)
python scripts/validate_draft.py --years 2022 2023 2024 2025 --drafts 24   # out-of-sample mock-draft validation   (~17 min)
python scripts/make_report.py               # regenerates docs/REPORT.md + REPORT.html (+ 28 figures)                           (~1 min)
python -m pytest -q                         # 36 tests, no network needed                                                       (~15 s)
```

`data/processed/` (gitignored): `nba.db` (~30 MB), `projections_2026.{parquet,csv}`, `backtest_summary.csv`, `backtest_oos.parquet`,
`rookie_profiles.csv`, `calibration.json`, `library_2026.npz` (~6 MB compressed / 62 MB in memory), validation CSVs/logs.
Rebuild order matters: ingest → projections → library → (validate) → report.

## 5. Architecture and the decisions behind it (don't re-litigate without evidence)

| Piece | File(s) | Key idea |
|---|---|---|
| Schema | `schema.sql`, `docs/data_model.md` | `points` is a *generated column* (2·FGM+3PM+FTM); FG%/FT% always ΣMakes/ΣAttempts. League/draft tables = append-only pick log (undo = delete last row). |
| Ingest | `scripts/ingest.py`, `src/data_ingestion/` | Idempotent; regular-season `002…` games only (NBA Cup final `006…` excluded); stints rebuilt from scratch each run. |
| Season table | `src/features/player_seasons.py` | Every modelled stat is a **ratio of two totals** (`STAT_DEFS`): per-36 rates, 2P%/3P%/FT%, mpg, availability. FGM/PTS/FG% are *derived*, never modelled. 2-pt and 3-pt shooting are separate. |
| Projections | `src/models/projection.py` | Per stat: shrunken recency-weighted baseline vs Ridge-on-residual vs LightGBM-on-residual; **winner chosen per stat by rolling-origin backtest** (T=2021–2025). Rookies/no-history: draft-slot bucket means. |
| Positions | `src/features/positions.py` | ESPN eligibility from `data/external/positions.csv` if present, else a heuristic from nba_api labels + ast/reb rates. `is_center` = C-eligible. |
| Calibration | `src/simulation/calibration.py` | NB dispersion, shooting overdispersion, 11×11 copula correlation, absence share `c`, talent σ (from backtest residuals), games pmf. |
| Schedule | `src/simulation/schedule.py` | Builds matchup periods from the `game` table (Cup week merges with the week *before*, All-Star with the week *after*); future-season missing games recorded but ignored by default. |
| Simulator | `src/simulation/copula.py`, `library.py` | Game-by-game Gaussian copula → NB counts, Binomial makes; per (sim, team) game-day masks; output `uint8 [players, periods, 14 days, 10 stats]`. |
| Engine | `src/draft/engine.py` | For each candidate: roll the rest of the draft forward (bots by `bot_base` = ESPN ADP if known else our rank, plus noise; center cap enforced), play every roster through the simulated periods with **daily greedy lineups** (dedicated slot → G/F → UTIL), score my team vs each opponent. Common random numbers across candidates. |
| State/UI | `src/draft/state.py`, `store.py`, `src/app/streamlit_app.py` | Snake order, center cap, SQLite persistence, flag-injured, plan-ahead mode, league/power-ranking tabs. |
| Validation | `scripts/validate_draft.py`, `src/draft/mock.py` | Per past season: project from earlier data only, build library on that season's real calendar, run paired mock drafts (engine vs best-available by value board) vs noisy bots, score on **actual day-by-day box scores**. |
| Report | `scripts/make_report.py`, `src/reporting/` | Every number/figure generated from data; text is conditional on results (it will say "not significant" if so). |

Things deliberately **not** built: coach/team-era data, historical injury reasons, contract status, auction drafts, waivers/trades,
IR modelling (the original handoff listed coach/injury/contract as ideas; the investigation plan says build only if analysis justifies).

## 6. Current state — results (honest)

**Data:** 281,164 player-games / 13,209 games (2015-16 → 2025-26); 614 players projected for 2026-27 (53 from the 2026 draft class);
published 2026-27 schedule = 1,200 games (every team shows 80 of 82). Box-score integrity: 99% of team-games sum to 240(+5/OT) minutes.

**Projections (rolling backtest, out-of-sample):** vs the shrunken baseline, MSE change: minutes −16% (LightGBM), 2-pt attempts −10% (Ridge),
availability −7.5% (LightGBM), FT attempts −6.5%, assists −5.8%, 3-pt attempts −3.6%, blocks −3.2%, turnovers −2.9%, 2P% −1.4%, 3P% −0.4%;
**rebounds, steals, FT% stay on the baseline** (learned models didn't help). Versus "same as last year": better on all nine categories
(MAE −6% to −24%; FT% R² 0.52 vs 0.07). Ranking players by projected 9-cat value: Spearman 0.670 vs 0.618, top-50 hit rate 62% vs 55% — but the
top-50 edge is uneven across seasons (ties/losses in 2 of 5).
Known mild biases (residual plots, report §4.5): over-projects age 28+ by a few %, under-projects the youngest; large apparent
under-projection for <15-mpg players is mostly a **selection artifact** of the ≥500-minute evaluation filter.

**Simulation calibration (2025-26 test, PIT):** outer deciles hold 10.8% / 12.8% (nominal 10%) — close, slightly over-confident in the upper tail.
A first version was clearly over-confident (15.8%/15.2%): the whole-week-absence share `c` had been estimated excluding long/season-ending injuries
(0.42); corrected to **0.72**. Widening the talent σ did *not* help (lower-tail excess was availability, not talent).

**Validation (96 paired mock drafts, 2022–2025, weekly-matchup win-probability gain vs "take best value left"):**

| Engine mode | Gain | Ahead in | By season (22/23/24/25) | Model's own predicted gain |
|---|---|---|---|---|
| Pure argmax | **−0.8 ± 1.7 pts** | 46% | +2.7 / −2.7 / −0.2 / −3.0 | +3.5 |
| 1-pt margin (override value rank only if clearly ahead) | **+2.5 ± 1.3 pts** | 53% | −0.2 / +4.1 / +3.0 / +3.0 | +3.1 |

Earlier runs under previous calendar/rookie/availability settings gave +1.6±1.3, +4.8±1.7, +2.4±1.5 — all consistent with a true effect between ~0 and a few points.
**Conclusion to carry forward: the engine is NOT shown to beat a good value ranking.** The projections/value board do nearly all the work.
The margin variant (margin chosen from estimate noise, not tuned) is promising but unconfirmed (≈1.8 SE, tested once on the same drafts).
Absolute win rates (~70%) are inflated because the bots are unrealistically noisy.

## 7. Known problems and areas for improvement (roughly by importance)

1. **Bots are unrealistic.** Each bot draws *independent* noise (validation: sd = 6 + 0.2·rank; engine internal: 3 + 0.12·rank) around our own value rank.
   In 300 simulated drafts a rank-16 player is taken anywhere from pick 7 to 28; rank-23 is still there after pick 45 about 1% of the time. Real drafters share one
   consensus (ESPN ADP) and cluster tightly. The engine exploits these artificial falls, which inflates its predicted gain and makes validation opponents too easy.
   → Use ESPN ADP + smaller, *shared* noise (`DraftEngine._bot_orders`, `mock.run_mock_draft`). Needs the ESPN fetch.
2. **Value board vs market.** The static board (`src/draft/values.py`) multiplies counting stats by projected availability (full discount). Cade Cunningham is #10 per game but #23
   after availability (and turnovers z = −2.6); Pascal Siakam is #87 (his 2025-26 *actual* 9-cat rank by the same method was 110th). The user expects Cade top-10/round 1 and
   Siakam top-3 rounds. In a daily-lineup league with bench + waivers a missed game costs less than full output, so a full discount is probably too harsh for the *board* (the engine's
   simulation already models absences with bench cover). → make the discount a setting (partial/none) and test it with `analysis.ranking_accuracy`; compare against ADP once available.
3. **Position eligibility is a heuristic** until `scripts/fetch_espn_positions.py` is run on a network that reaches ESPN (script's parsing is unit-tested on a synthetic payload but the
   live call has never run; slot-id mapping 0–4 = PG/SG/SF/PF/C is from memory — eyeball its printed sample).
4. **Validation is not fully leak-free.** The projections are rebuilt from earlier data per year, but `calibration.json` (dispersion, copula, `c`, talent σ, rookie σ) is fit on 2022–2025 and reused for every
   validation year; the report subtitle "built from pre-2025 data only" is therefore inaccurate for those constants. Also: per-stat model choice is selected on the same backtest that reports gains;
   decisions (the `c` fix, rookie-curve test, margin variant) were made after seeing test results; validation pool/teams use who actually played (`last_year`, first-game team); position labels are today's.
   → refit `calibrate(...)` inside `validate_year` with only earlier seasons and a rookie-profile CSV from `st[year < Y]`; reserve a truly untouched holdout (the 2026-27 season itself, tracked in-season).
5. **Margin variant unconfirmed.** Re-run with fresh bot seeds / more drafts before adopting. If adopted: add a `recommended` column to `engine.recommend` (highest value-rank among candidates within 1 pt of the best)
   and have the app's "Best pick" use it, while still showing the pure win-probability table.
6. **Rookies are generic within a draft-slot bucket** (two top-5 picks are identical; a smooth log(pick) curve was tested and was no more accurate). Players returning from overseas get the undrafted bucket.
   Use `data/external/projection_overrides.csv` for rookie numbers the user trusts; consider external priors (college/international stats) — see `docs/investigation_plan.md`.
7. **Minutes and availability are the dominant error source** (residuals across stats are correlated ~via minutes). Ideas: team context after offseason moves (the model ignores `team_id` changes), vacated usage,
   injury-status feed on draft day, age-specific durability.
8. **Two-week matchups:** "out for the whole period" is drawn once per period, overstating double-absence in weeks 7 and 17 (`c` was measured on one-week windows). Week 7 ignores ~30 unscheduled Cup games (user's choice) and is the noisiest week.
9. **Greedy daily lineup** (static-value order) instead of an optimal per-day assignment; no IR/waiver/streaming/trade modelling.
10. **Model-library choice** (LightGBM vs XGBoost) was never benchmarked; low stakes. Adding XGBoost as a fourth candidate in `predict_all` is ~10 lines.
11. **Report hygiene:** `docs/REPORT.html` is 3.7 MB in git (fine, but consider ignoring it and publishing via CI); the report's 8.4 "bug this check caught" section contains hard-coded history numbers.

## 8. Next steps (suggested order)

1. **User runs `scripts/fetch_espn_positions.py`** (or sends ESPN eligibility/ADP) → rebuild projections + library. Verify the printed sample against ESPN; check unmatched top-200 players.
2. **Fix bot realism** with ADP + shared noise; re-run validation. Expect the engine's apparent edge to shrink further; that is information.
3. **Make the availability discount configurable** and pick the board using `ranking_accuracy` plus ADP agreement; then re-check Cade/Edwards/Siakam-type disagreements.
4. **Leak-free validation** (per-year calibration refit) and fresh seeds for the margin variant; decide whether to ship the margin tie-break as the app default.
5. **Projection quality:** rookies (overrides/external), minutes/availability inputs, aging-curve bias, optional XGBoost comparison, SHAP for explanation.
6. **Product polish:** if the engine stays a "second opinion", lean into explanation (category needs, position-slot map, "why not X"); add 14-round option and IR handling if the user confirms the rules; playoff-odds objective if they give the playoff cutoff.
7. **Engineering:** CI running `pytest -q`; pin requirements; avoid committing large generated HTML; keep `REPORT.md` regenerated after any model/data change.

## 9. Open questions for the user

- 13 or 14 draft rounds (is the IR slot drafted)? Does the 3-center cap count any **C-eligible** player (incl. PF/C types) or only players slotted/listed at C?
- Can they run the ESPN script (or send eligibility + ADP)? Any rookie projections they trust?
- How many of the 10 teams make the playoffs (for a season-level objective)? Any draft timer/auto-draft, keepers, or pre-season drop rules?
- Which direction after the validation result: (a) engine as an explained second opinion, (b) invest in projection quality, (c) both?

## 10. Quick orientation for a new session

- Start with `docs/REPORT.md` §1 (summary), §10 (validation), §12 (limits). Then `src/draft/engine.py` (`DraftEngine.recommend`, `team_totals`, `evaluate`, `_bot_orders`).
- To check the app end-to-end headlessly: copy `data/processed/nba.db`, set `NBA_DB`, create a league with `store.save_league`, drive `AppTest` (see git history / report for the pattern).
- Tests cover: schema invariants, snake order/persistence, shrinkage math, projection pipeline on synthetic history, simulation moments and constraints, daily-lineup slot rules, center cap, schedule/matchup-period logic (incl. the Cup/All-Star merges), ESPN parsing.
- When changing models or the simulator, re-run: `build_projections.py` → `build_library.py` → `pytest -q` → (`validate_draft.py`) → `make_report.py`, and update this file's §6/§7.
