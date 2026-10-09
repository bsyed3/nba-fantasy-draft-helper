# Session handoff — NBA Fantasy Draft Helper

Written for a fresh Claude Code session (or any new contributor). Read this first, then `README.md` (setup/usage) and
`docs/REPORT.md` (the full methodology + results report, regenerated from code). This file is the "where are we, what's
shaky, what next" document. Last updated 2026-10-09 (second revision). Git branch `testing`; run `git log --oneline -5` and
`git status -sb` to see exactly what is committed vs pushed (the first push was `ed4d790`; later commits are local until the user asks to push).

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

**Deliverables that exist besides the code** (none of these are in git except where noted):
- `docs/REPORT.md` / `docs/REPORT.html` — the full written report with figures (in git).
- `C:\Users\SyedB\Personal\NBA Fantasy\player_projections_2026-27.xlsx` — all 614 projected players, sorted by board rank (Rank, Player, PPG, FGM, FGA, FG%, 3PM, FTM, FTA, FT%, REB, AST, STL, BLK, TOV + a Notes tab). One-off export, outside the repo, built with openpyxl (installed in the venv only, not in `requirements.txt`). Re-create from `data/processed/projections_2026.parquet` if needed.
- The Streamlit app is launchable through the preview tool: config is `C:\Users\SyedB\Personal\NBA Fantasy\.claude\launch.json` (name `draft-helper`, port 8501, absolute paths; deliberately outside the repo). Or just `streamlit run src/app/streamlit_app.py`.

## 2. Working agreements (from the user — follow them)

- **Branches:** `testing` = test environment, `main` = production. Commit/push **only when the user asks**, and push to `testing`.
  Never touch `main` without an explicit instruction. Remote: `github.com/bsyed3/nba-fantasy-draft-helper` (user is `bsyed3`).
- The user wants **honest reporting**: say plainly when a result is within noise or a check fails. Several earlier claims in this
  project had to be walked back (including one this session: whole-season-lost returners' availability is 0.48, not 0.69 — see §6); don't repeat that.
  Check whether a subgroup result is a **selection artifact** (e.g. only counting players who later played ≥500 minutes) before believing it.
- The user has **domain knowledge the data lacks** (injury type/severity, role changes). Treat their player-specific beliefs as inputs to be
  tested against data and, where the data can't settle it, supported through overrides — not as something to argue away or silently code in.
- Figures follow the dataviz guidance: palette in `src/reporting/figs.py`, takeaway headlines, **legends in a band under the title
  (never over data)**, highlighted + leader-labelled points in scatters, no dual axes, bars start at zero.
- The user said to "regroup" after the validation round and has still **not chosen** between the directions in §8/§9.

## 3. Environment gotchas (Windows)

- Python 3.14, project venv: `.venv\Scripts\python` (run from the repo root `nba-draft-bot`). `pip install -r requirements.txt`.
- Set `PYTHONIOENCODING=utf-8` when printing player names (accents). Scripts already call `sys.stdout.reconfigure`.
- **The Bash tool mangles multi-line heredocs containing quotes** (hangs or "unexpected EOF"). Create scratch scripts with the
  Write tool, then run them. A stray empty `python - <<EOF` launches an interactive REPL that hangs forever. Edit/Write on a file require
  it to have been Read in the session; Python patch scripts (`Path.read_text/replace/write_text`) sidestep that.
- `stats.nba.com` (nba_api) **works**. `fantasy.espn.com` / `lm-api-reads.fantasy.espn.com` are **blocked** by the dev machine's web
  filter and Python's TLS verification fails there. **Do not disable certificate verification or work around the filter.**
- Long jobs: `scripts/validate_draft.py` takes ~17 min sequentially on a quiet machine. Seasons can run in parallel:
  `--years 2022 --tag _2022` etc. (18 logical cores). Run in the background and don't launch heavy jobs alongside it.
- Headless UI test: `streamlit.testing.v1.AppTest`; point `NBA_DB` at a *copy* of the DB so test leagues don't pollute the real one. (AppTest cannot
  serialise multiselects whose options are ints — use string options.) The running dev server caches the engine with `st.cache_resource`:
  **restart the server after changing engine code** or it keeps the stale object.
- The real `data/processed/nba.db` holds the user's league/draft state once they use the app; don't wipe it. It is **committed in a public repo**, so avoid committing it mid-draft with real picks in it
  (currently it has one empty league, `ESPN 10-team`, slot 1, no picks). Each re-commit of the DB adds ~30 MB to git history; commit it only when the data actually changed.

## 4. How to run everything

```bash
python scripts/ingest.py                    # box scores 2015-16..latest + ages + PlayerIndex (incl. rookies) + published schedule  (~2 min)
python scripts/fetch_espn_positions.py      # ESPN eligibility + ADP -> data/external/positions.csv  (USER must run; ESPN blocked here)
python scripts/build_projections.py         # rolling backtest + 2026-27 projections (+ rookie profiles, OOS residuals)         (~15 s)
python scripts/build_library.py             # calibrate from history + simulate the season's real matchup periods               (~40 s)
streamlit run src/app/streamlit_app.py      # the live draft tool (state persists in SQLite after every pick)
python scripts/validate_draft.py --years 2022 2023 2024 2025 --drafts 24   # out-of-sample mock-draft validation   (~17 min)
python scripts/make_report.py               # regenerates docs/REPORT.md + REPORT.html (+ figures)                              (~1 min)
python -m pytest -q                         # 37 tests, no network needed                                                       (~15 s)
python scripts/analysis/<name>.py           # diagnostics behind §6 (see list below); each prints its findings
```

Diagnostics in `scripts/analysis/` (read-only, portable paths): `injury_recovery.py` (past top-60 players who lost most of a season vs healthy stars),
`returner_gap.py` (players who played zero games last season and returned), `availability_bias.py` (is availability/minutes biased for
"usually healthy, one bad year" / "recovered" groups — selection-free), `bot_spread_and_board.py` (how far the mock bots spread players; why the board disagrees with the market).

`data/processed/` (**committed to git** since 2026-10-09, ~38 MB total, so a fresh clone runs without re-ingesting; logs are gitignored; the repo is **public**): `nba.db` (~31 MB), `projections_2026.{parquet,csv}`, `backtest_summary.csv`, `backtest_oos.parquet`,
`rookie_profiles.csv`, `calibration.json`, `library_2026.npz` (~6 MB compressed / 62 MB in memory), validation CSVs/logs.
Rebuild order matters: ingest → projections → library → (validate) → report. The model code is currently the same as when the report/validation were last
generated (two experiments were reverted — §7), so those artifacts are consistent.

## 5. Architecture and the decisions behind it (don't re-litigate without evidence)

| Piece | File(s) | Key idea |
|---|---|---|
| Schema | `schema.sql`, `docs/data_model.md` | `points` is a *generated column* (2·FGM+3PM+FTM); FG%/FT% always ΣMakes/ΣAttempts. League/draft tables = append-only pick log (undo = delete last row). |
| Ingest | `scripts/ingest.py`, `src/data_ingestion/` | Idempotent; regular-season `002…` games only (NBA Cup final `006…` excluded); stints rebuilt from scratch each run. |
| Season table | `src/features/player_seasons.py` | Every modelled stat is a **ratio of two totals** (`STAT_DEFS`): per-36 rates, 2P%/3P%/FT%, mpg, availability. FGM/PTS/FG% are *derived*, never modelled. 2-pt and 3-pt shooting are separate. |
| Projections | `src/models/projection.py` | Per stat: shrunken recency-weighted baseline vs Ridge-on-residual vs LightGBM-on-residual; **winner chosen per stat by rolling-origin backtest** (T=2021–2025). Features: last 3 seasons, age, experience, draft slot, position, minutes/availability lags. **No team/roster context.** Rookies/no-history: draft-slot bucket means. Per-player hand overrides: `data/external/projection_overrides.csv` (`player_id` or `name` + any of mpg, avail, fga, fgm, tpa, tpm, fta, ftm, reb, ast, stl, blk, tov per game). |
| Positions | `src/features/positions.py` | ESPN eligibility from `data/external/positions.csv` if present, else a heuristic from nba_api labels + ast/reb rates. `is_center` = C-eligible. |
| Calibration | `src/simulation/calibration.py` | NB dispersion, shooting overdispersion, 11×11 copula correlation, absence share `c` (=0.72), talent σ (from backtest residuals), games pmf. |
| Schedule | `src/simulation/schedule.py` | Builds matchup periods from the `game` table (Cup week merges with the week *before*, All-Star with the week *after*); future-season missing games recorded but ignored by default. |
| Simulator | `src/simulation/copula.py`, `library.py` | Game-by-game Gaussian copula → NB counts, Binomial makes; per (sim, team) game-day masks; output `uint8 [players, periods, 14 days, 10 stats]`. |
| Engine | `src/draft/engine.py` | For each candidate: roll the rest of the draft forward (bots by `bot_base` = ESPN ADP if known else our board rank, plus noise; center cap enforced), play every roster through the simulated periods with **daily greedy lineups** (dedicated slot → G/F → UTIL), score my team vs each opponent. Common random numbers across candidates. **Candidate pool** = top `top_k` by board rank (+ optional `pergame_k` top players by *per-game* value, + any `extra_candidates`). Output includes `static_rank` (board) and `pergame_rank`. `pergame_k` defaults to 0 so validated behaviour is unchanged; the app uses 10. |
| State/UI | `src/draft/state.py`, `store.py`, `src/app/streamlit_app.py` | Snake order, center cap, SQLite persistence, flag-injured, plan-ahead mode, league/power-ranking tabs. **Recommender table now shows Proj GP (avail×82), MPG, PPG, 3PM, REB, AST, STL, BLK, FG%, FT%, TO plus board rank and per-game rank**; the Player board / My roster tabs show the same; sidebar **"Always evaluate these players"** forces chosen players into the simulation. |
| Validation | `scripts/validate_draft.py`, `src/draft/mock.py` | Per past season: project from earlier data only, build library on that season's real calendar, run paired mock drafts (engine vs best-available by value board) vs noisy bots, score on **actual day-by-day box scores**. |
| Report | `scripts/make_report.py`, `src/reporting/` | Every number/figure generated from data; text is conditional on results (it will say "not significant" if so). |

Things deliberately **not** built: coach/team-era data, historical injury *reasons*, contract status, auction drafts, waivers/trades,
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

### 6.1 User-raised model concerns, investigated this session

**Giannis (board rank 93) is not a bug.** Projection: 24.8 PPG, **29.0 MPG, 65% availability** (age 32), FT% 64% (z −4.7), 3PM z −1.7, FG% z +3.6 → total z −1.4.
Per-36 rates are normal; the drag is **minutes and games, both anchored on his injury-shortened 2025-26** (36 GP, 28.9 MPG; recency decay is 0.4). Counterfactuals:
no availability discount for anyone → rank 66; half discount → 84; his 2023-24 role (34.2 MPG, 82% of games) → **11th**. In the backtest, regulars coming off injury-shortened
years have *unbiased* minutes (projected 26.8 vs actual 26.7, n=117) and the elite-minutes subgroup is if anything over-projected (26.7 vs 25.8, n=27). The simulation (engine) also
rates him last of 32 evaluated candidates (49.4% vs Jokić 58.5%) — consistent with the projection. At pick 6 the *simulation's* top choice is **Cade Cunningham** even though the board has him 23rd.

**Board vs market (user's expectations: Cade round 1, Edwards round 1–2, Siakam top-3 rounds):** Edwards (#16) matches. Cade is #10 per game but #23 after the availability discount and turnovers (z −2.6).
Siakam is #87 per game and #87 board (his 2025-26 *actual* 9-cat rank by the same method was 110th). ESPN's board/ADP can't be seen from here (blocked) — use ADP to settle it.

**Injury-recovery analysis (user's suggestion; `scripts/analysis/injury_recovery.py`).** Past top-60 per-game players (≥50 GP the year before) who then played ≤41 GP, compared with
healthy top-60 players over the same two-year window:

| Group | n | Availability next year | Minutes change | PPG ratio (median) | Per-game value change | Keep ≥95% of PPG |
|---|---|---|---|---|---|---|
| All injury-hit stars | 51 | 0.69 | −3.1 | 0.90 | −1.54 | 39% |
| Healthy stars | 298 | 0.80 | −1.2 | 0.96 | −0.81 | 52% |
| **Durable** (≥75% GP in both prior years) injured | 23 | 0.65 | −3.6 | **0.80** | −1.86 | 30% |
| Durable healthy | 171 | 0.80 | −1.4 | 0.96 | −1.02 | 53% |

By age (injured vs healthy): ≤28: availability 0.70 vs 0.81, PPG ratio 0.91 vs 1.02. Huge spread: Kawhi 2016, Towns, SGA, LaVine, Capela, Covington held steady; Hayward, Thomas, Oladipo, Beal fell hard.
**So the user's "stars don't drop much" holds for some individuals but not on average.** Our model's backtest projections for these cases (PPG 18.1 projected vs 19.4 actual among those who played ≥500 min — a survivor-biased sample) are in line with history.
The data has **no injury type/severity**, which is exactly the distinction the user draws (acute one-off vs recurrent) — the model cannot see it.

**Whole-season-lost returners (`returner_gap.py`; relevant to Haliburton, Kyrie, Lillard, VanVleet — all missed 2025-26 and are projected at availability 0.37–0.49 / 27.5–29.9 MPG / board ranks 157 (Haliburton), 209 (Lillard), 258 (Kyrie), 294 (VanVleet)).**
Rotation players (≥25 MPG, ≥30 GP two seasons earlier) with zero games the prior season, 2017–2025: 40 rows → 24 still in the league → 20 played (83%); availability **0.48 overall (0.58 if they played)**;
minutes −7.7 (median −9.2; −4.9 for ≥28-MPG players); 4 stayed out another year. **The model's numbers are in line with this history** (0.37–0.49 projected vs 0.48 historical mean availability; 27.5–29.9 MPG vs prior 33.6–36.1). (An earlier narrow-window look suggested 0.69 and "9 of 9 under-projected" — that was an artifact of
a 5-season window in which everyone happened to return healthy; do not repeat it.)

**"Usually healthy, one bad year" (`availability_bias.py`, selection-free: players who barely played count as 0).** Bad year *last* year after healthy earlier years (n=58): projected 0.65 vs actual 0.65 — calibrated.
**Recovered** (healthy last year after a wrecked year in the prior two, rotation, n=94): projected 0.72 vs actual 0.77 (+0.05, 2.4–2.6 SE) — the one real, modest under-projection. A held-out "recovered" bump improves that group's MSE by only 2.4% and total error by 0.09%.
Consistently healthy players (n=227) are slightly *over*-projected (−0.02). KD's wrecked year is 4 seasons back, outside the model's 3-season memory.

**Team/roster context (user asked whether incoming/outgoing players and role changes are modelled): they are NOT.** The model uses only the player's own history; team is used only for the schedule.
Evidence that it matters: 22% of backtest player-seasons involve a team change; their typical miss is larger (minutes 17% of average vs 12% for stayers; points 25% vs 17%; assists 29% vs 21%), though the average bias is small (−0.15 MPG, −0.4 PPG).
A "vacated minutes / incoming minutes" feature is buildable from existing data (current `player.team_id` from PlayerIndex vs last season's team + minutes of players who left/arrived) — this is the original investigation-plan item #8 (vacated usage), never built.

## 7. Tried and rejected (don't redo without a new idea)

| Idea | Result |
|---|---|
| Smooth rookie curve in log(draft pick) instead of tier means | No better on held-out rookie classes (minutes MAE 4.88 vs 4.81) — kept tier means. |
| Widen talent σ in the simulation to fix over-confidence | Didn't fix the heavy lower tail; the real cause was the availability estimator (`c` 0.42 → 0.72). |
| 5-season "health/minutes track record" features (mean/min/max availability, healthy-season share, `recovered`, `one_bad_year` flags, peak MPG, MPG vs peak) | Availability MSE −7.5% → −7.9% vs baseline; minutes unchanged; recovered-group bias +0.052 → +0.048; named players' projections moved ≤0.03. **Reverted.** |
| "Returner" rule (lift availability/minutes for players with a lost last season to past returners' means, estimated on training seasons only) | Effectively a no-op (0.37 → 0.48 availability in early folds; current pool unchanged). **Reverted.** |
| Per-game candidate pool (`pergame_k`) to surface injury-discounted stars | Kept as an opt-in (helps e.g. Anthony Davis: board 32, per-game 7); does **not** surface Giannis (per-game rank 66). |

## 8. Known problems and areas for improvement (roughly by importance)

1. **Bots are unrealistic.** Each bot draws *independent* noise (validation: sd = 6 + 0.2·rank; engine internal: 3 + 0.12·rank) around our own value rank.
   In 300 simulated drafts a rank-16 player is taken anywhere from pick 7 to 28; rank-23 is still there after pick 45 about 1% of the time (`bot_spread_and_board.py`). Real drafters share one
   consensus (ESPN ADP) and cluster tightly. The engine exploits these artificial falls, which inflates its predicted gain and makes validation opponents too easy.
   → Use ESPN ADP + smaller, *shared* noise (`DraftEngine._bot_orders`, `mock.run_mock_draft`). Needs the ESPN fetch.
2. **Value board vs market.** The static board (`src/draft/values.py`) multiplies counting stats by projected availability (full discount) and is a *9-category z-sum*, which punishes punt-heavy stars
   (Giannis FT%/3PM). In a daily-lineup league with bench + waivers a missed game costs less than full output, so a full discount is probably too harsh for the *board* (the engine's simulation already
   models absences with bench cover; the board only drives candidate generation, bot ordering, lineup priority and the "best value left" baseline). → make the discount a setting (partial/none) and test it with
   `analysis.ranking_accuracy`; compare against ADP once available. Note: even with *no* discount Giannis is 66th — the rest is minutes and the punt profile.
3. **No team/roster/role context in the projections** (§6.1). Biggest identified model gap that *is* addressable with current data.
4. **No injury type/severity information**, so the model cannot separate acute one-off injuries from chronic ones. The pragmatic remedy is per-player overrides driven by the user's knowledge (see §9 decision A).
5. **Position eligibility is a heuristic** until `scripts/fetch_espn_positions.py` is run on a network that reaches ESPN (script's parsing is unit-tested on a synthetic payload but the
   live call has never run; slot-id mapping 0–4 = PG/SG/SF/PF/C is from memory — eyeball its printed sample).
6. **Validation is not fully leak-free.** The projections are rebuilt from earlier data per year, but `calibration.json` (dispersion, copula, `c`, talent σ, rookie σ) is fit on 2022–2025 and reused for every
   validation year; the report subtitle "built from pre-2025 data only" is therefore inaccurate for those constants. Also: per-stat model choice is selected on the same backtest that reports gains;
   decisions (the `c` fix, rookie-curve test, margin variant, this session's analyses) were made after seeing test results; validation pool/teams use who actually played (`last_year`, first-game team); position labels are today's.
   → refit `calibrate(...)` inside `validate_year` with only earlier seasons and a rookie-profile CSV from `st[year < Y]`; reserve a truly untouched holdout (the 2026-27 season itself, tracked in-season).
7. **Margin variant unconfirmed.** Re-run with fresh bot seeds / more drafts before adopting. If adopted: add a `recommended` column to `engine.recommend` (highest value-rank among candidates within 1 pt of the best)
   and have the app's "Best pick" use it, while still showing the pure win-probability table.
8. **Rookies are generic within a draft-slot bucket** (two top-5 picks are identical). Use `data/external/projection_overrides.csv` for rookie numbers the user trusts; consider external priors (college/international stats).
9. **Two-week matchups:** "out for the whole period" is drawn once per period, overstating double-absence in weeks 7 and 17 (`c` was measured on one-week windows). Week 7 ignores ~30 unscheduled Cup games (user's choice) and is the noisiest week.
10. **Greedy daily lineup** (static-value order) instead of an optimal per-day assignment; no IR/waiver/streaming/trade modelling.
11. **Model-library choice** (LightGBM vs XGBoost) was never benchmarked; low stakes. Adding XGBoost as a fourth candidate in `predict_all` is ~10 lines.
12. **Report hygiene:** the report does not yet include the injury-recovery / board-vs-market analyses from §6.1; `docs/REPORT.html` is 3.7 MB in git; the report's 8.4 "bug this check caught" section has hard-coded history numbers.

## 9. Next steps and pending decisions

**Decisions waiting on the user (they have not answered):**
- **A. Player adjustments.** Offer: a sheet or in-app editor that writes `data/external/projection_overrides.csv` (availability, minutes, per-game lines) so the user can encode what the data can't (KD post-Achilles, Paul George, Murray, Chet, Wemby, Kawhi, Giannis, Haliburton, Kyrie, Lillard, VanVleet…). Currently the CSV works but is manual.
- **B. Roster-change features** (vacated/incoming minutes, coaching/role context) — build next?
- **C. Push** the latest commits to `testing`? (Only commit/push when asked.)
- 13 or 14 draft rounds (is the IR slot drafted)? Does the 3-center cap count any **C-eligible** player (incl. PF/C types)? How many of 10 teams make the playoffs? Can they run the ESPN script (or send eligibility + ADP)? Any rookie projections they trust?
- Direction after the validation result: (a) engine as an explained second opinion, (b) invest in projection quality, (c) both?

**Suggested order of work:**
1. Get ESPN eligibility/ADP (user runs `scripts/fetch_espn_positions.py`) → rebuild projections + library; then **fix bot realism** (ADP + shared noise) and re-validate.
2. **Player adjustments workflow** (decision A) — fastest way to act on the user's injury knowledge; log which players were overridden.
3. **Roster-change / vacated-minutes features** (decision B), backtested the same rolling way; accept only if out-of-sample minutes/availability error improves for team-changers.
4. **Make the board's availability discount configurable** and choose it with `ranking_accuracy` + ADP agreement.
5. **Leak-free validation** (per-year calibration refit) and fresh seeds for the margin variant; decide whether to ship the margin tie-break as the app default.
6. Add the §6.1 analyses to `docs/REPORT.md` (new section + figures) and keep the report regenerated.
7. Engineering: CI running `pytest -q`; pin requirements; optional XGBoost comparison; SHAP for explanation.

## 10. Quick orientation for a new session

- Start with `docs/REPORT.md` §1 (summary), §10 (validation), §12 (limits), then this file's §6.1/§7. Then `src/draft/engine.py` (`DraftEngine.recommend`, `team_totals`, `evaluate`, `_bot_orders`) and `src/models/projection.py` (`predict_all`, `build_wide`, `feature_matrix`).
- To check the app end-to-end headlessly: copy `data/processed/nba.db`, set `NBA_DB`, create a league with `store.save_league`, drive `AppTest` (string multiselect options only).
- Tests (37) cover: schema invariants, snake order/persistence, shrinkage math, projection pipeline on synthetic history, rookie/returning-veteran handling, simulation moments and constraints, daily-lineup slot rules, center cap, schedule/matchup-period logic (incl. the Cup/All-Star merges), ESPN parsing, and the recommender's candidate-pool options.
- When changing models or the simulator, re-run: `build_projections.py` → `build_library.py` → `pytest -q` → (`validate_draft.py`) → `make_report.py`, restart the Streamlit server, and update this file's §6/§7/§8.
