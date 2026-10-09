# Investigation Phase Plan

Goal: before writing any modeling code, produce one chart or statistic per
model choice below, so every design decision in the proposal is backed by
evidence instead of intuition. This doubles as the "methodology
justification" material for the write-up.

Each task lists: the decision it justifies, the data needed, the
analysis/chart to produce, and what result would actually support the
choice (so a null result is a real, useful finding — not a failed task).

## How to use this

Assign tasks by column (they're mostly independent), track status as you
go, and drop each finished chart + one-line takeaway into `notebooks/`.
A task whose result *doesn't* support the proposed choice isn't wasted
work — it's a reason to simplify (e.g., drop a feature, use a plainer
model), which is exactly what a "be critical" reviewer will want to see.

| # | Decision to justify | Data needed | Analysis / chart | Result that supports the choice |
|---|---|---|---|---|
| 1 | Median (not mean) for weekly aggregation | Per-week fantasy stat totals per player | Distribution/histogram per stat, check skew | Clear left-skew from DNP/injury weeks |
| 2 | Negative Binomial (not Normal/Poisson) for counting stats | Per-game logs: STL, BLK, TOV, 3PM | Mean-vs-variance scatter per player/stat, with a variance=mean reference line | Points sit well above the line (overdispersion) |
| 3 | Copula correlation structure (not independent draws) | Per-game box scores, all 9 categories | Correlation heatmap; check stability across seasons | Meaningful pairwise correlations that hold up season to season |
| 4 | Shrinkage strength (the *k* in the Gamma-Poisson formula) | Player rate stats, bucketed by games/minutes played | Year-over-year rate correlation ("reliability") by sample-size bucket | Correlation clearly rises with sample size; the ~0.5 crossing point becomes your *k* |
| 5 | Non-linear aging curve; XGBoost for volume stats, Ridge for FG%/FT% | Stat output by age, all historical player-seasons | Age vs. average output curve, separately for volume stats and for FG%/FT% | Volume stats show a non-monotonic curve (trees justified); FG%/FT% curve is flatter (Ridge is enough) |
| 6 | Rolling time-series split (not random k-fold) | Same model, evaluated both ways | Compare error: random split vs. rolling-origin split | Random split reports noticeably better (inflated) accuracy — evidence of leakage |
| 7 | Contract-year flag as a feature | Per-minute production, walk year vs. other seasons, same players | Before/after comparison + effect size/significance test | A real, measurable bump — if not, drop the feature before building the scraper |
| 8 | Vacated-usage feature | Departed teammate's usage vs. returning players' usage change next season | Scatter/correlation | Confirms the relationship actually holds before you engineer it |
| 9 | Rookie variance by draft-capital tier | Historical rookie-season stats, grouped by draft slot tier | Box plot of coefficient of variation (σ/μ) by tier | CV clearly decreases for higher draft picks |
| 10 | Simplified games-played/injury bootstrap | Games played per season, per player, across history | Histogram; year-over-year comparison per player | Stable, unimodal pattern (if bimodal/volatile instead, note it as a stated limitation rather than overselling the simple model) |

## Data sources

- **nba_api**: player game logs (`playergamelogs`), league-wide season
  stats (`leaguedashplayerstats`), schedule/game density
  (`scheduleleaguev2` or similar — confirm the full season schedule is
  published before you need it; it's typically not final until
  mid-to-late August).
- **Not covered by nba_api** — flag these early since they need a separate
  source or manual collection:
  - Contract-year status → Spotrac/HoopsHype (scrape or manually maintain
    a CSV for the player pool you actually need).
  - Injury history (dates, games missed, type) → no clean free API;
    Pro Sports Transactions or a manually compiled log is the realistic
    option. Given #7 and #10 above might show the fancier version isn't
    even justified, don't build this scraper until those results are in.

## Suggested split of work

Two independent tracks, so you and your partner can work in parallel:

- **Track A — Projection justification** (#4, #5, #6, #7, #8, #9): feeds
  directly into Phase 1 model choices.
- **Track B — Simulation justification** (#1, #2, #3, #10): feeds directly
  into Phase 2's copula design.

## Exit criteria for this phase

Move to Milestone 1 (Phase 1 build) once: every row above has a chart and
a one-line takeaway, and any feature/technique that *didn't* hold up
(e.g., no contract-year effect, no vacated-usage relationship) has been
explicitly dropped or scoped down rather than carried forward "just in
case."
