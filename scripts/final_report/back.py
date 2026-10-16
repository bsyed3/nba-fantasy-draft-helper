"""References and appendices."""
from __future__ import annotations

import pandas as pd

from scripts.final_report.common import *  # noqa: F403
from src.models import projection as P


def build(ctx) -> str:
    S = ctx.S
    params = S["params"]
    ptab = pd.DataFrame([{"statistic": STAT_LABEL[s], "recency decay": params[s][0], "prior strength k": params[s][1]} for s in P.ALL_STATS])
    return f"""
# References

**Statistical methods**

1. Efron, B. and Morris, C. (1975). Data analysis using Stein's estimator and its generalizations. *Journal of the American Statistical Association*, 70(350), 311-319. (Shrinkage toward a prior mean, illustrated with baseball batting averages: the idea behind section 3.)
2. James, W. and Stein, C. (1961). Estimation with quadratic loss. *Proceedings of the Fourth Berkeley Symposium*, 1, 361-379.
3. Gelman, A., Carlin, J., Stern, H., Dunson, D., Vehtari, A. and Rubin, D. (2013). *Bayesian Data Analysis* (3rd ed.). CRC Press. (Beta-binomial and empirical-Bayes estimation of proportions.)
4. Hoerl, A. and Kennard, R. (1970). Ridge regression: biased estimation for nonorthogonal problems. *Technometrics*, 12(1), 55-67.
5. Ke, G. et al. (2017). LightGBM: a highly efficient gradient boosting decision tree. *Advances in Neural Information Processing Systems*, 30.
6. Koenker, R. and Bassett, G. (1978). Regression quantiles. *Econometrica*, 46(1), 33-50.
7. Hastie, T., Tibshirani, R. and Friedman, J. (2009). *The Elements of Statistical Learning* (2nd ed.). Springer. (Cross-validation, ridge regression, boosting; the caution about selecting a model with the data used to evaluate it, relevant to section 10.5.)
8. Nelsen, R. (2006). *An Introduction to Copulas* (2nd ed.). Springer. (Gaussian copula used in section 9.)
9. Dawid, A. P. (1984). Statistical theory: the prequential approach. *Journal of the Royal Statistical Society A*, 147(2), 278-292. (The probability-integral transform as a calibration check, section 9.6.)
10. Brier, G. (1950). Verification of forecasts expressed in terms of probability. *Monthly Weather Review*, 78(1), 1-3.
11. Hyndman, R. and Athanasopoulos, G. (2021). *Forecasting: Principles and Practice* (3rd ed.). OTexts. (Time-series cross-validation: the idea of the walk-forward backtest, section 4.2.)

**Data and software**

12. NBA statistics API accessed through the `nba_api` Python package (S. Patel et al.): endpoints `LeagueGameLog`, `PlayerIndex`, `LeagueDashPlayerBioStats`, `ScheduleLeagueV2`. Data © the NBA; used here for personal, non-commercial analysis.
13. ESPN injury report, espn.com/nba/injuries, read 14 October 2026 (players with an "Out" designation).
14. Yahoo Sports draft-class projections for rookies and the blended average-draft-position workbook (Yahoo, ESPN, Fantrax) dated 14 October 2026, supplied by the project owner; used for comparison only.
15. Software: Python 3.14, pandas, NumPy, SciPy, scikit-learn, LightGBM, Matplotlib, SQLite, Streamlit.

# Appendix A. Reproducing the results

From the repository root (`github.com/bsyed3/nba-fantasy-draft-helper`, branch `testing`), after creating a virtual environment and installing the requirements:

```
python scripts/ingest.py                 # box scores 2015-16 to latest + the 2026-27 schedule -> data/processed/nba.db
python scripts/build_projections.py      # walk-forward backtest + 2026-27 projections (projections_2026.parquet)
python scripts/build_library.py          # calibration, simulated weeks (21 periods incl. playoffs), policy table
python scripts/validate_draft.py --playoffs --arms adaptive   # mock drafts scored on real seasons 2021-2025
python -m scripts.final_report.build     # this report (Markdown + HTML + PDF) into the report folder
streamlit run src/app/streamlit_app.py  # the draft board
```

Data used by the backtest and the real-season validation never include the season being predicted: the walk-forward loop refits every model on seasons strictly before the test season. Random seeds are fixed in the scripts (e.g. policy table seed 24680, mock-draft seeds in the analysis scripts) so the numbers can be regenerated; the draft-engine experiments are the slow part (minutes to hours each, depending on the number of mock drafts).

# Appendix B. Fitted parameters of the shrinkage baseline

For each statistic the recency decay *d* (weight of a season *l* years back is *d*<sup>l−1</sup>) and prior strength *k* (equivalent number of attempts or minutes of prior information) were fitted by minimising out-of-sample error over the backtest folds.

{md_table(ptab, {"recency decay": "{:.2f}", "prior strength k": "{:.0f}"})}

# Appendix C. League settings used throughout

| Setting | Value |
|---|---|
| Platform and format | ESPN, snake draft, 10 teams, 13 rounds |
| Scoring | Head-to-head, 9 categories: PTS, 3PM, REB, AST, STL, BLK, FG%, FT%, TO (lower is better) |
| Win rule | A week is one win if a team wins more than 4.5 of 9 categories; margins are ignored |
| Season | 18 regular-season weeks + 3 playoff weeks (21 scoring periods) |
| Lineup | Daily; PG, SG, G, SF, PF, F, C, 3 UTIL; 3 bench; 1 IR; at most 3 centers |
| Management in scope | Draft only; no waivers, trades or free-agent pickups |
| Candidate pool | The top 200 players on the projection board |

# Appendix D. Glossary

* **ADP (average draft position).** The mean pick number at which a player is selected in many drafts; the *market order* ranks players by this value.
* **Availability.** Games played divided by the games a team played while the player was on its roster (after his first appearance), adjusted for known absences.
* **Backtest, walk-forward.** Repeatedly fit on seasons up to year *t*−1, predict season *t*, and score the prediction against what happened; moving *t* forward gives out-of-sample errors for several seasons.
* **Copula.** A way of attaching a dependence structure (here Gaussian) to variables whose individual (marginal) distributions are chosen separately.
* **Empirical Bayes / shrinkage.** Estimating each player's rate as a weighted average of his own data and a population prior, with the weight set by how much data he has.
* **Guard (margin).** A rule that the engine only departs from the board's best available player when its estimated win probability is higher by more than a stated amount.
* **MAE, RMSE.** Mean absolute error and root-mean-squared error of a prediction.
* **PIT (probability-integral transform).** The percentile of an observed value within its predicted distribution; uniform if the forecasts are calibrated.
* **Residual.** Actual minus projected value.
* **Roll-out.** Playing out the remainder of a draft (and the season) in simulation after a candidate pick to estimate how well it works.
* **Spearman correlation.** The correlation of the ranks of two sets of values; 1 means the same order.
* **z-score.** A value expressed in standard deviations from the mean of the player pool; the 9-category *value* is the sum of the nine z-scores (with turnovers reversed).
"""
