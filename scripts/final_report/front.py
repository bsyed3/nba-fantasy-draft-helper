"""Title block, abstract, key findings at a glance and contents."""
from __future__ import annotations

import pandas as pd

from scripts.final_report.common import *  # noqa: F403

TITLE = "Projecting, Simulating and Drafting: A Statistical Model for a 10-Team, 9-Category Head-to-Head Fantasy Basketball League"


def build(ctx) -> str:
    S, p = ctx.S, ctx.proj
    rs = S["real"]["policy_new"]
    rl, fresh = S["realism"], S["fresh"]
    top = p.sort_values("rank").head(12)
    mk = ctx.market.set_index("player_id")["blend_order"] if ctx.market is not None else pd.Series(dtype=float)
    tab = pd.DataFrame({"our rank": top["rank"].astype(int), "player": top["name"], "age": top["age_t"], "proj PPG": top["pts"], "proj GP": top["avail"] * 82,
                        "market order": [mk.get(pid, float("nan")) for pid in top["player_id"]]})
    return f"""# {TITLE}

<p class="meta">A personal applied-statistics project, written up as a technical report<br>
Prepared 16 October 2026 · League: ESPN snake draft, 10 teams, 9 categories, head-to-head (most categories wins the week) · Season projected: 2026-27</p>

<div class="abstract" markdown="1">

**Abstract.** This report documents a draft assistant for a single fantasy-basketball league, built in three phases and tested on seasons it had not seen. *Phase 1* projects the 2026-27 per-game statistics of every player from {S["n_playergames"]:,} player-games of box scores (2015-16 to 2025-26) using a shrinkage (empirical-Bayes) baseline, regularised and gradient-boosted corrections chosen by a walk-forward backtest, a purpose-built definition of *availability*, team-context features, a rookie prior and four documented expert judgments. On {S["n_oos"]:,} out-of-sample player-seasons the projections beat "same as last year" in all nine categories and ranked better in every one of five test seasons (mean Spearman correlation with true fantasy value {S["spearman_model"]:.3f} against {S["spearman_last"]:.3f}). *Phase 2* converts projections into thousands of simulated weeks on the real 2026-27 calendar (negative-binomial counts, a Gaussian copula for dependence between statistics, whole-week absences); on four past seasons the simulated weekly spread was about 5% too narrow while the category dependence and matchup probabilities were well calibrated. *Phase 3* chooses each pick by rolling out the rest of the draft and maximising the probability of winning a weekly matchup; on real box scores from 2021-22 to 2025-26 the final pick policy gained an average of **{rs["mean"]:+.1f} points of weekly win probability** over "take the best player left" (standard error {rs["se"]:.1f}, positive in {rs["pos"]} of {rs["n"]} seasons). The projected ranking correlates {S["rho"]:.2f} with the market's average draft position; the main disagreements are structural (assumed absences, returners from missed seasons, young big men, aging guards) rather than random. The report also documents the places where the models had to be reconsidered, including the case of Giannis Antetokounmpo, who started at 93rd and finished at 6th after three separately justified changes.

</div>

## Key findings at a glance

* **Projection accuracy.** Out-of-sample, the projections reduce mean absolute error in all nine categories and improve the ranking of players in 5 of 5 test seasons (Spearman {S["spearman_model"]:.3f} vs {S["spearman_last"]:.3f}). Learned corrections help for minutes, shot attempts and availability and are rejected where they do not beat the baseline (rebounds, steals, free-throw %).
* **Availability is the largest source of error** and its definition matters more than any model: counting only games after a player's first appearance and treating known absences as inputs removes most of the systematic bias.
* **A pure z-score sum misleads at the extremes.** Flooring each category's z-score at −3 changes nobody in the top 150 except Giannis Antetokounmpo, whose free-throw percentage is an extreme outlier; together with two other documented changes it moved him from 103rd to 6th (section 8.3).
* **The simulation is reliable for ranking decisions, not for absolute probabilities**: its weeks are about 5% too narrow and its matchup probabilities about 5% too extreme.
* **Draft policy.** A category-aware policy beat "best available" by {rs["mean"]:+.1f} points of weekly win probability on real seasons (about +0.9 wins over the 18-week regular season). An early lead for the policy over the plain engine disappeared at full sample, and looking at all 200 candidates does not help (sections 10.6-10.7).
* **Against the market.** Rank correlation {S["rho"]:.2f}; median gap {S["med_gap"]:.0f} places in the drafted range. The model is lower on assumed-absent players, returners from missed seasons and young bigs, and higher on efficient role players (section 11).
* **Where to go next: role.** A first look at the box scores (section 14) suggests that players back from a missed season lose minutes but not usage per minute (part-season returners lose some usage too), that minutes are restricted for about ten games after a layoff (about two games' worth in total), that stayers' usage rises when a *high-usage* teammate leaves but not when role players do, and that same-position crowding costs guards and centers minutes beyond what the projection knows.

**The model's top 12 for 2026-27, with the market's order alongside**

{md_table(tab, {"our rank": "{:.0f}", "age": "{:.0f}", "proj PPG": "{:.1f}", "proj GP": "{:.0f}", "market order": "{:.0f}"})}

## Contents

1. Introduction (purpose, why this is a statistics problem, overview)
2. Data and sources
3. Phase 1, step 1: the player-season table and a shrinkage baseline
4. Phase 1, step 2: learned corrections and how they were tested (walk-forward backtest, residual diagnostics)
5. Phase 1, step 3: availability, the biggest source of error
6. Phase 1, step 4: team context
7. Rookies and expert inputs
8. Phase 1, step 5: from projections to a ranking (the value score and the Giannis dilemma)
9. Phase 2: from an average to a distribution of weeks (simulation and its realism test)
10. Phase 3: from distributions to a draft decision (engine, pick policy, validation on real seasons)
11. Results: the 2026-27 board against the market
12. Last season's biggest jumpers and fallers
13. Conclusion
14. Next steps: areas to explore (role and usage after injury, roster change, positional depth, recommendations and questions)
- References
- Appendices: A. Reproducing the results · B. Fitted parameters · C. League settings · D. Glossary
"""
