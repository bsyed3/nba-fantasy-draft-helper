"""Data ingestion layer.

Responsible for pulling and caching raw data:
- nba_api pulls (player game logs, league dash stats, schedule/game density)
- External sources not covered by nba_api (contract-year status, injury
  history) — see docs/investigation_plan.md for what's needed and why.

Nothing implemented yet — this phase starts with investigation (see
docs/investigation_plan.md) before any ingestion code is written.
"""
