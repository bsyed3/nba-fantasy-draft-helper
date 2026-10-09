"""Phase 1: baseline talent projection models.

XGBoost/LightGBM regressors for volume stats, Ridge regression for FG%/FT%,
rookie cold-start (comp-based) model. Outputs per-player mean (mu) and
variance (sigma) per category for Phase 2.
"""
