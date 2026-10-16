"""
Scatter plot: our model's board rank vs the market's blended ADP (average draft pick).

    python scripts/plot_rank_vs_adp.py "C:/path/fantasy_basketball_adp_2026-27.xlsx" --out exports/rank_vs_adp.png

COMPARISON ONLY. The ADP workbook (Hashtag Basketball: Yahoo / ESPN / Fantrax / blend) is read here and nowhere else: it is not a model input
(no bot ordering, no eligibility, nothing written to data/external/ or the database). Players are matched by normalised name, with the team
breaking ties; unmatched names are printed.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding="utf-8")

from src.features.names import normalize_name
from src.reporting import figs as F

TEAM_ALIAS = {"SA": "SAS", "GS": "GSW", "NY": "NYK", "NO": "NOP", "PHO": "PHX", "UTAH": "UTA", "WSH": "WAS", "BRK": "BKN", "CHO": "CHA"}


def read_adp(xlsx: str) -> pd.DataFrame:
    """One row per ranked player: name, team, blend_adp, blend_order, plus the three sites' ADP."""
    raw = pd.read_excel(xlsx, sheet_name=0, header=None)
    hdr = next(i for i in range(min(len(raw), 15)) if "Rank" in set(raw.iloc[i].astype(str)) and "Player" in set(raw.iloc[i].astype(str)))
    df = pd.read_excel(xlsx, sheet_name=0, header=hdr)
    # the three sites repeat Pos / ADP / Order; pandas suffixes the repeats .1 .2 .3
    out = pd.DataFrame({"rank": pd.to_numeric(df["Rank"], errors="coerce"), "name": df["Player"].astype(str).str.strip(), "team": df["Team"].astype(str).str.strip(),
                        "yahoo_adp": pd.to_numeric(df["ADP"], errors="coerce"), "espn_adp": pd.to_numeric(df["ADP.1"], errors="coerce"),
                        "fantrax_adp": pd.to_numeric(df["ADP.2"], errors="coerce"), "blend_adp": pd.to_numeric(df["ADP.3"], errors="coerce"),
                        "blend_order": pd.to_numeric(df["Order.3"], errors="coerce")})
    return out[out["rank"].notna() & out["blend_adp"].notna()].reset_index(drop=True)


def match_to_model(adp: pd.DataFrame, proj: pd.DataFrame, team_abbr: dict[int, str]) -> tuple[pd.DataFrame, list[str]]:
    proj = proj.assign(key=proj["name"].map(normalize_name), abbr=proj["team_id"].map(team_abbr))
    by_key = {k: g for k, g in proj.groupby("key")}
    rows, missing = [], []
    for _, r in adp.iterrows():
        g = by_key.get(normalize_name(r["name"]))
        if g is not None and len(g) > 1:
            g = g[g["abbr"] == TEAM_ALIAS.get(r["team"], r["team"])]
        if g is None or len(g) != 1:
            missing.append(f"{r['name']} ({r['team']})")
            continue
        m = g.iloc[0]
        rows.append({**r.to_dict(), "player_id": int(m["player_id"]), "model_name": m["name"], "model_rank": int(m["rank"]), "games_out": float(m.get("games_out", 0) or 0),
                     "source": m["source"]})
    return pd.DataFrame(rows), missing


def main(xlsx: str, out: str, top: int = 150, n_label: int = 8, highlight: list[str] | None = None) -> None:
    from nba_api.stats.static import teams as static_teams

    proj = pd.read_parquet(ROOT / "data/processed/projections_2026.parquet")
    abbr = {t["id"]: t["abbreviation"] for t in static_teams.get_teams()}
    d, missing = match_to_model(read_adp(xlsx), proj, abbr)
    print(f"matched {len(d)} of {len(d) + len(missing)} ADP players to the model; unmatched: {missing[:12]}{' ...' if len(missing) > 12 else ''}")
    d["gap"] = d["model_rank"] - d["blend_order"]                     # + = the model ranks him LOWER (later) than the market's blended order does
    # the source flags ADPs bunched near 140 as a placeholder for rarely drafted players, not a real draft position: leave them out
    d["placeholder"] = d["blend_adp"].between(139.0, 141.0)
    real = d[~d["placeholder"]]
    both = real[(real["blend_order"] <= top) | (real["model_rank"] <= top)].copy()
    drafted = real[real["blend_order"] <= 130]                         # what a 10-team, 13-round draft actually takes
    rho = both[["model_rank", "blend_order"]].corr(method="spearman").iloc[0, 1]
    rho_d = drafted[["model_rank", "blend_order"]].corr(method="spearman").iloc[0, 1]
    print(f"{int(d['placeholder'].sum())} placeholder-ADP players excluded; {len(both)} plotted (market order or model rank within {top})")
    print(f"Spearman: {rho:.3f} (plotted) / {rho_d:.3f} (market order <= 130, n={len(drafted)}); median |gap| in the drafted range: {drafted['gap'].abs().median():.0f} places; "
          f"within 10: {(drafted['gap'].abs() <= 10).mean():.0%}, within 20: {(drafted['gap'].abs() <= 20).mean():.0%}, within 40: {(drafted['gap'].abs() <= 40).mean():.0%}")
    top30 = real[real["blend_order"] <= 30]
    print(f"of the market's top 30, the model has {(top30['model_rank'] <= 30).sum()} in its top 30 and {(top30['model_rank'] <= 40).sum()} in its top 40")
    cols = ["model_name", "blend_order", "blend_adp", "model_rank", "gap", "games_out", "source"]
    print("model ranks him HIGHER than the market (drafted range):"); print(both[both["blend_order"] <= 150].nsmallest(12, "gap")[cols].round(1).to_string(index=False))
    print("model ranks him LOWER than the market (drafted range):"); print(both[both["blend_order"] <= 150].nlargest(14, "gap")[cols].round(1).to_string(index=False))
    both.drop(columns=["rank"]).rename(columns={"model_name": "player"}).to_csv(Path(out).with_suffix(".csv"), index=False)
    draw(both, out, rho_d, drafted, int(d["placeholder"].sum()), top, highlight=highlight)


def draw(d: pd.DataFrame, out: str, rho: float, drafted: pd.DataFrame, n_placeholder: int, top: int, lim: int = 165, thresh: float = 30.0, n_label: int = 8,
         highlight: list[str] | None = None) -> None:
    highlight = highlight or []
    from src.reporting.analysis import short_name

    import matplotlib.pyplot as plt

    C = F.C
    F.setup()
    W, H = 11.8, 7.5
    fig = plt.figure(figsize=(W, H))
    inr = d[(d["blend_order"] <= lim) & (d["model_rank"] <= lim)]
    off_low = d[(d["model_rank"] > lim) & (d["blend_order"] <= top)].sort_values("blend_order")        # the market ranks him highly, we have him beyond the chart
    off_high = d[(d["blend_order"] > lim) & (d["model_rank"] <= top)].sort_values("model_rank")        # we rank him highly, the market ranks him lower than that
    x, y, gap = inr["blend_order"].to_numpy(), inr["model_rank"].to_numpy(), inr["gap"].to_numpy()
    higher, lower = gap <= -thresh, gap >= thresh                        # the model ranks him much higher / much lower than the market
    n_h, n_l = int(higher.sum()), int(lower.sum())
    F.headline(fig, f"Our board tracks the market's order (rank correlation {rho:.2f}), but only {(drafted['gap'].abs() <= 20).mean():.0%} of drafted-range players land within 20 places of it",
               f"{len(inr)} players plotted; {len(off_low) + len(off_high)} more fall off the chart (listed at right) and {n_placeholder} placeholder-ADP players are excluded. "
               f"Above the dashed line the model ranks a player lower (later) than the market's blended order, below it higher.")
    F.legend_top(fig, [F.swatch(C["blue"], f"Model much higher than the market ({thresh:.0f}+ places, {n_h} players)", "dot"),
                       F.swatch(C["orange"], f"Model much lower ({thresh:.0f}+ places, {n_l} players)", "dot"),
                       F.swatch(F.BG_POINT, f"Within {thresh:.0f} places", "dot")])
    # explicit layout: a square plot under the header + legend band, the off-chart list to its right
    top_in = fig._header_in + fig._legend_in + 0.12
    left_in, bottom_in, gutter = 0.95, 0.85, 0.5
    side_in = H - top_in - bottom_in
    ax = fig.add_axes([left_in / W, bottom_in / H, side_in / W, side_in / H])
    side = fig.add_axes([(left_in + side_in + gutter) / W, bottom_in / H, (W - left_in - side_in - gutter - 0.15) / W, side_in / H])
    F.apply_layout = lambda f: None                                      # the axes are placed by hand; skip tight_layout
    pad = 8
    ax.plot([0, lim], [0, lim], color=C["axis"], lw=1.3, ls=(0, (4, 3)), zorder=1)
    ax.scatter(x[~higher & ~lower], y[~higher & ~lower], s=26, color=F.BG_POINT, edgecolor="white", linewidth=0.6, zorder=3)
    ax.scatter(x[higher], y[higher], s=34, color=C["blue"], edgecolor="white", linewidth=0.8, zorder=4)
    ax.scatter(x[lower], y[lower], s=34, color=C["orange"], edgecolor="white", linewidth=0.8, zorder=4)
    ax.set_xlim(0, lim + pad); ax.set_ylim(0, lim + pad); ax.set_aspect("equal")                  # a little room past the cut-off so labels near it stay inside
    ticks = list(range(0, lim + 1, 20))
    ax.set_xticks(ticks); ax.set_yticks(ticks)
    ax.set_xlabel("Market order: players ranked by blended ADP across Yahoo, ESPN and Fantrax (1 = first off the board)")
    ax.set_ylabel("Our model's board rank (1 = best)")
    ax.text(0.015, 0.985, "model ranks him LOWER than the market", transform=ax.transAxes, ha="left", va="top", fontsize=8.5, color=C["muted"])
    ax.text(0.985, 0.015, "model ranks him HIGHER than the market", transform=ax.transAxes, ha="right", va="bottom", fontsize=8.5, color=C["muted"])

    def nm(full: str) -> str:
        return "Giannis" if "Antetokounmpo" in full else f"{full.split()[0][0]}. {short_name(full)}"

    pick = inr[inr["blend_order"] <= 130]
    labels = [(r.blend_order, r.model_rank, nm(r.model_name), C["blue"]) for r in pick.nsmallest(n_label, "gap").itertuples()]
    labels += [(r.blend_order, r.model_rank, nm(r.model_name), C["orange"]) for r in pick.nlargest(n_label, "gap").itertuples()]
    g = inr[inr["model_name"].str.contains("Antetokounmpo")]
    if len(g):                                                           # the one you asked about, for reference
        labels.append((float(g["blend_order"].iloc[0]), float(g["model_rank"].iloc[0]), "Giannis", C["ink"]))
    for full in highlight:                                               # any others you ask about, in black (unless already labelled)
        h = inr[inr["model_name"] == full]
        if len(h) and not any((lx, ly) == (float(h["blend_order"].iloc[0]), float(h["model_rank"].iloc[0])) for lx, ly, _, _ in labels):
            labels.append((float(h["blend_order"].iloc[0]), float(h["model_rank"].iloc[0]), nm(full), C["ink"]))

    # side panel: the players that fall off the chart, with their real numbers
    side.axis("off"); side.grid(False)
    y0 = 1.0

    def block(title: str, lines: list[str], y_top: float) -> float:
        side.text(0.0, y_top, title, transform=side.transAxes, ha="left", va="top", fontsize=9, fontweight="bold", color=C["ink"])
        y_cur = y_top - 0.06
        for ln in lines:
            side.text(0.0, y_cur, ln, transform=side.transAxes, ha="left", va="top", fontsize=8.4, color=C["ink2"])
            y_cur -= 0.036
        return y_cur - 0.03

    low_lines = [f"{nm(r.model_name)}: market {int(r.blend_order)}, ours {int(r.model_rank)}" + (f" ({int(r.games_out)} games out)" if r.games_out >= 20 else "") for r in off_low.head(11).itertuples()]
    y0 = block(f"Our rank is beyond {lim} (market order within {top})", low_lines + ([f"+ {len(off_low) - 11} more"] if len(off_low) > 11 else []), y0)
    high_lines = [f"{nm(r.model_name)}: ours {int(r.model_rank)}, market {int(r.blend_order)}" for r in off_high.head(8).itertuples()]
    y0 = block(f"Market order is beyond {lim} (our rank within {top})", high_lines + ([f"+ {len(off_high) - 8} more"] if len(off_high) > 8 else []), y0)
    side.text(0.0, max(y0, 0.02), "Games out = a known absence we assume\n(ESPN Out list plus two hand-set).", transform=side.transAxes, ha="left", va="top", fontsize=8, color=C["muted"])

    F.label_points(ax, labels, fontsize=8.3, marker_size=64)
    F.FIG_DIR = Path(out).parent
    F.save(fig, Path(out).stem)
    print(f"wrote {Path(out)} ({len(labels)} labelled; {len(off_low)} off-chart low, {len(off_high)} off-chart high)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("xlsx")
    ap.add_argument("--out", default=str(ROOT / "exports/rank_vs_adp.png"))
    ap.add_argument("--top", type=int, default=150)
    ap.add_argument("--highlight", nargs="*", default=[], help="extra players to label in black, by full name as the model has it (e.g. \"Pascal Siakam\")")
    a = ap.parse_args()
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    main(a.xlsx, a.out, a.top, highlight=a.highlight)
