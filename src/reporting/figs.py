"""
Figure style + helpers for the report (palette and chart rules from the dataviz guidance).

Layout rules enforced here so individual figures can't get them wrong:
  * the headline (takeaway) and grey subtitle are wrapped to the figure width and measured in inches;
  * legends live in their own band UNDER the header, never over the data (`legend_top`);
  * highlighted points get a solid marker with a white ring, labels are placed without overlapping
    each other or other highlights, and are tied to their point with a thin leader line (`label_points`).
"""
from __future__ import annotations

import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.text import Text

C = dict(blue="#2a78d6", orange="#eb6834", aqua="#1baf7a", yellow="#eda100", magenta="#e87ba4",
         green="#008300", violet="#4a3aa7", red="#e34948", ink="#0b0b0b", ink2="#52514e", muted="#898781",
         grid="#e1e0d9", axis="#c3c2b7", surface="#fcfcfb", mid="#f0efec")
SEQ = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
SEQ_CMAP = LinearSegmentedColormap.from_list("seq", SEQ)
DIV_CMAP = LinearSegmentedColormap.from_list("div", [C["red"], C["mid"], C["blue"]])
BG_POINT = "#b9c7da"            # muted cool grey-blue for the 'everyone else' cloud behind highlights

FIG_DIR = Path(__file__).resolve().parents[2] / "docs" / "report" / "figures"
TITLE_PT, SUB_PT = 12.5, 9.5
LEGEND_IN = 0.42               # height reserved for a legend band


def setup() -> None:
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Segoe UI", "DejaVu Sans", "Arial"],
        "figure.facecolor": C["surface"], "axes.facecolor": C["surface"], "savefig.facecolor": C["surface"],
        "axes.edgecolor": C["axis"], "axes.labelcolor": C["ink2"], "axes.titlecolor": C["ink"],
        "xtick.color": C["muted"], "ytick.color": C["muted"], "text.color": C["ink"],
        "axes.grid": True, "grid.color": C["grid"], "grid.linewidth": 0.6, "axes.axisbelow": True,
        "axes.spines.top": False, "axes.spines.right": False, "axes.spines.left": False,
        "axes.titlesize": 11, "axes.labelsize": 9.5, "xtick.labelsize": 9, "ytick.labelsize": 9,
        "legend.frameon": False, "legend.fontsize": 9, "figure.dpi": 100, "savefig.dpi": 150,
    })


def new(w: float = 8.0, h: float = 4.4, **kw):
    fig, ax = plt.subplots(figsize=(w, h), **kw)
    return fig, ax


def _wrap(text: str, width_in: float, pt: float, bold: bool = False) -> str:
    chars_per_inch = 72.0 / (pt * (0.60 if bold else 0.53))
    return "\n".join(textwrap.wrap(text, max(20, int(width_in * chars_per_inch))))


def headline(fig, text: str, sub: str | None = None) -> None:
    """Left-aligned takeaway title (+ grey subtitle), wrapped to the figure width. Records the header height."""
    w, h = fig.get_size_inches()
    t = _wrap(text, w - 0.3, TITLE_PT, bold=True)
    n_t = t.count("\n") + 1
    fig.text(0.012, 1 - 0.10 / h, t, ha="left", va="top", fontsize=TITLE_PT, fontweight="bold", color=C["ink"], linespacing=1.15)
    used = 0.10 + n_t * TITLE_PT * 1.25 / 72
    n_s = 0
    if sub:
        s = _wrap(sub, w - 0.3, SUB_PT)
        n_s = s.count("\n") + 1
        fig.text(0.012, 1 - (used + 0.05) / h, s, ha="left", va="top", fontsize=SUB_PT, color=C["ink2"], linespacing=1.2)
        used += 0.05 + n_s * SUB_PT * 1.3 / 72
    fig._header_in = used + 0.12


def legend_top(fig, handles=None, labels=None, ncol: int | None = None) -> None:
    """Legend in its own band under the header (never over the data). handles/labels default to the first axes'."""
    if handles is None:
        handles, labels = fig.axes[0].get_legend_handles_labels()
    w, h = fig.get_size_inches()
    y = 1 - getattr(fig, "_header_in", 0.7) / h
    n = len(handles)
    ncol = ncol or min(n, max(1, int(w // 2.6)))
    rows = -(-n // ncol)
    fig.legend(handles=handles, labels=labels, loc="upper left", bbox_to_anchor=(0.008, y), ncol=ncol,
               frameon=False, handlelength=1.5, columnspacing=1.8, borderaxespad=0.0, fontsize=9)
    fig._legend_in = 0.14 + 0.25 * rows


def swatch(color: str, label: str, kind: str = "patch"):
    """Legend handle: filled patch, line, or dot."""
    if kind == "line":
        return Line2D([0], [0], color=color, lw=2.4, label=label)
    if kind == "dot":
        return Line2D([0], [0], marker="o", color="none", markerfacecolor=color, markeredgecolor="white", markersize=9, label=label)
    return Patch(facecolor=color, label=label)


def apply_layout(fig) -> None:
    """tight_layout under the measured header (+ legend band)."""
    w, h = fig.get_size_inches()
    top = 1 - (getattr(fig, "_header_in", 0.7) + getattr(fig, "_legend_in", 0.0)) / h
    fig.tight_layout(rect=(0, 0, 1, top))


def save(fig, name: str, **_ignored) -> str:
    """Final layout, write the PNG; returns the path relative to docs/."""
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    apply_layout(fig)
    path = FIG_DIR / f"{name}.png"
    fig.savefig(path)
    plt.close(fig)
    return f"report/figures/{name}.png"


def label_points(ax, points, fontsize: float = 8.5, marker_size: float = 80) -> None:
    """
    Highlight and name points. points = [(x, y, text, color), ...].
    Each gets a solid coloured marker with a white ring. Labels are placed greedily, hardest first
    (points with the most close neighbours), at the first candidate offset whose box overlaps no earlier
    label, no highlighted marker, and stays inside the axes; each label is tied to its marker by a thin
    leader line. Call after the axes limits are final (the layout is applied first).
    """
    fig = ax.figure
    apply_layout(fig)
    for x, y, _, color in points:
        ax.scatter([x], [y], s=marker_size, color=color, edgecolor="white", linewidth=1.8, zorder=6)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    axbox = ax.get_window_extent(renderer)
    disp = [ax.transData.transform((x, y)) for x, y, _, _ in points]
    obstacles = [(px - 10, py - 10, px + 10, py + 10) for px, py in disp]          # the markers themselves

    # candidate label offsets in points: near first, then progressively farther, all directions
    ring1 = [(10, 9), (10, -15), (-10, 9), (-10, -15), (16, 0), (-16, 0), (0, 17), (0, -19)]
    ring2 = [(28, 16), (28, -22), (-28, 16), (-28, -22), (44, 4), (-44, 4), (4, 32), (4, -36)]
    ring3 = [(46, 30), (46, -40), (-46, 30), (-46, -40), (66, 10), (-66, 10), (12, 52), (12, -56)]
    offsets = ring1 + ring2 + ring3

    def overlaps(a, b) -> bool:
        return not (a[2] <= b[0] or a[0] >= b[2] or a[3] <= b[1] or a[1] >= b[3])

    def box_of(ann):
        bb = ann.get_window_extent(renderer)            # text + leader line
        return (bb.x0 - 3, bb.y0 - 3, bb.x1 + 3, bb.y1 + 3)

    # crowded points first so they get the nearest free slots
    crowd = [sum(np.hypot(dx - px, dy - py) < 70 for dx, dy in disp) for px, py in disp]
    order = sorted(range(len(points)), key=lambda i: -crowd[i])
    for i in order:
        x, y, text, color = points[i]
        placed = None
        for dx, dy in offsets:
            ha = "left" if dx > 3 else "right" if dx < -3 else "center"
            va = "bottom" if dy > 3 else "top" if dy < -3 else "center"
            ann = ax.annotate(text, (x, y), xytext=(dx, dy), textcoords="offset points", ha=ha, va=va, fontsize=fontsize,
                              color=color, fontweight="bold", zorder=7,
                              arrowprops=dict(arrowstyle="-", color=color, lw=0.9, shrinkA=0, shrinkB=5),
                              bbox=dict(boxstyle="round,pad=0.18", fc=C["surface"], ec="none", alpha=0.9))
            box = box_of(ann)
            # a leader line only conflicts with markers/labels other than its own marker
            own = obstacles[i]
            others = [o for o in obstacles if o is not own]
            inside = box[0] >= axbox.x0 and box[2] <= axbox.x1 and box[1] >= axbox.y0 and box[3] <= axbox.y1
            if inside and not any(overlaps(box, o) for o in others):
                placed = box
                break
            ann.remove()
        if placed is None:        # nothing fits cleanly: keep the label (don't drop information) at the first offset
            dx, dy = offsets[0]
            ann = ax.annotate(text, (x, y), xytext=(dx, dy), textcoords="offset points", fontsize=fontsize, color=color,
                              fontweight="bold", zorder=7, arrowprops=dict(arrowstyle="-", color=color, lw=0.9))
            placed = box_of(ann)
        obstacles.append(placed)


def md_table(df: pd.DataFrame, fmt: dict[str, str] | None = None, index: bool = False) -> str:
    """Minimal GitHub-markdown table from a DataFrame."""
    fmt = fmt or {}
    d = df.reset_index() if index else df
    cols = list(d.columns)
    lines = ["| " + " | ".join(str(c) for c in cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in d.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            if isinstance(v, (float, np.floating)):
                cells.append("" if np.isnan(v) else (fmt.get(c, "{:.2f}")).format(v))
            elif isinstance(v, (int, np.integer)) and c in fmt:
                cells.append(fmt[c].format(v))
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)
