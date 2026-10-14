"""
Turn a rookie-projection workbook (the 2026 draft class from Yahoo Sports / Fadeaway World: PTS, REB, AST, STL, BLK, FG%, 3P%, MIN per
game for the picks that have a published projection) into data/external/rookie_projections_yahoo.csv, which build_projections.py then uses
in place of the draft-slot profile for those rookies.

    python scripts/import_yahoo_rookies.py "C:/Users/you/Downloads/2026_nba_draft_class_projections.xlsx"

Expected layout: a header row containing Pick, Player, PTS, REB, AST, STL, BLK, FG%, 3P%, MIN (blank cells = not projected). Free throws,
3PM and turnovers are not in the source, so they keep the model's draft-slot values. Then re-run build_projections.py + build_library.py.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.stdout.reconfigure(encoding="utf-8")


def load_rookie_workbook(path: str) -> pd.DataFrame:
    raw = pd.read_excel(path, sheet_name=0, header=None)
    header_row = next(i for i in range(min(len(raw), 15)) if {"Pick", "Player", "PTS"} <= set(raw.iloc[i].astype(str)))
    df = pd.read_excel(path, sheet_name=0, header=header_row)
    df["Pick"] = pd.to_numeric(df["Pick"], errors="coerce")
    df = df[df["Pick"].notna() & df["PTS"].notna()]
    out = pd.DataFrame({
        "pick": df["Pick"].astype(int), "name": df["Player"].astype(str).str.strip(), "pos": df.get("Pos"), "team": df.get("Team (rights after draft)"),
        "pts": df["PTS"].astype(float), "reb": df["REB"].astype(float), "ast": df["AST"].astype(float), "stl": df["STL"].astype(float), "blk": df["BLK"].astype(float),
        "fg_pct": df["FG%"].astype(float), "tp_pct": df["3P%"].astype(float), "mpg": pd.to_numeric(df["MIN"], errors="coerce"),
    })
    return out.reset_index(drop=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("xlsx")
    ap.add_argument("--out", default=str(ROOT / "data/external/rookie_projections_yahoo.csv"))
    a = ap.parse_args()
    t = load_rookie_workbook(a.xlsx)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    t.to_csv(a.out, index=False)
    pd.set_option("display.width", 200)
    print(t.to_string(index=False))
    print(f"\nwrote {len(t)} rookie projections -> {a.out}\nNext: python scripts/build_projections.py && python scripts/build_library.py")
