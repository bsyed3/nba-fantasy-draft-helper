"""The outside ADP workbook (Hashtag Basketball) is a COMPARISON reference only: model code must never read or depend on it."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIRS = ["src/models", "src/draft", "src/features", "src/simulation"]
FORBIDDEN = ["adp_reference", "hashtag", "blend_adp", "fantasy_basketball_adp"]


def test_model_code_does_not_reference_the_outside_adp_file():
    hits = []
    for d in MODEL_DIRS:
        for p in (ROOT / d).rglob("*.py"):
            text = p.read_text(encoding="utf-8").lower()
            hits += [f"{p.relative_to(ROOT)}: {w}" for w in FORBIDDEN if w in text]
    assert not hits, f"outside ADP must stay out of the model: {hits}"


def test_the_pipelines_adp_input_only_comes_from_the_espn_positions_file():
    """`espn_adp` (what the draft bots follow) is filled solely from data/external/positions.csv, which the ADP workbook never writes to."""
    text = (ROOT / "src/features/positions.py").read_text(encoding="utf-8")
    assert "positions.csv" in text and "overrides_path" in text
    writers = [p.name for p in (ROOT / "scripts").glob("*.py") if "positions.csv" in p.read_text(encoding="utf-8") and "write" in p.read_text(encoding="utf-8").lower()]
    assert "plot_rank_vs_adp.py" not in writers
    assert "positions.csv" not in (ROOT / "scripts/plot_rank_vs_adp.py").read_text(encoding="utf-8")
