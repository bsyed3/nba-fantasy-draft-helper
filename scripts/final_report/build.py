"""
Build the final project report: Markdown + PDF (figures included), written to
C:\\Users\\SyedB\\Personal\\NBA Fantasy\\NBA Draft Model Report\\ and copied into the repository at docs/final_report/.

    python -m scripts.final_report.build                 # everything (and publish to docs/final_report/)
    python -m scripts.final_report.build --only sec_a    # one section group (for development); skips the PDF and publishing
    python -m scripts.final_report.build --no-pdf        # Markdown only (publishes the Markdown and figures, not the PDF)
    python -m scripts.final_report.build --publish-only  # copy the last build into docs/final_report/

The PDF is printed from HTML by Microsoft Edge in headless mode (no extra packages needed).
"""
from __future__ import annotations

import argparse
import importlib
import pickle
import re
import shutil
import subprocess
import sys
from pathlib import Path

import markdown

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding="utf-8")

from scripts.final_report import common  # noqa: E402

SECTIONS = ["sec_a", "sec_b", "sec_c", "sec_d", "sec_e"]
EDGE = [r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe", r"C:\Program Files\Microsoft\Edge\Application\msedge.exe"]
TITLE = "Projecting, Simulating and Drafting: A Statistical Model for a 10-Team, 9-Category Head-to-Head Fantasy Basketball League"
NAME = "NBA_Draft_Model_Report"
REPO_COPY = ROOT / "docs" / "final_report"

CSS = """
@page { size: Letter; margin: 0.8in 0.75in; }
body { font-family: 'Segoe UI', 'Helvetica Neue', Arial, sans-serif; font-size: 10.4pt; line-height: 1.45; color: #1b1b1b; max-width: 7.2in; margin: 0 auto; }
h1 { font-size: 19pt; margin-top: 30pt; border-bottom: 2px solid #2a78d6; padding-bottom: 4pt; page-break-before: always; }
h1.title { page-break-before: avoid; font-size: 23pt; border: none; margin-top: 60pt; line-height: 1.2; }
h2 { font-size: 14pt; margin-top: 20pt; color: #184f95; }
h3 { font-size: 11.5pt; margin-top: 14pt; color: #333; }
img { max-width: 100%; display: block; margin: 8pt auto 2pt auto; page-break-inside: avoid; }
p em:only-child { display: block; font-size: 9pt; color: #555; margin: 0 0 10pt 0; }
table { border-collapse: collapse; font-size: 8.6pt; margin: 8pt 0; width: 100%; page-break-inside: avoid; }
th, td { border: 1px solid #d3d3d3; padding: 2.5pt 5pt; text-align: left; vertical-align: top; }
th { background: #eef3fa; }
tr:nth-child(even) td { background: #fafafa; }
code { font-family: Consolas, monospace; font-size: 9pt; background: #f2f2f2; padding: 0 2pt; }
blockquote { margin: 6pt 0 6pt 8pt; padding: 4pt 10pt; border-left: 3px solid #2a78d6; background: #f5f8fc; }
.abstract { border: 1px solid #c7d6ea; background: #f7faff; padding: 8pt 12pt; margin: 14pt 0; }
.meta { color: #555; font-size: 10pt; }
"""


def to_html(md_text: str) -> str:
    body = markdown.markdown(md_text, extensions=["tables", "fenced_code", "sane_lists", "md_in_html"])
    body = body.replace("<h1>", "<h1 class='title'>", 1)                 # the first h1 is the report title (no page break before it)
    return f"<!doctype html><html><head><meta charset='utf-8'><title>{TITLE}</title><style>{CSS}</style></head><body>{body}</body></html>"


def to_pdf(html_path: Path, pdf_path: Path) -> bool:
    edge = next((p for p in EDGE if Path(p).exists()), None)
    if edge is None:
        print("Edge not found: PDF skipped (Markdown and HTML were written)")
        return False
    cmd = [edge, "--headless=new", "--disable-gpu", "--no-pdf-header-footer", f"--print-to-pdf={pdf_path}", html_path.as_uri()]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    ok = pdf_path.exists() and pdf_path.stat().st_size > 10_000
    print(("wrote " + str(pdf_path)) if ok else f"PDF failed: {r.stderr[-300:]}")
    return ok


def main(only: str | None, pdf: bool) -> None:
    ctx = common.load()
    s_path = common.OUT / "_S.pkl"
    if only and s_path.exists():                                   # numbers collected by earlier sections (so one section can be rebuilt alone)
        ctx.S.update(pickle.loads(s_path.read_bytes()))
    parts = {}
    for name in ([only] if only else SECTIONS):
        mod = importlib.import_module(f"scripts.final_report.{name}")
        print(f"building {name} ...", flush=True)
        parts[name] = mod.build(ctx)
        common.OUT.mkdir(parents=True, exist_ok=True)
        s_path.write_bytes(pickle.dumps(ctx.S))
    if only:
        (common.OUT / f"_{only}.md").write_text(parts[only], encoding="utf-8")
        print(f"wrote {common.OUT / f'_{only}.md'} ({ctx.nfig} figures)")
        return
    front = importlib.import_module("scripts.final_report.front").build(ctx)           # title page, abstract, contents (needs ctx.S)
    md = front + "\n".join(parts[n] for n in SECTIONS) + importlib.import_module("scripts.final_report.back").build(ctx)
    common.OUT.mkdir(parents=True, exist_ok=True)
    md_path = common.OUT / f"{NAME}.md"
    md_path.write_text(md, encoding="utf-8")
    html_path = common.OUT / f"{NAME}.html"
    html_path.write_text(to_html(md), encoding="utf-8")
    print(f"wrote {md_path} ({ctx.nfig} figures, {len(md.split()):,} words)")
    if pdf:
        to_pdf(html_path, common.OUT / f"{NAME}.pdf")
    publish(include_pdf=pdf)


def publish(include_pdf: bool = True) -> None:
    """Copy the Markdown, the PDF and the figures into the repository (docs/final_report/) so the committed report always matches the last build."""
    REPO_COPY.mkdir(parents=True, exist_ok=True)
    names = [f"{NAME}.md"] + ([f"{NAME}.pdf"] if include_pdf else [])
    for n in names:
        shutil.copy2(common.OUT / n, REPO_COPY / n)
    shutil.copytree(common.FIGS, REPO_COPY / "figures", dirs_exist_ok=True)
    print(f"published {', '.join(names)} and {len(list(common.FIGS.glob('*.png')))} figures to {REPO_COPY}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", default=None)
    ap.add_argument("--no-pdf", action="store_true")
    ap.add_argument("--publish-only", action="store_true", help="just copy the last build into docs/final_report/")
    a = ap.parse_args()
    if a.publish_only:
        publish()
    else:
        main(a.only, not a.no_pdf)
