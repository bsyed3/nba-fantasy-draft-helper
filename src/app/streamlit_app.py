"""
NBA fantasy league helper (Streamlit).

    streamlit run src/app/streamlit_app.py

One league (ESPN 10-team, 9-category H2H), four pages:
  Teams            every team's projected weekly stats, category ranks and win probability
  Free agents      the free-agent pool compared with my roster, and where roster transactions are recorded
  Trade evaluator  what a trade does to both teams' win probability
  Draft            enter picks as they happen; the engine ranks the available players by simulated matchup win probability
"""
from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.app import common, page_draft, page_free_agents, page_teams, page_trades  # noqa: E402

st.set_page_config(page_title="NBA Fantasy League Helper", page_icon="🏀", layout="wide")

ctx = common.load_context()


def teams() -> None:
    page_teams.render(ctx)


def free_agents() -> None:
    page_free_agents.render(ctx)


def trades() -> None:
    page_trades.render(ctx)


def draft() -> None:
    page_draft.render(ctx)


# Once the draft is done the season pages come first and open by default; during the draft it is the other way round.
done = ctx.state.done
pages = [
    st.Page(draft, title="Draft", icon="📝", url_path="draft", default=not done),
    st.Page(teams, title="Teams", icon="📊", url_path="teams", default=done),
    st.Page(free_agents, title="Free agents", icon="🔎", url_path="free-agents"),
    st.Page(trades, title="Trade evaluator", icon="🔁", url_path="trades"),
]
if done:
    pages = pages[1:] + pages[:1]

st.sidebar.title("🏀 League Helper")
common.sidebar(ctx)
st.navigation(pages).run()
