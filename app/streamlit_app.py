"""Streamlit dashboard for the FIFA World Cup 2026 Fantasy optimizer.

Run with:  streamlit run app/streamlit_app.py

Upload a price CSV (name, country, position, price) or use the built-in demo
data. Pick a tournament stage to see the optimal squad, or use the Live tab to
get transfer + booster recommendations for the next round.
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import pandas as pd
import streamlit as st

# Allow running without installing the package (src/ layout).
SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

warnings.filterwarnings("ignore")

from fantasy.data import make_seed_history, make_seed_price_list
from fantasy.ingest.prices import load_price_list
from fantasy.optimize.squad import optimize_squad
from fantasy.pipeline.live import recommend_update
from fantasy.pipeline.run import build_players
from fantasy.rules import Stage

st.set_page_config(page_title="WC2026 Fantasy Optimizer", layout="wide")
st.title("FIFA World Cup 2026 Fantasy Optimizer")


@st.cache_data
def _demo_data():
    history = make_seed_history()
    prices = make_seed_price_list(history)
    return history, prices


@st.cache_data
def _processed_history():
    from fantasy.ingest.storage import load_table
    try:
        return load_table("player_history")
    except FileNotFoundError:
        return None


def _player_table(players, ids=None):
    rows = []
    for p in players:
        if ids is not None and p.player_id not in ids:
            continue
        rows.append({
            "Pos": p.position.value, "Name": p.name, "Country": p.country,
            "Price": p.price, "EP": round(p.ep, 2),
        })
    return pd.DataFrame(rows)


# --- Sidebar: data + stage -------------------------------------------------
st.sidebar.header("Data")
use_demo = st.sidebar.toggle("Use demo data", value=True)

uploaded = None
if not use_demo:
    uploaded = st.sidebar.file_uploader("Price list CSV", type=["csv"])

stage = Stage(st.sidebar.selectbox("Tournament stage", [s.value for s in Stage], index=0))

if use_demo:
    history, price_df = _demo_data()
else:
    history = _processed_history()
    if history is None:
        st.warning("No processed history found. Run `python -m fantasy scrape` first, or use demo data.")
        st.stop()
    if uploaded is None:
        st.info("Upload a price list CSV to continue.")
        st.stop()
    tmp = Path("/tmp/_uploaded_prices.csv")
    tmp.write_bytes(uploaded.getvalue())
    price_df = load_price_list(tmp)

players, report = build_players(price_df, history)
st.sidebar.metric("Entity match rate", f"{report['match_rate']:.0%}", f"{report['matched']}/{report['total']}")

tab_squad, tab_live = st.tabs(["Optimal Squad", "Live: Transfers & Boosters"])

# --- Tab 1: optimal squad --------------------------------------------------
with tab_squad:
    sol = optimize_squad(players, stage=stage)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Formation", f"{sol.formation[0]}-{sol.formation[1]}-{sol.formation[2]}")
    c2.metric("Squad cost", f"${sol.total_cost:.1f}m")
    c3.metric("Expected pts (XI + C)", f"{sol.expected_points:.1f}")
    c4.metric("Captain", sol.captain.name)

    left, right = st.columns(2)
    with left:
        st.subheader("Starting XI")
        st.dataframe(_player_table(sol.starting_xi), hide_index=True, width="stretch")
    with right:
        st.subheader("Bench")
        st.dataframe(_player_table(sol.bench), hide_index=True, width="stretch")
        st.caption(f"Vice-captain: {sol.vice_captain.name}")

# --- Tab 2: live transfers + boosters --------------------------------------
with tab_live:
    st.write("Pick your current squad, then get transfer + booster recommendations for the next round.")
    base = optimize_squad(players, stage=stage)
    default_ids = [p.player_id for p in base.squad]
    id_to_name = {p.player_id: f"{p.name} ({p.country})" for p in players}

    current_ids = st.multiselect(
        "Current 15-man squad",
        options=list(id_to_name.keys()),
        default=default_ids,
        format_func=lambda i: id_to_name.get(i, i),
    )
    free = st.number_input("Free transfers (-1 = unlimited)", min_value=-1, value=2, step=1)

    if len(current_ids) != 15:
        st.info(f"Select exactly 15 players (currently {len(current_ids)}).")
    else:
        free_val = None if free < 0 else int(free)
        update = recommend_update(set(current_ids), price_df, history, stage, free_val)
        plan = update["transfer_plan"]
        c1, c2, c3 = st.columns(3)
        c1.metric("Transfers", plan.num_transfers)
        c2.metric("Point hit", f"-{plan.point_hit}")
        c3.metric("Net expected pts", f"{plan.net_expected_points:.1f}")

        if plan.transfers_in:
            st.subheader("Recommended transfers")
            st.dataframe(pd.DataFrame({
                "OUT": [p.name for p in plan.transfers_out],
                "IN": [p.name for p in plan.transfers_in],
                "IN EP": [round(p.ep, 2) for p in plan.transfers_in],
            }), hide_index=True, width="stretch")
        else:
            st.success("No transfers recommended - your squad is already optimal.")

        st.subheader("Booster ranking")
        st.dataframe(pd.DataFrame([{
            "Booster": b.booster,
            "Est. gain": "n/a" if b.estimated_gain is None else f"{b.estimated_gain:+.2f}",
            "Detail": b.detail,
        } for b in update["boosters"]]), hide_index=True, width="stretch")
