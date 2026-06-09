"""Entity-matching and price-loading tests."""

import pandas as pd

from fantasy.data import make_seed_history, make_seed_price_list
from fantasy.ingest.matching import match_players, match_report
from fantasy.ingest.prices import _normalize_position, load_price_list


def test_position_normalization():
    assert _normalize_position("Goalkeeper") == "GK"
    assert _normalize_position("def") == "DEF"
    assert _normalize_position("MID") == "MID"
    assert _normalize_position("Forward") == "FWD"


def test_load_price_list_roundtrip(tmp_path):
    df = pd.DataFrame({
        "name": ["A", "B"],
        "country": ["Brazil", "France"],
        "position": ["Forward", "GK"],
        "price": [10.5, 5.0],
    })
    p = tmp_path / "prices.csv"
    df.to_csv(p, index=False)
    loaded = load_price_list(p)
    assert list(loaded["position"]) == ["FWD", "GK"]
    assert "player_id" in loaded.columns
    assert loaded["price"].tolist() == [10.5, 5.0]


def test_load_real_format_prices_and_country_codes(tmp_path):
    # Mirrors players.csv: $-prefixed prices, FIFA codes, no position column.
    df = pd.DataFrame({
        "Name": ["Kane", "Mbappé", "Ronaldo"],
        "Country": ["ENG", "FRA", "POR"],
        "Price": ["$10.5m", "$10m", "$8m"],
    })
    p = tmp_path / "players.csv"
    df.to_csv(p, index=False)
    loaded = load_price_list(p)
    assert loaded["price"].tolist() == [10.5, 10.0, 8.0]
    assert loaded["country"].tolist() == ["england", "france", "portugal"]
    assert loaded["country_code"].tolist() == ["ENG", "FRA", "POR"]
    assert loaded["position"].isna().all()  # filled later from FBref


def test_fuzzy_match_handles_name_noise():
    history = make_seed_history()
    prices = make_seed_price_list(history)
    matched = match_players(prices, history)
    report = match_report(matched)
    # Names only have suffix noise + same country, so match rate should be high.
    assert report["match_rate"] >= 0.9


def test_override_forces_match():
    history = make_seed_history()
    prices = make_seed_price_list(history).head(1).copy()
    prices["player_id"] = ["x0"]
    target = history.iloc[0]["player_id"]
    matched = match_players(prices, history, overrides={"x0": target})
    assert matched.iloc[0]["matched_id"] == target
    assert matched.iloc[0]["match_method"] == "override"
