"""Offline tests for the FBref league-table combiner (no network).

Feeds synthetic soccerdata-style frames (MultiIndex columns + player-keys index)
through `combine_stat_frames` to validate column mapping, position/country
normalization, and that duplicate stat names across tables don't collide.
"""

import pandas as pd

from fantasy.ingest.fbref import KEYS, combine_stat_frames
from fantasy.ingest.schema import PLAYER_SEASON_COLUMNS

IDX = pd.MultiIndex.from_tuples(
    [
        ("Big 5", "2526", "Barcelona", "Lamine Yamal"),
        ("Big 5", "2526", "Liverpool", "Keeper Guy"),
    ],
    names=KEYS,
)


def _frames():
    standard = pd.DataFrame(
        [["es ESP", "FW", 30, 2500, 27.8, 9, 6, 3, 0],
         ["br BRA", "GK", 28, 2520, 28.0, 0, 0, 1, 0]],
        index=IDX,
        columns=pd.MultiIndex.from_tuples([
            ("m", "nation"), ("m", "pos"),
            ("Playing Time", "MP"), ("Playing Time", "Min"), ("Playing Time", "90s"),
            ("Performance", "Gls"), ("Performance", "Ast"),
            ("Performance", "CrdY"), ("Performance", "CrdR"),
        ]),
    )
    shooting = pd.DataFrame([[20], [0]], index=IDX,
                            columns=pd.MultiIndex.from_tuples([("Standard", "SoT")]))
    passing = pd.DataFrame([[55], [3]], index=IDX,
                           columns=pd.MultiIndex.from_tuples([("x", "KP")]))
    defense = pd.DataFrame([[18], [2]], index=IDX,
                           columns=pd.MultiIndex.from_tuples([("Tackles", "Tkl")]))
    misc = pd.DataFrame([[0, 2, 1], [0, 0, 0]], index=IDX,
                        columns=pd.MultiIndex.from_tuples([
                            ("Performance", "OG"), ("Performance", "PKwon"),
                            ("Performance", "PKcon")]))
    keeper = pd.DataFrame([[0, 0, 0, 0], [80, 3, 20, 10]], index=IDX,
                          columns=pd.MultiIndex.from_tuples([
                              ("Performance", "Saves"), ("Performance", "PKsv"),
                              ("Performance", "GA"), ("Performance", "CS")]))
    return {"standard": standard, "shooting": shooting, "passing": passing,
            "defense": defense, "misc": misc, "keeper": keeper}


def test_combine_maps_to_canonical_schema():
    out = combine_stat_frames(_frames(), "Big 5", "2526")
    assert list(out.columns) == PLAYER_SEASON_COLUMNS
    assert len(out) == 2

    yamal = out[out["player"] == "Lamine Yamal"].iloc[0]
    assert yamal["country"] == "spain"
    assert yamal["position"] == "FWD"
    assert yamal["goals"] == 9
    assert yamal["shots_on_target"] == 20
    assert yamal["key_passes"] == 55
    assert yamal["tackles"] == 18
    assert yamal["pens_won"] == 2
    assert yamal["player_id"] == "Lamine Yamal|spain"


def test_combine_handles_goalkeeper_and_missing_stats():
    out = combine_stat_frames(_frames(), "Big 5", "2526")
    gk = out[out["player"] == "Keeper Guy"].iloc[0]
    assert gk["position"] == "GK"
    assert gk["country"] == "brazil"
    assert gk["saves"] == 80
    assert gk["clean_sheets"] == 10
    assert gk["goals_against"] == 20
    assert gk["shots_on_target"] == 0


def test_combine_survives_missing_stat_table():
    frames = _frames()
    frames["keeper"] = pd.DataFrame()  # simulate a category that failed to fetch
    out = combine_stat_frames(frames, "Big 5", "2526")
    assert (out["saves"] == 0).all()
    assert list(out.columns) == PLAYER_SEASON_COLUMNS
