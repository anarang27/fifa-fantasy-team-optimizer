"""Hand-computed scoring tests verifying the engine against the rules table."""

from fantasy.rules import Position
from fantasy.scoring import MatchStats, score_match


def test_appearance_short_vs_long():
    assert score_match(MatchStats(Position.FWD, minutes=0)) == 0
    assert score_match(MatchStats(Position.FWD, minutes=30)) == 1
    assert score_match(MatchStats(Position.FWD, minutes=59)) == 1
    assert score_match(MatchStats(Position.FWD, minutes=60)) == 2
    assert score_match(MatchStats(Position.FWD, minutes=90)) == 2


def test_gk_clean_sheet_saves_and_penalty_save():
    # 90' + clean sheet(5) + 1 pen save(3) + 6 saves -> 2 -> total 12
    s = MatchStats(
        Position.GK, minutes=90, team_goals_conceded=0,
        saves=6, penalty_saves=1,
    )
    assert score_match(s) == 12


def test_gk_conceded_goals_penalty():
    # 90'(2) + conceded 3 -> -(3-1)=-2 + 3 saves -> 1  => 1
    s = MatchStats(Position.GK, minutes=90, team_goals_conceded=3, saves=3)
    assert score_match(s) == 1


def test_gk_scoring_a_goal():
    # 90'(2) + goal(9) + clean sheet(5) = 16
    s = MatchStats(Position.GK, minutes=90, team_goals_conceded=0, goals=1)
    assert score_match(s) == 16


def test_def_goal_and_clean_sheet():
    # 90'(2) + goal(7) + clean sheet(5) = 14
    s = MatchStats(Position.DEF, minutes=90, team_goals_conceded=0, goals=1)
    assert score_match(s) == 14


def test_def_conceded_and_yellow():
    # 90'(2) + conceded 2 -> -1 + yellow -1 = 0
    s = MatchStats(Position.DEF, minutes=90, team_goals_conceded=2, yellow_cards=1)
    assert score_match(s) == 0


def test_def_clean_sheet_requires_60_minutes():
    # 45'(1), conceded 0 but under 60 -> no clean sheet
    s = MatchStats(Position.DEF, minutes=45, team_goals_conceded=0)
    assert score_match(s) == 1


def test_mid_full_house():
    # 90'(2)+goal(6)+assist(3)+CS(1)+6 tackles->2 +4 chances->2 = 16
    s = MatchStats(
        Position.MID, minutes=90, team_goals_conceded=0,
        goals=1, assists=1, tackles=6, chances_created=4,
    )
    assert score_match(s) == 16


def test_mid_tackle_and_chance_thresholds_round_down():
    # 90'(2) + 5 tackles -> 1 + 3 chances -> 1, conceded 1 (no CS) = 4
    s = MatchStats(
        Position.MID, minutes=90, team_goals_conceded=1,
        tackles=5, chances_created=3,
    )
    assert score_match(s) == 4


def test_fwd_goal_and_shots():
    # 90'(2) + goal(5) + 5 SOT -> 2 = 9
    s = MatchStats(Position.FWD, minutes=90, goals=1, shots_on_target=5)
    assert score_match(s) == 9


def test_direct_free_kick_bonus():
    # MID 90'(2) + goal(6) + dfk bonus(1); conceded 1 so no clean sheet = 9
    s = MatchStats(
        Position.MID, minutes=90, team_goals_conceded=1,
        goals=1, direct_free_kick_goals=1,
    )
    assert score_match(s) == 9


def test_penalty_won_and_conceded():
    fwd = MatchStats(Position.FWD, minutes=70, goals=1, penalties_won=1, shots_on_target=3)
    # 2 + 5 + 2 + (3//2=1) = 10
    assert score_match(fwd) == 10


def test_disaster_game():
    # DEF 90'(2) + own goal(-2) + pen conceded(-1) + red(-2); conceded 1 -> 0
    s = MatchStats(
        Position.DEF, minutes=90, team_goals_conceded=1,
        own_goals=1, penalties_conceded=1, red_cards=1,
    )
    assert score_match(s) == -3
