import json
import os

import numpy as np
import pytest

from farkle.core import (
    Game,
    RollStruct,
    Turn,
    TurnState,
    TurnStateError,
)

PARAMETERS_FILENAME = "test_parameters.json"
tests_root = os.path.abspath(os.path.dirname(__file__))
parameters_file = os.path.join(tests_root, PARAMETERS_FILENAME)


def pytest_generate_tests(metafunc):
    """We search for any functions needing data and parameterize the data."""
    for fixture in metafunc.fixturenames:
        if fixture.startswith("data__"):
            tests = load_tests(fixture)
            metafunc.parametrize(fixture, tests)


def load_tests(fixture):
    """Pull parameters file and deserialize."""
    with open(parameters_file) as f:
        test_data = json.loads(f.read())
    test_data_key = fixture.removeprefix("data__")
    return test_data[test_data_key]


def test_roll_struct_scoring(data__scoring):
    """Test scoring algorithm."""
    roll = data__scoring["roll"]
    expected_score = data__scoring["score"]
    expected_keep = data__scoring["keep"]

    expected_keep_arr = np.array(expected_keep).astype(bool)
    roll_struct = RollStruct.fixed(roll)
    score, keep = roll_struct.score()
    assert score == expected_score
    np.testing.assert_array_equal(expected_keep_arr, keep)


def test_roll_struct_fixed():
    # Normal
    rs = RollStruct.fixed([1, 2, 3, 4, 5, 6])
    assert rs.counts == [1, 1, 1, 1, 1, 1]
    # Small
    rs = RollStruct.fixed([1])
    assert rs.counts == [1]
    # None
    with pytest.raises(ValueError):
        RollStruct.fixed([])
    # Too many
    with pytest.raises(ValueError):
        RollStruct.fixed([1] * 7)
    # Multidimensional
    with pytest.raises(ValueError):
        RollStruct.fixed([[1] * 7])
    # Bad values
    with pytest.raises(ValueError):
        RollStruct.fixed([-1])
    with pytest.raises(ValueError):
        RollStruct.fixed([7])
    with pytest.raises(ValueError):
        RollStruct.fixed([0])


def test_roll_struct_random():
    # Normal
    rs = RollStruct.random(6)
    assert rs.shape == 6


def test_roll_struct_sub_roll():
    rs = RollStruct.fixed([1, 2, 3, 4, 5, 6])
    sub_rs = rs.subroll(np.array([True] + [False] * 5))
    value, _ = sub_rs.score()
    assert value == 100


def test_turn_state_terminal():
    assert not TurnState.ROLL.is_terminal()
    assert not TurnState.SELECTION.is_terminal()
    assert TurnState.FARKLE.is_terminal()
    assert TurnState.HOLD.is_terminal()
    assert TurnState.ERROR.is_terminal()


def test_turn_state_assert():
    class FakeTurn:
        state: TurnState

    def needs_state(turn) -> None:
        return

    turn = FakeTurn()
    turn.state = TurnState.ROLL
    # Passes
    f = TurnState.ROLL.assert_state(needs_state)
    f(turn)
    # Fails
    turn.state = TurnState.SELECTION
    with pytest.raises(TurnStateError):
        f = TurnState.ROLL.assert_state(needs_state)
        f(turn)


def test_turn_complex():
    turn = Turn()
    # Partial scoring
    rs1 = RollStruct.fixed([1, 5, 3, 3, 4, 4])
    turn.roll(rs1)
    turn.select(np.array([True] + [False] * 5))
    # Sub roll
    rs2 = RollStruct.fixed([5, 2, 2, 3, 4])
    turn.roll(rs2)
    turn.select(turn.curr_scoring_mask)
    # Loop around
    rs3 = RollStruct.fixed([5, 5, 5, 5])
    turn.roll(rs3)
    turn.select()
    assert turn.open_spots == 6
    # Stop playing
    rs4 = RollStruct.fixed([6, 2, 3, 4, 6, 2])
    turn.roll(rs4)
    assert turn.state == TurnState.FARKLE


def test_turn_broken_operations_no_fail():
    turn = Turn()
    # add additional dice which will break roll structure
    turn.frozen = np.zeros((7,))
    turn.roll()
    assert turn.state == TurnState.ERROR
    turn = Turn()
    rs = RollStruct.fixed([1, 2, 3, 4, 5, 6])
    turn.roll(rs)
    # add additional dice which will break roll structure
    turn.frozen = np.zeros((7,))
    turn.curr_scoring_mask = np.array([False] * 7)
    turn.select(np.array([False] * 7))
    assert turn.state == TurnState.ERROR


def test_turn_continue_after_done():
    turn = Turn()
    rs1 = RollStruct.fixed([1, 5, 3, 3, 4, 4])
    turn.roll(rs1)
    turn.hold()
    with pytest.raises(TurnStateError):
        turn.state = TurnState.ROLL


def test_turn_borken_mask():
    turn = Turn()
    rs1 = RollStruct.fixed([1, 5, 3, 3, 4, 4])
    turn.roll(rs1)
    # Try to pull non-scoring die
    with pytest.raises(ValueError):
        turn.select(np.array([False] * 5 + [True]))
    # Try to selection with wrong mask shape
    with pytest.raises(ValueError):
        turn.select(np.array([False]))
    with pytest.raises(ValueError):
        turn.select(np.array([False] * 10))


def test_turn_check_mask():
    turn = Turn()
    rs1 = RollStruct.fixed([1, 5, 3, 3, 4, 4])
    turn.roll(rs1)
    score, _ = turn.check_mask(np.array([False] * 5 + [True]))
    assert score == 0
    score, _ = turn.check_mask(np.array([True] + [False] * 5))
    assert score == 100
    score, _ = turn.check_mask()
    assert score == 150


def test_game_simple():
    game = Game(2)
    game_gen = game.start()
    turn1 = next(game_gen)
    rs1 = RollStruct.fixed([1, 5, 3, 3, 4, 4])
    turn1.roll(rs1)
    turn1.hold()
    turn2 = next(game_gen)
    rs2 = RollStruct.fixed([1, 2, 3, 4, 5, 6])
    turn2.roll(rs2)
    turn2.hold()
    next(game_gen)
    assert game.player_score_map == {0: 150, 1: 1500}


def test_game_ending():
    game = Game(2)
    game_gen = game.start()
    turn1 = next(game_gen)
    turn1.value = 10_000
    turn1.state = TurnState.HOLD
    # Should allow one more turn but no more
    next(game_gen)
    with pytest.raises(StopIteration):
        next(game_gen)
