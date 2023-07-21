import numpy as np
import pytest

from farkle.core import (
    Action,
    BasePlayer,
    EnvironmentState,
    FixedRollFactory,
    Game,
    Roll,
    Turn,
    TurnState,
    TurnStateError,
)


def test_roll_struct_scoring(data__scoring):
    """Test scoring algorithm."""
    roll = data__scoring["roll"]
    expected_score = data__scoring["score"]
    expected_keep = data__scoring["keep"]

    expected_keep_arr = np.array(expected_keep).astype(bool)
    roll_struct = Roll.fixed(roll)
    assert expected_score == roll_struct.score
    np.testing.assert_array_equal(expected_keep_arr, roll_struct.scoring_mask)


def test_roll_struct_fixed():
    # Normal
    rs = Roll.fixed([1, 2, 3, 4, 5, 6])
    assert rs.counts == [1, 1, 1, 1, 1, 1]
    # Small
    rs = Roll.fixed([1])
    assert rs.counts == [1]
    # None
    with pytest.raises(ValueError):
        Roll.fixed([])
    # Too many
    with pytest.raises(ValueError):
        Roll.fixed([1] * 7)
    # Multidimensional
    with pytest.raises(ValueError):
        Roll.fixed([[1] * 7])
    # Bad values
    with pytest.raises(ValueError):
        Roll.fixed([-1])
    with pytest.raises(ValueError):
        Roll.fixed([7])
    with pytest.raises(ValueError):
        Roll.fixed([0])


def test_roll_struct_random():
    # Normal
    rs = Roll.random(6)
    assert rs.shape == 6


def test_roll_struct_sub_roll():
    rs = Roll.fixed([1, 2, 3, 4, 5, 6])
    sub_rs = rs.subset(np.array([True] + [False] * 5))
    assert sub_rs.score == 100


def test_turn_state_assert():
    class FakeTurn:
        state: TurnState

    def needs_state(turn) -> None:
        return

    turn = FakeTurn()
    turn.state = TurnState.PLAYING
    # Passes
    f = TurnState.PLAYING.assert_state(needs_state)
    f(turn)
    # Fails
    turn.state = TurnState.FARKLE
    with pytest.raises(TurnStateError):
        f = TurnState.PLAYING.assert_state(needs_state)
        f(turn)


def test_turn_complex():
    rs1 = Roll.fixed([1, 5, 3, 3, 4, 4])
    roll_factory = FixedRollFactory(rs1)
    turn = Turn(roll_factory=roll_factory)
    # Partial scoring
    rs2 = Roll.fixed([5, 2, 2, 3, 4])
    roll_factory.set_next_roll(rs2)
    turn.select(np.array([True] + [False] * 5))
    # Sub roll
    rs3 = Roll.fixed([5, 5, 5, 5])
    roll_factory.set_next_roll(rs3)
    turn.select()
    # Loop around
    rs4 = Roll.fixed([6, 2, 3, 4, 6, 2])
    roll_factory.set_next_roll(rs4)
    turn.select()
    assert turn.open_spots == 6
    # Stop playing
    assert turn.state == TurnState.FARKLE


def test_turn_broken_operations_no_fail():
    # break during roll
    turn = Turn()
    turn.frozen = np.zeros((7,))
    turn._update_roll(None)

    # break during select
    roll_factory = FixedRollFactory(Roll.fixed([1, 1, 1, 1, 1, 1]))
    turn = Turn(roll_factory=roll_factory)
    turn.frozen = np.zeros((7,))
    bad_scoring_mask = np.array([True] * 7)
    object.__setattr__(turn.roll, "scoring_mask", bad_scoring_mask)
    turn.select(bad_scoring_mask)
    assert turn.state == TurnState.ERROR


def test_turn_continue_after_done():
    rs1 = Roll.fixed([1, 5, 3, 3, 4, 4])
    roll_factory = FixedRollFactory(rs1)
    turn = Turn(roll_factory=roll_factory)
    turn.hold()
    with pytest.raises(TurnStateError):
        turn.state = TurnState.PLAYING


def test_turn_broken_mask():
    rs = Roll.fixed([1, 5, 3, 3, 4, 4])
    roll_factory = FixedRollFactory(rs)
    turn = Turn(roll_factory=roll_factory)
    # Try to pull non-scoring die
    with pytest.raises(ValueError):
        turn.select(np.array([False] * 5 + [True]))
    # Try to selection with wrong mask shape
    with pytest.raises(ValueError):
        turn.select(np.array([False]))
    with pytest.raises(ValueError):
        turn.select(np.array([False] * 10))
    # Selected none
    with pytest.raises(ValueError):
        turn.select(np.array([False] * 6))


def test_turn_randomized_play():
    for i in range(100):
        turn = Turn()
        while turn.is_active():
            turn.select()


@pytest.fixture
def env_state():
    return EnvironmentState(
        score_prior_to_turn=0,
        score_so_far_this_turn=0,
        potential_hold_score=3000,
        roll=Roll.fixed([1, 1, 1, 1, 1, 1]),
        frozen=np.array([0, 0, 0, 0, 0, 0], dtype=np.uint8),
        scoring_mask=np.array([True, True, True, True, True, True]),
        max_other_player_score=0,
    )


def test_environment_state_bad_state(env_state):
    turn = Turn()
    turn.state = TurnState.FARKLE
    with pytest.raises(TurnStateError):
        EnvironmentState.from_turn(turn, {})


def test_environment_state_copy(env_state):
    env2 = env_state.copy_with_updates(score_so_far_this_turn=100)
    assert env2.score_so_far_this_turn == 100


def test_action_eq():
    assert Action(None, False) == Action(None, False)
    assert Action(np.array([True]), False) == Action(np.array([True]), False)
    assert not Action(np.array([False]), False) == Action(np.array([True]), False)
    assert not Action(None, False) == Action(None, True)
    assert not Action(None, False) == "action"


def test_simple_random_game():
    class Player1(BasePlayer):
        score = 0

        def action(self, state):
            print("hotdog")
            if state.score_so_far_this_turn >= 100:
                return Action(None, True)
            return Action(None, False)

    game = Game([Player1(), Player1()])
    game.play()


def test_catch_broken_player_in_game(env_state):
    class Player1(BasePlayer):
        score = 0
        action = None

        def handle_turn_generator(self, turn, state):
            self.score += 10_000
            yield env_state

    game = Game([Player1()])
    game.play()
