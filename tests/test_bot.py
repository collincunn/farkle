from __future__ import annotations

import numpy as np
import pytest

from farkle.bot import (
    Action,
    DeterministicBotPlayer,
    EnvironmentState,
    StrategicDeterministicBotPlayer,
)
from farkle.core import (
    Game,
    RollStruct,
    TurnStateError,
)


@pytest.fixture
def env_state():
    return EnvironmentState(
        score_prior_to_turn=0,
        score_so_far_this_turn=0,
        potential_hold_score=3000,
        roll=RollStruct.fixed([1, 1, 1, 1, 1, 1]),
        frozen=np.array([0, 0, 0, 0, 0, 0], dtype=np.uint8),
        scoring_mask=np.array([True, True, True, True, True, True]),
        max_other_player_score=0,
    )


def test_environment_state_from_turn():
    game = Game(2)
    game.player_score_map = {0: 100, 1: 100}
    game_gen = game.start()
    turn = next(game_gen)
    with pytest.raises(TurnStateError):
        EnvironmentState.from_turn(turn)
    turn.roll(RollStruct.fixed([1, 1, 1, 1, 1, 1]))
    env = EnvironmentState.from_turn(turn)
    assert env.score_prior_to_turn == 100

    # Make it a gameless turn and try again
    turn.player_id = None
    turn.game = None
    env_no_game = EnvironmentState.from_turn(turn)
    assert env_no_game.score_prior_to_turn == 0


def test_environment_state_copy(env_state):
    env2 = env_state.copy_with_updates(score_so_far_this_turn=100)
    assert env2.score_so_far_this_turn == 100


def test_det_bot_player_bas_initialization(env_state):
    with pytest.raises(ValueError):
        DeterministicBotPlayer(-1)


def test_det_bot_player_simple(env_state):
    bot_low = DeterministicBotPlayer(1000)
    action = bot_low.action(env_state)
    expected_action = Action(None, True)
    assert action == expected_action

    bot_hi = DeterministicBotPlayer(4000)
    action = bot_hi.action(env_state)
    expected_action = Action(np.array([True] * 6), False)
    assert action == expected_action


def test_det_bot_endgame(env_state):
    env_state_endgame = env_state.copy_with_updates(
        max_other_player_score=10_000, score_prior_to_turn=0
    )
    bot = DeterministicBotPlayer(1000)
    _ = bot.action(env_state_endgame)
    assert bot.threshold > 10_000


def test_strat_det_bot_simple(env_state):
    bot = StrategicDeterministicBotPlayer(1000)

    all_scoring_over_threshold = env_state.copy_with_updates(
        score_so_far_this_turn=10_000,
    )
    action = bot.action(all_scoring_over_threshold)
    assert not action.hold


def test_action_eq():
    assert not Action(None, False) == "action"


def test_strat_det_bot_fives_logic(env_state):
    bot = StrategicDeterministicBotPlayer(5000)

    single_fives_all_scoring = env_state.copy_with_updates(
        roll=RollStruct.fixed([1, 1, 1, 1, 1, 5]), scoring_mask=np.array([True] * 6)
    )
    action = bot.action(single_fives_all_scoring)
    assert action.mask[5]

    three_fives = env_state.copy_with_updates(
        roll=RollStruct.fixed([5, 5, 5, 4]), scoring_mask=np.array([True] * 3 + [False])
    )

    action = bot.action(three_fives)
    np.testing.assert_array_equal(action.mask, [True] * 3 + [False])

    should_drop_five = env_state.copy_with_updates(
        roll=RollStruct.fixed([5, 5, 4, 4, 6]),
        scoring_mask=np.array([True] * 2 + [False] * 3),
    )
    action = bot.action(should_drop_five)
    np.testing.assert_array_equal(action.mask, [True] + [False] * 4)

    # drop five
    action = bot.action(three_fives)
    np.testing.assert_array_equal(action.mask, [True] * 3 + [False])

    should_drop_all_fives = env_state.copy_with_updates(
        roll=RollStruct.fixed([1, 5, 5, 4, 4, 6]),
        scoring_mask=np.array([True] * 3 + [False] * 3),
    )
    action = bot.action(should_drop_all_fives)
    np.testing.assert_array_equal(action.mask, [True] + [False] * 5)
