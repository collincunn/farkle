from __future__ import annotations

import numpy as np
import pytest

from farkle.bots import DeterministicBotPlayer, StrategicDeterministicBotPlayer
from farkle.core import Action, Roll


def test_det_bot_player_bad_initialization(env_state):
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


def test_strat_det_bot_fives_logic(env_state):
    bot = StrategicDeterministicBotPlayer(5000)

    single_fives_all_scoring = env_state.copy_with_updates(
        roll=Roll.fixed([1, 1, 1, 1, 1, 5]), scoring_mask=np.array([True] * 6)
    )
    action = bot.action(single_fives_all_scoring)
    assert action.mask[5]

    three_fives = env_state.copy_with_updates(
        roll=Roll.fixed([5, 5, 5, 4]), scoring_mask=np.array([True] * 3 + [False])
    )

    action = bot.action(three_fives)
    np.testing.assert_array_equal(action.mask, [True] * 3 + [False])

    should_drop_five = env_state.copy_with_updates(
        roll=Roll.fixed([5, 5, 4, 4, 6]),
        scoring_mask=np.array([True] * 2 + [False] * 3),
    )
    action = bot.action(should_drop_five)
    np.testing.assert_array_equal(action.mask, [True] + [False] * 4)

    # drop five
    action = bot.action(three_fives)
    np.testing.assert_array_equal(action.mask, [True] * 3 + [False])

    should_drop_all_fives = env_state.copy_with_updates(
        roll=Roll.fixed([1, 5, 5, 4, 4, 6]),
        scoring_mask=np.array([True] * 3 + [False] * 3),
    )
    action = bot.action(should_drop_all_fives)
    np.testing.assert_array_equal(action.mask, [True] + [False] * 5)
