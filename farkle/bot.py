from __future__ import annotations

from abc import ABCMeta, abstractmethod
from dataclasses import dataclass, fields
from typing import TYPE_CHECKING, Any

import numpy as np

from farkle.core import TurnState, TurnStateError

if TYPE_CHECKING:
    from farkle.core import (
        MaskType,
        RollArrayType,
        RollStruct,
        Turn,
    )


def _dataclass_asdict_shallow(obj: EnvironmentState) -> dict:
    return {field.name: getattr(obj, field.name) for field in fields(obj)}


@dataclass(frozen=True, slots=True)
class EnvironmentState:
    """All relevant environment state.

    Attributes:
        score_prior_to_turn: Player's score prior to this turn.
        score_so_far_this_turn: The current score accumulated this turn,
            excluding the current roll.
        potential_hold_score: The score that will be yielded from holding the
            current roll.
        roll: Dice available to select.
        frozen: Dice that have been held so far.
        max_other_player_score: Max score of other player. Used to deduce how
            close we are to the game ending. This gives bots the ability to
            make riskier moves towards the game's conclusion.
    """

    score_prior_to_turn: int
    score_so_far_this_turn: int
    potential_hold_score: int
    roll: RollStruct
    frozen: RollArrayType
    scoring_mask: MaskType
    max_other_player_score: int

    @classmethod
    def from_turn(cls, turn: Turn) -> EnvironmentState:
        """Create environment dynamically from core assets.

        Gather data from this game and turn for a specific player.

        Args:
            game: Current game
            player_id: Current player's ID.
            turn: Player's turn. Should be in ``SELECTION`` state.
        """
        if turn.state != TurnState.SELECTION:
            raise TurnStateError(
                "Must be in SELECTION state to create bot environment state"
            )
        game = turn.game
        player_id = turn.player_id
        score_so_far_this_turn = turn.value
        potential_hold_score = turn.curr_value + score_so_far_this_turn
        scoring_mask = turn.curr_scoring_mask
        roll = turn.curr_roll
        frozen = turn.frozen
        if game and player_id is not None:
            score_prior_to_turn = game.player_score_map[player_id]
            max_other_player_score = max(
                [v for id, v in game.player_score_map.items() if id != player_id]
            )
        else:
            score_prior_to_turn = max_other_player_score = 0
        return cls(
            score_prior_to_turn=score_prior_to_turn,
            score_so_far_this_turn=score_so_far_this_turn,
            potential_hold_score=potential_hold_score,
            roll=roll,
            frozen=frozen,
            scoring_mask=scoring_mask,
            max_other_player_score=max_other_player_score,
        )

    def copy_with_updates(self, **kwargs) -> EnvironmentState:
        other_dict = _dataclass_asdict_shallow(self)
        other_dict.update(kwargs)
        return EnvironmentState(**other_dict)


@dataclass(frozen=True, slots=True)
class Action:
    """Represents a game action.

    Attributes:
        mask: Mask for current roll. Should match shape of
            ``EnvironmentState.roll.shape``, with ``True``s for the dice to
            hold and ``False``s for dice staged for another roll.
        hold: Flag that indicates that the user will stop rolling and keep
            current score.
    """

    mask: MaskType | None
    hold: bool

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, Action):
            if self.hold:
                return other.hold
            return (
                other.mask is not None
                and self.mask is not None
                and np.array_equal(self.mask, other.mask)
            )
        return False


class BotPlayer(metaclass=ABCMeta):
    """Abstract base class for programmatic game players (aka bots).

    Bots, both deterministic and statistical, should inherit from this super
    class.
    """

    @abstractmethod
    def action(self, state: EnvironmentState) -> Action:
        """Mechanism through which the bot interacts with environment.

        Bot receives data about the current state of the game as and returns
        its decided action based on this data.

        Args:
            state: Current state of the game.

        Returns:
            Action: The action the bot has selected.
        """


class DeterministicBotPlayer(BotPlayer):
    """Simple threshold-based bot.

    This bot always plays based on the score accumulated so far this turn. If
    it is greater than a certain threshold, the bot always hold. Otherwise,
    it always rolls again — even when scoring is very low probability.

    Args:
        threshold: Score at which to hold.
    """

    threshold: int

    def __init__(self, threshold: int) -> None:
        if threshold <= 0:
            raise ValueError("Threshold should be a positive integer.")
        self.threshold = threshold

    def _update_threshold_if_endgame(self, state: EnvironmentState) -> None:
        """Change threshold if end game.

        If this is the last round, the player should continue rolling until
        they beat the player currently winning. Holding less than that value
        serves no purpose as the game will end this turn.
        """
        if state.max_other_player_score >= 10_000:
            diff = state.max_other_player_score - state.score_prior_to_turn
            self.threshold = diff + 1

    def action(self, state: EnvironmentState) -> Action:
        """Mechanism through which the bot interacts with environment.

        Bot receives data about the current state of the game as and returns
        its decided action based on this data.

        Args:
            state: Current state of the game.

        Returns:
            Action: The action the bot has selected.
        """
        self._update_threshold_if_endgame(state)
        if state.potential_hold_score < self.threshold:
            return Action(state.scoring_mask, False)
        return Action(None, True)


class StrategicDeterministicBotPlayer(DeterministicBotPlayer):
    """Threshold bot player with strategies.

    Plays based on score threshold with the addition of strategies. These
    strategies are listed as a series of ``staticmethod``s in this class. We
    apply strategies by updating the state.

    Args:
        threshold: Score at which to hold.
    """

    threshold: int

    @staticmethod
    def _always_roll_if_all_scoring(state: EnvironmentState) -> EnvironmentState:
        """Trick superclass into rolling if fresh set of dice.

        The probability of scoring with a full set of dice is so high that we
        should continue rolling even if the threshold is already met. We trick
        the superclass into always rolling in this case by setting the
        environment's ``score_so_far_this_turn`` to be zero, which used in the
        comparison against threshold. This is okay because state is ephemeral
        and only used once per action.
        """
        if np.all(state.scoring_mask):
            return state.copy_with_updates(potential_hold_score=0)
        return state

    @staticmethod
    def _drop_single_fives(state: EnvironmentState) -> EnvironmentState:
        """Drops single fives if beneficial.

        Fives are often not worth holding. This function codifies the following
        strategy:
        - All scoring: keep all. loops us to a full new round. Also, captures
          where fives are part of a larger scoring structure.
        - More than two fives: keep them as they are not single.
        - Only fives: keep one.
        - Else: drop  all fives.
        """
        roll = state.roll
        scoring_mask = state.scoring_mask
        is_five_mask = roll.roll_array == 5

        # All scoring
        if np.all(scoring_mask):
            return state
        # More than 2 fives
        if (num_fives := is_five_mask.sum()) >= 3:
            return state
        # Only fives and has one to drop
        if np.array_equal(is_five_mask, scoring_mask) and num_fives > 1:
            one_five = np.argmax(is_five_mask)
            one_five_mask = np.zeros_like(scoring_mask)
            one_five_mask[one_five] = True
            new_state = state.copy_with_updates(scoring_mask=one_five_mask)
        # Drop singles
        else:
            exclude_fives_mask = scoring_mask & ~is_five_mask
            new_state = state.copy_with_updates(scoring_mask=exclude_fives_mask)
        return new_state

    def action(self, state: EnvironmentState) -> Action:
        """Mechanism through which the bot interacts with environment.

        Bot receives data about the current state of the game as and returns
        its decided action based on this data.

        Args:
            state: Current state of the game.

        Returns:
            Action: The action the bot has selected.
        """
        state = StrategicDeterministicBotPlayer._drop_single_fives(state)
        state = StrategicDeterministicBotPlayer._always_roll_if_all_scoring(state)
        return super().action(state)
