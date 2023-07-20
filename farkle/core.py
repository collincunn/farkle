from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from functools import wraps
from itertools import cycle
import logging
from typing import (
    Callable,
    Concatenate,
    Final,
    Generator,
    ParamSpec,
    TypeVar,
)

import numpy as np
from numpy.typing import ArrayLike, NDArray

MaskType = NDArray[np.bool_]
RollArrayType = NDArray[np.uint8]

_KEEP_ALL = np.array([True] * 6)


@dataclass(frozen=True)
class RollStruct:
    """Represents a roll of N (``shape``) dice.

    Dataclass representing a single roll of multiple dice in the game. In
    addition to the raw numbers, we store useful metadata about the roll for
    later scoring.

    Attributes:
        roll: Array representing the actual rolled values.
        counts: Counts of each value stored in ascending order.
        last_n: The value of the highest rolled value. Equivalent to the value
            that is represents the last index of ``counts``.
        shape: The number of dice that were rolled. Must be in ``[1, 6]``.
    """

    roll_array: RollArrayType
    counts: list[int]
    last_n: int
    shape: int

    @classmethod
    def fixed(cls, roll: ArrayLike) -> RollStruct:
        """Create a ``RollStruct`` from a specific set of values.

        This is used for fixing a specific combination of values as a roll,
        and calculating the metadata implicitly.

        Args:
            roll (array-like): The selected roll.

        Returns:
            RollStruct: Object representing roll.
        """
        roll = np.array(roll, dtype=np.uint8)

        if roll.size == 0:
            raise ValueError("Roll is empty")
        if roll.shape[0] > 6:
            raise ValueError("Too many dice!")
        if roll.ndim != 1:
            raise ValueError("Should be one-dimensional")
        if roll.min() < 1 or roll.max() > 6:
            raise ValueError("Values should be in [1,6]")

        unique, counts = np.unique(roll, return_counts=True)
        indices = np.argsort(counts)
        counts = counts[indices].tolist()
        last_n = unique[indices[-1]]
        return cls(roll, counts, last_n, roll.shape[0])

    @classmethod
    def random(cls, shape: int = 6) -> RollStruct:
        """Simulate a roll.

        Create a random roll wrapped in a ``RollStruct``.

        Args:
            shape: Number of dice to roll.

        Returns:
            RollStruct: Object representing roll.
        """
        roll = np.random.randint(1, 7, size=shape)
        return cls.fixed(roll)

    def subroll(self, mask: NDArray[np.bool_]) -> RollStruct:
        """Create a new object from a subset of this object."""
        return RollStruct.fixed(self.roll_array[mask])

    def score(self) -> tuple[int, MaskType]:
        """Calculate score and mask for a given ``RollStruct``.

        Use ``RollStruct`` metadata to calculate the score and choose which
        dice to hold that maximizes our score. The different scoring patterns
        are represented as blocks in the match-case below.

        Returns:
            int: The score for the roll.
            array of bool: Boolean mask of which dice we will hold onto.
        """

        def update_with_scoring_singles(
            roll: RollArrayType,
            value: int,
            keep: MaskType,
        ) -> tuple[int, MaskType]:
            """Update value, mask pair based on ones and fives.

            Helper function that takes an existing score and update with the
            counts of ones and fives, worth 100 and 50 respectively.

            Args:
                roll (array of int): The selected roll.
                value: The score of the roll without ones and fives.
                keep (array of bool): A mask of which dice have already been
                    selected to keep.

            Returns:
                value: Updated score.
                keep (array of bool): Mask including ones and fives.
            """
            if keep.sum() == roll.shape:
                return value, keep

            is_one = roll == 1
            is_five = roll == 5
            one_count = is_one[~keep].sum()
            five_count = is_five[~keep].sum()

            value += 100 * one_count
            value += 50 * five_count
            if one_count + five_count == 0:
                return value, keep

            keep |= is_one
            keep |= is_five
            return value, keep

        match self:
            case RollStruct(roll, [6], scoring_n):
                """Six of a kind"""
                return 3000, _KEEP_ALL

            case RollStruct(roll, [*_, 5], scoring_n):
                """Five of a kind"""
                keep = roll == scoring_n
                value, keep = update_with_scoring_singles(roll, 2000, keep)
                return value, keep

            case RollStruct(roll, [2, 4], scoring_n):
                """Four of a kind + a pair"""
                return 1500, _KEEP_ALL

            case RollStruct(roll, [1, 1, 1, 1, 1, 1], scoring_n):
                """Straight"""
                return 1500, _KEEP_ALL

            case RollStruct(roll, [*_, 4], scoring_n):
                """Four of a kind"""
                keep = roll == scoring_n
                # Check if remainder is scoring
                value, keep = update_with_scoring_singles(roll, 1000, keep)
                return value, keep

            case RollStruct(roll, [3, 3], scoring_n):
                """Two triplets"""
                return 2500, _KEEP_ALL

            case RollStruct(roll, [2, 2, 2], scoring_n):
                """Three pairs"""
                return 1500, _KEEP_ALL

            case RollStruct(roll, [*_, 3], scoring_n) if scoring_n > 1:
                """Three of a kind (for numbers greater than one)"""
                value = scoring_n * 100
                keep = roll == scoring_n
                # Check if remainder is scoring
                value, keep = update_with_scoring_singles(roll, value, keep)
                return value, keep

            case RollStruct(roll, _):
                """Scoring numbers: Ones and fives"""
                keep_none = np.array([False] * self.shape)
                return update_with_scoring_singles(roll, 0, keep_none)

        raise RuntimeError(  # pragma: no cover
            "Unexpected error encountered during scoring! "
            "Unable to score RollStruct."
        )


class TurnStateError(RuntimeError):
    """Turn in wrong state for requested action."""


_Param = ParamSpec("_Param")
_RetType = TypeVar("_RetType")


class TurnState(Enum):
    """Enum of ``Turn`` states.

    Represents the possible states of a ``Turn`` object.
    """

    ROLL = auto()
    """Waiting for user to roll."""
    SELECTION = auto()
    """Waiting for user to select."""
    FARKLE = auto()
    """Non-scoring roll ends turn."""
    HOLD = auto()
    """User gracefully ends turn."""
    ERROR = auto()
    """Turn irrevocably failed"""

    def is_terminal(self):
        """Can state be updated after reaching this state."""
        return self in (TurnState.FARKLE, TurnState.HOLD, TurnState.ERROR)

    def assert_state(
        self, func: Callable[Concatenate[Turn, _Param], _RetType]
    ) -> Callable[Concatenate[Turn, _Param], _RetType]:
        """Ensure ``Turn`` has set state.

        Raises:
            TurnStateError: If any state except ``self``.
        """

        @wraps(func)
        def wrapper(cls_self: Turn, *args, **kwargs) -> _RetType:
            if cls_self.state != self:
                raise TurnStateError(
                    f"Cannot perform {func} in current state of {cls_self.state}"
                )
            return func(cls_self, *args, **kwargs)

        return wrapper


class Turn:
    """Turn in Farkle game.

    Interactive, stateful representation of a turn in a game. One object should
    be generated per turn, and mutated as the turn evolves, eventually landing
    in a terminal state such as "hold" or "farkle".

    You may notice that most functions return ``None``. This object is designed
    to change state through the methods, then let the caller reference the
    information as attributes rather than returning an arbitrary set of
    information.

    Args:
        game: Optional parameter to track the game this turn is a part of. This
            is managed programmatically. Do not update directly.
        player_id: Optional Player ID this turn belongs to. This is managed
            programmatically. Do not update directly.

    Attributes:
        frozen (array of int): Which dice have been scored, added to ``value``,
            then put aside.
        value: Total value accumulated during turn.
        state: The current ``TurnState``.
        open_spots (int): Number of non-frozen dice to roll.
        curr_roll: The current state of un-frozen dice.
        curr_value: The value of the current, un-frozen roll.
        curr_scoring_mask: A boolean mask of scoring values within current
            roll.
    """

    curr_roll: RollStruct
    curr_value: int
    curr_scoring_mask: MaskType

    def __init__(self, game: Game | None = None, player_id: int | None = None) -> None:
        self.game = game
        self.player_id = player_id
        self.frozen: RollArrayType = np.zeros((6,), dtype=np.uint8)
        self.value: int = 0
        self._state: TurnState = TurnState.ROLL
        self._logger = logging.getLogger(__name__)

    @property
    def state(self) -> TurnState:
        return self._state

    @state.setter
    def state(self, value: TurnState) -> None:
        """State can only be updated if in non-terminal state."""
        if self.state.is_terminal():
            raise TurnStateError(
                f"Cannot change state from terminal state: {self.state}"
            )
        self._logger.debug(f"Changing state: {self.state} → {value}")
        self._state = value

    @property
    def open_spots(self) -> int:
        return (self.frozen == 0).sum()

    @TurnState.ROLL.assert_state
    def roll(self, fixed: RollStruct | None = None) -> None:
        """Roll the dice!

        Rolls all un-frozen dice and updates state to ``SELECTION`` state. Can
        only be run when in ``ROLL`` state. Updates ``curr_`` prefixed attributes.

        Args:
            fixed: Pass a preset roll to the roll for testing.
        """
        try:
            self.curr_roll = fixed or RollStruct.random(self.open_spots)
            self.curr_value, self.curr_scoring_mask = self.curr_roll.score()

            if self.curr_value == 0:
                self.state = TurnState.FARKLE
            else:
                self.state = TurnState.SELECTION
            self._logger.debug(
                f"Rolled new: \n{self.curr_roll=}; "
                f"\n{self.curr_value=}; {self.curr_scoring_mask}"
            )
        except Exception:
            self._logger.exception("Unexpected error during roll: closing turn.")
            self.state = TurnState.ERROR

    @TurnState.SELECTION.assert_state
    def select(self, mask: MaskType | None = None) -> None:
        """Select scoring dice to freeze, with an intent to continue scoring.

        This is the selection phase of the turn, where the user has rolled and
        decides which dice to hold and which to make available for re-rolling.
        The object must be in the ``SELECTION`` state to execute, while doing
        so puts the object in the ``ROLL`` phase.

        Args:
            mask: Optional boolean mask of which dice to keep. Defaults to
            scoring dice captured in ``RollStruct.score``.
        """
        mask = mask if mask is not None else self.curr_scoring_mask
        if np.any(mask & ~self.curr_scoring_mask):
            raise ValueError("Tried to select non-scoring di(c)e")
        if mask.size != self.open_spots:
            raise ValueError("Mask size must be same as number of open spots")
        try:
            self._logger.debug(f"Selecting: {mask=} from {self.curr_roll=}")
            masked_roll = self.curr_roll.subroll(mask)
            masked_score, masked_keep = masked_roll.score()
            self.value += masked_score

            if np.all(mask):
                self._refresh_frozen()
            else:
                kept_from_roll = masked_roll.roll_array
                open_spots = ~self.frozen.astype(bool)
                open_spots[open_spots] = open_spots[open_spots] & mask
                self.frozen[open_spots] = kept_from_roll

            self._logger.debug(f"New score: {self.value=}")
            self.state = TurnState.ROLL
        except Exception:
            self._logger.exception("Unexpected error during select: closing turn.")
            self.state = TurnState.ERROR

    def _refresh_frozen(self) -> None:
        """Unfreeze all dice.

        If all dice are scoring, the player receives a new set of dice. Here we
        return the frozen sequence to be all zeros (indicating un-frozen).
        Note, we do not track the past frozen sets, only the current.
        """
        self.frozen[:] = 0

    @TurnState.SELECTION.assert_state
    def check_mask(self, mask: MaskType | None = None) -> tuple[int, MaskType]:
        """Check the score of a mask without changing state.

        This is optionally used to interactively test different combinations as
        the user selects them in a UI.

        Args:
            mask: Mask to check score from current roll.

        Returns:
            int: Score of selection.
            mask: The scoring subset of the mask.
        """
        roll = self.curr_roll
        if mask is not None:
            roll = self.curr_roll.subroll(mask)
        value, keep = roll.score()
        return value, keep

    @TurnState.SELECTION.assert_state
    def hold(self) -> None:
        """Gracelly end turn, keeping score.

        This is the only scoring terminal state in a turn of Farkle. The only
        other outcome is a "farkle" which yields zero points from the entire
        turn.
        """
        self.value += self.curr_value
        self.state = TurnState.HOLD


class Game:
    """Farkle game object.

    Wrapper for Farkle internals. Manages players, their turns, and scoring.
    Scores are accumulated on a turn basis until one player reaches 10,000.
    At this point, each player (besides this one) gets one final shot at
    beating them.

    Args:
        num_players: Number of players to cycle through

    Attributes:
        turn_cycler: Iterates through player ids.
        player_score_map: Stores player scores by id.
    """

    WINNING_SCORE: Final = 10_000
    """Threshold at which the end-game begins."""

    def __init__(self, num_players: int) -> None:
        player_range = range(num_players)
        self.num_players = num_players
        self.turn_cycler: cycle = cycle(player_range)
        self.player_score_map: dict[int, int] = {i: 0 for i in player_range}
        self._logger = logging.getLogger(__name__)

    def start(self) -> Generator[Turn, None, None]:
        winning_id = -1
        for player_id in self.turn_cycler:
            if player_id == winning_id:  # pragma: no cover
                return

            self._logger.debug(f"Starting turn for {player_id=}")
            turn = Turn(game=self, player_id=player_id)
            yield turn
            self._logger.debug(f"Closing turn for {player_id=}")

            if not turn.state.is_terminal():
                self._logger.critical(
                    "Turn was not over before starting next turn. "
                    "Force ending turn. Current roll points may be missed!"
                )
                turn.state = TurnState.ERROR

            self.player_score_map[player_id] += turn.value

            # If someone reaches 10,000 everyone gets one last turn to try to beat it
            if self.player_score_map[player_id] >= Game.WINNING_SCORE:
                winning_id = player_id

            # Every complete loop, we log the current scores to DEBUG
            if player_id == self.num_players - 1:
                self._logger.debug(f"Current scores: {self.player_score_map}")
