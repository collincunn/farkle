from __future__ import annotations

from abc import ABCMeta, abstractmethod
from dataclasses import (
    dataclass,
    field,
    replace,
)
from enum import Enum, auto
from functools import wraps
from itertools import cycle
import logging
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Concatenate,
    Final,
    Generator,
    ParamSpec,
    TypeVar,
)

import numpy as np
from numpy.typing import ArrayLike, NDArray

if TYPE_CHECKING:
    from collections.abc import Sequence

MaskType = NDArray[np.bool_]
RollArrayType = NDArray[np.uint8]

_KEEP_ALL = np.array([True] * 6)


@dataclass(frozen=True, slots=True)
class Roll:
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
        score: Score yielded from the roll.
        scoring_mask: A boolean array that indicates which dice are scoring and
            which are not.
    """

    roll_array: RollArrayType
    counts: list[int]
    last_n: int
    shape: int
    score: int = field(init=False)
    scoring_mask: MaskType = field(init=False)

    def __post_init__(self) -> None:
        """Add scoring information to dataclass.

        Depends on other variables so constructed post init.
        """
        score, scoring_mask = self._score()
        # Because the object is "immutable" we have to use object's setattr
        # call. This has a performance impact at creation time.
        object.__setattr__(self, "score", score)
        object.__setattr__(self, "scoring_mask", scoring_mask)

    @classmethod
    def fixed(cls, roll: ArrayLike) -> Roll:
        """Create a ``Roll`` from a specific set of values.

        This is used for fixing a specific combination of values as a roll,
        and calculating the metadata implicitly.

        Args:
            roll (array-like): The selected roll.

        Returns:
            Roll: Object representing roll.
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
    def random(cls, shape: int = 6) -> Roll:
        """Simulate a roll.

        Create a random roll wrapped in a ``Roll``.

        Args:
            shape: Number of dice to roll.

        Returns:
            Roll: Object representing roll.
        """
        roll = np.random.randint(1, 7, size=shape)
        return cls.fixed(roll)

    def is_farkle(self) -> bool:
        return self.score == 0

    def subset(self, mask: NDArray[np.bool_]) -> Roll:
        """Create a new object from a subset of this object."""
        return Roll.fixed(self.roll_array[mask])

    def _score(self) -> tuple[int, MaskType]:
        """Calculate score and mask.

        Use ``Roll`` metadata to calculate the score and choose which
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
            case Roll(roll, [6], scoring_n):
                """Six of a kind"""
                return 3000, _KEEP_ALL

            case Roll(roll, [*_, 5], scoring_n):
                """Five of a kind"""
                keep = roll == scoring_n
                value, keep = update_with_scoring_singles(roll, 2000, keep)
                return value, keep

            case Roll(roll, [2, 4], scoring_n):
                """Four of a kind + a pair"""
                return 1500, _KEEP_ALL

            case Roll(roll, [1, 1, 1, 1, 1, 1], scoring_n):
                """Straight"""
                return 1500, _KEEP_ALL

            case Roll(roll, [*_, 4], scoring_n):
                """Four of a kind"""
                keep = roll == scoring_n
                # Check if remainder is scoring
                value, keep = update_with_scoring_singles(roll, 1000, keep)
                return value, keep

            case Roll(roll, [3, 3], scoring_n):
                """Two triplets"""
                return 2500, _KEEP_ALL

            case Roll(roll, [2, 2, 2], scoring_n):
                """Three pairs"""
                return 1500, _KEEP_ALL

            case Roll(roll, [*_, 3], scoring_n) if scoring_n > 1:
                """Three of a kind (for numbers greater than one)"""
                value = scoring_n * 100
                keep = roll == scoring_n
                # Check if remainder is scoring
                value, keep = update_with_scoring_singles(roll, value, keep)
                return value, keep

            case Roll(roll, _):
                """Scoring numbers: Ones and fives"""
                keep_none = np.array([False] * self.shape)
                return update_with_scoring_singles(roll, 0, keep_none)

        raise RuntimeError(  # pragma: no cover
            "Unexpected error encountered during scoring! Unable to score Roll."
        )


class TurnStateError(RuntimeError):
    """Turn in wrong state for requested action."""


_Param = ParamSpec("_Param")
_RetType = TypeVar("_RetType")


class TurnState(Enum):
    """Enum of ``Turn`` states.

    Represents the possible states of a ``Turn`` object.
    """

    PLAYING = auto()
    """Play is still active."""
    FARKLE = auto()
    """Non-scoring roll ends turn."""
    HOLD = auto()
    """User gracefully ends turn."""
    ERROR = auto()
    """Turn irrevocably failed"""

    def is_terminal(self):
        """Can state be updated after reaching this state."""
        return self is not TurnState.PLAYING

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


RollFactoryType = Callable[[int], Roll]


class BaseRollFactory(metaclass=ABCMeta):
    """Gives next roll.

    Keeps track of state between rolls and gives next roll when called. Used
    for roll dependency injection into Turn.
    """

    @abstractmethod
    def __call__(self, open_spots: int) -> Roll:
        """Get next roll, given internal state and number of open spots."""


class NumpyRandomRollFactory(BaseRollFactory):
    """Stateless roll, getting next roll from ``Roll``'s numpy based random
    selection.
    """

    def __call__(self, open_spots: int) -> Roll:
        return Roll.random(open_spots)


class FixedRollFactory(BaseRollFactory):
    """Stateful roll incrementor that passes whatever the last roll that was
    sent.

    Used for testing and debugging.
    """

    next_roll: Roll

    def __init__(self, seed_roll: Roll) -> None:
        self.next_roll = seed_roll

    def set_next_roll(self, roll: Roll) -> None:
        """Fix the upcoming roll."""
        self.next_roll = roll

    def __call__(self, _: int) -> Roll:
        return self.next_roll


class Turn:
    """Turn in Farkle game.

    Interactive, stateful representation of a turn in a game. One object should
    be generated per turn, and mutated as the turn evolves through ``select``
    calls, eventually landing in a terminal state such as "hold" or "farkle".

    You may notice that most functions return ``None``. This object is designed
    to be a state machine. The caller can reference the information as attributes
    rather than returning an arbitrary set of information.

    Args:
        roll_factory: Optional parameter to specify a factory from which rolls
            will be generated.

    Attributes:
        frozen (array of int): Which dice have been scored, added to ``value``,
            then put aside.
        value: Total value accumulated during turn.
        state: The current ``TurnState``.
        open_spots (int): Number of non-frozen dice to roll.
        roll: The current un-frozen dice.
    """

    def __init__(self, *, roll_factory: RollFactoryType | None = None) -> None:
        self._logger = logging.getLogger(__name__)
        self.frozen: RollArrayType = np.zeros((6,), dtype=np.uint8)
        self.score: int = 0
        self._state: TurnState = TurnState.PLAYING
        self._roll_factory = roll_factory or NumpyRandomRollFactory()
        self._update_roll()

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
        self._logger.debug("Changing state: %s → %s", self.state, value)
        self._state = value

    @property
    def open_spots(self) -> int:
        return (self.frozen == 0).sum()

    def is_active(self) -> bool:
        """Is the turn still in a playable state?"""
        return self.state is TurnState.PLAYING

    def _update_roll(self, fixed: Roll | None = None) -> None:
        """Roll the dice!

        Rolls all un-frozen dice and updates ``roll`` attribute.

        Args:
            fixed: Pass a preset roll to the roll for testing.
        """
        try:
            self.roll = self._roll_factory(self.open_spots)
            self._logger.debug("Rolling: New roll = %s", self.roll)

            if self.roll.is_farkle():
                self.state = TurnState.FARKLE
                self.score = 0
        except Exception:
            self._logger.exception("Unexpected error during roll: closing turn.")
            self.state = TurnState.ERROR

    @TurnState.PLAYING.assert_state
    def select(self, mask: MaskType | None = None) -> None:
        """Select scoring dice to freeze, with an intent to continue scoring.

        This is the selection phase of the turn, where the user has rolled and
        decides which dice to hold and which to make available for re-rolling.
        The object must be in the ``PLAYING`` state to execute.

        Args:
            mask: Optional boolean mask of which dice to keep. Defaults to
                scoring dice captured in ``Roll.scoring_mask``.
        """
        mask = mask if mask is not None else self.roll.scoring_mask
        if np.any(mask & ~self.roll.scoring_mask):
            raise ValueError("Tried to select non-scoring di(c)e")
        if mask.size != self.open_spots:
            raise ValueError("Mask size must be same as number of open spots")
        if mask.sum() < 1:
            raise ValueError("No selection made, must choose at least one to keep.")
        try:
            self._logger.debug("Selecting: mask=%s from %s", mask, self.roll)
            masked_roll = self.roll.subset(mask)
            self.score += masked_roll.score

            if np.all(mask):
                self._refresh_frozen()
            else:
                kept_from_roll = masked_roll.roll_array
                open_spots = ~self.frozen.astype(bool)
                open_spots[open_spots] = open_spots[open_spots] & mask
                self.frozen[open_spots] = kept_from_roll

            self._logger.debug("New score: %s", self.score)

            self._update_roll()
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

    @TurnState.PLAYING.assert_state
    def hold(self) -> None:
        """Gracelly end turn, keeping score.

        This is the only scoring terminal state in a turn of Farkle. The only
        other outcome is a "farkle" which yields zero points from the entire
        turn.
        """
        self.score += self.roll.score
        self.state = TurnState.HOLD


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
    roll: Roll
    frozen: RollArrayType
    scoring_mask: MaskType
    max_other_player_score: int

    @classmethod
    def from_turn(cls, turn: Turn, game_state: dict[str, Any]) -> EnvironmentState:
        """Create environment dynamically from core assets.

        Gather data from this game and turn for a specific player.

        Args:
            turn: Current game
            game_state: arbitrary metadata about game that is set from game.
        """
        if turn.state != TurnState.PLAYING:
            raise TurnStateError(
                "Must be in SELECTION state to create bot environment state"
            )
        score_so_far_this_turn = turn.score
        potential_hold_score = turn.roll.score + score_so_far_this_turn
        scoring_mask = turn.roll.scoring_mask
        roll = turn.roll
        frozen = turn.frozen
        score_prior_to_turn = game_state.get("score_prior_to_turn", 0)
        max_other_player_score = game_state.get("max_other_player_score", 0)

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
        return replace(self, **kwargs)


@dataclass(frozen=True, slots=True, unsafe_hash=True)
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
            # TODO: This is overcomplicated
            if self.hold is not other.hold:
                return False
            if self.mask is None:
                return other.mask is None
            return other.mask is not None and np.array_equal(self.mask, other.mask)
        return False


class BasePlayer(metaclass=ABCMeta):
    """Abstract base class for game players."""

    score: int
    name: str

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

    def handle_turn_generator(
        self, turn: Turn, game_state: dict
    ) -> Generator[EnvironmentState, Action, None]:
        """Loop through turn and take actions until turn ends.

        Iterate over the rolling rounds in a turn and take actions that are
        defined by ``.action`` until the turn ends in either a Farkle or a
        "hold". The turn iterates as a generator, sending state and potentially
        receiving an overriding action. The latter should only be used for
        debugging.
        """
        while turn.is_active():
            state = EnvironmentState.from_turn(turn, game_state)
            action = self.action(state)
            received_action = yield state
            action = received_action or action
            if action.hold:
                turn.hold()
            else:
                turn.select(action.mask)
        self.score += turn.score


class Game:
    """Farkle game object.

    Wrapper for Farkle internals. Manages players, their turns, and scoring.
    Scores are accumulated on a turn basis until one player reaches 10,000.
    At this point, each player (besides this one) gets one final shot at
    beating them.

    Args:
        players: Sequence of player objects.

    Attributes:
        players: Sequence of player objects.
        players_cycler: Endlessly iterates through player ids.
    """

    WINNING_SCORE: Final = 10_000
    """Threshold at which the end-game begins."""

    def __init__(self, players: Sequence[BasePlayer]) -> None:
        self.players = players
        self.player_cycler: cycle = cycle(players)
        self._logger = logging.getLogger(__name__)

    def game_state(self, current_player: BasePlayer) -> dict[str, int]:
        if len(self.players) > 1:
            max_other_player_score = max(
                p.score for p in self.players if p is not current_player
            )
        else:
            max_other_player_score = 0
        return {
            "max_other_player_score": max_other_player_score,
            "score_prior_to_turn": current_player.score,
        }

    def play_generator(self) -> Generator[EnvironmentState, Action, None]:
        is_endgame = False
        for player in self.player_cycler:
            if is_endgame:  # pragma: no cover
                return

            self._logger.debug(f"Starting turn for {player}: starting {player.score=}")
            turn = Turn()
            game_state = self.game_state(player)
            handled_turn = player.handle_turn_generator(turn, game_state)
            yield from handled_turn

            self._logger.debug(f"Closing turn for {player}: ending {player.score=}")

            if not turn.state.is_terminal():
                self._logger.critical(
                    "Turn was not over before starting next turn. "
                    "Force ending turn. Current roll points may be missed!"
                )
                turn.state = TurnState.ERROR

            # If someone reaches 10,000 everyone gets one last turn to try to beat it
            if player.score >= Game.WINNING_SCORE:
                is_endgame = True

    def play(self) -> None:
        for _ in self.play_generator():
            continue
