import json
import os

import numpy as np
import pytest

from farkle.core import EnvironmentState, Roll

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
        test_data = json.load(f)
    test_data_key = fixture.removeprefix("data__")
    return test_data[test_data_key]


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
