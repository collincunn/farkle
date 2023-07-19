import json

import numpy as np

from tests.farkle.core import RollStruct, score_roll_struct


def pytest_generate_tests(metafunc):
    """This allows us to load tests from external files by
    parametrizing tests with each test case found in a data_X
    file"""
    for fixture in metafunc.fixturenames:
        if fixture.startswith("data__"):
            tests = load_tests(fixture)
            metafunc.parametrize(fixture, tests)


def load_tests(fixture):
    with open("test_parameters.json") as f:
        test_data = json.loads(f.read())
    test_data_key = fixture.removeprefix("data__")
    return test_data[test_data_key]


def test_scoring(data__scoring):
    roll = data__scoring["roll"]
    expected_score = data__scoring["score"]
    expected_keep = data__scoring["keep"]
    other_args = data__scoring.get("args", {})

    expected_keep_arr = np.array(expected_keep).astype(bool)
    roll_struct = RollStruct.fixed(roll)
    score, keep = score_roll_struct(roll_struct, **other_args)
    assert score == expected_score
    np.testing.assert_array_equal(expected_keep_arr, keep)
