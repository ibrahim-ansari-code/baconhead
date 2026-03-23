#!/usr/bin/env python3
"""Offline tests for llm_agent/scout: _parse_rewards, _frame_to_base64, DEFAULT_ACTIONS."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llm_agent.scout import (
    _parse_rewards,
    _frame_to_base64,
    DEFAULT_ACTIONS,
    SCOUT_MODEL,
)


def test_parse_rewards_all_present():
    t = "REWARD1=1.0 REWARD2=0.2 REWARD3=0.3 REWARD4=0.4 REWARD5=0.5 REWARD6=0.6 REWARD7=0.7 REWARD8=0.8 REWARD9=0.9 REWARD10=0.0"
    s = _parse_rewards(t, n=10)
    assert len(s) == 10
    assert s[0] == 1.0 and s[9] == 0.0


def test_parse_rewards_lowercase():
    t = "reward1=0.5 reward2=0.5"
    s = _parse_rewards(t, n=10)
    assert s[0] == 0.5 and s[1] == 0.5


def test_parse_rewards_spaces_around_eq():
    t = "REWARD1 = 0.9 REWARD2 = 0.1"
    s = _parse_rewards(t, n=10)
    assert s[0] == 0.9 and s[1] == 0.1


def test_parse_rewards_n_5():
    t = "REWARD1=1 REWARD2=2 REWARD3=3 REWARD4=4 REWARD5=5"
    s = _parse_rewards(t, n=5)
    assert len(s) == 5 and s[4] == 5.0


def test_parse_rewards_n_3():
    s = _parse_rewards("REWARD1=0.1 REWARD2=0.2 REWARD3=0.3", n=3)
    assert s == [0.1, 0.2, 0.3]


def test_parse_rewards_invalid_float():
    s = _parse_rewards("REWARD1=abc REWARD2=0.5", n=10)
    assert s[0] == 0.0 and s[1] == 0.5


def test_parse_rewards_extra_text():
    t = "Some text REWARD1=0.8 more text REWARD2=0.2 end"
    s = _parse_rewards(t, n=10)
    assert s[0] == 0.8 and s[1] == 0.2


def test_parse_rewards_repeated_key():
    t = "REWARD1=0.3 REWARD1=0.7"
    s = _parse_rewards(t, n=10)
    # re.search finds first match, so first value wins
    assert s[0] == 0.3


def test_frame_to_base64_shape():
    f = np.zeros((100, 200, 3), dtype=np.uint8)
    b64 = _frame_to_base64(f, max_size=512)
    assert isinstance(b64, str)
    assert len(b64) > 0


def test_frame_to_base64_decode():
    import base64
    f = np.ones((50, 50, 3), dtype=np.uint8) * 128
    b64 = _frame_to_base64(f)
    raw = base64.b64decode(b64)
    assert len(raw) > 0


def test_frame_to_base64_resize():
    f = np.zeros((1000, 800, 3), dtype=np.uint8)
    b64 = _frame_to_base64(f, max_size=256)
    import base64
    raw = base64.b64decode(b64)
    assert len(raw) < 500000  # smaller after resize


def test_default_actions_len():
    assert len(DEFAULT_ACTIONS) == 10


def test_default_actions_contains():
    assert "W" in DEFAULT_ACTIONS and "space" in DEFAULT_ACTIONS and "none" in DEFAULT_ACTIONS


def test_scout_model_nonempty():
    assert SCOUT_MODEL and "llama" in SCOUT_MODEL.lower()


def test_parse_rewards_zero():
    s = _parse_rewards("REWARD1=0 REWARD2=0.0", n=10)
    assert s[0] == 0.0 and s[1] == 0.0


def test_parse_rewards_one():
    s = _parse_rewards("REWARD1=1 REWARD2=1.0", n=10)
    assert s[0] == 1.0 and s[1] == 1.0


def test_parse_rewards_decimal():
    s = _parse_rewards("REWARD1=0.123 REWARD2=0.456", n=10)
    assert abs(s[0] - 0.123) < 1e-5 and abs(s[1] - 0.456) < 1e-5


def test_parse_rewards_empty_string():
    s = _parse_rewards("", n=10)
    assert len(s) == 10 and all(x == 0.0 for x in s)


def test_parse_rewards_no_reward_key():
    s = _parse_rewards("AVOID=0 OBJECTIVES=go", n=10)
    assert len(s) == 10 and all(x == 0.0 for x in s)


if __name__ == "__main__":
    tests = [
        test_parse_rewards_all_present, test_parse_rewards_lowercase, test_parse_rewards_spaces_around_eq,
        test_parse_rewards_n_5, test_parse_rewards_n_3, test_parse_rewards_invalid_float,
        test_parse_rewards_extra_text, test_parse_rewards_repeated_key, test_frame_to_base64_shape,
        test_frame_to_base64_decode, test_frame_to_base64_resize, test_default_actions_len,
        test_default_actions_contains, test_scout_model_nonempty, test_parse_rewards_zero,
        test_parse_rewards_one, test_parse_rewards_decimal, test_parse_rewards_empty_string,
        test_parse_rewards_no_reward_key,
    ]
    for t in tests:
        t()
        print(t.__name__, "OK")
    print("All scout offline tests passed.")
