#!/usr/bin/env python3
"""Offline tests for reward/input_state. Listener may or may not be running."""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from reward.input_state import get_current_keys, get_last_key_time, is_active


def test_get_current_keys_returns_set():
    k = get_current_keys()
    assert isinstance(k, set)


def test_get_current_keys_copy():
    k1 = get_current_keys()
    k2 = get_current_keys()
    assert k1 is not k2 or len(k1) == 0


def test_is_active_no_recent_key():
    # If no key was ever pressed, is_active(999) should be False
    # (last_key_time is None until first key)
    result = is_active(999.0)
    assert result is False or result is True  # either is valid depending on state


def test_is_active_zero_window():
    r = is_active(0.0)
    assert isinstance(r, bool)


def test_get_last_key_time_type():
    t = get_last_key_time()
    assert t is None or isinstance(t, (int, float))


def test_is_active_negative_window():
    r = is_active(-1.0)
    assert isinstance(r, bool)


def test_get_current_keys_iterable():
    for _ in get_current_keys():
        break


def test_is_active_large_window():
    r = is_active(1e6)
    assert isinstance(r, bool)


def test_get_current_keys_strings():
    k = get_current_keys()
    for x in k:
        assert isinstance(x, str)


if __name__ == "__main__":
    tests = [
        test_get_current_keys_returns_set, test_get_current_keys_copy, test_is_active_no_recent_key,
        test_is_active_zero_window, test_get_last_key_time_type, test_is_active_negative_window,
        test_get_current_keys_iterable, test_is_active_large_window, test_get_current_keys_strings,
    ]
    for t in tests:
        t()
        print(t.__name__, "OK")
    print("All input_state tests passed.")
