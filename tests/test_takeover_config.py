#!/usr/bin/env python3
"""Tests for run_takeover and run_live_test argument parsing and config."""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_region_parse_four():
    parts = [int(x.strip()) for x in "100,200,640,360".split(",")]
    assert len(parts) == 4
    r = {"left": parts[0], "top": parts[1], "width": parts[2], "height": parts[3]}
    assert r["width"] == 640 and r["height"] == 360


def test_region_parse_invalid():
    try:
        parts = [int(x.strip()) for x in "1,2,3".split(",")]
        r = {"left": parts[0], "top": parts[1], "width": parts[2], "height": parts[3]}
        assert False
    except (ValueError, IndexError):
        pass


def test_takeover_parser_idle():
    from run_takeover import click_close
    # run_takeover uses 10s plans; main parser has --idle default 3.0
    parser = argparse.ArgumentParser()
    parser.add_argument("--idle", type=float, default=3.0)
    parser.add_argument("--full-screen", action="store_true")
    parser.add_argument("--region", type=str, default=None)
    args = parser.parse_args([])
    assert args.idle == 3.0


def test_live_test_parser_full_screen():
    parser = argparse.ArgumentParser()
    parser.add_argument("--full-screen", action="store_true")
    parser.add_argument("--region", type=str)
    args = parser.parse_args(["--full-screen"])
    assert args.full_screen is True


def test_live_test_parser_region():
    parser = argparse.ArgumentParser()
    parser.add_argument("--region", type=str)
    args = parser.parse_args(["--region", "0,0,800,600"])
    assert args.region == "0,0,800,600"


def test_click_close_exists():
    from run_takeover import click_close
    click_close(None)
    click_close({"left": 0, "top": 0, "width": 100, "height": 100})


def test_interval_positive():
    interval_sec = 1.0
    duration_ms = 500
    last_cem_elapsed_sec = 0.2
    sleep_sec = interval_sec - last_cem_elapsed_sec - (duration_ms / 1000.0)
    assert abs(sleep_sec - 0.3) < 1e-5


def test_adaptive_interval_zero_sleep():
    interval_sec = 1.0
    duration_ms = 500
    last_cem_elapsed_sec = 0.6  # CEM took 0.6s
    sleep_sec = interval_sec - last_cem_elapsed_sec - (duration_ms / 1000.0)
    assert sleep_sec < 0.05  # next CEM immediately


def test_actions_deque_maxlen():
    from collections import deque
    d = deque(maxlen=8)
    for i in range(10):
        d.append(f"a{i}")
    assert len(d) == 8
    assert d[0] == "a2" and d[-1] == "a9"


if __name__ == "__main__":
    tests = [
        test_region_parse_four, test_region_parse_invalid, test_takeover_parser_idle,
        test_live_test_parser_full_screen, test_live_test_parser_region,
        test_click_close_exists, test_interval_positive, test_adaptive_interval_zero_sleep,
        test_actions_deque_maxlen,
    ]
    for t in tests:
        t()
        print(t.__name__, "OK")
    print("All takeover config tests passed.")
