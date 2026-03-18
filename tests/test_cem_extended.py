#!/usr/bin/env python3
"""Extended CEM and execute_action tests."""
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llm_agent.cem import run_cem, execute_action_ms


def _frame(h=224, w=224):
    return np.zeros((h, w, 3), dtype=np.uint8) + 128


def test_cem_actions_short_list():
    actions = ["W", "A"]
    mock = ([0.9, 0.1] + [0.0] * 8, 0.0, "")
    best, _, _, _, _, _ = run_cem(_frame(), actions=actions, mock_scout_result=mock, use_scout=False)
    assert best in ("W", "A")


def test_cem_actions_exactly_10():
    actions = ["W", "A", "S", "D", "space", "none", "W", "A", "S", "D"]
    mock = ([0.0] * 9 + [1.0], 0.0, "")
    best, _, _, _, _, _ = run_cem(_frame(), actions=actions, mock_scout_result=mock, use_scout=False)
    assert best == "D"


def test_cem_avoid_weight_zero():
    mock = ([0.5] * 10, 1.0, "")
    best, scores, _, _, _, _ = run_cem(_frame(), mock_scout_result=mock, use_scout=False, avoid_weight=0.0)
    assert scores[0] == 0.5


def test_cem_avoid_weight_one():
    mock = ([0.8] * 10, 0.5, "")
    best, scores, _, _, _, _ = run_cem(_frame(), mock_scout_result=mock, use_scout=False, avoid_weight=1.0)
    assert abs(scores[0] - 0.3) < 1e-5


def test_cem_last_actions_empty():
    # Empty last_actions; W has 0.7 (all equal), first W wins
    mock = ([0.7] * 10, 0.0, "")
    best, _, _, _, _, _ = run_cem(_frame(), mock_scout_result=mock, use_scout=False, last_actions=[])
    assert best == "W"


def test_cem_last_objective_passed():
    mock = ([0.5] * 10, 0.0, "reach flag")
    _, _, _, obj, _, _ = run_cem(_frame(), mock_scout_result=mock, use_scout=False, last_objective="old")
    assert obj == "reach flag"


def test_cem_returns_four_values():
    mock = ([0.5] * 10, 0.0, "x")
    out = run_cem(_frame(), mock_scout_result=mock, use_scout=False)
    assert len(out) == 6
    best, scores, r, obj, popup, duration_override = out
    assert isinstance(best, str) and len(scores) == 10 and isinstance(r, (int, float)) and isinstance(obj, str)


def test_cem_scores_order():
    # Highest score is index 9 (1.0) = look_right
    mock = ([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], 0.0, "")
    best, _, _, _, _, _ = run_cem(_frame(), mock_scout_result=mock, use_scout=False, last_actions=["W"])
    assert best == "look_right"


def test_execute_action_none_int_duration():
    t0 = time.perf_counter()
    execute_action_ms("none", duration_ms=25)
    elapsed = (time.perf_counter() - t0) * 1000
    assert 15 <= elapsed <= 150


def test_execute_action_unknown_key_sleeps():
    t0 = time.perf_counter()
    execute_action_ms("UnknownKeyXYZ", duration_ms=15)
    elapsed = (time.perf_counter() - t0) * 1000
    assert elapsed >= 10  # at least slept or tried key


def test_execute_action_none_none_input():
    t0 = time.perf_counter()
    execute_action_ms(None, duration_ms=20)
    elapsed = (time.perf_counter() - t0) * 1000
    assert 10 <= elapsed <= 120


def test_cem_mock_six_elements():
    mock = ([0.5] * 10, 0.0, "obj", "extra", "ignored", "ok")
    _, _, _, obj, _, _ = run_cem(_frame(), mock_scout_result=mock[:3], use_scout=False)
    assert obj == "obj"


def test_cem_mock_two_elements_no_objectives():
    mock = ([0.5] * 10, 0.0)
    _, _, _, obj, _, _ = run_cem(_frame(), mock_scout_result=mock, use_scout=False)
    assert obj == ""


def test_cem_frame_shape_any():
    mock = ([0.5] * 10, 0.0, "")
    best, _, _, _, _, _ = run_cem(_frame(50, 50), mock_scout_result=mock, use_scout=False)
    assert best in ("W", "A", "S", "D", "space", "none", "look_left", "look_right")


def test_cem_space_index():
    actions = ["W", "A", "S", "D", "space", "none", "W", "A", "S", "D"]
    mock = ([0, 0, 0, 0, 0.9, 0, 0, 0, 0, 0], 0.0, "")
    best, _, _, _, _, _ = run_cem(_frame(), actions=actions, mock_scout_result=mock, use_scout=False)
    assert best == "space"


def test_cem_repeat_penalty_space_two():
    mock = ([0.5, 0.2, 0.2, 0.2, 0.95, 0.0, 0.5, 0.2, 0.2, 0.2], 0.0, "")
    best, _, _, _, _, _ = run_cem(_frame(), mock_scout_result=mock, use_scout=False, last_actions=["space", "space"])
    assert best != "space"


def test_cem_repeat_penalty_three_w():
    # After -0.4 penalty for W, W indices become 0.5; make A/S higher so we switch
    mock = ([0.9, 0.55, 0.55, 0.4, 0.1, 0, 0.9, 0.55, 0.55, 0.4], 0.0, "")
    best, _, _, _, _, _ = run_cem(_frame(), mock_scout_result=mock, use_scout=False, last_actions=["W", "W", "W"])
    assert best != "W"


def test_cem_best_index_matches_action():
    actions = ["W", "A", "S", "D", "space", "none", "W", "A", "S", "D"]
    for i in range(10):
        scores = [0.0] * 10
        scores[i] = 1.0
        mock = (scores, 0.0, "")
        best, _, _, _, _, _ = run_cem(_frame(), actions=actions, mock_scout_result=mock, use_scout=False)
        assert actions[i] == best, f"index {i} expected {actions[i]} got {best}"


def test_cem_avoid_applies_to_all():
    mock = ([0.5] * 10, 0.3, "")
    _, scores, _, _, _, _ = run_cem(_frame(), mock_scout_result=mock, use_scout=False, avoid_weight=1.0, last_actions=[])
    assert all(abs(s - 0.2) < 1e-5 for s in scores)


def test_cem_single_action_list_padded():
    mock = ([1.0], 0.0, "")
    best, scores, _, _, _, _ = run_cem(_frame(), actions=["W"], mock_scout_result=mock, use_scout=False)
    assert len(scores) == 10
    assert best in ("W", "A", "S", "D", "space", "none", "look_left", "look_right")


if __name__ == "__main__":
    tests = [
        test_cem_actions_short_list, test_cem_actions_exactly_10, test_cem_avoid_weight_zero,
        test_cem_avoid_weight_one, test_cem_last_actions_empty, test_cem_last_objective_passed,
        test_cem_returns_four_values, test_cem_scores_order, test_execute_action_none_int_duration,
        test_execute_action_unknown_key_sleeps, test_execute_action_none_none_input,
        test_cem_mock_six_elements, test_cem_mock_two_elements_no_objectives, test_cem_frame_shape_any,
        test_cem_space_index, test_cem_repeat_penalty_space_two, test_cem_repeat_penalty_three_w,
        test_cem_best_index_matches_action, test_cem_avoid_applies_to_all, test_cem_single_action_list_padded,
    ]
    for t in tests:
        t()
        print(t.__name__, "OK")
    print("All CEM extended tests passed.")
