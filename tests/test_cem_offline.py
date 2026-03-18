#!/usr/bin/env python3
"""
Offline tests for CEM and action diversity. No Roblox, no real Scout API required.
Run: python -m pytest tests/test_cem_offline.py -v
Or:  python tests/test_cem_offline.py   (runs as script)
"""
import os
import sys
import time

import numpy as np

# Run from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llm_agent.cem import run_cem, execute_action_ms
from llm_agent.scout import _parse_rewards


def _fake_frame(h=224, w=224):
    return np.zeros((h, w, 3), dtype=np.uint8) + 128


def test_parse_rewards():
    text = "AVOID=0 OBJECTIVES=reach flag REWARD1=0.8 REWARD2=0.1 REWARD3=0.2 REWARD4=0.1 REWARD5=0.5 REWARD6=0 REWARD7=0.2 REWARD8=0.1 REWARD9=0.2 REWARD10=0.1"
    scores = _parse_rewards(text, n=10)
    assert len(scores) == 10
    assert scores[0] == 0.8 and scores[4] == 0.5 and scores[5] == 0.0


def test_parse_rewards_partial():
    """Missing some REWARDs should pad with 0.0."""
    text = "REWARD1=1.0 REWARD3=0.5"
    scores = _parse_rewards(text, n=10)
    assert len(scores) == 10
    assert scores[0] == 1.0 and scores[2] == 0.5
    assert scores[1] == 0.0 and scores[4] == 0.0


def test_parse_rewards_newlines():
    """Newline-separated REWARDs (real Scout sometimes does this)."""
    text = "AVOID=0 \nREWARD1=0.2 \nREWARD2=0.1 \nREWARD3=0.2 \nREWARD4=0.1 \nREWARD5=0.5 \nREWARD6=0.0 \nREWARD7=0.2 \nREWARD8=0.1 \nREWARD9=0.2 \nREWARD10=0.1"
    scores = _parse_rewards(text, n=10)
    assert len(scores) == 10 and scores[4] == 0.5


def test_cem_mock_prefer_w():
    """Without repeat penalty, mock that prefers W (index 0) should yield W."""
    frame = _fake_frame()
    # REWARD1=W, REWARD5=space, etc. Make W highest.
    mock = ([0.9, 0.2, 0.2, 0.2, 0.3, 0.0, 0.2, 0.2, 0.2, 0.2], 0.0, "move forward")
    best, scores, r, obj, _, _ = run_cem(frame, mock_scout_result=mock, use_scout=False)
    assert best == "W", f"expected W got {best}"


def test_cem_anti_repeat_space():
    """After 2x space, space should be downweighted and we can get W instead."""
    frame = _fake_frame()
    # Scout would return space=0.9, W=0.5. With anti-repeat we should get W.
    mock = ([0.5, 0.2, 0.2, 0.2, 0.9, 0.0, 0.5, 0.2, 0.2, 0.2], 0.0, "reach flag")
    best, _, _, _, _, _ = run_cem(
        frame,
        mock_scout_result=mock,
        use_scout=False,
        last_actions=["space", "space"],
    )
    assert best != "space", "anti-repeat should have chosen something other than space"


def test_cem_anti_repeat_generic():
    """After 3x W, W should be downweighted so we can get another action."""
    frame = _fake_frame()
    # All W and first W index dominate; after 3x W we downweight W.
    mock = ([0.9, 0.2, 0.3, 0.2, 0.1, 0.0, 0.85, 0.2, 0.3, 0.2], 0.0, "forward")
    best, _, _, _, _, _ = run_cem(
        frame,
        mock_scout_result=mock,
        use_scout=False,
        last_actions=["W", "W", "W"],
    )
    # After -0.4 penalty, W indices become 0.5 and 0.45; S is 0.3, A 0.3. So best could be W still if others are low.
    # Make S and A higher so we actually switch
    mock2 = ([0.9, 0.5, 0.6, 0.2, 0.1, 0.0, 0.85, 0.5, 0.6, 0.2], 0.0, "forward")
    best2, _, _, _, _, _ = run_cem(
        frame,
        mock_scout_result=mock2,
        use_scout=False,
        last_actions=["W", "W", "W"],
    )
    assert best2 != "W", "anti-repeat (3x) should have chosen something other than W when A/S are close"


def test_cem_loop_diversity():
    """Run a short CEM loop with mock that always returns same scores; repeat penalty should add variety."""
    frame = _fake_frame()
    actions_taken = []
    last_actions = []
    last_objective = None
    # Mock: W=0.7, others 0.3, so without penalty we'd always get W
    for step in range(8):
        mock = ([0.7, 0.35, 0.35, 0.35, 0.2, 0.0, 0.7, 0.35, 0.35, 0.35], 0.0, "go forward")
        best, _, _, obj, _, _ = run_cem(
            frame,
            mock_scout_result=mock,
            use_scout=False,
            last_actions=last_actions,
            last_objective=last_objective,
        )
        actions_taken.append(best)
        last_actions = (last_actions + [best])[-8:]
        if obj:
            last_objective = obj
    # Repeat penalty should cause at least 2 different actions (not 8x same)
    from collections import Counter
    counts = Counter(actions_taken)
    assert len(counts) >= 2, f"expected at least 2 distinct actions, got {actions_taken}"
    most_common_count = counts.most_common(1)[0][1]
    assert most_common_count <= 6, f"expected repeat penalty to limit same action to <=6, got {actions_taken}"


def test_cem_mock_avoid_penalty():
    """With avoid_pen=1, scores reduced; phase=move so W or A can win."""
    frame = _fake_frame()
    mock = ([0.6, 0.5, 0.2, 0.2, 0.1, 0.0, 0.6, 0.5, 0.2, 0.2], 1.0, "danger")
    best, scores, r, _, _, _ = run_cem(frame, mock_scout_result=mock, use_scout=False, avoid_weight=1.0, last_actions=[])
    assert r <= 0.6, "combined score should be reduced by avoid"
    assert best == "W"  # phase=move; W and A scores -avoid; W wins


def test_cem_mock_none_action():
    """When 'none' is best (index 5) in move phase, we get 'none'."""
    frame = _fake_frame()
    mock = ([0.1, 0.1, 0.1, 0.1, 0.1, 0.9, 0.1, 0.1, 0.1, 0.1], 0.0, "wait")
    best, _, r, _, _, _ = run_cem(frame, mock_scout_result=mock, use_scout=False, last_actions=[])
    assert best == "none"
    assert r == 0.9


def test_cem_mock_objectives_returned():
    """mock_scout_result objectives string should be returned."""
    frame = _fake_frame()
    mock = ([0.5] * 10, 0.0, "reach the red flag")
    _, _, _, obj, _, _ = run_cem(frame, mock_scout_result=mock, use_scout=False)
    assert obj == "reach the red flag"


def test_cem_mock_actions_list_stability():
    """Custom actions list length 10; phase=move so no look in list, A wins."""
    frame = _fake_frame()
    actions = ["W", "A", "S", "D", "space", "none", "W", "A", "S", "D"]
    mock = ([0.0, 0.9, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 0.0, "")
    best, _, _, _, _, _ = run_cem(frame, actions=actions, mock_scout_result=mock, use_scout=False, last_actions=[])
    assert best == "A"


def test_cem_no_phase_forcing_best_wins():
    """No phase forcing: best action by score wins regardless of last action."""
    frame = _fake_frame()
    mock = ([0.9, 0.8, 0.7, 0.6, 0.5, 0.1, 0.3, 0.4, 0.3, 0.4], 0.0, "")
    best, _, _, _, _, _ = run_cem(frame, mock_scout_result=mock, use_scout=False, last_actions=["W"])
    assert best == "W", f"highest score is W (0.9), got {best}"


def test_cem_no_phase_forcing_look_can_win():
    """No phase forcing: look can win when it has highest score (indices 6–9 are look)."""
    frame = _fake_frame()
    mock = ([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.9, 0.9, 0.9, 0.9], 0.0, "")
    best, _, _, _, _, _ = run_cem(frame, mock_scout_result=mock, use_scout=False, last_actions=["look_right"])
    assert best in ("look_left", "look_right"), f"highest scores are look actions, got {best}"


def test_execute_action_none():
    """execute_action_ms('none', ms) should only sleep, no key press; duration ~= ms."""
    t0 = time.perf_counter()
    execute_action_ms("none", duration_ms=30)
    elapsed_ms = (time.perf_counter() - t0) * 1000
    assert 20 <= elapsed_ms <= 200, f"expected ~30ms sleep, got {elapsed_ms:.0f}ms"


def test_execute_action_none_lowercase():
    """execute_action_ms('none') with lowercase should still be no-op."""
    t0 = time.perf_counter()
    execute_action_ms("None", duration_ms=20)
    elapsed_ms = (time.perf_counter() - t0) * 1000
    assert 10 <= elapsed_ms <= 150


def test_cem_runtime_mock():
    """Measure CEM runtime with mock (no API) to see if we can afford small intervals."""
    frame = _fake_frame()
    mock = ([0.5] * 10, 0.0, "goal")
    t0 = time.perf_counter()
    for _ in range(5):
        run_cem(frame, mock_scout_result=mock, use_scout=False)
    elapsed = time.perf_counter() - t0
    per_call = elapsed / 5
    print(f"\n[test] CEM (mock) per call: {per_call*1000:.0f} ms", flush=True)
    assert per_call < 0.1, "mock CEM should be very fast (<100ms)"


def run_with_real_scout_once():
    """If GROQ_API_KEY set, run one real Scout+CEM and report runtime (for manual tuning)."""
    if not os.environ.get("GROQ_API_KEY"):
        print("[test] GROQ_API_KEY not set; skipping real Scout runtime test.", flush=True)
        return
    from llm_agent.cem import run_cem
    frame = _fake_frame(360, 640)
    t0 = time.perf_counter()
    run_cem(frame, scout_api_key=os.environ["GROQ_API_KEY"], use_scout=True)
    elapsed = time.perf_counter() - t0
    print(f"[test] CEM (real Scout) one call: {elapsed*1000:.0f} ms", flush=True)


if __name__ == "__main__":
    tests = [
        (test_parse_rewards, "parse_rewards"),
        (test_parse_rewards_partial, "parse_rewards_partial"),
        (test_parse_rewards_newlines, "parse_rewards_newlines"),
        (test_cem_mock_prefer_w, "mock prefer W"),
        (test_cem_anti_repeat_space, "anti-repeat space"),
        (test_cem_anti_repeat_generic, "anti-repeat generic"),
        (test_cem_mock_avoid_penalty, "mock avoid penalty"),
        (test_cem_mock_none_action, "mock none action"),
        (test_cem_mock_objectives_returned, "mock objectives returned"),
        (test_cem_mock_actions_list_stability, "mock actions list"),
        (test_cem_no_phase_forcing_best_wins, "no phase forcing best wins"),
        (test_cem_no_phase_forcing_look_can_win, "no phase forcing look can win"),
        (test_execute_action_none, "execute_action none"),
        (test_execute_action_none_lowercase, "execute_action None"),
        (test_cem_loop_diversity, "loop diversity"),
        (test_cem_runtime_mock, "runtime mock"),
    ]
    for fn, name in tests:
        fn()
        print(f"{name} OK")
    run_with_real_scout_once()
    print("All offline tests passed.")
